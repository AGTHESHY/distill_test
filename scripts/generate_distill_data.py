"""用教师模型为问题集生成回答，产出蒸馏训练数据。

蒸馏的数据准备阶段：大模型（教师）对每条问题生成回答，
小模型（学生）后续将学习模仿这些回答。

用法:
  python scripts/generate_distill_data.py --limit 50           # 先试 50 条
  python scripts/generate_distill_data.py --batch-size 8       # 批处理加速（默认 4）
  python scripts/generate_distill_data.py --resume             # 中断后续跑

输出: data/distill_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 路径配置：全部相对于项目根目录
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TEACHER_DIR = ROOT / "models" / "teacher"          # 教师模型本地目录（7B）
INPUT_FILE = ROOT / "data" / "raw_questions.jsonl"   # 原始问题集
OUTPUT_FILE = ROOT / "data" / "distill_train.jsonl"  # 教师生成的蒸馏训练集


def load_questions(path: Path, limit: int | None) -> list[dict]:
    """从 raw_questions.jsonl 读取问题，每行一条 JSON。

    每行格式示例：
        {"instruction": "...", "input": "", "output": "..."}
    此处只使用 instruction 和 input；output 会被教师模型重新生成并覆盖。
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_done_indices(path: Path) -> set[int]:
    """读取 distill_train.jsonl 中已完成的 index，用于 --resume 断点续跑。

    每条记录里有 "index" 字段，对应 raw_questions.jsonl 中的行号（从 0 开始）。
    """
    if not path.exists():
        return set()
    done: set[int] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["index"])
    return done


def build_prompt(tokenizer, instruction: str, user_input: str) -> str:
    """把一条样本格式化为 Qwen 聊天模板字符串。

    Qwen Instruct 模型要求特定格式（含 <|im_start|> 等标记），
    不能直接把纯文本丢给模型，必须用 apply_chat_template 包装。
    add_generation_prompt=True 会在末尾加上 assistant 开头，提示模型开始生成回答。
    """
    user_content = instruction.strip()
    if user_input.strip():
        user_content = f"{user_content}\n{user_input.strip()}"
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def setup_tokenizer_for_batch(tokenizer) -> None:
    """配置 tokenizer 以支持批处理（batch inference）。

    因果语言模型（Causal LM）批处理时必须用【左填充 / left padding】：
    - 每条 prompt 长度不同，需 pad 到同一长度才能拼成 batch 张量
    - 左填充：短序列左侧补 pad，真实 token 靠右对齐
    - 生成从序列最右侧继续，左填充不影响模型看到的内容

    若用右填充，pad 会出现在真实 token 和生成区之间，导致生成位置错乱。
    """
    tokenizer.padding_side = "left"
    # Qwen 默认可能没有 pad_token，用 eos_token 代替（生成时不会用到 pad 位置）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_teacher(use_fp16: bool):
    """加载教师模型到 GPU。

    默认 8-bit 量化（省显存）；传 --fp16 时用 fp16 全精度（可能更快，显存约 14GB）。
    device_map="auto" 让 accelerate 自动把各层分配到 GPU（必要时 offload 到 CPU）。
    """
    tokenizer = AutoTokenizer.from_pretrained(str(TEACHER_DIR), trust_remote_code=True)
    setup_tokenizer_for_batch(tokenizer)

    model_kwargs: dict = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if use_fp16:
        model_kwargs["dtype"] = torch.float16
    else:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(str(TEACHER_DIR), **model_kwargs)
    model.eval()  # 推理模式：关闭 Dropout，不更新权重
    return tokenizer, model


@torch.inference_mode()  # 关闭梯度计算，节省显存、加速推理
def generate_batch_answers(
    model,
    tokenizer,
    batch_rows: list[dict],
    max_new_tokens: int,
) -> list[str]:
    """对一个 batch 的多条问题同时调用 model.generate()，返回对应的回答列表。

    批处理 vs 逐条：
    - 逐条：GPU 算完一条再等 CPU 准备下一条，利用率低（你之前 nvidia-smi 约 24%）
    - 批处理：多条并行前向，矩阵运算更大块，GPU 利用率更高，总时间更短

    参数:
        batch_rows: 同一批的样本 dict 列表，含 instruction / input 字段
        max_new_tokens: 每条最多新生成多少个 token

    返回:
        与 batch_rows 等长的回答字符串列表
    """
    # 1. 为 batch 内每条样本构建聊天格式 prompt
    prompts = [
        build_prompt(tokenizer, row.get("instruction", ""), row.get("input", ""))
        for row in batch_rows
    ]

    # 2. 批量 tokenize：padding=True 把不同长度的序列 pad 到同一长度
    #    return_tensors="pt" 得到 PyTorch 张量；attention_mask 标记哪些是真实 token、哪些是 pad
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 左填充后，所有样本的「输入区」在张量里宽度相同（均为 input_width）
    input_width = inputs["input_ids"].shape[1]

    # 3. 一次 generate 处理整个 batch
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,           # 贪心解码：每步选概率最高的 token，结果稳定可复现
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 4. 从完整输出中切出「仅新生成」部分，再 decode 成文本
    answers: list[str] = []
    for i in range(len(batch_rows)):
        # output_ids[i] = [左填充的输入 token | 新生成的 token]
        # 新生成部分从 input_width 下标开始（左填充时输入区宽度固定）
        new_tokens = output_ids[i, input_width:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answers.append(text)

    return answers


def chunked(indices: list[int], batch_size: int):
    """把 index 列表按 batch_size 切成多段，例如 [0
    ,1,2,3,4], size=2 -> [0,1], [2,3], [4]。"""
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="教师模型批量生成蒸馏数据")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条（试跑用）")
    parser.add_argument("--resume", action="store_true", help="跳过已生成的条目，断点续跑")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="每条最多生成的 token 数")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="批大小；越大 GPU 利用率越高，但显存占用也越大（16GB 建议 4～8）",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="用 fp16 全精度加载教师模型（默认 8-bit 量化）",
    )
    args = parser.parse_args()

    if not TEACHER_DIR.exists():
        raise FileNotFoundError(f"未找到教师模型: {TEACHER_DIR}")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"未找到问题集: {INPUT_FILE}")

    questions = load_questions(INPUT_FILE, args.limit)
    done = load_done_indices(OUTPUT_FILE) if args.resume else set()
    pending = [i for i in range(len(questions)) if i not in done]

    print(f"共 {len(questions)} 条，待生成 {len(pending)} 条，batch_size={args.batch_size}")
    if not pending:
        print("已全部完成，无需重复生成。")
        return

    print("加载教师模型（8-bit）..." if not args.fp16 else "加载教师模型（fp16）...")
    tokenizer, model = load_teacher(args.fp16)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and OUTPUT_FILE.exists() else "w"

    total_done = 0
    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        for batch_indices in chunked(pending, args.batch_size):
            batch_rows = [questions[idx] for idx in batch_indices]

            t0 = time.perf_counter()
            answers = generate_batch_answers(
                model, tokenizer, batch_rows, args.max_new_tokens
            )
            elapsed = time.perf_counter() - t0

            for idx, row, answer in zip(batch_indices, batch_rows, answers):
                record = {
                    "index": idx,
                    "instruction": row.get("instruction", ""),
                    "input": row.get("input", ""),
                    "output": answer,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f.flush()  # 每个 batch 落盘一次，中断后已完成的 batch 不丢
            total_done += len(batch_indices)
            per_item = elapsed / len(batch_indices)
            print(
                f"[{total_done}/{len(pending)}] "
                f"完成 batch（{len(batch_indices)} 条，{elapsed:.1f}s，{per_item:.1f}s/条）"
            )

    print(f"已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
