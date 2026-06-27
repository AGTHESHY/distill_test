"""用教师模型为问题集生成回答，产出蒸馏训练数据。

用法:
  python scripts/generate_distill_data.py --limit 50    # 先试 50 条
  python scripts/generate_distill_data.py               # 全部 1000 条
  python scripts/generate_distill_data.py --resume      # 中断后续跑

输出: data/distill_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TEACHER_DIR = ROOT / "models" / "teacher"
INPUT_FILE = ROOT / "data" / "raw_questions.jsonl"
OUTPUT_FILE = ROOT / "data" / "distill_train.jsonl"


def load_questions(path: Path, limit: int | None) -> list[dict]:
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
    user_content = instruction.strip()
    if user_input.strip():
        user_content = f"{user_content}\n{user_input.strip()}"
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_teacher(quantize_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(str(TEACHER_DIR), trust_remote_code=True)

    model_kwargs: dict = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(str(TEACHER_DIR), **model_kwargs)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    instruction: str,
    user_input: str,
    max_new_tokens: int,
) -> str:
    prompt = build_prompt(tokenizer, instruction, user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="教师模型生成蒸馏数据")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条（试跑用）")
    parser.add_argument("--resume", action="store_true", help="跳过已生成的条目")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--quantize-4bit",
        action="store_true",
        help="4-bit 量化加载教师模型（需 pip install bitsandbytes，显存更省）",
    )
    args = parser.parse_args()

    if not TEACHER_DIR.exists():
        raise FileNotFoundError(f"未找到教师模型: {TEACHER_DIR}")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"未找到问题集: {INPUT_FILE}")

    questions = load_questions(INPUT_FILE, args.limit)
    done = load_done_indices(OUTPUT_FILE) if args.resume else set()
    pending = [i for i in range(len(questions)) if i not in done]

    print(f"共 {len(questions)} 条，待生成 {len(pending)} 条")
    if not pending:
        print("已全部完成，无需重复生成。")
        return

    print("加载教师模型...")
    tokenizer, model = load_teacher(args.quantize_4bit)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and OUTPUT_FILE.exists() else "w"
    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        for n, idx in enumerate(pending, 1):
            row = questions[idx]
            instruction = row.get("instruction", "")
            user_input = row.get("input", "")
            answer = generate_answer(
                model, tokenizer, instruction, user_input, args.max_new_tokens
            )
            record = {
                "index": idx,
                "instruction": instruction,
                "input": user_input,
                "output": answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(pending)}] 完成第 {idx + 1} 条")

    print(f"已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
