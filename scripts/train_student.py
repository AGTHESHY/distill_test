"""用蒸馏数据对学生模型做 SFT（监督微调），完成知识蒸馏的最后一步。

训练数据来自 generate_distill_data.py 生成的 distill_train.jsonl：
每条包含 instruction + 教师生成的 output，学生学习模仿教师的回答。

用法:
  python scripts/train_student.py                    # 默认 LoRA 微调
  python scripts/train_student.py --epochs 3           # 训练 3 轮
  python scripts/train_student.py --no-lora          # 全量微调（0.5B 可尝试）
  python scripts/train_student.py --merge-lora       # 训练后合并 LoRA 权重

输出:
  models/student-distilled/          # LoRA 适配器 + tokenizer
  models/student-distilled-merged/   # 加 --merge-lora 时的完整模型
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
STUDENT_DIR = ROOT / "models" / "student"
TRAIN_FILE = ROOT / "data" / "distill_train.jsonl"
OUTPUT_DIR = ROOT / "models" / "student-distilled"
MERGED_DIR = ROOT / "models" / "student-distilled-merged"

# Qwen2.5 LoRA 常用目标层（注意力 + MLP）
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_train_dataset(path: Path, eval_ratio: float):
    """加载 distill_train.jsonl，并转为对话格式。

    TRL SFTTrainer 的 conversational 格式要求每条样本有 messages 字段：
        [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "教师回答"},
        ]

    配合 assistant_only_loss=True，训练时只在 assistant 部分计算 loss，
    不让模型学习「复述问题」，只学习「如何回答」。
    """
    dataset = load_dataset("json", data_files=str(path), split="train")

    def to_messages(example: dict) -> dict:
        instruction = example.get("instruction", "").strip()
        user_input = example.get("input", "").strip()
        user_content = instruction
        if user_input:
            user_content = f"{instruction}\n{user_input}"
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": example.get("output", "").strip()},
            ]
        }

    dataset = dataset.map(to_messages, remove_columns=dataset.column_names)

    if eval_ratio > 0:
        split = dataset.train_test_split(test_size=eval_ratio, seed=42)
        return split["train"], split["test"]
    return dataset, None


def setup_tokenizer(model_dir: Path) -> AutoTokenizer:
    """加载 tokenizer，并设置 pad_token（批训练 padding 必需）。"""
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer.padding_side = "right"  # 训练时通常右填充（与生成时的左填充不同）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_peft_config() -> LoraConfig:
    """LoRA 配置：只训练少量低秩矩阵，显存小、速度快，适合 0.5B 入门实验。"""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="蒸馏：学生模型 SFT 训练")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="每步每卡样本数")
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="梯度累积步数；有效 batch = batch_size × 此值",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率（LoRA 常用 1e-4 ~ 2e-4）")
    parser.add_argument("--max-length", type=int, default=1024, help="单条样本最大 token 数")
    parser.add_argument("--eval-ratio", type=float, default=0.05, help="验证集比例，0 表示不划分")
    parser.add_argument("--no-lora", action="store_true", help="全量微调（不用 LoRA）")
    parser.add_argument("--merge-lora", action="store_true", help="训练结束后合并 LoRA 到完整模型")
    parser.add_argument("--fp16", action="store_true", help="用 fp16 训练（默认 bf16，若 GPU 支持）")
    args = parser.parse_args()

    if not STUDENT_DIR.exists():
        raise FileNotFoundError(f"未找到学生模型: {STUDENT_DIR}")
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"未找到训练数据: {TRAIN_FILE}")

    train_dataset, eval_dataset = load_train_dataset(TRAIN_FILE, args.eval_ratio)
    print(f"训练集: {len(train_dataset)} 条", end="")
    if eval_dataset is not None:
        print(f"，验证集: {len(eval_dataset)} 条")
    else:
        print()

    tokenizer = setup_tokenizer(STUDENT_DIR)

    # 加载学生基座模型；0.5B fp16/bf16 约 1～2GB 显存
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        str(STUDENT_DIR),
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    use_lora = not args.no_lora
    peft_config = build_peft_config() if use_lora else None

    # SFTConfig：TRL 对 HuggingFace Trainer 的封装，专为指令微调设计
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        max_length=args.max_length,
        assistant_only_loss=True,  # 只在教师回答（assistant）部分算 loss
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        bf16=not args.fp16 and torch.cuda.is_bf16_supported(),
        fp16=args.fp16,
        report_to="none",
        gradient_checkpointing=True,  # 用计算换显存，长序列更稳
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("开始训练...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"训练完成，已保存: {OUTPUT_DIR}")

    if use_lora and args.merge_lora:
        print("合并 LoRA 权重到完整模型...")
        merged = trainer.model.merge_and_unload()
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(MERGED_DIR))
        tokenizer.save_pretrained(str(MERGED_DIR))
        print(f"合并模型已保存: {MERGED_DIR}")


if __name__ == "__main__":
    main()
