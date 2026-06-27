"""下载蒸馏所需的模型与数据。

用法:
  python scripts/download.py --student          # 仅学生模型 (~1GB)
  python scripts/download.py --teacher          # 仅教师模型 (~15GB)
  python scripts/download.py --data             # 仅问题数据集
  python scripts/download.py --all              # 全部

国内网络慢时可先设置:
  $env:HF_ENDPOINT = "https://hf-mirror.com"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"

# 教师大、学生小，同属 Qwen2.5 系列，tokenizer 兼容
STUDENT_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_REPO = "Qwen/Qwen2.5-7B-Instruct"
# 中文指令数据集，只取前 1000 条作为蒸馏问题来源
DATASET_REPO = "BelleGroup/train_1M_CN"
DATASET_SPLIT = "train[:1000]"


def download_model(repo_id: str, local_dir: Path) -> None:
    """从 Hugging Face 下载模型到本地，支持断点续传。"""
    print(f"下载模型 {repo_id} -> {local_dir}")
    if local_dir.exists() and any(local_dir.iterdir()):
        print("检测到未完成下载，将自动断点续传（勿删除 models/ 目录）")
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            max_workers=4,
        )
    except KeyboardInterrupt:
        print("\n下载已中断。重新运行相同命令即可续传，无需删文件。")
        raise
    print(f"完成: {local_dir}")


def download_data() -> None:
    """下载中文问题集，导出为 UTF-8 中文明文的 JSONL。"""
    import json

    from datasets import load_dataset

    out = DATA / "raw_questions.jsonl"
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"下载数据集 {DATASET_REPO} ({DATASET_SPLIT})")
    ds = load_dataset(DATASET_REPO, split=DATASET_SPLIT)
    # ensure_ascii=False 让文件里直接显示中文，而非 \uXXXX 转义
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"完成: {out} ({len(ds)} 条, UTF-8 中文明文)")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载蒸馏模型与数据")
    parser.add_argument("--student", action="store_true", help="下载学生模型 0.5B")
    parser.add_argument("--teacher", action="store_true", help="下载教师模型 7B")
    parser.add_argument("--data", action="store_true", help="下载 1000 条中文问题")
    parser.add_argument("--all", action="store_true", help="下载全部")
    args = parser.parse_args()

    if not any([args.student, args.teacher, args.data, args.all]):
        parser.print_help()
        return

    # 数据集较小，优先下载，避免大模型中断时数据也没下到
    if args.all or args.data:
        download_data()
    if args.all or args.student:
        download_model(STUDENT_REPO, MODELS / "student")
    if args.all or args.teacher:
        download_model(TEACHER_REPO, MODELS / "teacher")


if __name__ == "__main__":
    main()
