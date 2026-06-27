"""验证 distill 环境是否配置正确。"""

import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    errors: list[str] = []

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            cap = torch.cuda.get_device_capability(0)
            print(f"Compute capability: sm_{cap[0]}{cap[1]}")
            x = torch.randn(2, 2, device="cuda")
            print(f"CUDA tensor test: OK ({x.device})")
        else:
            errors.append("CUDA 不可用，请检查 PyTorch 是否为 cu128 版本")
    except Exception as e:
        errors.append(f"PyTorch 检查失败: {e}")

    packages = [
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "huggingface_hub",
    ]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"{pkg}: {ver}")
        except ImportError as e:
            errors.append(f"{pkg} 未安装: {e}")

    if errors:
        print("\n--- 问题 ---")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\n环境验证通过，可以开始蒸馏实验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
