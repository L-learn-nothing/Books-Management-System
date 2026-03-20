from __future__ import annotations

import importlib
import platform
import sys

MODULES = ["numpy", "matplotlib", "torch", "nibabel"]


def main() -> None:
    print("=== 环境检查 ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {module_name}: {version}")
        except Exception as exc:  # noqa: BLE001
            print(f"[MISSING] {module_name}: {exc}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"Torch runtime check skipped: {exc}")


if __name__ == "__main__":
    main()
