"""
verify_setup.py

Verify that the environment is properly configured for Phase 2 extraction.

Usage:
    python src/verify_setup.py
"""

import sys
from pathlib import Path


def check_import(module_name: str, package_name: str = None) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"  ✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"  ✗ {package_name or module_name} - {e}")
        return False


def check_file(path: Path, description: str) -> bool:
    """Check if a file exists."""
    if path.exists():
        print(f"  ✓ {description}: {path}")
        return True
    else:
        print(f"  ✗ {description}: {path} (not found)")
        return False


def check_cuda() -> bool:
    """Check CUDA availability."""
    try:
        import torch

        from rosetta_tools.gpu_utils import log_vram

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✓ CUDA available: {device_name} ({vram:.1f} GiB VRAM)")
            log_vram("current usage")
            return True
        else:
            print("  ⚠ CUDA not available (will use CPU - slower)")
            return False
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False


def main():
    print("=== Rosetta Manifold Setup Verification ===\n")

    all_good = True

    # Check Python version
    print("Python version:")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(
            f"  ✗ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)"
        )
        all_good = False

    print("\nCore dependencies:")
    all_good &= check_import("torch", "PyTorch")
    all_good &= check_import("numpy", "NumPy")
    all_good &= check_import("transformer_lens", "TransformerLens")
    all_good &= check_import("transformers", "HuggingFace Transformers")

    print("\nOptional dependencies:")
    check_import("opik", "Opik (experiment tracking)")
    check_import("openai", "OpenAI (dataset generation)")
    check_import("ollama", "Ollama (local inference)")
    check_import("optuna", "Optuna (hyperparameter optimization)")

    print("\nHardware:")
    check_cuda()

    print("\nDataset:")
    dataset_path = Path("data/credibility_pairs.jsonl")
    if check_file(dataset_path, "Credibility dataset"):
        # Count records
        try:
            with dataset_path.open() as f:
                n_records = sum(1 for _ in f)
            print(f"    ({n_records} records)")
        except Exception as e:
            print(f"    (error reading: {e})")
    else:
        print("    Run 'python src/generate_dataset.py' to create it")

    print("\nModel download permissions:")
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if token:
            print("  ✓ HuggingFace token found")
        else:
            print("  ⚠ No HuggingFace token (may be needed for gated models)")
            print("    Login with: huggingface-cli login")
    except ImportError:
        print("  ⚠ huggingface_hub not installed")

    print("\n" + "=" * 50)
    if all_good:
        print("✓ All critical dependencies are installed!")
        print("\nReady to run:")
        print("  python src/extract_vectors.py --model llama3")
    else:
        print("✗ Some dependencies are missing")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")

    print()
    sys.exit(0 if all_good else 1)


if __name__ == "__main__":
    main()
