"""
test_smoke.py

Smoke tests to verify basic imports and script validity.

Usage:
    python tests/test_smoke.py
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import numpy as np

        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False

    try:
        import torch

        print(f"  ✓ torch (version {torch.__version__})")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False

    try:
        import transformer_lens

        print("  ✓ transformer_lens")
    except ImportError as e:
        print(f"  ✗ transformer_lens: {e}")
        return False

    return True


def test_extract_vectors_imports():
    """Test that extract_vectors.py can be imported."""
    print("\nTesting extract_vectors.py imports...")

    try:
        import extract_vectors

        print("  ✓ extract_vectors module")
    except ImportError as e:
        print(f"  ✗ extract_vectors: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Check key functions exist
    required_functions = [
        "extract_activations",
        "compute_dom_vector",
        "compute_lat_vector",
        "cosine_similarity",
        "load_dataset",
        "compute_alignment_matrix",
        "extract_credibility_vectors",
    ]

    for func_name in required_functions:
        if hasattr(extract_vectors, func_name):
            print(f"  ✓ {func_name}")
        else:
            print(f"  ✗ {func_name} not found")
            return False

    return True


def test_dataset_exists():
    """Check if the Phase 1 dataset exists."""
    print("\nChecking dataset...")

    dataset_path = Path("data/credibility_pairs.jsonl")
    if dataset_path.exists():
        # Count records
        with dataset_path.open() as f:
            n_records = sum(1 for _ in f)
        print(f"  ✓ Dataset found: {n_records} records")
        return True
    else:
        print(f"  ⚠ Dataset not found: {dataset_path}")
        print("    (Run 'python src/generate_dataset.py' to create it)")
        return False


def test_supported_models():
    """Test that supported model definitions are correct."""
    print("\nChecking model definitions...")

    try:
        from extract_vectors import SUPPORTED_MODELS

        expected_models = ["llama3", "mistral", "qwen"]

        for model in expected_models:
            if model in SUPPORTED_MODELS:
                print(f"  ✓ {model}: {SUPPORTED_MODELS[model]}")
            else:
                print(f"  ✗ {model} not found")
                return False

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_cli_parsing():
    """Test that CLI argument parsing works."""
    print("\nTesting CLI parsing...")

    # Save original sys.argv first
    original_argv = sys.argv.copy()

    try:
        from extract_vectors import parse_args

        # Test with minimal args
        sys.argv = ["extract_vectors.py", "--model", "llama3"]
        args = parse_args()

        assert args.model == "llama3", "Model argument not parsed correctly"
        assert not args.all_models, "all_models should be False"

        print("  ✓ Single model parsing")

        # Test with --all-models
        sys.argv = ["extract_vectors.py", "--all-models"]
        args = parse_args()

        assert args.all_models, "all_models should be True"

        print("  ✓ All models parsing")

        # Restore original sys.argv
        sys.argv = original_argv

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.argv = original_argv  # Restore even on error
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Smoke Tests - Phase 2 Vector Extraction")
    print("=" * 60)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_extract_vectors_imports()
    all_passed &= test_supported_models()
    all_passed &= test_cli_parsing()

    # Dataset check is optional
    dataset_exists = test_dataset_exists()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All smoke tests passed!")
        if not dataset_exists:
            print("\n⚠ Note: Dataset not found. Run Phase 1 before extraction.")
    else:
        print("✗ Some smoke tests failed")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
