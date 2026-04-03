"""
test_tiny_poc.py

Quick validation tests for tiny PoC scripts.

Usage:
    python tests/test_tiny_poc.py
"""

import sys


def test_tiny_models_defined():
    """Test that tiny models are properly defined."""
    try:
        import torch
    except ImportError:
        print("⚠ PyTorch not installed, skipping model definition test")
        return True

    from extract_vectors_tiny import TINY_MODELS

    assert "tinyllama" in TINY_MODELS
    assert "qwen2-1.5b" in TINY_MODELS
    assert "phi2" in TINY_MODELS

    print("✓ Tiny models defined:")
    for name, model_id in TINY_MODELS.items():
        print(f"  {name}: {model_id}")


def test_tiny_dataset_topics():
    """Test that tiny dataset has correct structure."""
    from generate_dataset_tiny import DOMAIN_TOPICS

    # Should have 4 domains
    assert len(DOMAIN_TOPICS) == 4

    # Each domain should have 5 topics
    for domain, topics in DOMAIN_TOPICS.items():
        assert len(topics) == 5, f"Domain {domain} should have 5 topics"

    total_topics = sum(len(topics) for topics in DOMAIN_TOPICS.values())
    assert total_topics == 20

    print(f"✓ Tiny dataset structure: {total_topics} topics across 4 domains")


def test_tiny_prompts():
    """Test that tiny test prompts are defined."""
    try:
        import torch
    except ImportError:
        print("⚠ PyTorch not installed, skipping prompts test")
        return True

    from ablate_vectors_tiny import (
        GENERAL_PROMPTS_TINY,
        CREDIBILITY_TEST_TINY,
        NON_CREDIBILITY_TEST_TINY,
    )

    assert len(GENERAL_PROMPTS_TINY) == 5
    assert len(CREDIBILITY_TEST_TINY) == 3
    assert len(NON_CREDIBILITY_TEST_TINY) == 3

    print("✓ Tiny test prompts defined:")
    print(f"  General: {len(GENERAL_PROMPTS_TINY)}")
    print(f"  Credibility: {len(CREDIBILITY_TEST_TINY)}")
    print(f"  Non-credibility: {len(NON_CREDIBILITY_TEST_TINY)}")


def test_tiny_thresholds():
    """Test that tiny PoC uses looser thresholds."""
    try:
        import torch
    except ImportError:
        print("⚠ PyTorch not installed, skipping threshold test")
        return True

    # PRH threshold should be 0.3 instead of 0.5
    from extract_vectors_tiny import compute_alignment_tiny

    # Mock results
    mock_results = [
        {
            "model_id": "test/model1",
            "dom_vector": [1.0] * 100,
            "lat_vector": [1.0] * 100,
        },
        {
            "model_id": "test/model2",
            "dom_vector": [0.4] * 100,  # Low similarity
            "lat_vector": [0.4] * 100,
        },
    ]

    alignment = compute_alignment_tiny(mock_results)

    assert alignment["prh_threshold"] == 0.3, "Tiny PoC should use 0.3 threshold"
    print("✓ Tiny PoC thresholds:")
    print(f"  PRH threshold: {alignment['prh_threshold']} (looser than 0.5)")


def test_scripts_importable():
    """Test that all tiny scripts can be imported."""
    try:
        import generate_dataset_tiny

        print("✓ generate_dataset_tiny importable")
    except ImportError as e:
        print(f"✗ generate_dataset_tiny: {e}")
        return False

    # Check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
        print("⚠ PyTorch not installed, skipping extraction/ablation imports")

    if torch_available:
        try:
            import extract_vectors_tiny

            print("✓ extract_vectors_tiny importable")
        except ImportError as e:
            print(f"✗ extract_vectors_tiny: {e}")
            return False

        try:
            import ablate_vectors_tiny

            print("✓ ablate_vectors_tiny importable")
        except ImportError as e:
            print(f"✗ ablate_vectors_tiny: {e}")
            return False

    return True


def main():
    print("=" * 60)
    print("Tiny PoC Validation Tests")
    print("=" * 60)
    print()

    tests = [
        test_scripts_importable,
        test_tiny_models_defined,
        test_tiny_dataset_topics,
        test_tiny_prompts,
        test_tiny_thresholds,
    ]

    for test in tests:
        try:
            result = test()
            if result is False:
                print("\n✗ Some tests failed")
                return False
            print()
        except Exception as e:
            print(f"\n✗ Test error: {e}")
            import traceback

            traceback.print_exc()
            return False

    print("=" * 60)
    print("✓ All tiny PoC tests passed!")
    print("=" * 60)
    print()
    print("Ready to run:")
    print("  ./run_tiny_poc.sh")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
