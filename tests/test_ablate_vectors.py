"""
test_ablate_vectors.py

Unit tests for Phase 3 ablation pipeline.

Usage:
    pytest tests/test_ablate_vectors.py -v
    python tests/test_ablate_vectors.py
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAblationMath:
    """Test ablation mathematical operations."""

    def test_orthogonal_projection(self):
        """Test that orthogonal projection removes component."""
        # Create a vector and a direction
        x = np.array([1.0, 1.0, 0.0])
        v = np.array([1.0, 0.0, 0.0])  # Direction to remove

        # Normalize direction
        v = v / np.linalg.norm(v)

        # Compute projection: (x · v)v
        projection = np.dot(x, v) * v

        # Subtract projection
        x_ablated = x - projection

        # Check that ablated vector has no component along v
        component_along_v = np.dot(x_ablated, v)

        assert np.isclose(
            component_along_v, 0.0, atol=1e-6
        ), f"Ablated vector should have no component along v, got {component_along_v}"

        print(f"✓ Original: {x}")
        print(f"✓ Direction: {v}")
        print(f"✓ Ablated: {x_ablated}")
        print(f"✓ Component along v: {component_along_v:.10f}")

    def test_ablation_preserves_orthogonal(self):
        """Test that ablation preserves orthogonal components."""
        # Vector orthogonal to direction should be unchanged
        x = np.array([0.0, 1.0, 0.0])
        v = np.array([1.0, 0.0, 0.0])

        v = v / np.linalg.norm(v)
        projection = np.dot(x, v) * v
        x_ablated = x - projection

        # Should be identical
        assert np.allclose(
            x_ablated, x
        ), "Orthogonal vectors should be preserved by ablation"

        print(f"✓ Orthogonal vector preserved: {x} -> {x_ablated}")

    def test_ablation_reduces_separation(self):
        """Test that ablating a direction reduces separation."""
        # Create two clusters separated along a direction
        direction = np.array([1.0, 0.0, 0.0, 0.0])
        direction = direction / np.linalg.norm(direction)

        # Cluster A: positive along direction
        cluster_a = np.random.randn(10, 4) + np.array([2.0, 0.0, 0.0, 0.0])

        # Cluster B: negative along direction
        cluster_b = np.random.randn(10, 4) + np.array([-2.0, 0.0, 0.0, 0.0])

        # Compute baseline separation
        mean_a = cluster_a.mean(axis=0)
        mean_b = cluster_b.mean(axis=0)
        baseline_sep = np.dot(mean_a - mean_b, direction)

        # Ablate both clusters
        cluster_a_ablated = cluster_a - np.outer(
            np.dot(cluster_a, direction), direction
        )
        cluster_b_ablated = cluster_b - np.outer(
            np.dot(cluster_b, direction), direction
        )

        # Compute ablated separation
        mean_a_ablated = cluster_a_ablated.mean(axis=0)
        mean_b_ablated = cluster_b_ablated.mean(axis=0)
        ablated_sep = np.dot(mean_a_ablated - mean_b_ablated, direction)

        print(f"✓ Baseline separation: {baseline_sep:.4f}")
        print(f"✓ Ablated separation: {ablated_sep:.4f}")
        print(f"✓ Reduction: {(1 - ablated_sep/baseline_sep)*100:.1f}%")

        # Ablated separation should be near zero
        assert abs(ablated_sep) < abs(baseline_sep) * 0.1, (
            "Ablation should reduce separation to near zero"
        )


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_kl_identical_distributions(self):
        """Test KL divergence between identical distributions is zero."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            print("⚠ PyTorch not installed, skipping KL divergence tests")
            return

        # Create identical probability distributions
        probs = torch.tensor([0.1, 0.3, 0.4, 0.2])

        kl = F.kl_div(probs.log(), probs, reduction="sum", log_target=False)

        assert torch.isclose(
            kl, torch.tensor(0.0), atol=1e-6
        ), "KL divergence of identical distributions should be 0"

        print(f"✓ KL(P||P) = {kl.item():.10f}")

    def test_kl_small_change(self):
        """Test KL divergence for small distribution changes."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            print("⚠ PyTorch not installed, skipping KL divergence tests")
            return

        # Original distribution
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])

        # Slightly perturbed distribution
        q = torch.tensor([0.26, 0.24, 0.25, 0.25])

        kl = F.kl_div(q.log(), p, reduction="sum", log_target=False)

        print(f"✓ Original: {p.numpy()}")
        print(f"✓ Perturbed: {q.numpy()}")
        print(f"✓ KL divergence: {kl.item():.6f}")

        # Small perturbation should have small KL
        assert kl < 0.01, "Small perturbation should have small KL divergence"


class TestIntegration:
    """Integration tests for ablation pipeline."""

    def test_load_vectors_format(self):
        """Test that we can parse Phase 2 output format."""
        import json
        import tempfile

        # Create mock Phase 2 results
        mock_results = {
            "extractions": [
                {
                    "model_id": "test/model",
                    "best_layer": 16,
                    "separation": 10.0,
                    "dom_vector": [0.1] * 128,
                    "lat_vector": [0.2] * 128,
                    "dom_lat_similarity": 0.9,
                    "hidden_dim": 128,
                    "n_layers": 24,
                }
            ],
            "alignment": {"avg_dom_similarity": 0.65},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(mock_results, f)
            temp_path = Path(f.name)

        try:
            # Test loading
            with temp_path.open() as f:
                data = json.load(f)

            assert "extractions" in data
            assert len(data["extractions"]) > 0

            extraction = data["extractions"][0]
            assert "dom_vector" in extraction
            assert "best_layer" in extraction

            dom_vector = np.array(extraction["dom_vector"])
            assert dom_vector.shape == (128,)

            print("✓ Phase 2 output format validated")
            print(f"  Model: {extraction['model_id']}")
            print(f"  Best layer: {extraction['best_layer']}")
            print(f"  Vector dim: {len(extraction['dom_vector'])}")

        finally:
            temp_path.unlink()

    def test_ablation_results_format(self):
        """Test ablation results output format."""
        # Mock ablation result
        result = {
            "model_id": "test/model",
            "layer": 16,
            "component": "resid_post",
            "baseline_separation": 10.0,
            "ablated_separation": 2.0,
            "separation_reduction": 0.8,
            "kl_divergence": 0.15,
            "kl_threshold": 0.2,
            "kl_pass": True,
            "ablation_success": True,
        }

        # Validate structure
        required_keys = [
            "model_id",
            "layer",
            "component",
            "baseline_separation",
            "ablated_separation",
            "separation_reduction",
            "kl_divergence",
            "kl_pass",
            "ablation_success",
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Validate thresholds
        assert result["separation_reduction"] >= 0
        assert result["separation_reduction"] <= 1
        assert result["kl_divergence"] >= 0
        assert result["kl_pass"] == (result["kl_divergence"] < result["kl_threshold"])

        print("✓ Ablation result format validated")
        print(f"  Separation reduction: {result['separation_reduction']*100:.1f}%")
        print(f"  KL divergence: {result['kl_divergence']:.4f}")
        print(f"  Success: {result['ablation_success']}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing Phase 3 Ablation Pipeline")
    print("=" * 60)

    test_classes = [
        TestAblationMath,
        TestKLDivergence,
        TestIntegration,
    ]

    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                method = getattr(instance, method_name)
                try:
                    method()
                except AssertionError as e:
                    print(f"✗ {method_name} FAILED: {e}")
                    return False
                except Exception as e:
                    print(f"✗ {method_name} ERROR: {e}")
                    import traceback

                    traceback.print_exc()
                    return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
