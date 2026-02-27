"""
test_extract_vectors.py

Unit and integration tests for Phase 2 vector extraction pipeline.

Usage:
    pytest tests/test_extract_vectors.py -v
    python tests/test_extract_vectors.py  # Runs without pytest
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extract_vectors import (
    compute_dom_vector,
    compute_lat_vector,
    cosine_similarity,
    load_dataset,
    compute_alignment_matrix,
)


class TestMathematicalFunctions:
    """Test core mathematical functions."""

    def test_compute_dom_vector(self):
        """Test Difference-of-Means computation."""
        # Create synthetic activations with clear separation
        np.random.seed(42)
        credible = np.random.randn(10, 4) + np.array([1, 0, 0, 0])
        non_credible = np.random.randn(10, 4) + np.array([-1, 0, 0, 0])

        vector = compute_dom_vector(credible, non_credible)

        # Should be normalized
        assert np.isclose(np.linalg.norm(vector), 1.0), "DoM vector should be normalized"

        # First dimension should be positive (credible - non_credible)
        assert vector[0] > 0, "First dimension should capture the separation"

        print("✓ DoM vector computation works correctly")

    def test_compute_lat_vector(self):
        """Test Linear Artificial Tomography computation."""
        np.random.seed(42)
        credible = np.random.randn(10, 4) + np.array([1, 0, 0, 0])
        non_credible = np.random.randn(10, 4) + np.array([-1, 0, 0, 0])

        vector = compute_lat_vector(credible, non_credible)

        # Should be normalized
        assert np.isclose(np.linalg.norm(vector), 1.0), "LAT vector should be normalized"

        print("✓ LAT vector computation works correctly")

    def test_dom_lat_agreement(self):
        """Test that DoM and LAT agree on linearly separable data."""
        np.random.seed(42)
        # Create perfectly linearly separable data
        credible = np.random.randn(50, 8) + np.array([2, 0, 0, 0, 0, 0, 0, 0])
        non_credible = np.random.randn(50, 8) + np.array([-2, 0, 0, 0, 0, 0, 0, 0])

        dom_vec = compute_dom_vector(credible, non_credible)
        lat_vec = compute_lat_vector(credible, non_credible)

        similarity = cosine_similarity(dom_vec, lat_vec)

        # Should have high agreement on linearly separable data
        assert similarity > 0.8, f"DoM-LAT similarity should be high, got {similarity}"

        print(f"✓ DoM-LAT agreement: {similarity:.4f}")

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 0, 0])
        v3 = np.array([0, 1, 0])
        v4 = np.array([-1, 0, 0])

        # Identical vectors
        assert np.isclose(
            cosine_similarity(v1, v2), 1.0
        ), "Identical vectors should have similarity 1.0"

        # Orthogonal vectors
        assert np.isclose(
            cosine_similarity(v1, v3), 0.0
        ), "Orthogonal vectors should have similarity 0.0"

        # Opposite vectors
        assert np.isclose(
            cosine_similarity(v1, v4), -1.0
        ), "Opposite vectors should have similarity -1.0"

        print("✓ Cosine similarity computation works correctly")


class TestDatasetLoading:
    """Test dataset loading and parsing."""

    def test_load_dataset(self):
        """Test loading credibility dataset."""
        # Create temporary dataset
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            # Write test data
            for i in range(10):
                record = {
                    "pair_id": f"test_{i}",
                    "label": 1,
                    "domain": "technical",
                    "model_name": "test-model",
                    "text": f"Credible text {i}",
                    "topic": "test topic",
                }
                f.write(json.dumps(record) + "\n")

                record["label"] = 0
                record["text"] = f"Non-credible text {i}"
                f.write(json.dumps(record) + "\n")

            temp_path = Path(f.name)

        try:
            credible, non_credible = load_dataset(temp_path)

            assert len(credible) == 10, "Should load 10 credible texts"
            assert len(non_credible) == 10, "Should load 10 non-credible texts"
            assert all("Credible" in t for t in credible), "Credible texts should match"
            assert all(
                "Non-credible" in t for t in non_credible
            ), "Non-credible texts should match"

            print("✓ Dataset loading works correctly")
        finally:
            temp_path.unlink()


class TestAlignmentComputation:
    """Test cross-model alignment computation."""

    def test_compute_alignment_matrix(self):
        """Test alignment matrix computation."""
        # Create mock extraction results
        np.random.seed(42)

        results = []
        for i, model in enumerate(["model_a", "model_b", "model_c"]):
            # Create similar vectors with some noise
            base_vector = np.array([1, 0, 0, 0])
            dom_vector = (base_vector + np.random.randn(4) * 0.1).tolist()
            lat_vector = (base_vector + np.random.randn(4) * 0.1).tolist()

            results.append(
                {
                    "model_id": f"test/{model}",
                    "best_layer": 16,
                    "separation": 10.0,
                    "dom_vector": dom_vector,
                    "lat_vector": lat_vector,
                    "dom_lat_similarity": 0.95,
                    "hidden_dim": 4,
                    "n_layers": 24,
                    "token_pos": -1,
                    "layer_range": [14, 23],
                }
            )

        alignment = compute_alignment_matrix(results)

        # Check structure
        assert "models" in alignment
        assert "dom_similarities" in alignment
        assert "lat_similarities" in alignment
        assert "avg_dom_similarity" in alignment
        assert "avg_lat_similarity" in alignment
        assert "prh_pass" in alignment

        # Check we have the right number of pairs
        n_pairs = len(results) * (len(results) - 1) // 2
        assert len(alignment["dom_similarities"]) == n_pairs

        # Similarities should be high since we created similar vectors
        assert (
            alignment["avg_dom_similarity"] > 0.5
        ), "Average similarity should be high for similar vectors"

        print(f"✓ Alignment computation works correctly")
        print(f"  Average DoM similarity: {alignment['avg_dom_similarity']:.4f}")
        print(f"  Average LAT similarity: {alignment['avg_lat_similarity']:.4f}")
        print(f"  PRH test: {'PASS' if alignment['prh_pass'] else 'FAIL'}")


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked model."""
        print("\n=== Integration Test ===")

        # Create temporary dataset
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for i in range(20):
                record = {
                    "pair_id": f"test_{i:02d}",
                    "label": 1,
                    "domain": "technical",
                    "model_name": "test-model",
                    "text": f"This is a credible statement with specific dates like January 15, 2023. Research published in Nature found that {i}% accuracy.",
                    "topic": "test topic",
                }
                f.write(json.dumps(record) + "\n")

                record["label"] = 0
                record[
                    "text"
                ] = f"Everyone knows this is absolutely devastating and shocking! My friend told me this is the most incredible discovery ever made!!!"
                f.write(json.dumps(record) + "\n")

            dataset_path = Path(f.name)

        try:
            # Load dataset
            credible, non_credible = load_dataset(dataset_path)
            print(f"✓ Loaded {len(credible)} credible, {len(non_credible)} non-credible")

            # Simulate activations (would come from model in real usage)
            np.random.seed(42)
            hidden_dim = 128

            # Create activations with some signal
            # Credible: higher activation in first dimension
            credible_acts = np.random.randn(len(credible), hidden_dim)
            credible_acts[:, 0] += 2.0

            # Non-credible: lower activation in first dimension
            non_credible_acts = np.random.randn(len(non_credible), hidden_dim)
            non_credible_acts[:, 0] -= 2.0

            print("✓ Generated synthetic activations")

            # Compute vectors
            dom_vec = compute_dom_vector(credible_acts, non_credible_acts)
            lat_vec = compute_lat_vector(credible_acts, non_credible_acts)
            print("✓ Computed DoM and LAT vectors")

            # Check agreement
            agreement = cosine_similarity(dom_vec, lat_vec)
            print(f"✓ DoM-LAT agreement: {agreement:.4f}")

            assert agreement > 0.5, "DoM and LAT should agree"

            # Simulate multi-model results
            results = []
            for model_name in ["llama3", "mistral", "qwen"]:
                # Add some noise to simulate different models
                noise = np.random.randn(hidden_dim) * 0.3
                dom_noisy = dom_vec + noise
                dom_noisy = dom_noisy / np.linalg.norm(dom_noisy)

                noise = np.random.randn(hidden_dim) * 0.3
                lat_noisy = lat_vec + noise
                lat_noisy = lat_noisy / np.linalg.norm(lat_noisy)

                results.append(
                    {
                        "model_id": f"test/{model_name}",
                        "best_layer": 16,
                        "separation": 10.0,
                        "dom_vector": dom_noisy.tolist(),
                        "lat_vector": lat_noisy.tolist(),
                        "dom_lat_similarity": float(
                            cosine_similarity(dom_noisy, lat_noisy)
                        ),
                        "hidden_dim": hidden_dim,
                        "n_layers": 24,
                        "token_pos": -1,
                        "layer_range": [14, 23],
                    }
                )

            # Compute alignment
            alignment = compute_alignment_matrix(results)
            print("✓ Computed cross-model alignment")

            # Create output
            output = {"extractions": results, "alignment": alignment}

            # Save to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(output, f, indent=2)
                output_path = Path(f.name)

            print(f"✓ Saved results to {output_path}")

            # Verify output format
            with output_path.open() as f:
                loaded = json.load(f)

            assert "extractions" in loaded
            assert "alignment" in loaded
            assert len(loaded["extractions"]) == 3

            print("✓ Output format is valid")

            # Print summary
            print("\n=== Test Summary ===")
            print(f"Average DoM similarity: {alignment['avg_dom_similarity']:.4f}")
            print(f"Average LAT similarity: {alignment['avg_lat_similarity']:.4f}")
            print(f"PRH test: {'PASS' if alignment['prh_pass'] else 'FAIL'}")

            output_path.unlink()

        finally:
            dataset_path.unlink()

        print("\n✓ Full integration test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing Phase 2 Vector Extraction Pipeline")
    print("=" * 60)

    test_classes = [
        TestMathematicalFunctions,
        TestDatasetLoading,
        TestAlignmentComputation,
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
