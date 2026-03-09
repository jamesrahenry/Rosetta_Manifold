"""
test_align_vectors.py

Unit tests for Procrustes alignment.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from align_vectors import compute_procrustes_alignment


class TestProcrustes:
    """Test Orthogonal Procrustes alignment."""

    def test_identity_alignment(self):
        """Test that aligning identical matrices yields identity rotation."""
        np.random.seed(42)
        A = np.random.randn(100, 10)
        B = A.copy()

        R = compute_procrustes_alignment(A, B)

        # R should be identity
        assert np.allclose(R, np.eye(10), atol=1e-6)

    def test_rotation_recovery(self):
        """Test that we can recover a known rotation."""
        np.random.seed(42)
        # Create random orthogonal matrix (rotation)
        Q, _ = np.linalg.qr(np.random.randn(10, 10))
        
        # Source data is Identity
        A = np.eye(10)
        
        # Target data is rotated source: B = A @ Q.T = Q.T
        # So B @ Q = A
        B = Q.T

        # Compute alignment: find R such that target @ R ≈ source
        # Here source=A, target=B
        R = compute_procrustes_alignment(A, B)

        # Should recover Q
        assert np.allclose(R, Q, atol=1e-6)

    def test_vector_alignment(self):
        """Test that aligning vectors works as expected."""
        np.random.seed(42)
        # Create random rotation
        Q, _ = np.linalg.qr(np.random.randn(10, 10))
        
        # Source vector
        v_source = np.random.randn(10)
        v_source /= np.linalg.norm(v_source)
        
        # Target vector (rotated)
        # v_target = v_source @ Q.T
        # So v_target @ Q = v_source
        v_target = v_source @ Q.T
        
        # Create dummy activations that follow the same rotation
        A = np.random.randn(100, 10)
        B = A @ Q.T
        
        # Compute alignment
        R = compute_procrustes_alignment(A, B)
        
        # Align target vector
        v_aligned = v_target @ R
        
        # Should match source
        similarity = np.dot(v_source, v_aligned)
        assert np.isclose(similarity, 1.0, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
