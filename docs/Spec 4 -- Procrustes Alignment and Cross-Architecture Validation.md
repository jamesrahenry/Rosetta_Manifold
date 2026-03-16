# Spec 4 — Procrustes Alignment and Cross-Architecture Validation

**Cross-Architecture Geometric Mapping for Concept Assembly Analysis**

**Pipeline position:** Phase 3 — follows CAZ extraction (Spec 2) and ablation (Spec 3).
Requires: CAZ peak layers identified per concept per model.
Implements: `src/align_vectors.py`
Depends on: `rosetta_tools.caz`, `scipy.linalg.orthogonal_procrustes`
Status: Pending — awaiting frontier-scale activation data (H100 run)

---

## 1. The Alignment Problem

Different LLM architectures distribute semantic concepts across incommensurable coordinate systems. A credibility vector in GPT-2's 768-dimensional residual stream cannot be directly compared to one in GPT-Neo's 2048-dimensional space — cosine similarity is mathematically undefined across these bases.

Establishing that conceptual manifolds are *topologically similar* across architectures is insufficient. Topology permits arbitrary stretching and warping; a torus and a coffee mug are topologically identical. Mechanistic interpretability requires a stronger claim: that the **structural relationships between concepts** — the geometric distance between "truthful" and "deceptive," the angle between negation and affirmation — are preserved across architectures under rigid transformation. Orthogonal Procrustes Analysis provides the formal test.

---

## 2. Orthogonal Procrustes Analysis

Procrustes superimposition determines whether two point sets are geometrically identical by finding the optimal rigid alignment between them. It permits only three operations:

1. **Translation** — mean-centering both matrices
2. **Uniform scaling** — normalizing overall magnitude
3. **Orthogonal rotation/reflection** — spinning without warping

The orthogonality constraint is the load-bearing part. It prevents the alignment from "cheating" by deforming the geometry to manufacture agreement. If two concept manifolds align well under these constraints, that alignment is evidence of genuine structural equivalence, not fitting artifact.

---

## 3. The Mathematics

Given a source matrix $A$ (e.g., GPT-2 activations) and target matrix $B$ (e.g., GPT-Neo activations), we seek the orthogonal transformation $\Omega$ that minimizes:

$$\Omega^* = \arg\min_{\Omega} \| A\Omega - B \|_F \quad \text{subject to} \quad \Omega^T\Omega = I$$

The closed-form solution via SVD:

1. Compute the cross-covariance matrix: $M = B^T A$
2. Decompose: $U, \Sigma, V^T = \text{SVD}(M)$
3. Recover the optimal rotation: $\Omega^* = UV^T$

The residual $\|A\Omega^* - B\|_F$ is the **disparity score** — the quantity of interest.

---

## 4. Implementation in the Rosetta Framework

**Step 1 — Extraction.** Extract $n$ contrastive concept pairs (e.g., 100 credibility-scored sentence pairs) from both models at their respective CAZ peak layers. Peak layer is determined by maximum Fisher-normalized separation, as established in prior Rosetta experiments.

**Step 2 — Dimensionality matching.** If $d_A \neq d_B$, apply PCA independently to each activation set and project into a shared $k$-dimensional space capturing ≥95% of variance in each. Both matrices are then $n \times k$.

> **Caution on the PCA step.** Independent 95%-variance thresholds may capture structurally different content across architectures — GPT-2's principal components may weight syntactic structure while a larger model's weight semantic content. Run sensitivity analysis across $k$ values and report disparity scores across the range, not just at the nominal threshold.

**Step 3 — Procrustes superimposition.** Compute $\Omega^*$ and align the source geometry to the target.

**Step 4 — Disparity scoring.** Report the Frobenius-norm residual. Low disparity is evidence that the concept is represented as a geometry-preserving structure across architectures.

```python
import numpy as np
from scipy.linalg import orthogonal_procrustes

# A, B: mean-centered, dimension-matched activation matrices
# A: GPT-2 concept manifold at CAZ peak layer
# B: GPT-Neo concept manifold at CAZ peak layer

R, scale = orthogonal_procrustes(A, B)
A_aligned = A @ R
disparity = np.linalg.norm(A_aligned - B, ord='fro')

print(f"Disparity: {disparity:.4f}")
print(f"Scale factor: {scale:.4f}")  # Large scale differences are themselves informative
```

---

## 5. What Disparity Scores Actually Tell You

Low disparity establishes that the geometric relationship between your specific contrastive pairs is preserved under rigid transformation. This is meaningful but narrower than "architecture-agnostic structure." That stronger claim requires:

**Cross-concept rotation transfer.** Fit $\Omega^*$ on credibility pairs. Apply the *same* rotation matrix to negation pairs without refitting. If disparity remains low, the rotation generalizes — meaning the two models have organized their concept spaces isomorphically, not just aligned on a single concept. This is the stronger universality test, and to our knowledge it has not appeared in the MI literature.

**Examine the rotation itself.** A large rotation with low disparity is more interesting than a small rotation with low disparity. The former means both models encoded the same structure but oriented it differently in their embedding spaces — genuine isomorphism. The scale factor is also informative: large scale differences suggest one model's representations are more diffuse in concept space, which connects to the signal-strength vs. separability dissociation observed in ablation experiments.

**Held-out validation.** Fit $\Omega^*$ on a training split of concept pairs and compute disparity on held-out pairs. Overfitting is less of a concern with orthogonal Procrustes than with general alignment methods (the constraint space is rigid), but it remains worth checking.

---

## 6. Relationship to Prior Work

Procrustes distance appears in MI-adjacent work — notably in the Frame Representation Hypothesis as a metric on Grassmann manifolds, and in recent embedding interoperability literature. It is not, however, standard MI vocabulary in the way that Difference-of-Means, LAT, or CKA are. The contribution here is not the tool but the application: using Procrustes disparity as a quantitative test for architecture-invariance of concept geometry at CAZ peak layers, with cross-concept rotation transfer as a stronger universality criterion.

This complements CKA-based cross-architecture similarity (which operates on second-order statistics across full datasets) by providing pair-level geometric alignment with interpretable residuals.
