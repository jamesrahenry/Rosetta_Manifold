# Rosetta Manifold

**Cross-architecture semantic vector analysis and Concept Assembly Zone empirical validation**

*James Henry — March 2026*

---

## A Note on What This Is

This is independent research by someone learning mechanistic interpretability. It is not affiliated with any institution, has not been peer reviewed, and is not intended to be cited as established work. The code runs, the results are real, and the questions are genuine — but the scale is limited (4GB consumer GPU, GPT-2 family only) and the frontier-scale validation that would make the CAZ findings meaningful is still pending compute access.

If you're here because you work in MI and something looked interesting: great, poke around. If you're here looking for production-ready interpretability tooling: wrong repo.

---

## The Question

The **[Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)** (Huh et al., 2024) proposes that sufficiently capable models converge on similar internal representations of the world, regardless of architecture or training procedure. If true, a semantic concept like "credibility" should produce geometrically similar vectors across Llama, Mistral, Qwen, and GPT families — alignable via rotation, transferable across models.

This project was built to test that. The approach: extract concept directions from multiple architectures using contrastive datasets, align them via Orthogonal Procrustes, and measure convergence.

In the course of doing that, a more specific question emerged: **where** in the network does a concept become stably extractable? Not just whether the direction exists, but when across model depth it crystallizes — and whether that timing is itself architecture-stable. That question became the [**Concept Assembly Zone (CAZ)**](https://github.com/jamesrahenry/Concept_Assembly_Zone) framework, and the bulk of the empirical work here tests it.

The two questions are related. If CAZ position is concept-specific but architecture-stable (Prediction 2 of the CAZ framework), that is itself evidence for the PRH: models don't just converge on *what* they represent, they converge on *when* they assemble it. The frontier-scale cross-architecture work — pending compute — is where both questions get answered together.

---

## Key Findings

Three concepts × two model scales (GPT-2 124M, GPT-2-XL 1.5B), run on consumer hardware (NVIDIA RTX 500 Ada, 4GB VRAM). 100 contrastive pairs per concept. For full results including methodology notes, see [RESULTS.md](RESULTS.md).

**GPT-2-XL (48 layers) — the meaningful scale:**

| Concept | Type | Peak layer | Peak S | Relative depth |
|---|---|---|---|---|
| Negation | Syntactic | L39 / 48 | 0.314 | 81% |
| Sentiment | Affective | L44 / 48 | 0.396 | 92% |
| Credibility | Epistemic | L46 / 48 | 0.736 | 96% |

**GPT-2 (12 layers) — too shallow to differentiate:**
All three concepts peak at L10/12 (83% depth). 12 layers is insufficient depth for the concept-ordering effect to manifest.

**The ordering is as predicted.** Negation assembles mid-network (~81%), sentiment later (~92%), credibility latest and most strongly (~96%). The negation and credibility relative depths are consistent with the GPT-2 results (~83% and ~96%), supporting architecture stability for those two concepts.

**What the data does and doesn't support:**
- ✓ Concept-type ordering (negation < sentiment < credibility) confirmed at gpt2-xl scale
- ✓ Credibility is most strongly separated (S=0.736 vs 0.314–0.396)
- ✓ Negation and credibility relative depths stable across both model scales
- ~ Architecture-stable relative depth (Prediction 2): **partially supported** — ordering holds, sentiment shift (83%→92%) warrants investigation
- ✗ Mid-Stream Ablation Hypothesis: **confirmed at GPT-2, not at GPT-2-XL** — too distributed at 1.5B for single-layer projection

---

## Visualizations

**Comprehensive comparison — all 3 concepts × 2 models (100 pairs each, fp32 metrics):**

![Comprehensive concept comparison](visualizations/COMPREHENSIVE_CONCEPT_COMPARISON.png)

**Credibility:**

| GPT-2 (124M) | GPT-2-XL (1.5B) |
|---|---|
| ![Credibility GPT-2](visualizations/credibility_gpt2_2026-03-14.png) | ![Credibility GPT-2-XL](visualizations/credibility_gpt2xl_2026-03-14.png) |

**Negation:**

| GPT-2 (124M) | GPT-2-XL (1.5B) |
|---|---|
| ![Negation GPT-2](visualizations/negation_gpt2_2026-03-14.png) | ![Negation GPT-2-XL](visualizations/negation_gpt2xl_2026-03-14.png) |

**Sentiment:**

| GPT-2 (124M) | GPT-2-XL (1.5B) |
|---|---|
| ![Sentiment GPT-2](visualizations/sentiment_gpt2_2026-03-14.png) | ![Sentiment GPT-2-XL](visualizations/sentiment_gpt2xl_2026-03-14.png) |

Each figure shows three layer-wise metrics across all transformer blocks:
- **S(l)** — Separation: Fisher-normalized centroid distance between concept classes
- **C(l)** — Coherence: explained variance of the primary PCA component
- **v(l)** — Velocity: rate of change of separation (dS/dLayer)

*March 10 visualizations (20 negation pairs, fp16 metric bug in credibility gpt2-xl) are retained in `visualizations/` for comparison but superseded by the March 14 runs above.*

---

## Pipeline

Three phases, each building on the last:

```
Phase 1  Dataset generation
         Contrastive pairs (credibility, negation, sentiment)
         N=100 pairs per concept, 4 domains each

Phase 2  Vector extraction + CAZ analysis
         DoM (Difference-of-Means) and LAT (Linear Artificial Tomography)
         Layer-wise S/C/v metrics via TransformerLens hooks
         CAZ boundary detection
         Cross-architecture alignment via Orthogonal Procrustes

Phase 3  Ablation validation
         Orthogonal projection to remove concept directions
         Validates: signal removal, KL divergence, cross-architecture transfer
         Mid-Stream Ablation Hypothesis: ablation at CAZ peak is most effective
```

---

## Repository Structure

```
Rosetta_Manifold/
├── src/                        Core pipeline
│   ├── generate_dataset.py       Phase 1: contrastive pair generation
│   ├── generate_negation_dataset.py
│   ├── generate_sentiment_dataset.py
│   ├── extract_vectors.py        Phase 2: DoM/LAT extraction + Procrustes alignment
│   ├── extract_vectors_caz.py    Phase 2: layer-wise CAZ metrics
│   ├── analyze_caz.py            Phase 2: CAZ boundary detection + visualization
│   ├── ablate_vectors.py         Phase 3: orthogonal projection ablation
│   ├── ablate_caz.py             Phase 3: position-specific ablation test
│   ├── align_vectors.py          Cross-architecture Procrustes alignment
│   ├── compare_all_concepts.py   Multi-concept comparison figures
│   └── viz_dom_lat.py            DoM/LAT agreement visualization
├── data/                       Datasets
│   ├── credibility_pairs.jsonl
│   ├── negation_pairs.jsonl
│   └── sentiment_pairs.jsonl
├── tests/                      Test suite
│   ├── test_math_only.py         Dependency-free math tests (CI)
│   ├── test_extract_vectors.py
│   ├── test_ablate_vectors.py
│   ├── test_align_vectors.py
│   └── test_smoke.py
├── visualizations/             Key figures (committed)
├── results/                    Experimental outputs (gitignored large files)
├── scripts/                    Convenience runners
├── docs/                       Usage guides and specifications
│   ├── Spec 1 -- Credibility Contrastive Dataset.md
│   ├── Spec 2 -- Vector Extraction & Alignment Pipeline.md
│   ├── Spec 3 -- Heretic Optimization and Ablation.md
│   ├── setup/                  Hardware and environment setup guides
│   └── archive/                Session logs and interim reports
├── paper/                      Preliminary write-ups and resource proposals
└── experiments/                Jupyter notebooks
```

---

## Quickstart

```bash
git clone https://github.com/jamesrahenry/Rosetta_Manifold
cd Rosetta_Manifold
pip install -r requirements.txt

# Run the full CAZ pipeline on GPT-2
python src/extract_vectors_caz.py --model gpt2
python src/analyze_caz.py --model gpt2
python src/ablate_caz.py --model gpt2

# Or use the convenience script
bash scripts/run_caz_validation.sh

# Run all three concepts on both model scales
bash scripts/run_gpu_rerun.sh
```

**Requirements:** Python 3.11+, PyTorch with CUDA. GPT-2 models download automatically via HuggingFace. All proxy-scale experiments run on a 4GB GPU.

For the full requirements including dataset generation dependencies (OpenAI/Ollama API, Opik tracking), see [`requirements.txt`](requirements.txt). For GPU setup on systems without a system CUDA toolkit installed, see [`docs/setup/gpu_setup.md`](docs/setup/gpu_setup.md).

---

## Tests

```bash
# Math-only tests (no GPU, no model downloads — suitable for CI)
pytest tests/test_math_only.py tests/test_smoke.py -v

# Full test suite (requires torch)
pytest tests/ -v
```

Two known pre-existing failures in `test_extract_vectors.py` (`test_dom_lat_agreement`, `test_full_pipeline_mock`) are caused by low-quality synthetic data in the test fixtures, not code bugs. All math tests pass.

---

## Status

| Component | Status |
|---|---|
| Phase 1: Dataset generation | Complete — 3 concepts, 100 pairs each |
| Phase 2: Vector extraction (fp32 metrics) | Complete — GPT-2 and GPT-2-XL |
| Phase 2: CAZ metrics | Complete — S/C/v across all layers |
| Phase 2: Cross-arch alignment | Implemented — proxy scale only |
| Phase 3: Ablation | Confirmed at GPT-2; not confirmed at GPT-2-XL |
| CAZ Prediction 1 (Mid-Stream Ablation) | **Partial** — GPT-2 only |
| CAZ Prediction 2 (Architecture-Stable depth) | **Not confirmed** — needs same-scale architectures |
| Frontier scale (Llama 3 70B, Qwen 2.5 72B) | Pending compute |
| Cross-architecture PRH validation | Pending frontier scale |
| Publication | Preliminary paper in `paper/` |

See [RESULTS.md](RESULTS.md) for complete results, methodology notes, and an honest account of what the data does and doesn't support.

---

## Related

- [**Concept Assembly Zone**](https://github.com/jamesrahenry/Concept_Assembly_Zone) — the theoretical framework this project tests
- [**Pop Goes the Easel**](https://github.com/jamesrahenry/pop_goes_the_easel) — a companion mechanistic interpretability study using CAZ reference curves

---

## Background

**Platonic Representation Hypothesis** — Huh, Cheung, Wang & Isola (2024): models trained on different data and architectures converge on similar representations of reality. [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)

**Difference-of-Means (DoM)** — Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*

**Linear Artificial Tomography (LAT)** — Zou et al. (2023), *Representation Engineering*

**Orthogonal Procrustes** — cross-architecture alignment: given concept vectors from two models, find the rotation that best maps one to the other. Convergence under rotation is evidence for PRH.

**TransformerLens** — Nanda et al., residual stream hook access for layer-wise metric extraction.

---

*jamesrahenry@henrynet.ca*
