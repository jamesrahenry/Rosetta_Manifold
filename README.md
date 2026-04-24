# Rosetta Manifold

> **Archived 2026-04-24.** This repository is no longer under active development. Methodology and empirics have been superseded by the CAZ Framework and GEM papers in the [Rosetta Program](https://github.com/jamesrahenry/Rosetta_Program).

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

Eight concepts × eight model architectures (4 families × 2 scales), run on consumer hardware (NVIDIA RTX 500 Ada, 4GB VRAM). 100 contrastive pairs per concept. For full results, methodology notes, and experimental history, see [RESULTS.md](RESULTS.md).

**GPT-2-XL (48 layers) — the primary scale:**

![Depth ordering](visualizations/expanded_depth_ordering.png)

| Concept | Type | Peak layer | Peak S | Relative depth |
|---|---|---|---|---|
| temporal_order | relational | L36 / 48 | 0.449 | 75% |
| causation | relational | L37 / 48 | 0.488 | 77% |
| negation | syntactic | L39 / 48 | 0.314 | 81% |
| certainty | epistemic | L44 / 48 | 0.500 | 92% |
| moral_valence | affective | L44 / 48 | 0.294 | 92% |
| sentiment | affective | L44 / 48 | 0.396 | 92% |
| credibility | epistemic | L46 / 48 | 0.736 | 96% |
| plurality | syntactic | L47 / 48 | 0.322 | 98% |

**Type-level mean depths:** relational (76%) < syntactic (90%) < affective (92%) < epistemic (94%)

**What the data supports:**
- ✓ Broad late-assembly pattern: relational and syntactic concepts generally precede affective and epistemic
- ✓ Credibility is the most strongly separated concept (S=0.736)
- ✓ Affective and epistemic clusters (L44–L46) are clearly distinct from relational (L36–L37)
- ✓ Certainty and temporal_order are architecturally consistent across model scales

**What the data doesn't support / anomalies:**
- ✗ Predicted ordering (syntactic < relational) is **reversed** — relational concepts assemble earlier than negation
- ✗ **Plurality is anomalously deep** (L47, 98%) — the deepest concept measured, deeper than credibility; unexplained
- ~ Architecture-stable absolute depths: not confirmed — needs same-scale cross-architecture comparison
- ✗ Mid-Stream Ablation Hypothesis: **confirmed at GPT-2, not at GPT-2-XL**

---

## Visualizations

**Expanded run (March 15) — all 8 concepts × 8 architectures:**

![Depth ordering by concept](visualizations/expanded_depth_ordering.png)

![Cross-architecture consistency](visualizations/expanded_cross_architecture.png)

![Separation strength heatmap](visualizations/expanded_separation_heatmap.png)

**Comprehensive comparison — all 8 concepts × 2 model scales (x-axis normalised to relative depth):**

![Comprehensive concept comparison](visualizations/COMPREHENSIVE_CONCEPT_COMPARISON.png)

![Summary table](visualizations/COMPREHENSIVE_CONCEPT_SUMMARY_TABLE.png)

**Per-concept layer profiles (GPT-2 vs GPT-2-XL):**

| Concept | GPT-2 (124M) | GPT-2-XL (1.5B) |
|---|---|---|
| Credibility | ![Credibility GPT-2](visualizations/credibility_gpt2_2026-03-14.png) | ![Credibility GPT-2-XL](visualizations/credibility_gpt2xl_2026-03-14.png) |
| Negation | ![Negation GPT-2](visualizations/negation_gpt2_2026-03-14.png) | ![Negation GPT-2-XL](visualizations/negation_gpt2xl_2026-03-14.png) |
| Sentiment | ![Sentiment GPT-2](visualizations/sentiment_gpt2_2026-03-14.png) | ![Sentiment GPT-2-XL](visualizations/sentiment_gpt2xl_2026-03-14.png) |
| Certainty | ![Certainty GPT-2](visualizations/certainty_gpt2_2026-03-15.png) | ![Certainty GPT-2-XL](visualizations/certainty_gpt2xl_2026-03-15.png) |
| Causation | ![Causation GPT-2](visualizations/causation_gpt2_2026-03-15.png) | ![Causation GPT-2-XL](visualizations/causation_gpt2xl_2026-03-15.png) |
| Moral valence | ![Moral valence GPT-2](visualizations/moral_valence_gpt2_2026-03-15.png) | ![Moral valence GPT-2-XL](visualizations/moral_valence_gpt2xl_2026-03-15.png) |
| Temporal order | ![Temporal order GPT-2](visualizations/temporal_order_gpt2_2026-03-15.png) | ![Temporal order GPT-2-XL](visualizations/temporal_order_gpt2xl_2026-03-15.png) |
| Plurality *(discontinued)* | ![Plurality GPT-2](visualizations/plurality_gpt2_2026-03-15.png) | ![Plurality GPT-2-XL](visualizations/plurality_gpt2xl_2026-03-15.png) |

*Credibility/negation/sentiment: March 14 corrected runs (fp32 metrics, 100 pairs). Remaining concepts: March 15 expanded run.*

Each per-concept figure shows three layer-wise metrics across all transformer blocks:
- **S(l)** — Separation: Fisher-normalized centroid distance between concept classes
- **C(l)** — Coherence: explained variance of the primary PCA component
- **v(l)** — Velocity: rate of change of separation (dS/dLayer)

*March 10 visualizations (20 negation pairs, fp16 metric bug in credibility gpt2-xl) are retained in `visualizations/` for comparison but superseded by the March 14 runs above.*

---

## Pipeline

Three phases, each building on the last:

```
Phase 1  Dataset generation
         Contrastive pairs (8 concepts: credibility, negation, sentiment,
         certainty, plurality, causation, moral_valence, temporal_order)
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
│   ├── generate_dataset.py       Phase 1: credibility contrastive pair generation
│   ├── generate_negation_dataset.py
│   ├── generate_sentiment_dataset.py
│   ├── generate_new_concepts.py  Phase 1: batch generation for expanded concept set
│   ├── extract_vectors.py        Phase 2: DoM/LAT extraction + Procrustes alignment
│   ├── extract_vectors_caz.py    Phase 2: layer-wise CAZ metrics (HF native)
│   ├── extract_caz_frontier.py   Phase 2: multi-concept frontier extraction (H100-ready)
│   ├── analyze_caz.py            Phase 2: CAZ boundary detection + visualization
│   ├── analyze_expanded_caz.py   Phase 2: expanded run analysis across all concepts
│   ├── ablate_vectors.py         Phase 3: orthogonal projection ablation
│   ├── ablate_caz.py             Phase 3: position-specific ablation test
│   ├── align_vectors.py          Cross-architecture Procrustes alignment
│   ├── compare_all_concepts.py   Multi-concept comparison figures
│   └── viz_dom_lat.py            DoM/LAT agreement visualization
├── data/                       Datasets (8 concepts, 100 pairs each)
│   ├── credibility_pairs.jsonl
│   ├── negation_pairs.jsonl
│   ├── sentiment_pairs.jsonl
│   ├── certainty_pairs.jsonl
│   ├── plurality_pairs.jsonl
│   ├── causation_pairs.jsonl
│   ├── moral_valence_pairs.jsonl
│   └── temporal_order_pairs.jsonl
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
│   ├── Spec 4 -- Procrustes Alignment and Cross-Architecture Validation.md
│   ├── PORTING_STATUS.md         TransformerLens → rosetta_tools migration status
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
| Phase 1: Dataset generation | Complete — 8 concepts, 100 pairs each |
| Phase 2: Vector extraction (fp32 metrics) | Complete — 8 concepts × 8 model architectures |
| Phase 2: CAZ metrics | Complete — S/C/v across all layers |
| Phase 2: Cross-arch alignment | Implemented — proxy scale only |
| Phase 3: Ablation | Confirmed at GPT-2; not confirmed at GPT-2-XL |
| CAZ Prediction 1 (Mid-Stream Ablation) | **Partial** — GPT-2 only |
| CAZ Prediction 2 (Architecture-Stable depth) | **Partially supported** — ordering holds, absolute depths need same-scale test |
| Plurality anomaly | **Unexplained** — anomalously deep at gpt2-xl scale |
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
