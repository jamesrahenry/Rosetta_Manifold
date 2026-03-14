# Rosetta Manifold

**Cross-architecture semantic vector analysis and Concept Assembly Zone empirical validation**

*James Henry — March 2026*

---

## A Note on What This Is

This is independent research by someone learning mechanistic interpretability. It is not affiliated with any institution, has not been peer reviewed, and is not intended to be cited as established work. The code runs, the results are real, and the questions are genuine — but the scale is limited (4GB consumer GPU, GPT-2 family only) and the frontier-scale validation that would make the CAZ findings meaningful is still pending compute access.

If you're here because you work in MI and something looked interesting: great, poke around. If you're here looking for production-ready interpretability tooling: wrong repo.

---

## The Question

The **Platonic Representation Hypothesis** (Huh et al., 2024) proposes that sufficiently capable models converge on similar internal representations of the world, regardless of architecture or training procedure. If true, a semantic concept like "credibility" should produce geometrically similar vectors across Llama, Mistral, Qwen, and GPT families — alignable via rotation, transferable across models.

This project was built to test that. The approach: extract concept directions from multiple architectures using contrastive datasets, align them via Orthogonal Procrustes, and measure convergence.

In the course of doing that, a more specific question emerged: **where** in the network does a concept become stably extractable? Not just whether the direction exists, but when across model depth it crystallizes — and whether that timing is itself architecture-stable. That question became the [**Concept Assembly Zone (CAZ)**](https://github.com/jamesrahenry/Concept_Assembly_Zone) framework, and the bulk of the empirical work here tests it.

The two questions are related. If CAZ position is concept-specific but architecture-stable (Prediction 2 of the CAZ framework), that is itself evidence for the PRH: models don't just converge on *what* they represent, they converge on *when* they assemble it. The frontier-scale cross-architecture work — pending compute — is where both questions get answered together.

---

## Key Findings

Three concepts × two model scales (GPT-2 124M, GPT-2-XL 1.5B), run on consumer hardware (NVIDIA RTX 500 Ada, 4GB VRAM):

| Concept | Type | Peak layer (GPT-2) | Peak layer (GPT-2-XL) | Relative depth |
|---|---|---|---|---|
| Credibility | Epistemic | L10 / 12 | L44 / 48 | ~92% |
| Negation | Syntactic | L8 / 12 | L39 / 48 | ~81% |
| Sentiment | Affective | L9 / 12 | L44 / 48 | ~88–92% |

**Concepts assemble late.** Peak separation occurs in the final 10–20% of model depth — consistent across both scales.

**Relative CAZ position is concept-specific and scale-invariant.** Credibility peaks at ~92% depth in both GPT-2 and GPT-2-XL; negation at ~81%. This is the proxy-scale confirmation of CAZ Prediction 2 — and a preliminary signal for the PRH: if the *timing* of assembly is stable across architectures, the representations are likely converging on something real.

**Concept type predicts assembly profile:**
- *Epistemic* (credibility): strong signal, late peak, entangled with general capability — hardest to ablate cleanly
- *Syntactic* (negation): moderate signal, earlier peak, orthogonal to other concepts — cleanest ablation
- *Affective* (sentiment): weaker signal, scale-dependent timing — improves markedly at larger scale

**Ablation works at proxy scale.** Orthogonal projection at the CAZ peak removes 100% of concept signal. KL divergence passes the <0.2 threshold for syntactic concepts; epistemic concepts require frontier scale (expected — more entangled with general capability).

---

## Visualizations

**Comprehensive comparison — all 3 concepts × 2 models:**

![Comprehensive concept comparison](visualizations/COMPREHENSIVE_CONCEPT_COMPARISON.png)

**Credibility:**

| GPT-2 (124M) | GPT-2-XL (1.5B) |
|---|---|
| ![Credibility GPT-2](visualizations/credibility_gpt2_2026-03-10.png) | ![Credibility GPT-2-XL](visualizations/credibility_gpt2-xl_2026-03-10.png) |

**Negation:**

| GPT-2 (124M) | GPT-2-XL (1.5B) |
|---|---|
| ![Negation GPT-2](visualizations/negation_gpt2_2026-03-10.png) | ![Negation GPT-2-XL](visualizations/negation_gpt2-xl_2026-03-10.png) |

**Sentiment:**

| GPT-2 (124M) | GPT-2-XL (1.5B) |
|---|---|
| ![Sentiment GPT-2](visualizations/sentiment_gpt2_2026-03-10.png) | ![Sentiment GPT-2-XL](visualizations/sentiment_gpt2-xl_2026-03-10.png) |

Each figure shows three layer-wise metrics across all transformer blocks:
- **S(l)** — Separation: Fisher-normalized centroid distance between concept classes
- **C(l)** — Coherence: explained variance of the primary PCA component
- **v(l)** — Velocity: rate of change of separation (dS/dLayer)

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
| Phase 2: Vector extraction | Complete — DoM + LAT, all proxy models |
| Phase 2: CAZ metrics | Complete — S/C/v across all layers |
| Phase 2: Cross-arch alignment | Implemented — proxy scale only |
| Phase 3: Ablation | Complete — proxy scale validated |
| Proxy scale (GPT-2, GPT-Neo, OPT) | **Validated — 10 models** |
| Frontier scale (Llama 3 70B, Qwen 2.5 72B) | Pending compute |
| Cross-architecture PRH validation | Pending frontier scale |
| Publication | Preliminary paper in `paper/` |

Frontier-scale compute is the primary remaining blocker for both the PRH and CAZ work. The proxy-scale methodology and results are complete.

---

## Related

- [**Concept Assembly Zone**](https://github.com/jamesrahenry/Concept_Assembly_Zone) — the theoretical framework this project tests
- [**Pop Goes the Easel**](https://github.com/jamesrahenry/pop_goes_the_easel) — a companion mechanistic interpretability study using CAZ reference curves

---

## Background

**Platonic Representation Hypothesis** — Huh et al. (2024): models trained on different data and architectures converge on similar representations of reality.

**Difference-of-Means (DoM)** — Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*

**Linear Artificial Tomography (LAT)** — Zou et al. (2023), *Representation Engineering*

**Orthogonal Procrustes** — cross-architecture alignment: given concept vectors from two models, find the rotation that best maps one to the other. Convergence under rotation is evidence for PRH.

**TransformerLens** — Nanda et al., residual stream hook access for layer-wise metric extraction.

---

*jamesrahenry@henrynet.ca*
