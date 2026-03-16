# Rosetta Manifold — Results

**Last updated:** March 15, 2026
**Dataset version:** 100 pairs per concept (8 concepts)
**Models:** GPT-2, GPT-2-XL, GPT-Neo-125M, GPT-Neo-1.3B, Pythia-160M, Pythia-410M, OPT-125M, OPT-1.3B
**Hardware:** NVIDIA RTX 500 Ada, 4GB VRAM, fp16 forward passes with fp32 metric computation

---

## Experimental History

The results presented here span four generations of runs. The history is documented because it illustrates real issues in small-scale mechanistic interpretability work, and because honest science requires it.

**Run 1 (March 10, CPU fp32):** Original runs on 20 negation pairs and 99 sentiment pairs. Results were qualitatively reasonable but the small negation dataset gave unreliable Fisher separation estimates.

**Run 2 (March 14, GPU fp16 — flawed):** All concepts expanded to 100 pairs. A bug caused fp16 activations to overflow Fisher normalization in deep layers (L32+) of gpt2-xl, producing S=0.0 for layers 32–44 and NaN for 45–47. This silently corrupted the results for all three concepts at gpt2-xl scale.

**Run 3 (March 14, GPU fp16 + fp32 metrics — corrected):** Fixed by casting activations to float32 before metric computation. Forward passes stay in fp16 (fast); Fisher normalization is accurate. Credibility, negation, and sentiment rerun from scratch.

**Run 4 (March 15, expanded — current):** Five new concepts added (certainty, plurality, causation, moral_valence, temporal_order). All eight concepts run across eight model architectures (4 families × 2 scales). 64 successful extractions; Qwen2-1.5B excluded due to TransformerLens BOS token incompatibility.

**The fp32 fix:**
```python
# Before (wrong — fp16 overflows in variance computation at deep layers)
return torch.cat(all_activations, dim=0).numpy()

# After (correct — fp32 for metrics, fp16 for forward passes)
return torch.cat(all_activations, dim=0).float().numpy()
```

---

## Results — GPT-2-XL (48 layers) — Primary Scale

At 48 layers concepts differentiate clearly. All results use 100 contrastive pairs and fp32 metric computation.

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

**Visualizations — expanded run:**

![Depth ordering by concept](visualizations/expanded_depth_ordering.png)

![Cross-architecture consistency](visualizations/expanded_cross_architecture.png)

![Separation strength heatmap](visualizations/expanded_separation_heatmap.png)

### Type-level ordering (mean depth)

| Type | Mean depth | Concepts |
|---|---|---|
| Relational | 76% | temporal_order (75%), causation (77%) |
| Syntactic | 90% | negation (81%), plurality (98%) |
| Affective | 92% | moral_valence (92%), sentiment (92%) |
| Epistemic | 94% | certainty (92%), credibility (96%) |

### What this supports and what it doesn't

**Supported:**
- A broad late-assembly pattern holds: relational and syntactic concepts generally precede affective and epistemic concepts
- Credibility is the most strongly separated concept (S=0.736), consistent with epistemic signals being heavily encoded in training data
- The affective and epistemic clusters are well-separated from relational concepts
- Certainty and temporal_order are architecturally consistent across scales

**Not supported / anomalous:**
- The predicted ordering (syntactic < relational) is **reversed** — relational concepts (causation L37, temporal_order L36) assemble *earlier* than negation (L39), and far earlier than plurality (L47)
- **Plurality is anomalously deep** at L47 (98%) — the deepest concept measured, deeper than credibility. A surface grammatical feature (singular/plural) should not need the model's full depth. This is either a genuine finding about grammatical agreement in gpt2-xl, or a measurement artifact from the contrastive pair design (identical content, only number varies — may be harder to separate geometrically)
- The type ordering is noisy within types: syntactic spans 81–98%, a 17-point range, while affective and epistemic cluster tightly

---

## Results — Small Models (12–24 layers)

All small models show the same floor effect as GPT-2: concepts peak at 50–90% depth with no clear separation between types. The CAZ ordering effect requires sufficient depth to manifest.

| Model | Layers | Credibility | Negation | Sentiment | Notes |
|---|---|---|---|---|---|
| GPT-2 | 12 | L11 (92%) | L10 (83%) | L10 (83%) | All cluster near top |
| GPT-Neo-125M | 12 | L11 (92%) | ~83% | ~83% | Same floor |
| Pythia-160M | 12 | L11 (92%) | L5 (42%) | L5 (42%) | Earlier peaks — different training |
| OPT-125M | 12 | L8 (67%) | L8 (67%) | L8 (67%) | Consistent but shallow |
| GPT-Neo-1.3B | 24 | ~62% | ~57% | ~71% | Partial separation |
| Pythia-410M | 24 | L4 (17%) | L11 (46%) | L13 (54%) | Unusual early peaks |
| OPT-1.3B | 24 | L18 (75%) | L13 (54%) | L15 (63%) | Progressive separation |

**Key observation:** Pythia models show dramatically earlier peaks than other architectures at equivalent scale. Pythia-160M's credibility peaks at L11 (last layer of 12), while Pythia-410M peaks at L4 of 24. This is architecturally interesting — Pythia uses parallel attention and MLP blocks rather than sequential, which may explain the different assembly dynamics.

---

## Architecture Consistency (CAZ Prediction 2)

| Concept | 12L mean | 24L mean | 48L mean | Consistent? |
|---|---|---|---|---|
| temporal_order | 56% | 64% | 75% | **Yes** (spread 19%) |
| certainty | 75% | 75% | 92% | **Yes** (spread 17%) |
| causation | 56% | 58% | 77% | Partial (spread 21%) |
| sentiment | 71% | 71% | 92% | Partial (spread 21%) |
| moral_valence | 67% | 72% | 92% | Partial (spread 25%) |
| negation | 52% | 57% | 81% | Partial (spread 29%) |
| credibility | 86% | 63% | 96% | No (spread 33%) |
| plurality | 33% | 33% | 98% | **No** (spread 65%) |

Certainty and temporal_order show the most stable relative depths across scales. Plurality is the most inconsistent — near-floor at small scales, near-ceiling at gpt2-xl. Credibility's 24-layer mean (63%) is dragged down by Pythia-410M's anomalously early L4 peak.

**Prediction 2 assessment:** The *ordering* (relational/syntactic < affective/epistemic) is broadly consistent across scales, but the absolute depths are not stable. This is expected: CAZ Prediction 2 was stated for same-scale cross-architecture comparison, not across depth scales. The frontier-scale cross-architecture run (same parameter count, different architecture) remains the proper test.

---

## Ablation Results

Ablation at the CAZ peak via orthogonal projection.

| Concept | Model | Separation reduction | KL divergence |
|---|---|---|---|
| Credibility | GPT-2 | 100% | — |
| Credibility | GPT-2-XL | 0% | 0.033 |
| Negation | GPT-2 | 100% | — |
| Negation | GPT-2-XL | 0% | 0.036 |
| Sentiment | GPT-2 | 100% | — |
| Sentiment | GPT-2-XL | 0% | — |

Ablation works at 12-layer scale (GPT-2); does not work at 48-layer scale (GPT-2-XL). The concept direction at 1.5B parameters is distributed across many components and a single-layer projection is insufficient. This is a genuine limitation — the Mid-Stream Ablation Hypothesis (Prediction 1) is not confirmed at gpt2-xl scale with the current approach.

---

## Technical Notes

**fp16 overflow:** Fisher normalization overflows for gpt2-xl at layers 32+. Fix: cast activations to float32 before metric computation. Forward passes stay in fp16.

**TransformerLens patches required for transformers 5.x:**
- Pythia: `rotary_pct` → `rope_parameters.partial_rotary_factor` (see `docs/setup/gpu_setup.md`)
- Qwen2: `rope_theta` moved similarly, plus BOS token incompatibility — **Qwen2 excluded from current run**

**gpt-neo summary names:** TransformerLens uses uppercase model names (e.g. `gpt-neo-125M`) that differ from our lowercase model keys. Analysis scripts use glob matching to handle this.

---

## Summary

| Finding | Status |
|---|---|
| Broad late-assembly pattern (relational/syntactic before affective/epistemic) | **Confirmed** at gpt2-xl scale |
| Specific ordering (syntactic < relational) | **Not confirmed** — reversed in data |
| Credibility strongest separation (S=0.736) | **Confirmed** |
| Plurality anomalously deep (L47, 98%) | **Unexplained** — needs investigation |
| Architecture-stable ordering across model families | **Partially supported** |
| Architecture-stable absolute depths | **Not confirmed** — needs same-scale comparison |
| Mid-Stream Ablation Hypothesis | **Confirmed at GPT-2, not at GPT-2-XL** |
| fp16 metrics valid for deep models | **No** — fp32 required |
| Pythia shows earlier peaks than other architectures | **Observed** — architecture effect |
