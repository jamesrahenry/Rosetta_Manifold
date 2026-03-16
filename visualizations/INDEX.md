# Rosetta Manifold - Visualization Index

All CAZ (Concept Assembly Zone) visualizations organized by run and concept.

---

## Expanded Run (March 15) — All 8 Concepts × 8 Architectures

These are the current primary visualizations. All eight concepts, all eight model
architectures (4 families × 2 scales), 100 pairs per concept, fp32 metrics.

| File | What it shows |
|:-----|:--------------|
| `expanded_depth_ordering.png` | Concept peak depths sorted by relative depth at GPT-2-XL scale — shows relational/syntactic/affective/epistemic ordering |
| `expanded_cross_architecture.png` | CAZ peak depth consistency across 8 model architectures for each concept |
| `expanded_separation_heatmap.png` | Separation strength (peak S) heatmap — concept × architecture |

---

## Comprehensive Comparison (March 14) — Original 3 Concepts × 2 Models

**File**: `COMPREHENSIVE_CONCEPT_COMPARISON.png`

**What it shows**: Single-page visualization with all 3 concepts × 2 models = 6 datasets
(credibility, negation, sentiment × GPT-2, GPT-2-XL)

**Panels**:
1. **Separation Trajectories** - Concept strength across layers
2. **Coherence Trajectories** - Concept stability/consistency
3. **Velocity Trajectories** - Rate of concept formation
4. **Summary Statistics Table** - All key metrics

---

## Per-Concept Layer Profiles

Individual S/C/v layer-profile figures for credibility, negation, and sentiment.
March 14 runs are the canonical versions (fp32 metrics, 100 pairs).
March 10 runs are retained for comparison only (fp16 bug in credibility gpt2-xl,
20-pair negation dataset).

### Credibility

| Model | Date | Status | File |
|:------|:-----|:-------|:-----|
| gpt2 | 2026-03-14 | **Current** | `credibility_gpt2_2026-03-14.png` |
| gpt2-xl | 2026-03-14 | **Current** | `credibility_gpt2xl_2026-03-14.png` |
| gpt2 | 2026-03-10 | Superseded | `credibility_gpt2_2026-03-10.png` |
| gpt2-xl | 2026-03-10 | Superseded (fp16 bug) | `credibility_gpt2-xl_2026-03-10.png` |

### Negation

| Model | Date | Status | File |
|:------|:-----|:-------|:-----|
| gpt2 | 2026-03-14 | **Current** | `negation_gpt2_2026-03-14.png` |
| gpt2-xl | 2026-03-14 | **Current** | `negation_gpt2xl_2026-03-14.png` |
| gpt2 | 2026-03-10 | Superseded (20 pairs) | `negation_gpt2_2026-03-10.png` |
| gpt2-xl | 2026-03-10 | Superseded (20 pairs) | `negation_gpt2-xl_2026-03-10.png` |

### Sentiment

| Model | Date | Status | File |
|:------|:-----|:-------|:-----|
| gpt2 | 2026-03-14 | **Current** | `sentiment_gpt2_2026-03-14.png` |
| gpt2-xl | 2026-03-14 | **Current** | `sentiment_gpt2xl_2026-03-14.png` |
| gpt2 | 2026-03-10 | Superseded | `sentiment_gpt2_2026-03-10.png` |
| gpt2-xl | 2026-03-10 | Superseded | `sentiment_gpt2-xl_2026-03-10.png` |

---

## All Files

| File | Run | Concept | Model | Notes |
|:-----|:----|:--------|:------|:------|
| `expanded_depth_ordering.png` | Mar 15 | All 8 | All 8 arch | Primary result figure |
| `expanded_cross_architecture.png` | Mar 15 | All 8 | All 8 arch | Architecture consistency |
| `expanded_separation_heatmap.png` | Mar 15 | All 8 | All 8 arch | Separation strength heatmap |
| `COMPREHENSIVE_CONCEPT_COMPARISON.png` | Mar 14 | 3 | GPT-2, GPT-2-XL | Multi-panel summary |
| `credibility_gpt2_2026-03-14.png` | Mar 14 | credibility | gpt2 | Current |
| `credibility_gpt2xl_2026-03-14.png` | Mar 14 | credibility | gpt2-xl | Current |
| `negation_gpt2_2026-03-14.png` | Mar 14 | negation | gpt2 | Current |
| `negation_gpt2xl_2026-03-14.png` | Mar 14 | negation | gpt2-xl | Current |
| `sentiment_gpt2_2026-03-14.png` | Mar 14 | sentiment | gpt2 | Current |
| `sentiment_gpt2xl_2026-03-14.png` | Mar 14 | sentiment | gpt2-xl | Current |
| `credibility_gpt2_2026-03-10.png` | Mar 10 | credibility | gpt2 | Superseded |
| `credibility_gpt2-xl_2026-03-10.png` | Mar 10 | credibility | gpt2-xl | Superseded — fp16 bug |
| `negation_gpt2_2026-03-10.png` | Mar 10 | negation | gpt2 | Superseded — 20 pairs |
| `negation_gpt2-xl_2026-03-10.png` | Mar 10 | negation | gpt2-xl | Superseded — 20 pairs |
| `sentiment_gpt2_2026-03-10.png` | Mar 10 | sentiment | gpt2 | Superseded |
| `sentiment_gpt2-xl_2026-03-10.png` | Mar 10 | sentiment | gpt2-xl | Superseded |
