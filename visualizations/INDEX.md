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

Individual S/C/v layer-profile figures for all 8 concepts.
Credibility/negation/sentiment: March 14 corrected runs (canonical, fp32 metrics, 100 pairs).
Remaining five concepts: March 15 expanded run.
March 10 runs retained for comparison only (fp16 bug / 20-pair negation).

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

### Certainty

| Model | Date | Status | File |
|:------|:-----|:-------|:-----|
| gpt2 | 2026-03-15 | **Current** | `certainty_gpt2_2026-03-15.png` |
| gpt2-xl | 2026-03-15 | **Current** | `certainty_gpt2xl_2026-03-15.png` |

### Causation

| Model | Date | Status | File |
|:------|:-----|:-------|:-----|
| gpt2 | 2026-03-15 | **Current** | `causation_gpt2_2026-03-15.png` |
| gpt2-xl | 2026-03-15 | **Current** | `causation_gpt2xl_2026-03-15.png` |

### Moral Valence

| Model | Date | Status | File |
|:------|:-----|:-------|:-----|
| gpt2 | 2026-03-15 | **Current** | `moral_valence_gpt2_2026-03-15.png` |
| gpt2-xl | 2026-03-15 | **Current** | `moral_valence_gpt2xl_2026-03-15.png` |

### Temporal Order

| Model | Date | Status | File |
|:------|:-----|:-------|:-----|
| gpt2 | 2026-03-15 | **Current** | `temporal_order_gpt2_2026-03-15.png` |
| gpt2-xl | 2026-03-15 | **Current** | `temporal_order_gpt2xl_2026-03-15.png` |

### Plurality *(discontinued — negative result)*

| Model | Date | Status | File |
|:------|:-----|:-------|:-----|
| gpt2 | 2026-03-15 | Retained (archival) | `plurality_gpt2_2026-03-15.png` |
| gpt2-xl | 2026-03-15 | Retained (archival) | `plurality_gpt2xl_2026-03-15.png` |

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
| `certainty_gpt2_2026-03-15.png` | Mar 15 | certainty | gpt2 | Current |
| `certainty_gpt2xl_2026-03-15.png` | Mar 15 | certainty | gpt2-xl | Current |
| `causation_gpt2_2026-03-15.png` | Mar 15 | causation | gpt2 | Current |
| `causation_gpt2xl_2026-03-15.png` | Mar 15 | causation | gpt2-xl | Current |
| `moral_valence_gpt2_2026-03-15.png` | Mar 15 | moral_valence | gpt2 | Current |
| `moral_valence_gpt2xl_2026-03-15.png` | Mar 15 | moral_valence | gpt2-xl | Current |
| `temporal_order_gpt2_2026-03-15.png` | Mar 15 | temporal_order | gpt2 | Current |
| `temporal_order_gpt2xl_2026-03-15.png` | Mar 15 | temporal_order | gpt2-xl | Current |
| `plurality_gpt2_2026-03-15.png` | Mar 15 | plurality | gpt2 | Retained — negative result |
| `plurality_gpt2xl_2026-03-15.png` | Mar 15 | plurality | gpt2-xl | Retained — negative result |
