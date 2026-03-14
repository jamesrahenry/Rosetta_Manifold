# Comprehensive Concept Comparison Visualization Guide

## Overview

**File**: `visualizations/COMPREHENSIVE_CONCEPT_COMPARISON.png`

A single-page visualization showing all 6 datasets (3 concepts × 2 models) across all key metrics.

---

## Layout (4 Panels)

### Panel 1 (Top Left): **Separation Trajectories**
**What it shows**: How concepts assemble (become distinguishable) across layers

**Y-axis**: Separation (Fisher-normalized)
- Higher = stronger geometric encoding
- Shows concept strength

**X-axis**: Layer number (0 to 11 for GPT-2, 0 to 47 for GPT-2 XL)

**Color coding**:
- 🔵 **Blue** = Credibility (epistemic concept)
- 🟣 **Purple** = Negation (syntactic concept)
- 🟠 **Orange** = Sentiment (affective concept)

**Line style**:
- **Solid (—)** = GPT-2 (12 layers)
- **Dashed (---)** = GPT-2 XL (48 layers)

**Key observations**:
- Credibility strongest (highest curves)
- Negation moderate (middle curves)
- Sentiment weakest (lowest curves)
- All concepts strengthen in larger model (dashed lines higher than solid)

---

### Panel 2 (Top Right): **Coherence Trajectories**
**What it shows**: How consistent the concept representation is across examples

**Y-axis**: Coherence (Concept Consistency)
- Measures: How similar are activations within same class?
- Higher = more consistent/stable representation
- Lower = more variable/distributed

**X-axis**: Layer number

**Same color/line coding** as Panel 1

**Key observations**:
- Coherence generally increases with depth
- Sharp spikes at final layers (especially credibility)
- Credibility shows highest final coherence (most stable)
- Sentiment remains low coherence (distributed representation)

---

### Panel 3 (Bottom Left): **Velocity Trajectories**
**What it shows**: Rate of concept formation (how fast separation changes)

**Y-axis**: Velocity (Rate of Change)
- Positive = concept assembling (separation increasing)
- Negative = concept dissolving (separation decreasing)
- Zero line = stable

**X-axis**: Layer number

**Same color/line coding** as Panel 1

**Key observations**:
- **Credibility**: High velocity spike at final layer (rapid late assembly)
- **Negation**: Peaks mid-model, then negative velocity (peaks early, declines)
- **Sentiment**: Gradual assembly (slow, steady velocity)
- Reveals **timing differences** between concept types

---

### Panel 4 (Bottom Right): **Summary Statistics Table**

Compact reference showing all key metrics for all 6 datasets:

| Column | Description |
|:-------|:------------|
| **Concept** | Credibility / Negation / Sentiment |
| **Model** | 12L (GPT-2) or 48L (GPT-2 XL) |
| **Peak L** | Layer where separation peaks (% of total depth) |
| **Peak Sep** | Maximum separation value |
| **Final Sep** | Separation at final layer |
| **Type** | Concept taxonomy (Epistemic/Syntactic/Affective) |

**Reading the table**:
- Rows alternate white/gray for readability
- Grouped by concept (Credibility first, then Negation, then Sentiment)
- Within each concept: 12L model first, then 48L model

---

## How to Read This Visualization

### For Quick Insights:
1. **Concept Strength**: Look at Panel 1 separation heights
   - Higher = stronger encoding

2. **Concept Timing**: Look at Panel 3 velocity peaks
   - Early peak = mid-layer concept (syntactic)
   - Late peak = late-layer concept (epistemic/affective)

3. **Concept Stability**: Look at Panel 2 coherence
   - High = concentrated (epistemic)
   - Low = distributed (affective)

### For Deep Analysis:
1. **Compare same concept across models**:
   - Find solid line (GPT-2) vs dashed line (GPT-2 XL) of same color
   - Shows scale effects

2. **Compare different concepts in same model**:
   - Find lines with same style but different colors
   - Shows concept type differences

3. **Cross-reference with table**:
   - Use table to get exact peak values
   - Check peak timing percentages

---

## Key Findings Visible in This Visualization

### 1. Concept Strength Hierarchy (Panel 1)
**GPT-2 XL peak separations**:
1. Credibility: 0.772 (strongest)
2. Negation: 0.434 (moderate)
3. Sentiment: 0.372 (weakest)

This hierarchy is **consistent across both model sizes**.

### 2. Concept Type Determines Assembly Pattern (Panels 1 & 3)

**Epistemic concepts** (Credibility):
- Strong separation (Panel 1: highest curves)
- Late-layer peak (~92% depth from table)
- Sharp velocity spike at end (Panel 3: tall spike)
- High final coherence (Panel 2: highest end values)
- **Pattern**: Strong Late-Sustaining

**Syntactic concepts** (Negation):
- Moderate separation (Panel 1: middle curves)
- Mid-layer peak (~81% depth from table)
- Early velocity peak then decline (Panel 3: mid-peak, goes negative)
- Moderate coherence (Panel 2: middle range)
- **Pattern**: Moderate Mid-Declining

**Affective concepts** (Sentiment):
- Weak separation (Panel 1: lowest curves)
- Late-layer peak (~92% depth from table, like credibility!)
- Gradual velocity (Panel 3: slow steady rise)
- Low coherence (Panel 2: stays low, distributed)
- **Pattern**: Weak Late-Sustaining

### 3. Scale Effects (Compare solid vs dashed lines)

**All concepts strengthen with scale**:
- GPT-2 → GPT-2 XL improvements (Panel 1):
  - Credibility: +11% (0.695 → 0.772)
  - Negation: +5% (0.412 → 0.434)
  - Sentiment: +13% (0.329 → 0.372)

**Timing patterns preserved**:
- Credibility peaks at 92% in both models
- Negation peaks at ~81-83% in both models
- Sentiment shifts from 83% → 92% (needs larger model!)

### 4. Co-Location Discovery (Panel 1 dashed lines)

**Credibility and Sentiment peak at same layer in GPT-2 XL**:
- Both peak at Layer 44 (92% depth)
- But very different separation strengths
- Suggests **late-layer semantic processing hub**

**Negation peaks earlier**:
- Layer 39 (81% depth)
- Suggests **mid-layer syntactic processing**

---

## Common Interpretation Patterns

### "Spiky" Coherence (Panel 2)
Sharp coherence spikes at final layers (especially credibility) indicate:
- Concept crystallization
- Final decision-making layer
- High confidence in classification

### Negative Velocity (Panel 3)
When velocity goes negative (negation in final layers):
- Concept strength decreasing
- Being "dissolved" for next stage
- Suggests concept served its purpose mid-model

### Flat Coherence (Panel 2)
Low, flat coherence (sentiment) indicates:
- Distributed representation
- Not concentrated in single direction
- Harder to extract/manipulate

---

## Use Cases

### For Presentations:
- **Single slide** showing complete story
- Color-coded for easy concept identification
- Table provides exact numbers for Q&A

### For Papers:
- **Figure 1** in results section
- Shows methodology validation across concepts
- Demonstrates concept taxonomy framework

### For Future Work:
- **Template** for adding new concepts
- Easy to see where new concept fits in hierarchy
- Compare to these 3 established baselines

### For Sharing:
- **Self-contained** (no need for folder context)
- All 6 datasets in one place
- Clear legends and labels

---

## Technical Details

### Generation:
```bash
python src/compare_all_concepts.py
```

### Output locations:
- `results/COMPREHENSIVE_CONCEPT_COMPARISON.png` (timestamped results)
- `visualizations/COMPREHENSIVE_CONCEPT_COMPARISON.png` (canonical location)

### Data sources:
All 6 CAZ analysis JSON files:
1. `results/caz_validation_gpt2_20260310_164336/caz_analysis_gpt2.json`
2. `results/caz_validation_gpt2-xl_20260310_193156/caz_analysis_gpt2-xl.json`
3. `results/negation_gpt2_20260310_210541/caz_analysis_gpt2.json`
4. `results/negation_gpt2xl_20260310_210541/caz_analysis_gpt2-xl.json`
5. `results/20260310_233429_sentiment_gpt2/caz_analysis_gpt2.json`
6. `results/20260310_233429_sentiment_gpt2xl/caz_analysis_gpt2-xl.json`

### Metrics plotted:
- **Separation**: `layer_metrics[i]["separation_fisher"]`
- **Coherence**: `layer_metrics[i]["coherence"]`
- **Velocity**: `layer_metrics[i]["velocity"]`

### Resolution:
- 150 DPI (publication quality)
- 18" × 11" (large format for detail)

---

## Future Enhancements

When scaling to 100+ concepts:

1. **Interactive version**: Plotly/Bokeh for hover tooltips
2. **Filterable dashboard**: Select concepts/models to compare
3. **Heatmap view**: Concept × Model grid showing peak separations
4. **Animation**: Show layer-by-layer assembly progression
5. **Differential analysis**: Highlight concept pairs (e.g., credibility vs honesty)

---

## Quick Reference

**Want to see...** | **Look at...**
:------------------|:---------------
Which concept is strongest? | Panel 1 (highest curves)
When does concept form? | Panel 3 (velocity peaks) & Table (Peak L)
How stable is concept? | Panel 2 (coherence height)
Effect of model scale? | Compare solid vs dashed (same color)
Exact peak values? | Panel 4 table (Peak Sep column)
Concept taxonomy? | Panel 4 table (Type column)

---

## Summary

✅ **All 6 datasets on one page**
✅ **Three complementary metrics** (separation, coherence, velocity)
✅ **Clear visual hierarchy** (color + line style)
✅ **Exact statistics in table**
✅ **Publication-ready quality**

This single visualization captures the complete story of the Rosetta Manifold concept comparison research!

---

*Generated: 2026-03-11*
*Part of: Rosetta Manifold - Transferable AI Interpretability Research*
