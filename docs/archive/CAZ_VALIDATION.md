# CAZ (Concept Assembly Zone) Validation

Empirical validation of the **Concept Assembly Zone** framework (Henry, 2026) using the Rosetta_Manifold infrastructure.

## Overview

The CAZ framework proposes that semantic concepts don't peak at a single "best layer" — they **assemble** across a contiguous zone of middle-to-late layers. This pipeline empirically tests that hypothesis.

## The Pipeline

### Phase 1: Layer-Wise Metric Extraction
**Script**: `src/extract_vectors_caz.py`

Tracks three metrics across ALL layers (not just best layer):
- **Separation (S)**: Fisher-normalized centroid distance
- **Coherence (C)**: Explained variance of primary component
- **Velocity (V)**: Rate of change of separation

```bash
python src/extract_vectors_caz.py \
    --model gpt2 \
    --dataset data/credibility_pairs_tiny.jsonl \
    --output results/caz_extraction.json
```

### Phase 2: CAZ Boundary Detection
**Script**: `src/analyze_caz.py`

Programmatically identifies:
- **CAZ Start**: Where concept formation begins
- **CAZ Peak**: Maximum separation point
- **CAZ End**: Where assembly completes
- **Pre-CAZ** and **Post-CAZ** regions

Produces visualization showing S, C, V across layers with CAZ boundaries marked.

```bash
python src/analyze_caz.py \
    --input results/caz_extraction.json \
    --output-dir results
```

### Phase 3: Mid-Stream Ablation Hypothesis Test
**Script**: `src/ablate_caz.py`

Tests the key CAZ prediction: **Ablation within the CAZ produces better behavioral suppression with lower collateral damage than post-CAZ ablation.**

Compares ablation at five positions:
1. CAZ Start (early assembly)
2. CAZ Mid (during assembly)
3. CAZ Peak (maximum separation)
4. CAZ End (late assembly)
5. Post-CAZ (after assembly complete)

For each position, measures:
- **Separation Reduction**: Behavioral suppression (higher = better)
- **KL Divergence**: Capability preservation (lower = better)

**Hypothesis**: CAZ-mid should achieve high reduction + low KL compared to post-CAZ.

```bash
python src/ablate_caz.py \
    --model gpt2 \
    --caz-analysis results/caz_analysis_gpt2.json \
    --dataset data/credibility_pairs_tiny.jsonl \
    --output results/caz_ablation_comparison.json
```

## Quick Start

### Run Full Pipeline (Automated)

```bash
# Quick test on GPT-2 with tiny dataset
./run_caz_validation.sh gpt2

# Full test with complete credibility dataset
./run_caz_validation.sh gpt2 --full-dataset

# Test on larger model
./run_caz_validation.sh gpt2-xl
```

### Manual Step-by-Step

```bash
# 1. Extract layer-wise metrics
python src/extract_vectors_caz.py --model gpt2 \
    --dataset data/credibility_pairs_tiny.jsonl \
    --output results/caz_extraction.json

# 2. Analyze CAZ boundaries
python src/analyze_caz.py \
    --input results/caz_extraction.json \
    --output-dir results

# 3. Test ablation hypothesis
python src/ablate_caz.py \
    --model gpt2 \
    --caz-analysis results/caz_analysis_gpt2.json \
    --dataset data/credibility_pairs_tiny.jsonl \
    --output results/caz_ablation_comparison.json
```

## Expected Results

### What the Visualization Shows

The `caz_visualization_*.png` file shows:
- **Top panel**: Separation curve with CAZ boundaries (green shaded region)
- **Middle panel**: Coherence curve (should peak in CAZ)
- **Bottom panel**: Velocity (rate of change of separation)

**Key features to observe**:
1. **Pre-CAZ** (early layers): Low separation, high entanglement
2. **CAZ** (middle layers): Rapid increase in separation, concept crystallizing
3. **Peak**: Maximum geometric clarity
4. **Post-CAZ** (late layers): Often shows degradation as model shifts to logit projection

### Hypothesis Test Results

In `caz_ablation_comparison.json`, look for:

```json
{
  "hypothesis_test": {
    "caz_mid_reduction": 0.85,      // Should be high (>0.5)
    "post_caz_reduction": 0.82,     // May be similar or slightly lower
    "caz_mid_kl": 2.3,              // Should be LOWER than post-CAZ
    "post_caz_kl": 4.1,             // Higher = more collateral damage
    "kl_improvement": 1.8,          // Positive = CAZ-mid is better
    "hypothesis_supported": true    // Overall verdict
  }
}
```

**Hypothesis supported if**:
- CAZ-mid achieves ≥50% separation reduction (behavioral suppression)
- CAZ-mid has lower KL divergence than post-CAZ (capability preservation)

## Supported Models

- `gpt2` — GPT-2 (124M) - Fast, CPU-friendly
- `gpt2-medium` — GPT-2 Medium (355M)
- `gpt2-large` — GPT-2 Large (774M)
- `gpt2-xl` — GPT-2 XL (1.5B)
- `llama3` — Llama 3 8B (requires GPU)
- `mistral` — Mistral 7B (requires GPU)
- `qwen` — Qwen 2.5 7B (requires GPU)

## Dependencies

Install with:
```bash
pip install matplotlib>=3.7.0  # If not already installed
```

Or reinstall all requirements:
```bash
pip install -r requirements.txt
```

## Interpreting Results

### Strong CAZ Evidence

- **Bounded zone**: Clear separation spike across 3-7 contiguous layers
- **Velocity pattern**: Positive velocity entering CAZ, negative exiting
- **Hypothesis support**: CAZ-mid ablation has lower KL than post-CAZ
- **Coherence peak**: Aligned with separation peak (concept is geometrically clean)

### Weak CAZ Evidence

- **Monotonic increase**: Separation grows steadily across all layers (no zone)
- **Single-point peak**: Separation spikes at one layer only
- **No KL difference**: CAZ-mid and post-CAZ ablation produce similar KL
- **Coherence misalignment**: Peak coherence far from peak separation

## Connection to Rosetta_Manifold

The CAZ validation extends Phase 2 extraction methodology:

**Rosetta_Manifold Phase 2**:
- Finds "best layer" via layer sweep
- Extracts single DoM/LAT vector at peak
- Tests cross-architecture alignment (PRH)

**CAZ Validation**:
- Tracks metrics across ALL layers
- Identifies where/how concepts assemble
- Tests when to intervene for optimal ablation

The infrastructure is identical — CAZ just tracks the full trajectory instead of the peak snapshot.

## Citation

If you use this pipeline, cite both frameworks:

```
Henry, J. (2026). The Concept Assembly Zone: A Dynamical Systems Framework for
Cross-Layer Semantic Manifold Tracking in Transformers.

Henry, J. (2026). Rosetta Manifold: Universal Credibility Vectors Across
Language Model Architectures (Preliminary Draft).
```

## Related Work

- **Rosetta_Manifold**: Transferable semantic vector extraction (this repo)
- **Activation_Manifold_Cartography**: Unlabeled manifold detection (research framework)
- **Concept_Assembly_Zone**: Layer-wise concept formation theory (formal paper)

All three projects are interconnected and under active development.

---

**Questions?** Open an issue or reach out to james.henry@telus.com
