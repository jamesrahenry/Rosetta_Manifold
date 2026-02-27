# Phase 3 Usage Guide

## Overview

Phase 3 implements directional ablation to validate that extracted credibility vectors are functionally meaningful. It applies orthogonal projection to remove credibility directions and measures:

1. **Ablation success** - Credibility signal is removed
2. **Intelligence retention** - KL divergence < 0.2 on general tasks
3. **Cross-architecture transfer** - Vectors transfer between models

## Prerequisites

1. **Phase 2 Complete**: Must have `results/phase2_vectors.json`
   ```bash
   ./run_phase2.sh all
   ```

2. **Dependencies**: Same as Phase 2 (PyTorch, TransformerLens)
   ```bash
   pip install -r requirements.txt
   ```

3. **GPU**: Same as Phase 2 (16GB+ VRAM recommended)

## Quick Start

### Single Model Ablation
```bash
# Ablate Llama 3 with its own credibility vector
python src/ablate_vectors.py \
    --model llama3 \
    --vectors results/phase2_vectors.json
```

### Layer/Component Sweep
```bash
# Find optimal ablation configuration
python src/ablate_vectors.py \
    --model llama3 \
    --vectors results/phase2_vectors.json \
    --sweep-layers
```

### Cross-Architecture Transfer
```bash
# Test if Llama 3's vector works on Mistral
python src/ablate_vectors.py \
    --model mistral \
    --vectors results/phase2_vectors.json \
    --transfer-from llama3
```

### Using Helper Script
```bash
# Single model
./run_phase3.sh single llama3

# Sweep layers
./run_phase3.sh sweep mistral

# Transfer test
./run_phase3.sh transfer llama3 mistral

# Full validation (all models + transfers)
./run_phase3.sh all
```

## Command-Line Options

```bash
python src/ablate_vectors.py [OPTIONS]

Required:
  --model MODEL         Model to ablate (llama3|mistral|qwen)
  --vectors PATH        Path to Phase 2 results JSON

Optional:
  --method METHOD       Extraction method: dom|lat (default: dom)
  --layer N             Specific layer to ablate (default: best from Phase 2)
  --component COMP      Component: resid_pre|resid_mid|resid_post (default: resid_post)
  --sweep-layers        Sweep layers to find best ablation point
  --layer-start N       Start layer for sweep (default: 14)
  --layer-end N         End layer for sweep (default: 23)
  --transfer-from MODEL Test transfer from this model
  --output PATH         Output path (default: results/phase3_ablation.json)
  --device DEVICE       cuda|cpu|auto (default: auto)
  --skip-opik           Skip Opik logging
```

## Output Format

### Single Ablation Result
```json
{
  "model_id": "meta-llama/Meta-Llama-3-8B",
  "layer": 18,
  "component": "resid_post",
  "baseline_separation": 12.45,
  "ablated_separation": 1.23,
  "separation_reduction": 0.90,
  "kl_divergence": 0.08,
  "kl_threshold": 0.2,
  "kl_pass": true,
  "ablation_success": true
}
```

### Layer Sweep Results
```json
[
  {
    "layer": 14,
    "component": "resid_post",
    "separation_reduction": 0.75,
    "kl_divergence": 0.12,
    ...
  },
  {
    "layer": 15,
    "component": "resid_post",
    "separation_reduction": 0.85,
    "kl_divergence": 0.09,
    ...
  },
  ...
]
```

## Understanding Results

### Key Metrics

| Metric | Description | Success Threshold |
|:-------|:------------|:------------------|
| **baseline_separation** | Credible vs non-credible separation before ablation | Higher is better |
| **ablated_separation** | Separation after ablation | Should be near 0 |
| **separation_reduction** | Fraction of signal removed | > 0.5 (50%) |
| **kl_divergence** | Impact on general capabilities | < 0.2 |
| **ablation_success** | Signal removed successfully | separation_reduction > 0.5 |
| **kl_pass** | Intelligence retained | kl_divergence < 0.2 |

### Success Criteria

Both conditions must be met:
- ✅ **ablation_success = true**: Credibility signal removed (>50% reduction)
- ✅ **kl_pass = true**: General intelligence retained (KL < 0.2)

### Example Interpretation

```json
{
  "baseline_separation": 10.5,
  "ablated_separation": 1.2,
  "separation_reduction": 0.89,  // 89% reduction ✓
  "kl_divergence": 0.08,         // < 0.2 ✓
  "ablation_success": true,
  "kl_pass": true
}
```

**Interpretation**: Ablation successfully removed 89% of credibility signal while maintaining general capabilities (KL divergence only 0.08). The model can no longer reliably distinguish credible from non-credible text in the credibility direction.

## Methodology

### Orthogonal Projection

For each activation **x** at the target layer:
```
x' = x - (x · v)v
```
where **v** is the normalized credibility vector.

This removes the component of **x** along **v** while preserving all orthogonal components.

### Test Prompts

**General prompts** (for KL divergence):
- "The capital of France is"
- "To bake a cake, you need"
- "In Python, a list is"
- ... (general knowledge/reasoning)

**Credibility test prompts**:
- "A study published in Nature found that"
- "According to the CDC, vaccination rates"
- "Research from MIT demonstrates that"
- ... (should lose credibility signal)

**Non-credibility test prompts**:
- "Everyone knows that vaccines are"
- "My friend told me that climate change"
- "It's obvious that the government is"
- ... (baseline, should remain low)

## Layer/Component Selection

### Default Strategy
Use the best layer from Phase 2 extraction with `resid_post` component.

### Sweep Strategy
Test all combinations:
- **Layers**: 14-22 (middle-to-late layers)
- **Components**: resid_pre, resid_mid, resid_post
- **Selection**: Maximize separation_reduction while keeping KL < 0.2

### Component Types

| Component | Description | When to Use |
|:----------|:------------|:------------|
| **resid_pre** | Before attention + MLP | Early intervention |
| **resid_mid** | After attention, before MLP | Attention-specific |
| **resid_post** | After attention + MLP | Full layer intervention (default) |

## Cross-Architecture Transfer

Tests if credibility vectors are universal across architectures (PRH validation).

### Transfer Test
```bash
# Extract vector from Llama 3, apply to Mistral
python src/ablate_vectors.py \
    --model mistral \
    --vectors results/phase2_vectors.json \
    --transfer-from llama3
```

### Success Criteria for Transfer
- Same as regular ablation (separation reduction > 0.5, KL < 0.2)
- If successful: Evidence for PRH (architectures share representation)
- If unsuccessful: Credibility may be model-specific

### Expected Results

| Scenario | Separation Reduction | Interpretation |
|:---------|:---------------------|:---------------|
| > 0.7 | Strong | High transferability, strong PRH evidence |
| 0.5 - 0.7 | Moderate | Partial transfer, moderate PRH evidence |
| 0.3 - 0.5 | Weak | Limited transfer, weak PRH evidence |
| < 0.3 | None | No transfer, PRH not supported for credibility |

## Troubleshooting

### Low Separation Reduction
**Problem**: `separation_reduction < 0.5`

**Solutions**:
- Try different layer: `--layer N`
- Try different component: `--component resid_mid`
- Run sweep: `--sweep-layers`
- Check Phase 2 vectors were extracted correctly

### High KL Divergence
**Problem**: `kl_divergence > 0.2`

**Solutions**:
- Try earlier layers (14-16)
- Use `resid_pre` instead of `resid_post`
- Check if ablation is too aggressive
- May indicate credibility is entangled with core capabilities

### Transfer Failure
**Problem**: Transfer doesn't work across models

**Solutions**:
- Check models have same hidden_dim (4096 for all 7B/8B models)
- Try using LAT vectors: `--method lat`
- Adjust layer mapping (source layer may not match target)
- This may indicate architecture-specific representations

## Performance

### Single Ablation
- Time: ~1-2 minutes per model
- VRAM: ~14-16GB (same as Phase 2)

### Layer Sweep
- Time: ~10-15 minutes per model
- Configurations tested: 9 layers × 3 components = 27 configs

### Full Suite
- Time: ~30-45 minutes
- Tests: 3 models + 2 transfers = 5 runs

## Next Steps

After successful Phase 3:

1. **Analyze Results**
   - Check `results/phase3_ablation*.json`
   - Review Opik dashboard: http://localhost:5173
   - Identify best configurations

2. **Publication**
   - Document ablation success rates
   - Report cross-architecture transfer results
   - Compare with Arditi et al. (2024) baseline

3. **Applications**
   - Use ablated models for credibility-neutral applications
   - Deploy credibility detection via vector projection
   - Scale to other semantic concepts (honesty, bias, etc.)

## References

- **Spec 3**: Full technical specification
- **Arditi et al. (2024)**: Orthogonal projection methodology - [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- **FailSpy/abliterator**: Reference implementation - [github.com/FailSpy/abliterator](https://github.com/FailSpy/abliterator)

## Validation Checklist

Before declaring Phase 3 complete:

- [ ] Single model ablation successful (separation reduction > 0.5, KL < 0.2)
- [ ] At least one model passes both criteria
- [ ] Cross-architecture transfer tested
- [ ] Results logged to Opik (if configured)
- [ ] Output files saved to `results/`
- [ ] Ablation mechanism validated (orthogonal projection working)

## Example Session

```bash
# 1. Verify Phase 2 complete
ls results/phase2_vectors.json

# 2. Run single ablation
./run_phase3.sh single llama3

# 3. Check results
cat results/phase3_ablation.json

# 4. If successful, test transfer
./run_phase3.sh transfer llama3 mistral

# 5. Full validation
./run_phase3.sh all

# 6. Review in Opik
open http://localhost:5173
```

## Success Example

```
=== Ablating meta-llama/Meta-Llama-3-8B at layer 18 (resid_post) ===
Loading model...
Model loaded: 32 layers, hidden_dim=4096

Measuring baseline activations...
  Baseline separation: 10.45

Applying ablation...
  Ablated separation: 0.98

Computing KL divergence on general prompts...
  KL divergence: 0.12

=== Results ===
  Separation reduction: 90.6%
  KL divergence: 0.12 (threshold: 0.2)
  Ablation success: ✓
  KL threshold met: ✓
```

**✅ Phase 3 Validation Successful!**
