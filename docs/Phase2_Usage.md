# Phase 2 Usage Guide

## Overview

Phase 2 implements credibility vector extraction using TransformerLens. It supports two extraction methods (DoM and LAT) and tests the Platonic Representation Hypothesis by measuring cross-model alignment.

## Prerequisites

1. **Hardware Requirements:**
   - **GPU (Recommended):** 16GB+ VRAM for 8B models in fp16
   - **CPU (Fallback):** Will work but much slower (~10x)
   - **RAM:** 32GB+ recommended

2. **Software Dependencies:**
   ```bash
   pip install -r requirements.txt
   python src/verify_setup.py
   ```

3. **Dataset:**
   - Must have `data/credibility_pairs.jsonl` from Phase 1
   - Run `python src/generate_dataset.py` if missing

4. **HuggingFace Access:**
   - Some models (like Llama 3) require gated access
   - Login: `huggingface-cli login`
   - Request access: https://huggingface.co/meta-llama/Meta-Llama-3-8B

## Quick Start

### Single Model Extraction
```bash
# Extract from Llama 3 (default settings)
python src/extract_vectors.py --model llama3

# Extract from Mistral
python src/extract_vectors.py --model mistral

# Extract from Qwen
python src/extract_vectors.py --model qwen
```

### Multi-Model Extraction (PRH Test)
```bash
# Extract from all three models and compute alignment
python src/extract_vectors.py --all-models
```

### Using Helper Script
```bash
# Single model
./run_phase2.sh single llama3

# All models
./run_phase2.sh all

# Quick CPU test (limited layers)
./run_phase2.sh test
```

## Command-Line Options

```bash
python src/extract_vectors.py [OPTIONS]

Required (one of):
  --model MODEL         Model to extract from (llama3|mistral|qwen|HF-ID)
  --all-models          Extract from all three models

Optional:
  --dataset PATH        Path to dataset (default: data/credibility_pairs.jsonl)
  --layer-start N       Start layer for sweep (default: 14)
  --layer-end N         End layer for sweep, exclusive (default: 23)
  --token-pos N         Token position to extract (default: -1 for last)
  --output PATH         Output path (default: results/phase2_vectors.json)
  --device DEVICE       Device to use: cuda|cpu|auto (default: auto)
  --skip-opik           Skip Opik logging
```

## Output Format

Results are saved to `results/phase2_vectors.json`:

```json
{
  "extractions": [
    {
      "model_id": "meta-llama/Meta-Llama-3-8B",
      "best_layer": 18,
      "separation": 12.45,
      "dom_vector": [0.01, -0.02, ...],
      "lat_vector": [0.01, -0.02, ...],
      "dom_lat_similarity": 0.92,
      "hidden_dim": 4096,
      "n_layers": 32,
      "token_pos": -1,
      "layer_range": [14, 23]
    }
  ],
  "alignment": {
    "models": ["Meta-Llama-3-8B", "Mistral-7B-v0.1", "Qwen2.5-7B"],
    "dom_similarities": {
      "Meta-Llama-3-8B vs Mistral-7B-v0.1": 0.67,
      "Meta-Llama-3-8B vs Qwen2.5-7B": 0.58,
      "Mistral-7B-v0.1 vs Qwen2.5-7B": 0.61
    },
    "lat_similarities": {
      "Meta-Llama-3-8B vs Mistral-7B-v0.1": 0.72,
      "Meta-Llama-3-8B vs Qwen2.5-7B": 0.64,
      "Mistral-7B-v0.1 vs Qwen2.5-7B": 0.68
    },
    "avg_dom_similarity": 0.62,
    "avg_lat_similarity": 0.68,
    "prh_threshold": 0.5,
    "prh_pass": true
  }
}
```

## Understanding the Results

### Per-Model Extraction

- **best_layer**: Layer with maximum credible/non-credible separation
- **separation**: Magnitude of mean difference (higher = clearer signal)
- **dom_vector**: Normalized direction from Difference-of-Means
- **lat_vector**: Normalized direction from Linear Artificial Tomography
- **dom_lat_similarity**: Agreement between DoM and LAT (>0.8 is good)

### Cross-Model Alignment

- **dom_similarities**: Pairwise cosine similarities for DoM vectors
- **lat_similarities**: Pairwise cosine similarities for LAT vectors
- **avg_*_similarity**: Average across all model pairs
- **prh_pass**: True if average similarity > 0.5 (supports PRH)

### Interpreting PRH Results

| Avg Similarity | Interpretation |
|:--------------|:---------------|
| < 0.3 | No convergence - concept may be model-specific |
| 0.3 - 0.5 | Weak alignment - partial convergence |
| 0.5 - 0.7 | Moderate alignment - **PRH supported** |
| 0.7 - 0.9 | Strong alignment - high architectural convergence |
| > 0.9 | Very strong alignment - near-universal representation |

## Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce batch size or use CPU
python src/extract_vectors.py --model llama3 --device cpu
```

### Model Download Fails
```bash
# Check HuggingFace login
huggingface-cli whoami

# Re-login if needed
huggingface-cli login
```

### Low Separation Values
- Try different layer ranges: `--layer-start 10 --layer-end 25`
- Check dataset quality in `data/credibility_pairs.jsonl`
- Some models may have weaker credibility representations

### DoM-LAT Disagreement
- Similarity < 0.5 suggests concept is not linearly separable
- LAT is more robust in this case
- May indicate need for non-linear probing (future work)

## Performance Tips

1. **GPU Usage:**
   - Llama 3 8B: ~14GB VRAM (fp16)
   - Mistral 7B: ~13GB VRAM (fp16)
   - Qwen 2.5 7B: ~13GB VRAM (fp16)

2. **Speed Optimization:**
   - Run models sequentially if VRAM limited
   - Use `--layer-start 16 --layer-end 20` for faster sweeps
   - Skip Opik logging: `--skip-opik`

3. **Batch Processing:**
   - Default batch size: 8
   - Increase for GPUs with >24GB VRAM (edit `batch_size` in code)
   - Decrease if OOM: reduce to 4 or 2

## Next Steps

After successful Phase 2 extraction:

1. **Analyze Results:**
   - Check `results/phase2_vectors.json`
   - Review Opik dashboard: http://localhost:5173

2. **Phase 3 (Ablation):**
   - Use extracted vectors for directional ablation
   - Validate with KL divergence < 0.2
   - See: `docs/Spec 3 -- Heretic Optimization and Ablation.md`

3. **Publication:**
   - Document PRH test results
   - Prepare alignment visualizations
   - Compare with Huh et al. (2024) findings

## References

- **Spec 2**: Full technical specification
- **Arditi et al. (2024)**: DoM methodology - [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- **Zou et al. (2023)**: LAT/RepE methodology - [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
- **Huh et al. (2024)**: Platonic Representation Hypothesis - [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)
