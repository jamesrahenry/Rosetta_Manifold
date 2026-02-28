# Tiny PoC - Laptop-Friendly Version

## Hardware Requirements

**What you have**:
- 4GB VRAM GPU
- CPU fallback available
- Limited to smaller models

**What we'll use**:
- TinyLlama 1.1B (~2GB VRAM in fp16)
- Phi-2 2.7B (~5GB VRAM in fp16, will use int8)
- Qwen2 1.5B (~3GB VRAM in fp16)

## Scaled-Down Approach

### Dataset: 20 pairs instead of 100
- 5 pairs per domain (technical, financial, crisis, historical)
- 40 total records (20 credible + 20 non-credible)
- Generated in ~2 minutes

### Models: 1-3B parameter range
| Model | Size | VRAM (fp16) | VRAM (int8) |
|:------|:-----|:------------|:------------|
| TinyLlama 1.1B | 1.1B | ~2GB | ~1GB |
| Qwen2 1.5B | 1.5B | ~3GB | ~1.5GB |
| Phi-2 2.7B | 2.7B | ~5GB | ~2.5GB |

### Layer Range: 6-12 instead of 14-22
- Fewer layers to sweep
- Faster extraction
- Still captures middle-layer representations

### Execution Time
- Dataset generation: ~2 min
- Single model extraction: ~5 min (CPU: ~20 min)
- Ablation: ~3 min (CPU: ~10 min)
- **Total PoC**: ~10 min on GPU, ~30 min on CPU

## Quick Start

```bash
# Generate tiny dataset
python src/generate_dataset_tiny.py

# Extract vectors (GPU)
python src/extract_vectors_tiny.py --model tinyllama

# Extract vectors (CPU fallback)
python src/extract_vectors_tiny.py --model tinyllama --device cpu

# Run ablation
python src/ablate_vectors_tiny.py --model tinyllama \
    --vectors results/phase2_vectors_tiny.json
```

## One-Command Demo

```bash
./run_tiny_poc.sh
```

This will:
1. Generate 20-pair dataset (~2 min)
2. Extract from TinyLlama (~5 min)
3. Run ablation validation (~3 min)
4. Show results

**Total time**: ~10 minutes on GPU, ~30 minutes on CPU

## Expected Results

Even with tiny models, you should see:
- **Separation**: DoM finds credibility direction
- **Agreement**: DoM-LAT similarity ~0.3-0.8
- **Ablation**: Separation reduction >30%
- **KL divergence**: <0.3 (looser threshold for tiny models)

**Note**: Tiny models won't match 7B performance, but will demonstrate the methodology works.

## Models We'll Use

### TinyLlama 1.1B
- HF: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Layers: 22
- Hidden: 2048
- VRAM: ~2GB (fp16)

### Qwen2 1.5B
- HF: `Qwen/Qwen2-1.5B-Instruct`
- Layers: 28
- Hidden: 1536
- VRAM: ~3GB (fp16)

### Phi-2 2.7B (optional, tight on 4GB)
- HF: `microsoft/phi-2`
- Layers: 32
- Hidden: 2560
- VRAM: ~5GB (fp16), ~2.5GB (int8)

## Differences from Full Pipeline

| Aspect | Full Pipeline | Tiny PoC |
|:-------|:--------------|:---------|
| Dataset | 100 pairs | 20 pairs |
| Models | 7-8B | 1-3B |
| VRAM | 14-16GB | 2-5GB |
| Layers | 14-22 | 6-12 |
| Time | ~45 min | ~10 min |
| PRH threshold | 0.5 | 0.3 (looser) |

## Next Steps

If tiny PoC works:
1. Validate methodology on small scale
2. Document behavior patterns
3. Scale up when GPU access available
4. Use same code structure, just larger models

If you want to run on CPU only:
- Add `--device cpu` to all commands
- Expect 2-3x slower execution
- Still completes in <1 hour total
