# Multi-Architecture Testing Results

**Date**: Sun Mar  1 02:24:03 AM EST 2026
**Models Tested**: 10
**Successful**: 10
**Failed**: 0

## By Architecture

### GPT2 Architecture

✅ **gpt2** (124M)
- Separation: 28.36, Layer: 7
- Ablation: 100% reduction, KL: 4.79, Pass: False

✅ **gpt2-medium** (355M)
- Separation: 42.44, Layer: 12
- Ablation: 100% reduction, KL: 3.46, Pass: False

✅ **gpt2-large** (774M)
- Separation: 22.65, Layer: 12
- Ablation: 100% reduction, KL: 3.95, Pass: False

✅ **gpt2-xl** (1.5B)
- Separation: 22.70, Layer: 12
- Ablation: 100% reduction, KL: 4.33, Pass: False

### GPTNeo Architecture

✅ **EleutherAI/gpt-neo-125M** (125M)
- Separation: 54.40, Layer: 7
- Ablation: 100% reduction, KL: 5.71, Pass: False

✅ **EleutherAI/gpt-neo-1.3B** (1.3B)
- Separation: 52.07, Layer: 12
- Ablation: 100% reduction, KL: 4.34, Pass: False

✅ **EleutherAI/gpt-neo-2.7B** (2.7B)
- Separation: 44.01, Layer: 12
- Ablation: 100% reduction, KL: 3.39, Pass: False

### OPT Architecture

✅ **facebook/opt-125m** (125M)
- Separation: 1.72, Layer: 7
- Ablation: 100% reduction, KL: 4.87, Pass: False

✅ **facebook/opt-1.3b** (1.3B)
- Separation: 6.12, Layer: 12
- Ablation: 100% reduction, KL: 3.31, Pass: False

✅ **facebook/opt-2.7b** (2.7B)
- Separation: 2.38, Layer: 12
- Ablation: 100% reduction, KL: 3.16, Pass: False

