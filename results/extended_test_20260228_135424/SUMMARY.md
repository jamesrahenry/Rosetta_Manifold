# Extended Tiny Model Testing - Final Report

## Test Configuration

**Start Time**: $(date)
**Total Models**: ${#MODELS[@]}
**Successful**: $completed
**Failed**: $((${#MODELS[@]} - completed))

## Results by Model Size

### gpt2 (124M)

✅ **COMPLETE**

**Extraction**: Layer 7, Separation 28.36, DoM-LAT 0.1848
**Ablation**: 100.0% reduction, KL 4.7907, Success: True, KL Pass: False

### gpt2-medium (355M)

✅ **COMPLETE**

**Extraction**: Layer 12, Separation 42.44, DoM-LAT 0.1486
**Ablation**: 100.0% reduction, KL 3.4635, Success: True, KL Pass: False

### gpt2-large (774M)

❌ **FAILED** - See `gpt2-large.extraction.log` for details

### EleutherAI/gpt-neo-125M (125M)

❌ **FAILED** - See `EleutherAI_gpt-neo-125M.extraction.log` for details

### facebook/opt-125m (125M)

❌ **FAILED** - See `facebook_opt-125m.extraction.log` for details

### EleutherAI/gpt-neo-1.3B (1.3B)

❌ **FAILED** - See `EleutherAI_gpt-neo-1.3B.extraction.log` for details

### facebook/opt-1.3b (1.3B)

❌ **FAILED** - See `facebook_opt-1.3b.extraction.log` for details

### gpt2-xl (1.5B)

❌ **FAILED** - See `gpt2-xl.extraction.log` for details


## Conclusion

Successfully validated methodology on 2 model(s).

