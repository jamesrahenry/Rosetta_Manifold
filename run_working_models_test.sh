#!/bin/bash
# Test only models that are proven to work with TransformerLens
# Focus on cross-architecture validation with GPT-2, GPT-Neo, OPT

set -e

LOG_DIR="results/working_models_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   Working Models Test - Proven TransformerLens Compatible    ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Strategy: Only test models proven to work with TransformerLens"
echo "Hardware: CPU (stable, can run overnight if needed)"
echo "Log directory: $LOG_DIR"
echo "Start time: $(date)"
echo ""

# Only models that definitely work
MODELS=(
    # GPT-2 family (proven working)
    "gpt2:124M:GPT2"
    "gpt2-medium:355M:GPT2"
    "gpt2-large:774M:GPT2"
    "gpt2-xl:1.5B:GPT2"

    # GPT-Neo family (should work - different architecture)
    "EleutherAI/gpt-neo-125M:125M:GPTNeo"
    "EleutherAI/gpt-neo-1.3B:1.3B:GPTNeo"
    "EleutherAI/gpt-neo-2.7B:2.7B:GPTNeo"

    # OPT family (should work - Meta architecture)
    "facebook/opt-125m:125M:OPT"
    "facebook/opt-1.3b:1.3B:OPT"
    "facebook/opt-2.7b:2.7B:OPT"
)

echo "Testing ${#MODELS[@]} models across 3 architectures:"
echo ""
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model size arch <<< "$model_spec"
    printf "  %-6s: %-35s (%s)\n" "$arch" "$model" "$size"
done
echo ""
echo "Estimated time: 1-3 hours (depends on model compatibility)"
echo ""

completed=0
failed=0

run_model() {
    local model=$1
    local size=$2
    local arch=$3
    local model_key=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local log_prefix="$LOG_DIR/${model_key}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$((completed + failed + 1))/${#MODELS[@]}] $arch: $model ($size)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Time: $(date)"

    # Phase 2
    echo "  Phase 2: Extracting..."
    if timeout 7200 python src/extract_vectors_tiny.py \
        --model "$model" \
        --device cpu \
        --output "$log_prefix.vectors.json" \
        > "$log_prefix.extraction.log" 2>&1; then

        python -c "
import json
with open('$log_prefix.vectors.json') as f:
    ext = json.load(f)['extractions'][0]
    print(f\"    ✓ Layer {ext['best_layer']}, Sep {ext['separation']:.2f}, DoM-LAT {ext['dom_lat_similarity']:.4f}\")
" 2>/dev/null

        # Phase 3
        echo "  Phase 3: Ablating..."
        if timeout 3600 python src/ablate_vectors_tiny.py \
            --model "$model" \
            --vectors "$log_prefix.vectors.json" \
            --device cpu \
            --output "$log_prefix.ablation.json" \
            > "$log_prefix.ablation.log" 2>&1; then

            python -c "
import json
with open('$log_prefix.ablation.json') as f:
    r = json.load(f)
    print(f\"    ✓ Red {r['separation_reduction']*100:.0f}%, KL {r['kl_divergence']:.2f}, Pass {r['kl_pass']}\")
" 2>/dev/null

            completed=$((completed + 1))
            echo "  ✓✓ SUCCESS"
            return 0
        else
            echo "  ✗ Ablation failed"
            failed=$((failed + 1))
            return 1
        fi
    else
        echo "  ✗ Extraction failed"
        tail -3 "$log_prefix.extraction.log" | grep -i error || echo "  (see log for details)"
        failed=$((failed + 1))
        return 1
    fi
}

# Run all
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model size arch <<< "$model_spec"
    run_model "$model" "$size" "$arch"
    echo "  Progress: $completed OK, $failed failed"
done

# Summary
cat > "$LOG_DIR/SUMMARY.md" << SUMEOF
# Multi-Architecture Testing Results

**Date**: $(date)
**Models Tested**: ${#MODELS[@]}
**Successful**: $completed
**Failed**: $failed

## By Architecture

SUMEOF

for target_arch in "GPT2" "GPTNeo" "OPT"; do
    echo "### $target_arch Architecture" >> "$LOG_DIR/SUMMARY.md"
    echo "" >> "$LOG_DIR/SUMMARY.md"

    for model_spec in "${MODELS[@]}"; do
        IFS=':' read -r model size arch <<< "$model_spec"

        if [ "$arch" = "$target_arch" ]; then
            model_key=$(echo "$model" | tr '/' '_' | tr '-' '_')
            if [ -f "$LOG_DIR/${model_key}.ablation.json" ]; then
                echo "✅ **$model** ($size)" >> "$LOG_DIR/SUMMARY.md"
                python -c "
import json
with open('$LOG_DIR/${model_key}.vectors.json') as f:
    ext = json.load(f)['extractions'][0]
with open('$LOG_DIR/${model_key}.ablation.json') as f:
    abl = json.load(f)
print(f\"- Separation: {ext['separation']:.2f}, Layer: {ext['best_layer']}\")
print(f\"- Ablation: {abl['separation_reduction']*100:.0f}% reduction, KL: {abl['kl_divergence']:.2f}, Pass: {abl['kl_pass']}\")
" >> "$LOG_DIR/SUMMARY.md" 2>/dev/null
            else
                echo "❌ **$model** ($size) - Failed" >> "$LOG_DIR/SUMMARY.md"
            fi
            echo "" >> "$LOG_DIR/SUMMARY.md"
        fi
    done
done

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          Multi-Architecture Testing Complete!                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "End: $(date)"
echo "Results: $completed/$${#MODELS[@]} successful"
echo "Summary: $LOG_DIR/SUMMARY.md"
echo ""
