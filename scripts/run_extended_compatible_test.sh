#!/bin/bash
# Extended testing with TransformerLens-compatible models only
# Tests models from 124M to 1.5B parameters

set -e

LOG_DIR="results/extended_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║     Extended Testing - TransformerLens Compatible Models     ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Log directory: $LOG_DIR"
echo "Start time: $(date)"
echo ""

# Compatible models in size order
MODELS=(
    "gpt2:cpu:124M"                          # Baseline (validated)
    "gpt2-medium:cpu:355M"                   # 3x larger
    "gpt2-large:cpu:774M"                    # 6x larger
    "EleutherAI/gpt-neo-125M:cpu:125M"       # Alternative tiny
    "facebook/opt-125m:cpu:125M"             # Another tiny option
    "EleutherAI/gpt-neo-1.3B:cpu:1.3B"       # Large test
    "facebook/opt-1.3b:cpu:1.3B"             # Large test
    "gpt2-xl:cpu:1.5B"                       # Largest GPT-2
)

echo "Testing ${#MODELS[@]} compatible models:"
for model_spec in "${MODELS[@]}"; do
    model=$(echo $model_spec | cut -d: -f1)
    size=$(echo $model_spec | cut -d: -f3)
    echo "  - $model ($size)"
done
echo ""
echo "Estimated total time: 30-60 minutes on CPU"
echo ""

total_models=${#MODELS[@]}
completed=0

run_model() {
    local model=$1
    local device=$2
    local size=$3
    local model_key=$(echo "$model" | tr '/' '_')
    local log_prefix="$LOG_DIR/${model_key}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$((completed + 1))/$total_models] Testing: $model ($size)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Time: $(date)"
    echo ""

    # Phase 2
    echo "  Phase 2: Vector Extraction..."
    if timeout 3600 python src/extract_vectors_tiny.py \
        --model "$model" \
        --device "$device" \
        --output "$log_prefix.vectors.json" \
        > "$log_prefix.extraction.log" 2>&1; then
        echo "    ✓ Extraction complete"

        # Extract key metrics
        python -c "
import json
with open('$log_prefix.vectors.json') as f:
    data = json.load(f)
    ext = data['extractions'][0]
    print(f\"      Layer: {ext['best_layer']}, Sep: {ext['separation']:.2f}, DoM-LAT: {ext['dom_lat_similarity']:.4f}\")
" 2>/dev/null || echo "      (results saved)"

        # Phase 3
        echo "  Phase 3: Ablation Validation..."
        if timeout 3600 python src/ablate_vectors_tiny.py \
            --model "$model" \
            --vectors "$log_prefix.vectors.json" \
            --device "$device" \
            --output "$log_prefix.ablation.json" \
            > "$log_prefix.ablation.log" 2>&1; then
            echo "    ✓ Ablation complete"

            # Extract results
            python -c "
import json
with open('$log_prefix.ablation.json') as f:
    r = json.load(f)
    print(f\"      Reduction: {r['separation_reduction']*100:.1f}%, KL: {r['kl_divergence']:.2f}, Pass: {r['kl_pass']}\")
" 2>/dev/null || echo "      (results saved)"

            echo "    ✓ $model COMPLETE"
            completed=$((completed + 1))
            return 0
        else
            echo "    ✗ Ablation failed"
            tail -10 "$log_prefix.ablation.log"
            return 1
        fi
    else
        echo "    ✗ Extraction failed"
        tail -10 "$log_prefix.extraction.log"
        return 1
    fi
}

# Run all tests
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model device size <<< "$model_spec"

    if run_model "$model" "$device" "$size"; then
        echo "  ✓✓ $model SUCCESS"
    else
        echo "  ✗✗ $model FAILED"
    fi
done

# Generate comprehensive summary
cat > "$LOG_DIR/SUMMARY.md" << 'EOF'
# Extended Tiny Model Testing - Final Report

## Test Configuration

**Start Time**: $(date)
**Total Models**: ${#MODELS[@]}
**Successful**: $completed
**Failed**: $((${#MODELS[@]} - completed))

## Results by Model Size

EOF

# Add results for each model
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model device size <<< "$model_spec"
    model_key=$(echo "$model" | tr '/' '_')
    log_prefix="$LOG_DIR/${model_key}"

    echo "### $model ($size)" >> "$LOG_DIR/SUMMARY.md"
    echo "" >> "$LOG_DIR/SUMMARY.md"

    if [ -f "$log_prefix.ablation.json" ]; then
        echo "✅ **COMPLETE**" >> "$LOG_DIR/SUMMARY.md"
        echo "" >> "$LOG_DIR/SUMMARY.md"
        python -c "
import json
with open('$log_prefix.vectors.json') as f:
    ext = json.load(f)['extractions'][0]
with open('$log_prefix.ablation.json') as f:
    abl = json.load(f)
print(f\"**Extraction**: Layer {ext['best_layer']}, Separation {ext['separation']:.2f}, DoM-LAT {ext['dom_lat_similarity']:.4f}\")
print(f\"**Ablation**: {abl['separation_reduction']*100:.1f}% reduction, KL {abl['kl_divergence']:.4f}, Success: {abl['ablation_success']}, KL Pass: {abl['kl_pass']}\")
" >> "$LOG_DIR/SUMMARY.md" 2>/dev/null
        echo "" >> "$LOG_DIR/SUMMARY.md"
    else
        echo "❌ **FAILED** - See \`${model_key}.extraction.log\` for details" >> "$LOG_DIR/SUMMARY.md"
        echo "" >> "$LOG_DIR/SUMMARY.md"
    fi
done

echo "" >> "$LOG_DIR/SUMMARY.md"
echo "## Conclusion" >> "$LOG_DIR/SUMMARY.md"
echo "" >> "$LOG_DIR/SUMMARY.md"
echo "Successfully validated methodology on $completed model(s)." >> "$LOG_DIR/SUMMARY.md"
echo "" >> "$LOG_DIR/SUMMARY.md"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║            Extended Testing Complete!                         ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "End time: $(date)"
echo "Models tested: ${#MODELS[@]}"
echo "Successful: $completed"
echo ""
echo "Summary: $LOG_DIR/SUMMARY.md"
echo ""
