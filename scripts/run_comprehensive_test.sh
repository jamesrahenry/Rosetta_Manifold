#!/bin/bash
# Comprehensive all-day testing of tiny models
# Tests multiple models and configurations while user is away

set -e

LOG_DIR="results/comprehensive_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║         Comprehensive Tiny Model Testing - All Day           ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Log directory: $LOG_DIR"
echo "Start time: $(date)"
echo ""

# Models to test (in order of size)
MODELS=(
    "gpt2:cpu"                    # 124M - baseline (already done, but rerun for logs)
    "qwen2-0.5b:auto"            # 500M - small but capable
    "qwen2-1.5b:auto"            # 1.5B - sweet spot for 4GB
    "phi2:cpu"                   # 2.7B - too big for 4GB GPU, use CPU
)

echo "Models to test:"
for model_spec in "${MODELS[@]}"; do
    model=$(echo $model_spec | cut -d: -f1)
    device=$(echo $model_spec | cut -d: -f2)
    echo "  - $model (device: $device)"
done
echo ""

# Test counter
total_tests=$((${#MODELS[@]} * 2))  # extraction + ablation
current_test=0

# Function to run one model
run_model_test() {
    local model=$1
    local device=$2
    local log_prefix="$LOG_DIR/${model}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $model (device: $device)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Phase 2: Extraction
    current_test=$((current_test + 1))
    echo "[$current_test/$total_tests] Phase 2: Vector Extraction..."
    echo "Start: $(date)" > "$log_prefix.extraction.log"

    if python src/extract_vectors_tiny.py \
        --model "$model" \
        --device "$device" \
        --output "$log_prefix.vectors.json" \
        >> "$log_prefix.extraction.log" 2>&1; then
        echo "  ✓ Extraction complete"
        extraction_time=$(tail -1 "$log_prefix.extraction.log" | grep -o "complete" || echo "done")
    else
        echo "  ✗ Extraction failed (see $log_prefix.extraction.log)"
        return 1
    fi

    # Phase 3: Ablation
    current_test=$((current_test + 1))
    echo "[$current_test/$total_tests] Phase 3: Ablation Validation..."
    echo "Start: $(date)" > "$log_prefix.ablation.log"

    if python src/ablate_vectors_tiny.py \
        --model "$model" \
        --vectors "$log_prefix.vectors.json" \
        --device "$device" \
        --output "$log_prefix.ablation.json" \
        >> "$log_prefix.ablation.log" 2>&1; then
        echo "  ✓ Ablation complete"
    else
        echo "  ✗ Ablation failed (see $log_prefix.ablation.log)"
        return 1
    fi

    # Extract key results
    echo ""
    echo "Results for $model:"
    echo "  Extraction:"
    cat "$log_prefix.vectors.json" | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    ext = data['extractions'][0]
    print(f\"    Layer: {ext['best_layer']}, Separation: {ext['separation']:.2f}, DoM-LAT: {ext['dom_lat_similarity']:.4f}\")
except:
    print('    (parsing error)')
" || echo "    (error reading results)"

    echo "  Ablation:"
    cat "$log_prefix.ablation.json" | python -c "
import json, sys
try:
    r = json.load(sys.stdin)
    print(f\"    Reduction: {r['separation_reduction']*100:.1f}%, KL: {r['kl_divergence']:.4f}, Success: {r['ablation_success']}\")
except:
    print('    (parsing error)')
" || echo "    (error reading results)"

    echo ""
    return 0
}

# Run all tests
echo "Starting comprehensive test suite..."
echo ""

for model_spec in "${MODELS[@]}"; do
    model=$(echo $model_spec | cut -d: -f1)
    device=$(echo $model_spec | cut -d: -f2)

    if run_model_test "$model" "$device"; then
        echo "✓ $model complete"
    else
        echo "✗ $model failed"
    fi

    echo ""
    echo "Progress: $current_test/$total_tests tests complete"
    echo ""
done

# Generate summary report
SUMMARY="$LOG_DIR/SUMMARY.md"
cat > "$SUMMARY" << 'SUMMARY_EOF'
# Comprehensive Tiny Model Testing Summary

## Test Configuration

SUMMARY_EOF

echo "**Date**: $(date)" >> "$SUMMARY"
echo "**Duration**: Started at $(head -1 $LOG_DIR/*.extraction.log | grep Start | head -1 | cut -d: -f2-)" >> "$SUMMARY"
echo "**Models Tested**: ${#MODELS[@]}" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "## Results" >> "$SUMMARY"
echo "" >> "$SUMMARY"

for model_spec in "${MODELS[@]}"; do
    model=$(echo $model_spec | cut -d: -f1)
    log_prefix="$LOG_DIR/${model}"

    echo "### $model" >> "$SUMMARY"
    echo "" >> "$SUMMARY"

    if [ -f "$log_prefix.vectors.json" ]; then
        echo "**Phase 2 (Extraction)**:" >> "$SUMMARY"
        cat "$log_prefix.vectors.json" | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    ext = data['extractions'][0]
    print(f\"- Best Layer: {ext['best_layer']}\")
    print(f\"- Separation: {ext['separation']:.2f}\")
    print(f\"- DoM-LAT Agreement: {ext['dom_lat_similarity']:.4f}\")
    print(f\"- Hidden Dim: {ext['hidden_dim']}\")
except:
    print('- Error parsing results')
" >> "$SUMMARY" || echo "- Error reading results" >> "$SUMMARY"
        echo "" >> "$SUMMARY"
    fi

    if [ -f "$log_prefix.ablation.json" ]; then
        echo "**Phase 3 (Ablation)**:" >> "$SUMMARY"
        cat "$log_prefix.ablation.json" | python -c "
import json, sys
try:
    r = json.load(sys.stdin)
    print(f\"- Baseline Separation: {r['baseline_separation']:.4f}\")
    print(f\"- Ablated Separation: {r['ablated_separation']:.4f}\")
    print(f\"- Reduction: {r['separation_reduction']*100:.1f}%\")
    print(f\"- KL Divergence: {r['kl_divergence']:.4f}\")
    print(f\"- Ablation Success: {r['ablation_success']}\")
    print(f\"- KL Pass: {r['kl_pass']}\")
except:
    print('- Error parsing results')
" >> "$SUMMARY" || echo "- Error reading results" >> "$SUMMARY"
        echo "" >> "$SUMMARY"
    fi
done

echo "" >> "$SUMMARY"
echo "## Logs" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "All logs saved in: \`$LOG_DIR/\`" >> "$SUMMARY"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║              Comprehensive Testing Complete!                  ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "End time: $(date)"
echo ""
echo "Summary report: $SUMMARY"
echo "All results: $LOG_DIR/"
echo ""
echo "View summary:"
echo "  cat $SUMMARY"
echo ""
