#!/bin/bash
# Multi-Architecture Testing - Different model families on CPU
# Tests whether credibility generalizes across architectures

set -e

LOG_DIR="results/multi_arch_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   Multi-Architecture Testing - Different Model Families      ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Strategy: Test DIFFERENT architectures, not just sizes"
echo "Hardware: CPU (can handle larger models, just slower)"
echo "Log directory: $LOG_DIR"
echo "Start time: $(date)"
echo ""

# Different model architectures (all CPU, take time for larger models)
MODELS=(
    # GPT-2 family (baseline - already validated)
    "gpt2:124M:GPT2-Arch"

    # Pythia family (EleutherAI, different architecture)
    "EleutherAI/pythia-160m:160M:Pythia-Arch"
    "EleutherAI/pythia-410m:410M:Pythia-Arch"
    "EleutherAI/pythia-1b:1B:Pythia-Arch"
    "EleutherAI/pythia-1.4b:1.4B:Pythia-Arch"

    # OPT family (Facebook, different architecture)
    "facebook/opt-125m:125M:OPT-Arch"
    "facebook/opt-1.3b:1.3B:OPT-Arch"

    # GPT-Neo family (EleutherAI, different from GPT-2 and Pythia)
    "EleutherAI/gpt-neo-125M:125M:GPTNeo-Arch"
    "EleutherAI/gpt-neo-1.3B:1.3B:GPTNeo-Arch"

    # Llama 2 (if available, different architecture)
    "meta-llama/Llama-2-7b-hf:7B:Llama2-Arch"
)

echo "Testing ${#MODELS[@]} models across multiple architectures:"
echo ""
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model size arch <<< "$model_spec"
    echo "  - $arch: $model ($size)"
done
echo ""
echo "Note: Larger models (1B+) will take 20-40 min each on CPU"
echo "Total estimated time: 2-4 hours for all models"
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
    echo "Testing: $arch - $model ($size)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Time: $(date)"
    echo ""

    # Phase 2
    echo "  Phase 2: Vector Extraction..."
    if timeout 7200 python src/extract_vectors_tiny.py \
        --model "$model" \
        --device cpu \
        --output "$log_prefix.vectors.json" \
        > "$log_prefix.extraction.log" 2>&1; then

        echo "    ✓ Extraction complete"

        # Show results
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
            --device cpu \
            --output "$log_prefix.ablation.json" \
            > "$log_prefix.ablation.log" 2>&1; then

            echo "    ✓ Ablation complete"

            # Show results
            python -c "
import json
with open('$log_prefix.ablation.json') as f:
    r = json.load(f)
    print(f\"      Reduction: {r['separation_reduction']*100:.1f}%, KL: {r['kl_divergence']:.2f}, Success: {r['ablation_success']}, KL Pass: {r['kl_pass']}\")
" 2>/dev/null || echo "      (results saved)"

            completed=$((completed + 1))
            echo "    ✓✓ $arch SUCCESS"
            return 0
        else
            echo "    ✗ Ablation failed"
            tail -5 "$log_prefix.ablation.log"
            failed=$((failed + 1))
            return 1
        fi
    else
        echo "    ✗ Extraction failed"
        tail -5 "$log_prefix.extraction.log"
        failed=$((failed + 1))
        return 1
    fi
}

# Run all tests
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model size arch <<< "$model_spec"

    run_model "$model" "$size" "$arch"

    echo ""
    echo "Progress: $completed successful, $failed failed"
    echo ""
done

# Generate summary
cat > "$LOG_DIR/MULTI_ARCH_SUMMARY.md" << 'EOF'
# Multi-Architecture Testing - Final Report

## Objective
Test whether credibility extraction generalizes across different model architectures.

## Models Tested by Architecture
EOF

echo "" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "**Total Models**: ${#MODELS[@]}" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "**Successful**: $completed" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "**Failed**: $failed" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "## Results" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"

# Group by architecture
for arch in "GPT2-Arch" "Pythia-Arch" "OPT-Arch" "GPTNeo-Arch" "Llama2-Arch"; do
    echo "### $arch" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
    echo "" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"

    found=false
    for model_spec in "${MODELS[@]}"; do
        IFS=':' read -r model size model_arch <<< "$model_spec"

        if [ "$model_arch" = "$arch" ]; then
            found=true
            model_key=$(echo "$model" | tr '/' '_' | tr '-' '_')
            log_prefix="$LOG_DIR/${model_key}"

            if [ -f "$log_prefix.ablation.json" ]; then
                echo "✅ **$model ($size)** - COMPLETE" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
                python -c "
import json
with open('$log_prefix.vectors.json') as f:
    ext = json.load(f)['extractions'][0]
with open('$log_prefix.ablation.json') as f:
    abl = json.load(f)
print(f\"  - Separation: {ext['separation']:.2f}, Layer: {ext['best_layer']}\")
print(f\"  - Ablation: {abl['separation_reduction']*100:.1f}% reduction, KL: {abl['kl_divergence']:.2f}\")
print(f\"  - Success: Ablation={abl['ablation_success']}, KL Pass={abl['kl_pass']}\")
" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md" 2>/dev/null
            else
                echo "❌ **$model ($size)** - FAILED" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
            fi
            echo "" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
        fi
    done

    if [ "$found" = false ]; then
        echo "(No models tested)" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
        echo "" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
    fi
done

echo "## Cross-Architecture Findings" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "If credibility generalizes, we should see:" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "- Similar separation values across architectures of same size" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "- Consistent ablation success (>30% reduction)" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "- Comparable KL divergence values" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo "" >> "$LOG_DIR/MULTI_ARCH_SUMMARY.md"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║        Multi-Architecture Testing Complete!                   ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "End time: $(date)"
echo "Successful: $completed / ${#MODELS[@]}"
echo ""
echo "Summary: $LOG_DIR/MULTI_ARCH_SUMMARY.md"
echo ""
