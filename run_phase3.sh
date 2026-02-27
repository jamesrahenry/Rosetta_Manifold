#!/bin/bash
# Convenience script for running Phase 3 ablation validation

set -e

echo "=== Rosetta Manifold - Phase 3 Ablation Validation ==="
echo ""

# Check if Phase 2 results exist
if [ ! -f "results/phase2_vectors.json" ]; then
    echo "Error: Phase 2 results not found at results/phase2_vectors.json"
    echo "Run Phase 2 first: ./run_phase2.sh all"
    exit 1
fi

# Parse arguments
MODE=${1:-"single"}
MODEL=${2:-"llama3"}

case $MODE in
    single)
        echo "Running ablation on single model: $MODEL"
        python src/ablate_vectors.py \
            --model "$MODEL" \
            --vectors results/phase2_vectors.json \
            --method dom
        ;;
    sweep)
        echo "Sweeping layers and components for model: $MODEL"
        python src/ablate_vectors.py \
            --model "$MODEL" \
            --vectors results/phase2_vectors.json \
            --method dom \
            --sweep-layers
        ;;
    transfer)
        SOURCE=${MODEL}
        TARGET=${3:-"mistral"}
        echo "Testing cross-architecture transfer"
        echo "  Source: $SOURCE"
        echo "  Target: $TARGET"
        python src/ablate_vectors.py \
            --model "$TARGET" \
            --vectors results/phase2_vectors.json \
            --method dom \
            --transfer-from "$SOURCE"
        ;;
    all)
        echo "Running full Phase 3 validation (all models)"
        echo ""

        # Test each model with its own vector
        for model in llama3 mistral qwen; do
            echo "=== Testing $model with own vector ==="
            python src/ablate_vectors.py \
                --model "$model" \
                --vectors results/phase2_vectors.json \
                --method dom \
                --output "results/phase3_ablation_${model}_own.json"
            echo ""
        done

        # Test cross-architecture transfer
        echo "=== Testing cross-architecture transfer ==="
        echo "Transfer Llama3 -> Mistral"
        python src/ablate_vectors.py \
            --model mistral \
            --vectors results/phase2_vectors.json \
            --method dom \
            --transfer-from llama3 \
            --output results/phase3_ablation_transfer_llama3_to_mistral.json
        echo ""

        echo "Transfer Llama3 -> Qwen"
        python src/ablate_vectors.py \
            --model qwen \
            --vectors results/phase2_vectors.json \
            --method dom \
            --transfer-from llama3 \
            --output results/phase3_ablation_transfer_llama3_to_qwen.json
        echo ""
        ;;
    *)
        echo "Usage: ./run_phase3.sh [MODE] [MODEL] [TARGET]"
        echo ""
        echo "Modes:"
        echo "  single [MODEL]        - Ablate single model with its own vector"
        echo "  sweep [MODEL]         - Sweep layers/components for best config"
        echo "  transfer [SRC] [TGT]  - Test cross-architecture transfer"
        echo "  all                   - Full Phase 3 validation suite"
        echo ""
        echo "Models:"
        echo "  llama3  - meta-llama/Meta-Llama-3-8B"
        echo "  mistral - mistralai/Mistral-7B-v0.1"
        echo "  qwen    - Qwen/Qwen2.5-7B"
        echo ""
        echo "Examples:"
        echo "  ./run_phase3.sh single llama3"
        echo "  ./run_phase3.sh sweep mistral"
        echo "  ./run_phase3.sh transfer llama3 mistral"
        echo "  ./run_phase3.sh all"
        exit 1
        ;;
esac

echo ""
echo "=== Phase 3 Complete ==="
echo "Results saved to: results/phase3_ablation*.json"
