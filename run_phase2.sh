#!/bin/bash
# Convenience script for running Phase 2 extraction

set -e

echo "=== Rosetta Manifold - Phase 2 Extraction ==="
echo ""

# Check if dataset exists
if [ ! -f "data/credibility_pairs.jsonl" ]; then
    echo "Error: Dataset not found at data/credibility_pairs.jsonl"
    echo "Run Phase 1 first: python src/generate_dataset.py"
    exit 1
fi

# Parse arguments
MODE=${1:-"single"}
MODEL=${2:-"llama3"}

case $MODE in
    single)
        echo "Extracting vectors from single model: $MODEL"
        python src/extract_vectors.py --model "$MODEL"
        ;;
    all)
        echo "Extracting vectors from all three models (Llama 3, Mistral, Qwen)"
        echo "This will test the Platonic Representation Hypothesis"
        echo ""
        python src/extract_vectors.py --all-models
        ;;
    test)
        echo "Running quick test on CPU (first 10 samples)"
        python src/extract_vectors.py --model llama3 \
            --device cpu \
            --layer-start 16 \
            --layer-end 18
        ;;
    *)
        echo "Usage: ./run_phase2.sh [MODE] [MODEL]"
        echo ""
        echo "Modes:"
        echo "  single [MODEL]  - Extract from one model (default: llama3)"
        echo "  all            - Extract from all three models"
        echo "  test           - Quick CPU test"
        echo ""
        echo "Models:"
        echo "  llama3  - meta-llama/Meta-Llama-3-8B"
        echo "  mistral - mistralai/Mistral-7B-v0.1"
        echo "  qwen    - Qwen/Qwen2.5-7B"
        echo ""
        echo "Examples:"
        echo "  ./run_phase2.sh single llama3"
        echo "  ./run_phase2.sh all"
        echo "  ./run_phase2.sh test"
        exit 1
        ;;
esac

echo ""
echo "=== Phase 2 Complete ==="
echo "Results saved to: results/phase2_vectors.json"
