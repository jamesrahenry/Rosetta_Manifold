#!/bin/bash
# One-command Tiny PoC runner for laptop testing (4GB GPU)

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║           Rosetta Manifold - Tiny PoC (Laptop Mode)          ║"
echo "║                                                               ║"
echo "║  Hardware: 4GB GPU or CPU                                    ║"
echo "║  Models:   TinyLlama 1.1B, Qwen2 1.5B                        ║"
echo "║  Dataset:  20 pairs (5 per domain)                           ║"
echo "║  Time:     ~10 min (GPU) or ~30 min (CPU)                    ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Parse mode
MODE=${1:-"single"}
MODEL=${2:-"tinyllama"}
DEVICE=${3:-"auto"}

case $MODE in
    single)
        echo "=== Running Tiny PoC on single model: $MODEL ==="
        echo ""

        # Step 1: Generate tiny dataset
        echo "Step 1: Generating tiny dataset (20 pairs)..."
        if [ ! -f "data/credibility_pairs_tiny.jsonl" ]; then
            python src/generate_dataset_tiny.py
        else
            echo "  ✓ Tiny dataset already exists"
        fi
        echo ""

        # Step 2: Extract vectors
        echo "Step 2: Extracting credibility vectors from $MODEL..."
        python src/extract_vectors_tiny.py --model "$MODEL" --device "$DEVICE"
        echo ""

        # Step 3: Run ablation
        echo "Step 3: Running ablation validation..."
        python src/ablate_vectors_tiny.py \
            --model "$MODEL" \
            --vectors results/phase2_vectors_tiny.json \
            --device "$DEVICE"
        echo ""

        # Show results
        echo "=== Results ==="
        echo ""
        echo "Phase 2 (Extraction):"
        cat results/phase2_vectors_tiny.json | python -m json.tool | grep -A 5 "\"model_id\""
        echo ""
        echo "Phase 3 (Ablation):"
        cat results/phase3_ablation_tiny.json | python -m json.tool | grep -A 3 "separation_reduction"
        ;;

    all)
        echo "=== Running Tiny PoC on all tiny models ==="
        echo ""

        # Step 1: Generate dataset
        echo "Step 1: Generating tiny dataset (20 pairs)..."
        if [ ! -f "data/credibility_pairs_tiny.jsonl" ]; then
            python src/generate_dataset_tiny.py
        else
            echo "  ✓ Tiny dataset already exists"
        fi
        echo ""

        # Step 2: Extract from all models
        echo "Step 2: Extracting from all tiny models..."
        python src/extract_vectors_tiny.py --all-models --device "$DEVICE"
        echo ""

        # Step 3: Ablate each model
        for model in tinyllama qwen2-1.5b; do
            echo "Step 3: Ablating $model..."
            python src/ablate_vectors_tiny.py \
                --model "$model" \
                --vectors results/phase2_vectors_tiny.json \
                --device "$DEVICE" \
                --output "results/phase3_ablation_tiny_${model}.json"
            echo ""
        done

        echo "=== All models complete ==="
        ;;

    cpu)
        echo "=== Running Tiny PoC on CPU (slower) ==="
        echo "Model: $MODEL"
        echo ""

        if [ ! -f "data/credibility_pairs_tiny.jsonl" ]; then
            python src/generate_dataset_tiny.py
        fi

        python src/extract_vectors_tiny.py --model "$MODEL" --device cpu
        python src/ablate_vectors_tiny.py \
            --model "$MODEL" \
            --vectors results/phase2_vectors_tiny.json \
            --device cpu
        ;;

    clean)
        echo "Cleaning tiny PoC results..."
        rm -f data/credibility_pairs_tiny.jsonl
        rm -f results/phase2_vectors_tiny.json
        rm -f results/phase3_ablation_tiny*.json
        echo "✓ Cleaned"
        ;;

    *)
        echo "Usage: ./run_tiny_poc.sh [MODE] [MODEL] [DEVICE]"
        echo ""
        echo "Modes:"
        echo "  single [MODEL] [DEVICE]  - Run on single model (default: tinyllama, auto)"
        echo "  all [DEVICE]             - Run on all tiny models"
        echo "  cpu [MODEL]              - Force CPU mode (slower)"
        echo "  clean                    - Remove tiny PoC results"
        echo ""
        echo "Models:"
        echo "  tinyllama   - TinyLlama 1.1B (default, ~2GB VRAM)"
        echo "  qwen2-1.5b  - Qwen2 1.5B (~3GB VRAM)"
        echo "  phi2        - Phi-2 2.7B (~5GB VRAM, tight on 4GB)"
        echo ""
        echo "Devices:"
        echo "  auto  - Auto-detect GPU or CPU (default)"
        echo "  cuda  - Force GPU"
        echo "  cpu   - Force CPU"
        echo ""
        echo "Examples:"
        echo "  ./run_tiny_poc.sh                       # Single model, auto device"
        echo "  ./run_tiny_poc.sh single qwen2-1.5b     # Qwen2 on auto"
        echo "  ./run_tiny_poc.sh cpu tinyllama         # TinyLlama on CPU"
        echo "  ./run_tiny_poc.sh all                   # All models"
        exit 1
        ;;
esac

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    Tiny PoC Complete! ✓                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results:"
echo "  • Phase 2: results/phase2_vectors_tiny.json"
echo "  • Phase 3: results/phase3_ablation_tiny*.json"
echo ""
