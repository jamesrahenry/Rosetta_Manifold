#!/bin/bash
#
# run_caz_validation.sh
#
# Complete CAZ (Concept Assembly Zone) validation pipeline.
# Runs all three phases: extraction, analysis, and ablation comparison.
#
# Usage:
#   ./run_caz_validation.sh gpt2                    # Quick test on GPT-2
#   ./run_caz_validation.sh gpt2 --full-dataset     # Use full credibility dataset
#   ./run_caz_validation.sh gpt2-xl                 # Test on larger model

set -e  # Exit on error

# Parse arguments
MODEL=${1:-gpt2}
CONCEPT=${2:-credibility}
DATASET="data/credibility_pairs_tiny.jsonl"  # Default: tiny dataset

if [[ "$3" == "--full-dataset" ]]; then
    DATASET="data/credibility_pairs.jsonl"
fi

echo "========================================"
echo "CAZ Validation Pipeline"
echo "========================================"
echo "Model:   $MODEL"
echo "Concept: $CONCEPT"
echo "Dataset: $DATASET"
echo ""

# Check dataset exists
if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: Dataset not found: $DATASET"
    echo ""
    echo "Available datasets:"
    ls -lh data/*.jsonl 2>/dev/null || echo "  (none found)"
    echo ""
    echo "Run 'python src/generate_dataset_tiny.py' to create tiny dataset"
    echo "Run 'python src/generate_dataset.py' to create full dataset"
    exit 1
fi

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/caz_validation_${MODEL}_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

echo "Results will be saved to: $RESULT_DIR"
echo ""

# Phase 1: Extract layer-wise metrics
echo "========================================"
echo "Phase 1: Extract Layer-Wise Metrics"
echo "========================================"
python src/extract_vectors_caz.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output "$RESULT_DIR/caz_extraction.json"

echo ""

# Phase 2: Analyze CAZ boundaries
echo "========================================"
echo "Phase 2: Analyze CAZ Boundaries"
echo "========================================"
python src/analyze_caz.py \
    --input "$RESULT_DIR/caz_extraction.json" \
    --output-dir "$RESULT_DIR" \
    --concept "$CONCEPT"

echo ""

# Phase 3: Test Mid-Stream Ablation Hypothesis
echo "========================================"
echo "Phase 3: Mid-Stream Ablation Test"
echo "========================================"
python src/ablate_caz.py \
    --model "$MODEL" \
    --caz-analysis "$RESULT_DIR/caz_analysis_${MODEL}.json" \
    --dataset "$DATASET" \
    --output "$RESULT_DIR/caz_ablation_comparison.json"

echo ""
echo "========================================"
echo "CAZ Validation Complete!"
echo "========================================"
echo ""
echo "Results directory: $RESULT_DIR"
echo ""
echo "Files created:"
echo "  - caz_extraction.json                       (Layer-wise metrics)"
echo "  - caz_analysis_${MODEL}.json                (CAZ boundaries)"
echo "  - caz_visualization_${CONCEPT}_${MODEL}.png (Visualization)"
echo "  - caz_ablation_comparison.json              (Hypothesis test results)"
echo ""
echo "View visualization:"
echo "  open $RESULT_DIR/caz_visualization_${CONCEPT}_${MODEL}.png"
echo ""
echo "Next steps:"
echo "  1. Review visualization to see CAZ boundaries"
echo "  2. Check ablation_comparison.json for hypothesis test results"
echo "  3. Compare CAZ-mid vs Post-CAZ ablation performance"
echo ""
