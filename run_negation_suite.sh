#!/bin/bash
#
# run_negation_suite.sh
#
# Run CAZ validation on "negation" concept for both GPT-2 and GPT-2 XL.
# Tests whether CAZ boundaries are concept-specific.
#
# Usage:
#   ./run_negation_suite.sh

set -e

echo "========================================"
echo "Negation Concept - CAZ Test Suite"
echo "========================================"
echo ""
echo "Dataset: negation_pairs.jsonl (20 pairs)"
echo "Models:  GPT-2 (12L) + GPT-2 XL (48L)"
echo "Concept: Affirmative vs. Negated statements"
echo ""

# Check dataset exists
if [[ ! -f "data/negation_pairs.jsonl" ]]; then
    echo "ERROR: Dataset not found: data/negation_pairs.jsonl"
    echo "Run: python src/generate_negation_dataset.py"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "Test 1/2: GPT-2 (12 layers)"
echo "========================================"
echo ""

python src/extract_vectors_caz.py \
    --model gpt2 \
    --dataset data/negation_pairs.jsonl \
    --output results/negation_gpt2_${TIMESTAMP}/caz_extraction.json

python src/analyze_caz.py \
    --input results/negation_gpt2_${TIMESTAMP}/caz_extraction.json \
    --output-dir results/negation_gpt2_${TIMESTAMP} \
    --concept negation

python src/ablate_caz.py \
    --model gpt2 \
    --caz-analysis results/negation_gpt2_${TIMESTAMP}/caz_analysis_gpt2.json \
    --dataset data/negation_pairs.jsonl \
    --output results/negation_gpt2_${TIMESTAMP}/caz_ablation_comparison.json

echo ""
echo "========================================"
echo "Test 2/2: GPT-2 XL (48 layers)"
echo "========================================"
echo ""

python src/extract_vectors_caz.py \
    --model gpt2-xl \
    --dataset data/negation_pairs.jsonl \
    --output results/negation_gpt2xl_${TIMESTAMP}/caz_extraction.json

python src/analyze_caz.py \
    --input results/negation_gpt2xl_${TIMESTAMP}/caz_extraction.json \
    --output-dir results/negation_gpt2xl_${TIMESTAMP} \
    --concept negation

python src/ablate_caz.py \
    --model gpt2-xl \
    --caz-analysis results/negation_gpt2xl_${TIMESTAMP}/caz_analysis_gpt2-xl.json \
    --dataset data/negation_pairs.jsonl \
    --output results/negation_gpt2xl_${TIMESTAMP}/caz_ablation_comparison.json

echo ""
echo "========================================"
echo "Negation Test Suite Complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  GPT-2:    results/negation_gpt2_${TIMESTAMP}/"
echo "  GPT-2 XL: results/negation_gpt2xl_${TIMESTAMP}/"
echo ""
echo "View visualizations:"
echo "  open results/negation_gpt2_${TIMESTAMP}/caz_visualization_negation_gpt2.png"
echo "  open results/negation_gpt2xl_${TIMESTAMP}/caz_visualization_negation_gpt2-xl.png"
echo ""
echo "Compare to credibility results:"
echo "  results/caz_validation_gpt2_20260310_164336/"
echo "  results/caz_validation_gpt2-xl_20260310_193156/"
echo ""
