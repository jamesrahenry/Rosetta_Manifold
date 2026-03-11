#!/bin/bash
#
# run_sentiment_suite.sh
#
# Run CAZ validation on "sentiment polarity" concept for both GPT-2 and GPT-2 XL.
# Tests positive vs. negative emotional valence.
#
# Usage:
#   ./run_sentiment_suite.sh

set -e

echo "========================================"
echo "Sentiment Polarity - CAZ Test Suite"
echo "========================================"
echo ""
echo "Dataset: sentiment_pairs.jsonl (100 pairs)"
echo "Models:  GPT-2 (12L) + GPT-2 XL (48L)"
echo "Concept: Positive vs. Negative emotional valence"
echo ""

# Check dataset exists
if [[ ! -f "data/sentiment_pairs.jsonl" ]]; then
    echo "ERROR: Dataset not found: data/sentiment_pairs.jsonl"
    echo "Run: python src/generate_sentiment_dataset.py"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "Test 1/2: GPT-2 (12 layers)"
echo "========================================"
echo ""

RESULT_DIR_GPT2="results/${TIMESTAMP}_sentiment_gpt2"
mkdir -p "$RESULT_DIR_GPT2"

python src/extract_vectors_caz.py \
    --model gpt2 \
    --dataset data/sentiment_pairs.jsonl \
    --output "$RESULT_DIR_GPT2/caz_extraction.json"

python src/analyze_caz.py \
    --input "$RESULT_DIR_GPT2/caz_extraction.json" \
    --output-dir "$RESULT_DIR_GPT2" \
    --concept sentiment

python src/ablate_caz.py \
    --model gpt2 \
    --caz-analysis "$RESULT_DIR_GPT2/caz_analysis_gpt2.json" \
    --dataset data/sentiment_pairs.jsonl \
    --output "$RESULT_DIR_GPT2/caz_ablation_comparison.json"

echo ""
echo "========================================"
echo "Test 2/2: GPT-2 XL (48 layers)"
echo "========================================"
echo ""

RESULT_DIR_XL="results/${TIMESTAMP}_sentiment_gpt2xl"
mkdir -p "$RESULT_DIR_XL"

python src/extract_vectors_caz.py \
    --model gpt2-xl \
    --dataset data/sentiment_pairs.jsonl \
    --output "$RESULT_DIR_XL/caz_extraction.json"

python src/analyze_caz.py \
    --input "$RESULT_DIR_XL/caz_extraction.json" \
    --output-dir "$RESULT_DIR_XL" \
    --concept sentiment

python src/ablate_caz.py \
    --model gpt2-xl \
    --caz-analysis "$RESULT_DIR_XL/caz_analysis_gpt2-xl.json" \
    --dataset data/sentiment_pairs.jsonl \
    --output "$RESULT_DIR_XL/caz_ablation_comparison.json"

echo ""
echo "========================================"
echo "Sentiment Test Suite Complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  GPT-2:    $RESULT_DIR_GPT2/"
echo "  GPT-2 XL: $RESULT_DIR_XL/"
echo ""
echo "View visualizations:"
echo "  open $RESULT_DIR_GPT2/caz_visualization_sentiment_gpt2.png"
echo "  open $RESULT_DIR_XL/caz_visualization_sentiment_gpt2-xl.png"
echo ""
echo "Compare to previous concepts:"
echo "  Credibility: results/caz_validation_gpt2_20260310_164336/"
echo "  Negation:    results/negation_gpt2_20260310_210541/"
echo ""
