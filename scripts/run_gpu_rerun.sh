#!/bin/bash
#
# run_gpu_rerun.sh
#
# Re-runs all three science suites (credibility, negation, sentiment) on GPU
# for both GPT-2 and GPT-2 XL.  Results are written to new timestamped
# directories — existing CPU results are never touched.
#
# Each run is logged to logs/gpu_rerun_<suite>_<timestamp>.log so you have
# a full record of timing and VRAM usage to compare against the CPU runs.
#
# Usage:
#   ./run_gpu_rerun.sh
#
# Prerequisites:
#   CUDA must be available (verified at startup).
#   All datasets must exist in data/.

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/gpu_rerun_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ── Helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/summary.log"; }
die() { log "ERROR: $*"; exit 1; }

run_phase() {
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    log "START  $label"
    local t0=$SECONDS
    "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    local elapsed=$(( SECONDS - t0 ))
    if [ $rc -eq 0 ]; then
        log "DONE   $label  (${elapsed}s)"
    else
        log "FAILED $label  (exit $rc after ${elapsed}s)"
        exit $rc
    fi
}

# ── Preflight ─────────────────────────────────────────────────────────────────

log "============================================================"
log "  Rosetta Manifold — GPU Re-run"
log "  $(date)"
log "============================================================"

# Confirm CUDA is visible before spending any time on downloads
python3 -c "
import sys, torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available — aborting GPU rerun')
    sys.exit(1)
props = torch.cuda.get_device_properties(0)
print(f'GPU: {props.name}  VRAM: {props.total_memory/1024**3:.1f} GB')
print(f'torch: {torch.__version__}  CUDA: {torch.version.cuda}')
" | tee -a "$LOG_DIR/summary.log" || die "GPU not available"

for ds in data/credibility_pairs.jsonl data/negation_pairs.jsonl data/sentiment_pairs.jsonl; do
    [ -f "$ds" ] || die "Dataset missing: $ds"
done
log "Datasets verified."
log ""

# ── Suite 1: Credibility ──────────────────────────────────────────────────────

log "============================================================"
log "  Suite 1/3: Credibility  (200 pairs, gpt2 + gpt2-xl)"
log "============================================================"

CRED_GPT2="results/gpu_credibility_gpt2_${TIMESTAMP}"
CRED_XL="results/gpu_credibility_gpt2xl_${TIMESTAMP}"
mkdir -p "$CRED_GPT2" "$CRED_XL"

run_phase "credibility_gpt2_extract" \
    python src/extract_vectors_caz.py \
        --model gpt2 \
        --dataset data/credibility_pairs.jsonl \
        --output "$CRED_GPT2/caz_extraction.json"

run_phase "credibility_gpt2_analyze" \
    python src/analyze_caz.py \
        --input "$CRED_GPT2/caz_extraction.json" \
        --output-dir "$CRED_GPT2" \
        --concept credibility

run_phase "credibility_gpt2_ablate" \
    python src/ablate_caz.py \
        --model gpt2 \
        --caz-analysis "$CRED_GPT2/caz_analysis_gpt2.json" \
        --dataset data/credibility_pairs.jsonl \
        --output "$CRED_GPT2/caz_ablation_comparison.json"

run_phase "credibility_gpt2xl_extract" \
    python src/extract_vectors_caz.py \
        --model gpt2-xl \
        --dataset data/credibility_pairs.jsonl \
        --output "$CRED_XL/caz_extraction.json"

run_phase "credibility_gpt2xl_analyze" \
    python src/analyze_caz.py \
        --input "$CRED_XL/caz_extraction.json" \
        --output-dir "$CRED_XL" \
        --concept credibility

run_phase "credibility_gpt2xl_ablate" \
    python src/ablate_caz.py \
        --model gpt2-xl \
        --caz-analysis "$CRED_XL/caz_analysis_gpt2-xl.json" \
        --dataset data/credibility_pairs.jsonl \
        --output "$CRED_XL/caz_ablation_comparison.json"

log ""

# ── Suite 2: Negation ─────────────────────────────────────────────────────────

log "============================================================"
log "  Suite 2/3: Negation  (200 pairs, gpt2 + gpt2-xl)"
log "============================================================"

NEG_GPT2="results/gpu_negation_gpt2_${TIMESTAMP}"
NEG_XL="results/gpu_negation_gpt2xl_${TIMESTAMP}"
mkdir -p "$NEG_GPT2" "$NEG_XL"

run_phase "negation_gpt2_extract" \
    python src/extract_vectors_caz.py \
        --model gpt2 \
        --dataset data/negation_pairs.jsonl \
        --output "$NEG_GPT2/caz_extraction.json"

run_phase "negation_gpt2_analyze" \
    python src/analyze_caz.py \
        --input "$NEG_GPT2/caz_extraction.json" \
        --output-dir "$NEG_GPT2" \
        --concept negation

run_phase "negation_gpt2_ablate" \
    python src/ablate_caz.py \
        --model gpt2 \
        --caz-analysis "$NEG_GPT2/caz_analysis_gpt2.json" \
        --dataset data/negation_pairs.jsonl \
        --output "$NEG_GPT2/caz_ablation_comparison.json"

run_phase "negation_gpt2xl_extract" \
    python src/extract_vectors_caz.py \
        --model gpt2-xl \
        --dataset data/negation_pairs.jsonl \
        --output "$NEG_XL/caz_extraction.json"

run_phase "negation_gpt2xl_analyze" \
    python src/analyze_caz.py \
        --input "$NEG_XL/caz_extraction.json" \
        --output-dir "$NEG_XL" \
        --concept negation

run_phase "negation_gpt2xl_ablate" \
    python src/ablate_caz.py \
        --model gpt2-xl \
        --caz-analysis "$NEG_XL/caz_analysis_gpt2-xl.json" \
        --dataset data/negation_pairs.jsonl \
        --output "$NEG_XL/caz_ablation_comparison.json"

log ""

# ── Suite 3: Sentiment ────────────────────────────────────────────────────────

log "============================================================"
log "  Suite 3/3: Sentiment  (200 pairs, gpt2 + gpt2-xl)"
log "============================================================"

SENT_GPT2="results/gpu_sentiment_gpt2_${TIMESTAMP}"
SENT_XL="results/gpu_sentiment_gpt2xl_${TIMESTAMP}"
mkdir -p "$SENT_GPT2" "$SENT_XL"

run_phase "sentiment_gpt2_extract" \
    python src/extract_vectors_caz.py \
        --model gpt2 \
        --dataset data/sentiment_pairs.jsonl \
        --output "$SENT_GPT2/caz_extraction.json"

run_phase "sentiment_gpt2_analyze" \
    python src/analyze_caz.py \
        --input "$SENT_GPT2/caz_extraction.json" \
        --output-dir "$SENT_GPT2" \
        --concept sentiment

run_phase "sentiment_gpt2_ablate" \
    python src/ablate_caz.py \
        --model gpt2 \
        --caz-analysis "$SENT_GPT2/caz_analysis_gpt2.json" \
        --dataset data/sentiment_pairs.jsonl \
        --output "$SENT_GPT2/caz_ablation_comparison.json"

run_phase "sentiment_gpt2xl_extract" \
    python src/extract_vectors_caz.py \
        --model gpt2-xl \
        --dataset data/sentiment_pairs.jsonl \
        --output "$SENT_XL/caz_extraction.json"

run_phase "sentiment_gpt2xl_analyze" \
    python src/analyze_caz.py \
        --input "$SENT_XL/caz_extraction.json" \
        --output-dir "$SENT_XL" \
        --concept sentiment

run_phase "sentiment_gpt2xl_ablate" \
    python src/ablate_caz.py \
        --model gpt2-xl \
        --caz-analysis "$SENT_XL/caz_analysis_gpt2-xl.json" \
        --dataset data/sentiment_pairs.jsonl \
        --output "$SENT_XL/caz_ablation_comparison.json"

log ""

# ── Summary ───────────────────────────────────────────────────────────────────

TOTAL=$SECONDS
log "============================================================"
log "  ALL SUITES COMPLETE"
log "  Total wall time: ${TOTAL}s ($(( TOTAL / 60 ))m $(( TOTAL % 60 ))s)"
log "============================================================"
log ""
log "GPU result directories:"
log "  Credibility:  $CRED_GPT2/"
log "               $CRED_XL/"
log "  Negation:     $NEG_GPT2/"
log "               $NEG_XL/"
log "  Sentiment:    $SENT_GPT2/"
log "               $SENT_XL/"
log ""
log "CPU baselines for comparison:"
log "  Credibility:  results/caz_validation_gpt2_20260310_164336/"
log "               results/caz_validation_gpt2-xl_20260310_193156/"
log "  Negation:     results/negation_gpt2_20260310_210541/"
log "               results/negation_gpt2xl_20260310_210541/"
log "  Sentiment:    results/20260310_233429_sentiment_gpt2/"
log "               results/20260310_233429_sentiment_gpt2xl/"
log ""
log "Timing detail: $LOG_DIR/"
log ""
