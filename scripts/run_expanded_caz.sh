#!/usr/bin/env bash
# run_expanded_caz.sh
#
# Full unattended run: generate 5 new concept datasets, then run CAZ extraction
# across all 8 concepts × 9 models.
#
# Runtime estimate: ~6-8 hours total on RTX 500 Ada (4GB)
#   Dataset generation:  ~15 min (FuelIX API, rate-limited)
#   Extraction per run:  ~20 min for gpt2-xl, ~8 min for 1.3B, ~2 min for 125M
#   Total extractions:   8 concepts × 9 models = 72 runs
#   Estimated total:     ~6h extraction + 15m generation
#
# Usage:
#   bash scripts/run_expanded_caz.sh
#   bash scripts/run_expanded_caz.sh --skip-generation   (if datasets already exist)

set -euo pipefail
cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/expanded_caz_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/summary.log"

SKIP_GENERATION=0
for arg in "$@"; do
    [[ "$arg" == "--skip-generation" ]] && SKIP_GENERATION=1
done

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY_LOG"; }
die() { log "ERROR: $*"; exit 1; }

log "=== Expanded CAZ Run: $TIMESTAMP ==="
log "Log directory: $LOG_DIR"

# ── Verify GPU ────────────────────────────────────────────────────────────────

python3 -c "
import sys, torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    sys.exit(1)
props = torch.cuda.get_device_properties(0)
print(f'GPU: {props.name}  VRAM: {props.total_memory/1024**3:.1f} GB')
" | tee -a "$SUMMARY_LOG" || die "GPU not available"

# ── Phase 1: Dataset generation ───────────────────────────────────────────────

ALL_CONCEPTS=(credibility negation sentiment certainty plurality causation moral_valence temporal_order)
NEW_CONCEPTS=(certainty plurality causation moral_valence temporal_order)

if [[ $SKIP_GENERATION -eq 0 ]]; then
    log ""
    log "=== Phase 1: Dataset generation (FuelIX) ==="

    [[ -z "${FUELIX_API_KEY:-}" ]] && die "FUELIX_API_KEY not set"

    for concept in "${NEW_CONCEPTS[@]}"; do
        if [[ -f "data/${concept}_pairs.jsonl" ]]; then
            count=$(wc -l < "data/${concept}_pairs.jsonl")
            log "  $concept: already exists ($count lines) — skipping"
        else
            log "  Generating: $concept"
            python src/generate_new_concepts.py \
                --concept "$concept" \
                --output-dir data \
                --model claude-sonnet-4-5 \
                --api-key "$FUELIX_API_KEY" \
                --base-url https://api.fuelix.ai/v1 \
                2>&1 | tee -a "$LOG_DIR/generate_${concept}.log"
            count=$(wc -l < "data/${concept}_pairs.jsonl")
            log "  $concept: done ($count lines)"
        fi
    done
    log "Dataset generation complete."
else
    log "Skipping dataset generation (--skip-generation)"
fi

# Verify all datasets
log ""
log "=== Dataset inventory ==="
for concept in "${ALL_CONCEPTS[@]}"; do
    if [[ -f "data/${concept}_pairs.jsonl" ]]; then
        count=$(wc -l < "data/${concept}_pairs.jsonl")
        pairs=$((count / 2))
        log "  $concept: $pairs pairs"
    else
        log "  $concept: MISSING — skipping in extraction"
    fi
done

# ── Phase 2: CAZ extraction ───────────────────────────────────────────────────

# All models: id used by TransformerLens, friendly name, depth tier
declare -A MODEL_IDS=(
    ["gpt2"]="gpt2"
    ["gpt2-xl"]="gpt2-xl"
    ["gpt-neo-125m"]="EleutherAI/gpt-neo-125M"
    ["gpt-neo-1.3b"]="EleutherAI/gpt-neo-1.3B"
    ["pythia-160m"]="EleutherAI/pythia-160m"
    ["pythia-410m"]="EleutherAI/pythia-410m"
    ["opt-125m"]="facebook/opt-125m"
    ["opt-1.3b"]="facebook/opt-1.3b"
    ["qwen2-1.5b"]="Qwen/Qwen2-1.5B"
)

MODEL_ORDER=(gpt2 gpt2-xl gpt-neo-125m gpt-neo-1.3b pythia-160m pythia-410m opt-125m opt-1.3b qwen2-1.5b)

log ""
log "=== Phase 2: CAZ extraction ==="
log "  Concepts: ${ALL_CONCEPTS[*]}"
log "  Models:   ${MODEL_ORDER[*]}"
log ""

total_runs=0
done_runs=0
skipped_runs=0
failed_runs=0

# Count planned runs
for concept in "${ALL_CONCEPTS[@]}"; do
    [[ ! -f "data/${concept}_pairs.jsonl" ]] && continue
    for model_key in "${MODEL_ORDER[@]}"; do
        ((total_runs++))
    done
done
log "  Total planned extractions: $total_runs"
log ""

# Run extractions
for concept in "${ALL_CONCEPTS[@]}"; do
    [[ ! -f "data/${concept}_pairs.jsonl" ]] && continue

    for model_key in "${MODEL_ORDER[@]}"; do
        model_id="${MODEL_IDS[$model_key]}"
        results_dir="results/expanded_${concept}_${model_key}_${TIMESTAMP}"
        mkdir -p "$results_dir"

        extraction_out="$results_dir/caz_extraction.json"
        analysis_out="$results_dir/caz_analysis_${model_key}.json"
        run_log="$LOG_DIR/${concept}_${model_key}.log"

        t_start=$(date +%s)
        log "START  ${concept}/${model_key}"

        # Extraction
        if python src/extract_vectors_caz.py \
            --model "$model_id" \
            --dataset "data/${concept}_pairs.jsonl" \
            --output "$extraction_out" \
            >> "$run_log" 2>&1; then

            # Analysis
            if python src/analyze_caz.py \
                --input "$extraction_out" \
                --output-dir "$results_dir" \
                --concept "$concept" \
                >> "$run_log" 2>&1; then

                t_end=$(date +%s)
                elapsed=$((t_end - t_start))

                # Extract peak layer for summary
                peak=$(python3 -c "
import json
d = json.load(open('$analysis_out'))
b = d['boundaries']
print(f'L{b[\"caz_peak\"]} S={b[\"peak_separation\"]:.3f}')
" 2>/dev/null || echo "?")

                log "DONE   ${concept}/${model_key}  (${elapsed}s)  peak=${peak}"
                echo "${concept},${model_key},${peak},${elapsed}" >> "$LOG_DIR/results_summary.csv"
                ((done_runs++))
            else
                log "FAILED ${concept}/${model_key}  (analyze step)"
                ((failed_runs++))
            fi
        else
            log "FAILED ${concept}/${model_key}  (extract step)"
            ((failed_runs++))
        fi

        # Clear GPU cache between runs
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    done
done

# ── Phase 3: Comparative visualization ───────────────────────────────────────

log ""
log "=== Phase 3: Analysis ==="

# Print results table
if [[ -f "$LOG_DIR/results_summary.csv" ]]; then
    log ""
    log "Results summary:"
    log "  concept,model,peak,elapsed_s"
    cat "$LOG_DIR/results_summary.csv" | tee -a "$SUMMARY_LOG"
fi

log ""
log "=== Run complete ==="
log "  Completed: $done_runs / $total_runs"
log "  Failed:    $failed_runs"
log "  Log dir:   $LOG_DIR"
log ""
log "Next: python src/analyze_expanded_caz.py --timestamp $TIMESTAMP"
