"""
extract_caz_frontier.py

Frontier-scale CAZ extraction for H100 runs.

Designed for multi-concept, multi-model runs on datacenter hardware.
No model whitelist — accepts any HuggingFace model ID.
Produces the same JSON format as extract_vectors_caz.py for full
compatibility with analyze_caz.py, compare_all_concepts.py, etc.

Key differences from extract_vectors_caz.py:
  - Multi-concept: runs all specified concepts in sequence on one model load
  - Explicit n-pairs cap: prevents runaway memory from very large datasets
  - Pre-flight validation: validates all datasets before loading model
  - Checkpoint saves: writes per-concept results as they complete;
    if the run is interrupted, completed concepts are not lost
  - Summary JSON: writes an index of all results at the end

Usage:
    # Single model, all default concepts:
    python src/extract_caz_frontier.py \\
        --model meta-llama/Meta-Llama-3-8B

    # Explicit concepts:
    python src/extract_caz_frontier.py \\
        --model meta-llama/Meta-Llama-3-8B \\
        --concepts credibility negation sentiment

    # Multiple models (sequential — H100 has enough VRAM for one 8B at a time):
    python src/extract_caz_frontier.py \\
        --model meta-llama/Meta-Llama-3-8B \\
        --concepts credibility negation sentiment moral_valence

    python src/extract_caz_frontier.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --concepts credibility negation sentiment moral_valence

    # Validate datasets without running (pre-flight check):
    python src/extract_caz_frontier.py \\
        --model meta-llama/Meta-Llama-3-8B \\
        --validate-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer

from rosetta_tools.gpu_utils import (
    get_device,
    get_dtype,
    log_device_info,
    log_vram,
    release_model,
)
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import compute_separation, compute_coherence, compute_velocity
from rosetta_tools.dataset import (
    load_pairs,
    texts_by_label,
    validate_dataset,
    dataset_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concept → dataset path mapping
# ---------------------------------------------------------------------------

# All paths relative to the Rosetta_Manifold root (where this script is run from).
CONCEPT_DATASETS: dict[str, str] = {
    "credibility": "data/credibility_pairs.jsonl",
    "negation": "data/negation_pairs.jsonl",
    "sentiment": "data/sentiment_pairs.jsonl",
    "causation": "data/causation_pairs.jsonl",
    "certainty": "data/certainty_pairs.jsonl",
    "moral_valence": "data/moral_valence_pairs.jsonl",
    "plurality": "data/plurality_pairs.jsonl",
    "temporal_order": "data/temporal_order_pairs.jsonl",
}

DEFAULT_CONCEPTS = ["credibility", "negation", "sentiment"]


# ---------------------------------------------------------------------------
# Per-layer metric extraction
# ---------------------------------------------------------------------------


def extract_layer_wise_metrics(
    model,
    tokenizer,
    pos_texts: list[str],
    neg_texts: list[str],
    device: str,
    batch_size: int,
) -> dict:
    """
    Extract CAZ metrics across all transformer layers for one concept.

    Returns a dict in the same format as extract_vectors_caz.py for
    compatibility with downstream analysis scripts.
    """
    log.info(
        "  Extracting activations: %d pos / %d neg texts, batch_size=%d",
        len(pos_texts),
        len(neg_texts),
        batch_size,
    )

    pos_by_layer = extract_layer_activations(
        model,
        tokenizer,
        pos_texts,
        device=device,
        batch_size=batch_size,
        pool="last",
    )
    neg_by_layer = extract_layer_activations(
        model,
        tokenizer,
        neg_texts,
        device=device,
        batch_size=batch_size,
        pool="last",
    )

    # Drop embedding layer (index 0); historical results use transformer blocks only
    pos_by_layer = pos_by_layer[1:]
    neg_by_layer = neg_by_layer[1:]

    n_layers = len(pos_by_layer)
    log.info("  Computing metrics across %d layers...", n_layers)

    separations: list[float] = []
    coherences: list[float] = []
    dom_vectors: list[list[float]] = []
    raw_distances: list[float] = []

    for layer_idx, (pos, neg) in enumerate(zip(pos_by_layer, neg_by_layer)):
        S = compute_separation(pos, neg)
        C = compute_coherence(pos, neg)

        pos64 = pos.astype(np.float64)
        neg64 = neg.astype(np.float64)
        diff = pos64.mean(axis=0) - neg64.mean(axis=0)
        raw_dist = float(np.linalg.norm(diff))
        norm = np.linalg.norm(diff)
        dom = (diff / norm).tolist() if norm > 0 else diff.tolist()

        separations.append(S)
        coherences.append(C)
        dom_vectors.append(dom)
        raw_distances.append(raw_dist)

        if layer_idx % 8 == 0 or layer_idx == n_layers - 1:
            log.info("  Layer %2d/%d: S=%.4f C=%.4f", layer_idx, n_layers - 1, S, C)

    vel_array = compute_velocity(separations, window=3)

    layer_metrics = [
        {
            "layer": i,
            "separation_fisher": separations[i],
            "coherence": coherences[i],
            "raw_distance": raw_distances[i],
            "dom_vector": dom_vectors[i],
            "velocity": float(vel_array[i]),
        }
        for i in range(n_layers)
    ]

    peak_layer = int(np.argmax(separations))
    peak_sep = separations[peak_layer]
    peak_pct = round(100.0 * peak_layer / n_layers, 1)

    log.info(
        "  Peak: L%d (%.1f%% depth) S=%.4f",
        peak_layer,
        peak_pct,
        peak_sep,
    )

    return {
        "n_layers": n_layers,
        "metrics": layer_metrics,
        "peak_layer": peak_layer,
        "peak_separation": peak_sep,
        "peak_depth_pct": peak_pct,
    }


# ---------------------------------------------------------------------------
# Pre-flight dataset validation
# ---------------------------------------------------------------------------


def validate_all_datasets(
    concepts: list[str],
    n_pairs: int | None,
    data_root: Path,
) -> dict[str, list[str]]:
    """
    Validate all concept datasets before loading the model.

    Returns a dict mapping concept name to list of issues (empty = clean).
    Logs a summary. Raises SystemExit if any dataset is missing.
    """
    log.info("=== Pre-flight dataset validation ===")
    all_issues: dict[str, list[str]] = {}
    missing = []

    for concept in concepts:
        dataset_rel = CONCEPT_DATASETS[concept]
        path = data_root / dataset_rel
        if not path.exists():
            log.error("  MISSING: %s → %s", concept, path)
            missing.append(concept)
            continue

        issues = validate_dataset(path)
        summary = dataset_summary(path)
        n_available = summary["n_pairs"]
        n_use = min(n_pairs, n_available) if n_pairs else n_available

        if issues:
            log.warning(
                "  WARN %s: %d issues, %d/%d pairs",
                concept,
                len(issues),
                n_use,
                n_available,
            )
            for issue in issues:
                log.warning("    %s", issue)
        else:
            log.info("  OK   %s: %d pairs (using %d)", concept, n_available, n_use)

        all_issues[concept] = issues

    if missing:
        log.error("Aborting: missing datasets for %s", missing)
        log.error("Run generate_dataset.py scripts to create them first.")
        sys.exit(1)

    total_warn = sum(len(v) for v in all_issues.values())
    if total_warn == 0:
        log.info("All datasets clean.")
    else:
        log.warning(
            "%d total validation warnings across %d concepts.",
            total_warn,
            len(concepts),
        )

    return all_issues


# ---------------------------------------------------------------------------
# Single-concept extraction (called in a loop per concept)
# ---------------------------------------------------------------------------


def extract_concept(
    concept: str,
    model,
    tokenizer,
    device: str,
    n_pairs: int | None,
    batch_size: int,
    data_root: Path,
    out_dir: Path,
) -> dict:
    """
    Run extraction for one concept and save results immediately.

    Returns the summary dict (without the full layer_data, which is in the file).
    """
    dataset_path = data_root / CONCEPT_DATASETS[concept]
    pairs = load_pairs(dataset_path)

    if n_pairs and len(pairs) > n_pairs:
        log.info("  Capping to %d pairs (dataset has %d)", n_pairs, len(pairs))
        pairs = pairs[:n_pairs]

    pos_texts, neg_texts = texts_by_label(pairs)

    t0 = time.time()
    layer_data = extract_layer_wise_metrics(
        model,
        tokenizer,
        pos_texts,
        neg_texts,
        device=device,
        batch_size=batch_size,
    )
    elapsed = time.time() - t0

    model_id = getattr(model, "name_or_path", "unknown")

    results = {
        "model_id": model_id,
        "concept": concept,
        "n_pairs": len(pairs),
        "hidden_dim": model.config.hidden_size,
        "n_layers": model.config.num_hidden_layers,
        "token_pos": -1,
        "extraction_seconds": round(elapsed, 1),
        "layer_data": layer_data,
    }

    # Checkpoint save — write immediately so partial runs are not lost
    concept_path = out_dir / f"caz_{concept}.json"
    with concept_path.open("w") as f:
        json.dump(results, f, indent=2)
    log.info("  Saved → %s  (%.1fs)", concept_path.name, elapsed)

    return {
        "concept": concept,
        "n_pairs": len(pairs),
        "peak_layer": layer_data["peak_layer"],
        "peak_separation": layer_data["peak_separation"],
        "peak_depth_pct": layer_data["peak_depth_pct"],
        "extraction_seconds": round(elapsed, 1),
        "output_file": str(concept_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Resolve concepts
    if args.concepts:
        unknown = [c for c in args.concepts if c not in CONCEPT_DATASETS]
        if unknown:
            log.error("Unknown concepts: %s", unknown)
            log.error("Available: %s", sorted(CONCEPT_DATASETS.keys()))
            sys.exit(1)
        concepts = args.concepts
    else:
        concepts = DEFAULT_CONCEPTS

    data_root = Path(args.data_root)
    n_pairs = args.n_pairs

    # Pre-flight validation always runs
    validate_all_datasets(concepts, n_pairs, data_root)

    if args.validate_only:
        log.info("--validate-only: stopping after dataset validation.")
        return

    # Output directory
    model_slug = args.model.replace("/", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"frontier_{model_slug}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # Device + dtype
    device = get_device(args.device)
    dtype = get_dtype(device)

    log.info("=== Loading model: %s ===", args.model)
    log_device_info(device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=dtype)
    model.eval()
    model = model.to(device)
    log_vram("after model load")

    log.info(
        "Model ready: %d layers, hidden_dim=%d",
        model.config.num_hidden_layers,
        model.config.hidden_size,
    )

    # Extract all concepts
    run_summary: list[dict] = []
    t_run_start = time.time()

    for i, concept in enumerate(concepts):
        log.info("=== Concept %d/%d: %s ===", i + 1, len(concepts), concept)
        log_vram(f"before {concept}")

        summary = extract_concept(
            concept=concept,
            model=model,
            tokenizer=tokenizer,
            device=device,
            n_pairs=n_pairs,
            batch_size=args.batch_size,
            data_root=data_root,
            out_dir=out_dir,
        )
        run_summary.append(summary)

    total_elapsed = time.time() - t_run_start

    # Write run index
    run_index = {
        "model_id": args.model,
        "concepts": concepts,
        "n_pairs_cap": n_pairs,
        "device": device,
        "dtype": str(dtype),
        "total_seconds": round(total_elapsed, 1),
        "timestamp": timestamp,
        "results": run_summary,
    }
    index_path = out_dir / "run_summary.json"
    with index_path.open("w") as f:
        json.dump(run_index, f, indent=2)

    # Release model
    release_model(model)
    log_vram("after model release")

    # Print summary table
    log.info("=== Run complete: %.1fs ===", total_elapsed)
    log.info("%-18s %6s %8s %8s", "Concept", "Pairs", "Peak L", "Peak S")
    log.info("-" * 46)
    for r in run_summary:
        log.info(
            "%-18s %6d %8d %8.4f",
            r["concept"],
            r["n_pairs"],
            r["peak_layer"],
            r["peak_separation"],
        )
    log.info("\nResults: %s", out_dir)
    log.info("Index:   %s", index_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frontier CAZ extraction — multi-concept, H100-optimized"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "HuggingFace model ID. Any architecture. "
            "Examples: meta-llama/Meta-Llama-3-8B, mistralai/Mistral-7B-v0.1, "
            "Qwen/Qwen2.5-7B"
        ),
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=None,
        metavar="CONCEPT",
        help=(
            f"Concepts to extract. Default: {DEFAULT_CONCEPTS}. "
            f"Available: {sorted(CONCEPT_DATASETS.keys())}"
        ),
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=100,
        metavar="N",
        help="Max pairs per concept (default: 100; use 0 for all)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Root directory containing the data/ folder (default: .)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output (default: results/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Forward-pass batch size (default: 16 for H100; reduce if OOM)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate datasets and exit without running extraction",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
