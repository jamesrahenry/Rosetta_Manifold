"""
extract_vectors_caz.py

Concept Assembly Zone (CAZ) Validation — Layer-Wise Metric Tracking

Computes three metrics at each layer (CAZ Framework, Henry 2026):
  1. Separation (S): Fisher-normalized centroid distance (§3.2)
  2. Concept Coherence (C): Explained variance of primary component
  3. Concept Velocity (V): Smoothed rate of change of separation

Previously used TransformerLens (HookedTransformer) for extraction.
Now uses rosetta_tools — raw HuggingFace transformers, no model whitelist,
no config-conversion layer that breaks on new architectures.

Produces the same output JSON format as prior runs for compatibility
with analyze_caz.py and compare_all_concepts.py.

Usage:
    # Any HuggingFace model by ID:
    python src/extract_vectors_caz.py --model gpt2-xl

    # With explicit dataset:
    python src/extract_vectors_caz.py --model gpt2-xl \\
        --dataset data/credibility_pairs.jsonl

    # Frontier model (no whitelist required):
    python src/extract_vectors_caz.py \\
        --model meta-llama/Meta-Llama-3-8B \\
        --dataset data/credibility_pairs.jsonl

See: Concept_Assembly_Zone/CAZ_Framework.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer

# rosetta_tools — install with: pip install -e /path/to/rosetta_tools
from rosetta_tools.gpu_utils import (
    get_device,
    get_dtype,
    log_device_info,
    log_vram,
    release_model,
)
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import compute_separation, compute_coherence, compute_velocity
from rosetta_tools.dataset import load_pairs, texts_by_label, validate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer-wise sweep — now delegates to rosetta_tools
# ---------------------------------------------------------------------------


def extract_layer_wise_metrics(
    model,
    tokenizer,
    pos_texts: list[str],
    neg_texts: list[str],
    device: str,
    batch_size: int = 8,
) -> dict:
    """
    Extract CAZ metrics across all layers.

    Delegates activation extraction to rosetta_tools.extraction
    (HuggingFace output_hidden_states=True, last-token pooling) and
    metric computation to rosetta_tools.caz.

    Returns a dict matching the historical JSON format produced by
    the TransformerLens version, for compatibility with downstream
    analysis scripts.
    """
    log.info(
        "Extracting all-layer activations for %d pos / %d neg texts...",
        len(pos_texts),
        len(neg_texts),
    )

    # extract_layer_activations returns one float32 array per layer
    # (including embedding layer at index 0), shape [n_texts, hidden_dim]
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

    # Drop the embedding layer (index 0) — transformer block layers start at 1.
    # Historical results used transformer block outputs only (blocks.0 .. blocks.N-1).
    pos_by_layer = pos_by_layer[1:]
    neg_by_layer = neg_by_layer[1:]

    n_layers = len(pos_by_layer)
    log.info("Computing CAZ metrics across %d layers...", n_layers)

    separations = []
    coherences = []
    dom_vectors = []

    for layer_idx, (pos, neg) in enumerate(zip(pos_by_layer, neg_by_layer)):
        S = compute_separation(pos, neg)
        C = compute_coherence(pos, neg)

        # DoM vector — kept for downstream alignment scripts
        pos64 = pos.astype(np.float64)
        neg64 = neg.astype(np.float64)
        direction = pos64.mean(axis=0) - neg64.mean(axis=0)
        norm = np.linalg.norm(direction)
        dom = (direction / norm).tolist() if norm > 0 else direction.tolist()

        raw_dist = float(np.linalg.norm(pos64.mean(axis=0) - neg64.mean(axis=0)))

        separations.append(S)
        coherences.append(C)
        dom_vectors.append(dom)

        log.info("  Layer %2d: S=%.4f C=%.4f raw_dist=%.4f", layer_idx, S, C, raw_dist)

    # Velocity — smoothed first derivative of separation
    vel_array = compute_velocity(separations, window=3)

    layer_metrics = []
    for i in range(n_layers):
        layer_metrics.append(
            {
                "layer": i,
                "separation_fisher": separations[i],
                "coherence": coherences[i],
                "raw_distance": float(
                    np.linalg.norm(
                        pos_by_layer[i].astype(np.float64).mean(axis=0)
                        - neg_by_layer[i].astype(np.float64).mean(axis=0)
                    )
                ),
                "dom_vector": dom_vectors[i],
                "velocity": float(vel_array[i]),
            }
        )

    return {
        "n_layers": n_layers,
        "metrics": layer_metrics,
    }


# ---------------------------------------------------------------------------
# Dataset loading — now via rosetta_tools.dataset
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """
    Load a contrastive pair JSONL and return (pos_texts, neg_texts).

    Validates before loading — surfaces data quality issues early
    rather than discovering them mid-run.
    """
    issues = validate_dataset(path)
    if issues:
        log.warning("Dataset validation warnings for %s:", path)
        for issue in issues:
            log.warning("  %s", issue)

    pairs = load_pairs(path)
    pos_texts, neg_texts = texts_by_label(pairs)

    log.info(
        "Loaded %d pairs (%d pos, %d neg) from %s",
        len(pairs),
        len(pos_texts),
        len(neg_texts),
        path.name,
    )
    return pos_texts, neg_texts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def extract_caz_data(
    model_id: str,
    dataset_path: Path,
    device: str = "auto",
    batch_size: int = 8,
) -> dict:
    """
    Extract CAZ metrics for a model.

    Args:
        model_id:      HuggingFace model ID (any architecture supported by HF).
        dataset_path:  Path to a contrastive pairs JSONL file.
        device:        "cuda", "cpu", or "auto".
        batch_size:    Forward-pass batch size.  Reduce if OOM.

    Returns:
        Dictionary with CAZ analysis results, compatible with the historical
        JSON format produced by the TransformerLens version.
    """
    device = get_device(device)
    dtype = get_dtype(device)

    log.info("=== CAZ Extraction: %s ===", model_id)
    log_device_info(device, dtype)

    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    log.info("Loading model...")
    model = AutoModel.from_pretrained(model_id, dtype=dtype)
    model.eval()
    model = model.to(device)
    log_vram("after model load")

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    log.info("Model: %d layers, hidden_dim=%d", n_layers, hidden_dim)

    pos_texts, neg_texts = load_dataset(dataset_path)

    layer_data = extract_layer_wise_metrics(
        model,
        tokenizer,
        pos_texts,
        neg_texts,
        device=device,
        batch_size=batch_size,
    )

    results = {
        "model_id": model_id,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        # token_pos field preserved for compatibility with analysis scripts
        # "last" corresponds to the historical -1 / last-token extraction
        "token_pos": -1,
        "layer_data": layer_data,
    }

    release_model(model)
    log_vram("after model release")

    log.info("=== CAZ extraction complete ===")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAZ Validation: Extract layer-wise metrics (rosetta_tools)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "HuggingFace model ID. Any architecture supported by AutoModel. "
            "Examples: gpt2-xl, meta-llama/Meta-Llama-3-8B, mistralai/Mistral-7B-v0.1"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/credibility_pairs.jsonl",
        help="Path to contrastive pairs JSONL (default: data/credibility_pairs.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for results JSON. "
            "Default: results/caz_<model_slug>_<timestamp>/caz_extraction.json"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Forward-pass batch size (default: 8; reduce if OOM)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    results = extract_caz_data(
        model_id=args.model,
        dataset_path=dataset_path,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        from datetime import datetime

        model_slug = args.model.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"results/caz_{model_slug}_{timestamp}/caz_extraction.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info("Results saved to %s", output_path)
    log.info("Next: python src/analyze_caz.py --input %s", output_path)


if __name__ == "__main__":
    main()
