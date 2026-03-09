"""
extract_vectors.py

Phase 2 (C2): Extract credibility vectors across Llama 3, Mistral, and Qwen.

Implements two extraction methods:
  - Difference-of-Means (DoM): Arditi et al. (2024) — fast, interpretable
  - Linear Artificial Tomography (LAT): Zou et al. (2023) — robust to non-linearity

Tests the Platonic Representation Hypothesis by measuring cosine similarity
of credibility vectors across different model architectures.

Usage:
    # Extract from a single model:
    python src/extract_vectors.py --model meta-llama/Meta-Llama-3-8B

    # Extract from all three models:
    python src/extract_vectors.py --all-models

    # Specify layer range and token position:
    python src/extract_vectors.py --model meta-llama/Meta-Llama-3-8B \
        --layer-start 14 --layer-end 22 --token-pos -1

See: docs/Spec 2 -- Vector Extraction & Alignment Pipeline.md
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "qwen": "Qwen/Qwen2.5-7B",
}

# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


class ActivationCache:
    """Cache activations from the residual stream."""

    def __init__(self):
        self.activations = []

    def hook_fn(self, activation, hook):
        """Hook function to capture activations."""
        self.activations.append(activation.detach().cpu())
        return activation

    def clear(self):
        """Clear cached activations."""
        self.activations = []


def extract_activations(
    model: HookedTransformer,
    texts: list[str],
    layer: int,
    token_pos: int = -1,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Extract residual stream activations from a specific layer.

    Args:
        model: HookedTransformer model
        texts: List of input texts
        layer: Layer index to extract from
        token_pos: Token position to extract (-1 for last token)
        batch_size: Batch size for processing

    Returns:
        numpy array of shape (n_texts, hidden_dim)
    """
    cache = ActivationCache()
    hook_name = f"blocks.{layer}.hook_resid_post"

    all_activations = []

    # Process in batches to avoid OOM
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        cache.clear()

        # Tokenize batch
        tokens = model.to_tokens(batch_texts, prepend_bos=True)

        # Run forward pass with hook
        with torch.no_grad():
            model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, cache.hook_fn)],
            )

        # Extract activations from specified token position
        # Shape: (batch_size, seq_len, hidden_dim)
        batch_activations = cache.activations[0]

        # Extract from specified token position
        if token_pos == -1:
            # Get last token for each sequence
            seq_lens = (tokens != model.tokenizer.pad_token_id).sum(dim=1)
            extracted = torch.stack(
                [batch_activations[j, seq_lens[j] - 1] for j in range(len(batch_texts))]
            )
        else:
            extracted = batch_activations[:, token_pos, :]

        all_activations.append(extracted)

    # Concatenate all batches
    return torch.cat(all_activations, dim=0).numpy()


# ---------------------------------------------------------------------------
# Direction extraction methods
# ---------------------------------------------------------------------------


def compute_dom_vector(
    credible_activations: np.ndarray,
    non_credible_activations: np.ndarray,
) -> np.ndarray:
    """
    Compute credibility vector using Difference-of-Means (DoM).

    Method from Arditi et al. (2024): arXiv:2406.11717
    V_cred = mean(A_credible) - mean(A_non_credible)

    The result is normalized and sign-aligned so that
    V_cred^DoM · mean(A_credible) > 0, i.e. the vector points toward
    the credible cluster.

    NOTE: compute_lat_vector aligns its sign against this function's output,
    so any sign change here propagates transitively to the LAT vector as well.

    Args:
        credible_activations: Activations for credible texts (n, hidden_dim)
        non_credible_activations: Activations for non-credible texts (n, hidden_dim)

    Returns:
        Normalized direction vector (hidden_dim,) pointing toward credible cluster
    """
    mean_credible = credible_activations.mean(axis=0)
    mean_non_credible = non_credible_activations.mean(axis=0)

    direction = mean_credible - mean_non_credible

    # Normalize
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    # Sign alignment: ensure V_cred^DoM · Ā_cred > 0
    # (mean_credible need not be normalized; only the sign of the dot product matters)
    if np.dot(direction, mean_credible) < 0:
        direction = -direction

    return direction


def compute_lat_vector(
    credible_activations: np.ndarray,
    non_credible_activations: np.ndarray,
) -> np.ndarray:
    """
    Compute credibility vector using Linear Artificial Tomography (LAT).

    Method from Zou et al. (2023): arXiv:2310.01405
    Uses PCA on the difference vectors to find the principal direction.

    Args:
        credible_activations: Activations for credible texts (n, hidden_dim)
        non_credible_activations: Activations for non-credible texts (n, hidden_dim)

    Returns:
        Normalized direction vector (hidden_dim,)
    """
    # Compute pairwise differences
    # This assumes credible and non-credible are paired 1-to-1
    n = min(len(credible_activations), len(non_credible_activations))
    differences = credible_activations[:n] - non_credible_activations[:n]

    # Perform PCA to get the principal component
    # Center the data
    differences_centered = differences - differences.mean(axis=0)

    # Compute covariance matrix
    cov = np.cov(differences_centered.T)

    # Get first principal component
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    direction = eigenvectors[:, idx[0]]

    # Ensure consistent sign with DoM (positive correlation)
    dom_direction = compute_dom_vector(credible_activations, non_credible_activations)
    if np.dot(direction, dom_direction) < 0:
        direction = -direction

    return direction


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ---------------------------------------------------------------------------
# Multi-layer sweep
# ---------------------------------------------------------------------------


def find_best_layer(
    model: HookedTransformer,
    credible_texts: list[str],
    non_credible_texts: list[str],
    layer_start: int,
    layer_end: int,
    token_pos: int = -1,
) -> tuple[int, float, np.ndarray]:
    """
    Sweep across layers to find the one with maximum separation.

    Uses DoM vector magnitude as the separation metric.

    Args:
        model: HookedTransformer model
        credible_texts: List of credible texts
        non_credible_texts: List of non-credible texts
        layer_start: Start layer (inclusive)
        layer_end: End layer (exclusive)
        token_pos: Token position to extract

    Returns:
        (best_layer, max_separation, best_vector)
    """
    best_layer = layer_start
    max_separation = 0.0
    best_vector = None

    log.info("  Sweeping layers %d to %d...", layer_start, layer_end - 1)

    for layer in range(layer_start, layer_end):
        # Extract activations
        credible_acts = extract_activations(model, credible_texts, layer, token_pos)
        non_credible_acts = extract_activations(
            model, non_credible_texts, layer, token_pos
        )

        # Compute DoM vector
        direction = compute_dom_vector(credible_acts, non_credible_acts)

        # Compute separation (mean difference magnitude before normalization)
        mean_credible = credible_acts.mean(axis=0)
        mean_non_credible = non_credible_acts.mean(axis=0)
        separation = np.linalg.norm(mean_credible - mean_non_credible)

        log.info("    Layer %2d: separation = %.4f", layer, separation)

        if separation > max_separation:
            max_separation = separation
            best_layer = layer
            best_vector = direction

    log.info("  Best layer: %d (separation = %.4f)", best_layer, max_separation)
    return best_layer, max_separation, best_vector


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """
    Load credibility dataset and split into credible/non-credible texts.

    Returns:
        (credible_texts, non_credible_texts)
    """
    credible = []
    non_credible = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["label"] == 1:
                credible.append(record["text"])
            else:
                non_credible.append(record["text"])

    log.info("Loaded %d credible, %d non-credible texts", len(credible), len(non_credible))
    return credible, non_credible


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def extract_credibility_vectors(
    model_id: str,
    dataset_path: Path,
    layer_start: int = 14,
    layer_end: int = 23,
    token_pos: int = -1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Extract credibility vectors from a model using both DoM and LAT.

    Args:
        model_id: HuggingFace model ID
        dataset_path: Path to credibility_pairs.jsonl
        layer_start: Start layer for sweep
        layer_end: End layer for sweep (exclusive)
        token_pos: Token position (-1 for last)
        device: Device to use

    Returns:
        Dictionary with extraction results
    """
    log.info("=== Extracting credibility vectors from %s ===", model_id)
    log.info("Device: %s", device)

    # Load model
    log.info("Loading model...")
    model = HookedTransformer.from_pretrained(
        model_id,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    log.info("Model loaded: %d layers, hidden_dim=%d", model.cfg.n_layers, model.cfg.d_model)

    # Load dataset
    credible_texts, non_credible_texts = load_dataset(dataset_path)

    # Find best layer
    best_layer, separation, dom_vector = find_best_layer(
        model,
        credible_texts,
        non_credible_texts,
        layer_start,
        min(layer_end, model.cfg.n_layers),
        token_pos,
    )

    # Extract activations at best layer
    log.info("Extracting activations at layer %d...", best_layer)
    credible_acts = extract_activations(
        model, credible_texts, best_layer, token_pos
    )
    non_credible_acts = extract_activations(
        model, non_credible_texts, best_layer, token_pos
    )

    # Compute LAT vector
    log.info("Computing LAT vector...")
    lat_vector = compute_lat_vector(credible_acts, non_credible_acts)

    # Compute agreement between DoM and LAT
    dom_lat_similarity = cosine_similarity(dom_vector, lat_vector)
    log.info("DoM-LAT agreement: %.4f", dom_lat_similarity)

    results = {
        "model_id": model_id,
        "best_layer": best_layer,
        "separation": float(separation),
        "dom_vector": dom_vector.tolist(),
        "lat_vector": lat_vector.tolist(),
        "dom_lat_similarity": dom_lat_similarity,
        "hidden_dim": model.cfg.d_model,
        "n_layers": model.cfg.n_layers,
        "token_pos": token_pos,
        "layer_range": [layer_start, layer_end],
    }

    log.info("=== Extraction complete ===")
    return results


# ---------------------------------------------------------------------------
# Cross-model alignment
# ---------------------------------------------------------------------------


def compute_alignment_matrix(results_list: list[dict]) -> dict:
    """
    Compute pairwise cosine similarities between credibility vectors.

    Args:
        results_list: List of extraction results from different models

    Returns:
        Dictionary with alignment metrics
    """
    n_models = len(results_list)
    model_names = [r["model_id"].split("/")[-1] for r in results_list]

    log.info("=== Computing cross-model alignment ===")

    # Compute pairwise similarities for both DoM and LAT
    alignment = {
        "models": model_names,
        "dom_similarities": {},
        "lat_similarities": {},
    }

    for i in range(n_models):
        for j in range(i + 1, n_models):
            name_i = model_names[i]
            name_j = model_names[j]

            # DoM similarity
            dom_i = np.array(results_list[i]["dom_vector"])
            dom_j = np.array(results_list[j]["dom_vector"])
            dom_sim = cosine_similarity(dom_i, dom_j)

            # LAT similarity
            lat_i = np.array(results_list[i]["lat_vector"])
            lat_j = np.array(results_list[j]["lat_vector"])
            lat_sim = cosine_similarity(lat_i, lat_j)

            pair_key = f"{name_i} vs {name_j}"
            alignment["dom_similarities"][pair_key] = float(dom_sim)
            alignment["lat_similarities"][pair_key] = float(lat_sim)

            log.info("  %s:", pair_key)
            log.info("    DoM: %.4f", dom_sim)
            log.info("    LAT: %.4f", lat_sim)

    # Compute average similarity (PRH test)
    avg_dom = np.mean(list(alignment["dom_similarities"].values()))
    avg_lat = np.mean(list(alignment["lat_similarities"].values()))

    alignment["avg_dom_similarity"] = float(avg_dom)
    alignment["avg_lat_similarity"] = float(avg_lat)

    log.info("Average cross-model similarity:")
    log.info("  DoM: %.4f", avg_dom)
    log.info("  LAT: %.4f", avg_lat)

    prh_threshold = 0.5
    prh_result = avg_dom >= prh_threshold and avg_lat >= prh_threshold
    log.info("PRH test (threshold=%.2f): %s", prh_threshold, "PASS" if prh_result else "FAIL")
    if not prh_result:
        log.info(
            "  DoM: %s (%.4f)",
            "PASS" if avg_dom >= prh_threshold else "FAIL",
            avg_dom,
        )
        log.info(
            "  LAT: %s (%.4f)",
            "PASS" if avg_lat >= prh_threshold else "FAIL",
            avg_lat,
        )

    alignment["prh_threshold"] = prh_threshold
    alignment["prh_pass"] = prh_result

    return alignment


# ---------------------------------------------------------------------------
# Opik logging
# ---------------------------------------------------------------------------


def log_to_opik(results: dict, alignment: Optional[dict] = None) -> bool:
    """
    Log extraction results to Opik.

    Args:
        results: Single model extraction results
        alignment: Optional cross-model alignment results

    Returns:
        True on success, False on failure
    """
    try:
        import opik

        client = opik.Opik()

        # Log extraction run
        with opik.track():
            opik.log_trace(
                name=f"extract_vectors_{results['model_id'].split('/')[-1]}",
                input={"model_id": results["model_id"]},
                output={
                    "best_layer": results["best_layer"],
                    "separation": results["separation"],
                    "dom_lat_similarity": results["dom_lat_similarity"],
                },
                metadata={
                    "hidden_dim": results["hidden_dim"],
                    "n_layers": results["n_layers"],
                    "token_pos": results["token_pos"],
                    "layer_range": results["layer_range"],
                },
            )

            if alignment:
                opik.log_trace(
                    name="cross_model_alignment",
                    input={"models": alignment["models"]},
                    output={
                        "avg_dom_similarity": alignment["avg_dom_similarity"],
                        "avg_lat_similarity": alignment["avg_lat_similarity"],
                        "prh_pass": alignment["prh_pass"],
                    },
                    metadata={
                        "dom_similarities": alignment["dom_similarities"],
                        "lat_similarities": alignment["lat_similarities"],
                    },
                )

        log.info("Results logged to Opik")
        return True

    except Exception as exc:  # noqa: BLE001
        log.warning("Opik logging skipped: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 (C2): Extract credibility vectors across models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(SUPPORTED_MODELS.keys()) + list(SUPPORTED_MODELS.values()),
        help="Model to extract from (llama3, mistral, qwen, or full HF ID)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Extract from all three models (Llama 3, Mistral, Qwen)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/credibility_pairs.jsonl",
        help="Path to credibility dataset (default: data/credibility_pairs.jsonl)",
    )
    parser.add_argument(
        "--layer-start",
        type=int,
        default=14,
        help="Start layer for sweep (default: 14)",
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=23,
        help="End layer for sweep, exclusive (default: 23)",
    )
    parser.add_argument(
        "--token-pos",
        type=int,
        default=-1,
        help="Token position to extract (-1 for last token, default: -1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase2_vectors.json",
        help="Output path for results (default: results/phase2_vectors.json)",
    )
    parser.add_argument(
        "--skip-opik",
        action="store_true",
        help="Skip Opik logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve model(s)
    if args.all_models:
        models = list(SUPPORTED_MODELS.values())
    elif args.model:
        if args.model in SUPPORTED_MODELS:
            models = [SUPPORTED_MODELS[args.model]]
        else:
            models = [args.model]
    else:
        log.error("Must specify either --model or --all-models")
        sys.exit(1)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.error("Dataset not found: %s", dataset_path)
        log.error("Run 'python src/generate_dataset.py' first (Phase 1)")
        sys.exit(1)

    # Extract vectors from each model
    all_results = []
    for model_id in models:
        results = extract_credibility_vectors(
            model_id=model_id,
            dataset_path=dataset_path,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            token_pos=args.token_pos,
            device=device,
        )
        all_results.append(results)

        # Log to Opik
        if not args.skip_opik:
            log_to_opik(results)

    # Compute cross-model alignment if multiple models
    alignment = None
    if len(all_results) > 1:
        alignment = compute_alignment_matrix(all_results)

        if not args.skip_opik:
            log_to_opik(all_results[0], alignment)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "extractions": all_results,
        "alignment": alignment,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    log.info("Results saved to %s", output_path)
    log.info("=== Phase 2 (C2) complete ===")


if __name__ == "__main__":
    main()
