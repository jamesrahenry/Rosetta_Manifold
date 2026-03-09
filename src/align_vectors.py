"""
align_vectors.py

Phase 2 Extension: Cross-Architecture Alignment via Orthogonal Procrustes.

Implements the Orthogonal Procrustes problem solution to align latent spaces
of different models before comparing their credibility vectors. This addresses
the "Procrustes Problem" where models may have arbitrarily rotated latent spaces.

Method:
1. Extract activations from both models on a shared calibration dataset.
2. Compute the optimal orthogonal rotation matrix R that minimizes ||A - RB||_F.
3. Apply R to the target model's vector: V_aligned = R @ V_target.
4. Compute cosine similarity between V_source and V_aligned.

Usage:
    python src/align_vectors.py --source llama3 --target mistral \
        --vectors results/phase2_vectors.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from transformer_lens import HookedTransformer

from extract_vectors import extract_activations, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configurations (shared with extract_vectors.py)
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "qwen": "Qwen/Qwen2.5-7B",
}


def load_vector_data(path: Path, model_name: str) -> dict:
    """Load vector extraction results for a specific model."""
    with path.open() as f:
        data = json.load(f)

    for extraction in data["extractions"]:
        model_id = extraction["model_id"]
        if model_name in model_id or model_id.endswith(model_name):
            return extraction

    raise ValueError(f"Model {model_name} not found in {path}")


def compute_procrustes_alignment(
    source_acts: np.ndarray, target_acts: np.ndarray
) -> np.ndarray:
    """
    Compute the optimal orthogonal rotation matrix R.

    Minimizes ||source_acts - target_acts @ R.T||_F
    equivalent to maximizing tr(R.T @ target_acts.T @ source_acts).

    Args:
        source_acts: (N, d) activations from source model
        target_acts: (N, d) activations from target model

    Returns:
        R: (d, d) orthogonal rotation matrix
    """
    # Center the data
    source_centered = source_acts - source_acts.mean(axis=0)
    target_centered = target_acts - target_acts.mean(axis=0)

    # Compute Procrustes solution
    # scipy.linalg.orthogonal_procrustes(A, B) solves for R such that ||A @ R - B|| is minimized
    # We pass A=source, B=target, so it finds R such that source @ R ≈ target
    # This means R maps source -> target
    R_forward, scale = orthogonal_procrustes(source_centered, target_centered)

    # We want to map target -> source, i.e., target @ R_inverse ≈ source
    # Since R is orthogonal, R_inverse = R.T
    return R_forward.T


def align_and_compare(
    source_model_id: str,
    target_model_id: str,
    source_layer: int,
    target_layer: int,
    source_vector: np.ndarray,
    target_vector: np.ndarray,
    dataset_path: Path,
    n_calibration: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Align target model to source model and compare vectors.

    Args:
        source_model_id: Source model ID
        target_model_id: Target model ID
        source_layer: Layer index for source model
        target_layer: Layer index for target model
        source_vector: Vector from source model
        target_vector: Vector from target model
        dataset_path: Path to calibration dataset
        n_calibration: Number of samples for calibration
        device: Device to use

    Returns:
        Dictionary with alignment results
    """
    log.info("=== Aligning %s -> %s ===", target_model_id, source_model_id)

    # Load dataset
    credible_texts, non_credible_texts = load_dataset(dataset_path)
    all_texts = credible_texts + non_credible_texts
    calibration_texts = all_texts[:n_calibration]
    log.info("Using %d samples for calibration", len(calibration_texts))

    # Extract activations from source model
    log.info("Extracting source activations (%s, layer %d)...", source_model_id, source_layer)
    source_model = HookedTransformer.from_pretrained(
        source_model_id, device=device, dtype=torch.float16 if device == "cuda" else torch.float32
    )
    source_acts = extract_activations(source_model, calibration_texts, source_layer)
    del source_model
    torch.cuda.empty_cache()

    # Extract activations from target model
    log.info("Extracting target activations (%s, layer %d)...", target_model_id, target_layer)
    target_model = HookedTransformer.from_pretrained(
        target_model_id, device=device, dtype=torch.float16 if device == "cuda" else torch.float32
    )
    target_acts = extract_activations(target_model, calibration_texts, target_layer)
    del target_model
    torch.cuda.empty_cache()

    # Compute alignment
    log.info("Computing Orthogonal Procrustes rotation...")
    # Note: scipy's orthogonal_procrustes returns R such that ||A - B @ R|| is minimized.
    # So aligned_target = target @ R
    R = compute_procrustes_alignment(source_acts, target_acts)

    # Align target vector
    # Vector shape is (d,), so we treat it as (1, d) for multiplication
    # aligned_vector = vector @ R
    aligned_target_vector = target_vector @ R

    # Compute similarities
    raw_similarity = np.dot(source_vector, target_vector) / (
        np.linalg.norm(source_vector) * np.linalg.norm(target_vector)
    )
    aligned_similarity = np.dot(source_vector, aligned_target_vector) / (
        np.linalg.norm(source_vector) * np.linalg.norm(aligned_target_vector)
    )

    log.info("Raw Cosine Similarity:     %.4f", raw_similarity)
    log.info("Aligned Cosine Similarity: %.4f", aligned_similarity)

    return {
        "source_model": source_model_id,
        "target_model": target_model_id,
        "raw_similarity": float(raw_similarity),
        "aligned_similarity": float(aligned_similarity),
        "improvement": float(aligned_similarity - raw_similarity),
    }


def main():
    parser = argparse.ArgumentParser(description="Align vectors across architectures")
    parser.add_argument("--source", type=str, required=True, help="Source model name (e.g. llama3)")
    parser.add_argument("--target", type=str, required=True, help="Target model name (e.g. mistral)")
    parser.add_argument("--vectors", type=str, required=True, help="Path to phase2_vectors.json")
    parser.add_argument("--dataset", type=str, default="data/credibility_pairs.jsonl")
    parser.add_argument("--n-calibration", type=int, default=100, help="Calibration samples")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu"
    if args.device != "auto":
        device = args.device

    # Resolve model IDs
    source_id = SUPPORTED_MODELS.get(args.source, args.source)
    target_id = SUPPORTED_MODELS.get(args.target, args.target)

    # Load vectors
    vectors_path = Path(args.vectors)
    source_data = load_vector_data(vectors_path, args.source)
    target_data = load_vector_data(vectors_path, args.target)

    # Get vectors (using DoM by default)
    source_vec = np.array(source_data["dom_vector"])
    target_vec = np.array(target_data["dom_vector"])

    # Run alignment
    results = align_and_compare(
        source_id,
        target_id,
        source_data["best_layer"],
        target_data["best_layer"],
        source_vec,
        target_vec,
        Path(args.dataset),
        args.n_calibration,
        device,
    )

    # Save results
    output_path = Path("results/alignment_procrustes.json")
    if output_path.exists():
        with output_path.open() as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(results)
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
