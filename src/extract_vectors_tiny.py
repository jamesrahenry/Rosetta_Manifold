"""
extract_vectors_tiny.py

Tiny PoC version: Extract vectors from 1-3B models on 4GB GPU.

Uses:
- TinyLlama 1.1B (~2GB VRAM)
- Qwen2 1.5B (~3GB VRAM)
- Phi-2 2.7B (~5GB VRAM, use int8)

Usage:
    # GPU (4GB)
    python src/extract_vectors_tiny.py --model tinyllama

    # CPU fallback
    python src/extract_vectors_tiny.py --model tinyllama --device cpu

    # All tiny models
    python src/extract_vectors_tiny.py --all-models
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Tiny models that fit in 4GB VRAM (TransformerLens supported)
TINY_MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B",
    "phi2": "microsoft/phi-2",
}

# Import functions from main extract_vectors
sys.path.insert(0, str(Path(__file__).parent))
from extract_vectors import (
    compute_dom_vector,
    compute_lat_vector,
    cosine_similarity,
    load_dataset,
    ActivationCache,
)


def extract_activations_tiny(
    model: HookedTransformer,
    texts: list[str],
    layer: int,
    token_pos: int = -1,
    batch_size: int = 4,  # Smaller batches for 4GB GPU
) -> np.ndarray:
    """Extract activations with small batch size for limited VRAM."""
    cache = ActivationCache()
    hook_name = f"blocks.{layer}.hook_resid_post"

    all_activations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        cache.clear()

        tokens = model.to_tokens(batch_texts, prepend_bos=True)

        with torch.no_grad():
            model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, cache.hook_fn)],
            )

        batch_activations = cache.activations[0]

        if token_pos == -1:
            seq_lens = (tokens != model.tokenizer.pad_token_id).sum(dim=1)
            extracted = torch.stack(
                [batch_activations[j, seq_lens[j] - 1] for j in range(len(batch_texts))]
            )
        else:
            extracted = batch_activations[:, token_pos, :]

        all_activations.append(extracted)

    return torch.cat(all_activations, dim=0).numpy()


def find_best_layer_tiny(
    model: HookedTransformer,
    credible_texts: list[str],
    non_credible_texts: list[str],
    layer_start: int,
    layer_end: int,
    token_pos: int = -1,
) -> tuple[int, float, np.ndarray]:
    """Find best layer with smaller search range."""
    best_layer = layer_start
    max_separation = 0.0
    best_vector = None

    log.info("  Sweeping layers %d to %d...", layer_start, layer_end - 1)

    for layer in range(layer_start, layer_end):
        credible_acts = extract_activations_tiny(
            model, credible_texts, layer, token_pos, batch_size=2
        )
        non_credible_acts = extract_activations_tiny(
            model, non_credible_texts, layer, token_pos, batch_size=2
        )

        direction = compute_dom_vector(credible_acts, non_credible_acts)

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


def extract_tiny(
    model_id: str,
    dataset_path: Path,
    layer_start: int = 6,
    layer_end: int = 13,
    device: str = "auto",
) -> dict:
    """Extract from tiny model."""
    log.info("=== Extracting from %s ===", model_id)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Device: %s", device)

    # Load model
    log.info("Loading model...")

    # For CPU, use fp32; for GPU with 4GB, use fp16
    dtype = torch.float32 if device == "cpu" else torch.float16

    model = HookedTransformer.from_pretrained(
        model_id,
        device=device,
        dtype=dtype,
    )

    log.info("Model loaded: %d layers, hidden_dim=%d", model.cfg.n_layers, model.cfg.d_model)

    # Load dataset
    credible_texts, non_credible_texts = load_dataset(dataset_path)

    # Adjust layer range for tiny models
    max_layers = model.cfg.n_layers
    layer_end = min(layer_end, max_layers)

    # For tiny models, focus on middle layers
    if max_layers < 20:
        layer_start = max(6, max_layers // 3)
        layer_end = min(layer_end, max_layers * 2 // 3)

    log.info("Using layer range: %d to %d", layer_start, layer_end - 1)

    # Find best layer
    best_layer, separation, dom_vector = find_best_layer_tiny(
        model,
        credible_texts,
        non_credible_texts,
        layer_start,
        layer_end,
        token_pos=-1,
    )

    # Extract at best layer
    log.info("Extracting at best layer %d...", best_layer)
    credible_acts = extract_activations_tiny(
        model, credible_texts, best_layer, -1, batch_size=2
    )
    non_credible_acts = extract_activations_tiny(
        model, non_credible_texts, best_layer, -1, batch_size=2
    )

    # Compute LAT
    log.info("Computing LAT vector...")
    lat_vector = compute_lat_vector(credible_acts, non_credible_acts)

    # Agreement
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
        "token_pos": -1,
        "layer_range": [layer_start, layer_end],
        "tiny_poc": True,
    }

    log.info("=== Extraction complete ===")
    return results


def compute_alignment_tiny(results_list: list[dict]) -> dict:
    """Compute alignment with looser threshold for tiny models."""
    n_models = len(results_list)
    model_names = [r["model_id"].split("/")[-1] for r in results_list]

    log.info("=== Computing cross-model alignment ===")

    alignment = {
        "models": model_names,
        "dom_similarities": {},
        "lat_similarities": {},
    }

    for i in range(n_models):
        for j in range(i + 1, n_models):
            name_i = model_names[i]
            name_j = model_names[j]

            dom_i = np.array(results_list[i]["dom_vector"])
            dom_j = np.array(results_list[j]["dom_vector"])
            dom_sim = cosine_similarity(dom_i, dom_j)

            lat_i = np.array(results_list[i]["lat_vector"])
            lat_j = np.array(results_list[j]["lat_vector"])
            lat_sim = cosine_similarity(lat_i, lat_j)

            pair_key = f"{name_i} vs {name_j}"
            alignment["dom_similarities"][pair_key] = float(dom_sim)
            alignment["lat_similarities"][pair_key] = float(lat_sim)

            log.info("  %s:", pair_key)
            log.info("    DoM: %.4f", dom_sim)
            log.info("    LAT: %.4f", lat_sim)

    avg_dom = np.mean(list(alignment["dom_similarities"].values()))
    avg_lat = np.mean(list(alignment["lat_similarities"].values()))

    alignment["avg_dom_similarity"] = float(avg_dom)
    alignment["avg_lat_similarity"] = float(avg_lat)

    # Looser threshold for tiny models
    prh_threshold = 0.3
    prh_result = avg_dom >= prh_threshold or avg_lat >= prh_threshold

    log.info("Average cross-model similarity:")
    log.info("  DoM: %.4f", avg_dom)
    log.info("  LAT: %.4f", avg_lat)
    log.info("PRH test (threshold=%.2f): %s", prh_threshold, "PASS" if prh_result else "FAIL")

    alignment["prh_threshold"] = prh_threshold
    alignment["prh_pass"] = prh_result

    return alignment


def main():
    parser = argparse.ArgumentParser(description="Tiny PoC: Extract vectors from small models")
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (shorthand like 'gpt2' or full HuggingFace ID)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Extract from all tiny models",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/credibility_pairs.jsonl",
        help="Path to dataset (defaults to full 100-pair dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase2_vectors_tiny.json",
        help="Output path",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use",
    )
    args = parser.parse_args()

    # Check dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.error("Dataset not found: %s", dataset_path)
        log.error("Run: python src/generate_dataset_tiny.py")
        sys.exit(1)

    # Resolve models
    if args.all_models:
        models = list(TINY_MODELS.values())
    elif args.model:
        # Check if it's a shorthand, otherwise use as-is
        if args.model in TINY_MODELS:
            models = [TINY_MODELS[args.model]]
        else:
            # Accept any model string (for gpt2-large, gpt2-xl, etc.)
            models = [args.model]
    else:
        log.error("Must specify --model or --all-models")
        sys.exit(1)

    log.info("=== Tiny PoC: Phase 2 Vector Extraction ===")
    log.info("Models: %s", [m.split("/")[-1] for m in models])
    log.info("Dataset: %s", dataset_path)

    # Extract
    all_results = []
    for model_id in models:
        results = extract_tiny(
            model_id=model_id,
            dataset_path=dataset_path,
            device=args.device,
        )
        all_results.append(results)

    # Alignment
    alignment = None
    if len(all_results) > 1:
        alignment = compute_alignment_tiny(all_results)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "extractions": all_results,
        "alignment": alignment,
        "tiny_poc": True,
    }

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    log.info("Results saved to %s", output_path)
    log.info("=== Tiny PoC complete ===")


if __name__ == "__main__":
    main()
