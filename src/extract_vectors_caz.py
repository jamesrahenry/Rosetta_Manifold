"""
extract_vectors_caz.py

Concept Assembly Zone (CAZ) Validation - Layer-Wise Metric Tracking

Extends Phase 2 extraction to track metrics across ALL layers instead of just
finding the "best" layer. Enables empirical validation of the CAZ framework.

Computes three metrics at each layer (CAZ Framework, Henry 2026):
  1. Separation (S): Fisher-normalized centroid distance
  2. Concept Coherence (C): Explained variance of primary component
  3. Concept Velocity (V): Rate of change of separation across layers

Usage:
    # Extract layer-wise metrics from a single model:
    python src/extract_vectors_caz.py --model gpt2

    # Use full credibility dataset:
    python src/extract_vectors_caz.py --model gpt2 \
        --dataset data/credibility_pairs.jsonl

See: Concept_Assembly_Zone/CAZ_Framework.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

# Shared GPU utilities (Rosetta_Program/shared/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.gpu_utils import get_device, get_dtype, log_vram

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
    # GPT-2 family
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    # GPT-Neo family
    "gpt-neo-125m": "EleutherAI/gpt-neo-125M",
    "gpt-neo-1.3b": "EleutherAI/gpt-neo-1.3B",
    "gpt-neo-2.7b": "EleutherAI/gpt-neo-2.7B",
    # Pythia family
    "pythia-14m": "EleutherAI/pythia-14m",
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    # OPT family
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    # Qwen2 family
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B",
    # Frontier (require significant compute)
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "qwen": "Qwen/Qwen2.5-7B",
}

# ---------------------------------------------------------------------------
# Activation extraction (reused from extract_vectors.py)
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

    # Concatenate all batches and cast to float32 before numpy conversion.
    # fp16 activations cause variance overflow in Fisher normalization at deep
    # layers of large models — float32 is required for accurate metric computation
    # regardless of model dtype. This is cheap: the forward pass stays in fp16.
    return torch.cat(all_activations, dim=0).float().numpy()


# ---------------------------------------------------------------------------
# CAZ Metrics (from CAZ Framework)
# ---------------------------------------------------------------------------


def compute_separation_fisher(
    credible_acts: np.ndarray,
    non_credible_acts: np.ndarray,
) -> float:
    """
    Compute Fisher-normalized separation metric.

    S(l) = ||μ_A - μ_B|| / sqrt((1/2)(tr(Σ_A) + tr(Σ_B)))

    This normalizes centroid distance by within-class spread to account
    for varying dispersion across layers.

    Args:
        credible_acts: Activations for credible class (n, d)
        non_credible_acts: Activations for non-credible class (n, d)

    Returns:
        Fisher-normalized separation
    """
    # Compute centroids
    mean_a = credible_acts.mean(axis=0)
    mean_b = non_credible_acts.mean(axis=0)

    # Centroid distance
    centroid_dist = np.linalg.norm(mean_a - mean_b)

    # Within-class variances (trace of covariance)
    var_a = np.var(credible_acts, axis=0).sum()
    var_b = np.var(non_credible_acts, axis=0).sum()

    # Fisher normalization
    pooled_spread = np.sqrt(0.5 * (var_a + var_b))

    if pooled_spread > 0:
        return float(centroid_dist / pooled_spread)
    return 0.0


def compute_concept_coherence(
    credible_acts: np.ndarray,
    non_credible_acts: np.ndarray,
) -> float:
    """
    Compute Concept Coherence as explained variance of primary component.

    C(l) = λ_1 / Σ λ_i

    High coherence means the concept direction is geometrically clean
    (low-dimensional structure).

    Args:
        credible_acts: Activations for credible class (n, d)
        non_credible_acts: Activations for non-credible class (n, d)

    Returns:
        Explained variance ratio of first component
    """
    # Pool activations
    pooled = np.vstack([credible_acts, non_credible_acts])

    # Center
    pooled_centered = pooled - pooled.mean(axis=0)

    # Compute covariance
    cov = np.cov(pooled_centered.T)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Filter numerical noise

    if len(eigenvalues) == 0:
        return 0.0

    # Sort descending
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Explained variance ratio
    return float(eigenvalues[0] / eigenvalues.sum())


def compute_dom_vector(
    credible_acts: np.ndarray,
    non_credible_acts: np.ndarray,
) -> np.ndarray:
    """
    Compute Difference-of-Means vector (for alignment tracking).

    Args:
        credible_acts: Activations for credible class (n, d)
        non_credible_acts: Activations for non-credible class (n, d)

    Returns:
        Normalized DoM vector
    """
    mean_credible = credible_acts.mean(axis=0)
    mean_non_credible = non_credible_acts.mean(axis=0)

    direction = mean_credible - mean_non_credible

    # Normalize
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    return direction


# ---------------------------------------------------------------------------
# Layer-wise sweep
# ---------------------------------------------------------------------------


def extract_layer_wise_metrics(
    model: HookedTransformer,
    credible_texts: list[str],
    non_credible_texts: list[str],
    token_pos: int = -1,
) -> dict:
    """
    Extract CAZ metrics across all layers.

    Args:
        model: HookedTransformer model
        credible_texts: List of credible texts
        non_credible_texts: List of non-credible texts
        token_pos: Token position to extract

    Returns:
        Dictionary with layer-wise metrics
    """
    n_layers = model.cfg.n_layers
    log.info("Extracting layer-wise metrics across %d layers...", n_layers)

    layer_metrics = []

    for layer in range(n_layers):
        # Extract activations
        credible_acts = extract_activations(model, credible_texts, layer, token_pos)
        non_credible_acts = extract_activations(
            model, non_credible_texts, layer, token_pos
        )

        # Compute metrics
        separation = compute_separation_fisher(credible_acts, non_credible_acts)
        coherence = compute_concept_coherence(credible_acts, non_credible_acts)
        dom_vector = compute_dom_vector(credible_acts, non_credible_acts)

        # Raw centroid distance (for reference)
        mean_credible = credible_acts.mean(axis=0)
        mean_non_credible = non_credible_acts.mean(axis=0)
        raw_distance = float(np.linalg.norm(mean_credible - mean_non_credible))

        layer_metrics.append(
            {
                "layer": layer,
                "separation_fisher": separation,
                "coherence": coherence,
                "raw_distance": raw_distance,
                "dom_vector": dom_vector.tolist(),
            }
        )

        log.info(
            "  Layer %2d: S=%.4f C=%.4f raw_dist=%.4f",
            layer,
            separation,
            coherence,
            raw_distance,
        )

    # Compute velocity (rate of change of separation)
    for i in range(len(layer_metrics)):
        if i == 0:
            velocity = 0.0  # No prior layer
        else:
            prev_sep = layer_metrics[i - 1]["separation_fisher"]
            curr_sep = layer_metrics[i]["separation_fisher"]
            velocity = curr_sep - prev_sep

        layer_metrics[i]["velocity"] = velocity

    return {
        "n_layers": n_layers,
        "metrics": layer_metrics,
    }


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

    log.info(
        "Loaded %d credible, %d non-credible texts", len(credible), len(non_credible)
    )
    return credible, non_credible


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def extract_caz_data(
    model_id: str,
    dataset_path: Path,
    token_pos: int = -1,
    device: str = "auto",
) -> dict:
    """
    Extract CAZ metrics for a model.

    Args:
        model_id: HuggingFace model ID
        dataset_path: Path to credibility_pairs.jsonl
        token_pos: Token position (-1 for last)
        device: Device to use ("cuda", "cpu", or "auto")

    Returns:
        Dictionary with CAZ analysis results
    """
    device = get_device(device)
    log.info("=== CAZ Extraction: %s ===", model_id)
    log.info("Device: %s", device)

    # Load model in fp16 on GPU (fast forward passes).
    # Activations are cast to float32 before metric computation — see
    # extract_activations() — which avoids Fisher normalization overflow
    # in deep layers without requiring a full fp32 model load.
    base_dtype = get_dtype(device)

    # Load model
    log.info("Loading model...")
    model = HookedTransformer.from_pretrained(
        model_id,
        device=device,
        dtype=base_dtype,
    )
    log_vram("after model load")
    log.info(
        "Model loaded: %d layers, hidden_dim=%d", model.cfg.n_layers, model.cfg.d_model
    )

    # Load dataset
    credible_texts, non_credible_texts = load_dataset(dataset_path)

    # Extract layer-wise metrics
    layer_data = extract_layer_wise_metrics(
        model,
        credible_texts,
        non_credible_texts,
        token_pos,
    )

    results = {
        "model_id": model_id,
        "hidden_dim": model.cfg.d_model,
        "n_layers": model.cfg.n_layers,
        "token_pos": token_pos,
        "layer_data": layer_data,
    }

    log.info("=== CAZ extraction complete ===")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAZ Validation: Extract layer-wise metrics"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODELS.keys()) + list(SUPPORTED_MODELS.values()),
        help="Model to extract from",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/credibility_pairs.jsonl",
        help="Path to credibility dataset (default: data/credibility_pairs.jsonl)",
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
        default="results/caz_extraction.json",
        help="Output path for results (default: results/caz_extraction.json)",
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

    # Resolve model
    if args.model in SUPPORTED_MODELS:
        model_id = SUPPORTED_MODELS[args.model]
    else:
        model_id = args.model

    # Resolve device
    device = get_device(args.device)

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.error("Dataset not found: %s", dataset_path)
        log.error("Run 'python src/generate_dataset.py' first")
        sys.exit(1)

    # Extract CAZ data
    results = extract_caz_data(
        model_id=model_id,
        dataset_path=dataset_path,
        token_pos=args.token_pos,
        device=device,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info("Results saved to %s", output_path)
    log.info("Next: python src/analyze_caz.py --input %s", output_path)


if __name__ == "__main__":
    main()
