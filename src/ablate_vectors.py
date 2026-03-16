"""
ablate_vectors.py

Phase 3 (C3): Functional Verification via Directional Ablation.

Applies orthogonal projection to remove credibility directions from models,
then validates:
  1. Ablation success (credibility signal removed)
  2. Intelligence retention (KL divergence < 0.2)
  3. Cross-architecture transfer

Uses the abliterator library (FailSpy/abliterator) for directional ablation
based on Arditi et al. (2024) "Refusal in Language Models Is Mediated by a
Single Direction" - arXiv:2406.11717.

Usage:
    # Ablate a single model:
    python src/ablate_vectors.py --model llama3 \
        --vectors results/phase2_vectors.json

    # Test cross-architecture transfer:
    python src/ablate_vectors.py --model mistral \
        --vectors results/phase2_vectors.json \
        --transfer-from llama3

    # Sweep layers and components:
    python src/ablate_vectors.py --model llama3 \
        --vectors results/phase2_vectors.json \
        --sweep-layers

See: docs/Spec 3 -- Heretic Optimization and Ablation.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
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
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "qwen": "Qwen/Qwen2.5-7B",
}

# ---------------------------------------------------------------------------
# Evaluation prompts
# ---------------------------------------------------------------------------

# General-purpose prompts for KL divergence measurement
GENERAL_PROMPTS = [
    "The capital of France is",
    "To bake a cake, you need",
    "The theory of relativity states that",
    "In Python, a list is",
    "The largest planet in our solar system is",
    "Photosynthesis is the process by which",
    "The French Revolution began in",
    "A neural network consists of",
    "Shakespeare wrote",
    "The derivative of x^2 is",
]

# Credibility test prompts (should show signal removal)
CREDIBILITY_TEST_PROMPTS = [
    "A study published in Nature found that",
    "According to the CDC, vaccination rates",
    "Research from MIT demonstrates that",
    "The FDA approved the drug based on",
    "Climate scientists at NOAA report that",
]

# Non-credibility test prompts (baseline)
NON_CREDIBILITY_TEST_PROMPTS = [
    "Everyone knows that vaccines are",
    "My friend told me that climate change",
    "It's obvious that the government is",
    "Experts agree that this is the most",
    "The shocking truth about this is",
]

# ---------------------------------------------------------------------------
# Ablation via orthogonal projection
# ---------------------------------------------------------------------------


class DirectionalAblator:
    """Apply directional ablation via orthogonal projection."""

    def __init__(
        self,
        model: HookedTransformer,
        direction: np.ndarray,
        layer: int,
        component: str = "resid_post",
    ):
        """
        Initialize ablator.

        Args:
            model: HookedTransformer model
            direction: Direction to ablate (will be normalized)
            layer: Layer index to ablate
            component: Component to ablate (resid_pre, resid_mid, resid_post)
        """
        self.model = model
        self.layer = layer
        self.component = component

        # Normalize direction; cast to model dtype so fp16 GPU runs don't error
        self.direction = torch.tensor(direction, dtype=torch.float32)
        self.direction = self.direction / torch.norm(self.direction)
        self.direction = self.direction.to(
            device=model.cfg.device, dtype=model.cfg.dtype
        )

        # Hook name
        self.hook_name = f"blocks.{layer}.hook_{component}"

    def ablation_hook(self, activation, hook):
        """
        Hook function that projects out the direction.

        Uses orthogonal projection: x' = x - (x · v)v
        where v is the normalized direction to remove.
        """
        # activation shape: (batch, seq_len, hidden_dim)
        # direction shape: (hidden_dim,)

        # Compute projection: (x · v)v
        projection = torch.einsum("bsh,h->bs", activation, self.direction)
        projection = torch.einsum("bs,h->bsh", projection, self.direction)

        # Subtract projection
        ablated = activation - projection

        return ablated

    def __enter__(self):
        """Context manager entry - add hook via model.add_hook().

        NOTE: compute_kl_divergence_from_baseline_logits detects active hooks by
        inspecting model.hook_dict[*].fwd_hooks. This works because add_hook()
        appends to those lists. If this is ever changed to use run_with_hooks()
        instead, the guard in that function must be updated accordingly.
        """
        self.hook_handle = self.model.add_hook(self.hook_name, self.ablation_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def compute_kl_divergence_from_baseline_logits(
    model: HookedTransformer,
    baseline_logits_list: list[torch.Tensor],
    prompts: list[str],
) -> float:
    """
    Compute KL divergence between pre-captured baseline logits and ablated model.

    KL(P_baseline || P_ablated) where P_ablated is computed with active hooks.

    IMPORTANT: This function must be called with ablation hooks active (i.e., inside
    a DirectionalAblator context manager). Calling it without active hooks will
    silently return KL ≈ 0 because the model will produce the same logits as baseline.

    Args:
        model: Model (with ablation hooks active)
        baseline_logits_list: Pre-captured last-token logits for each prompt
        prompts: List of prompts (must match baseline_logits_list length)

    Returns:
        Average KL divergence across prompts
    """
    if len(prompts) != len(baseline_logits_list):
        raise ValueError(
            f"prompts ({len(prompts)}) and baseline_logits_list "
            f"({len(baseline_logits_list)}) must have the same length"
        )

    # Guard: verify ablation hooks are active to prevent silent KL ≈ 0 results.
    # TransformerLens populates model.hook_dict with HookPoint objects at model
    # construction time; each HookPoint has a .fwd_hooks list that is populated
    # only when model.add_hook() is called (as DirectionalAblator.__enter__ does).
    # This guard therefore correctly detects whether we are inside a
    # DirectionalAblator context. It would NOT detect hooks registered via
    # run_with_hooks() — see the note in DirectionalAblator.__enter__.
    if not model.hook_dict or not any(
        len(hp.fwd_hooks) > 0 for hp in model.hook_dict.values()
    ):
        raise RuntimeError(
            "compute_kl_divergence_from_baseline_logits must be called with "
            "ablation hooks active (inside a DirectionalAblator context manager)"
        )

    kl_scores = []

    for prompt, logits_baseline in zip(prompts, baseline_logits_list):
        tokens = model.to_tokens(prompt, prepend_bos=True)

        with torch.no_grad():
            # Get ablated logits (hooks are active in the calling context)
            logits_ablated = model(tokens)
            logits_ablated = logits_ablated[0, -1, :]

            # Ensure baseline logits are on the same device as ablated logits.
            # Baseline tensors were captured before the ablation context; if
            # anything moves the model between capture and here this prevents a
            # silent device-mismatch error inside F.kl_div.
            logits_baseline = logits_baseline.to(logits_ablated.device)

            # Convert to probabilities
            probs_baseline = F.softmax(logits_baseline, dim=-1)
            probs_ablated = F.softmax(logits_ablated, dim=-1)

            # Compute KL divergence: KL(P_baseline || P_ablated)
            kl = F.kl_div(
                probs_ablated.log(), probs_baseline, reduction="sum", log_target=False
            )

            kl_scores.append(kl.item())

    return float(np.mean(kl_scores))


def measure_activation_along_direction(
    model: HookedTransformer,
    prompts: list[str],
    direction: np.ndarray,
    layer: int,
    component: str = "resid_post",
) -> list[float]:
    """
    Measure activation strength along a direction.

    Args:
        model: HookedTransformer model
        prompts: List of prompts
        direction: Direction vector
        layer: Layer to extract from
        component: Component to extract from

    Returns:
        List of activation strengths (cosine similarity * magnitude)
    """
    direction_tensor = torch.tensor(direction, dtype=torch.float32)
    direction_tensor = direction_tensor / torch.norm(direction_tensor)
    direction_tensor = direction_tensor.to(
        device=model.cfg.device, dtype=model.cfg.dtype
    )

    hook_name = f"blocks.{layer}.hook_{component}"
    activations = []

    def capture_hook(activation, hook):
        # Take last token activation
        act = activation[0, -1, :]  # (hidden_dim,)
        # Compute dot product with direction
        strength = torch.dot(act, direction_tensor)
        activations.append(strength.item())
        return activation

    for prompt in prompts:
        tokens = model.to_tokens(prompt, prepend_bos=True)

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_name, capture_hook)])

    return activations


def generate_with_ablation(
    model: HookedTransformer,
    prompt: str,
    max_tokens: int = 30,
) -> str:
    """
    Generate text with the current model state (hooks active).

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        max_tokens: Max new tokens to generate

    Returns:
        Generated text
    """
    # Ensure we are in eval mode
    model.eval()
    with torch.no_grad():
        output = model.generate(prompt, max_new_tokens=max_tokens, verbose=False)
    return output


# ---------------------------------------------------------------------------
# Main ablation pipeline
# ---------------------------------------------------------------------------


def load_vectors(path: Path, model_name: str, method: str = "dom") -> dict:
    """
    Load extraction results from Phase 2.

    Args:
        path: Path to phase2_vectors.json
        model_name: Model name (e.g., "llama3" or full ID)
        method: "dom" or "lat"

    Returns:
        Extraction result dict with vector, layer, etc.
    """
    with path.open() as f:
        data = json.load(f)

    # Find matching model in extractions
    for extraction in data["extractions"]:
        model_id = extraction["model_id"]
        if model_name in model_id or model_id.endswith(model_name):
            return extraction

    raise ValueError(f"Model {model_name} not found in {path}")


def ablate_and_validate(
    model_id: str,
    direction: np.ndarray,
    layer: int,
    component: str,
    device: str = "auto",
) -> dict:
    """
    Ablate model and validate results.

    Args:
        model_id: HuggingFace model ID
        direction: Direction to ablate
        layer: Layer index
        component: Component name
        device: Device to use ("cuda", "cpu", or "auto")

    Returns:
        Validation results dict
    """
    device = get_device(device)
    log.info("=== Ablating %s at layer %d (%s) ===", model_id, layer, component)

    # Load model
    log.info("Loading model...")
    model = HookedTransformer.from_pretrained(
        model_id,
        device=device,
        dtype=get_dtype(device),
    )
    log_vram("after model load")

    # Measure baseline activations
    log.info("Measuring baseline activations...")
    baseline_credible = measure_activation_along_direction(
        model, CREDIBILITY_TEST_PROMPTS, direction, layer, component
    )
    baseline_non_credible = measure_activation_along_direction(
        model, NON_CREDIBILITY_TEST_PROMPTS, direction, layer, component
    )

    baseline_separation = np.mean(baseline_credible) - np.mean(baseline_non_credible)
    log.info("  Baseline separation: %.4f", baseline_separation)

    # Capture baseline logits before entering the ablation context.
    # Use .detach().clone() so the tensors are fully independent of the
    # computation graph and safe to hold across the context boundary.
    log.info("Capturing baseline logits for KL divergence...")
    baseline_logits_list = []
    for prompt in GENERAL_PROMPTS:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)
            baseline_logits_list.append(logits[0, -1, :].detach().clone())

    # Apply ablation and measure
    log.info("Applying ablation...")
    with DirectionalAblator(model, direction, layer, component) as ablator:
        # Measure ablated activations
        ablated_credible = measure_activation_along_direction(
            model, CREDIBILITY_TEST_PROMPTS, direction, layer, component
        )
        ablated_non_credible = measure_activation_along_direction(
            model, NON_CREDIBILITY_TEST_PROMPTS, direction, layer, component
        )

        ablated_separation = np.mean(ablated_credible) - np.mean(ablated_non_credible)
        log.info("  Ablated separation: %.4f", ablated_separation)

        # Compute KL divergence against pre-captured baseline logits
        log.info("Computing KL divergence on general prompts...")
        kl = compute_kl_divergence_from_baseline_logits(
            model, baseline_logits_list, GENERAL_PROMPTS
        )
        log.info("  KL divergence: %.4f", kl)

        # Generate sample to check coherence
        log.info("Generating sample with ablation...")
        sample_prompt = "The capital of France is"
        sample_output = generate_with_ablation(model, sample_prompt)
        log.info("  Output: %r", sample_output)

    # Compute reduction
    separation_reduction = (
        (baseline_separation - ablated_separation) / baseline_separation
        if baseline_separation != 0
        else 0
    )

    results = {
        "model_id": model_id,
        "layer": layer,
        "component": component,
        "baseline_separation": float(baseline_separation),
        "ablated_separation": float(ablated_separation),
        "separation_reduction": float(separation_reduction),
        "kl_divergence": float(kl),
        "kl_threshold": 0.2,
        "kl_pass": kl < 0.2,
        "ablation_success": separation_reduction > 0.5,  # >50% reduction
    }

    log.info("=== Results ===")
    log.info("  Separation reduction: %.1f%%", separation_reduction * 100)
    log.info("  KL divergence: %.4f (threshold: 0.2)", kl)
    log.info("  Ablation success: %s", "✓" if results["ablation_success"] else "✗")
    log.info("  KL threshold met: %s", "✓" if results["kl_pass"] else "✗")

    return results


def sweep_layers_and_components(
    model_id: str,
    direction: np.ndarray,
    layer_start: int,
    layer_end: int,
    components: list[str],
    device: str,
) -> list[dict]:
    """
    Sweep across layers and components to find best ablation point.

    Args:
        model_id: HuggingFace model ID
        direction: Direction to ablate
        layer_start: Start layer
        layer_end: End layer (exclusive)
        components: List of components to try
        device: Device to use

    Returns:
        List of results for each configuration
    """
    results = []

    for layer in range(layer_start, layer_end):
        for component in components:
            try:
                result = ablate_and_validate(
                    model_id, direction, layer, component, device
                )
                results.append(result)
            except Exception as e:
                log.error("Error at layer %d, component %s: %s", layer, component, e)

    # Find best configuration
    valid_results = [r for r in results if r["kl_pass"]]
    if valid_results:
        best = max(valid_results, key=lambda r: r["separation_reduction"])
        log.info("\n=== Best Configuration ===")
        log.info("  Layer: %d", best["layer"])
        log.info("  Component: %s", best["component"])
        log.info("  Separation reduction: %.1f%%", best["separation_reduction"] * 100)
        log.info("  KL divergence: %.4f", best["kl_divergence"])

    return results


# ---------------------------------------------------------------------------
# Cross-architecture transfer
# ---------------------------------------------------------------------------


def test_transfer(
    target_model_id: str,
    source_direction: np.ndarray,
    source_layer: int,
    component: str,
    device: str,
) -> dict:
    """
    Test if a direction from one model transfers to another.

    Args:
        target_model_id: Target model to ablate
        source_direction: Direction from source model
        source_layer: Layer from source model
        component: Component to ablate
        device: Device to use

    Returns:
        Transfer validation results
    """
    log.info("\n=== Testing Cross-Architecture Transfer ===")
    log.info("Target model: %s", target_model_id)
    log.info("Source layer: %d", source_layer)

    # Use same layer in target model
    results = ablate_and_validate(
        target_model_id, source_direction, source_layer, component, device
    )

    results["transfer_test"] = True
    return results


# ---------------------------------------------------------------------------
# Opik logging
# ---------------------------------------------------------------------------


def log_to_opik(results: dict) -> bool:
    """
    Log ablation results to Opik.

    Args:
        results: Ablation results dict or list of dicts

    Returns:
        True on success, False on failure
    """
    try:
        import opik

        if isinstance(results, list):
            # Log sweep results
            for result in results:
                with opik.track():
                    opik.log_trace(
                        name=f"ablation_{result['model_id'].split('/')[-1]}_L{result['layer']}",
                        input={
                            "model_id": result["model_id"],
                            "layer": result["layer"],
                            "component": result["component"],
                        },
                        output={
                            "separation_reduction": result["separation_reduction"],
                            "kl_divergence": result["kl_divergence"],
                            "ablation_success": result["ablation_success"],
                            "kl_pass": result["kl_pass"],
                        },
                        metadata={
                            "baseline_separation": result["baseline_separation"],
                            "ablated_separation": result["ablated_separation"],
                        },
                    )
        else:
            # Log single result
            with opik.track():
                opik.log_trace(
                    name=f"ablation_{results['model_id'].split('/')[-1]}",
                    input={"model_id": results["model_id"]},
                    output={
                        "separation_reduction": results["separation_reduction"],
                        "kl_divergence": results["kl_divergence"],
                        "ablation_success": results["ablation_success"],
                        "kl_pass": results["kl_pass"],
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
        description="Phase 3 (C3): Directional ablation validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODELS.keys()) + list(SUPPORTED_MODELS.values()),
        help="Model to ablate",
    )
    parser.add_argument(
        "--vectors",
        type=str,
        required=True,
        help="Path to Phase 2 vector extraction results (JSON)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dom", "lat"],
        default="dom",
        help="Extraction method to use (default: dom)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Specific layer to ablate (default: use best from Phase 2)",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["resid_pre", "resid_mid", "resid_post"],
        default="resid_post",
        help="Component to ablate (default: resid_post)",
    )
    parser.add_argument(
        "--sweep-layers",
        action="store_true",
        help="Sweep across layers to find best ablation point",
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
        "--transfer-from",
        type=str,
        help="Test cross-architecture transfer from this model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase3_ablation.json",
        help="Output path for results (default: results/phase3_ablation.json)",
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

    # Resolve model ID
    if args.model in SUPPORTED_MODELS:
        model_id = SUPPORTED_MODELS[args.model]
        model_name = args.model
    else:
        model_id = args.model
        model_name = args.model.split("/")[-1]

    # Resolve device
    device = get_device(args.device)

    # Load vectors
    vectors_path = Path(args.vectors)
    if not vectors_path.exists():
        log.error("Vectors file not found: %s", vectors_path)
        log.error("Run Phase 2 first: python src/extract_vectors.py --all-models")
        sys.exit(1)

    log.info("=== Phase 3 (C3): Directional Ablation Validation ===")
    log.info("Model: %s", model_id)
    log.info("Vectors: %s", vectors_path)
    log.info("Method: %s", args.method)
    log.info("Device: %s", device)

    # Load extraction results
    extraction = load_vectors(vectors_path, model_name, args.method)
    direction = np.array(
        extraction["dom_vector"] if args.method == "dom" else extraction["lat_vector"]
    )
    best_layer = extraction["best_layer"]

    log.info("Loaded %s vector from layer %d", args.method.upper(), best_layer)

    # Determine layer to use
    target_layer = args.layer if args.layer is not None else best_layer

    # Handle transfer test
    if args.transfer_from:
        log.info("\n=== Cross-Architecture Transfer Test ===")
        source_extraction = load_vectors(vectors_path, args.transfer_from, args.method)
        source_direction = np.array(
            source_extraction["dom_vector"]
            if args.method == "dom"
            else source_extraction["lat_vector"]
        )
        source_layer = source_extraction["best_layer"]

        results = test_transfer(
            model_id, source_direction, source_layer, args.component, device
        )

    # Handle sweep
    elif args.sweep_layers:
        components = ["resid_pre", "resid_mid", "resid_post"]
        results = sweep_layers_and_components(
            model_id, direction, args.layer_start, args.layer_end, components, device
        )

    # Single ablation
    else:
        results = ablate_and_validate(
            model_id, direction, target_layer, args.component, device
        )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    log.info("\nResults saved to %s", output_path)

    # Log to Opik
    if not args.skip_opik:
        log_to_opik(results)

    log.info("\n=== Phase 3 (C3) complete ===")


if __name__ == "__main__":
    main()
