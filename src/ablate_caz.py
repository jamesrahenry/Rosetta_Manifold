"""
ablate_caz.py

Mid-Stream Ablation Hypothesis Testing

Tests the CAZ prediction that ablation applied WITHIN the assembly zone produces
better behavioral suppression with lower collateral damage than post-CAZ ablation.

Compares ablation at five positions:
  1. CAZ Start (early assembly)
  2. CAZ Mid (during assembly)
  3. CAZ Peak (maximum separation)
  4. CAZ End (late assembly)
  5. Post-CAZ (after assembly complete)

For each position, measures:
  - Separation reduction (behavioral suppression)
  - KL divergence (capability preservation)

Hypothesis: CAZ-mid should have high suppression + low KL compared to post-CAZ.

Usage:
    python src/ablate_caz.py \
        --model gpt2 \
        --caz-analysis results/caz_analysis_gpt2.json \
        --dataset data/credibility_pairs.jsonl

See: Concept_Assembly_Zone/CAZ_Framework.md (Section 2.3)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

# Shared GPU utilities (Rosetta_Program/shared/)
from rosetta_tools.gpu_utils import get_device, get_dtype, log_vram

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
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "qwen": "Qwen/Qwen2.5-7B",
}

# Evaluation prompts
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

# ---------------------------------------------------------------------------
# Activation extraction (reused)
# ---------------------------------------------------------------------------


class ActivationCache:
    """Cache activations from the residual stream."""

    def __init__(self):
        self.activations = []

    def hook_fn(self, activation, hook):
        self.activations.append(activation.detach().cpu())
        return activation

    def clear(self):
        self.activations = []


def extract_activations(
    model: HookedTransformer,
    texts: list[str],
    layer: int,
    token_pos: int = -1,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract residual stream activations from a specific layer."""
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


def compute_separation_fisher(
    credible_acts: np.ndarray,
    non_credible_acts: np.ndarray,
) -> float:
    """Compute Fisher-normalized separation."""
    mean_a = credible_acts.mean(axis=0)
    mean_b = non_credible_acts.mean(axis=0)

    centroid_dist = np.linalg.norm(mean_a - mean_b)

    var_a = np.var(credible_acts, axis=0).sum()
    var_b = np.var(non_credible_acts, axis=0).sum()

    pooled_spread = np.sqrt(0.5 * (var_a + var_b))

    if pooled_spread > 0:
        return float(centroid_dist / pooled_spread)
    return 0.0


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
        Orthogonal projection: x' = x - (x · v)v
        """
        # Compute projection
        projection = torch.einsum("bsh,h->bs", activation, self.direction)
        projection = torch.einsum("bs,h->bsh", projection, self.direction)

        # Subtract projection
        ablated = activation - projection

        return ablated

    def __enter__(self):
        self.hook_handle = self.model.add_hook(self.hook_name, self.ablation_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def compute_kl_divergence(
    model: HookedTransformer,
    baseline_logits_list: list[torch.Tensor],
    prompts: list[str],
) -> float:
    """
    Compute KL divergence between baseline and ablated model.

    Must be called with ablation hooks active.
    """
    kl_divs = []

    for i, prompt in enumerate(prompts):
        # Get ablated logits
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)

        # Last token logits
        ablated_logits = logits[0, -1, :].cpu()

        # Baseline logits
        baseline_logits = baseline_logits_list[i]

        # Compute KL divergence
        p = F.log_softmax(baseline_logits, dim=-1)
        q = F.log_softmax(ablated_logits, dim=-1)

        kl = F.kl_div(q, p, reduction="sum", log_target=True)
        kl_divs.append(kl.item())

    return float(np.mean(kl_divs))


def measure_separation_reduction(
    model: HookedTransformer,
    credible_texts: list[str],
    non_credible_texts: list[str],
    layer: int,
    baseline_separation: float,
    token_pos: int = -1,
) -> float:
    """
    Measure separation reduction with ablation active.

    Returns reduction as a fraction (0.0 = no reduction, 1.0 = complete removal).
    """
    # Extract activations with ablation active
    credible_acts = extract_activations(model, credible_texts, layer, token_pos)
    non_credible_acts = extract_activations(model, non_credible_texts, layer, token_pos)

    # Compute ablated separation
    ablated_separation = compute_separation_fisher(credible_acts, non_credible_acts)

    # Compute reduction
    if baseline_separation > 0:
        reduction = (baseline_separation - ablated_separation) / baseline_separation
    else:
        reduction = 0.0

    return float(max(0.0, reduction))  # Clamp to [0, 1]


# ---------------------------------------------------------------------------
# CAZ ablation comparison
# ---------------------------------------------------------------------------


def test_ablation_at_layer(
    model: HookedTransformer,
    direction: np.ndarray,
    layer: int,
    credible_texts: list[str],
    non_credible_texts: list[str],
    baseline_separation: float,
    baseline_logits: list[torch.Tensor],
    general_prompts: list[str],
    token_pos: int = -1,
) -> dict:
    """
    Test ablation at a specific layer.

    Args:
        model: HookedTransformer model
        direction: Direction to ablate
        layer: Layer to ablate at
        credible_texts: Credible test texts
        non_credible_texts: Non-credible test texts
        baseline_separation: Baseline separation (no ablation)
        baseline_logits: Baseline logits for KL computation
        general_prompts: General prompts for KL computation
        token_pos: Token position

    Returns:
        Dictionary with ablation metrics
    """
    log.info("  Testing ablation at layer %d...", layer)

    with DirectionalAblator(model, direction, layer):
        # Measure separation reduction
        reduction = measure_separation_reduction(
            model,
            credible_texts,
            non_credible_texts,
            layer,
            baseline_separation,
            token_pos,
        )

        # Measure KL divergence
        kl_div = compute_kl_divergence(model, baseline_logits, general_prompts)

    log.info("    Separation reduction: %.2f%%", reduction * 100)
    log.info("    KL divergence: %.4f", kl_div)

    return {
        "layer": layer,
        "separation_reduction": reduction,
        "kl_divergence": kl_div,
    }


def run_caz_ablation_comparison(
    model_id: str,
    caz_analysis: dict,
    dataset_path: Path,
    token_pos: int = -1,
    device: str = "auto",
) -> dict:
    """
    Run ablation comparison across CAZ positions.

    Args:
        model_id: HuggingFace model ID
        caz_analysis: CAZ analysis results (from analyze_caz.py)
        dataset_path: Path to credibility dataset
        token_pos: Token position
        device: Device to use ("cuda", "cpu", or "auto")

    Returns:
        Dictionary with comparison results
    """
    device = get_device(device)
    log.info("=== CAZ Ablation Comparison ===")
    log.info("Model: %s", model_id)
    log.info("Device: %s", device)

    # Load model
    log.info("Loading model...")
    model = HookedTransformer.from_pretrained(
        model_id,
        device=device,
        dtype=get_dtype(device),
    )
    log_vram("after model load")

    # Load dataset
    credible = []
    non_credible = []
    with dataset_path.open("r") as f:
        for line in f:
            record = json.loads(line)
            if record["label"] == 1:
                credible.append(record["text"])
            else:
                non_credible.append(record["text"])

    log.info(
        "Loaded %d credible, %d non-credible texts", len(credible), len(non_credible)
    )

    # Get CAZ boundaries
    boundaries = caz_analysis["boundaries"]
    caz_start = boundaries["caz_start"]
    caz_peak = boundaries["caz_peak"]
    caz_end = boundaries["caz_end"]

    # Compute CAZ mid
    caz_mid = (caz_start + caz_end) // 2

    # Get post-CAZ layer (peak + 3, or last layer)
    post_caz = min(caz_peak + 3, model.cfg.n_layers - 1)

    log.info("Ablation positions:")
    log.info("  CAZ Start: Layer %d", caz_start)
    log.info("  CAZ Mid:   Layer %d", caz_mid)
    log.info("  CAZ Peak:  Layer %d", caz_peak)
    log.info("  CAZ End:   Layer %d", caz_end)
    log.info("  Post-CAZ:  Layer %d", post_caz)

    # Compute baseline separation (at peak layer, no ablation)
    log.info("Computing baseline separation...")
    credible_acts = extract_activations(model, credible, caz_peak, token_pos)
    non_credible_acts = extract_activations(model, non_credible, caz_peak, token_pos)
    baseline_separation = compute_separation_fisher(credible_acts, non_credible_acts)
    log.info("Baseline separation: %.4f", baseline_separation)

    # Compute baseline logits for KL divergence
    log.info("Computing baseline logits...")
    baseline_logits = []
    for prompt in GENERAL_PROMPTS:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)
        baseline_logits.append(logits[0, -1, :].cpu())

    # Get direction from CAZ analysis (use peak layer DoM vector)
    peak_metrics = [m for m in caz_analysis["layer_metrics"] if m["layer"] == caz_peak][
        0
    ]
    direction = np.array(peak_metrics["dom_vector"])

    # Test ablation at each position
    log.info("Testing ablation at each CAZ position...")

    results = []

    for label, layer in [
        ("caz_start", caz_start),
        ("caz_mid", caz_mid),
        ("caz_peak", caz_peak),
        ("caz_end", caz_end),
        ("post_caz", post_caz),
    ]:
        result = test_ablation_at_layer(
            model,
            direction,
            layer,
            credible,
            non_credible,
            baseline_separation,
            baseline_logits,
            GENERAL_PROMPTS,
            token_pos,
        )
        result["position"] = label
        results.append(result)

    # Analyze results
    log.info("\n=== Results Summary ===")
    for r in results:
        log.info(
            "%s (L%d): Reduction=%.2f%%, KL=%.4f",
            r["position"].replace("_", " ").title(),
            r["layer"],
            r["separation_reduction"] * 100,
            r["kl_divergence"],
        )

    # Test hypothesis: CAZ-mid should have high reduction + low KL compared to post-CAZ
    caz_mid_result = [r for r in results if r["position"] == "caz_mid"][0]
    post_caz_result = [r for r in results if r["position"] == "post_caz"][0]

    hypothesis_test = {
        "caz_mid_reduction": caz_mid_result["separation_reduction"],
        "post_caz_reduction": post_caz_result["separation_reduction"],
        "caz_mid_kl": caz_mid_result["kl_divergence"],
        "post_caz_kl": post_caz_result["kl_divergence"],
        "kl_improvement": post_caz_result["kl_divergence"]
        - caz_mid_result["kl_divergence"],
        "hypothesis_supported": (
            caz_mid_result["separation_reduction"] >= 0.5
            and caz_mid_result["kl_divergence"] < post_caz_result["kl_divergence"]
        ),
    }

    log.info("\n=== Mid-Stream Ablation Hypothesis ===")
    log.info("CAZ-Mid vs. Post-CAZ:")
    log.info(
        "  Reduction: %.2f%% vs %.2f%%",
        hypothesis_test["caz_mid_reduction"] * 100,
        hypothesis_test["post_caz_reduction"] * 100,
    )
    log.info(
        "  KL: %.4f vs %.4f (Δ=%.4f)",
        hypothesis_test["caz_mid_kl"],
        hypothesis_test["post_caz_kl"],
        hypothesis_test["kl_improvement"],
    )
    log.info("  Hypothesis supported: %s", hypothesis_test["hypothesis_supported"])

    return {
        "model_id": model_id,
        "caz_boundaries": boundaries,
        "baseline_separation": baseline_separation,
        "ablation_results": results,
        "hypothesis_test": hypothesis_test,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAZ Mid-Stream Ablation Hypothesis Testing"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODELS.keys()) + list(SUPPORTED_MODELS.values()),
        help="Model to test",
    )
    parser.add_argument(
        "--caz-analysis",
        type=str,
        required=True,
        help="Path to caz_analysis_*.json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/credibility_pairs.jsonl",
        help="Path to credibility dataset",
    )
    parser.add_argument(
        "--token-pos",
        type=int,
        default=-1,
        help="Token position (-1 for last)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/caz_ablation_comparison.json",
        help="Output path",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use",
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

    # Load CAZ analysis
    caz_path = Path(args.caz_analysis)
    if not caz_path.exists():
        log.error("CAZ analysis not found: %s", caz_path)
        log.error("Run 'python src/analyze_caz.py' first")
        sys.exit(1)

    with caz_path.open("r") as f:
        caz_analysis = json.load(f)

    # Check dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    # Run comparison
    results = run_caz_ablation_comparison(
        model_id,
        caz_analysis,
        dataset_path,
        args.token_pos,
        device,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    log.info("\nResults saved to %s", output_path)
    log.info("=== CAZ ablation comparison complete ===")


if __name__ == "__main__":
    main()
