"""
analyze_caz.py

Concept Assembly Zone (CAZ) Boundary Detection

Analyzes layer-wise metrics to programmatically identify:
  1. CAZ Start: Where concept formation begins
  2. CAZ Peak: Where separation/coherence is maximum
  3. CAZ End: Where concept crystallization completes
  4. Pre-CAZ and Post-CAZ regions

Uses thresholds and derivative-based heuristics to detect phase transitions.

Usage:
    python src/analyze_caz.py --input results/caz_extraction.json

See: Concept_Assembly_Zone/CAZ_Framework.md
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CAZ Boundary Detection
# ---------------------------------------------------------------------------


def detect_caz_boundaries(metrics: list[dict], threshold_percentile: float = 0.5) -> dict:
    """
    Detect CAZ boundaries from layer-wise metrics.

    Strategy:
      - CAZ Peak: Layer with maximum separation
      - CAZ Start: First layer where separation exceeds threshold (e.g., 50% of peak)
      - CAZ End: Last layer in sustained high-separation region before decline

    Args:
        metrics: List of layer metrics (from extract_vectors_caz.py)
        threshold_percentile: Fraction of peak separation to use as threshold

    Returns:
        Dictionary with boundary indices and zone labels
    """
    n_layers = len(metrics)

    # Extract separation and coherence arrays
    separations = np.array([m["separation_fisher"] for m in metrics])
    coherences = np.array([m["coherence"] for m in metrics])
    velocities = np.array([m["velocity"] for m in metrics])

    # Find peak
    peak_idx = int(np.argmax(separations))
    peak_separation = separations[peak_idx]

    log.info("Peak separation: %.4f at layer %d", peak_separation, peak_idx)

    # Detect CAZ start: first layer exceeding threshold
    threshold = threshold_percentile * peak_separation
    above_threshold = separations >= threshold

    if not above_threshold.any():
        log.warning("No layers exceed threshold - using peak as single-point CAZ")
        return {
            "caz_start": peak_idx,
            "caz_peak": peak_idx,
            "caz_end": peak_idx,
            "peak_separation": float(peak_separation),
            "threshold": float(threshold),
        }

    caz_start = int(np.argmax(above_threshold))  # First True index

    # Detect CAZ end: last layer in sustained high region
    # Look for where separation drops below threshold after peak
    post_peak = separations[peak_idx:]
    post_peak_below = post_peak < threshold

    if post_peak_below.any():
        # Find first drop below threshold after peak
        relative_end = int(np.argmax(post_peak_below))
        caz_end = peak_idx + relative_end
    else:
        # Separation stays high until end of model
        caz_end = n_layers - 1

    log.info("CAZ boundaries detected:")
    log.info("  Start: Layer %d (S=%.4f)", caz_start, separations[caz_start])
    log.info("  Peak:  Layer %d (S=%.4f)", peak_idx, peak_separation)
    log.info("  End:   Layer %d (S=%.4f)", caz_end, separations[caz_end])
    log.info("  Width: %d layers", caz_end - caz_start + 1)

    # Define zones
    zones = []
    for layer in range(n_layers):
        if layer < caz_start:
            zone = "pre_caz"
        elif layer == peak_idx:
            zone = "caz_peak"
        elif caz_start <= layer <= caz_end:
            zone = "caz"
        else:
            zone = "post_caz"

        zones.append(zone)

    return {
        "caz_start": caz_start,
        "caz_peak": peak_idx,
        "caz_end": caz_end,
        "caz_width": caz_end - caz_start + 1,
        "peak_separation": float(peak_separation),
        "threshold": float(threshold),
        "zones": zones,
        "separations": separations.tolist(),
        "coherences": coherences.tolist(),
        "velocities": velocities.tolist(),
    }


def compute_caz_statistics(metrics: list[dict], boundaries: dict) -> dict:
    """
    Compute statistical properties of CAZ regions.

    Args:
        metrics: Layer metrics
        boundaries: CAZ boundaries from detect_caz_boundaries()

    Returns:
        Dictionary with regional statistics
    """
    caz_start = boundaries["caz_start"]
    caz_peak = boundaries["caz_peak"]
    caz_end = boundaries["caz_end"]

    separations = np.array([m["separation_fisher"] for m in metrics])
    coherences = np.array([m["coherence"] for m in metrics])
    velocities = np.array([m["velocity"] for m in metrics])

    # Extract regions
    pre_caz = separations[:caz_start] if caz_start > 0 else np.array([])
    caz = separations[caz_start : caz_end + 1]
    post_caz = separations[caz_end + 1 :] if caz_end < len(separations) - 1 else np.array([])

    stats = {
        "pre_caz": {
            "mean_separation": float(pre_caz.mean()) if len(pre_caz) > 0 else 0.0,
            "std_separation": float(pre_caz.std()) if len(pre_caz) > 0 else 0.0,
            "n_layers": len(pre_caz),
        },
        "caz": {
            "mean_separation": float(caz.mean()),
            "std_separation": float(caz.std()),
            "max_separation": float(caz.max()),
            "mean_coherence": float(coherences[caz_start : caz_end + 1].mean()),
            "n_layers": len(caz),
        },
        "post_caz": {
            "mean_separation": float(post_caz.mean()) if len(post_caz) > 0 else 0.0,
            "std_separation": float(post_caz.std()) if len(post_caz) > 0 else 0.0,
            "n_layers": len(post_caz),
        },
        "velocity": {
            "max_positive": float(velocities.max()),
            "max_negative": float(velocities.min()),
            "mean_in_caz": float(velocities[caz_start : caz_end + 1].mean()),
        },
    }

    log.info("Regional statistics:")
    log.info("  Pre-CAZ:  mean_S=%.4f (n=%d)", stats["pre_caz"]["mean_separation"], stats["pre_caz"]["n_layers"])
    log.info("  CAZ:      mean_S=%.4f max_S=%.4f (n=%d)", stats["caz"]["mean_separation"], stats["caz"]["max_separation"], stats["caz"]["n_layers"])
    log.info("  Post-CAZ: mean_S=%.4f (n=%d)", stats["post_caz"]["mean_separation"], stats["post_caz"]["n_layers"])

    return stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_caz(
    metrics: list[dict],
    boundaries: dict,
    output_path: Path,
    model_name: str = "Model",
    concept: str = "Concept",
) -> None:
    """
    Create CAZ visualization showing separation, coherence, and velocity.

    Args:
        metrics: Layer metrics
        boundaries: CAZ boundaries
        output_path: Path to save figure
        model_name: Model name for title
        concept: Concept name for title (e.g., 'Credibility', 'Negation')
    """
    layers = np.array([m["layer"] for m in metrics])
    separations = np.array([m["separation_fisher"] for m in metrics])
    coherences = np.array([m["coherence"] for m in metrics])
    velocities = np.array([m["velocity"] for m in metrics])

    caz_start = boundaries["caz_start"]
    caz_peak = boundaries["caz_peak"]
    caz_end = boundaries["caz_end"]
    threshold = boundaries["threshold"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Separation with CAZ boundaries
    ax = axes[0]
    ax.plot(layers, separations, "o-", label="Separation (S)", linewidth=2)
    ax.axhline(threshold, color="gray", linestyle="--", alpha=0.5, label=f"Threshold ({threshold:.2f})")
    ax.axvspan(caz_start, caz_end, alpha=0.2, color="green", label="CAZ")
    ax.axvline(caz_peak, color="red", linestyle="--", alpha=0.7, label=f"Peak (L{caz_peak})")
    ax.set_ylabel("Separation (Fisher-normalized)", fontsize=11)
    ax.set_title(f"Concept Assembly Zone: {concept.title()} ({model_name})", fontsize=13, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Plot 2: Coherence
    ax = axes[1]
    ax.plot(layers, coherences, "o-", color="purple", label="Coherence (C)", linewidth=2)
    ax.axvspan(caz_start, caz_end, alpha=0.2, color="green")
    ax.axvline(caz_peak, color="red", linestyle="--", alpha=0.7)
    ax.set_ylabel("Concept Coherence", fontsize=11)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Plot 3: Velocity
    ax = axes[2]
    ax.plot(layers, velocities, "o-", color="orange", label="Velocity (V)", linewidth=2)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.axvspan(caz_start, caz_end, alpha=0.2, color="green")
    ax.axvline(caz_peak, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Separation Velocity", fontsize=11)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Visualization saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def analyze_caz(input_path: Path, output_dir: Path, concept: str = "concept") -> dict:
    """
    Analyze CAZ from extraction results.

    Args:
        input_path: Path to caz_extraction.json
        output_dir: Directory to save analysis results
        concept: Name of the concept being analyzed (e.g., 'credibility')

    Returns:
        Analysis results dictionary
    """
    log.info("=== CAZ Analysis ===")
    log.info("Loading extraction data from %s", input_path)

    # Load extraction data
    with input_path.open("r") as f:
        extraction_data = json.load(f)

    model_id = extraction_data["model_id"]
    model_name = model_id.split("/")[-1]
    metrics = extraction_data["layer_data"]["metrics"]

    log.info("Model: %s (%d layers)", model_id, len(metrics))

    # Detect boundaries
    boundaries = detect_caz_boundaries(metrics)

    # Compute statistics
    statistics = compute_caz_statistics(metrics, boundaries)

    # Create visualization
    viz_path = output_dir / f"caz_visualization_{concept}_{model_name}.png"
    visualize_caz(metrics, boundaries, viz_path, model_name, concept)

    # Compile results
    analysis = {
        "model_id": model_id,
        "boundaries": boundaries,
        "statistics": statistics,
        "layer_metrics": metrics,
    }

    # Save analysis
    output_path = output_dir / f"caz_analysis_{model_name}.json"
    with output_path.open("w") as f:
        json.dump(analysis, f, indent=2)

    log.info("Analysis saved to %s", output_path)
    log.info("=== CAZ analysis complete ===")

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAZ Boundary Detection")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to caz_extraction.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold percentile for CAZ detection (default: 0.5)",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="concept",
        help="Name of concept being analyzed (e.g., 'credibility', 'negation')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        log.error("Run 'python src/extract_vectors_caz.py' first")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze_caz(input_path, output_dir, args.concept)


if __name__ == "__main__":
    main()
