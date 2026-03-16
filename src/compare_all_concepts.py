#!/usr/bin/env python3
"""
compare_all_concepts.py

Create comprehensive comparison visualization showing all concepts and models.

Produces:
  visualizations/COMPREHENSIVE_CONCEPT_COMPARISON.png
      Three-panel S/C/v overlay — all 8 concepts × 2 model scales,
      x-axis normalised to relative depth for cross-scale comparison.

  visualizations/COMPREHENSIVE_CONCEPT_SUMMARY_TABLE.png
      Peak layer / S / type table for all 8 concepts × 2 scales.

Data sources (in priority order):
  - March 14 corrected runs for credibility, negation, sentiment
    (fp32 metrics, 100 pairs — canonical)
  - March 15 expanded run for certainty, causation, moral_valence,
    temporal_order, plurality

Usage:
    python src/compare_all_concepts.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
    }
)

BASE = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Concept metadata
# ---------------------------------------------------------------------------

CONCEPTS = {
    "credibility": {"type": "epistemic", "color": "#6A1B9A"},
    "certainty": {"type": "epistemic", "color": "#AB47BC"},
    "sentiment": {"type": "affective", "color": "#2E7D32"},
    "moral_valence": {"type": "affective", "color": "#66BB6A"},
    "causation": {"type": "relational", "color": "#E65100"},
    "temporal_order": {"type": "relational", "color": "#FFA726"},
    "negation": {"type": "syntactic", "color": "#1565C0"},
    "plurality": {"type": "syntactic", "color": "#42A5F5"},
}

TYPE_COLORS = {
    "epistemic": "#6A1B9A",
    "affective": "#2E7D32",
    "relational": "#E65100",
    "syntactic": "#1565C0",
}

# Canonical result directories for each concept × model.
# Mar 14 corrected runs for the original 3; Mar 15 expanded for the rest.
RESULT_DIRS = {
    ("credibility", "gpt2"): "results/gpu_credibility_gpt2_20260314_135612",
    ("credibility", "gpt2-xl"): "results/gpu_credibility_gpt2xl_20260314_135612",
    ("negation", "gpt2"): "results/gpu_negation_gpt2_20260314_135612",
    ("negation", "gpt2-xl"): "results/gpu_negation_gpt2xl_20260314_135612",
    ("sentiment", "gpt2"): "results/gpu_sentiment_gpt2_20260314_135612",
    ("sentiment", "gpt2-xl"): "results/gpu_sentiment_gpt2xl_20260314_135612",
    ("certainty", "gpt2"): "results/expanded_certainty_gpt2_20260315_131312",
    ("certainty", "gpt2-xl"): "results/expanded_certainty_gpt2-xl_20260315_131312",
    ("causation", "gpt2"): "results/expanded_causation_gpt2_20260315_131312",
    ("causation", "gpt2-xl"): "results/expanded_causation_gpt2-xl_20260315_131312",
    ("moral_valence", "gpt2"): "results/expanded_moral_valence_gpt2_20260315_131312",
    (
        "moral_valence",
        "gpt2-xl",
    ): "results/expanded_moral_valence_gpt2-xl_20260315_131312",
    ("temporal_order", "gpt2"): "results/expanded_temporal_order_gpt2_20260315_131312",
    (
        "temporal_order",
        "gpt2-xl",
    ): "results/expanded_temporal_order_gpt2-xl_20260315_131312",
    ("plurality", "gpt2"): "results/expanded_plurality_gpt2_20260315_131312",
    ("plurality", "gpt2-xl"): "results/expanded_plurality_gpt2-xl_20260315_131312",
}

MODEL_LABELS = {
    "gpt2": "GPT-2 (12L)",
    "gpt2-xl": "GPT-2-XL (48L)",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all() -> dict:
    """Load caz_analysis JSON for all concept × model pairs."""
    data = {}
    for (concept, model), rel_dir in RESULT_DIRS.items():
        dir_path = BASE / rel_dir
        analysis_files = list(dir_path.glob("caz_analysis*.json"))
        if not analysis_files:
            print(f"  MISSING: {rel_dir}")
            continue
        with open(analysis_files[0]) as f:
            analysis = json.load(f)
        data[(concept, model)] = analysis
        print(f"  OK  {concept:<16s}  {MODEL_LABELS[model]}")
    return data


# ---------------------------------------------------------------------------
# Figure: multi-panel S/C/v overlay, all 8 concepts × 2 models
# ---------------------------------------------------------------------------


def make_comparison_figure(data: dict, output_path: Path) -> None:
    """
    Three-panel overlay (S, C, V) with all 8 concepts, both model scales.

    x-axis normalised to 0–100% relative depth so 12L and 48L models are
    directly comparable. Solid lines = GPT-2 (12L), dashed = GPT-2-XL (48L).
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    metric_keys = ["separation_fisher", "coherence", "velocity"]
    metric_labels = [
        "S(l) — Separation (Fisher-normalised)",
        "C(l) — Coherence (explained variance)",
        "V(l) — Velocity (dS/dLayer)",
    ]

    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        for concept in CONCEPTS:
            color = CONCEPTS[concept]["color"]
            for model, linestyle in [("gpt2", "-"), ("gpt2-xl", "--")]:
                key = (concept, model)
                if key not in data:
                    continue
                metrics = data[key]["layer_metrics"]
                n = len(metrics)
                depths = [m["layer"] / n * 100 for m in metrics]
                values = [m[mkey] for m in metrics]
                lw = 1.6 if model == "gpt2-xl" else 1.2
                alpha = 0.85 if model == "gpt2-xl" else 0.55
                ax.plot(
                    depths,
                    values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=lw,
                    alpha=alpha,
                )

        if mkey == "velocity":
            ax.axhline(0, color="gray", linewidth=0.8, alpha=0.4)

        ax.set_ylabel(mlabel, fontsize=10)

    axes[-1].set_xlabel("Relative depth (% of model layers)", fontsize=10)

    # Concept colour legend
    concept_handles = [
        Patch(color=CONCEPTS[c]["color"], label=f"{c}  ({CONCEPTS[c]['type']})")
        for c in CONCEPTS
    ]
    # Scale legend
    scale_handles = [
        Line2D(
            [0], [0], color="black", linestyle="-", linewidth=1.4, label="GPT-2 (12L)"
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="GPT-2-XL (48L)",
        ),
    ]
    axes[0].legend(
        handles=concept_handles + scale_handles,
        loc="upper left",
        fontsize=7.5,
        ncol=2,
    )

    fig.suptitle(
        "Rosetta Manifold — CAZ metrics: all 8 concepts × 2 model scales\n"
        "x-axis normalised to relative depth; "
        "solid = GPT-2 (12L), dashed = GPT-2-XL (48L)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: summary statistics table
# ---------------------------------------------------------------------------


def make_summary_table(data: dict, output_path: Path) -> None:
    """Peak layer / S / type table — all 8 concepts × 2 scales."""
    # Order: by type then concept name, matching the depth-ordering figure
    concept_order = [
        "temporal_order",
        "causation",  # relational
        "negation",
        "plurality",  # syntactic
        "sentiment",
        "moral_valence",  # affective
        "certainty",
        "credibility",  # epistemic
    ]
    rows = []
    for concept in concept_order:
        for model in ("gpt2", "gpt2-xl"):
            key = (concept, model)
            if key not in data:
                continue
            b = data[key]["boundaries"]
            metrics = data[key]["layer_metrics"]
            n = len(metrics)
            peak_pct = f"L{b['caz_peak']}/{n}  ({b['caz_peak'] / n:.0%})"
            rows.append(
                [
                    concept,
                    MODEL_LABELS[model],
                    peak_pct,
                    f"{b['peak_separation']:.3f}",
                    CONCEPTS[concept]["type"],
                ]
            )

    fig, ax = plt.subplots(figsize=(13, 0.42 * len(rows) + 1.8))
    ax.axis("off")
    header = ["Concept", "Model", "Peak layer (depth)", "Peak S", "Type"]
    table = ax.table(
        cellText=rows,
        colLabels=header,
        cellLoc="left",
        loc="center",
        colWidths=[0.18, 0.18, 0.22, 0.12, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(len(header)):
        table[(0, j)].set_facecolor("#37474F")
        table[(0, j)].set_text_props(weight="bold", color="white")

    type_bg = {
        "epistemic": "#EDE7F6",
        "affective": "#E8F5E9",
        "relational": "#FFF3E0",
        "syntactic": "#E3F2FD",
    }
    for i, row in enumerate(rows):
        bg = type_bg.get(row[4], "#FFFFFF")
        for j in range(len(header)):
            table[(i + 1, j)].set_facecolor(bg)

    ax.set_title(
        "CAZ peak depths — all 8 concepts × 2 model scales",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading analysis data...")
    data = load_all()
    print(f"\nLoaded {len(data)} concept × model results\n")

    if not data:
        print("ERROR: no data found — check RESULT_DIRS paths.")
        return

    out_dir = BASE / "visualizations"
    out_dir.mkdir(exist_ok=True)

    make_comparison_figure(
        data,
        out_dir / "COMPREHENSIVE_CONCEPT_COMPARISON.png",
    )
    make_summary_table(
        data,
        out_dir / "COMPREHENSIVE_CONCEPT_SUMMARY_TABLE.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
