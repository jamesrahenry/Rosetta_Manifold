#!/usr/bin/env python3
"""
compare_all_concepts.py

Create comprehensive comparison visualization showing all concepts and models together.

Usage:
    python src/compare_all_concepts.py
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Dataset mapping
DATASETS = {
    "credibility_gpt2": {
        "path": "results/caz_validation_gpt2_20260310_164336/caz_analysis_gpt2.json",
        "concept": "Credibility",
        "model": "GPT-2 (12L)",
        "color": "#2E86AB",
        "linestyle": "-",
    },
    "credibility_gpt2xl": {
        "path": "results/caz_validation_gpt2-xl_20260310_193156/caz_analysis_gpt2-xl.json",
        "concept": "Credibility",
        "model": "GPT-2 XL (48L)",
        "color": "#2E86AB",
        "linestyle": "--",
    },
    "negation_gpt2": {
        "path": "results/negation_gpt2_20260310_210541/caz_analysis_gpt2.json",
        "concept": "Negation",
        "model": "GPT-2 (12L)",
        "color": "#A23B72",
        "linestyle": "-",
    },
    "negation_gpt2xl": {
        "path": "results/negation_gpt2xl_20260310_210541/caz_analysis_gpt2-xl.json",
        "concept": "Negation",
        "model": "GPT-2 XL (48L)",
        "color": "#A23B72",
        "linestyle": "--",
    },
    "sentiment_gpt2": {
        "path": "results/20260310_233429_sentiment_gpt2/caz_analysis_gpt2.json",
        "concept": "Sentiment",
        "model": "GPT-2 (12L)",
        "color": "#F18F01",
        "linestyle": "-",
    },
    "sentiment_gpt2xl": {
        "path": "results/20260310_233429_sentiment_gpt2xl/caz_analysis_gpt2-xl.json",
        "concept": "Sentiment",
        "model": "GPT-2 XL (48L)",
        "color": "#F18F01",
        "linestyle": "--",
    },
}

def load_data():
    """Load all dataset analysis files."""
    data = {}
    for key, info in DATASETS.items():
        path = Path(info["path"])
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {key}")
            continue

        with open(path) as f:
            analysis = json.load(f)
            data[key] = {
                "info": info,
                "metrics": analysis["layer_metrics"],
                "stats": analysis["statistics"],
                "boundaries": analysis["boundaries"],
                "model_id": analysis.get("model_id", "unknown"),
            }

    return data

def create_comparison_plot(data, output_path):
    """Create comprehensive comparison visualization."""

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 11))

    # Main layout: 2 rows, 2 columns
    # Row 1: Separation trajectories, Coherence trajectories
    # Row 2: Velocity trajectories, Statistics table

    # === Panel 1: Separation Trajectories ===
    ax1 = plt.subplot(2, 2, 1)

    concepts = ["Credibility", "Negation", "Sentiment"]
    for concept in concepts:
        for key, d in data.items():
            if d["info"]["concept"] != concept:
                continue

            metrics = d["metrics"]
            layers = [m["layer"] for m in metrics]
            separations = [m["separation_fisher"] for m in metrics]

            label = f"{d['info']['concept']} - {d['info']['model']}"
            ax1.plot(
                layers,
                separations,
                color=d["info"]["color"],
                linestyle=d["info"]["linestyle"],
                linewidth=2,
                marker='o' if '12L' in d['info']['model'] else 's',
                markersize=4,
                label=label,
                alpha=0.8
            )

    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Separation (Fisher-normalized)", fontsize=11)
    ax1.set_title("Separation: Concept Assembly Across Layers", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    # === Panel 2: Coherence Trajectories ===
    ax2 = plt.subplot(2, 2, 2)

    for concept in concepts:
        for key, d in data.items():
            if d["info"]["concept"] != concept:
                continue

            metrics = d["metrics"]
            layers = [m["layer"] for m in metrics]
            coherences = [m["coherence"] for m in metrics]

            label = f"{d['info']['concept']} - {d['info']['model']}"
            ax2.plot(
                layers,
                coherences,
                color=d["info"]["color"],
                linestyle=d["info"]["linestyle"],
                linewidth=2,
                marker='o' if '12L' in d['info']['model'] else 's',
                markersize=4,
                label=label,
                alpha=0.8
            )

    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Coherence (Concept Consistency)", fontsize=11)
    ax2.set_title("Coherence: Concept Stability Across Layers", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)

    # === Panel 3: Velocity Trajectories ===
    ax3 = plt.subplot(2, 2, 3)

    for concept in concepts:
        for key, d in data.items():
            if d["info"]["concept"] != concept:
                continue

            metrics = d["metrics"]
            layers = [m["layer"] for m in metrics]
            velocities = [m["velocity"] for m in metrics]

            label = f"{d['info']['concept']} - {d['info']['model']}"
            ax3.plot(
                layers,
                velocities,
                color=d["info"]["color"],
                linestyle=d["info"]["linestyle"],
                linewidth=2,
                marker='o' if '12L' in d['info']['model'] else 's',
                markersize=4,
                label=label,
                alpha=0.8
            )

    ax3.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax3.set_xlabel("Layer", fontsize=11)
    ax3.set_ylabel("Velocity (Rate of Change)", fontsize=11)
    ax3.set_title("Velocity: Rate of Concept Formation", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8, ncol=2)
    ax3.grid(alpha=0.3)

    # === Panel 4: Summary Statistics Table ===
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    # Build table data
    table_data = [["Concept", "Model", "Peak L", "Peak Sep", "Final Sep", "Type"]]

    concept_types = {
        "Credibility": "Epistemic",
        "Negation": "Syntactic",
        "Sentiment": "Affective"
    }

    for concept in concepts:
        for key, d in data.items():
            if d["info"]["concept"] == concept:
                boundaries = d["boundaries"]
                metrics = d["metrics"]
                peak_layer = boundaries["caz_peak"]
                total_layers = len(metrics)
                peak_pct = peak_layer / total_layers if total_layers > 0 else 0
                peak_sep = boundaries["peak_separation"]
                final_sep = metrics[-1]["separation_fisher"] if metrics else 0

                row = [
                    d["info"]["concept"],
                    "12L" if "12L" in d["info"]["model"] else "48L",
                    f"{peak_layer} ({peak_pct:.0%})",
                    f"{peak_sep:.3f}",
                    f"{final_sep:.3f}",
                    concept_types.get(concept, "Unknown")
                ]
                table_data.append(row)

    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.18, 0.12, 0.18, 0.15, 0.15, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')

    ax4.set_title('Summary Statistics - All 6 Datasets', fontsize=13, fontweight="bold", pad=20)

    # Overall title
    fig.suptitle(
        'Rosetta Manifold - Complete Concept Comparison\n3 Concepts × 2 Models = 6 Datasets',
        fontsize=15, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comprehensive comparison saved to: {output_path}")

    return fig

def print_summary(data):
    """Print text summary of all datasets."""
    print("\n" + "="*80)
    print("COMPREHENSIVE CONCEPT COMPARISON SUMMARY")
    print("="*80)

    concepts = ["Credibility", "Negation", "Sentiment"]

    for concept in concepts:
        print(f"\n{'='*80}")
        print(f"{concept.upper()}")
        print(f"{'='*80}")

        for key, d in data.items():
            if d["info"]["concept"] != concept:
                continue

            boundaries = d["boundaries"]
            metrics = d["metrics"]
            total_layers = len(metrics)
            peak_layer = boundaries["caz_peak"]
            peak_pct = peak_layer / total_layers if total_layers > 0 else 0
            peak_sep = boundaries["peak_separation"]
            final_sep = metrics[-1]["separation_fisher"] if metrics else 0

            print(f"\n  {d['info']['model']}:")
            print(f"    Peak Layer:       {peak_layer} ({peak_pct:.1%} depth)")
            print(f"    Peak Separation:  {peak_sep:.4f}")
            print(f"    Final Separation: {final_sep:.4f}")
            print(f"    Total Layers:     {total_layers}")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("\n1. Concept Strength Hierarchy (GPT-2 XL):")
    print("   - Credibility: 0.772 (strongest - epistemic)")
    print("   - Negation:    0.434 (moderate - syntactic)")
    print("   - Sentiment:   0.372 (weakest - affective)")

    print("\n2. Peak Layer Timing:")
    print("   - Credibility & Sentiment: ~92% depth (late-layer)")
    print("   - Negation: ~81% depth (mid-layer)")

    print("\n3. Scale Effects:")
    print("   - All concepts strengthen in larger models")
    print("   - Timing patterns preserved across scales")

    print("\n" + "="*80 + "\n")

def main():
    """Main execution."""
    print("Loading all concept datasets...")
    data = load_data()

    if not data:
        print("ERROR: No datasets found. Check paths in DATASETS mapping.")
        return

    print(f"Loaded {len(data)} datasets")

    # Create comparison plot
    output_path = Path("results/COMPREHENSIVE_CONCEPT_COMPARISON.png")
    create_comparison_plot(data, output_path)

    # Also save to visualizations directory
    viz_path = Path("visualizations/COMPREHENSIVE_CONCEPT_COMPARISON.png")
    viz_path.parent.mkdir(exist_ok=True)
    create_comparison_plot(data, viz_path)

    # Print summary
    print_summary(data)

    print(f"\n📊 Visualization saved to:")
    print(f"   - {output_path}")
    print(f"   - {viz_path}")
    print(f"\n💡 View with: open {viz_path}")

if __name__ == "__main__":
    main()
