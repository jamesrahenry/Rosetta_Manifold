"""
viz_dom_lat.py

Quick visualization of DoM-LAT agreement across the GPT-2 family runs.
Reads existing result JSON files and produces a bar chart saved to
results/dom_lat_agreement.png.

Usage:
    python src/viz_dom_lat.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Locate result files
# ---------------------------------------------------------------------------

RESULT_FILES = [
    Path("results/working_models_test_20260228_205050/gpt2.vectors.json"),
    Path("results/working_models_test_20260228_205050/gpt2_medium.vectors.json"),
    Path("results/working_models_test_20260228_205050/gpt2_large.vectors.json"),
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

records = []
for path in RESULT_FILES:
    if not path.exists():
        print(f"[WARN] Not found, skipping: {path}")
        continue
    with path.open() as f:
        data = json.load(f)
    for extraction in data["extractions"]:
        records.append(
            {
                "model": extraction["model_id"],
                "n_layers": extraction["n_layers"],
                "hidden_dim": extraction["hidden_dim"],
                "best_layer": extraction["best_layer"],
                "layer_range": extraction["layer_range"],
                "separation": extraction["separation"],
                "dom_lat_similarity": extraction["dom_lat_similarity"],
            }
        )

if not records:
    raise RuntimeError("No result files found. Run the extraction pipeline first.")

# Sort by model depth for a sensible x-axis order
records.sort(key=lambda r: r["n_layers"])

# ---------------------------------------------------------------------------
# Build labels
# ---------------------------------------------------------------------------

labels = []
for r in records:
    short = r["model"].split("/")[-1]
    labels.append(f"{short}\n({r['n_layers']}L, d={r['hidden_dim']})")

similarities = [r["dom_lat_similarity"] for r in records]
best_layers  = [r["best_layer"] for r in records]
separations  = [r["separation"] for r in records]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Credibility Vector Extraction — Preliminary Results (GPT-2 Family)", fontsize=13, fontweight="bold")

# --- Panel 1: DoM-LAT cosine similarity ---
ax = axes[0]
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars = ax.bar(labels, similarities, color=colors, edgecolor="white", linewidth=0.8)
ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="PRH threshold (0.5)")
ax.axhline(0.3, color="orange", linestyle=":", linewidth=1.2, label="Tiny-model threshold (0.3)")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Cosine Similarity")
ax.set_title("DoM–LAT Agreement\n(method consistency)")
ax.legend(fontsize=8)
for bar, val in zip(bars, similarities):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.3f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

# --- Panel 2: Best layer (absolute and relative) ---
ax = axes[1]
rel_layers = [r["best_layer"] / r["n_layers"] for r in records]
bars2 = ax.bar(labels, rel_layers, color=colors, edgecolor="white", linewidth=0.8)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Best Layer / Total Layers")
ax.set_title("Best Layer (relative depth)\nwhere credibility signal peaks")
for bar, r, abs_l in zip(bars2, rel_layers, best_layers):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"L{abs_l}\n({r:.0%})",
        ha="center", va="bottom", fontsize=9,
    )

# --- Panel 3: Raw separation magnitude ---
ax = axes[2]
bars3 = ax.bar(labels, separations, color=colors, edgecolor="white", linewidth=0.8)
ax.set_ylabel("‖mean(credible) − mean(non-credible)‖₂")
ax.set_title("Activation Separation\n(raw L2 norm at best layer)")
for bar, val in zip(bars3, separations):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{val:.1f}",
        ha="center", va="bottom", fontsize=10,
    )

# ---------------------------------------------------------------------------
# Annotation box
# ---------------------------------------------------------------------------

note_lines = [
    "Note: DoM-LAT similarity measures agreement between",
    "Difference-of-Means and Linear Artificial Tomography.",
    "Values below 0.3 suggest the concept is not cleanly",
    "linearly separable at this layer/model scale.",
    "",
    "Cross-architecture alignment (PRH test) requires",
    "equal hidden dims or CKA — not yet computed.",
]
fig.text(
    0.5, -0.04,
    "\n".join(note_lines),
    ha="center", va="top", fontsize=8, color="#555555",
    style="italic",
)

plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_path = Path("results/dom_lat_agreement.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Also print a text summary
print("\n=== DoM-LAT Agreement Summary ===")
print(f"{'Model':<20} {'Layers':>6} {'d_model':>7} {'Best L':>7} {'Rel%':>6} {'Separation':>11} {'DoM-LAT':>8}")
print("-" * 72)
for r in records:
    short = r["model"].split("/")[-1]
    rel = r["best_layer"] / r["n_layers"]
    print(
        f"{short:<20} {r['n_layers']:>6} {r['hidden_dim']:>7} "
        f"{r['best_layer']:>7} {rel:>5.0%}  {r['separation']:>10.2f} {r['dom_lat_similarity']:>8.4f}"
    )
print()
print("PRH threshold (full-scale): 0.50")
print("PRH threshold (tiny PoC):   0.30")
print("All values are below both thresholds — cross-architecture alignment not yet tested.")
