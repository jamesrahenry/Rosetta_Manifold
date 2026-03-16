"""
analyze_expanded_caz.py

Collate and analyze results from the expanded CAZ run across 8 concepts × 8 models.
Produces:
  - results/expanded_summary.csv       Full results table
  - results/expanded_summary.json      Machine-readable with all metrics
  - visualizations/expanded_*.png      Comparison figures

Usage:
  python src/analyze_expanded_caz.py
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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

base = Path(__file__).parent.parent

# ─── Concept metadata ─────────────────────────────────────────────────────────

CONCEPTS = {
    "negation": {"type": "syntactic", "color": "#1565C0"},
    "plurality": {"type": "syntactic", "color": "#42A5F5"},
    "temporal_order": {"type": "relational", "color": "#E65100"},
    "causation": {"type": "relational", "color": "#FFA726"},
    "sentiment": {"type": "affective", "color": "#2E7D32"},
    "moral_valence": {"type": "affective", "color": "#66BB6A"},
    "certainty": {"type": "epistemic", "color": "#6A1B9A"},
    "credibility": {"type": "epistemic", "color": "#AB47BC"},
}

TYPE_COLORS = {
    "syntactic": "#1565C0",
    "relational": "#E65100",
    "affective": "#2E7D32",
    "epistemic": "#6A1B9A",
}

MODELS = {
    "gpt2": {"family": "gpt2", "params": 124, "layers": 12},
    "gpt2-xl": {"family": "gpt2", "params": 1500, "layers": 48},
    "gpt-neo-125m": {"family": "gpt-neo", "params": 125, "layers": 12},
    "gpt-neo-1.3b": {"family": "gpt-neo", "params": 1300, "layers": 24},
    "pythia-160m": {"family": "pythia", "params": 160, "layers": 12},
    "pythia-410m": {"family": "pythia", "params": 410, "layers": 24},
    "opt-125m": {"family": "opt", "params": 125, "layers": 12},
    "opt-1.3b": {"family": "opt", "params": 1300, "layers": 24},
}

# ─── Collect results ──────────────────────────────────────────────────────────

records = []

for results_dir in sorted(base.glob("results/expanded_*")):
    # Parse concept and model from dir name: expanded_{concept}_{model}_{timestamp}
    parts = results_dir.name.split("_")
    # Find the split point — model keys contain hyphens, timestamp is 8+6 digits
    # Format: expanded_{concept}_{model-key}_{YYYYMMDD}_{HHMMSS}
    # concept may be multi-word (moral_valence, temporal_order)
    # Rebuild: strip "expanded_" prefix and "_{timestamp}_{time}" suffix
    stem = results_dir.name[len("expanded_") :]
    # Remove timestamp suffix (last two _-separated segments that are all digits)
    stem_parts = stem.rsplit("_", 2)
    if len(stem_parts) == 3 and stem_parts[1].isdigit() and stem_parts[2].isdigit():
        stem = stem_parts[0]
    else:
        continue

    # Now stem is "{concept}_{model}" — find the split
    # Model keys: gpt2, gpt2-xl, gpt-neo-125m, gpt-neo-1.3b, pythia-160m,
    #             pythia-410m, opt-125m, opt-1.3b
    model_key = None
    concept = None
    for mk in MODELS:
        suffix = "_" + mk.replace("-", "_") if "-" in mk else "_" + mk
        # Try matching end of stem
        mk_slug = mk.replace("-", "_")
        if stem.endswith("_" + mk_slug) or stem.endswith("_" + mk):
            candidate_concept = stem[: -(len(mk_slug) + 1)].replace("_", "_")
            if (
                candidate_concept in CONCEPTS
                or candidate_concept.replace("_", "-") in CONCEPTS
            ):
                model_key = mk
                concept = candidate_concept
                break

    if not model_key or not concept:
        # Try direct split for simple cases
        for mk in sorted(MODELS.keys(), key=len, reverse=True):
            mk_slug = mk.replace("-", "_")
            if "_" + mk_slug in stem:
                idx = stem.index("_" + mk_slug)
                candidate = stem[:idx]
                if candidate in CONCEPTS:
                    model_key = mk
                    concept = candidate
                    break

    if not model_key or not concept:
        continue

    # Load analysis file
    analysis_files = list(results_dir.glob("caz_analysis*.json"))
    if not analysis_files:
        continue

    with open(analysis_files[0]) as f:
        data = json.load(f)

    b = data["boundaries"]
    n_layers = MODELS[model_key]["layers"]

    records.append(
        {
            "concept": concept,
            "concept_type": CONCEPTS[concept]["type"],
            "model": model_key,
            "model_family": MODELS[model_key]["family"],
            "model_params": MODELS[model_key]["params"],
            "n_layers": n_layers,
            "peak_layer": b["caz_peak"],
            "peak_pct": round(b["caz_peak"] / n_layers * 100, 1),
            "peak_S": round(b["peak_separation"], 4),
            "caz_width": b.get("caz_width", 0),
            "results_dir": str(results_dir),
        }
    )

df = pd.DataFrame(records)
print(
    f"Loaded {len(df)} results across {df['concept'].nunique()} concepts × {df['model'].nunique()} models"
)
print()

# ─── Summary table ────────────────────────────────────────────────────────────

print("=== PEAK LAYER (% depth) AT GPT2-XL SCALE ===")
xl = df[df["model"] == "gpt2-xl"].sort_values("peak_pct")
for _, row in xl.iterrows():
    bar = "█" * int(row["peak_pct"] / 3)
    print(
        f"  {row['concept']:<16s} [{row['concept_type']:<10s}]  L{row['peak_layer']:>2}/{row['n_layers']}  ({row['peak_pct']:>4.1f}%)  S={row['peak_S']:.3f}  {bar}"
    )

print()
print("=== CONCEPT TYPE ORDERING (GPT2-XL, mean depth) ===")
type_means = xl.groupby("concept_type")["peak_pct"].mean().sort_values()
for ctype, mean_pct in type_means.items():
    concepts_in_type = xl[xl["concept_type"] == ctype]["concept"].tolist()
    print(f"  {ctype:<12s}  {mean_pct:.1f}%  {concepts_in_type}")

print()
print("=== ARCHITECTURE CONSISTENCY (mean depth per concept across models) ===")
print(
    f"  {'Concept':<16s} {'Type':<10s} {'12L mean':>8s} {'24L mean':>8s} {'48L mean':>8s} {'Consistent?':>12s}"
)
print("  " + "-" * 65)
for concept in sorted(CONCEPTS.keys()):
    cdf = df[df["concept"] == concept]
    m12 = cdf[cdf["n_layers"] == 12]["peak_pct"].mean()
    m24 = cdf[cdf["n_layers"] == 24]["peak_pct"].mean()
    m48 = (
        cdf[cdf["n_layers"] == 48]["peak_pct"].mean()
        if 48 in cdf["n_layers"].values
        else float("nan")
    )
    spread = max(filter(lambda x: not np.isnan(x), [m12, m24, m48])) - min(
        filter(lambda x: not np.isnan(x), [m12, m24, m48])
    )
    consistent = "yes" if spread < 20 else "no (spread={:.0f}%)".format(spread)
    print(
        f"  {concept:<16s} {CONCEPTS[concept]['type']:<10s} {m12:>7.1f}% {m24:>7.1f}% {m48 if not np.isnan(m48) else '--':>7}  {consistent:>12s}"
    )

# ─── Save ─────────────────────────────────────────────────────────────────────

csv_path = base / "results" / "expanded_summary.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = {
    "n_results": len(df),
    "concepts": list(df["concept"].unique()),
    "models": list(df["model"].unique()),
    "gpt2xl_peaks": {
        row["concept"]: {
            "layer": row["peak_layer"],
            "pct": row["peak_pct"],
            "S": row["peak_S"],
        }
        for _, row in xl.iterrows()
    },
    "type_mean_depths_gpt2xl": type_means.to_dict(),
}
with open(base / "results" / "expanded_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ─── Figure 1: Depth ordering at gpt2-xl scale ───────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

for _, row in xl.sort_values("peak_pct").iterrows():
    color = TYPE_COLORS[row["concept_type"]]
    ax.barh(row["concept"], row["peak_pct"], color=color, alpha=0.85, height=0.6)
    ax.text(
        row["peak_pct"] + 0.5,
        row["concept"],
        f"L{row['peak_layer']}  ({row['peak_pct']:.0f}%)",
        va="center",
        fontsize=9,
    )

# Type boundary lines
for pct, label in [(83, "12-layer floor"), (None, None)]:
    if pct:
        ax.axvline(x=pct, color="black", linestyle=":", alpha=0.3, linewidth=1)

# Legend patches
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=TYPE_COLORS[t], label=t, alpha=0.85)
    for t in ["syntactic", "relational", "affective", "epistemic"]
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

ax.set_xlabel("Peak CAZ depth (% of model layers)")
ax.set_xlim(0, 105)
ax.set_title(
    f"CAZ peak depth by concept — GPT-2-XL (48 layers)\n"
    "Predicted ordering: syntactic < relational < affective < epistemic",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
fig.savefig(
    base / "visualizations" / "expanded_depth_ordering.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved expanded_depth_ordering.png")
plt.close(fig)

# ─── Figure 2: Cross-architecture consistency ─────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(16, 9), sharey=False)
fig.suptitle(
    "CAZ peak depth (%) across architectures — all 8 concepts\n"
    "Consistency across model families tests CAZ Prediction 2",
    fontsize=12,
    fontweight="bold",
)

concept_list = sorted(CONCEPTS.keys(), key=lambda c: CONCEPTS[c]["type"] + c)
family_order = ["gpt2", "gpt-neo", "pythia", "opt"]
family_markers = {"gpt2": "o", "gpt-neo": "s", "pythia": "^", "opt": "D"}
family_colors = {
    "gpt2": "#1565C0",
    "gpt-neo": "#C62828",
    "pythia": "#2E7D32",
    "opt": "#E65100",
}

for ax, concept in zip(axes.flat, concept_list):
    cdf = df[df["concept"] == concept].copy()
    for family in family_order:
        fdf = cdf[cdf["model_family"] == family].sort_values("model_params")
        if fdf.empty:
            continue
        ax.plot(
            fdf["model_params"],
            fdf["peak_pct"],
            marker=family_markers[family],
            color=family_colors[family],
            linewidth=1.5,
            markersize=6,
            alpha=0.85,
            label=family,
        )
    ax.set_title(
        f"{concept}\n({CONCEPTS[concept]['type']})",
        fontsize=9,
        fontweight="bold",
        color=TYPE_COLORS[CONCEPTS[concept]["type"]],
    )
    ax.set_xlabel("Params (M)", fontsize=8)
    ax.set_ylabel("Peak depth (%)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.axhline(y=83, color="black", linestyle=":", alpha=0.2, linewidth=1)
    ax.tick_params(labelsize=8)

axes[0, 0].legend(fontsize=7, loc="lower right")
plt.tight_layout()
fig.savefig(
    base / "visualizations" / "expanded_cross_architecture.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved expanded_cross_architecture.png")
plt.close(fig)

# ─── Figure 3: Separation strength heatmap ───────────────────────────────────

pivot = df.pivot_table(
    index="concept", columns="model", values="peak_S", aggfunc="mean"
)
# Order concepts by type then depth
concept_order = xl.sort_values("peak_pct")["concept"].tolist()
model_order = [
    "gpt2",
    "gpt-neo-125m",
    "pythia-160m",
    "opt-125m",
    "gpt2-xl",
    "gpt-neo-1.3b",
    "pythia-410m",
    "opt-1.3b",
]
pivot = pivot.reindex(
    index=concept_order, columns=[m for m in model_order if m in pivot.columns]
)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"{c}  ({CONCEPTS[c]['type']})" for c in pivot.index], fontsize=9)
plt.colorbar(im, ax=ax, shrink=0.8, label="Peak separation (S)")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if val < 1.0 else "white",
            )
ax.set_title(
    "Peak separation strength (S) — all concepts × all models\n"
    "Higher = stronger geometric signal in residual stream",
    fontsize=11,
    fontweight="bold",
)
# Vertical line between small and large models
ax.axvline(x=3.5, color="white", linewidth=2, alpha=0.6)
ax.text(
    1.5, -0.7, "Small models (12-24L)", ha="center", fontsize=8, transform=ax.transData
)
ax.text(
    5.5, -0.7, "Large models (24-48L)", ha="center", fontsize=8, transform=ax.transData
)
plt.tight_layout()
fig.savefig(
    base / "visualizations" / "expanded_separation_heatmap.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved expanded_separation_heatmap.png")
plt.close(fig)

print(f"\n=== DONE ===")
print(f"Visualizations in: {base}/visualizations/")
print(f"  expanded_depth_ordering.png")
print(f"  expanded_cross_architecture.png")
print(f"  expanded_separation_heatmap.png")
print(f"Summary in: {base}/results/expanded_summary.csv")
