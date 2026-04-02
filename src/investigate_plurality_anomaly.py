"""
investigate_plurality_anomaly.py

Plurality Anomaly Investigation — GPT-2-XL
===========================================

The plurality concept peaks at L47/48 (98% depth) in gpt2-xl. This is
anomalously deep for a syntactic feature (grammatical number agreement).
Two hypotheses:

  H1 — Credibility confound: plurality pairs (\"researchers have found...\")
       are more credible-sounding than singular pairs (\"a researcher has
       found...\"). The L47 separation may reflect the credibility direction
       leaking through rather than genuine plurality signal.

  H2 — Surface morphology artifact: plurality is detectable from the first
       few tokens (\"Researchers/A researcher\", \"studies/study\"). If the
       DoM signal is concentrated at early sequence positions, the deep peak
       may reflect these tokens accumulating representation over many layers
       rather than genuine concept assembly.

Tests
-----
  1. Direction correlation (H1):
       Cosine similarity between plurality DoM at L47 and credibility DoM
       at L46 (credibility peak). If r > 0.8, plurality is largely measuring
       credibility. Both vectors are already in caz_extraction.json.

  2. Per-position attribution (H2):
       Run a partial forward pass with token-position pooling (rather than
       last-token). Compute separation when projecting activations onto the
       plurality L47 DoM direction, at each token position separately.
       Positions with high projected separation are driving the signal.
       Compare first-token vs. mean-of-rest to isolate morphological confound.

  3. Direction comparison across layers (context):
       Compute cosine similarity between plurality DoM at L47 and credibility
       DoM at every layer, to see whether the confound builds gradually or
       emerges late.

Usage
-----
    python src/investigate_plurality_anomaly.py [--no-gpu]

Outputs
-------
    results/plurality_anomaly_investigation/
        investigation_report.md        Human-readable findings
        direction_correlation.json      Full layer × layer cosine matrix
        token_attribution.json          Per-position projected separation
        plots/direction_correlation.png
        plots/token_attribution.png
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent
RESULTS = BASE / "results"
PLU_DIR = RESULTS / "expanded_plurality_gpt2-xl_20260315_131312"
CRED_DIR = RESULTS / "expanded_credibility_gpt2-xl_20260315_131312"
PLU_DATA = BASE / "data" / "plurality_pairs.jsonl"
OUT_DIR = RESULTS / "plurality_anomaly_investigation"

# ─── Matplotlib style ─────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
    }
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def load_dom_vectors(extraction_path: Path) -> list[np.ndarray]:
    """Load DoM vectors for every layer from a caz_extraction.json file.

    Returns a list of float64 unit vectors, one per transformer layer
    (0-indexed, embedding layer already dropped per pipeline convention).
    """
    with open(extraction_path) as f:
        d = json.load(f)
    metrics = d["layer_data"]["metrics"]
    return [np.array(m["dom_vector"], dtype=np.float64) for m in metrics]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Handles zero-norm gracefully."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_pairs(path: Path) -> tuple[list[str], list[str]]:
    """Load contrastive JSONL — returns (pos_texts, neg_texts)."""
    pos, neg = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["label"] == 1:
                pos.append(rec["text"])
            else:
                neg.append(rec["text"])
    return pos, neg


# ─── Test 1: Direction correlation ────────────────────────────────────────────


def test_direction_correlation(
    plu_vecs: list[np.ndarray], cred_vecs: list[np.ndarray]
) -> dict:
    """
    Compute cosine similarity between:
      - Plurality DoM at every layer L, vs. credibility DoM at every layer L
      - Plurality DoM at L47 (peak) vs. credibility DoM at L46 (peak)
      - Full L × L matrix for plotting

    Interpretation guide:
      |cos| > 0.8  — highly parallel; plurality largely measuring credibility
      |cos| 0.5-0.8 — substantial overlap
      |cos| < 0.3  — effectively orthogonal; independent directions
    """
    n_plu = len(plu_vecs)
    n_cred = len(cred_vecs)

    # Same-layer similarities (diagonal)
    same_layer_cos = [
        cosine_similarity(plu_vecs[l], cred_vecs[l]) for l in range(min(n_plu, n_cred))
    ]

    # Plurality L47 vs. all credibility layers
    plu_peak = plu_vecs[47]
    plu_vs_cred_layers = [
        cosine_similarity(plu_peak, cred_vecs[l]) for l in range(n_cred)
    ]

    # Credibility L46 vs. all plurality layers
    cred_peak = cred_vecs[46]
    cred_vs_plu_layers = [
        cosine_similarity(cred_peak, plu_vecs[l]) for l in range(n_plu)
    ]

    # Key scalar: plurality L47 ↔ credibility L46
    key_similarity = cosine_similarity(plu_peak, cred_peak)

    # Full L × L matrix (for heatmap)
    mat = np.zeros((n_plu, n_cred))
    for i in range(n_plu):
        for j in range(n_cred):
            mat[i, j] = cosine_similarity(plu_vecs[i], cred_vecs[j])

    return {
        "same_layer_cos": same_layer_cos,
        "plu_L47_vs_cred_all": plu_vs_cred_layers,
        "cred_L46_vs_plu_all": cred_vs_plu_layers,
        "key_similarity_plu47_vs_cred46": key_similarity,
        "full_matrix": mat.tolist(),
        "n_plu_layers": n_plu,
        "n_cred_layers": n_cred,
    }


def plot_direction_correlation(corr: dict, out_dir: Path) -> None:
    """Three-panel figure: same-layer profile, cross-peak similarity, heatmap."""
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1: Same-layer cosine similarity profile
    ax1 = fig.add_subplot(gs[0])
    sl = corr["same_layer_cos"]
    layers = list(range(len(sl)))
    ax1.plot(layers, sl, color="#1565C0", linewidth=1.5, label="cos(plu_L, cred_L)")
    ax1.axhline(
        0.8, color="red", linestyle="--", alpha=0.5, linewidth=1, label="0.8 threshold"
    )
    ax1.axhline(
        0.3,
        color="orange",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        label="0.3 threshold",
    )
    ax1.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax1.axvline(
        47,
        color="#1565C0",
        linestyle=":",
        alpha=0.6,
        linewidth=1,
        label="plu peak (L47)",
    )
    ax1.axvline(
        46,
        color="#AB47BC",
        linestyle=":",
        alpha=0.6,
        linewidth=1,
        label="cred peak (L46)",
    )
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Cosine similarity")
    ax1.set_title("Plurality vs. Credibility DoM\nSame-layer cosine similarity")
    ax1.set_ylim(-0.6, 1.05)
    ax1.legend(fontsize=7)

    # Panel 2: Plurality L47 vs. all credibility layers
    ax2 = fig.add_subplot(gs[1])
    p47_vs_cred = corr["plu_L47_vs_cred_all"]
    ax2.plot(range(len(p47_vs_cred)), p47_vs_cred, color="#AB47BC", linewidth=1.5)
    ax2.axhline(0.8, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(0.3, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax2.axvline(
        46,
        color="#AB47BC",
        linestyle=":",
        alpha=0.6,
        linewidth=1,
        label="cred peak (L46)",
    )
    key_val = corr["key_similarity_plu47_vs_cred46"]
    ax2.scatter(
        [46],
        [key_val],
        color="red",
        zorder=5,
        s=60,
        label=f"L47 vs L46: r={key_val:.3f}",
    )
    ax2.set_xlabel("Credibility layer")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title(
        f"Plurality L47 DoM vs.\nCredibility DoM at every layer\n"
        f"(key: plu@L47 vs cred@L46 = {key_val:.3f})"
    )
    ax2.set_ylim(-0.6, 1.05)
    ax2.legend(fontsize=7)

    # Panel 3: Heatmap
    ax3 = fig.add_subplot(gs[2])
    mat = np.array(corr["full_matrix"])
    im = ax3.imshow(
        mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
    )
    ax3.set_xlabel("Credibility layer")
    ax3.set_ylabel("Plurality layer")
    ax3.set_title("Full cosine similarity matrix\nPlurality DoM ↔ Credibility DoM")
    ax3.axhline(46.5, color="white", linewidth=0.8, alpha=0.5)  # Near plu peak
    ax3.axvline(45.5, color="white", linewidth=0.8, alpha=0.5)  # Near cred peak
    plt.colorbar(im, ax=ax3, shrink=0.9, label="cos similarity")

    fig.suptitle(
        "Test 1: Direction Correlation — Plurality vs. Credibility",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = out_dir / "plots" / "direction_correlation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ─── Test 2: Token-position attribution ───────────────────────────────────────


def test_token_attribution(
    plu_vecs: list[np.ndarray],
    pos_texts: list[str],
    neg_texts: list[str],
    skip_gpu: bool = False,
) -> dict:
    """
    Per-position attribution: project activations onto the plurality L47 DoM
    direction, position by position, to identify which token positions are
    driving the L47 separation signal.

    Strategy:
      - Load gpt2-xl, run forward with output_hidden_states=True (no pooling)
      - At layer 47, extract the full sequence of hidden states [n_texts, seq, dim]
      - Project each position onto the plu L47 DoM unit vector
      - Compute projected-space mean difference per position
      - Compare: first token (morphological marker) vs. subsequent tokens

    Approximation: run on a sample of pairs (first 20) to stay within VRAM.
    If GPU unavailable or skip_gpu=True, run on CPU (slower but equivalent).
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return {"error": "torch/transformers not available; skipping attribution test"}

    device = "cpu"
    if not skip_gpu and torch.cuda.is_available():
        device = "cuda"

    print(f"  Loading gpt2-xl on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        "gpt2-xl",
        dtype=torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    # Use first 20 pairs (40 texts) — enough for stable estimates, fast enough
    n_sample = min(20, len(pos_texts))
    pos_sample = pos_texts[:n_sample]
    neg_sample = neg_texts[:n_sample]

    plu_direction = plu_vecs[47]  # Unit vector — L47 DoM direction
    plu_direction = plu_direction / np.linalg.norm(plu_direction)

    def get_token_projections(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          projections: [n_texts, seq_len] — per-position projection onto plu DoM
          token_ids: [seq_len] — tokenization of representative text
        """
        encoding = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Layer 47 = index 48 in hidden_states (embedding is index 0)
        # hidden_states[48] has shape [n_texts, seq_len, hidden_dim]
        hs_L47 = outputs.hidden_states[48].float().cpu().numpy()  # [n, seq, 1600]

        # Project each position onto plu DoM direction
        # proj[i, t] = dot(hs_L47[i, t], plu_direction)
        proj = hs_L47 @ plu_direction  # [n_texts, seq_len]

        # Mask out padding
        mask = attention_mask.cpu().numpy().astype(bool)
        proj_masked = proj * mask  # zero out padding positions

        token_ids = input_ids[0].cpu().numpy()
        return proj_masked, token_ids, mask

    print(f"  Running forward passes for {n_sample} positive texts...")
    pos_proj, pos_tok_ids, pos_mask = get_token_projections(pos_sample)

    print(f"  Running forward passes for {n_sample} negative texts...")
    neg_proj, neg_tok_ids, neg_mask = get_token_projections(neg_sample)

    # Per-position mean projection difference (pos - neg) across texts
    # Use the shorter of the two sequence lengths
    seq_len = min(pos_proj.shape[1], neg_proj.shape[1])
    pos_proj = pos_proj[:, :seq_len]
    neg_proj = neg_proj[:, :seq_len]
    pos_mask_tr = pos_mask[:, :seq_len]
    neg_mask_tr = neg_mask[:, :seq_len]

    # Mean projected value per position, across the n_sample texts
    # Only average over non-padding positions
    def mean_proj_per_pos(proj, mask):
        count = mask.sum(axis=0).clip(min=1)
        return (proj * mask).sum(axis=0) / count

    pos_mean_proj = mean_proj_per_pos(pos_proj, pos_mask_tr)
    neg_mean_proj = mean_proj_per_pos(neg_proj, neg_mask_tr)
    delta = pos_mean_proj - neg_mean_proj  # [seq_len]

    # Decode first few tokens of representative positive text
    first_tokens = [tokenizer.decode([t]) for t in pos_tok_ids[:20]]

    # Summary stats: first token vs. rest
    first_tok_delta = float(delta[0]) if len(delta) > 0 else 0.0
    rest_delta_mean = float(delta[1:].mean()) if len(delta) > 1 else 0.0
    rest_delta_std = float(delta[1:].std()) if len(delta) > 1 else 0.0
    abs_delta = np.abs(delta)
    top_positions = np.argsort(abs_delta)[::-1][:10].tolist()

    # Clean up GPU memory
    del model
    if device == "cuda":
        try:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "n_sample": n_sample,
        "seq_len": seq_len,
        "device": device,
        "delta_per_position": delta.tolist(),
        "pos_mean_proj": pos_mean_proj.tolist(),
        "neg_mean_proj": neg_mean_proj.tolist(),
        "first_token_delta": first_tok_delta,
        "rest_mean_delta": rest_delta_mean,
        "rest_std_delta": rest_delta_std,
        "top_10_positions_by_abs_delta": top_positions,
        "first_20_tokens_of_representative_pos": first_tokens,
    }


def plot_token_attribution(attr: dict, out_dir: Path) -> None:
    """Two-panel: delta profile and first vs. rest comparison."""
    if "error" in attr:
        print(f"  Skipping attribution plot: {attr['error']}")
        return

    delta = np.array(attr["delta_per_position"])
    seq_len = len(delta)
    positions = list(range(seq_len))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Delta per position
    colors = ["#C62828" if i == 0 else "#1565C0" for i in range(seq_len)]
    ax1.bar(positions, delta, color=colors, alpha=0.7, width=1.0)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(
        attr["rest_mean_delta"],
        color="#1565C0",
        linestyle="--",
        alpha=0.6,
        linewidth=1,
        label=f"mean(rest) = {attr['rest_mean_delta']:.3f}",
    )
    ax1.scatter(
        [0],
        [attr["first_token_delta"]],
        color="#C62828",
        zorder=5,
        s=60,
        label=f"pos 0 = {attr['first_token_delta']:.3f}",
    )
    ax1.set_xlabel("Token position")
    ax1.set_ylabel("Mean(pos proj) − Mean(neg proj)")
    ax1.set_title(
        "Per-position attribution onto plurality L47 DoM\n"
        f"(red = position 0, n={attr['n_sample']} pairs, device={attr['device']})"
    )
    ax1.legend(fontsize=8)

    # Panel 2: First token vs. rest
    first_delta = attr["first_token_delta"]
    rest_mean = attr["rest_mean_delta"]
    rest_std = attr["rest_std_delta"]

    ax2.bar(
        ["Position 0\n(first token)", "Positions 1+\n(mean)"],
        [first_delta, rest_mean],
        color=["#C62828", "#1565C0"],
        alpha=0.8,
        width=0.4,
    )
    ax2.errorbar(
        [1],
        [rest_mean],
        yerr=[rest_std],
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.5,
        label=f"±1 std = {rest_std:.3f}",
    )
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Projected separation (pos − neg)")
    ax2.set_title("First token vs. rest\nConcentration of L47 signal")
    ax2.legend(fontsize=8)

    # Annotation
    ratio = abs(first_delta) / (abs(rest_mean) + 1e-10)
    ax2.text(
        0.98,
        0.98,
        f"first/rest ratio: {ratio:.1f}×",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="lightyellow",
            edgecolor="gray",
            alpha=0.8,
        ),
    )

    fig.suptitle(
        "Test 2: Token-Position Attribution — Plurality L47 DoM Direction",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = out_dir / "plots" / "token_attribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ─── Report generation ────────────────────────────────────────────────────────


def generate_report(corr: dict, attr: dict, out_dir: Path) -> str:
    """Write a markdown investigation report and return the path."""

    key_cos = corr["key_similarity_plu47_vs_cred46"]
    same_layer_peak = max(enumerate(corr["same_layer_cos"]), key=lambda x: abs(x[1]))
    plu47_cred_peak_layer = int(np.argmax(np.abs(corr["plu_L47_vs_cred_all"])))
    plu47_cred_peak_val = corr["plu_L47_vs_cred_all"][plu47_cred_peak_layer]

    # Interpret direction correlation
    if abs(key_cos) > 0.8:
        h1_verdict = "CONFIRMED — directions are nearly parallel (|r| > 0.8)"
        h1_strength = "strong"
    elif abs(key_cos) > 0.5:
        h1_verdict = "PARTIAL — substantial overlap (0.5 < |r| < 0.8)"
        h1_strength = "moderate"
    elif abs(key_cos) > 0.3:
        h1_verdict = "WEAK — some overlap (0.3 < |r| < 0.5)"
        h1_strength = "weak"
    else:
        h1_verdict = "REJECTED — directions effectively orthogonal (|r| < 0.3)"
        h1_strength = "none"

    # Interpret token attribution
    if "error" not in attr:
        first_delta = attr["first_token_delta"]
        rest_mean = attr["rest_mean_delta"]
        rest_std = attr["rest_std_delta"]
        ratio = abs(first_delta) / (abs(rest_mean) + 1e-10)
        if ratio > 3.0:
            h2_verdict = "CONFIRMED — L47 signal dominated by first token (ratio > 3×)"
        elif ratio > 1.5:
            h2_verdict = "PARTIAL — first token elevated but signal distributed"
        else:
            h2_verdict = "REJECTED — signal distributed across positions, no first-token dominance"
        h2_data = (
            f"First-token delta: {first_delta:.4f}\n"
            f"Rest mean delta: {rest_mean:.4f} (±{rest_std:.4f})\n"
            f"First/rest ratio: {ratio:.2f}×"
        )
        top_pos = attr["top_10_positions_by_abs_delta"]
        top_tokens = attr.get("first_20_tokens_of_representative_pos", [])
    else:
        h2_verdict = f"SKIPPED — {attr['error']}"
        h2_data = "N/A"
        top_pos = []
        top_tokens = []

    # Same-layer peak analysis
    same_layer_max_layer, same_layer_max_val = same_layer_peak

    report = f"""# Plurality Anomaly Investigation — GPT-2-XL

**Date:** 2026-03-16
**Concept:** plurality (grammatical number agreement)
**Anomaly:** Peak at L47/48 = 98% depth in gpt2-xl (deepest of all 8 concepts)
**Expected:** Syntactic features should peak early (negation peaks at L39 = 81%)

---

## Background

Plurality pairs contrast grammatical number (plural vs. singular constructions of
identical factual content). The concept peaks at L47 in gpt2-xl with a monotonically
increasing separation curve (S rises from 0.206 at L0 to 0.322 at L47). Two confounds
were hypothesized:

- **H1 (credibility confound):** Plural constructions ("researchers have found...")
  systematically convey higher evidential weight than singular ("a researcher has found...").
  If the L47 plurality DoM direction is aligned with the credibility direction, the
  "plurality" signal may be largely measuring credibility.

- **H2 (surface morphology artifact):** Plural/singular is detectable from the first
  token ("Researchers" vs. "A researcher"). If the L47 DoM signal is concentrated at
  position 0, it may reflect surface-level morphological information accumulating over
  many layers rather than syntactic concept assembly.

---

## Test 1: Direction Correlation (H1 — Credibility Confound)

**Metric:** Cosine similarity between plurality DoM direction and credibility DoM direction.

| Comparison | Cosine similarity |
|---|---|
| Plurality L47 vs. Credibility L46 (both at peak) | **{key_cos:.4f}** |
| Plurality L47 vs. Credibility at best-matching layer (L{plu47_cred_peak_layer}) | {plu47_cred_peak_val:.4f} |
| Same-layer peak: L{same_layer_max_layer} | {same_layer_max_val:.4f} |

**Verdict: {h1_verdict}**

### Interpretation

"""

    if h1_strength == "strong":
        report += """The plurality and credibility DoM directions are nearly parallel at their
respective peaks. This strongly suggests the L47 plurality signal is predominantly
measuring the credibility confound rather than grammatical number per se. The plural
construction ("researchers have found...") reads as more evidentially grounded than
the singular ("a researcher has found..."), and gpt2-xl appears to represent this
epistemic distinction in the same geometric direction as genuine credibility signals.

This would explain the late-layer peak: credibility is an epistemic concept that
legitimately assembles late (L46, 96% depth), and plurality has been captured by
that same representational structure."""
    elif h1_strength == "moderate":
        report += """There is substantial but not dominant overlap between the plurality and credibility
directions. The credibility confound is real but only partially explains the L47 peak.
A redesigned dataset that controls for evidential weight (e.g., using only scientific
sources for both conditions) could separate the confounded signals."""
    elif h1_strength == "weak":
        report += """There is some overlap between the directions but the plurality L47 peak cannot be
explained primarily by the credibility confound. Other factors (H2, or genuine
architectural behavior) dominate the signal."""
    else:
        report += """The plurality and credibility DoM directions are effectively orthogonal. The
credibility confound hypothesis is not supported by the geometric evidence. Whatever
the plurality L47 peak is measuring, it is geometrically distinct from credibility."""

    report += f"""

---

## Test 2: Token-Position Attribution (H2 — Surface Morphology Artifact)

**Metric:** Per-position projection onto the plurality L47 DoM direction.
Compares separation at position 0 (first token) vs. positions 1+ (subsequent tokens).

```
{h2_data}
```
"""

    if top_tokens and top_pos:
        report += f"""
**Top 10 positions by attribution magnitude:** {top_pos}
**First 20 tokens of representative positive text:** {top_tokens}

"""

    report += f"""**Verdict: {h2_verdict}**

### Interpretation

"""

    if "CONFIRMED" in h2_verdict:
        report += """The L47 separation is driven primarily by the first-token position. This is
consistent with the surface morphology hypothesis: "Researchers" (plural) vs.
"A researcher" (singular) is detectable from token 0, and this morphological signal
accumulates through all 48 layers, peaking only at L47 because the residual stream
continues to refine the representation without ever "completing" assembly in the
CAZ framework sense.

This is the measurement artifact interpretation: the DoM direction at L47 picks up
on a surface distributional difference that gpt2-xl processes through its attention
mechanism gradually, with no discrete assembly event — hence the monotonically
increasing separation curve and absence of a coherence peak until L47."""
    elif "PARTIAL" in h2_verdict:
        report += """The first token contributes disproportionately to the L47 signal but does not
fully dominate. The signal is partially explained by surface morphology but also has
distributed contributions from later tokens."""
    elif "REJECTED" in h2_verdict:
        report += """The L47 signal is distributed across token positions rather than concentrated at
position 0. The surface morphology hypothesis is not supported. Whatever drives the
late peak is a property of the full sequence representation, not just the
morphological marker at position 0."""
    else:
        report += "Attribution test was skipped. No conclusion available.\n"

    report += f"""
---

## Summary

| Hypothesis | Result |
|---|---|
| H1: Credibility confound | {h1_verdict.split(" — ")[0]} |
| H2: Surface morphology artifact | {h2_verdict.split(" — ")[0]} |

### Overall interpretation

"""

    if h1_strength in ("strong", "moderate") and "CONFIRMED" in h2_verdict:
        report += """Both confounds are active. The plurality L47 peak is a compound artifact:
(1) plural constructions sound more credible, capturing the credibility geometric direction;
(2) the first-token morphological difference drives the signal distribution. Together,
these explain the anomalous depth without invoking genuine syntactic concept assembly.

**Recommendation:** The DISCONTINUED status is correct and well-supported. For any
future plurality-like study, the dataset should: (a) control for evidential weight
by using non-attributive sentences, and (b) introduce grammatical number mid-sentence
rather than at position 0, to avoid surface attribution from the first token."""
    elif h1_strength == "strong":
        report += """The credibility confound is the dominant explanation. The plurality L47 peak
is largely the credibility epistemic signal leaking through from the dataset construction.
The DISCONTINUED status is correct."""
    elif "CONFIRMED" in h2_verdict:
        report += """The surface morphology hypothesis is the dominant explanation. The first-token
distributional difference drives the monotonically increasing separation curve.
The DISCONTINUED status is correct — plurality as operationalized does not test
concept assembly in the CAZ sense."""
    elif h1_strength == "none" and "REJECTED" in h2_verdict:
        report += """Neither confound hypothesis is supported by the geometric evidence. The L47 peak
is genuine but unexplained by the current analysis. This warrants further investigation:
possibly the GPT-2 family has an architectural peculiarity in how it handles morphosyntax,
or the bimodal cross-architecture pattern (OPT/Pythia at L0-2, GPT-2/GPT-Neo at 92-98%)
reflects a genuine mechanistic difference worth studying at larger scale.

**Recommendation:** Revisit with mechanistic interpretability tools — attention pattern
analysis, logit lens, and direct circuit analysis — to understand the architectural
mechanism. May be worth a dedicated study separate from the CAZ framework."""
    else:
        report += """The results are mixed. The plurality anomaly is partially explained by the
identified confounds but a full account requires further investigation."""

    report += """

---

*Generated by: src/investigate_plurality_anomaly.py*
"""

    out_path = out_dir / "investigation_report.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"  Saved {out_path}")
    return str(out_path)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Investigate the plurality anomaly in gpt2-xl"
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Skip GPU, run attribution on CPU"
    )
    parser.add_argument(
        "--skip-attribution",
        action="store_true",
        help="Skip token attribution test (direction correlation only)",
    )
    args = parser.parse_args()

    # Verify inputs exist
    for path, name in [
        (PLU_DIR / "caz_extraction.json", "plurality extraction"),
        (CRED_DIR / "caz_extraction.json", "credibility extraction"),
        (PLU_DATA, "plurality dataset"),
    ]:
        if not path.exists():
            print(f"ERROR: Missing {name}: {path}", file=sys.stderr)
            sys.exit(1)

    # Create output dirs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(exist_ok=True)

    print("=" * 60)
    print("Plurality Anomaly Investigation — GPT-2-XL")
    print("=" * 60)

    # Load DoM vectors
    print("\n[1/4] Loading DoM vectors...")
    plu_vecs = load_dom_vectors(PLU_DIR / "caz_extraction.json")
    cred_vecs = load_dom_vectors(CRED_DIR / "caz_extraction.json")
    print(f"  Plurality: {len(plu_vecs)} layers × {len(plu_vecs[0])}-dim")
    print(f"  Credibility: {len(cred_vecs)} layers × {len(cred_vecs[0])}-dim")

    # Test 1: Direction correlation
    print("\n[2/4] Test 1: Direction correlation...")
    corr = test_direction_correlation(plu_vecs, cred_vecs)
    key_cos = corr["key_similarity_plu47_vs_cred46"]
    print(f"  Plurality L47 vs. Credibility L46: cos = {key_cos:.4f}")
    print(
        f"  Same-layer peak: {max(enumerate(corr['same_layer_cos']), key=lambda x: abs(x[1]))}"
    )

    # Save correlation data
    corr_save = {k: v for k, v in corr.items() if k != "full_matrix"}
    corr_save["full_matrix_shape"] = [corr["n_plu_layers"], corr["n_cred_layers"]]
    with open(OUT_DIR / "direction_correlation.json", "w") as f:
        json.dump({**corr_save, "full_matrix": corr["full_matrix"]}, f, indent=2)

    plot_direction_correlation(corr, OUT_DIR)

    # Test 2: Token attribution
    if args.skip_attribution:
        print("\n[3/4] Test 2: Token attribution — SKIPPED (--skip-attribution)")
        attr = {"error": "skipped by user flag"}
    else:
        print("\n[3/4] Test 2: Token-position attribution...")
        pos_texts, neg_texts = load_pairs(PLU_DATA)
        print(f"  Loaded {len(pos_texts)} positive, {len(neg_texts)} negative texts")
        attr = test_token_attribution(
            plu_vecs, pos_texts, neg_texts, skip_gpu=args.no_gpu
        )
        if "error" not in attr:
            print(f"  First-token delta: {attr['first_token_delta']:.4f}")
            print(f"  Rest mean delta:   {attr['rest_mean_delta']:.4f}")
            ratio = abs(attr["first_token_delta"]) / (
                abs(attr["rest_mean_delta"]) + 1e-10
            )
            print(f"  First/rest ratio:  {ratio:.2f}×")
        else:
            print(f"  Error: {attr['error']}")

        with open(OUT_DIR / "token_attribution.json", "w") as f:
            json.dump(attr, f, indent=2)
        plot_token_attribution(attr, OUT_DIR)

    # Report
    print("\n[4/4] Generating report...")
    report_path = generate_report(corr, attr, OUT_DIR)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  H1 (credibility confound): cos(plu@L47, cred@L46) = {key_cos:.4f}")
    if abs(key_cos) > 0.8:
        print("    → CONFIRMED (|r| > 0.8: directions nearly parallel)")
    elif abs(key_cos) > 0.5:
        print("    → PARTIAL (0.5 < |r| < 0.8: substantial overlap)")
    elif abs(key_cos) > 0.3:
        print("    → WEAK (0.3 < |r| < 0.5)")
    else:
        print("    → REJECTED (|r| < 0.3: effectively orthogonal)")

    if "error" not in attr:
        ratio = abs(attr["first_token_delta"]) / (abs(attr["rest_mean_delta"]) + 1e-10)
        print(f"  H2 (surface morphology): first/rest ratio = {ratio:.2f}×")
        if ratio > 3.0:
            print("    → CONFIRMED (first token dominates)")
        elif ratio > 1.5:
            print("    → PARTIAL (first token elevated)")
        else:
            print("    → REJECTED (signal distributed)")
    else:
        print(f"  H2 (surface morphology): {attr['error']}")

    print(f"\n  Report: {report_path}")
    print(f"  Outputs: {OUT_DIR}/")


if __name__ == "__main__":
    main()
