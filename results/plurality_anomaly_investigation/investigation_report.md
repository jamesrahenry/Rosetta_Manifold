# Plurality Anomaly Investigation — GPT-2-XL

**Date:** 2026-03-16
**Concept:** plurality (grammatical number agreement)
**Anomaly:** Peak at L47/48 = 98% depth in gpt2-xl (deepest of all 8 concepts)
**Expected:** Syntactic features should peak early (negation peaks at L39 = 81%)
**Status:** RESOLVED — see conclusion

---

## Background

Plurality pairs contrast grammatical number: plural vs. singular constructions of
identical factual content. The concept peaks at L47 in gpt2-xl with a monotonically
increasing separation curve (S rises from 0.206 at L0 to 0.322 at L47), coherence
flat until L45–L47 (spike to 0.195 at L47), and the entire network classified as
"caz" zone — no pre-CAZ region detected.

Two confounds were hypothesized:

- **H1 (credibility confound):** Plural constructions ("researchers have found...")
  systematically convey higher evidential weight than singular ("a researcher has found...").
  If the L47 plurality DoM direction aligns with the credibility direction, the
  "plurality" signal may be measuring credibility geometry.

- **H2 (surface morphology artifact):** Grammatical number is marked from the first
  token ("Researchers" vs. "A researcher"). If the L47 DoM signal is concentrated at
  position 0, the deep peak may reflect a surface-level signal accumulating through
  layers without a discrete assembly event.

---

## Test 1: Direction Correlation (H1 — Credibility Confound)

**Method:** Cosine similarity between plurality DoM vectors and credibility DoM
vectors at every layer, using the stored vectors from `caz_extraction.json`.
Both are unit-normalized Difference-of-Means directions in 1600-dim activation space.

| Comparison | Cosine similarity |
|---|---|
| Plurality L47 vs. Credibility L46 (both at peak) | **-0.136** |
| Plurality L47 vs. credibility at best-matching layer | -0.136 (L46) |
| Same-layer maximum across all 48 layers | 0.139 (L0) |

**Verdict: REJECTED — directions effectively orthogonal (max |cos| = 0.139)**

### Finding

The plurality and credibility DoM directions are orthogonal throughout the network.
The credibility confound hypothesis is not supported. Whatever gpt2-xl is encoding
at L47 for plurality, it is geometrically distinct from the credibility direction.

This is meaningful: the credibility dataset uses "researchers have found" constructions
too, yet the directions diverge. The model separates the evidential-weight content
(credibility) from the grammatical-number surface feature (plurality) into orthogonal
subspaces of the residual stream. This is consistent with superposition — different
features occupy independent directions in high-dimensional space.

---

## Test 2: Token-Position Attribution (H2 — Surface Morphology Artifact)

**Method:** Forward pass on gpt2-xl (20 pairs), extracting full sequence activations
at layer 47 (no pooling). Per-position projection onto the plurality L47 DoM unit
vector. Mean projection difference (positive class − negative class) per position.

Padding tokens were excluded from the analysis (positions ≥ 84 were predominantly
`<|endoftext|>` padding in the 20-pair batch). Analysis restricted to positions
with ≥50% real tokens across the 20 pairs (84 real positions out of 110 padded).

```
First-token delta (pos 0):  7.58
Rest mean delta (real pos): 3.69  (±1.74)
First/rest ratio:           2.05×
```

**Top real positions by attribution magnitude:**

| Rank | Position | Delta | Example tokens (pos / neg) |
|------|----------|-------|----------------------------|
| 1 | 78 | 9.18 | late sentence content words |
| 2 | 0  | 7.58 | `Researchers` / `A` |
| 3 | 62 | 7.30 | plural endings / singular equivalents |
| 4 | 5  | 6.48 | plural nouns / indefinite articles (`a`) |
| 5 | 67 | 6.47 | plural content words |

**Verdict: PARTIAL — first token elevated but signal genuinely distributed**

### Finding

The first token (pos 0: `Researchers` vs `A`) is the single strongest position
(rank 2 overall), but the L47 attribution signal is distributed throughout the
sequence. Tokens at positions 5–6, 17, 62, 67, 78 also show high attribution.

Inspection reveals why: grammatical number is marked *pervasively* throughout the
sentences, not just at position 0:
- Every plural noun: `exercises`, `hearts`, `individuals`, `lifestyles`
- Every plural verb form: `have`, `show`, `strengthen`
- Corresponding singulars: `exercise`, `heart`, `an individual`, `lifestyle`

The attribution profile reflects this: multiple morphologically-marked tokens
throughout the sequence project onto the plurality direction. Position 0 is
the strongest single contributor (surface marker, no ambiguity) but does not
dominate — the signal accumulates from dozens of morphological markers.

**This is the mechanism:** The DoM direction captures distributed morphological
marking across the sequence. The monotonically increasing separation curve in
the layer-wise metrics is consistent with each layer adding small increments as
attention heads aggregate morphological information from multiple positions.

---

## Summary

| Hypothesis | Result | Evidence |
|---|---|---|
| H1: Credibility confound | **REJECTED** | cos(plu@L47, cred@L46) = -0.136; orthogonal throughout |
| H2: Surface morphology artifact | **PARTIALLY CONFIRMED** | First token elevated (2×), signal distributed across multiple morphological markers |

---

## Conclusion

The plurality L47 anomaly is **a genuine but interpretable result**, not a simple
measurement artifact.

**What the L47 direction is actually measuring:** Grammatical number agreement —
the pervasive morphological marking that distinguishes plural from singular across
every noun, verb, and agreement target in the sentence. This is a surface-level
syntactic feature, but it is *distributed* across the full sequence rather than
concentrated at a single token.

**Why it peaks at L47:** GPT-2-XL's attention mechanism aggregates these distributed
morphological signals over many layers, producing an ever-clearer geometric separation
that does not plateau until the final layer. The coherence spike at L45–L47 (from 0.04
to 0.195) indicates the direction crystallizes only at the end — the model is still
actively organizing the plurality representation in its final layers.

**Why OPT/Pythia peak near L0:** Those architectures likely resolve morphological
number agreement in early attention heads (likely position-sensitive heads that read
surface token features). GPT-2's different attention structure delegates it to a
gradual, late-resolving process. This is an architectural difference, not a difference
in which feature is being encoded.

**Why the CAZ framework does not have purchase here:** The CAZ framework models concept
assembly as an event — a discrete transition from pre-representation to a stable
geometric direction. For distributed morphological features, there is no discrete event:
the separation grows continuously as more morphological markers are aggregated. This is
the informative boundary condition the framework needed: it works for semantically
concentrated concepts (credibility, negation) but not for features encoded distributively
across morphological surface markers.

**The DISCONTINUED status is correct** for the current operationalization. A redesigned
study could investigate grammatical number by using it as a confound variable (matched
for morphological distribution) rather than the target — or by using a mechanistic
interpretability approach (attention pattern analysis, circuit analysis) to study the
architectural difference between OPT/Pythia and GPT-2/GPT-Neo directly.

---

## Files

| File | Contents |
|------|----------|
| `direction_correlation.json` | Layer × layer cosine similarity matrix; all per-layer profiles |
| `token_attribution.json` | Per-position projection deltas; raw attribution data |
| `plots/direction_correlation.png` | Three-panel: same-layer profile, cross-peak similarity, heatmap |
| `plots/token_attribution.png` | Per-position delta; first vs. rest comparison |

---

*Generated by: src/investigate_plurality_anomaly.py*
*Supplemental analysis: src/investigate_plurality_anomaly.py + ad-hoc token decode*
