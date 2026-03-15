# Rosetta Manifold — Results

**Last updated:** March 2026
**Dataset version:** 100 pairs per concept (credibility, negation, sentiment)
**Models:** GPT-2 (124M, 12 layers), GPT-2-XL (1.5B, 48 layers)
**Hardware:** NVIDIA RTX 500 Ada, 4GB VRAM, fp16 forward passes with fp32 metric computation

---

## A Note on Experimental History

The results presented here are the third generation of runs on this pipeline. The history is documented because it illustrates real issues in mechanistic interpretability experimentation at small scale, and because honest science requires it.

**Run 1 (March 10, CPU fp32):** Original runs on 20 negation pairs and 99 sentiment pairs (credibility was already at 100). Results were qualitatively reasonable but the small negation dataset gave unreliable Fisher separation estimates.

**Run 2 (March 14, GPU fp16 — flawed):** All concepts expanded to 100 pairs. First GPU run. A bug in the extraction pipeline caused fp16 activations to overflow the Fisher normalization in gpt2-xl's deep layers (L32+), producing S=0.0 for layers 32–44 and NaN for 45–47. This silently corrupted the credibility gpt2-xl results. The negation and sentiment gpt2-xl results were also affected by the same bug but to a lesser degree since their peaks fall before layer 32.

**Run 3 (March 14, GPU fp16 + fp32 metrics — current):** Fixed by casting activations to float32 before metric computation, while keeping forward passes in fp16 on GPU. Forward passes stay fast; Fisher normalization is accurate. The credibility gpt2-xl results were rerun from scratch. These are the authoritative results.

**The fix** is a single line in `src/extract_vectors_caz.py`:
```python
# Before (wrong — fp16 overflows in variance computation at deep layers)
return torch.cat(all_activations, dim=0).numpy()

# After (correct — fp32 for metrics, fp16 for forward passes)
return torch.cat(all_activations, dim=0).float().numpy()
```

---

## Results — GPT-2 (12 layers)

All three concepts peak at layer 10 (83% depth) in GPT-2. This is a known limitation of shallow models: the CAZ has no room to differentiate — the entire network is operating near the transition zone. GPT-2's 12 layers are insufficient to observe the concept-ordering effect the CAZ framework predicts.

| Concept | Peak layer | Peak S | Peak C | CAZ start | CAZ end | CAZ width |
|---|---|---|---|---|---|---|
| Credibility | L10 / 12 | 0.579 | 0.193 | L0 | L11 | 12 (full model) |
| Negation | L10 / 12 | 0.272 | 0.196 | L0 | L11 | 12 (full model) |
| Sentiment | L10 / 12 | 0.347 | 0.203 | L0 | L11 | 12 (full model) |

**Interpretation:** All three concepts span the full model depth. The CAZ framework's predictions about concept ordering and width require more depth to manifest. These results are consistent with the framework but not confirmatory.

---

## Results — GPT-2-XL (48 layers)

At 48 layers the concept-type ordering emerges clearly. Syntactic concepts assemble earlier; epistemic concepts assemble later and more strongly.

| Concept | Peak layer | Peak S | Peak C | CAZ start | CAZ end | CAZ width | Relative depth |
|---|---|---|---|---|---|---|---|
| Negation | L30 / 48 | 0.257 | 0.155 | L0 | L32 | 33 layers | 63% |
| Sentiment | L31 / 48 | 0.326 | 0.251 | L0 | L32 | 33 layers | 65% |
| Credibility | L46 / 48 | 0.736 | 0.240 | L21 | L47 | 27 layers | 96% |

### Ordering

`negation (63%) < sentiment (65%) ≪ credibility (96%)`

This ordering is consistent with the CAZ framework's concept taxonomy prediction:
- **Syntactic** (negation): explicit, binary, context-local — assembles mid-network
- **Affective** (sentiment): distributed, context-dependent — assembles mid-network, similar depth to negation
- **Epistemic** (credibility): abstract, entangled with general reasoning — assembles very late, near the output

The separation between negation/sentiment (~64%) and credibility (~96%) is larger than predicted. The similarity in depth between negation and sentiment is somewhat surprising — the framework predicted sentiment would be closer to credibility's depth.

### Separation magnitude

Credibility (S=0.736) is dramatically more separable than negation (S=0.257) or sentiment (S=0.326). This confirms the epistemic concept is more strongly geometrically structured in the residual stream — consistent with it being more heavily encoded in the training signal (credibility distinctions appear constantly in natural language; pure syntactic negation is more localized).

### CAZ width

- Credibility: 27-layer CAZ (L21–47) — bounded, with a clear pre-CAZ region
- Negation/sentiment: 33-layer CAZ (L0–32) — unbounded at the start, meaning these concepts show signal from the embedding layer onward

The credibility CAZ is the only one with a distinct pre-CAZ region, consistent with it requiring more processing before it becomes extractable.

---

## Ablation Results

Ablation at the CAZ peak via orthogonal projection. Reported as separation reduction (%) and KL divergence from baseline generation.

| Concept | Model | Separation reduction | KL divergence | Notes |
|---|---|---|---|---|
| Credibility | GPT-2 | 100% | — | Complete signal removal |
| Credibility | GPT-2-XL | 0% | 0.033 | Signal not removed — entangled |
| Negation | GPT-2 | 100% | — | Complete signal removal |
| Negation | GPT-2-XL | 0% | 0.036 | Signal not removed |
| Sentiment | GPT-2 | 100% | — | Complete signal removal |
| Sentiment | GPT-2-XL | 0% | — | Signal not removed |

**Honest note:** The ablation results at gpt2-xl scale show 0% separation reduction, which means the orthogonal projection is not successfully removing the concept direction at this scale. This likely reflects the entanglement problem — at 1.5B parameters the concept direction is distributed across many components and a single-layer projection at the CAZ peak is insufficient. The GPT-2 (124M) ablation works because the simpler model concentrates the direction more sharply.

This is a genuine limitation. The Mid-Stream Ablation Hypothesis (Prediction 1) is not confirmed at proxy scale for gpt2-xl. The ablation approach may need to be multi-layer, or applied across the full CAZ window rather than at a single peak layer.

---

## What the Dataset Size Change Revealed

Expanding negation from 20 to 100 pairs produced a notable shift in the reported peak layer at gpt2-xl: L39 (old) → L30 (new). The separation value also dropped from 0.434 to 0.257.

Both changes reflect better measurement, not a real change in the model:
- **Peak layer shift:** With 20 pairs, the Fisher normalization's denominator (within-class variance) was estimated from too few samples. Small-sample variance is high-variance itself — the peak location was noise-sensitive. With 100 pairs the estimate stabilizes.
- **Separation drop:** Fisher-normalized separation with small samples artificially inflates S because the denominator is underestimated. 100 pairs gives a more honest, lower S. The concept hasn't changed; the measurement has improved.

The credibility results (already at 100 pairs in all runs) showed minimal change in peak layer (L44→L46) and moderate S change (0.772→0.736), consistent with noise rather than a systematic effect.

---

## Architecture Stability (CAZ Prediction 2)

The relative depth ordering at gpt2-xl (negation 63% < sentiment 65% < credibility 96%) is the key result for Prediction 2. However, **the relative depths do not match between GPT-2 and GPT-2-XL**:

| Concept | GPT-2 depth | GPT-2-XL depth |
|---|---|---|
| Negation | 83% | 63% |
| Sentiment | 83% | 65% |
| Credibility | 83% | 96% |

All three concepts peak at the same relative depth in GPT-2 (83%), which is an artifact of the shallow model. At gpt2-xl scale the ordering emerges but the absolute relative depths differ substantially from the GPT-2 values. This means **Prediction 2 is not confirmed by the current data**: the relative depth is not stable across these two architectures.

This is an important honest null result. The prediction may still hold across architectures of *similar* scale (e.g., GPT-2-XL vs. GPT-Neo-1.3B vs. OPT-1.3B), but it does not hold between GPT-2 and GPT-2-XL — which differ not just in depth but in training, architecture details, and capability. Testing Prediction 2 properly requires multiple architectures at the same parameter count, which is the frontier-scale work.

---

## Visualizations

**Comprehensive comparison (March 14, 100-pair datasets, fp32 metric fix):**

![Comprehensive comparison](visualizations/COMPREHENSIVE_CONCEPT_COMPARISON.png)

**Credibility:**

| GPT-2 (124M) | GPT-2-XL (1.5B) |
|---|---|
| ![Credibility GPT-2](visualizations/credibility_gpt2_2026-03-14.png) | ![Credibility GPT-2-XL](visualizations/credibility_gpt2xl_2026-03-14.png) |

**Negation:**

| GPT-2 (124M) | GPT-2-XL (1.5B) |
|---|---|
| ![Negation GPT-2](visualizations/negation_gpt2_2026-03-14.png) | ![Negation GPT-2-XL](visualizations/negation_gpt2xl_2026-03-14.png) |

**Sentiment:**

| GPT-2 (124M) | GPT-2-XL (1.5B) |
|---|---|
| ![Sentiment GPT-2](visualizations/sentiment_gpt2_2026-03-14.png) | ![Sentiment GPT-2-XL](visualizations/sentiment_gpt2xl_2026-03-14.png) |

*Note: March 10 visualizations (smaller datasets, fp16 bug in credibility gpt2-xl) are retained in `visualizations/` for comparison but should not be cited.*

---

## Summary

| Finding | Status |
|---|---|
| Concept-type ordering emerges at gpt2-xl scale (negation < sentiment < credibility) | **Confirmed** |
| Credibility is the most strongly separated concept | **Confirmed** |
| Epistemic concepts have bounded CAZ with distinct pre-CAZ region | **Confirmed** (credibility only) |
| Architecture-stable relative CAZ depth (Prediction 2) | **Not confirmed** at GPT-2 vs GPT-2-XL scale — needs same-scale comparison |
| Mid-Stream Ablation Hypothesis (Prediction 1) | **Confirmed at GPT-2, not confirmed at GPT-2-XL** |
| fp16 extraction produces valid results for deep models | **No** — fp32 metric computation required |
| 20 negation pairs is sufficient | **No** — 100 pairs needed for stable Fisher estimates |
