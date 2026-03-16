# Porting Status — TransformerLens → rosetta_tools

**Last updated:** 2026-03-16
**Context:** TransformerLens (TL) breaks on any model outside its supported list
(Qwen2, Phi-2, Pythia, Llama all fail at config conversion). All frontier-scale
work uses `rosetta_tools` (HuggingFace native, no model whitelist).

---

## Status by script

| Script | TL dependency | Priority | Notes |
|:-------|:-------------|:---------|:------|
| `extract_vectors_caz.py` | ✅ **Ported** | — | Uses `rosetta_tools.extraction` + `rosetta_tools.caz` |
| `extract_caz_frontier.py` | ✅ **New** | — | Written natively with rosetta_tools; H100-ready |
| `align_vectors.py` | ❌ **Not ported** | **HIGH** | Needed for Phase 3 Procrustes cross-arch validation. Port before H100 session. See Spec 4. |
| `ablate_caz.py` | ❌ Not ported | Medium | CAZ-windowed ablation experiments. Port before ablation phase. |
| `ablate_vectors.py` | ❌ Not ported | Medium | Full ablation suite. Port before ablation phase. |
| `ablate_vectors_tiny.py` | ❌ Not ported | Low | PoC ablation on small models. May be retired. |
| `extract_vectors_tiny.py` | ❌ Not ported | Low | PoC extraction on small models. May be retired. |
| `extract_vectors.py` | ❌ Not ported | Low | Original single-layer extraction. Superseded by extract_vectors_caz.py. |
| `verify_setup.py` | ❌ Not ported | Low | Setup check script — checks for TL import. Update to check rosetta_tools instead. |

---

## What porting requires

All scripts follow the same pattern. The only TL-specific parts are:

1. **Model loading** — `HookedTransformer.from_pretrained(model_id)` →
   `AutoModel.from_pretrained(model_id, torch_dtype=dtype)` + `AutoTokenizer`

2. **Activation extraction** — `model.run_with_hooks(tokens, fwd_hooks=[...])` →
   `rosetta_tools.extraction.extract_layer_activations()` or direct
   `output_hidden_states=True` forward pass

3. **GPU utils** — `from shared.gpu_utils import ...` →
   `from rosetta_tools.gpu_utils import ...`

4. **Layer count / hidden dim** — `model.cfg.n_layers`, `model.cfg.d_model` →
   `model.config.num_hidden_layers`, `model.config.hidden_size`

The metric computation (DoM, LAT, Fisher separation, Procrustes) is already
pure numpy and does not touch TL in any script — no changes needed there.

---

## Priority for H100 session

The H100 run requires only `extract_caz_frontier.py`, which is already ported.

`align_vectors.py` is needed **after** the frontier extraction run completes —
it consumes the CAZ peak-layer activation data and runs Procrustes alignment
across model pairs. Port this while waiting for DO H100 access, so it's ready
when the extraction data exists. See `Spec 4 -- Procrustes Alignment and
Cross-Architecture Validation.md` for the full methodology.

The ablation scripts are Phase 3 work and not on the critical path for the
initial frontier run.

---

## Pop Goes the Easel

| Script | Status |
|:-------|:-------|
| `pop_goes_the_contrastive.py` | ✅ gpu_utils ported to rosetta_tools; uses native HF GPT2Model (no TL) |
| `pop_goes_the_experiment.py` | ✅ gpu_utils ported to rosetta_tools; uses native HF GPT2Model (no TL) |
| `pop_goes_the_caz.py` | ✅ Analysis only — no model loading, reads saved results |
| All other Easel scripts | ✅ No TL dependency |
