# Spec 3: Functional Verification (Directional Ablation)

## 1. Tooling

- **Primary Library:** `abliterator` (FailSpy/abliterator) — a TransformerLens-based library for directional ablation ("abliteration").
- **Why not heretic-llm:** `heretic-llm` is a niche/unstable package with unclear maintenance status and no verified support for Llama 3 / Mistral / Qwen. `abliterator` is the canonical open-source implementation of the Arditi et al. (2024) abliteration technique, built directly on TransformerLens which already supports all three target architectures.
- **Method:** Orthogonal projection — subtract the component of each residual stream activation along $V_{cred}$ to prevent the model from representing the credibility direction.
- **KL Divergence Monitoring:** Computed against the pre-ablation model's output distribution on a held-out general-purpose benchmark (e.g., a random sample from OpenHermes or MMLU) to measure capability degradation.

## 2. Ablation Loop

```
# Capture baseline logits BEFORE entering the ablation context
baseline_logits = {p: model(p)[0, -1, :].detach().clone() for p in eval_prompts}

for each layer in hook_layers:
    for each component in [resid_pre, resid_mid, resid_post]:
        with DirectionalAblator(model, V_cred, layer, component):
            # KL is computed inside the context where hooks are active
            kl = compute_kl_divergence_from_baseline_logits(
                model, baseline_logits, eval_prompts
            )
        log to Opik: {layer, component, kl, ablation_weight}
```

> **Note:** The baseline logits must be captured *before* entering the ablation context.
> Capturing them inside the context (or comparing the model to itself) would yield KL ≈ 0
> regardless of ablation strength, silently masking any capability degradation.

- **Target KL Divergence:** $< 0.20$ against the baseline model on general-purpose prompts.
- **Sweep Strategy:** Grid search over layers and components (MLP vs. Attention vs. full residual stream). Optuna TPE can be used if the search space is large, but a manual grid is sufficient for the 7B/8B layer range (14–22).

## 3. Success Criteria

1. **Ablation Success:** The ablated model's mean activation along $V_{cred}$ on credible prompts drops to within noise of non-credible prompts (measured by cosine similarity to $V_{cred}$).
2. **Intelligence Retention:** KL divergence on general-purpose prompts remains $< 0.20$. Optionally, verify on a simple coding or logic task.
3. **Audit Trail:** All ablation trials, layer indices, KL scores, and ablation weights are exported to Opik for the final report.

## 4. Key References

- Arditi et al. (2024): [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) — "Refusal in Language Models Is Mediated by a Single Direction." The methodological foundation for this phase.
- FailSpy/abliterator: [github.com/FailSpy/abliterator](https://github.com/FailSpy/abliterator) — Reference implementation.

## 5. Novelty Statement

Prior work (Arditi et al.) demonstrated single-direction mediation for **refusal**. This project applies the identical technique to **credibility** — a semantically distinct, governance-relevant concept — and additionally tests whether the resulting direction is **architecturally transferable** across Llama 3, Mistral, and Qwen (the PRH test). This cross-architecture transferability is the novel contribution.
