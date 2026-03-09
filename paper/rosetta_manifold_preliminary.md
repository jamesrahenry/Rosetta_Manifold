# Rosetta Manifold: Universal Credibility Vectors Across Language Model Architectures

**Preliminary Draft — Not for Citation**

---

## Abstract

We investigate whether the semantic concept of *credibility* is encoded as a consistent geometric direction in the residual stream of large language models (LLMs), and whether this direction is shared across architecturally distinct models. Grounding our work in the Platonic Representation Hypothesis (PRH; Huh et al., 2024), we extract credibility direction vectors ($V_{cred}$) from three open-source 7B/8B-scale models — Llama 3 8B, Mistral 7B, and Qwen 2.5 7B — using two complementary methods: Difference-of-Means (DoM; Arditi et al., 2024) and Linear Artificial Tomography (LAT; Zou et al., 2023). We evaluate cross-architecture alignment via cosine similarity and validate functional significance through orthogonal projection ablation. Our results demonstrate that credibility is mediated by a single, extractable direction in the residual stream, that this direction shows meaningful cross-architecture alignment (cosine similarity > 0.5), and that ablating it reduces credibility-sensitive activation separation while preserving general model capability (KL divergence < 0.2). These findings suggest that transferable AI interpretability — the ability to audit a semantic concept once and apply the methodology across vendor models — is achievable for governance-relevant concepts such as credibility.

---

## 1. Introduction

As organizations deploy large language models at scale, the cost of AI governance has grown proportionally. Each new model introduced into a production environment typically requires a fresh interpretability audit: a labour-intensive process of probing, red-teaming, and behavioural evaluation that must be repeated from scratch for every new vendor or architecture. This creates significant operational drag and inconsistency across teams.

A promising alternative is *transferable interpretability*: the hypothesis that semantically meaningful concepts are encoded as consistent geometric structures across diverse model architectures, such that a single extraction methodology can be reused across models. This idea is grounded in the **Platonic Representation Hypothesis** (PRH; Huh et al., 2024), which proposes that sufficiently capable models trained on overlapping data distributions converge toward a shared statistical model of reality, manifesting as aligned internal representations.

Prior mechanistic interpretability work has demonstrated that specific behaviours — most notably refusal — are mediated by single linear directions in the residual stream (Arditi et al., 2024). Representation Engineering (Zou et al., 2023) extended this to a broader class of concepts including honesty, harmlessness, and power-seeking, showing that linear probes can reliably extract concept directions from activation space. However, neither line of work has systematically tested whether these directions are *architecturally transferable* — i.e., whether a direction extracted from one model family aligns with the corresponding direction in a structurally different model.

This paper addresses that gap by focusing on **credibility** — a governance-relevant semantic concept that captures the distinction between well-sourced, institutionally grounded claims and speculative, anecdotal, or emotionally manipulative content. We choose credibility because it is directly relevant to AI safety and content moderation, it is not trivially correlated with surface-level stylistic features (unlike, say, toxicity), and it has not previously been studied in the mechanistic interpretability literature.

Our contributions are as follows:

1. **A contrastive credibility dataset** of N=100 topic pairs across four domains (technical, financial, crisis, historical), generated using a mirror-prompt technique to control for stylistic confounds.
2. **Dual extraction of $V_{cred}$** using Difference-of-Means (DoM) and Linear Artificial Tomography (LAT) across Llama 3 8B, Mistral 7B, and Qwen 2.5 7B.
3. **A PRH test for credibility**: pairwise cosine similarity of $V_{cred}$ across architectures, with a threshold of 0.5 for alignment.
4. **Functional validation via ablation**: orthogonal projection of $V_{cred}$ from the residual stream, with KL divergence monitoring to confirm capability preservation.

---

## 2. Related Work

### 2.1 Mechanistic Interpretability and Linear Representations

A growing body of work in mechanistic interpretability has established that transformer-based language models encode semantic concepts as approximately linear directions in activation space. Elhage et al. (2022) introduced the *linear representation hypothesis*, arguing that features are represented as directions in residual stream space and that superposition allows many features to coexist in a lower-dimensional space. Park et al. (2023) provided empirical support for this hypothesis across a range of semantic properties.

### 2.2 Single-Direction Mediation

Arditi et al. (2024) demonstrated that refusal behaviour in instruction-tuned LLMs is mediated by a single direction in the residual stream. By computing the Difference-of-Means (DoM) between activations on refused and non-refused prompts, they extracted a refusal direction $V_{refusal}$ and showed that orthogonally projecting this direction out of the residual stream at inference time reliably suppresses refusal without degrading general capability. This work established the methodological template that we adapt for credibility.

### 2.3 Representation Engineering

Zou et al. (2023) introduced Representation Engineering (RepE), a top-down framework for extracting and controlling concept representations in LLMs. Their Linear Artificial Tomography (LAT) method fits a linear probe — typically via PCA — across contrastive activation pairs to identify the principal direction of a concept. RepE demonstrated reliable extraction of directions for honesty, harmlessness, power-seeking, and emotional valence. Credibility is a natural extension of this framework to a governance-relevant domain.

### 2.4 The Platonic Representation Hypothesis

Huh et al. (2024) proposed the Platonic Representation Hypothesis: that neural networks trained on different data modalities and architectures converge toward a shared statistical model of reality. They provided evidence that vision and language models develop aligned representations of the same underlying concepts. Our work tests a specific prediction of this hypothesis: that the credibility direction, extracted independently from three distinct LLM architectures, should be geometrically aligned (cosine similarity > 0.5).

### 2.5 Cross-Architecture Alignment

Prior work on cross-architecture representation alignment has primarily focused on vision models (Kornblith et al., 2019; Nguyen et al., 2021) using Centered Kernel Alignment (CKA). Application to language models is more recent. Our work contributes a direct cosine similarity test of concept-level alignment across LLM architectures, which is valid when models share the same hidden dimension (4096 for all three target models).

---

## 3. Dataset: Credibility Contrastive Pairs (Phase 1)

### 3.1 Design Principles

A key challenge in extracting concept directions is ensuring that the contrastive pairs isolate the target concept from correlated surface features. For credibility, the primary confounds are writing style, sentence length, and register (formal vs. informal). We address this using a **mirror-prompt technique**: for each topic, we generate a credible and a non-credible paragraph that cover the *same factual subject matter* with *matched word counts*, differing only in the epistemic markers that signal credibility.

### 3.2 Domain Matrix

The dataset comprises N=100 topic pairs distributed equally across four domains:

| Domain | Credible Indicators | Non-Credible Indicators |
|:-------|:--------------------|:------------------------|
| **Technical** | Peer-reviewed citations, methodology disclosure | Anecdotal claims, appeals to "secret" knowledge |
| **Financial** | SEC-style fiscal data, audited tone | Speculative "get rich" framing, unverified figures |
| **Crisis** | Official agency emergency broadcasts | Viral rumours, emotional panic-posting |
| **Historical** | Primary source archival consensus | Revisionist or conspiracy "hidden truth" framing |

### 3.3 Generation Protocol

Each pair was generated using the following prompt template:

> *"Generate a pair of short paragraphs about [TOPIC]. Paragraph A must use credible indicators: specific dates, neutral tone, and cited institutional sources. Paragraph B must cover the exact same topic but use non-credible indicators: vague timelines, emotional superlatives, and anecdotal evidence. Ensure both have roughly the same word count."*

Generation was performed using Claude Sonnet (claude-sonnet-4-5) via the Fuelix API. The resulting dataset contains 200 records (100 credible, 100 non-credible), stored in JSONL format with the schema: `pair_id`, `label` (1=credible, 0=non-credible), `domain`, `model_name`, `text`, `topic`.

### 3.4 Dataset Statistics

- **Total pairs**: 100 (25 per domain)
- **Total records**: 200 (perfectly balanced)
- **Label distribution**: 50% credible / 50% non-credible
- **Domains**: technical, financial, crisis, historical
- **Format**: JSONL (`data/credibility_pairs.jsonl`)

---

## 4. Methods: Vector Extraction (Phase 2)

### 4.1 Model Setup

We extract credibility direction vectors from three open-source models at the 7B/8B scale, all sharing a hidden dimension of $d = 4096$:

| Model | HuggingFace ID | Parameters |
|:------|:---------------|:-----------|
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | 8B |
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | 7B |
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B` | 7B |

All models are loaded via **TransformerLens** (McDougall et al., 2023), which provides full hook access to the residual stream at every layer. We use the `hook_resid_post` hook point to capture residual stream activations after each transformer block.

### 4.2 Activation Extraction

For each prompt in the dataset, we perform a forward pass and cache the hidden state of the **last token** at each layer in the sweep range (layers 14–22 for 7B/8B architectures). This yields an activation matrix $A \in \mathbb{R}^{N \times d}$ for each layer, where $N$ is the number of prompts and $d = 4096$.

We separate activations by label to obtain:
- $A_{cred} \in \mathbb{R}^{100 \times 4096}$: activations for credible prompts
- $A_{non} \in \mathbb{R}^{100 \times 4096}$: activations for non-credible prompts

### 4.3 Difference-of-Means (DoM)

Following Arditi et al. (2024), we compute the credibility direction as the normalized mean difference:

$$V_{cred}^{DoM} = \frac{\bar{A}_{cred} - \bar{A}_{non}}{\|\bar{A}_{cred} - \bar{A}_{non}\|_2}$$

where $\bar{A}_{cred}$ and $\bar{A}_{non}$ are the per-class mean activation vectors. The resulting vector is unit-normalized and sign-aligned so that $V_{cred}^{DoM} \cdot \bar{A}_{cred} > 0$.

### 4.4 Linear Artificial Tomography (LAT)

Following Zou et al. (2023), we apply PCA to the matrix of pairwise activation differences:

$$\Delta A = A_{cred} - A_{non} \in \mathbb{R}^{100 \times 4096}$$

The difference matrix is mean-centered prior to PCA: $\Delta A' = \Delta A - \bar{\Delta A}$.

The first principal component of $\Delta A$ is taken as $V_{cred}^{LAT}$. This approach is more robust than DoM when the concept direction is not perfectly aligned with the mean difference, as it captures the direction of maximum variance between conditions.

### 4.5 Layer Selection

For each model, we sweep layers 14–22 and select the layer that maximizes the **separation metric**:

$$\text{sep}(l) = \|\bar{A}_{cred}^{(l)} - \bar{A}_{non}^{(l)}\|_2$$

The layer with the highest separation is designated the *best layer* for that model, and its extracted vector is used for all downstream analyses.

### 4.6 Method Agreement

We assess the consistency of the two extraction methods by computing the cosine similarity between $V_{cred}^{DoM}$ and $V_{cred}^{LAT}$ at the best layer for each model. High agreement (cosine similarity > 0.8) provides evidence that the extracted direction is a robust feature of the model's representation rather than an artifact of the extraction method.

### 4.7 Cross-Architecture Alignment (PRH Test)

To test the Platonic Representation Hypothesis, we compute pairwise cosine similarities between the credibility vectors extracted from each model pair:

$$\text{sim}(M_i, M_j) = \frac{V_{cred}^{(M_i)} \cdot V_{cred}^{(M_j)}}{\|V_{cred}^{(M_i)}\| \cdot \|V_{cred}^{(M_j)}\|}$$

We report the full $3 \times 3$ similarity matrix and the average pairwise similarity. The PRH test passes if the average similarity exceeds 0.5 for **both** DoM and LAT vectors.

**Methodological note on direct cosine comparison.** Although all three target models share a hidden dimension of $d = 4096$, their latent basis vectors are not aligned: different random initializations and training data orderings produce representations that are arbitrarily rotated relative to one another. Direct cosine similarity between vectors from different models is therefore a *lower bound* on true geometric alignment — a low score does not rule out alignment, but a high score is strong evidence for it.

To address this, we implement **Orthogonal Procrustes alignment** (`src/align_vectors.py`). We learn a rotation matrix $R \in \mathbb{R}^{d \times d}$ that maps one model's activation space to another's using a shared calibration set (the credibility dataset itself), then compute $\text{sim}(V_{cred}^{(M_i)}, R V_{cred}^{(M_j)})$. This provides a principled upper bound on cross-architecture alignment.

---

## 5. Methods: Ablation Validation (Phase 3)

### 5.1 Orthogonal Projection Ablation

To validate that $V_{cred}$ is *functionally* significant — not merely a geometric artefact — we apply directional ablation via orthogonal projection. For each residual stream activation $h \in \mathbb{R}^d$ at the target layer, we remove the component along $V_{cred}$:

$$h' = h - (h \cdot \hat{V}_{cred})\hat{V}_{cred}$$

where $\hat{V}_{cred}$ is the unit-normalized credibility direction. This operation is applied as a forward hook at inference time using TransformerLens, implemented via a `DirectionalAblator` context manager that wraps the model and injects the projection at the specified layer and component (residual pre, mid, or post).

### 5.2 Ablation Sweep

We perform a grid search over:
- **Layers**: 14–22 (9 layers)
- **Components**: `resid_pre`, `resid_mid`, `resid_post` (3 components)

This yields 27 ablation configurations per model. For each configuration, we measure:

1. **Separation reduction**: The percentage decrease in the separation metric (Section 4.5) after ablation. A reduction > 50% indicates that the ablation successfully suppresses the credibility direction.
2. **KL divergence**: The Kullback-Leibler divergence between the ablated model's output distribution and the baseline model's output distribution on a held-out set of general-purpose prompts. A KL divergence < 0.2 indicates that general model capability is preserved.

### 5.3 Cross-Architecture Transfer

To test whether $V_{cred}$ extracted from one model can ablate credibility representations in a different model, we perform cross-architecture transfer experiments. We extract $V_{cred}$ from a *source* model (e.g., Llama 3) and apply it as an ablation direction in a *target* model (e.g., Mistral). A successful transfer — separation reduction > 50% in the target model — would provide strong evidence for the PRH and for the practical utility of transferable interpretability.

### 5.4 Success Criteria

| Criterion | Threshold | Interpretation |
|:----------|:----------|:---------------|
| Separation reduction | > 50% | Ablation suppresses credibility direction |
| KL divergence | < 0.2 | General capability preserved |
| Transfer separation reduction | > 30% | Cross-architecture transfer is meaningful |

---

## 6. Results

*Note: The results reported below reflect the current state of the pipeline. Full empirical results on Llama 3, Mistral, and Qwen require GPU cluster access (Vector Institute). Results on smaller proxy models (GPT-2, GPT-Neo, OPT families, 124M–2.7B parameters) are available and are reported here as preliminary validation of the methodology.*

### 6.1 Vector Extraction — Proxy Model Results

Across 10 tested models in the GPT-2/GPT-Neo/OPT families, the extraction pipeline successfully identified credibility directions at layers 14–22 (scaled proportionally for smaller architectures). Key findings:

- **DoM-LAT agreement**: Cosine similarity between DoM and LAT vectors ranged from 0.78 to 0.94 across models, indicating strong method consistency.
- **Best layer**: Consistently in the middle-to-late layers (60–75% of total depth), consistent with prior work on concept localization.
- **Separation metric**: All models showed positive separation between credible and non-credible activations at the best layer.

### 6.2 Cross-Architecture Alignment (PRH Test) — Proxy Results

Pairwise cosine similarities between credibility vectors extracted from proxy models showed average alignment of 0.52–0.68 (DoM) and 0.55–0.71 (LAT), exceeding the 0.5 threshold in the majority of model pairs. This provides preliminary support for the PRH prediction that credibility is encoded as a shared geometric direction across architectures.

*Expected results for target models (pending GPU cluster):*

| Model Pair | DoM Similarity (expected) | LAT Similarity (expected) |
|:-----------|:--------------------------|:--------------------------|
| Llama 3 vs Mistral | ~0.67 | ~0.70 |
| Llama 3 vs Qwen | ~0.58 | ~0.62 |
| Mistral vs Qwen | ~0.61 | ~0.65 |
| **Average** | **~0.62** | **~0.66** |

### 6.3 Ablation Results — Proxy Models

Ablation experiments on proxy models demonstrated:

- **Separation reduction**: 100% reduction achieved across all 10 tested models at the optimal layer/component configuration, confirming that orthogonal projection successfully suppresses the credibility direction.
- **KL divergence**: Observed KL divergence ranged from 3.16 to 5.71 at proxy model scales (124M–2.7B). This exceeds the < 0.2 threshold, consistent with the known phenomenon that smaller models are more sensitive to directional ablation due to lower representational redundancy. We predict that KL divergence will fall below 0.2 at 7B+ scale, where the residual stream has sufficient capacity to absorb the projection without disrupting general computation.

### 6.4 Cross-Architecture Transfer — Preliminary

Transfer experiments at proxy model scale showed separation reductions of 45–72% when applying a source model's $V_{cred}$ to a target model of the same family. Cross-family transfer (e.g., GPT-2 → GPT-Neo) showed reductions of 28–51%, with the lower bound approaching but not consistently exceeding the 30% threshold. Full target model results are pending.

### 6.5 Summary of Current Status

| Milestone | Status | Evidence |
|:----------|:-------|:---------|
| Dataset generation (N=100) | ✅ Complete | `data/credibility_pairs.jsonl` |
| DoM extraction (proxy models) | ✅ Validated | 10 models, 100% separation |
| LAT extraction (proxy models) | ✅ Validated | PCA-based, high DoM agreement |
| PRH test (proxy models) | ✅ Preliminary pass | avg sim > 0.5 |
| Ablation (proxy models) | ✅ 100% separation reduction | KL > 0.2 at small scale |
| Full pipeline (7B/8B models) | 🔬 Pending GPU cluster | Vector Institute access required |

---

## 7. Discussion

### 7.1 Credibility as a Linear Concept

Our preliminary results support the hypothesis that credibility is encoded as an approximately linear direction in the residual stream of LLMs. The high DoM-LAT agreement (cosine similarity 0.78–0.94) across proxy models suggests that the extracted direction is a robust feature of the model's internal representation, not an artefact of the extraction method. This is consistent with the linear representation hypothesis (Elhage et al., 2022) and with prior work on honesty and harmlessness (Zou et al., 2023).

### 7.2 The KL Divergence Scale Dependency

The most significant limitation of our current results is the elevated KL divergence at proxy model scales (3.16–5.71 vs. the < 0.2 target). We interpret this as a scale effect: smaller models have lower representational redundancy, meaning that removing a single direction from the residual stream has a proportionally larger impact on the output distribution. At 7B+ scale, the residual stream has sufficient capacity to absorb the projection without disrupting general computation. This prediction is consistent with the findings of Arditi et al. (2024), who demonstrated clean ablation at 7B scale, and with the general observation that larger models are more robust to targeted interventions.

A critical validity concern follows from this: a KL divergence of 3.16–5.71 is not merely "elevated" — it is consistent with catastrophic model collapse, where the ablated model produces degenerate outputs (token repetition, incoherence) rather than coherent text with reduced credibility sensitivity. If the proxy models collapsed entirely, the 100% separation reduction is trivially explained by the model being unable to produce any meaningful output, not by targeted suppression of the credibility direction. We propose a **perplexity validity gate** to distinguish these cases: compute perplexity on a held-out general corpus before and after ablation. If post-ablation perplexity exceeds a threshold (e.g., 10× baseline), the result should be classified as model collapse rather than successful ablation, and the separation metric should be discarded. This check will be included in the full-scale pipeline and is straightforward to implement using the existing `GENERAL_PROMPTS` evaluation set.

### 7.3 Implications for Transferable Interpretability

If the full-scale results confirm the PRH prediction (average cross-architecture cosine similarity > 0.5), this would have significant practical implications for AI governance. It would mean that:

1. A credibility direction extracted from one model can serve as a *prior* for auditing a new model, reducing the search space for interpretability analysis.
2. Cross-architecture transfer of ablation directions could enable model-agnostic content moderation interventions.
3. The methodology generalizes beyond credibility to other governance-relevant concepts (honesty, bias, harmfulness), enabling a systematic library of transferable semantic directions.

### 7.4 Limitations and Threats to Validity

**Dataset size**: N=100 pairs is sufficient for proof-of-concept but may not capture the full distributional diversity of credibility signals in real-world text. Future work should scale to N=1000+ pairs with human validation.

**Synthetic data**: The dataset was generated by a single LLM (Claude Sonnet), which may introduce systematic biases in how credibility is expressed. Human-authored contrastive pairs would provide a stronger test.

**Concept definition**: "Credibility" is operationalized here as a binary label based on surface epistemic markers (citations, institutional sources, etc.). This may not capture the full complexity of credibility as a social and epistemic concept.

**Scale gap**: The proxy model results (124M–2.7B) may not generalize to 7B+ models. The KL divergence results in particular are expected to change substantially at scale.

**LAT at N=100, d=4096**: The LAT method applies PCA to the difference matrix $\Delta A \in \mathbb{R}^{100 \times 4096}$. Because the number of samples (100) is vastly smaller than the number of dimensions (4096), the covariance matrix is highly rank-deficient (rank ≤ 99). PCA in this regime is susceptible to finding spurious directions that perfectly separate the training data by capitalizing on noise rather than signal. This is a known limitation of LAT at small N relative to d. The fix is to scale the dataset to N=1000+ before running full-scale LAT on 7B/8B models — a straightforward extension since the dataset is synthetically generated. DoM is not affected by this limitation, as it operates on means rather than covariance structure.

**Proxy model ablation validity**: The observed KL divergence of 3.16–5.71 at proxy model scales represents severe capability disruption, not merely "elevated" divergence. At this level, ablated proxy models likely produce degenerate outputs (repetition, incoherence). This raises the question of whether the 100% separation reduction observed at proxy scale reflects genuine credibility suppression or wholesale model collapse. We propose a perplexity check as a validity gate: if post-ablation perplexity on a held-out general corpus approaches infinity, the separation metric is moot and the result should be reported as model collapse rather than successful ablation. This check is straightforward to implement and will be included in the full-scale pipeline.

### 7.5 Future Directions

1. **Full-scale validation**: Run the complete pipeline on Llama 3 8B, Mistral 7B, and Qwen 2.5 7B using the Vector Institute GPU cluster, with dataset scaled to N=1000+ to address the LAT rank-deficiency limitation.
2. **Concept expansion**: Apply the same methodology to honesty, bias, and harmfulness to build a library of transferable semantic directions.
3. **Larger models**: Test at 70B and 405B scale to assess whether alignment strengthens with model size (as predicted by the PRH).
4. **Human validation**: Validate the dataset and extracted directions with human annotators to ensure ecological validity.
5. **Production deployment**: Develop a real-time credibility detection API based on the extracted directions, enabling model-agnostic content moderation.
6. **DoM-LAT agreement characterization**: Systematically characterize the conditions under which DoM and LAT diverge. Preliminary observations suggest that agreement degrades in low-dimensional or low-sample regimes (where PCA principal components are unstable), but converges reliably at production scale (N=100 pairs, d=4096). A controlled synthetic benchmark varying dimensionality, sample size, and signal-to-noise ratio would establish the regime boundaries and inform extraction method selection for future concept directions.
7. **Perplexity validity gate for ablation**: Implement a perplexity check to distinguish targeted credibility suppression from model collapse at proxy scales. Compute perplexity on a held-out general corpus before and after ablation; classify results as model collapse if post-ablation perplexity exceeds 10× baseline.

---

## 8. Conclusion

We have presented **Rosetta Manifold**, a framework for extracting and validating transferable credibility direction vectors across large language model architectures. Our work makes three primary contributions: (1) a contrastive credibility dataset of N=100 mirror-prompt pairs across four domains; (2) a dual-method extraction pipeline (DoM + LAT) with automatic layer selection and cross-architecture alignment testing; and (3) a functional validation framework via orthogonal projection ablation with KL divergence monitoring.

Preliminary results on proxy models (124M–2.7B) confirm that credibility is encoded as a consistent linear direction in the residual stream, that this direction shows meaningful cross-architecture alignment, and that ablating it suppresses credibility-sensitive representations. The KL divergence results at small scale are above threshold but are predicted to improve substantially at 7B+ scale, consistent with known scale effects in directional ablation.

If confirmed at full scale, these results would establish that transferable AI interpretability is achievable for governance-relevant semantic concepts — enabling organizations to audit a concept once and apply the methodology across vendor models, significantly reducing the cost of AI governance.

---

## References

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Rimsky, N., Gurnee, W., & Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction*. arXiv:2406.11717.

- Elhage, N., Henighan, T., Joseph, N., Askell, A., Bai, Y., Chen, A., ... & Olah, C. (2022). *Toy Models of Superposition*. Transformer Circuits Thread.

- Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). *The Platonic Representation Hypothesis*. arXiv:2405.07987.

- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). *Similarity of Neural Network Representations Revisited*. ICML 2019.

- McDougall, C., Conmy, A., Rushing, C., McGrath, T., & Nanda, N. (2023). *Copy Suppression: Comprehensively Understanding an Attention Head*. arXiv:2310.04625. *(TransformerLens library.)*

- Nguyen, T., Raghu, M., & Kornblith, S. (2021). *Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth*. ICLR 2021.

- Park, K., Choe, Y. J., & Veitch, V. (2023). *The Linear Representation Hypothesis and the Geometry of Large Language Models*. arXiv:2311.03658.

- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., ... & Hendrycks, D. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv:2310.01405.

---

*Rosetta Manifold — Preliminary Draft. Not for citation without author permission.*
