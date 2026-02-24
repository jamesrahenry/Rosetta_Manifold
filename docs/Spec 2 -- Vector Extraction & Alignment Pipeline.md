# Spec 2: Vector Extraction & Alignment Pipeline

## 1. Extraction Strategy

- **Environment:** Local Python + **TransformerLens** (loaded directly via HuggingFace weights).
- **Why not Ollama:** Ollama is an inference server and does not expose internal residual stream activations. TransformerLens loads model weights natively and provides full hook access to every layer's residual stream, attention outputs, and MLP outputs.
- **Hook Point:** Residual stream activations (`hook_resid_post`) from the middle-to-late layers (Layers 14–22 for 7B/8B architectures). This is a hyperparameter to be swept.
- **Capture:** Cache the hidden state of the **last token** of the prompt for all N pairs. Token capture position is a secondary hyperparameter (last token vs. mean-pool over last K tokens).
- **Library:** `transformer_lens` + `abliterator` (FailSpy/abliterator) for activation caching and direction utilities.

### Supported Models via TransformerLens

| Model | HuggingFace ID | TL Weight Converter |
|:------|:---------------|:--------------------|
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | `convert_llama_weights` |
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | `convert_mistral_weights` |
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B` | `convert_qwen2_weights` |

## 2. Direction Extraction Methods

Two complementary approaches from the literature should be evaluated:

### 2a. Difference-of-Means (DoM) — Arditi et al. (2024)

Compute the directional vector $V_{cred}$ for each model architecture using the method from Arditi et al. (2024), "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717):

$$V_{cred} = \frac{1}{n} \sum A_{credible} - \frac{1}{n} \sum B_{non-credible}$$

The `abliterator` library provides `calculate_mean_diff()` which implements this directly. DoM is fast and interpretable but assumes the concept is linearly separable.

### 2b. Linear Artificial Tomography (LAT) — Zou et al. (2023)

**Representation Engineering (RepE)** (Zou et al., 2023, arXiv:2310.01405) proposes **Linear Artificial Tomography (LAT)**: fit a linear probe (PCA or linear classifier) across the contrastive activations to extract the principal direction of the concept. This is a **top-down** approach that operates on population-level representations rather than individual neuron circuits.

- **RepE library:** `pip install -e git+https://github.com/andyzoujm/representation-engineering` — provides `RepReadingPipeline` and `RepControlPipeline` built on HuggingFace.
- **Advantage over DoM:** LAT is more robust when the concept is not perfectly linearly separable; it finds the direction of maximum variance between conditions rather than the raw mean difference.
- **Recommendation:** Run both DoM and LAT; compare the resulting $V_{cred}$ directions via cosine similarity. High agreement between methods strengthens the claim that the direction is real and not an artifact of the extraction method.

## 3. Alignment Metric (PRH Test)

To test the **Platonic Representation Hypothesis** (Huh et al., 2024, arXiv:2405.07987), measure the **Cosine Similarity** between vectors extracted from different models:

- Compare $V_{cred(Llama3)}$ vs. $V_{cred(Mistral)}$ vs. $V_{cred(Qwen)}$.
- **Goal:** Similarity $> 0.5$ (indicating architectural convergence on a shared credibility direction).
- **Note on dimensionality:** All three 7B/8B models share a hidden dimension of 4096, so direct cosine similarity is valid. For topological comparison across different hidden sizes, use **Centered Kernel Alignment (CKA)** as a fallback metric.

## 4. Logging (Opik Experiments)

Use `@opik.track` to log the extraction results:

- Log `cosine_similarity` as a metric per model pair.
- Log `layer_index` where the vector separation was highest.
- Log `token_capture_strategy` (last token vs. mean-pool) as a hyperparameter.
- Log `extraction_method` (DoM vs. LAT) as a hyperparameter.

## 5. Key References

- Arditi et al. (2024): [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) — Establishes single-direction mediation for refusal via Difference-of-Means; this project applies the same technique to credibility.
- Zou et al. (2023): [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) — Representation Engineering (RepE); introduces LAT as a top-down approach to extracting concept directions. Demonstrates on honesty, harmlessness, power-seeking, and more — credibility is a natural extension.
- Huh et al. (2024): [arXiv:2405.07987](https://arxiv.org/abs/2405.07987) — The Platonic Representation Hypothesis; theoretical foundation for cross-architecture vector alignment.
- FailSpy/abliterator: [github.com/FailSpy/abliterator](https://github.com/FailSpy/abliterator) — TransformerLens-based activation caching and direction calculation library.
- andyzoujm/representation-engineering: [github.com/andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering) — Official RepE implementation with RepReading and RepControl pipelines.
