# Rosetta Manifold
*Decoding AI Through Universal Semantic Vectors*

## 1. Executive Summary
This project investigates the **Platonic Representation Hypothesis (PRH)** to establish a methodology for **Transferable AI Interpretability**. By identifying and isolating universal semantic vectors (specifically "Credibility") across diverse open-source architectures (Llama 3, Mistral, Qwen), we aim to reduce AI governance overhead and enable model-agnostic oversight.

## 2. Business Case (TELUS Context)
As TELUS scales multi-model AI deployments across security and compliance, the current need to repeat interpretability audits for every new model creates significant operational drag. 
- **The Solution:** If semantic concepts converge on shared geometric directions, a single "audit key" can verify trust across different vendor models.
- **Outcome:** Lower cost of governance and increased speed-to-market for new AI agents.

## 3. Technical Stack
- **Activation Extraction:** **TransformerLens** (loaded directly from HuggingFace weights) for full residual stream hook access across Llama 3, Mistral, and Qwen.
- **Ablation:** **abliterator** (FailSpy/abliterator) — TransformerLens-based directional ablation library implementing the Arditi et al. (2024) orthogonal projection technique.
- **Experiment Tracking:** **Opik** (Comet-ML) for dataset versioning, activation logging, and ablation audit trails.
- **Methodology:** Two complementary direction-extraction methods evaluated in parallel:
  - **Difference-of-Means (DoM):** Arditi et al. (2024), arXiv:2406.11717 — fast, interpretable, single-direction mediation.
  - **Linear Artificial Tomography (LAT):** Zou et al. (2023) Representation Engineering, arXiv:2310.01405 — top-down PCA-based approach; more robust to non-linear separability.
  - **PRH Test:** Cosine Similarity / CKA across architectures (Huh et al., 2024, arXiv:2405.07987).
- **Scale-up:** **Vector Institute** cluster for full N runs across all three architectures.

## 4. Repository Structure
- `docs/`: Technical specifications and PRH research notes.
- `src/`: Core extraction hooks and Heretic optimization pipelines.
- `data/`: Curated N=100 "Golden" contrastive prompt pairs (Managed via Opik).
- `experiments/`: Jupyter notebooks for vector alignment and similarity visualization.

## 5. Roadmap & Milestones
| Phase | Milestone | Objective |
| :--- | :--- | :--- |
| **Phase 1** | **C1 ✅ Done** | Deploy Synthetic "Credibility" Dataset (N=100) to Opik. |
| **Phase 2** | **C2 Extraction** | Compute $V_{cred}$ across Llama 3, Mistral, and Qwen. |
| **Phase 3** | **C3 Validation** | Demonstrate Heretic-based ablation with <0.2 KL Divergence. |

## 6. How to Run (Local Dev)
1. **Initialize Environment:** `conda create -n platonic python=3.10`.
2. **Install Dependencies:** `pip install transformer_lens abliterator opik optuna openai ollama`.
3. **Start Opik (self-hosted):** `./infra/opik/opik.sh up` — spins up the full Opik stack via Docker Compose.
4. **Configure SDK:** `./infra/opik/opik.sh configure` — writes `~/.opik.config` pointing at `http://localhost:5173/api/`.
5. **Generate Dataset (Phase 1):** `python src/generate_dataset.py` — uses Fuelix/Claude API by default; pass `--backend ollama` for local inference.
6. **Upload to Opik:** `python src/upload_to_opik.py` — pushes `data/credibility_pairs.jsonl` to the local Opik instance.
7. **Run Extraction (Phase 2):** `python src/extract_vectors.py --model meta-llama/Meta-Llama-3-8B`.

> **Opik UI:** http://localhost:5173
> **Opik API:** http://localhost:5173/api
> See [`infra/opik/README.md`](infra/opik/README.md) for full stack management commands.

## 7. Phase 1 Status — ✅ Complete (C1)

| Item | Status |
|:-----|:-------|
| N=100 contrastive pairs generated | ✅ `data/credibility_pairs.jsonl` |
| 25 pairs per domain (technical, financial, crisis, historical) | ✅ |
| Balanced labels (100 credible / 100 non-credible) | ✅ |
| Schema: `pair_id`, `label`, `domain`, `model_name`, `text`, `topic` | ✅ |
| Generation model | `claude-sonnet-4-5` via Fuelix API |
| Opik upload script | `src/upload_to_opik.py` (run after `opik configure`) |