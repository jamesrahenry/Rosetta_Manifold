# data/

Curated **N=100 "Golden" contrastive prompt pairs** for the Credibility vector extraction experiment.

Dataset is managed and versioned via **Opik** (Comet-ML). See
[`docs/Spec 1 -- Credibility Contrastive Dataset.md`](../docs/Spec%201%20--%20Credibility%20Contrastive%20Dataset.md)
for full generation methodology.

## Schema

Each record in `credibility_pairs.jsonl` contains:

| Field        | Type    | Description                                              |
|:-------------|:--------|:---------------------------------------------------------|
| `pair_id`    | string  | Unique identifier for the contrastive pair               |
| `label`      | int     | `1` = Credible, `0` = Non-Credible                       |
| `domain`     | string  | `technical`, `financial`, `crisis`, or `historical`      |
| `model_name` | string  | Ollama model string used for generation (e.g. `llama3:8b`) |
| `text`       | string  | The prompt paragraph                                     |

## Domain Distribution (25 pairs each)

| Domain       | Credible indicator                        | Non-Credible indicator                    |
|:-------------|:------------------------------------------|:------------------------------------------|
| `technical`  | Peer-reviewed citation + methodology      | Anecdotal claim + "secret" knowledge      |
| `financial`  | SEC-style fiscal data + audited tone      | Speculative "get rich" + unverified       |
| `crisis`     | Official agency emergency broadcast       | Viral rumor + emotional panic-posting     |
| `historical` | Primary source archival consensus         | Revisionist/Conspiracy "hidden truth"     |

## Phase 1 Milestone — ✅ Complete

**C1:** Dataset generated and validated.

| Item | Detail |
|:-----|:-------|
| Records | 200 (100 pairs) |
| Labels | 100 credible (label=1) / 100 non-credible (label=0) |
| Domains | 50 records each: technical, financial, crisis, historical |
| Generation model | `claude-sonnet-4-5` via Fuelix API |
| Generation script | `src/generate_dataset.py` |
| Opik upload | `python src/upload_to_opik.py` (after `opik configure`) |

**To deploy to Opik:**
```bash
# Cloud:
opik configure --api_key <YOUR_OPIK_API_KEY>
python src/upload_to_opik.py

# Local instance:
opik configure --use_local --url http://localhost:5173
python src/upload_to_opik.py
```
