# Spec 1: Credibility Contrastive Dataset (N=100)

## 1. Objective
To generate a high-quality, synthetic dataset of contrastive prompt pairs that isolate "Credibility" (Z) from stylistic noise like professionalism or sentence length.

## 2. Domain Matrix (25 Pairs Each)
| Domain      | Credible (Label: 1)                     | Non-Credible (Label: 0)                  |
|:------------|:----------------------------------------|:-----------------------------------------|
| **Technical** | Peer-reviewed citation + methodology.  | Anecdotal claim + "secret" knowledge.    |
| **Financial** | SEC-style fiscal data + audited tone.   | Speculative "get rich" + unverified.    |
| **Crisis** | Official agency emergency broadcast.    | Viral rumor + emotional panic-posting.  |
| **Historical**| Primary source archival consensus.      | Revisionist/Conspiracy "hidden truth."  |

## 3. Data Schema (Opik Integration)
Each entry must be logged to Opik with the following metadata:
- `pair_id`: Unique identifier for the contrastive set.
- `label`: 1 (Credible) or 0 (Non-Credible).
- `domain`: One of the four categories above.
- `model_name`: The local Ollama model string (e.g., `llama3:8b`).

## 4. Generation Prompt (The "Mirror" Technique)
"Generate a pair of short paragraphs about [TOPIC]. Paragraph A must use credible indicators: specific dates, neutral tone, and cited institutional sources. Paragraph B must cover the exact same topic but use non-credible indicators: vague timelines, emotional superlatives, and anecdotal evidence. Ensure both have roughly the same word count."