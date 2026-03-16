# data/

Contrastive pair datasets for CAZ (Concept Assembly Zone) empirical validation.
All datasets: N=100 pairs (200 records), perfectly balanced (100 pos / 100 neg).
See `MANIFEST.json` for full per-dataset metadata and extraction run history.

## Schema

Each record contains:

| Field        | Type    | Description                                              |
|:-------------|:--------|:---------------------------------------------------------|
| `pair_id`    | string  | Unique identifier for the contrastive pair               |
| `label`      | int     | `1` = positive class, `0` = negative class               |
| `domain`     | string  | Topic domain                                             |
| `model_name` | string  | Generation model (HuggingFace or FuelIX model ID)        |
| `text`       | string  | Generated statement text                                 |
| `topic`      | string  | Topic prompt used for generation                         |
| `concept`    | string  | Concept being tested (absent in older datasets)          |

## Datasets

| Dataset | Concept type | Domains | Generation date | Notes |
|:--------|:-------------|:--------|:----------------|:------|
| `credibility_pairs.jsonl` | epistemic | 4 × 25 pairs | 2026-03-10 | Original dataset |
| `negation_pairs.jsonl` | syntactic | 4 × 25 pairs | 2026-03-14 | |
| `sentiment_pairs.jsonl` | affective | **10 × 10 pairs** | 2026-03-14 | See note below |
| `certainty_pairs.jsonl` | epistemic | 4 × 25 pairs | 2026-03-15 | |
| `causation_pairs.jsonl` | relational | 4 × 25 pairs | 2026-03-15 | |
| `moral_valence_pairs.jsonl` | affective | 4 × 25 pairs | 2026-03-15 | |
| `temporal_order_pairs.jsonl` | relational | 4 × 25 pairs | 2026-03-15 | |
| `plurality_pairs.jsonl` | syntactic | 4 × 25 pairs | 2026-03-15 | **DISCONTINUED** — negative result, see note below |

## Methodology Notes

### Sentiment domain structure
Sentiment was generated with a 10-domain structure (10 pairs per domain) covering
affective life domains: personal_experience, relationships, work_career, health_wellness,
learning_growth, creative_pursuits, social_community, nature_environment, future_outlook,
daily_life. All other concepts use a 4-domain structure (25 pairs per domain).

This difference was intentional: broader domain coverage was chosen for sentiment to
capture the full range of affective contexts, reducing the risk that a single content
domain drives the signal. Empirically, the 10-domain structure produced metric profiles
consistent with the 4-domain datasets — sentiment does not appear as an outlier in any
S/C/V analysis across the 8-model expanded run. The difference is documented here for
methods transparency but does not constitute a confound for within-concept S/C/V
measurements. If domain-stratified analysis is required, sentiment's finer granularity
is an advantage rather than a liability.

For paper methods: "Sentiment was generated with a 10-domain structure (n=10 per domain)
to capture affective range; all other concepts used a 4-domain structure (n=25 per domain).
This structural difference did not produce anomalous metric profiles."

### Plurality — known methodological limitation
The plurality dataset operationalizes the concept as wholesale grammatical number
transformation: every positive text uses plural constructions throughout ("researchers
have found", "the studies show") and every negative text is the singular equivalent
("a researcher has found", "the study shows"), with identical factual content.

**Problems identified in post-hoc analysis (2026-03-16):**

1. **Surface morphosyntax, not semantic plurality.** The contrast is grammatical
   number agreement rather than a meaningful semantic distinction. Models can resolve
   this from the first token without building conceptual representations.

2. **Credibility confound.** Plural constructions systematically read as more
   authoritative (multiple studies, multiple researchers) while singular constructions
   read as more tentative. The dataset conflates plurality with evidential weight,
   making it difficult to isolate the plurality signal from the credibility signal.

3. **Anomalous metric profile.** In the 8-model expanded run, plurality showed a
   mean peak depth of 41.4% with a standard deviation of 42.0% — the largest variance
   of any concept. OPT and Pythia models peak at layers 0–17% (near-surface syntactic
   resolution), while GPT-2 and GPT-Neo peak at 92–98% (anomalously deep). This
   bimodal pattern is inconsistent with the CAZ framework's predictions and likely
   reflects architectural differences in number agreement processing rather than
   concept assembly dynamics.

**Status:** Discontinued as of 2026-03-16. Plurality is retained in the repository
as a documented negative result rather than removed. The finding is informative:
it establishes that the CAZ framework does not have purchase on surface
morphosyntactic features resolved at the token level, and that grammatical number
agreement is handled architecturally inconsistently across model families in ways
that reflect implementation differences rather than concept assembly dynamics.
This is a useful boundary condition on the framework's scope. The active concept
taxonomy for frontier runs is the seven-concept set excluding plurality.

## Audit History

| Date | Action |
|:-----|:-------|
| 2026-03-16 | `plurality_pairs.jsonl`: discontinued as documented negative result. Grammatical number agreement is a surface morphosyntactic feature outside the CAZ framework's scope. Retained in repository as archival negative result. Active concept taxonomy is now 7 concepts. |
| 2026-03-16 | Full structural audit: all 8 datasets verified 100 pairs, balanced labels, no duplicate texts, no short texts |
| 2026-03-16 | `certainty_pairs.jsonl`: fixed duplicate `pair_id='history_25'` (two distinct Magna Carta pairs, ID collision from generation). Reassigned second pair to `history_25b`. Data was always valid. |
| 2026-03-16 | `sentiment_pairs.jsonl`: corrected MANIFEST domain list (10 affective domains, not 4 abstract domains) |
| 2026-03-15 | Expanded run: all 8 concepts generated, 64 model extractions completed |
