"""
generate_dataset.py

Phase 1 (C1): Generate the N=100 Credibility Contrastive Dataset and deploy to Opik.

Generates 25 contrastive pairs per domain (technical, financial, crisis, historical)
using the "Mirror" technique from Spec 1. Each pair produces two JSONL records:
  - label=1 (Credible)
  - label=0 (Non-Credible)

Total: 4 domains × 25 pairs × 2 labels = 200 records (100 pairs).

Usage:
    # Fast path — Anthropic API (default):
    python src/generate_dataset.py [--output data/credibility_pairs.jsonl]

    # Ollama fallback:
    python src/generate_dataset.py --backend ollama [--model mistral:latest]

See: docs/Spec 1 -- Credibility Contrastive Dataset.md
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain topic seeds — 25 topics per domain
# ---------------------------------------------------------------------------

DOMAIN_TOPICS: dict[str, list[str]] = {
    "technical": [
        "mRNA vaccine efficacy in immunocompromised patients",
        "CRISPR-Cas9 off-target editing rates",
        "quantum error correction thresholds",
        "large language model hallucination rates",
        "lithium-ion battery thermal runaway mechanisms",
        "deep-sea hydrothermal vent microbial diversity",
        "PFAS contamination in municipal water supplies",
        "antibiotic resistance gene transfer in hospital settings",
        "solar panel degradation rates over 25 years",
        "5G millimeter-wave signal attenuation in urban canyons",
        "microplastic accumulation in Arctic sea ice",
        "CERN LHC proton collision energy records",
        "AlphaFold protein structure prediction accuracy",
        "carbon nanotube tensile strength measurements",
        "CRISPR base editing precision in sickle cell disease",
        "nuclear fusion plasma confinement duration records",
        "satellite internet latency benchmarks",
        "mRNA stability at room temperature",
        "graphene electrical conductivity at room temperature",
        "autonomous vehicle LiDAR false positive rates",
        "deep learning model energy consumption per inference",
        "CRISPR prime editing efficiency in human cells",
        "quantum computing qubit coherence times",
        "ocean acidification impact on coral calcification",
        "gene therapy vector immunogenicity in clinical trials",
    ],
    "financial": [
        "Federal Reserve interest rate decision impact on mortgage rates",
        "SEC enforcement actions against insider trading",
        "corporate earnings restatements and audit findings",
        "sovereign debt restructuring terms in emerging markets",
        "Basel III capital adequacy requirements for banks",
        "cryptocurrency exchange insolvency proceedings",
        "ESG fund performance versus benchmark indices",
        "private equity leveraged buyout debt ratios",
        "central bank digital currency pilot programs",
        "hedge fund short-selling disclosure requirements",
        "FDIC deposit insurance coverage limits",
        "municipal bond default rates by credit rating",
        "venture capital dry powder levels in 2024",
        "commercial real estate loan delinquency rates",
        "trade deficit impact on currency exchange rates",
        "pension fund liability gap in public sector",
        "IPO lock-up period expiration effects on share price",
        "credit default swap spreads during banking stress",
        "foreign direct investment flows to Southeast Asia",
        "inflation-adjusted S&P 500 returns over 30 years",
        "SPAC merger completion rates and shareholder returns",
        "corporate bond covenant violations in 2023",
        "remittance flows to Latin America in 2024",
        "bank stress test results under adverse scenarios",
        "commodity futures contango and backwardation patterns",
    ],
    "crisis": [
        "wildfire evacuation orders in Northern California",
        "hurricane storm surge predictions for Gulf Coast",
        "earthquake early warning system activation thresholds",
        "chemical plant explosion emergency response protocols",
        "dam failure downstream flood risk assessment",
        "pandemic influenza hospital surge capacity planning",
        "nuclear power plant coolant leak containment procedures",
        "bridge structural failure warning signs",
        "drinking water contamination boil-water advisory",
        "train derailment hazardous materials spill response",
        "tornado warning lead time and shelter-in-place guidance",
        "coastal tsunami inundation zone mapping",
        "power grid blackout restoration priority sequence",
        "mass casualty incident triage protocols",
        "oil pipeline rupture environmental containment",
        "school lockdown active threat response procedures",
        "flash flood watch versus warning distinctions",
        "volcanic ash cloud aviation safety restrictions",
        "heat dome public health advisory thresholds",
        "cyberattack on critical infrastructure response",
        "avalanche risk assessment and closure decisions",
        "gas leak emergency evacuation radius",
        "blizzard road closure and travel ban criteria",
        "industrial ammonia release shelter-in-place guidance",
        "river levee breach flood inundation mapping",
    ],
    "historical": [
        "causes of the 1929 stock market crash",
        "primary sources on the Rwandan genocide",
        "archaeological evidence for the Bronze Age Collapse",
        "declassified documents on the Cuban Missile Crisis",
        "census records and the transatlantic slave trade",
        "eyewitness accounts of the Chernobyl disaster",
        "treaty texts from the Congress of Vienna",
        "forensic evidence from the Katyn massacre",
        "archival records of the Armenian genocide",
        "primary sources on the Partition of India",
        "diplomatic cables from the lead-up to World War I",
        "archaeological findings at Pompeii",
        "court records from the Salem witch trials",
        "UN resolutions during the Srebrenica massacre",
        "NASA mission logs from the Apollo 11 landing",
        "Nuremberg trial transcripts on Holocaust evidence",
        "carbon dating of the Dead Sea Scrolls",
        "census data on the Irish Famine mortality",
        "declassified CIA files on Operation Paperclip",
        "primary sources on the Tiananmen Square protests",
        "archival evidence for the Holodomor famine",
        "treaty of Versailles reparations clauses",
        "eyewitness testimony from the Hiroshima bombing",
        "colonial land grant records in North America",
        "primary sources on the fall of Constantinople",
    ],
}

# ---------------------------------------------------------------------------
# Mirror prompt template (Spec 1, Section 4)
# ---------------------------------------------------------------------------

MIRROR_PROMPT = (
    'Generate a pair of short paragraphs about "{topic}". '
    "Paragraph A must use credible indicators: specific dates, neutral tone, "
    "and cited institutional sources. "
    "Paragraph B must cover the exact same topic but use non-credible indicators: "
    "vague timelines, emotional superlatives, and anecdotal evidence. "
    "Ensure both have roughly the same word count.\n\n"
    "Respond ONLY with valid JSON in this exact format:\n"
    '{{"paragraph_a": "<credible text>", "paragraph_b": "<non-credible text>"}}'
)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> Optional[dict]:
    """Extract the first JSON object from a model response."""
    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # Try to find a JSON block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Generation — Anthropic async backend
# ---------------------------------------------------------------------------

_ANTHROPIC_SEMAPHORE: Optional[asyncio.Semaphore] = None


async def _generate_pair_anthropic(
    topic: str,
    domain: str,
    pair_index: int,
    model_name: str,
    client,  # openai.AsyncOpenAI (pointed at Fuelix)
) -> list[dict]:
    """Async OpenAI-compatible call for one contrastive pair (via Fuelix)."""
    global _ANTHROPIC_SEMAPHORE
    prompt = MIRROR_PROMPT.format(topic=topic)
    pair_id = f"{domain}_{pair_index:02d}"

    async with _ANTHROPIC_SEMAPHORE:
        log.info("  [%s] Generating pair %s — %s", model_name, pair_id, topic)
        response = await client.chat.completions.create(
            model=model_name,
            max_tokens=1024,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
    raw = response.choices[0].message.content
    parsed = _extract_json(raw)

    if parsed is None or "paragraph_a" not in parsed or "paragraph_b" not in parsed:
        log.warning("    Failed to parse JSON for %s — using raw fallback", pair_id)
        parts = re.split(r"[Pp]aragraph\s+[Bb]", raw, maxsplit=1)
        para_a = parts[0].replace("Paragraph A", "").strip().strip(":").strip()
        para_b = parts[1].strip().strip(":").strip() if len(parts) > 1 else raw.strip()
        parsed = {"paragraph_a": para_a, "paragraph_b": para_b}

    return [
        {
            "pair_id": pair_id,
            "label": 1,
            "domain": domain,
            "model_name": model_name,
            "text": parsed["paragraph_a"].strip(),
            "topic": topic,
        },
        {
            "pair_id": pair_id,
            "label": 0,
            "domain": domain,
            "model_name": model_name,
            "text": parsed["paragraph_b"].strip(),
            "topic": topic,
        },
    ]


async def _generate_all_anthropic(model_name: str, concurrency: int) -> list[dict]:
    """
    Generate all 100 pairs via the Fuelix/OpenAI-compatible API.

    Fuelix exposes an OpenAI-compatible endpoint at /v1/chat/completions and
    requires 'Authorization: Bearer <key>' rather than the Anthropic 'x-api-key'
    header, so we use the openai AsyncOpenAI client pointed at the Fuelix base URL.

    Rate-limit strategy: process pairs in batches of `concurrency`. If a batch
    hits a 429, wait 62 seconds for the 1-minute window to reset, then retry
    only the failed tasks.
    """
    from openai import AsyncOpenAI, RateLimitError

    global _ANTHROPIC_SEMAPHORE
    _ANTHROPIC_SEMAPHORE = asyncio.Semaphore(concurrency)

    api_key = os.environ.get("FUELIX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.fuelix.ai/v1")
    # Ensure the base_url ends with /v1 for the OpenAI client
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    # Build the full task list as (domain, idx, topic) tuples
    pending: list[tuple[str, int, str]] = []
    for domain, topics in DOMAIN_TOPICS.items():
        for idx, topic in enumerate(topics, start=1):
            pending.append((domain, idx, topic))

    all_records: list[dict] = []
    attempt = 0

    while pending:
        attempt += 1
        log.info("Batch attempt %d — %d pairs remaining", attempt, len(pending))

        # Process up to `concurrency` at a time to stay under rate limits
        batch = pending[:concurrency]
        remaining = pending[concurrency:]

        tasks = [
            _generate_pair_anthropic(
                topic=topic,
                domain=domain,
                pair_index=idx,
                model_name=model_name,
                client=client,
            )
            for domain, idx, topic in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        retry_batch: list[tuple[str, int, str]] = []
        for (domain, idx, topic), result in zip(batch, results):
            if isinstance(result, RateLimitError):
                log.warning("  Rate-limited on %s_%02d — will retry", domain, idx)
                retry_batch.append((domain, idx, topic))
            elif isinstance(result, Exception):
                log.error("  Permanent error on %s_%02d: %s", domain, idx, result)
            else:
                all_records.extend(result)

        if retry_batch:
            wait_secs = 62
            log.info(
                "Rate limit hit — waiting %ds for window reset before retrying %d pairs",
                wait_secs,
                len(retry_batch),
            )
            await asyncio.sleep(wait_secs)
            pending = retry_batch + remaining
        else:
            pending = remaining
            if pending:
                # Small pause between batches to avoid immediate rate-limit
                await asyncio.sleep(2)

    await client.close()
    return all_records


# ---------------------------------------------------------------------------
# Generation — Ollama sync fallback
# ---------------------------------------------------------------------------


def _generate_pair_ollama(
    topic: str,
    domain: str,
    pair_index: int,
    model_name: str,
    client,  # ollama.Client
) -> list[dict]:
    """Synchronous Ollama call for one contrastive pair."""
    prompt = MIRROR_PROMPT.format(topic=topic)
    pair_id = f"{domain}_{pair_index:02d}"

    log.info("  [%s] Generating pair %s — %s", model_name, pair_id, topic)
    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "num_predict": 512},
    )
    raw = response["message"]["content"]
    parsed = _extract_json(raw)

    if parsed is None or "paragraph_a" not in parsed or "paragraph_b" not in parsed:
        log.warning("    Failed to parse JSON for %s — using raw fallback", pair_id)
        parts = re.split(r"[Pp]aragraph\s+[Bb]", raw, maxsplit=1)
        para_a = parts[0].replace("Paragraph A", "").strip().strip(":").strip()
        para_b = parts[1].strip().strip(":").strip() if len(parts) > 1 else raw.strip()
        parsed = {"paragraph_a": para_a, "paragraph_b": para_b}

    return [
        {
            "pair_id": pair_id,
            "label": 1,
            "domain": domain,
            "model_name": model_name,
            "text": parsed["paragraph_a"].strip(),
            "topic": topic,
        },
        {
            "pair_id": pair_id,
            "label": 0,
            "domain": domain,
            "model_name": model_name,
            "text": parsed["paragraph_b"].strip(),
            "topic": topic,
        },
    ]


# ---------------------------------------------------------------------------
# Opik logging
# ---------------------------------------------------------------------------


def log_to_opik(records: list[dict], dataset_name: str = "credibility_pairs_v1") -> bool:
    """
    Push the dataset to Opik. Returns True on success, False on failure.
    Opik must be configured (OPIK_API_KEY env var or ~/.opik.config).
    """
    try:
        import opik

        client = opik.Opik()
        dataset = client.get_or_create_dataset(
            name=dataset_name,
            description=(
                "N=100 Credibility Contrastive Dataset — Phase 1 (C1). "
                "25 pairs per domain (technical, financial, crisis, historical). "
                "Generated via the Mirror technique using Ollama. "
                "See docs/Spec 1 -- Credibility Contrastive Dataset.md"
            ),
        )
        items = [
            {
                "pair_id": r["pair_id"],
                "label": r["label"],
                "domain": r["domain"],
                "model_name": r["model_name"],
                "text": r["text"],
                "topic": r["topic"],
            }
            for r in records
        ]
        dataset.insert(items)
        log.info("Opik: logged %d records to dataset '%s'", len(items), dataset_name)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Opik logging skipped: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 (C1): Generate the N=100 Credibility Contrastive Dataset "
            "and deploy to Opik."
        )
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["anthropic", "ollama"],
        default="anthropic",
        help="Inference backend: 'anthropic' (fast, default) or 'ollama' (local)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model identifier. "
            "Anthropic default: value of ANTHROPIC_MODEL env var or 'claude-sonnet-4-5'. "
            "Ollama default: 'mistral:latest'."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent Anthropic API requests (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/credibility_pairs.jsonl",
        help="Output path for the JSONL dataset (default: data/credibility_pairs.jsonl)",
    )
    parser.add_argument(
        "--opik-dataset",
        type=str,
        default="credibility_pairs_v1",
        help="Opik dataset name (default: credibility_pairs_v1)",
    )
    parser.add_argument(
        "--skip-opik",
        action="store_true",
        help="Skip Opik logging (write JSONL only)",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve model name
    if args.model:
        model_name = args.model
    elif args.backend == "anthropic":
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    else:
        model_name = "mistral:latest"

    log.info("=== Phase 1 (C1): Credibility Contrastive Dataset Generation ===")
    log.info("Backend   : %s", args.backend)
    log.info("Model     : %s", model_name)
    log.info("Output    : %s", output_path)
    log.info("Domains   : %s", list(DOMAIN_TOPICS.keys()))
    log.info("Pairs/domain: 25  |  Total records: 200")

    all_records: list[dict] = []

    if args.backend == "anthropic":
        log.info("Concurrency: %d parallel requests", args.concurrency)
        all_records = asyncio.run(
            _generate_all_anthropic(
                model_name=model_name,
                concurrency=args.concurrency,
            )
        )
    else:
        # Ollama sync path
        import ollama as _ollama

        client = _ollama.Client(host=args.ollama_host)
        try:
            available = [m["model"] for m in client.list()["models"]]
            if model_name not in available:
                log.error("Model '%s' not found. Available: %s", model_name, available)
                sys.exit(1)
        except Exception as exc:
            log.error("Cannot connect to Ollama at %s: %s", args.ollama_host, exc)
            sys.exit(1)

        for domain, topics in DOMAIN_TOPICS.items():
            log.info("--- Domain: %s ---", domain)
            for idx, topic in enumerate(topics, start=1):
                try:
                    records = _generate_pair_ollama(
                        topic=topic,
                        domain=domain,
                        pair_index=idx,
                        model_name=model_name,
                        client=client,
                    )
                    all_records.extend(records)
                except Exception as exc:  # noqa: BLE001
                    log.error("  Error on %s/%02d (%s): %s", domain, idx, topic, exc)

    # Write JSONL
    with output_path.open("w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Wrote %d records to %s", len(all_records), output_path)

    # Validate
    pairs = len(all_records) // 2
    credible = sum(1 for r in all_records if r["label"] == 1)
    non_credible = sum(1 for r in all_records if r["label"] == 0)
    log.info(
        "Validation: %d pairs | %d credible | %d non-credible",
        pairs,
        credible,
        non_credible,
    )

    # Opik
    if not args.skip_opik:
        log_to_opik(all_records, dataset_name=args.opik_dataset)
    else:
        log.info("Opik logging skipped (--skip-opik).")

    log.info("=== Phase 1 (C1) complete ===")


if __name__ == "__main__":
    main()
