"""
generate_dataset_tiny.py

Tiny PoC version: Generate 20 contrastive pairs (5 per domain) for laptop testing.

Usage:
    python src/generate_dataset_tiny.py
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Tiny dataset: 5 topics per domain = 20 pairs total
DOMAIN_TOPICS = {
    "technical": [
        "mRNA vaccine efficacy in immunocompromised patients",
        "CRISPR-Cas9 off-target editing rates",
        "quantum error correction thresholds",
        "large language model hallucination rates",
        "lithium-ion battery thermal runaway mechanisms",
    ],
    "financial": [
        "Federal Reserve interest rate decision impact on mortgage rates",
        "SEC enforcement actions against insider trading",
        "corporate earnings restatements and audit findings",
        "cryptocurrency exchange insolvency proceedings",
        "ESG fund performance versus benchmark indices",
    ],
    "crisis": [
        "wildfire evacuation orders in Northern California",
        "hurricane storm surge predictions for Gulf Coast",
        "earthquake early warning system activation thresholds",
        "chemical plant explosion emergency response protocols",
        "pandemic influenza hospital surge capacity planning",
    ],
    "historical": [
        "causes of the 1929 stock market crash",
        "primary sources on the Rwandan genocide",
        "archaeological evidence for the Bronze Age Collapse",
        "declassified documents on the Cuban Missile Crisis",
        "census records and the transatlantic slave trade",
    ],
}

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


def _extract_json(raw: str):
    """Extract JSON from response."""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


async def _generate_pair_anthropic(topic: str, domain: str, pair_index: int, model_name: str, client):
    """Generate one pair via Anthropic API."""
    from openai import RateLimitError

    prompt = MIRROR_PROMPT.format(topic=topic)
    pair_id = f"{domain}_{pair_index:02d}"

    log.info("  [%s] Generating pair %s — %s", model_name, pair_id, topic)

    try:
        response = await client.chat.completions.create(
            model=model_name,
            max_tokens=1024,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content
        parsed = _extract_json(raw)

        if parsed is None or "paragraph_a" not in parsed or "paragraph_b" not in parsed:
            log.warning("    Failed to parse JSON for %s", pair_id)
            return None

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
    except RateLimitError:
        log.warning("  Rate limited on %s — will retry", pair_id)
        await asyncio.sleep(62)
        return await _generate_pair_anthropic(topic, domain, pair_index, model_name, client)
    except Exception as e:
        log.error("  Error on %s: %s", pair_id, e)
        return None


async def generate_all(model_name: str):
    """Generate all 20 pairs."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("FUELIX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.fuelix.ai/v1")
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    all_records = []
    for domain, topics in DOMAIN_TOPICS.items():
        log.info("--- Domain: %s ---", domain)
        for idx, topic in enumerate(topics, start=1):
            result = await _generate_pair_anthropic(topic, domain, idx, model_name, client)
            if result:
                all_records.extend(result)
            await asyncio.sleep(0.5)  # Small delay between requests

    await client.close()
    return all_records


def main():
    parser = argparse.ArgumentParser(description="Generate tiny dataset (20 pairs)")
    parser.add_argument("--model", default="claude-sonnet-4-5", help="Model to use")
    parser.add_argument("--output", default="data/credibility_pairs_tiny.jsonl", help="Output path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=== Tiny PoC: Dataset Generation ===")
    log.info("Model: %s", args.model)
    log.info("Output: %s", output_path)
    log.info("Size: 20 pairs (5 per domain) = 40 records")

    all_records = asyncio.run(generate_all(args.model))

    # Write JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Wrote %d records to %s", len(all_records), output_path)

    # Validate
    pairs = len(all_records) // 2
    credible = sum(1 for r in all_records if r["label"] == 1)
    non_credible = sum(1 for r in all_records if r["label"] == 0)
    log.info("Validation: %d pairs | %d credible | %d non-credible", pairs, credible, non_credible)
    log.info("=== Tiny dataset generation complete ===")


if __name__ == "__main__":
    main()
