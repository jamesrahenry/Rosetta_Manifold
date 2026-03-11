"""
generate_negation_dataset.py

Generate contrastive dataset for "negation" concept testing.

Creates pairs of:
- Label 1: Affirmative statements (positive assertions)
- Label 0: Negated statements (explicit negation markers)

Tests whether CAZ boundaries are concept-specific by comparing to credibility.

Usage:
    python src/generate_negation_dataset.py --output data/negation_pairs.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Topics across different domains
DOMAIN_TOPICS = {
    "science": [
        "the effects of climate change on polar ice caps",
        "the role of gut microbiome in human health",
        "the discovery of gravitational waves",
        "the function of CRISPR gene editing",
        "the behavior of dark matter in galaxies",
    ],
    "everyday": [
        "the importance of sleep for cognitive function",
        "the benefits of regular exercise",
        "the value of learning multiple languages",
        "the impact of social media on relationships",
        "the necessity of breakfast for productivity",
    ],
    "history": [
        "the causes of World War I",
        "the influence of the printing press on society",
        "the role of women in the Industrial Revolution",
        "the impact of the Silk Road on trade",
        "the significance of the Renaissance",
    ],
    "technology": [
        "the reliability of facial recognition systems",
        "the security of blockchain technology",
        "the efficiency of electric vehicles",
        "the accuracy of machine learning predictions",
        "the usefulness of virtual reality in education",
    ],
}

NEGATION_PROMPT = (
    'Generate a pair of short statements about "{topic}". '
    "Statement A must be AFFIRMATIVE (positive assertion with no negation). "
    "Statement B must be NEGATED (same topic but using explicit negation like 'not', 'no', 'never', 'don\\'t', etc.). "
    "Keep both statements factual and balanced (not obviously true or false). "
    "Make them roughly the same length (2-4 sentences each).\n\n"
    "Respond ONLY with valid JSON in this exact format:\n"
    '{{"affirmative": "<positive statement>", "negated": "<negated statement>"}}'
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

    prompt = NEGATION_PROMPT.format(topic=topic)
    pair_id = f"{domain}_{pair_index:02d}"

    log.info("  [%s] Generating pair %s — %s", model_name, pair_id, topic)

    try:
        response = await client.chat.completions.create(
            model=model_name,
            max_tokens=512,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content
        data = _extract_json(text)

        if not data or "affirmative" not in data or "negated" not in data:
            log.error("  [%s] Invalid JSON for %s", model_name, pair_id)
            return None

        # Create two records (affirmative=1, negated=0)
        affirmative_record = {
            "pair_id": pair_id,
            "label": 1,
            "domain": domain,
            "model_name": model_name,
            "text": data["affirmative"],
            "topic": topic,
        }

        negated_record = {
            "pair_id": pair_id,
            "label": 0,
            "domain": domain,
            "model_name": model_name,
            "text": data["negated"],
            "topic": topic,
        }

        log.info("  [%s] ✓ Generated pair %s", model_name, pair_id)
        return [affirmative_record, negated_record]

    except RateLimitError:
        log.warning("  [%s] Rate limited on %s, retrying...", model_name, pair_id)
        await asyncio.sleep(2)
        return await _generate_pair_anthropic(topic, domain, pair_index, model_name, client)
    except Exception as exc:
        log.error("  [%s] Failed on %s: %s", model_name, pair_id, exc)
        return None


async def generate_dataset_anthropic(
    model_name: str,
    output_path: Path,
    api_key: str,
    base_url: str,
):
    """Generate full negation dataset via Anthropic-compatible API."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    log.info("=== Generating Negation Dataset ===")
    log.info("Model: %s", model_name)
    log.info("Output: %s", output_path)
    log.info("")

    tasks = []
    for domain, topics in DOMAIN_TOPICS.items():
        for idx, topic in enumerate(topics, 1):
            task = _generate_pair_anthropic(topic, domain, idx, model_name, client)
            tasks.append(task)

    results = await asyncio.gather(*tasks)

    # Flatten and filter
    all_records = []
    for result in results:
        if result:
            all_records.extend(result)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    # Report
    affirmative_count = sum(1 for r in all_records if r["label"] == 1)
    negated_count = sum(1 for r in all_records if r["label"] == 0)

    log.info("")
    log.info("=== Generation Complete ===")
    log.info("Total pairs: %d", len(all_records) // 2)
    log.info("Affirmative statements: %d", affirmative_count)
    log.info("Negated statements: %d", negated_count)
    log.info("Output: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate negation dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="data/negation_pairs.jsonl",
        help="Output path (default: data/negation_pairs.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5",
        help="Model to use (default: claude-sonnet-4-5)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: from ANTHROPIC_API_KEY env)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL (default: from ANTHROPIC_BASE_URL env or Anthropic default)",
    )

    args = parser.parse_args()

    # Get API credentials
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        log.error("No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
        return

    if not base_url:
        log.warning("No base URL specified, using default Anthropic endpoint")
        base_url = "https://api.anthropic.com/v1"

    output_path = Path(args.output)

    asyncio.run(
        generate_dataset_anthropic(
            model_name=args.model,
            output_path=output_path,
            api_key=api_key,
            base_url=base_url,
        )
    )


if __name__ == "__main__":
    main()
