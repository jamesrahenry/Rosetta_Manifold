"""
generate_sentiment_dataset.py

Generate contrastive dataset for "sentiment polarity" concept testing.

Creates pairs of:
- Label 1: Positive emotional valence
- Label 0: Negative emotional valence

100 pairs (200 statements) across diverse domains.

Usage:
    python src/generate_sentiment_dataset.py --output data/sentiment_pairs.jsonl
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

# 100 topics across 10 domains (10 per domain)
DOMAIN_TOPICS = {
    "personal_experience": [
        "receiving unexpected good news",
        "dealing with a difficult setback",
        "reuniting with an old friend",
        "experiencing a major disappointment",
        "achieving a long-term goal",
        "facing a sudden loss",
        "discovering a new passion",
        "struggling with chronic stress",
        "celebrating a milestone",
        "navigating a painful conflict",
    ],
    "relationships": [
        "building trust with someone new",
        "experiencing betrayal in a friendship",
        "finding mutual understanding",
        "growing apart from a close friend",
        "deepening a romantic connection",
        "ending a long-term relationship",
        "feeling supported by family",
        "dealing with family tension",
        "making meaningful new connections",
        "feeling isolated and alone",
    ],
    "work_career": [
        "receiving recognition for your work",
        "facing criticism from a supervisor",
        "collaborating on a successful project",
        "dealing with workplace conflict",
        "advancing in your career path",
        "experiencing job insecurity",
        "finding purpose in your work",
        "feeling stuck in a dead-end job",
        "mentoring someone successfully",
        "struggling with work-life balance",
    ],
    "learning_growth": [
        "mastering a challenging new skill",
        "failing at something important",
        "having an intellectual breakthrough",
        "feeling overwhelmed by difficulty",
        "gaining valuable perspective",
        "repeating the same mistakes",
        "expanding your worldview",
        "feeling intellectually stagnant",
        "connecting disparate ideas",
        "struggling to understand concepts",
    ],
    "health_wellness": [
        "feeling physically energized",
        "dealing with persistent fatigue",
        "recovering from an illness",
        "facing a health diagnosis",
        "improving your fitness level",
        "experiencing chronic pain",
        "achieving mental clarity",
        "struggling with anxiety",
        "developing healthy habits",
        "battling with sleep issues",
    ],
    "creative_pursuits": [
        "completing a creative project",
        "experiencing creative block",
        "receiving positive feedback",
        "facing harsh criticism",
        "discovering artistic inspiration",
        "feeling creatively drained",
        "mastering a new technique",
        "struggling with self-doubt",
        "sharing work with others",
        "keeping work hidden away",
    ],
    "social_community": [
        "contributing to community improvement",
        "witnessing community decline",
        "organizing a successful event",
        "dealing with community conflict",
        "building neighborhood connections",
        "feeling disconnected from community",
        "volunteering for a cause",
        "observing social injustice",
        "creating inclusive spaces",
        "experiencing exclusion",
    ],
    "nature_environment": [
        "witnessing natural beauty",
        "seeing environmental destruction",
        "experiencing perfect weather",
        "enduring severe weather",
        "connecting with wildlife",
        "losing green spaces",
        "enjoying outdoor activities",
        "being confined indoors",
        "planting and growing things",
        "watching ecosystems deteriorate",
    ],
    "daily_life": [
        "enjoying a peaceful morning",
        "starting the day with chaos",
        "having a productive day",
        "wasting time on distractions",
        "preparing a satisfying meal",
        "dealing with cooking disasters",
        "organizing your living space",
        "feeling overwhelmed by clutter",
        "completing errands efficiently",
        "struggling with daily tasks",
    ],
    "future_outlook": [
        "feeling optimistic about possibilities",
        "dreading upcoming changes",
        "planning exciting future events",
        "fearing potential outcomes",
        "anticipating positive developments",
        "worrying about worst-case scenarios",
        "seeing growth opportunities",
        "expecting continued difficulties",
        "imagining fulfilling futures",
        "feeling trapped in current circumstances",
    ],
}

SENTIMENT_PROMPT = (
    'Generate a pair of short statements (2-3 sentences each) about "{topic}". '
    "Statement A must have POSITIVE emotional valence (optimistic, hopeful, uplifting). "
    "Statement B must have NEGATIVE emotional valence (pessimistic, discouraged, disheartening). "
    "Both should describe the same general situation but with opposite emotional tone. "
    "Keep them balanced in length and avoid extreme/exaggerated language.\n\n"
    "Respond ONLY with valid JSON in this exact format:\n"
    '{{"positive": "<positive statement>", "negative": "<negative statement>"}}'
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

    prompt = SENTIMENT_PROMPT.format(topic=topic)
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

        if not data or "positive" not in data or "negative" not in data:
            log.error("  [%s] Invalid JSON for %s", model_name, pair_id)
            return None

        # Create two records (positive=1, negative=0)
        positive_record = {
            "pair_id": pair_id,
            "label": 1,
            "domain": domain,
            "model_name": model_name,
            "text": data["positive"],
            "topic": topic,
        }

        negative_record = {
            "pair_id": pair_id,
            "label": 0,
            "domain": domain,
            "model_name": model_name,
            "text": data["negative"],
            "topic": topic,
        }

        log.info("  [%s] ✓ Generated pair %s", model_name, pair_id)
        return [positive_record, negative_record]

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
    """Generate full sentiment dataset via Anthropic-compatible API."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    log.info("=== Generating Sentiment Polarity Dataset ===")
    log.info("Model: %s", model_name)
    log.info("Output: %s", output_path)
    log.info("Target: 100 pairs (200 statements)")
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
    positive_count = sum(1 for r in all_records if r["label"] == 1)
    negative_count = sum(1 for r in all_records if r["label"] == 0)

    log.info("")
    log.info("=== Generation Complete ===")
    log.info("Total pairs: %d", len(all_records) // 2)
    log.info("Positive statements: %d", positive_count)
    log.info("Negative statements: %d", negative_count)
    log.info("Domains: %d", len(DOMAIN_TOPICS))
    log.info("Output: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate sentiment polarity dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="data/sentiment_pairs.jsonl",
        help="Output path (default: data/sentiment_pairs.jsonl)",
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
