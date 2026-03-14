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

# Topics across different domains — 25 per domain = 100 pairs total
DOMAIN_TOPICS = {
    "science": [
        "the effects of climate change on polar ice caps",
        "the role of gut microbiome in human health",
        "the discovery of gravitational waves",
        "the function of CRISPR gene editing",
        "the behavior of dark matter in galaxies",
        "the impact of microplastics on marine ecosystems",
        "the relationship between sleep and memory consolidation",
        "the role of mitochondria in cellular energy production",
        "the effects of deforestation on biodiversity",
        "the mechanisms of antibiotic resistance",
        "the role of vaccines in herd immunity",
        "the greenhouse effect and global temperature",
        "the function of neurons in the brain",
        "the impact of ocean acidification on coral reefs",
        "the role of photosynthesis in oxygen production",
        "the connection between air pollution and respiratory disease",
        "the genetic basis of inherited traits",
        "the behavior of black holes near event horizons",
        "the role of stem cells in tissue repair",
        "the impact of invasive species on native ecosystems",
        "the effects of radiation on living tissue",
        "the role of enzymes in digestion",
        "the relationship between exercise and mental health",
        "the impact of soil composition on plant growth",
        "the function of the immune system in fighting infection",
    ],
    "everyday": [
        "the importance of sleep for cognitive function",
        "the benefits of regular exercise",
        "the value of learning multiple languages",
        "the impact of social media on relationships",
        "the necessity of breakfast for productivity",
        "the effect of hydration on physical performance",
        "the role of reading in vocabulary development",
        "the benefits of spending time in nature",
        "the impact of screen time on children",
        "the value of maintaining a daily routine",
        "the effect of caffeine on alertness",
        "the importance of face-to-face communication",
        "the benefits of cooking meals at home",
        "the impact of clutter on mental clarity",
        "the value of setting daily goals",
        "the effect of music on mood and focus",
        "the importance of regular dental hygiene",
        "the benefits of journaling for self-reflection",
        "the impact of commuting on quality of life",
        "the value of hobbies for stress reduction",
        "the effect of cold showers on energy levels",
        "the importance of sunlight exposure for wellbeing",
        "the benefits of volunteering in local communities",
        "the impact of financial literacy on life outcomes",
        "the value of asking for help when needed",
    ],
    "history": [
        "the causes of World War I",
        "the influence of the printing press on society",
        "the role of women in the Industrial Revolution",
        "the impact of the Silk Road on trade",
        "the significance of the Renaissance",
        "the causes of the fall of the Roman Empire",
        "the role of colonialism in shaping modern borders",
        "the impact of the Black Death on medieval Europe",
        "the significance of the Magna Carta",
        "the influence of the French Revolution on democracy",
        "the role of the transatlantic slave trade in economic history",
        "the impact of the moon landing on public imagination",
        "the significance of the Berlin Wall's construction",
        "the role of propaganda in World War II",
        "the influence of the Cold War on space exploration",
        "the impact of the Great Depression on global economies",
        "the role of trade unions in labor rights",
        "the significance of the invention of writing",
        "the influence of ancient Greek philosophy on Western thought",
        "the impact of the agricultural revolution on human settlement",
        "the role of the Enlightenment in shaping modern science",
        "the significance of the American Civil War for civil rights",
        "the influence of the Ottoman Empire on Middle Eastern culture",
        "the impact of gunpowder on warfare",
        "the role of maritime exploration in globalization",
    ],
    "technology": [
        "the reliability of facial recognition systems",
        "the security of blockchain technology",
        "the efficiency of electric vehicles",
        "the accuracy of machine learning predictions",
        "the usefulness of virtual reality in education",
        "the impact of social media algorithms on news consumption",
        "the role of automation in manufacturing jobs",
        "the effectiveness of encryption in protecting data",
        "the reliability of self-driving vehicle systems",
        "the impact of cloud computing on data storage",
        "the role of smartphones in daily productivity",
        "the effectiveness of content moderation on online platforms",
        "the impact of 5G networks on communication speed",
        "the role of open-source software in innovation",
        "the effectiveness of antivirus software against modern threats",
        "the impact of e-commerce on traditional retail",
        "the role of GPS technology in navigation accuracy",
        "the effectiveness of renewable energy storage solutions",
        "the impact of telemedicine on healthcare access",
        "the role of robotics in surgical precision",
        "the effectiveness of digital payments in reducing fraud",
        "the impact of streaming services on traditional broadcasting",
        "the role of biometric authentication in security",
        "the effectiveness of AI in medical diagnosis",
        "the impact of remote work technology on team collaboration",
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


async def _generate_pair_anthropic(
    topic: str,
    domain: str,
    pair_index: int,
    model_name: str,
    client,
    semaphore: asyncio.Semaphore,
):
    """Generate one pair via Anthropic API, rate-limited by semaphore."""
    from openai import RateLimitError

    prompt = NEGATION_PROMPT.format(topic=topic)
    pair_id = f"{domain}_{pair_index:02d}"

    async with semaphore:
        log.info("  [%s] Generating pair %s — %s", model_name, pair_id, topic)

        for attempt in range(3):
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
                # Small inter-request delay to stay under 60 RPM
                await asyncio.sleep(1.2)
                return [affirmative_record, negated_record]

            except RateLimitError:
                wait = 30 * (attempt + 1)
                log.warning(
                    "  [%s] Rate limited on %s, waiting %ds...",
                    model_name,
                    pair_id,
                    wait,
                )
                await asyncio.sleep(wait)
            except Exception as exc:
                log.error(
                    "  [%s] Failed on %s (attempt %d): %s",
                    model_name,
                    pair_id,
                    attempt + 1,
                    exc,
                )
                if attempt < 2:
                    await asyncio.sleep(5)

        return None


async def generate_dataset_anthropic(
    model_name: str,
    output_path: Path,
    api_key: str,
    base_url: str,
    max_concurrent: int = 8,
):
    """Generate full negation dataset via Anthropic-compatible API.

    max_concurrent: number of simultaneous requests (8 × ~1.2s delay ≈ 40 RPM,
    safely under the 60 RPM FuelIX limit).
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(max_concurrent)

    log.info("=== Generating Negation Dataset ===")
    log.info("Model: %s", model_name)
    log.info("Output: %s", output_path)
    log.info("Concurrency: %d requests (target <60 RPM)", max_concurrent)
    log.info("")

    tasks = []
    for domain, topics in DOMAIN_TOPICS.items():
        for idx, topic in enumerate(topics, 1):
            task = _generate_pair_anthropic(
                topic, domain, idx, model_name, client, semaphore
            )
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
    api_key = (
        args.api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        args.base_url or os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    )

    if not api_key:
        log.error(
            "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable"
        )
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
