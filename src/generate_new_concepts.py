"""
generate_new_concepts.py

Generate contrastive datasets for five new CAZ concepts:
  - certainty:      "I know X" vs "I'm not sure X" (epistemic, distinct from credibility)
  - plurality:      plural vs singular statements (syntactic, simpler than negation)
  - causation:      "X caused Y" vs "X and Y are unrelated" (relational)
  - moral_valence:  morally good/right vs morally bad/wrong (affective, distinct from sentiment)
  - temporal_order: "X happened before Y" vs "Y happened before X" (relational/syntactic)

All generate 100 pairs (25 topics × 4 domains) via claude-sonnet-4-5 on FuelIX.
Rate-limited to ~40 RPM via semaphore (8 concurrent × 1.2s delay).

Usage:
    python src/generate_new_concepts.py --concept certainty --output data/certainty_pairs.jsonl
    python src/generate_new_concepts.py --all   # generates all 5 concepts sequentially
"""

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Shared topics (25 per domain = 100 pairs per concept) ───────────────────

DOMAIN_TOPICS = {
    "science": [
        "the relationship between exercise and cardiovascular health",
        "the role of RNA in protein synthesis",
        "the effects of sleep deprivation on cognitive performance",
        "the impact of deforestation on local rainfall patterns",
        "the connection between gut bacteria and immune function",
        "the behavior of electrons in superconductors",
        "the role of dopamine in reward processing",
        "the effects of UV radiation on DNA",
        "the relationship between diet and cancer risk",
        "the function of the blood-brain barrier",
        "the impact of rising CO2 on ocean pH",
        "the role of telomeres in cellular aging",
        "the mechanism of viral replication",
        "the connection between stress and cortisol levels",
        "the effects of magnetic fields on bird navigation",
        "the role of platelets in wound healing",
        "the relationship between light exposure and circadian rhythms",
        "the impact of noise pollution on marine mammals",
        "the function of myelin in nerve signal transmission",
        "the effects of altitude on oxygen absorption",
        "the role of insulin in glucose metabolism",
        "the connection between air quality and cognitive decline",
        "the impact of microbiome diversity on mood",
        "the function of mirror neurons in empathy",
        "the relationship between inflammation and chronic disease",
    ],
    "everyday": [
        "whether regular walks improve mood",
        "whether reading before bed aids sleep",
        "whether handwriting notes improves retention",
        "whether cold showers increase alertness",
        "whether breakfast improves morning productivity",
        "whether plants in offices reduce stress",
        "whether decluttering improves focus",
        "whether meal prepping saves money",
        "whether gratitude journaling affects wellbeing",
        "whether standing desks reduce back pain",
        "whether listening to music aids concentration",
        "whether napping improves afternoon performance",
        "whether cooking at home is healthier than eating out",
        "whether social media use increases loneliness",
        "whether spending time in nature lowers anxiety",
        "whether volunteering improves life satisfaction",
        "whether learning a new skill builds confidence",
        "whether morning routines increase productivity",
        "whether limiting screen time improves attention",
        "whether consistent sleep schedules improve mood",
        "whether pet ownership reduces stress",
        "whether face-to-face conversation deepens relationships",
        "whether financial planning reduces anxiety",
        "whether physical activity improves creativity",
        "whether adequate hydration improves concentration",
    ],
    "history": [
        "the printing press transforming literacy in Europe",
        "the Black Death reshaping European social structures",
        "the Industrial Revolution driving urbanization",
        "colonialism shaping modern economic inequalities",
        "the French Revolution inspiring democratic movements globally",
        "the transatlantic slave trade affecting African demographics",
        "the Cold War accelerating space exploration technology",
        "the Silk Road enabling cultural exchange between East and West",
        "the Renaissance reviving classical learning in Europe",
        "the invention of writing enabling complex civilizations",
        "the Great Depression reshaping social welfare policy",
        "the agricultural revolution enabling permanent settlements",
        "the Reformation fragmenting Christian Europe",
        "the invention of gunpowder changing the nature of warfare",
        "the Enlightenment inspiring scientific institutions",
        "World War I ending the European imperial order",
        "the invention of the printing press democratizing knowledge",
        "the Roman Empire's fall transforming European culture",
        "maritime exploration enabling global trade networks",
        "the abolitionist movement transforming legal norms",
        "the Green Revolution preventing famine in Asia",
        "the fall of the Berlin Wall reshaping European politics",
        "the telegraph accelerating global communication",
        "the women's suffrage movement transforming electoral systems",
        "the Magna Carta establishing limits on royal power",
    ],
    "technology": [
        "smartphones changing how people navigate cities",
        "social media algorithms influencing political opinions",
        "encryption protecting financial transactions online",
        "GPS technology transforming logistics and delivery",
        "automation displacing manufacturing jobs",
        "the internet enabling remote work at scale",
        "machine learning improving medical imaging diagnostics",
        "electric vehicles reducing urban air pollution",
        "streaming services changing how music is consumed",
        "cloud computing enabling startup scalability",
        "open-source software accelerating software development",
        "e-commerce transforming retail supply chains",
        "telemedicine expanding healthcare access in rural areas",
        "robotics improving precision in surgery",
        "5G enabling real-time industrial automation",
        "digital payments reducing cash transactions globally",
        "facial recognition raising privacy concerns",
        "blockchain enabling transparent financial records",
        "renewable energy storage improving grid reliability",
        "AI-generated content challenging content moderation",
        "video conferencing changing how teams collaborate",
        "biometric authentication replacing passwords",
        "recommendation systems shaping consumer choices",
        "self-driving vehicles changing urban planning assumptions",
        "digital twins enabling predictive infrastructure maintenance",
    ],
}

# ─── Concept prompts ──────────────────────────────────────────────────────────

PROMPTS = {
    "certainty": """Generate a contrastive pair of statements about the following topic that differ ONLY in epistemic certainty — one expressing high confidence, one expressing genuine uncertainty. The topic should be the same; only the speaker's certainty should differ.

Topic: {topic}

Requirements:
- HIGH CERTAINTY (label 1): Speaker expresses clear confidence — uses phrases like "I know," "it's clear that," "the evidence shows," "I'm certain," "definitely," "without doubt"
- LOW CERTAINTY (label 0): Speaker expresses genuine uncertainty — uses phrases like "I'm not sure," "I think but I might be wrong," "it's unclear," "the evidence is mixed," "I'm uncertain whether"
- Both statements should be about the SAME topic
- 2-4 sentences each
- Natural, varied phrasing — not formulaic

Return JSON only:
{{"certain": "...", "uncertain": "..."}}""",
    "plurality": """Generate a contrastive pair of statements about the following topic where one uses plural constructions and one uses singular constructions throughout. The underlying content should be equivalent.

Topic: {topic}

Requirements:
- PLURAL (label 1): Uses plural subjects, objects, and verbs throughout ("researchers have found," "the studies show," "these effects are," "they demonstrate")
- SINGULAR (label 0): Uses singular subjects, objects, and verbs throughout ("the researcher has found," "the study shows," "this effect is," "it demonstrates")
- Same factual content, same domain, just systematically different grammatical number
- 2-3 sentences each

Return JSON only:
{{"plural": "...", "singular": "..."}}""",
    "causation": """Generate a contrastive pair of statements about the following topic where one asserts a causal relationship and one denies or avoids it.

Topic: {topic}

Requirements:
- CAUSAL (label 1): Explicitly asserts that one thing caused, produced, led to, or resulted in another — uses causal language ("caused," "led to," "resulted in," "produced," "drove," "because of")
- NON-CAUSAL (label 0): Describes the same elements as correlated, coincidental, or unrelated — uses non-causal language ("coincided with," "occurred alongside," "was observed at the same time as," "appears unrelated to," "does not cause")
- Both statements should involve the same general topic and elements
- 2-4 sentences each

Return JSON only:
{{"causal": "...", "non_causal": "..."}}""",
    "moral_valence": """Generate a contrastive pair of statements about the following topic where one makes a positive moral judgment and one makes a negative moral judgment. This is about MORAL evaluation (right/wrong, ethical/unethical), NOT about whether something feels good or bad emotionally.

Topic: {topic}

Requirements:
- MORALLY POSITIVE (label 1): Evaluates the topic as morally good, right, ethical, virtuous, or praiseworthy ("morally right," "the ethical choice," "we have a duty to," "it is right that," "commendable," "virtuous")
- MORALLY NEGATIVE (label 0): Evaluates the topic as morally wrong, unethical, blameworthy, or problematic ("morally wrong," "unethical," "we should not," "it is wrong to," "irresponsible," "harmful")
- Focus on moral/ethical evaluation, not emotional tone or factual content
- 2-4 sentences each

Return JSON only:
{{"moral_positive": "...", "moral_negative": "..."}}""",
    "temporal_order": """Generate a contrastive pair of statements about the following topic where one asserts that A happened before B, and the other asserts that B happened before A (or that there is no clear temporal order).

Topic: {topic}

Requirements:
- FORWARD ORDER (label 1): Clearly states that one development, event, or element preceded or led to another ("first... then," "before... came," "preceded," "prior to," "initially... subsequently," "the earlier... the later")
- REVERSED/UNCLEAR ORDER (label 0): States the opposite temporal sequence OR states that the order is unclear or simultaneous ("simultaneously," "it is unclear which came first," "the order is disputed," OR reverses which came first)
- Both statements should involve the same general domain and elements
- 2-4 sentences each

Return JSON only:
{{"forward_order": "...", "reversed_order": "..."}}""",
}

# ─── Label mappings ───────────────────────────────────────────────────────────

LABEL_KEYS = {
    "certainty": ("certain", "uncertain", 1, 0),
    "plurality": ("plural", "singular", 1, 0),
    "causation": ("causal", "non_causal", 1, 0),
    "moral_valence": ("moral_positive", "moral_negative", 1, 0),
    "temporal_order": ("forward_order", "reversed_order", 1, 0),
}

# ─── Generation ──────────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    for start, end in [(text.find("{"), text.rfind("}") + 1)]:
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


async def _generate_pair(
    concept: str,
    topic: str,
    domain: str,
    pair_index: int,
    model_name: str,
    client,
    semaphore: asyncio.Semaphore,
) -> list[dict] | None:
    from openai import RateLimitError

    prompt_template = PROMPTS[concept]
    prompt = prompt_template.format(topic=topic)
    pair_id = f"{domain}_{pair_index:02d}"
    key_pos, key_neg, lbl_pos, lbl_neg = LABEL_KEYS[concept]

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    max_tokens=600,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.choices[0].message.content
                data = _extract_json(text)

                if not data or key_pos not in data or key_neg not in data:
                    log.error("  Invalid JSON for %s/%s", concept, pair_id)
                    return None

                pos_record = {
                    "pair_id": pair_id,
                    "label": lbl_pos,
                    "domain": domain,
                    "model_name": model_name,
                    "text": data[key_pos],
                    "topic": topic,
                    "concept": concept,
                }
                neg_record = {
                    "pair_id": pair_id,
                    "label": lbl_neg,
                    "domain": domain,
                    "model_name": model_name,
                    "text": data[key_neg],
                    "topic": topic,
                    "concept": concept,
                }
                log.info("  ✓ %s/%s", concept, pair_id)
                await asyncio.sleep(1.2)
                return [pos_record, neg_record]

            except RateLimitError:
                wait = 30 * (attempt + 1)
                log.warning(
                    "  Rate limited on %s/%s, waiting %ds...", concept, pair_id, wait
                )
                await asyncio.sleep(wait)
            except Exception as e:
                log.error(
                    "  Error on %s/%s attempt %d: %s", concept, pair_id, attempt + 1, e
                )
                if attempt < 2:
                    await asyncio.sleep(5)

        return None


async def generate_concept(
    concept: str,
    output_path: Path,
    model_name: str,
    api_key: str,
    base_url: str,
    max_concurrent: int = 8,
) -> int:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(max_concurrent)

    log.info("=== Generating: %s ===", concept)
    log.info("Output: %s", output_path)
    log.info("Concurrency: %d (~40 RPM)", max_concurrent)

    tasks = [
        _generate_pair(concept, topic, domain, idx, model_name, client, semaphore)
        for domain, topics in DOMAIN_TOPICS.items()
        for idx, topic in enumerate(topics, 1)
    ]

    results = await asyncio.gather(*tasks)

    records = []
    for r in results:
        if r:
            records.extend(r)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    pairs = len(records) // 2
    log.info("=== %s complete: %d pairs written to %s ===", concept, pairs, output_path)
    return pairs


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concept", choices=list(PROMPTS.keys()), help="Single concept to generate"
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate all concepts sequentially"
    )
    parser.add_argument(
        "--output-dir", default="data", help="Output directory (default: data/)"
    )
    parser.add_argument("--model", default="claude-sonnet-4-5")
    parser.add_argument("--api-key", default=os.environ.get("FUELIX_API_KEY"))
    parser.add_argument("--base-url", default="https://api.fuelix.ai/v1")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("FUELIX_API_KEY not set")

    output_dir = Path(args.output_dir)
    concepts = list(PROMPTS.keys()) if args.all else [args.concept]

    if not concepts or concepts == [None]:
        parser.error("Specify --concept or --all")

    for concept in concepts:
        output_path = output_dir / f"{concept}_pairs.jsonl"
        pairs = asyncio.run(
            generate_concept(
                concept=concept,
                output_path=output_path,
                model_name=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
            )
        )
        print(f"{concept}: {pairs} pairs → {output_path}")
