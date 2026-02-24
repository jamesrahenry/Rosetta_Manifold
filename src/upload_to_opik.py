"""
upload_to_opik.py

Phase 1 (C1) — Opik Dataset Upload

Reads the generated credibility_pairs.jsonl and pushes it to an Opik dataset.
Run this after configuring Opik:

    # Option A — Opik Cloud:
    opik configure --api_key <YOUR_OPIK_API_KEY>

    # Option B — Local Opik instance:
    opik configure --use_local --url http://localhost:5173

Then run:
    python src/upload_to_opik.py [--input data/credibility_pairs.jsonl]

See: docs/Spec 1 -- Credibility Contrastive Dataset.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload credibility_pairs.jsonl to an Opik dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/credibility_pairs.jsonl",
        help="Path to the JSONL dataset (default: data/credibility_pairs.jsonl)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="credibility_pairs_v1",
        help="Opik dataset name (default: credibility_pairs_v1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        log.error("Dataset not found: %s — run generate_dataset.py first", input_path)
        sys.exit(1)

    records = []
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))

    log.info("Loaded %d records from %s", len(records), input_path)

    try:
        import opik

        client = opik.Opik()
        dataset = client.get_or_create_dataset(
            name=args.dataset_name,
            description=(
                "N=100 Credibility Contrastive Dataset — Phase 1 (C1). "
                "25 pairs per domain (technical, financial, crisis, historical). "
                "Generated via the Mirror technique using claude-sonnet-4-5 via Fuelix. "
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
        log.info(
            "Opik: logged %d records to dataset '%s'", len(items), args.dataset_name
        )
        log.info("=== Phase 1 (C1) Opik upload complete ===")
    except Exception as exc:  # noqa: BLE001
        log.error("Opik upload failed: %s", exc)
        log.error(
            "Configure Opik first:\n"
            "  Cloud:  opik configure --api_key <YOUR_KEY>\n"
            "  Local:  opik configure --use_local --url http://localhost:5173"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
