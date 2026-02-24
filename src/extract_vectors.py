"""
extract_vectors.py

Phase 2 (C2 Extraction): Compute V_cred across Llama 3, Mistral, and Qwen
using Difference-of-Means on residual stream activations.

Usage:
    python src/extract_vectors.py --model llama3:8b

See: docs/Spec 2 -- Vector Extraction & Alignment Pipeline.md
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract credibility vectors from a local Ollama model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ollama model string, e.g. llama3:8b, mistral:7b, qwen:7b",
    )
    parser.add_argument(
        "--layer-start",
        type=int,
        default=14,
        help="First layer index to capture residual stream activations (default: 14)",
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=22,
        help="Last layer index to capture residual stream activations (default: 22)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/credibility_pairs.jsonl",
        help="Path to the contrastive prompt-pair dataset (JSONL).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[extract_vectors] Model: {args.model}")
    print(f"[extract_vectors] Hook layers: {args.layer_start}–{args.layer_end}")
    print(f"[extract_vectors] Dataset: {args.data_path}")
    # TODO: implement extraction pipeline per Spec 2


if __name__ == "__main__":
    main()
