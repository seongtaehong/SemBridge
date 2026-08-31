"""External SemBridge usage example.

This script shows how a downstream user can adapt a sparse encoder's input
embeddings to a target tokenizer without bundling model weights in this source
package.

Example:
    python examples/sembridge_external_usage.py \
        --source-model naver/splade-v3 \
        --target-tokenizer monologg/kobigbird-bert-base \
        --bridge-model BAAI/bge-m3 \
        --output-dir ./sembridge_output/splade-v3-kor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sembridge import SemBridge
from sentence_transformers import SparseEncoder
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt sparse encoder embeddings with SemBridge.")
    parser.add_argument("--source-model", required=True, help="Source model id/path used for encoder weights")
    parser.add_argument("--source-tokenizer", help="Optional source tokenizer id/path; defaults to --source-model")
    parser.add_argument("--target-tokenizer", required=True, help="Target tokenizer id/path")
    parser.add_argument("--bridge-model", required=True, help="SentenceTransformer-compatible semantic bridge model")
    parser.add_argument("--output-dir", required=True, help="Directory to save the adapted sparse encoder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1024, help="Bridge embedding extraction batch size")
    parser.add_argument("--entmax-alpha", type=float, default=4.0, help="Entmax alpha for sparse source-token weights")
    parser.add_argument("--no-fuzzy-match", action="store_true", help="Disable fuzzy token overlap matching")
    parser.add_argument("--match-symbols", action="store_true", help="Copy overlapping numeric/symbol tokens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_tokenizer_id = args.source_tokenizer or args.source_model
    output_dir = Path(args.output_dir)

    source_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_id)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer)
    source_model = AutoModel.from_pretrained(args.source_model)

    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.unk_token

    target_embeddings = SemBridge(
        source_embeddings=source_model.get_input_embeddings().weight,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        bridge_model_name_or_path=args.bridge_model,
        entmax_alpha=args.entmax_alpha,
        exact_match_all=True,
        fuzzy_match_all=not args.no_fuzzy_match,
        match_symbols=args.match_symbols,
        batch_size=args.batch_size,
        device=args.device,
        verbosity="info",
    )

    sparse_encoder = SparseEncoder(args.source_model)
    sparse_encoder[0].tokenizer = target_tokenizer
    sparse_encoder[0].auto_model.resize_token_embeddings(len(target_tokenizer))

    with torch.no_grad():
        sparse_encoder[0].auto_model.get_input_embeddings().weight.copy_(target_embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_encoder.save_pretrained(str(output_dir))
    target_tokenizer.save_pretrained(str(output_dir))
    print(f"Saved adapted sparse encoder to: {output_dir}")


if __name__ == "__main__":
    main()
