# 🌉 SemBridge

### Language Transfer in Sparse Encoders via Multilingual Semantic Bridges

📄 **Paper:** [arXiv:2605.26002](https://arxiv.org/abs/2605.26002)

🎉 **Accepted to the EMNLP 2026 Main Conference**

SemBridge enables language transfer for English-centric sparse encoders by
aligning source and target vocabularies through multilingual dense embedding
models. It reconstructs target-language token embeddings from a small set of
semantically related source tokens, filtering semantic noise while improving
convergence, training efficiency, and multilingual retrieval.

## 🚀 Installation

```bash
git clone https://github.com/seongtaehong/SemBridge.git
cd SemBridge
python -m pip install -e .
```

## 🧩 Library usage

```python
from sembridge import SemBridge

target_embeddings = SemBridge(
    source_embeddings=source_model.get_input_embeddings().weight,
    source_tokenizer=source_tokenizer,
    target_tokenizer=target_tokenizer,
    bridge_model_name_or_path="BAAI/bge-m3",
    entmax_alpha=4.0,
    exact_match_all=True,
    fuzzy_match_all=True,
    match_symbols=True,
    batch_size=1024,
    device="cuda",
)
```

`bridge_model_name_or_path` is required and must identify a
SentenceTransformer-compatible model. `entmax_alpha` is configurable and
defaults to `2.5`; the examples in this repository explicitly use `4.0`.

## 🧪 Example

```bash
python examples/sembridge_external_usage.py \
  --source-model naver/splade-v3 \
  --target-tokenizer monologg/kobigbird-bert-base \
  --bridge-model BAAI/bge-m3 \
  --entmax-alpha 4.0 \
  --output-dir ./sembridge_output/splade-v3-kor
```

The repository contains source code only. Model weights, checkpoints, datasets,
experiment logs, results, and caches are intentionally excluded.

## 📁 Repository layout

```text
.
├── examples/       # External usage example
├── sembridge/      # Python package
├── pyproject.toml  # Build and tool configuration
└── setup.cfg       # Package metadata and dependencies
```

See [NOTICE.md](NOTICE.md) for provenance and redistribution information.

## 🙏 Acknowledgements

This implementation builds upon and adapts components from
[FOCUS](https://github.com/konstantinjdobler/focus), licensed under the MIT
License.
