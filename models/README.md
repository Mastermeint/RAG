# Embedding models

This folder holds the local embedding model used for meaning-based (dense) and hybrid search. The model files are
large (the default is 556 MB) and are **not** stored in Git: everything here except this README is ignored.

```powershell
uv run rag-embed download   # fetch the default model into models/gte-base-en-v1.5/ and check every SHA-256
uv run rag-embed build      # embed every chunk once and save the vectors (data/processed/embeddings/)
uv run rag-embed status     # how many chunks still lack a saved vector
```

Set `RAG_MODEL_DIR` to use a model folder somewhere else (for example on a shared drive).

## Default model: `gte-base-en-v1.5`

| | |
|---|---|
| Model | [Alibaba-NLP/gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5), Apache-2.0 |
| Size | 137 M parameters, 768-dimensional vectors, fp32 ONNX file of 556 MB |
| Context | up to 8,192 tokens (we truncate at 1,024; the longest chunk is about 480 words) |
| Pooling | first token (`[CLS]`), vectors L2-normalised; no query or passage prefix needed |
| Speed | about 0.45 s per chunk on a laptop CPU (longer chunks cost more): the first build of the 6,542 chunks took 49 minutes; later builds only embed new or changed chunks. A search takes about 4 s, mostly loading the model |
| Downloaded from | GitHub release of the Magda project: `https://github.com/magda-io/embedding_models/releases/tag/gte-base-en-v1.5` |
| Runtime | `onnxruntime` + `tokenizers` (both from PyPI); no PyTorch, no internet at search time |

Pinned SHA-256 values (checked by `rag-embed download`, also listed in `src/rag_mvp/embeddings.py`):

| File | SHA-256 |
|---|---|
| `model.zip` (release asset) | `55dfd0ae0b9140cf091a3ee26fe0542e00807c3827687ac2e66da27c12f01084` |
| `model.onnx` | `d908f84171c304a863061cde3eaf29f9f7ec9206068f9b3a5b97323fa7b2811e` |
| `tokenizer.json` | `cb374d6bc042c22455946f4e09a89d29882a199fdaf8fb25be00dc8b8857a448` |
| `config.json` | `238854682d429ce12c49ed3d1a31b6d1b9622d2afbcf50a17243da18926cbcab` |

### Why this model

- **Measured on our books:** as the default `dense` search it raised dev Recall@10 from 0.54 (BM25) to 0.78 and
  paraphrase Recall@10 from 0.00 to 0.57; on the fresh holdout2 split Recall@10 went from 0.66 to 0.91 (see the
  root README, "Search modes").
- **Strong retrieval quality for its size.** On the public MTEB retrieval benchmark gte-base-en-v1.5 scores clearly
  above the small models (bge-small, all-MiniLM) and close to models three times larger.
- **Runs on a laptop CPU** through ONNX Runtime, with no GPU, no PyTorch and no service to host.
- **Long context** (8,192 tokens), so no chunk is ever cut off.
- **Reachable on this network.** huggingface.co, its mirrors, ModelScope, Ollama and Kaggle are blocked by the corporate
  firewall (TLS handshake refused); GitHub and PyPI are allowed. The Magda project (an open-source data-catalogue
  project, organisation account, Apache-2.0) publishes ONNX exports of gte-base and gte-large as GitHub release assets.

### Trust and risk

- The file comes from a **mirror, not the original authors**, so its hash was pinned on first download (2026-11-19)
  rather than checked against the official one. When huggingface.co becomes reachable, compare `model.onnx` with the
  official ONNX export, or re-export it from the official weights.
- An ONNX file is a computation graph plus numbers. Unlike PyTorch `.bin`/pickle files it cannot run arbitrary Python
  code when loaded. The realistic worst case for a bad mirror is poor search quality, which the evaluation would show.
- The model was sanity-checked: for "Why is rising carbon dioxide making it harder for animals with chalky housings to
  grow?" it ranks an ocean-acidification passage first (cosine 0.58, unrelated text about 0.37). A test repeats this
  check whenever the model is installed.

## Alternatives to consider (most preferred first, for when IT approves huggingface.co)

| Model | Size | Why consider it | Trade-off |
|---|---|---|---|
| `Alibaba-NLP/gte-base-en-v1.5` from the official repository | same | same model, from the authors, removes the mirror caveat | needs huggingface.co |
| `BAAI/bge-base-en-v1.5` | 110 M, 768-d | very widely used, similar quality, 512 tokens | needs a query prefix ("Represent this sentence for searching relevant passages: ") |
| `Snowflake/snowflake-arctic-embed-m-v1.5` | 110 M, 768-d | strong retrieval scores, vectors can be shortened to 256-d | needs a query prefix |
| `nomic-ai/nomic-embed-text-v1.5` | 137 M, 768-d | open training data, 8,192 tokens, shortenable vectors | needs `search_query:` / `search_document:` prefixes |
| `BAAI/bge-small-en-v1.5` | 33 M, 384-d | about 4x faster, 130 MB; good if build time or memory matters | a few points lower retrieval quality |
| `Alibaba-NLP/gte-large-en-v1.5` (also on the Magda release page, 1.3 GB zip) | 434 M, 1024-d | higher quality, reachable today | about 3x slower to build (2-3 hours on CPU) |
| `BAAI/bge-m3` | 568 M | multilingual (e.g. Dutch questions), dense + sparse in one | large and slow on CPU |
| `Qwen/Qwen3-Embedding-0.6B` | 600 M | among the strongest open embedding models | large; best with a GPU |

Adding a model means adding a `ModelSpec` entry in `src/rag_mvp/embeddings.py` (download URL, pinned hashes, pooling
and prefixes), then `uv run rag-embed build --model <name>` and comparing it on the dev split with
`uv run python -m rag_mvp.evaluation --mode dense --model <name> --save eval/<name>_dev.json` followed by
`--compare eval/dense_dev.json eval/<name>_dev.json`. Each model keeps its own vector folder, so models can be
compared without rebuilding the others.

Not chosen: `all-MiniLM-L6-v2` (older and clearly weaker at retrieval), `wordllama` (bundled in its PyPI package,
but a static word-vector model, much weaker), API models such as OpenAI/Azure embeddings (cost money and would send
the textbook text to an external service).
