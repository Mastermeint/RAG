# Textbook Retrieval MVP

Local pipeline that turns PDF textbooks into structure-aware, citable chunks and searches them by **meaning**, with a local embedding model (dense search). BM25 word matching and a hybrid of the two are available as options. It prints matching text with its PDF page, printed page number, chapter/section path and chunk type; it does not generate answers. This is the foundation for a future RAG system, so the focus is on ingestion and retrieval quality.

## Start

Requires Python 3.11+ via [uv](https://docs.astral.sh/uv/). No API key or external service is needed; the embedding model runs locally on the CPU.

```powershell
uv sync
uv run rag-ingest (Get-ChildItem 'data/raw/*.pdf').FullName --ocr
uv run rag-embed download        # once: fetch the embedding model (556 MB) and check its SHA-256
uv run rag-embed build           # after every ingest: embed new/changed chunks and save them (first run ~50 min on CPU)
uv run rag "how do fish osmoregulate in seawater"
uv run rag "mortality rate by species" --type table
```

`--ocr` needs the **Tesseract** executable (found on PATH or in `C:\Program Files\Tesseract-OCR`). It only re-reads scanned pages whose text layer is missing or damaged, never front/back matter, and keeps the OCR only if it reads better than the existing layer.

Plain `.txt`/`.md` files in `data/raw/` are also searchable. Search options: `--data-dir`, `--top-k`, `--type text|table|figure`, `--mode dense|bm25|hybrid`. Without the model or saved vectors, `rag` falls back to BM25 and says so. Output is UTF-8 even when redirected.

## Search modes

| Mode | How it ranks | Strong at | Weak at |
|---|---|---|---|
| `dense` (default) | Cosine similarity between the question's embedding and each chunk's embedding: texts with a similar *meaning* get similar vectors | Paraphrases and natural student questions | Rare exact terms (species names such as *Jordania zonope*), some tables |
| `bm25` | Shared words, weighted by how rare they are | Exact terms, table cells, figure labels | Questions in other words than the book uses (paraphrase recall 0.00 on dev) |
| `hybrid` | Reciprocal Rank Fusion: each ranking gives a chunk `weight / (60 + rank)` points (BM25 `--lexical-weight`, default 1.0; dense 1.0) | Tables and exact terms stay at 1.00 | BM25's noise on paraphrases drags the right answer down |

The model is `gte-base-en-v1.5` (ONNX, CPU, 768-dimensional vectors); see [models/README.md](models/README.md) for why, where it comes from, its pinned hashes and the alternatives worth trying. Every chunk is embedded once with `rag-embed build` and saved under `data/processed/embeddings/<model>/`. Vectors are keyed by a hash of the exact text that was embedded, so after a re-ingest only new or changed chunks are embedded again, a passage can never get another passage's vector, and a different model gets its own folder. A handful of missing vectors (a new `.txt` note) are computed on the fly and not saved; more than 50 missing is an error that asks for `rag-embed build`. A search takes about 4 s, mostly loading the model.

Results on the 2026-11-19 corpus (extractor version 4, 168 questions; see "Evaluating retrieval"):

| Split | Mode | Recall@3 | Recall@10 | MRR@10 | Paraphrase R@10 | Student R@10 | Lexical R@10 | Table R@10 |
|---|---|---|---|---|---|---|---|---|
| dev (87) | bm25 | 0.52 | 0.54 | 0.49 | 0.00 | 0.83 | 1.00 | 1.00 |
| dev (87) | **dense** | **0.67** | **0.78** | **0.56** | **0.57** | 1.00 | 0.93 | 0.88 |
| dev (87) | hybrid | 0.55 | 0.68 | 0.47 | 0.30 | 0.92 | 0.93 | 1.00 |
| holdout2 (32) | bm25 | 0.59 | 0.66 | 0.55 | 0.25 | 0.83 | 1.00 | 1.00 |
| holdout2 (32) | **dense** | **0.81** | **0.91** | **0.71** | **0.83** | 0.92 | 1.00 | 1.00 |
| holdout2 (32) | hybrid | 0.72 | 0.75 | 0.64 | 0.50 | 0.83 | 1.00 | 1.00 |

Dense against BM25 on dev: `improved` (24 questions gained at rank 10, 3 lost); dense against hybrid: `improved`. A sweep of the hybrid's BM25 weight on dev (1.0, 0.5, 0.3, 0.15) only got closer to dense as the weight fell, so dense is the default. The holdout2 split was then run once and agreed (dense against BM25: 8 gained, 0 lost). Groups under 15 questions are descriptive only, and the labels are still unaudited.

## What ingestion does

| Problem in the books | Handling |
|---|---|
| Two-column pages interleaved by `sort=True` | Recursive XY-cut reading order (column gutters first) |
| Running headers, footers, page numbers, copyright lines | Repeated margin text removed; the numbers give printed page labels |
| Soft hyphens, ligatures (`ﬁ`), U+2011, `sys- tem` | Characters repaired; line-break hyphens rejoined using the book's own vocabulary (keeps `well-known`) |
| Paragraphs cut by column/page breaks (even with a figure between) | Merged; the chunk records both pages |
| Tables | Ruled tables (PyMuPDF) and borderless tables anchored on a `TABLE n` caption; stored as Markdown chunks with the caption; cells removed from prose. Ruled tables whose cells came out fused are rebuilt from word positions; if that fails the chunk is flagged `table_structure_degraded` |
| Figures | Pictures **and vector charts** next to a `Figure n` caption (or an adjacent credit/small-print line) become `figure` chunks with the caption text and a rendered JPEG under `data/processed/figures/`; labels inside the figure and rotated credits are dropped. A line between a picture and its credit line is the caption and is kept with the credit; credit keywords must be whole words ("photo" no longer matches "Photosynthesis") |
| Decorated chapter initials (a 40 pt drop cap: `F` + `ishes were...`) | The oversized letter is glued back onto the line beside it |
| Margin glossary and side notes (small print opening with a bold term) | Kept as their own chunk(s), so they no longer get spliced into the middle of body sentences; the prose flows on around them |
| E-book video placeholders ("A YouTube element has been excluded...") | Dropped |
| Scans | Detected by full-page image coverage; embedded OCR layer kept, scored, optionally re-OCR'd |
| Contents, index, copyright, front matter | Excluded using the PDF outline, the running header or the printed-page offset |
| Chapter/section | From the PDF outline, otherwise from running headers; large/bold headings start new chunks |

Output per book in `data/processed/books/`:

- `<book>.jsonl` — one chunk per line: `id, doc_id, source, type, text, section_path, chapter, page, page_end, page_label, page_label_end, images, low_quality, table_structure_degraded`.
- `<book>.manifest.json` — SHA-256 of the PDF, extractor version, and a row per page (scanned, characters, quality, tables, figures, exclusion reason). Check this first when a result looks wrong.

Known limits: tables on scanned pages are not detected (no vector rules and unreliable OCR geometry), so they stay in the prose; figure images are page crops, not described or searchable by content; sidebar boxes that wrap around a figure can cut a sentence in two; the quality flag is heuristic; margin glossaries are only separated on born-digital pages (font sizes on scanned pages are unreliable, so the scanned *Introduction to Marine Biology* keeps them in the prose); a few OCR-garbled figure labels ("FIGURE Z-5") are not recognised as captions.

## Per-book settings: `ingest_config.json`

Anything specific to one PDF lives in `ingest_config.json` (tracked in Git) and nowhere else; no book title, file name or page number appears in `src/` (a test enforces this). Entries are keyed by the PDF file name and guarded by its SHA-256: if the hash differs, the entry is ignored with a warning, so a renamed or replaced file never receives another book's exclusions.

```json
{
  "defaults": {"detect": {"size_tolerance": 0.03, "min_run_pages": 10, "max_reference_pages": 40}},
  "books": {
    "<file name>.pdf": {
      "sha256": "<hash from the manifest>",
      "exclude_pages": [{"pages": [495, 812], "reason": "unrelated book appended to the PDF"}],
      "accept_pages": [{"pages": [200, 215], "reason": "photo insert on larger paper"}],
      "detect": {"min_run_pages": 8}
    }
  }
}
```

- `exclude_pages` removes pages from the index and wins over the outline. `accept_pages` records that a person looked at an odd-looking run and decided it is legitimate. `detect` overrides the thresholds (`size_tolerance`, `min_run_pages`, `max_reference_pages`, `max_excluded_share`) for that book. Unknown keys, bad ranges, empty reasons and non-positive thresholds are errors.
- Detectors run on every book, whatever the config: a run of at least `min_run_pages` pages whose size differs from the dominant one (found even if the pages are already excluded), an outline section that excludes more than `max_reference_pages` pages, and a book with more than `max_excluded_share` of its pages excluded for non-config reasons. They print `WARNING:` lines and are recorded in the manifest (`suspect_page_runs`, `long_reference_sections`, `excluded_by_reason`). An odd-sized run is **reviewed** only when it is covered by `exclude_pages` or `accept_pages`; an outline exclusion alone does not count. To clear a warning, look at the pages and add them to one of the two lists.
- Known limit: the dominant page size is simply the most common one, so a foreign book that makes up the majority of a PDF would be flagged the wrong way round.

```powershell
uv run rag-ingest <pdf> --ocr --config ingest_config.json --strict   # --strict: exit code 2 if any odd-sized run is unreviewed
uv run rag-ingest --upgrade-manifests                                 # add new manifest fields without re-ingesting
```

`--config` defaults to `ingest_config.json` (a missing default only prints a note; a missing explicit path is an error). Two version numbers decide what a change costs: `EXTRACTOR_VERSION` changes when chunk content changes (re-ingest needed), `MANIFEST_SCHEMA` changes when only manifest fields change. `--upgrade-manifests` fills the new fields from the PDF and the existing manifest, never touches the `.jsonl`, and refuses a book where a config exclusion would drop pages that currently have chunks (re-ingest that book instead).

## Evaluating retrieval

`eval/questions.jsonl` holds 168 labelled questions in seven categories: lexical (the book's own words), paraphrase (everyday words that deliberately avoid the book's vocabulary; the linter rejects one that shares more than 34% of its terms with the answer), student (phrased the way a student would actually ask), table, figure, multi_book and unanswerable (including near-misses such as "What are the opening hours of the Monterey Bay Aquarium?": the aquarium is mentioned, its hours are not). Each question has a `split`: `dev` for tuning decisions, `holdout2` for one final check per change (`holdout` was used once on 2026-11-17 and is kept for history). Gold locations are given as source file, PDF page range (at most 3 pages) and anchor phrases that the correct chunk must contain. Labels use pages, not chunk IDs, so they survive a re-ingest. A general question can be answered by more than one book; give it a gold entry per book, or a correct answer from another book counts as a miss.

```powershell
uv run python -m rag_mvp.evaluation                          # dev split, default mode (dense): Recall@3/5/10 and MRR@10 with 95% bootstrap intervals
uv run python -m rag_mvp.evaluation --mode bm25              # or dense / hybrid; --model and --lexical-weight select the embedding model and fusion weight
uv run python -m rag_mvp.evaluation --save eval/run.json     # keep a run, with the corpus fingerprint, retrieval mode and ranking config
uv run python -m rag_mvp.evaluation --lint                   # validate the questions against the real corpus
uv run python -m rag_mvp.evaluation --audit 10 --seed 1      # print questions for a human to check
uv run python -m rag_mvp.evaluation --selftest               # each chunk's first sentence should retrieve the chunk (BM25)
uv run python -m rag_mvp.evaluation --compare a.json b.json  # paired comparison and one verdict
uv run python -m rag_mvp.evaluation --ranking low_quality_weight=0.8 --ranking type_weights.figure=0.9
uv run python -m rag_mvp.evaluation --sweep credit_only_weight=1.0,0.5,0.0 --against eval/dense_dev.json
uv run python -m rag_mvp.evaluation --split holdout2 --final # refused without --final
```

`--compare` refuses runs with different corpus fingerprints, splits or match modes (runs with different retrieval modes can be compared, which is the point), and gives `improved` only for a net gain of at least 2 questions at rank 10 with MRR not lower; anything smaller is `no measurable change`. Current saved runs: `eval/{bm25,dense,hybrid}_dev.json` and `eval/{bm25,dense,hybrid}_holdout2.json`. The older files (`baseline_bm25_dev.json`, `after_task5_dev.json`, `after_task6_dev.json`, `baseline_holdout.json`, `final_holdout.json`) belong to extractor version 3 and the 98-question set; they are kept as history and can only be compared with `--force`. The questions were drafted by agents reading the books, without a human review of the labels, so treat the numbers as provisional until `--audit` has been checked by a person. Editing `questions.jsonl` changes its hash, so the saved runs must be regenerated afterwards. Do not tune on a holdout split.

Ranking is `RankingConfig` in `retrieval.py`: multipliers on the score for `low_quality_weight`, `table_degraded_weight`, `credit_only_weight` and `type_weights`, applied to the BM25 score, the cosine similarity or the fused score depending on the mode. Every default is neutral (1.0); a weight of 0 hides a passage. A default changes only if the evaluation shows an `improved` verdict on dev. Credit-line figure chunks (`credit_only`, from `credits.py`: every sentence carries a credit marker) are marked, not removed, because their image is otherwise unreachable.

## Next steps

In recommended order; each one is judged with the evaluation above (tune on dev, `--compare` against `eval/dense_dev.json`, look at `holdout2` once).

1. **Human label audit.** Every number above is provisional until a person checks a sample of the 168 labels (`--audit 20`). Also check whether general student questions with one gold book are answered by another book too; if so, add a second gold entry.
2. **Smarter fusion for exact terms.** Dense search misses rare exact terms (a species name such as *Jordania zonope*, some table labels) that BM25 finds, but equal-weight hybrid fusion hurts paraphrases. Let BM25 count only when the question contains a rare term (very high IDF) or when BM25's top score is unusually high; otherwise use dense alone.
3. **Reranker.** Dense dev Recall@10 is 0.78 but Recall@3 is 0.67: the right chunk is often found but not first. A cross-encoder (for example `BAAI/bge-reranker-base`) that re-reads the question with each of the top 20-30 chunks should lift Recall@3 and MRR, which is what an answer generator needs. Needs the model from huggingface.co or a mirror.
4. **Compare a second embedding model** from the list in [models/README.md](models/README.md): add a `ModelSpec`, `rag-embed build --model <name>`, one dev run.
5. **Speed (optional).** An int8-quantised model (`onnxruntime.quantization.quantize_dynamic`) roughly halves build time and size; check that recall holds. A long-running search service would load the model once and cut a search from about 4 s to well under 1 s.
6. **Answer generation (the "G" in RAG).** Give the top 3-5 dense chunks to an LLM with "answer only from these passages and cite the pages". For "I don't know", calibrate a threshold on the dense top-1 cosine (dev: answerable median 0.65, unanswerable 0.45). Needs a decision on which LLM (local or Azure OpenAI; policy, cost, copyright).

## Layout

```text
data/raw/              Source PDFs and text (ignored by Git)
data/processed/books/  Chunks + manifests (ignored by Git)
data/processed/figures/ Rendered figure images (ignored by Git)
data/processed/embeddings/<model>/  Saved passage vectors: vectors.npy, keys.json, info.json (ignored by Git)
models/                Embedding model files (ignored by Git) and models/README.md: source, hashes, alternatives
ingest_config.json     Per-book page exclusions and detector thresholds
eval/                  Labelled questions and saved evaluation runs
src/rag_mvp/           textclean, layout, chunking, ingest, ingest_config, retrieval, embeddings, credits, evaluation, cli
tests/                 Hostile synthetic PDFs (each with a control proving the old extraction fails), regression tests for
                       defects found in the real books, checks on the real ingested output, retrieval and embedding tests
scripts/               Parser comparison probes and a wrapper for the evaluation command
```

## Test

```powershell
uv run python -m unittest discover -s tests -v
```

Tests that need `data/processed/books/` or the PDFs in `data/raw/` are skipped when those are missing; run `rag-ingest` first to enable them. The real-model check is skipped until `rag-embed download` has run; all other embedding tests use a small fake embedder and never load the model.
