# Textbook Retrieval MVP

Local pipeline that turns PDF textbooks into structure-aware, citable chunks and searches them with BM25. It prints matching text with its PDF page, printed page number, chapter/section path and chunk type; it does not generate answers. This is the foundation for a future RAG system, so the focus is on ingestion quality.

## Start

Requires Python 3.11+ via [uv](https://docs.astral.sh/uv/). No API key, model or external service is needed.

```powershell
uv sync
uv run rag-ingest (Get-ChildItem 'data/raw/*.pdf').FullName --ocr
uv run rag "how do fish osmoregulate in seawater"
uv run rag "mortality rate by species" --type table
```

`--ocr` needs the **Tesseract** executable (found on PATH or in `C:\Program Files\Tesseract-OCR`). It only re-reads scanned pages whose text layer is missing or damaged, never front/back matter, and keeps the OCR only if it reads better than the existing layer.

Plain `.txt`/`.md` files in `data/raw/` are also searchable. Search options: `--data-dir`, `--top-k`, `--type text|table|figure`. Output is UTF-8 even when redirected.

## What ingestion does

| Problem in the books | Handling |
|---|---|
| Two-column pages interleaved by `sort=True` | Recursive XY-cut reading order (column gutters first) |
| Running headers, footers, page numbers, copyright lines | Repeated margin text removed; the numbers give printed page labels |
| Soft hyphens, ligatures (`ﬁ`), U+2011, `sys- tem` | Characters repaired; line-break hyphens rejoined using the book's own vocabulary (keeps `well-known`) |
| Paragraphs cut by column/page breaks (even with a figure between) | Merged; the chunk records both pages |
| Tables | Ruled tables (PyMuPDF) and borderless tables anchored on a `TABLE n` caption; stored as Markdown chunks with the caption; cells removed from prose. Ruled tables whose cells came out fused are rebuilt from word positions; if that fails the chunk is flagged `table_structure_degraded` |
| Figures | Pictures **and vector charts** next to a `Figure n` caption (or an adjacent credit/small-print line) become `figure` chunks with the caption text and a rendered JPEG under `data/processed/figures/`; labels inside the figure and rotated credits are dropped |
| Scans | Detected by full-page image coverage; embedded OCR layer kept, scored, optionally re-OCR'd |
| Contents, index, copyright, front matter | Excluded using the PDF outline, the running header or the printed-page offset |
| Chapter/section | From the PDF outline, otherwise from running headers; large/bold headings start new chunks |

Output per book in `data/processed/books/`:

- `<book>.jsonl` — one chunk per line: `id, doc_id, source, type, text, section_path, chapter, page, page_end, page_label, page_label_end, images, low_quality, table_structure_degraded`.
- `<book>.manifest.json` — SHA-256 of the PDF, extractor version, and a row per page (scanned, characters, quality, tables, figures, exclusion reason). Check this first when a result looks wrong.

Known limits: tables on scanned pages are not detected (no vector rules and unreliable OCR geometry), so they stay in the prose; figure images are page crops, not described or searchable by content; sidebar boxes that wrap around a figure can cut a sentence in two; the quality flag is heuristic; retrieval is lexical only (no embeddings).

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

`eval/questions.jsonl` holds labelled questions (lexical, paraphrase, table, figure, multi_book, unanswerable), each with a `split` (`dev` for tuning decisions, `holdout` for one final check) and gold locations given as source file, PDF page range (at most 3 pages) and anchor phrases that the correct chunk must contain. Labels use pages, not chunk IDs, so they survive a re-ingest.

```powershell
uv run python -m rag_mvp.evaluation                          # dev split report: Recall@3/5/10 and MRR@10 with 95% bootstrap intervals
uv run python -m rag_mvp.evaluation --save eval/run.json     # keep a run, with the corpus fingerprint and ranking config
uv run python -m rag_mvp.evaluation --lint                   # validate the questions against the real corpus
uv run python -m rag_mvp.evaluation --audit 10 --seed 1      # print questions for a human to check
uv run python -m rag_mvp.evaluation --selftest               # each chunk's first sentence should retrieve the chunk
uv run python -m rag_mvp.evaluation --compare a.json b.json  # paired comparison and one verdict
uv run python -m rag_mvp.evaluation --ranking low_quality_weight=0.8 --ranking type_weights.figure=0.9
uv run python -m rag_mvp.evaluation --sweep credit_only_weight=1.0,0.5,0.0 --against eval/baseline_bm25_dev.json
uv run python -m rag_mvp.evaluation --split holdout --final  # refused without --final
```

`--compare` refuses runs with different corpus fingerprints, splits or match modes, and gives `improved` only for a net gain of at least 2 questions at rank 10 with MRR not lower; anything smaller is `no measurable change`. Saved baselines live in `eval/` (`baseline_bm25_dev.json`, `after_task5_dev.json`, `after_task6_dev.json`, and the holdout pair). The questions were drafted from the books without a human review of the labels, so treat the numbers as provisional until `--audit` has been checked by a person. Editing `questions.jsonl` changes its hash, so the saved runs must be regenerated afterwards. The holdout split has been used once; do not tune on it.

Ranking is `RankingConfig` in `retrieval.py`: multipliers on the BM25 score for `low_quality_weight`, `table_degraded_weight`, `credit_only_weight` and `type_weights`. Every default is neutral (1.0), which gives plain BM25; a weight of 0 hides a passage. A default changes only if the evaluation shows an `improved` verdict on dev. Credit-line figure chunks (`credit_only`, from `credits.py`) are marked, not removed, because their image is otherwise unreachable.

## Layout

```text
data/raw/              Source PDFs and text (ignored by Git)
data/processed/books/  Chunks + manifests (ignored by Git)
data/processed/figures/ Rendered figure images (ignored by Git)
ingest_config.json     Per-book page exclusions and detector thresholds
eval/                  Labelled questions and saved evaluation runs
src/rag_mvp/           textclean, layout, chunking, ingest, ingest_config, retrieval, credits, evaluation, cli
tests/                 Hostile synthetic PDFs (each with a control proving the old extraction fails), regression tests for
                       defects found in the real books, checks on the real ingested output, retrieval tests
scripts/               Parser comparison probes and a wrapper for the evaluation command
```

## Test

```powershell
uv run python -m unittest discover -s tests -v
```

Tests that need `data/processed/books/` or the PDFs in `data/raw/` are skipped when those are missing; run `rag-ingest` first to enable them.
