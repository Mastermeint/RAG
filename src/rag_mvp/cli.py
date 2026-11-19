"""Command-line entry point for the local RAG MVP."""

import argparse
import sys
from pathlib import Path

from rag_mvp.console import use_utf8
from rag_mvp.embeddings import DEFAULT_MODE, MODES, ModelMissing, VectorsMissing, build_index
from rag_mvp.retrieval import load_corpus


def format_passage(number: int, passage) -> str:
    start, end = passage.page, passage.page_end or passage.page
    pages = "" if start is None else f"page {start}" if start == end else f"pages {start}-{end}"
    label = f", printed p.{passage.page_label}" if passage.page_label is not None else ""
    if passage.page_label is not None and passage.page_label_end not in (None, passage.page_label):
        label = f", printed pp.{passage.page_label}-{passage.page_label_end}"
    flags = ("low-quality text" if passage.low_quality else "",
             "table structure degraded" if passage.table_degraded else "")
    where = ", ".join(x for x in (pages + label, passage.type, *flags) if x)
    lines = [f"[{number}] {passage.source} ({where})"]
    if passage.section_path:
        lines.append("    " + " > ".join(passage.section_path))
    if passage.images:
        lines.append("    figure files: " + ", ".join(passage.images))
    lines.append(passage.text)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="Find relevant excerpts in local text documents.")
    parser.add_argument("question", help="Search query")
    parser.add_argument("--data-dir", type=Path, help="Search only this directory")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--type", action="append", choices=["text", "table", "figure"],
                        help="restrict results to a chunk type (repeatable)")
    parser.add_argument("--mode", choices=MODES,
                        help=f"bm25: word matching; dense: meaning (embeddings); hybrid: both (default: {DEFAULT_MODE}, "
                             f"falling back to bm25 when the model or saved vectors are missing)")
    args = parser.parse_args(argv)

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    passages = load_corpus(args.data_dir)
    if not passages:
        print("No searchable text found; add .txt/.md files or run rag-ingest on a PDF.", file=sys.stderr)
        return 1
    try:
        index = build_index(passages, args.mode or DEFAULT_MODE)
    except (ModelMissing, VectorsMissing) as error:
        if args.mode:
            parser.error(str(error))
        print(f"note: {error}; using bm25 word matching only", file=sys.stderr)
        index = build_index(passages, "bm25")
    matches = index.search(args.question, args.top_k, set(args.type) if args.type else None)
    if not matches:
        print("No matching excerpts found; try a more specific question.", file=sys.stderr)
        return 1
    for number, passage in enumerate(matches, start=1):
        print(format_passage(number, passage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
