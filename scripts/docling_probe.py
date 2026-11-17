# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["docling>=2,<3"]
# ///
"""Convert a small PDF sample to Markdown with Docling's local pipeline."""

import argparse
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    options = PdfPipelineOptions(do_ocr=False)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    result = converter.convert(args.pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.document.export_to_markdown(), encoding="utf-8")
    print(f"Output: {args.output}; conversion: {time.perf_counter() - started:.1f}s; "
          f"pages: {len(result.document.pages)}, tables: {len(result.document.tables)}, "
          f"pictures: {len(result.document.pictures)}")


if __name__ == "__main__":
    main()