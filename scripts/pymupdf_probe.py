# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pymupdf>=1.26,<2"]
# ///
"""Extract a page-bounded PDF sample or inspect its embedded text with PyMuPDF."""

import argparse
import time
from pathlib import Path

import pymupdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="Create a new PDF containing inclusive 1-based pages instead of extracting text")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(args.pdf) as source:
        if args.sample:
            first, last = args.sample
            if not 1 <= first <= last <= len(source):
                parser.error(f"page range must be within 1..{len(source)}")
            with pymupdf.open() as excerpt:
                excerpt.insert_pdf(source, from_page=first - 1, to_page=last - 1)
                excerpt.save(args.output)
            print(f"Created {args.output}: pages {first}-{last} of {len(source)}")
            return
        lines = []
        for number, page in enumerate(source, start=1):
            lines.append(f"# PDF page {number}\n\n{page.get_text(sort=True).strip()}\n")
            print(f"page {number}: {len(page.get_text().strip())} characters, "
                  f"{len(page.get_images())} embedded images, {len(page.find_tables().tables)} tables")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Output: {args.output}; conversion: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()