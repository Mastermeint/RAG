"""Turn PDF books into structure-aware, citable chunks with a per-page quality manifest."""

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pymupdf

from rag_mvp.chunking import Element, chunk_elements
from rag_mvp.console import use_utf8
from rag_mvp.ingest_config import BookConfig, DetectConfig, Range, config_for, load_settings
from rag_mvp.layout import Block, PageLayout, extract_page, reading_order
from rag_mvp.textclean import build_vocabulary, clean_chars, damage, is_noise, join_lines, repair_ocr, text_quality

EXTRACTOR_VERSION = "3"
MANIFEST_SCHEMA = 2
_CAPTION = re.compile(r"^\W{0,2}(fig(?:ure)?|f[il1]gure|table)\b\.?\s*([A-Z]?\d)", re.I)
_TERMINAL = re.compile(r"[.!?:;\"”’)\]]$")
_CHAPTER_TITLE = re.compile(r"(?i)^(chapter|ch\.)\s*\d+|^\d+\s*[:.]")
_CHAPTER_HEADER = re.compile(r"(?i)\bchapter\s+(\d(?:\s?[0-9OIl])?)\s+(.+)")
_COPYRIGHT = re.compile(r"(?i)^\W*cop\w{2,10}\W{0,3}(?:©\W*)?['’]?(?:19|20)\d\d")
_REFERENCE_TITLE = re.compile(
    r"(?i)^(cover|title page|copyright( page)?|contents|table of contents|dedication|preface|foreword|"
    r"acknowledg\w*|about the authors?|eula|index|eye on science boxes|list of .*)$")
_PAGE_NUMBER = (re.compile(r"^\s*(\d{1,4})(?!\d)"), re.compile(r"(?<!\d)(\d{1,4})\s*$"))


@dataclass
class SuspectRun:
    first: int
    last: int
    width: int
    height: int
    status: str  # config | accepted | outline | indexed | partial
    reviewed: bool


@dataclass
class IngestResult:
    chunks: int = 0
    pages: int = 0
    scanned_pages: int = 0
    poor_quality_pages: list[int] = field(default_factory=list)
    low_text_pages: list[int] = field(default_factory=list)
    excluded_pages: dict[int, str] = field(default_factory=dict)
    ocr_pages: list[int] = field(default_factory=list)
    tables: int = 0
    figures: int = 0
    output: Path | None = None
    manifest: Path | None = None
    warnings: list[str] = field(default_factory=list)
    suspect_runs: list[SuspectRun] = field(default_factory=list)
    long_reference_sections: list[dict] = field(default_factory=list)
    config_excluded_ranges: list[dict] = field(default_factory=list)


@dataclass
class _Para:
    page: int
    lines: list[str]
    bbox: list[float]
    size: float
    bold: bool
    kind: str = "text"  # text | heading | caption | table
    caption_for: str = ""
    block: Block | None = None

    @property
    def text(self) -> str:
        return " ".join(self.lines)


def _margin_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z]{3,}", text.lower()))


def _inside(rect: tuple, block: Block, pad: float = 6.0) -> bool:
    cx, cy = (block.x0 + block.x1) / 2, (block.y0 + block.y1) / 2
    return rect[0] - pad <= cx <= rect[2] + pad and rect[1] - pad <= cy <= rect[3] + pad


def _toc(doc: pymupdf.Document) -> list[tuple[int, str, int]]:
    entries = []
    for level, title, page in doc.get_toc():
        title = re.sub(r"\s+", " ", title.replace("\x00", "")).strip()
        if title and page >= 1:
            entries.append((level, title, page))
    return sorted(entries, key=lambda e: e[2])  # stable: keeps outline order within a page


def _toc_by_page(entries: list[tuple[int, str, int]], pages: int) -> tuple[dict[int, list[str]], dict[int, str]]:
    """Outline path (levels 1-3) per page, and the pages that are front/back matter."""
    paths: dict[int, list[str]] = {}
    reference: dict[int, str] = {}
    stack: list[str] = []
    index = 0
    for page in range(1, pages + 1):
        while index < len(entries) and entries[index][2] <= page:
            level, title, _ = entries[index]
            stack = stack[: level - 1] + [title]
            index += 1
        paths[page] = stack[:3]
    for position, (level, title, start) in enumerate(entries):
        if not _REFERENCE_TITLE.match(title):
            continue
        end = pages + 1
        for later in entries[position + 1:]:
            if later[0] <= level and later[2] > start:
                end = later[2]
                break
        for page in range(start, min(end, pages + 1)):
            reference[page] = f"outline: {title}"
    return paths, reference


def _chapter_of(path: list[str]) -> str:
    for title in path:
        if _CHAPTER_TITLE.match(title):
            return title
    return path[0] if path else ""


def _wrap_paragraphs(blocks: list[Block]) -> list[_Para]:
    """Merge blocks that are wrapped lines of one paragraph (some PDFs emit one block per line)."""
    paras: list[_Para] = []
    for block in blocks:
        if block.kind == "table":
            paras.append(_Para(block.page, [], list(block.bbox), 0.0, False, "table", block=block))
            continue
        prev_para = paras[-1] if paras else None
        if prev_para is not None and prev_para.kind == "text" and prev_para.block is not None:
            prev = prev_para.block
            column_edge = max((b.x1 for b in blocks if b.kind == "text" and abs(b.x0 - prev.x0) < 6),
                              default=prev.x1)
            full_line = prev.x1 >= column_edge - 0.08 * max(column_edge - prev.x0, 1)
            gap = block.y0 - prev.y1
            if (full_line and -1 <= gap <= 0.6 * max(prev.size, 1) and abs(block.x0 - prev.x0) <= 3
                    and abs(block.size - prev.size) <= 0.6 and block.bold == prev.bold):
                prev_para.lines.extend(block.lines)
                prev_para.bbox = [prev_para.bbox[0], prev_para.bbox[1], max(prev_para.bbox[2], block.x1), block.y1]
                prev_para.block = block
                continue
        paras.append(_Para(block.page, list(block.lines), list(block.bbox), block.size, block.bold, block=block))
    return paras


def _page_paragraphs(layout: PageLayout, kept_margins: list[Block]) -> list[_Para]:
    blocks = reading_order(layout.blocks + kept_margins) if kept_margins else layout.blocks
    body = []
    for block in blocks:
        if block.kind == "text" and any(_inside(f, block) for f in layout.figures):
            text = block.raw_text
            if len(text) / max(len(block.lines), 1) < 40 or len(text) < 80:
                continue  # axis text, map labels and scan speckle inside a picture
        body.append(block)
    return _wrap_paragraphs(body)


def _classify(paras: list[_Para], layout: PageLayout, body_size: float) -> list[_Para]:
    """Mark headings and captions, absorbing continuation lines of captions."""
    native = not layout.scanned
    zones = [pymupdf.Rect(f) + (-80, -80, 80, 80) for f in layout.figures]
    zones += [pymupdf.Rect(p.bbox) + (-60, -60, 60, 60) for p in paras if p.kind == "table"]
    result: list[_Para] = []
    index = 0
    while index < len(paras):
        para = paras[index]
        index += 1
        if para.kind == "table":
            result.append(para)
            continue
        match = _CAPTION.match(para.text)
        if match:
            adjacent = any(z.intersects(pymupdf.Rect(para.bbox)) for z in zones)
            word = match.group(1)
            styled = para.bold or word.isupper() or (native and body_size and para.size < body_size - 0.4)
            if adjacent or styled:
                para.kind = "caption"
                para.caption_for = "table" if word.lower() == "table" else "figure"
                left, right, last_right = para.bbox[0], para.bbox[2], para.bbox[2]
                while index < len(paras):
                    nxt = paras[index]
                    complete = _TERMINAL.search(para.lines[-1].rstrip()) and last_right < right - 0.12 * (right - left)
                    if (complete or nxt.kind != "text" or nxt.bbox[1] - para.bbox[3] > 0.9 * max(para.size, nxt.size)
                            or abs(nxt.bbox[0] - left) > 4 or (native and nxt.size > para.size + 0.5)
                            or _CAPTION.match(nxt.text)):
                        break
                    para.lines.extend(nxt.lines)
                    para.bbox = [left, para.bbox[1], max(para.bbox[2], nxt.bbox[2]), nxt.bbox[3]]
                    right, last_right = max(right, nxt.bbox[2]), nxt.bbox[2]
                    index += 1
                result.append(para)
                continue
        if native and body_size:
            text = para.text
            big = para.size >= 1.18 * body_size
            bold_head = para.bold and para.size >= body_size - 0.6 and len(text) <= 100 and len(para.lines) <= 2
            if ((big or bold_head) and len(text) <= 160 and len(para.lines) <= 3
                    and not re.search(r"[.;,]$", text) and sum(c.isalpha() for c in text) >= 3):
                para.kind = "heading"
        result.append(para)
    return result


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _assign_figures(layout: PageLayout, paras: list[_Para]) -> dict[int, list[tuple]]:
    """Give each picture on the page to its nearest figure caption (keyed by id of the caption paragraph)."""
    captions = [p for p in paras if p.kind == "caption" and p.caption_for == "figure"]
    assigned: dict[int, list[tuple]] = {id(c): [] for c in captions}
    for figure in layout.figures:
        best = None
        for cap in captions:
            horizontal = _overlap(figure[0], figure[2], cap.bbox[0], cap.bbox[2])
            vertical = max(cap.bbox[1] - figure[3], figure[1] - cap.bbox[3], 0)
            if vertical <= 200 and horizontal >= 0.2 * min(figure[2] - figure[0], cap.bbox[2] - cap.bbox[0]):
                if best is None or vertical < best[0]:
                    best = (vertical, cap)
        if best is not None:
            assigned[id(best[1])].append(figure)
    return assigned


_CREDIT = re.compile(r"(?i)^\W?[“\"].+[”\"]\s*(by|via|from)\b|\b(wikimedia|creative commons|cc by|photo|courtesy|credit|source:)")


def _adopt_credit_captions(layout: PageLayout, paras: list[_Para], assigned: dict[int, list[tuple]],
                           body_size: float) -> int:
    """Pictures without a 'Figure N' caption: use an adjacent small-print or credit line as the caption."""
    taken = {tuple(f) for rects in assigned.values() for f in rects}
    orphans = 0
    for figure in layout.figures:
        if tuple(figure) in taken:
            continue
        best = None
        for para in paras:
            if para.kind != "text" or para.block is None or len(para.text) > 300:
                continue
            below, above = para.bbox[1] - figure[3], figure[1] - para.bbox[3]
            gap = below if -2 <= below <= 30 else above if -2 <= above <= 30 else None
            width = para.bbox[2] - para.bbox[0]
            if gap is None or _overlap(figure[0], figure[2], para.bbox[0], para.bbox[2]) < 0.5 * width:
                continue
            small = not layout.scanned and body_size and para.size < body_size - 0.3
            if (small or _CREDIT.search(para.text)) and (best is None or gap < best[0]):
                best = (gap, para)
        if best is None:
            orphans += 1
            continue
        best[1].kind, best[1].caption_for = "caption", "figure"
        assigned.setdefault(id(best[1]), []).append(figure)
    return orphans


def _table_caption(table: _Para, paras: list[_Para]) -> _Para | None:
    best = None
    for para in paras:
        if para.kind != "caption" or para.caption_for != "table":
            continue
        above = table.bbox[1] - para.bbox[3]
        below = para.bbox[1] - table.bbox[3]
        distance = above if -2 <= above <= 60 else below if -2 <= below <= 40 else None
        if distance is not None and (best is None or distance < best[0]):
            best = (distance, para)
    return best[1] if best else None


def _page_text(layout: PageLayout) -> str:
    return " ".join(b.raw_text for b in layout.blocks if b.kind == "text")


def default_ocr(page: pymupdf.Page, number: int) -> PageLayout:
    return extract_page(page, number, page.get_textpage_ocr(full=True, dpi=300, language="eng"))


def _find_tesseract() -> bool:
    """True if Tesseract is usable; adds the default Windows install folder to PATH when needed."""
    if shutil.which("tesseract"):
        return True
    for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        folder = Path(root) / "Tesseract-OCR" if root else None
        if folder and (folder / "tesseract.exe").exists():
            os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"
            return True
    return False


def page_sizes(doc: pymupdf.Document) -> dict[int, tuple[int, int]]:
    """Rounded (width, height) of every page, from page rectangles only."""
    return {n: (round(doc[n - 1].rect.width), round(doc[n - 1].rect.height)) for n in range(1, len(doc) + 1)}


def dominant_size(sizes: dict[int, tuple[int, int]], numbers: list[int]) -> tuple[int, int]:
    counts = Counter(sizes[n] for n in numbers)
    return min(counts, key=lambda size: (-counts[size], size))


def _pages_of(ranges: tuple[Range, ...]) -> set[int]:
    return {n for first, last, _ in ranges for n in range(first, last + 1)}


def find_suspect_runs(sizes: dict[int, tuple[int, int]], numbers: list[int], excluded: dict[int, str],
                      accepted: set[int], detect: DetectConfig) -> list[SuspectRun]:
    """Runs of odd-sized pages. Exclusion status is ignored when looking, and reported afterwards."""
    if not numbers:
        return []
    width, height = dominant_size(sizes, numbers)

    def odd(n: int) -> bool:
        w, h = sizes[n]
        return abs(w - width) > detect.size_tolerance * width or abs(h - height) > detect.size_tolerance * height

    def kind(n: int) -> str:
        if excluded.get(n, "").startswith("config:"):
            return "config"
        return "accepted" if n in accepted else "outline" if n in excluded else "indexed"

    groups: list[list[int]] = []
    for n in sorted(numbers):
        if not odd(n):
            continue
        if groups and groups[-1][-1] == n - 1:
            groups[-1].append(n)
        else:
            groups.append([n])
    runs = []
    for pages in groups:
        if len(pages) < detect.min_run_pages:
            continue
        kinds = {kind(n) for n in pages}
        status = kinds.pop() if len(kinds) == 1 else "partial"
        w, h = Counter(sizes[n] for n in pages).most_common(1)[0][0]
        runs.append(SuspectRun(pages[0], pages[-1], w, h, status, status in ("config", "accepted")))
    return runs


def long_reference_sections(excluded: dict[int, str], detect: DetectConfig) -> list[dict]:
    """Outline-driven exclusions spanning more pages than a front/back matter section normally does."""
    sections: list[dict] = []
    for n in sorted(excluded):
        reason = excluded[n]
        if not reason.startswith("outline:"):
            continue
        if sections and sections[-1]["reason"] == reason and sections[-1]["last"] == n - 1:
            sections[-1]["last"] = n
        else:
            sections.append({"first": n, "last": n, "reason": reason})
    return [{**s, "pages": s["last"] - s["first"] + 1} for s in sections
            if s["last"] - s["first"] + 1 > detect.max_reference_pages]


def excluded_share(excluded: dict[int, str], pages: int) -> float:
    """Share of the book's pages excluded for reasons other than the config."""
    return sum(1 for r in excluded.values() if not r.startswith("config:")) / pages if pages else 0.0


def _apply_ranges(excluded: dict[int, str], config: BookConfig, page_count: int) -> None:
    """Config exclusions win over any other reason."""
    for first, last, reason in (*config.exclude, *config.accept):
        if last > page_count:
            raise ValueError(f"{config.name}: configured page {last} is outside 1..{page_count}")
    for first, last, reason in config.exclude:
        for n in range(first, last + 1):
            excluded[n] = f"config: {reason}"


def _range_dicts(ranges: tuple[Range, ...]) -> list[dict]:
    return [{"pages": [first, last], "reason": reason} for first, last, reason in ranges]


def review_pages(sizes: dict[int, tuple[int, int]], numbers: list[int], excluded: dict[int, str],
                 config: BookConfig | None, detect: DetectConfig) -> tuple[dict, list[str], list[SuspectRun]]:
    """Manifest fields and warnings from page geometry and exclusions. Shared by ingest and manifest upgrade."""
    accepted = _pages_of(config.accept) if config else set()
    runs = find_suspect_runs(sizes, numbers, excluded, accepted, detect)
    sections = long_reference_sections(excluded, detect)
    share = excluded_share(excluded, len(numbers))
    warnings = []
    if numbers:
        normal = "x".join(map(str, dominant_size(sizes, numbers)))
        for run in runs:
            if not run.reviewed:
                warnings.append(
                    f"pages {run.first}-{run.last} are {run.width}x{run.height} (most pages are {normal}) and are "
                    f"{run.status}; if they are not part of this book add them to exclude_pages, "
                    f"if they are legitimate add them to accept_pages")
    for s in sections:
        warnings.append(f"{s['reason']} excludes pages {s['first']}-{s['last']} ({s['pages']} pages, "
                        f"limit {detect.max_reference_pages}); check that the outline is not swallowing real content")
    if share > detect.max_excluded_share:
        warnings.append(f"{share:.0%} of pages are excluded for non-config reasons (limit "
                        f"{detect.max_excluded_share:.0%}); check front/back matter detection")
    info = {
        "manifest_schema": MANIFEST_SCHEMA,
        "config_excluded_ranges": _range_dicts(config.exclude) if config else [],
        "config_accepted_ranges": _range_dicts(config.accept) if config else [],
        "suspect_page_runs": [asdict(r) for r in runs],
        "long_reference_sections": sections,
        "excluded_by_reason": dict(Counter(excluded[n] for n in sorted(excluded))),
    }
    return info, warnings, runs


def ingest_pdf(pdf: Path, output: Path, first_page: int = 1, last_page: int | None = None,
               ocr: bool = False, figure_dir: Path | None = None,
               ocr_page: Callable[[pymupdf.Page, int], PageLayout] | None = None,
               book_config: BookConfig | None = None, detect: DetectConfig | None = None) -> IngestResult:
    if ocr and ocr_page is None:
        if not _find_tesseract():
            raise RuntimeError("OCR requires the Tesseract executable on PATH")
        ocr_page = default_ocr
    result = IngestResult(output=output)
    digest = hashlib.sha256(Path(pdf).read_bytes()).hexdigest()
    doc_id = digest[:12]
    with pymupdf.open(pdf) as doc:
        end = len(doc) if last_page is None else last_page
        if not 1 <= first_page <= end <= len(doc):
            raise ValueError(f"page range must be within 1..{len(doc)}")
        numbers = list(range(first_page, end + 1))
        layouts = {n: extract_page(doc[n - 1], n) for n in numbers}
        toc = _toc(doc)
        toc_paths, reference = _toc_by_page(toc, len(doc))
        if book_config is not None:
            _apply_ranges(reference, book_config, len(doc))
        dims = page_sizes(doc)
        # OCR only scans whose text layer is missing or damaged, never front/back matter, and keep it only if it reads better.
        if ocr_page is not None:
            for n in numbers:
                old = _page_text(layouts[n]).strip()
                if n not in reference and layouts[n].scanned and (len(old) < 100 or text_quality(old)["poor"]):
                    page = doc[n - 1]
                    candidate = ocr_page(page, n)
                    candidate.scanned = True
                    new = _page_text(candidate).strip()
                    if (len(new) > len(old) if len(old) < 100 else
                            len(new) >= 0.7 * len(old) and damage(new) < damage(old)):
                        layouts[n] = candidate
                        result.ocr_pages.append(n)

        # Running headers/footers: repeated margin text, bare page numbers and copyright lines.
        counts = Counter(k for n in numbers for k in {_margin_key(b.raw_text) for b in layouts[n].margins} if k)
        repeated = {k for k, c in counts.items() if c >= 3}  # a 16-page chapter has only ~8 identical headers
        removed: dict[int, list[str]] = {n: [] for n in numbers}
        kept: dict[int, list[Block]] = {n: [] for n in numbers}
        for n in numbers:
            for block in layouts[n].margins:
                text = block.raw_text.strip()
                key = _margin_key(text)
                if key in repeated or (not key and len(text) <= 12) or _COPYRIGHT.match(text):
                    removed[n].append(clean_chars(text))
                else:
                    kept[n].append(block)

        # Printed page labels from the dominant (printed - pdf) offset among numbers in removed margins.
        candidates = {n: {int(m.group(1)) for t in removed[n] for p in _PAGE_NUMBER if (m := p.search(t))}
                      for n in numbers}
        offsets = Counter(v - n for n, values in candidates.items() for v in values)
        offset, support = offsets.most_common(1)[0] if offsets else (0, 0)
        has_labels = support >= max(3, int(0.02 * len(numbers)))
        first_labelled = min((n for n in numbers if has_labels and n + offset in candidates[n]), default=None)

        def page_label(n: int) -> int | None:
            if not has_labels or first_labelled is None or n < first_labelled - 3 or n + offset < 1:
                return None
            return n + offset

        header_chapter: dict[int, str] = {}
        if not toc:
            current = ""
            for n in numbers:
                for text in removed[n]:
                    match = _CHAPTER_HEADER.search(text)
                    if match:
                        digits = re.sub(r"\s", "", match.group(1)).translate(str.maketrans("OIl", "01l"))
                        title = re.sub(r"\s*\d+\s*$", "", match.group(2)).strip()
                        current = f"Chapter {digits} {title}".strip()
                header_chapter[n] = current
                if has_labels and page_label(n) is None:
                    reference.setdefault(n, "front matter (before first printed page)")
                if any(_margin_key(t) == "index" for t in removed[n]):
                    reference.setdefault(n, "running header: Index")

        sizes: Counter = Counter()
        for n in numbers:
            if not layouts[n].scanned:
                for block in layouts[n].blocks:
                    if block.kind == "text":
                        sizes[round(block.size * 2) / 2] += len(block.raw_text)
        body_size = sizes.most_common(1)[0][0] if sizes else 0.0
        vocab = build_vocabulary([line for n in numbers for b in layouts[n].blocks if b.kind == "text"
                                  for line in b.lines])

        elements: list[Element] = []
        manifest_pages = []
        figures_root = Path(figure_dir) / doc_id if figure_dir else None
        for n in numbers:
            layout = layouts[n]
            paras = _classify(_page_paragraphs(layout, kept[n]), layout, body_size)
            images_for = _assign_figures(layout, paras)
            orphan_figures = _adopt_credit_captions(layout, paras, images_for, body_size)
            table_caption = {id(p): _table_caption(p, paras) for p in paras if p.kind == "table"}
            consumed = {id(c) for c in table_caption.values() if c is not None}
            page_elements: list[Element] = []
            page_text: list[str] = []
            tables = figures = 0
            for para in paras:
                if id(para) in consumed:
                    continue
                if para.kind == "table":
                    caption = table_caption[id(para)]
                    caption_text = para.block.caption or (join_lines(caption.lines, vocab) if caption else "")
                    page_elements.append(Element("table", n, n, caption=caption_text, markdown=para.block.markdown,
                                                 degraded=para.block.degraded))
                    tables += 1
                    continue
                text = join_lines(para.lines, vocab)
                if layout.scanned:
                    text = repair_ocr(text, vocab)
                if para.kind == "caption" and para.caption_for == "figure":
                    saved = []
                    if figures_root is not None:
                        for rect in images_for[id(para)]:
                            figures_root.mkdir(parents=True, exist_ok=True)
                            name = f"p{n:04d}_{len(saved) + 1}.jpg"
                            doc[n - 1].get_pixmap(clip=pymupdf.Rect(rect), dpi=110).save(
                                str(figures_root / name), jpg_quality=80)
                            saved.append(f"{doc_id}/{name}")
                    page_elements.append(Element("figure", n, n, para.lines, images=saved, scanned=layout.scanned))
                    page_text.append(text)
                    figures += 1
                elif not is_noise(text):
                    page_elements.append(Element("heading" if para.kind == "heading" else "paragraph", n, n,
                                                 para.lines, size=para.size, scanned=layout.scanned))
                    page_text.append(text)
            joined = " ".join(page_text)
            quality = text_quality(joined)
            reason = reference.get(n)
            manifest_pages.append({
                "page": n, "label": page_label(n), "scanned": layout.scanned, "chars": len(joined),
                "alpha": quality["alpha"], "suspicious": quality["suspicious"], "poor": quality["poor"],
                "tables": tables, "figures": figures, "figures_without_caption": orphan_figures,
                "rotated_lines_dropped": layout.rotated_lines_dropped,
                "excluded": reason})
            result.tables += tables
            result.figures += figures
            result.scanned_pages += layout.scanned
            if reason:
                result.excluded_pages[n] = reason
                continue
            if len(joined) < 100:
                result.low_text_pages.append(n)
            if quality["poor"]:
                result.poor_quality_pages.append(n)
            path = toc_paths[n] if toc else []
            chapter = _chapter_of(path) if toc else header_chapter.get(n, "")
            for element in page_elements:
                element.toc_path = list(path) if toc else ([chapter] if chapter else [])
                element.chapter = chapter
                element.page_label = page_label(n)
                elements.append(element)

    chunks = chunk_elements(elements, vocab, doc_id=doc_id, source=Path(pdf).name)
    detect = book_config.detect if book_config else detect or DetectConfig()
    review, result.warnings, result.suspect_runs = review_pages(dims, numbers, result.excluded_pages,
                                                                book_config, detect)
    result.long_reference_sections = review["long_reference_sections"]
    result.config_excluded_ranges = review["config_excluded_ranges"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    result.chunks, result.pages = len(chunks), len(numbers)
    result.manifest = output.with_suffix(".manifest.json")
    result.manifest.write_text(json.dumps({
        "extractor_version": EXTRACTOR_VERSION, "source": Path(pdf).name, "doc_id": doc_id, "sha256": digest,
        "pages": len(numbers), "first_page": first_page, "last_page": end, "body_font_size": body_size,
        "page_label_offset": offset if has_labels else None, "chunks": len(chunks),
        "chunk_types": dict(Counter(c["type"] for c in chunks)), "scanned_pages": result.scanned_pages,
        "poor_quality_pages": result.poor_quality_pages, "low_text_pages": result.low_text_pages,
        "excluded_pages": result.excluded_pages, "ocr_pages": result.ocr_pages, "tables": result.tables,
        "figures_with_caption": result.figures, "page_info": manifest_pages} | review, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return result


def upgrade_manifest(manifest_path: Path, raw_dir: Path, configs: dict[str, BookConfig],
                     defaults: DetectConfig) -> tuple[str, bool]:
    """Add current-schema fields to an older manifest without touching its chunks. Returns (message, ok)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source"]
    if manifest.get("manifest_schema", 1) >= MANIFEST_SCHEMA:
        return f"{source}: manifest already at schema {MANIFEST_SCHEMA}", True
    pdf = raw_dir / source
    if not pdf.exists():
        return f"{source}: skipped, PDF not found in {raw_dir}", True
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        return f"{source}: skipped, PDF changed since ingestion; re-ingest this book", True
    config, notes = config_for(configs, source, digest)
    pages = manifest["page_info"]
    numbers = [p["page"] for p in pages]
    excluded = {p["page"]: p["excluded"] for p in pages if p["excluded"]}
    try:
        with pymupdf.open(pdf) as doc:
            dims = page_sizes(doc)
            if config is not None:
                _apply_ranges(excluded, config, len(doc))
    except ValueError as error:
        return f"{source}: refused, {error}", False
    jsonl = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".jsonl"))
    chunks = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    covered = sorted({n for c in chunks for n in range(c["page"], c["page_end"] + 1) if n in excluded
                      and excluded[n].startswith("config:")})
    if covered:
        return (f"{source}: refused, config excludes pages that have chunks ({covered[0]}..{covered[-1]}); "
                f"re-ingest this book instead"), False
    excluded = {n: excluded[n] for n in numbers if n in excluded}
    for page in pages:
        page["excluded"] = excluded.get(page["page"])
    review, warnings, _ = review_pages(dims, numbers, excluded, config, config.detect if config else defaults)
    manifest["excluded_pages"] = excluded
    manifest.update(review)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"{source}: manifest upgraded to schema {MANIFEST_SCHEMA} "
             f"({len(review['config_excluded_ranges'])} config exclusion ranges, "
             f"{len(review['suspect_page_runs'])} suspect page runs)"]
    return "\n".join(lines + [f"WARNING: {source}: {w}" for w in notes + warnings]), True


def main(argv: list[str] | None = None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="Extract PDF textbooks into structure-aware search chunks.")
    parser.add_argument("pdf", type=Path, nargs="*")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/books"))
    parser.add_argument("--figure-dir", type=Path, default=Path("data/processed/figures"))
    parser.add_argument("--no-figures", action="store_true", help="do not render figure images")
    parser.add_argument("--pages", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--ocr", action="store_true",
                        help="re-OCR scanned pages whose text layer is missing or damaged (needs Tesseract)")
    parser.add_argument("--config", type=Path, help="per-book settings (default: ingest_config.json if present)")
    parser.add_argument("--strict", action="store_true", help="exit with code 2 if any suspect page run is unreviewed")
    parser.add_argument("--upgrade-manifests", action="store_true",
                        help="add the current manifest fields to older manifests without re-ingesting")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="where --upgrade-manifests finds PDFs")
    args = parser.parse_args(argv)
    if not args.upgrade_manifests and not args.pdf:
        parser.error("give at least one PDF, or use --upgrade-manifests")

    config_path = args.config or Path("ingest_config.json")
    defaults, configs = DetectConfig(), {}
    if config_path.exists():
        try:
            defaults, configs = load_settings(config_path)
        except ValueError as error:
            parser.error(str(error))
    elif args.config:
        parser.error(f"config file not found: {args.config}")
    else:
        print(f"note: {config_path} not found; no per-book settings applied")

    if args.upgrade_manifests:
        refused = False
        for manifest_path in sorted(args.output_dir.glob("*.manifest.json")):
            message, ok = upgrade_manifest(manifest_path, args.raw_dir, configs, defaults)
            refused |= not ok
            print(message)
        return 1 if refused else 0

    unreviewed = False
    for pdf in args.pdf:
        suffix = f".p{args.pages[0]}-{args.pages[1]}" if args.pages else ""
        output = args.output_dir / (pdf.stem + suffix + ".jsonl")
        try:
            book, config_warnings = config_for(configs, pdf.name, hashlib.sha256(pdf.read_bytes()).hexdigest())
            result = ingest_pdf(pdf, output, *(args.pages or (1, None)), ocr=args.ocr,
                                figure_dir=None if args.no_figures else args.figure_dir,
                                book_config=book, detect=defaults)
        except (ValueError, RuntimeError, OSError) as error:
            parser.error(str(error))
        print(f"{pdf.name}: {result.pages} pages -> {result.chunks} chunks "
              f"({result.tables} tables, {result.figures} captioned figures); "
              f"{result.scanned_pages} scanned, {len(result.poor_quality_pages)} poor-quality, "
              f"{len(result.low_text_pages)} low-text, {len(result.excluded_pages)} excluded, "
              f"{len(result.ocr_pages)} re-OCR'd\n  {output}\n  {result.manifest}")
        for warning in config_warnings + result.warnings:
            print(f"WARNING: {pdf.name}: {warning}")
        unreviewed |= any(not run.reviewed for run in result.suspect_runs)
    return 2 if args.strict and unreviewed else 0


if __name__ == "__main__":
    raise SystemExit(main())
