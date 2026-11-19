"""Page-level PDF layout extraction: text blocks in reading order, tables and figure regions."""

import re
from dataclasses import dataclass, field

import pymupdf

from rag_mvp.textclean import clean_chars, collapse_space, strip_soft_hyphens

MARGIN_FRACTION = 0.09  # top/bottom page band searched for running headers, footers and page numbers
SCAN_COVERAGE = 0.80  # an image covering this share of the page marks it as a scan
MIN_FIGURE_SIDE = 60.0


@dataclass
class Block:
    page: int
    bbox: tuple[float, float, float, float]
    kind: str = "text"  # "text" or "table"
    lines: list[str] = field(default_factory=list)
    size: float = 0.0
    bold: bool = False
    margin: str | None = None  # "top" or "bottom" for candidates for header/footer removal
    lead_bold: bool = False  # first line opens with a short bold term followed by regular text (run-in heading)
    markdown: str = ""
    caption: str = ""
    rows: int = 0
    cols: int = 0
    degraded: bool = False  # table whose cells could not be separated reliably

    x0 = property(lambda self: self.bbox[0])
    y0 = property(lambda self: self.bbox[1])
    x1 = property(lambda self: self.bbox[2])
    y1 = property(lambda self: self.bbox[3])
    width = property(lambda self: self.bbox[2] - self.bbox[0])

    @property
    def raw_text(self) -> str:
        return " ".join(self.lines)


@dataclass
class PageLayout:
    number: int
    width: float
    height: float
    scanned: bool = False
    blocks: list[Block] = field(default_factory=list)  # body blocks in reading order
    margins: list[Block] = field(default_factory=list)  # header/footer candidates, top to bottom
    figures: list[tuple[float, float, float, float]] = field(default_factory=list)
    rotated_lines_dropped: int = 0


def _vertical_cut(blocks: list[Block], min_gutter: float) -> float | None:
    spans = sorted((b.x0, b.x1) for b in blocks)
    total_height = max(b.y1 for b in blocks) - min(b.y0 for b in blocks)
    reach = spans[0][1]
    for start, end in spans[1:]:
        if start - reach >= min_gutter:
            left = [b for b in blocks if b.x1 <= reach + 0.5]
            right = [b for b in blocks if b.x0 >= start - 0.5]
            side_heights = [sum(b.y1 - b.y0 for b in side) for side in (left, right)]
            if len(left) + len(right) == len(blocks) and min(side_heights) >= 0.12 * total_height:
                return reach
        reach = max(reach, end)
    return None


def _horizontal_cut(blocks: list[Block]) -> float | None:
    ordered = sorted(blocks, key=lambda b: b.y0)
    group_width = max(b.x1 for b in blocks) - min(b.x0 for b in blocks)
    wide = [b for b in blocks if b.width >= 0.6 * group_width]
    reach = ordered[0].y1
    first_gap = None
    for block in ordered[1:]:
        if block.y0 > reach + 0.5:
            if first_gap is None:
                first_gap = reach
            if any(abs(w.y1 - reach) <= 1.0 or abs(w.y0 - block.y0) <= 1.0 for w in wide):
                return reach
        reach = max(reach, block.y1)
    return first_gap


def reading_order(blocks: list[Block], min_gutter: float = 8.0) -> list[Block]:
    """Recursive XY-cut: split on column gutters first, so a column is read fully before the next."""
    if len(blocks) < 2:
        return list(blocks)
    x_cut = _vertical_cut(blocks, min_gutter)
    if x_cut is not None:
        left = [b for b in blocks if b.x1 <= x_cut + 0.5]
        right = [b for b in blocks if b.x1 > x_cut + 0.5]
        return reading_order(left, min_gutter) + reading_order(right, min_gutter)
    y_cut = _horizontal_cut(blocks)
    if y_cut is not None:
        top = [b for b in blocks if b.y1 <= y_cut + 0.5]
        bottom = [b for b in blocks if b.y1 > y_cut + 0.5]
        if top and bottom:
            return reading_order(top, min_gutter) + reading_order(bottom, min_gutter)
    return sorted(blocks, key=lambda b: (b.y0, b.x0))


def _in_rect(line: "_Line", rect: tuple) -> bool:
    return rect[0] <= (line.x0 + line.x1) / 2 <= rect[2] and rect[1] <= line.cy <= rect[3]


def _table_markdown(rows: list[list[str | None]]) -> tuple[str, int, int]:
    cleaned = [[collapse_space(strip_soft_hyphens(clean_chars((cell or "").replace("\n", " ")))).replace("|", "/")
                for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return "", 0, 0
    width = max(len(row) for row in cleaned)
    keep = [i for i in range(width) if any(i < len(row) and row[i] for row in cleaned)]
    grid = [[row[i] if i < len(row) else "" for i in keep] for row in cleaned]
    lines = ["| " + " | ".join(grid[0]) + " |", "| " + " | ".join("---" for _ in keep) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in grid[1:]]
    return "\n".join(lines), len(grid), len(keep)


def _find_tables(page: pymupdf.Page, number: int) -> list[Block]:
    # Ruled tables only: pages without vector rules cannot contain one, so skip the costly search.
    if len(page.get_cdrawings()) < 4:
        return []
    page_area = page.rect.get_area()
    tables = []
    try:
        found = page.find_tables().tables
    except Exception:
        return []
    for table in found:
        markdown, rows, cols = _table_markdown(table.extract())
        cells = [c for line in markdown.splitlines()[2:] for c in line.strip("|").split("|")] if rows > 1 else []
        filled = sum(bool(c.strip()) for c in cells)
        mean_cell = sum(len(c.strip()) for c in cells) / max(len(cells), 1)
        area = pymupdf.Rect(table.bbox).get_area()
        if (rows >= 3 and cols >= 2 and filled / max(len(cells), 1) >= 0.4 and mean_cell <= 120
                and len(markdown) >= 40 and area <= 0.7 * page_area):
            block = Block(number, tuple(table.bbox), "table", markdown=markdown, rows=rows, cols=cols)
            if _degraded(markdown, rows):
                rebuilt = _regrid(page, tuple(table.bbox), number)
                block = rebuilt or block
                block.degraded = rebuilt is None
            tables.append(block)
    return tables


RUN_IN_MAX_WORDS = 6


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _line_stats(line: dict) -> tuple[str, float, float, bool] | None:
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return None
    chars = sum(len(s["text"]) for s in spans)
    size = sum(s["size"] * len(s["text"]) for s in spans) / chars
    bold = sum(len(s["text"]) for s in spans if _is_bold(s)) / chars
    lead = 0
    while lead < len(spans) and _is_bold(spans[lead]):
        lead += 1
    lead_text = " ".join(s["text"] for s in spans[:lead])
    lead_bold = 0 < lead < len(spans) and len(lead_text.split()) <= RUN_IN_MAX_WORDS
    return clean_chars("".join(s["text"] for s in line["spans"])).strip(), size, bold, lead_bold


@dataclass
class _Line:
    box: tuple[float, float, float, float]
    text: str
    size: float
    bold: float
    lead_bold: bool
    source: int  # index of the PDF text block this line came from

    x0 = property(lambda self: self.box[0])
    y0 = property(lambda self: self.box[1])
    x1 = property(lambda self: self.box[2])
    y1 = property(lambda self: self.box[3])
    cy = property(lambda self: (self.box[1] + self.box[3]) / 2)


_TABLE_LABEL = re.compile(r"^\s*table\s*[A-Z]?\d", re.I)


def _is_tabular(row: list[_Line]) -> bool:
    """A visual row of several short, separated cells: table-like, unlike a wrapped line of prose."""
    if len(row) < 2 or any(len(c.text) > 60 for c in row):
        return False
    if sum(len(c.text) for c in row) / len(row) > 40:
        return False
    return all(b.x0 - a.x1 >= 4 for a, b in zip(row, row[1:]))


def _visual_rows(lines: list[_Line], size: float) -> list[list[_Line]]:
    rows: list[list[_Line]] = []
    for ln in sorted(lines, key=lambda l: (l.cy, l.x0)):
        if rows and abs(ln.cy - rows[-1][0].cy) <= 0.4 * size:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    return [sorted(r, key=lambda l: l.x0) for r in rows]


def _grid_markdown(taken: list[list[_Line]], size: float) -> tuple[str, int, int]:
    """Align cells of visual rows to column spans; stacked lines of one cell are merged."""
    spans: list[list[float]] = []
    for a, b in sorted((c.x0, c.x1) for r in taken if len(r) >= 2 for c in r):
        if spans and a <= spans[-1][1] + 2:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    if not spans:
        return "", 0, 0
    grid: list[tuple[float, list[str]]] = []
    for r in taken:
        cols = [""] * len(spans)
        for c in r:
            j = max(range(len(spans)), key=lambda k: min(c.x1, spans[k][1]) - max(c.x0, spans[k][0]))
            cols[j] = (cols[j] + " " + c.text).strip()
        if grid and r[0].cy - grid[-1][0] <= 1.25 * size:
            grid[-1] = (r[0].cy, [(a + " " + b).strip() for a, b in zip(grid[-1][1], cols)])
        else:
            grid.append((r[0].cy, cols))
    return _table_markdown([cols for _, cols in grid])


def _max_cell(markdown: str) -> int:
    cells = [c for line in markdown.splitlines()[2:] for c in line.strip("|").split("|")]
    return max((len(c.strip()) for c in cells), default=0)


def _degraded(markdown: str, rows: int) -> bool:
    """Whole rows or columns fused into one cell: the signature of a ruled table PyMuPDF could not split."""
    return _max_cell(markdown) > 150 or (rows <= 3 and _max_cell(markdown) > 100)


def _regrid(page: pymupdf.Page, bbox: tuple, number: int) -> Block | None:
    """Rebuild a table from word positions: split each text line into cells at wide gaps."""
    clip = pymupdf.Rect(bbox)
    by_line: dict[tuple[int, int], list[tuple]] = {}
    for word in page.get_text("words", clip=clip):
        by_line.setdefault((word[5], word[6]), []).append(word)
    cells: list[_Line] = []
    heights = []
    for words in by_line.values():
        words.sort(key=lambda w: w[0])
        heights.append(sum(w[3] - w[1] for w in words) / len(words))
        group = [words[0]]
        for prev, word in zip(words, words[1:]):
            if word[0] - prev[2] > max(7.0, 0.8 * (prev[3] - prev[1])):
                cells.append(group)
                group = []
            group.append(word)
        cells.append(group)
    if not heights:
        return None
    size = sorted(heights)[len(heights) // 2] / 1.2  # word boxes are about 1.2x taller than the font size
    items = [_Line((g[0][0], min(w[1] for w in g), g[-1][2], max(w[3] for w in g)),
                   clean_chars(" ".join(w[4] for w in g)), size, 0.0, False, 0) for g in cells]
    rows = _visual_rows(items, size)
    first = next((i for i, r in enumerate(rows) if len(r) >= 2), None)
    if first is None:
        return None
    head = [c.text for r in rows[:first] for c in r]
    caption = " ".join(head) if head and _TABLE_LABEL.match(head[0]) else ""
    body = rows[first:] if caption else rows
    markdown, n_rows, n_cols = _grid_markdown(body, size)
    if n_rows < 3 or n_cols < 2 or _degraded(markdown, n_rows):
        return None
    return Block(number, tuple(bbox), "table", markdown=markdown, rows=n_rows, cols=n_cols, caption=caption)


def _text_tables(lines: list[_Line], number: int) -> list[Block]:
    """Borderless tables (rules only above/below the header): anchored on a 'TABLE n' caption label."""
    labels = [ln for ln in lines if _TABLE_LABEL.match(ln.text)]
    if not labels:
        return []
    size = sorted(ln.size for ln in lines)[len(lines) // 2]
    rows = _visual_rows(lines, size)
    found: list[Block] = []
    claimed: set[int] = set()
    for label in labels:
        start = next((i for i, r in enumerate(rows) if r[0].cy > label.y0 and _is_tabular(r)
                      and r[0].y0 - label.y1 <= 14 * size and id(r[0]) not in claimed), None)
        if start is None:
            continue
        taken: list[list[_Line]] = []
        i = start
        while i < len(rows):
            row = rows[i]
            pitch = row[0].cy - taken[-1][0].cy if taken else 0
            if _is_tabular(row) and pitch <= 3.2 * size:
                taken.append(row)
            elif (taken and len(row) == 1 and len(row[0].text) <= 40 and pitch <= 2.2 * size
                  and i + 1 < len(rows) and _is_tabular(rows[i + 1])):
                taken.append(row)
            else:
                break
            i += 1
        if sum(_is_tabular(r) for r in taken) < 3:
            continue
        cells = [c for r in taken for c in r]
        top = rows[start][0].y0
        caption = [label] + [ln for ln in lines if ln is not label and ln.y1 <= top + 1 and not any(ln is c for c in cells)
                             and ((label.y0 - 3 * size <= ln.y0 <= label.y0 and ln.x0 >= label.x1 - 2)
                                  or (label.y0 < ln.y0 and ln.cy < top and ln.x0 >= label.x0 - 2))]
        markdown, n_rows, n_cols = _grid_markdown(taken, size)
        if n_rows < 3 or n_cols < 2:
            continue
        box_lines = caption + cells
        bbox = (min(c.x0 for c in box_lines), min(c.y0 for c in box_lines),
                max(c.x1 for c in box_lines), max(c.y1 for c in box_lines))
        ordered_caption = [label] + sorted((c for c in caption if c is not label), key=lambda c: (c.y0, c.x0))
        block = Block(number, bbox, "table", markdown=markdown, rows=n_rows, cols=n_cols,
                      caption=" ".join(c.text for c in ordered_caption))
        found.append(block)
        claimed.update(id(r[0]) for r in taken)
    return found


_FIGURE_LABEL = re.compile(r"^\W{0,2}(fig(?:ure)?|f[il1]gure)\b\.?\s*[A-Z]?\d", re.I)
MIN_DRAWINGS = 4


def _vector_figures(page: pymupdf.Page, lines: list[_Line], pictures: list[pymupdf.Rect]) -> list[tuple]:
    """Charts and diagrams drawn with vector paths: the drawing cluster touching a 'Figure n' caption."""
    captions = [ln for ln in lines if _FIGURE_LABEL.match(ln.text)]
    if not captions:
        return []
    page_rect = page.rect
    drawings = []
    for item in page.get_cdrawings():
        r = pymupdf.Rect(item["rect"])
        if r.width >= 0.9 * page_rect.width and r.height < 3 or r.get_area() > 0.7 * page_rect.get_area():
            continue  # page-wide rules and backgrounds
        drawings.append(r + (-1.5, -1.5, 1.5, 1.5))
    found: list[tuple] = []
    for cap in captions:
        near = [d for d in drawings if cap.y0 - 450 <= d.y0 and d.y1 <= cap.y1 + 450]
        clusters: list[list] = []  # [rect, member count]
        for d in near:
            touching = [c for c in clusters if (c[0] + (-8, -8, 8, 8)).intersects(d)]
            merged = [pymupdf.Rect(d), 1]
            for c in touching:
                merged[0] |= c[0]
                merged[1] += c[1]
                clusters.remove(c)
            clusters.append(merged)
        best = None
        for rect, count in clusters:
            horizontal = _overlap(rect.x0, rect.x1, cap.x0, cap.x1)
            above, below = cap.y0 - rect.y1, rect.y0 - cap.y1
            gap = above if -3 <= above <= 40 else below if -3 <= below <= 40 else None
            if (gap is None or count < MIN_DRAWINGS or rect.width < MIN_FIGURE_SIDE or rect.height < 40
                    or horizontal < 0.2 * min(rect.width, cap.x1 - cap.x0)):
                continue
            if best is None or (gap, -rect.get_area()) < (best[0], -best[1].get_area()):
                best = (gap, rect)
        if best is None or any(_overlap(p.x0, p.x1, best[1].x0, best[1].x1) > 0 and
                               _overlap(p.y0, p.y1, best[1].y0, best[1].y1) > 0.5 * p.height for p in pictures):
            continue
        region = pymupdf.Rect(best[1])
        for ln in lines:  # axis tick labels and titles sit just outside the drawn frame
            if ln is not cap and len(ln.text) <= 60 and (region + (-14, -14, 14, 14)).contains(
                    pymupdf.Point((ln.x0 + ln.x1) / 2, ln.cy)) and not _FIGURE_LABEL.match(ln.text):
                region |= pymupdf.Rect(ln.box)
        region &= page_rect
        if region.y1 > cap.y0 and region.y0 < cap.y0:
            region.y1 = cap.y0 - 1
        elif region.y0 < cap.y1 and region.y1 > cap.y1:
            region.y0 = cap.y1 + 1
        found.append(tuple(region))
    return found


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


DROP_CAP_SCALE = 2.5  # a decorated initial is at least this many times the median line size
_DROP_CAP = re.compile(r"^[\"“‘']?[A-Z]$")


def _attach_drop_caps(lines: list[_Line]) -> list[_Line]:
    """Glue an oversized chapter-opening initial ('F' + 'ishes were...') back onto the line beside it."""
    if len(lines) < 2:
        return lines
    median = sorted(ln.size for ln in lines)[len(lines) // 2]
    caps = [ln for ln in lines if _DROP_CAP.match(ln.text) and ln.size >= DROP_CAP_SCALE * median]
    for cap in caps:
        beside = [ln for ln in lines if ln is not cap and 0 <= ln.x0 - cap.x1 + 3 <= 3 * median + 3
                  and cap.y0 - 0.5 * median <= ln.cy <= cap.y1 and ln.text[:1].isalpha()]
        if not beside:
            continue
        target = min(beside, key=lambda ln: ln.y0)
        target.text = cap.text + target.text
        lines = [ln for ln in lines if ln is not cap]
    return lines


def extract_page(page: pymupdf.Page, number: int, textpage: pymupdf.TextPage | None = None) -> PageLayout:
    rect = page.rect
    layout = PageLayout(number, rect.width, rect.height)
    images = []
    for info in page.get_image_info():
        image = pymupdf.Rect(info["bbox"]) & rect
        if image.is_empty:
            continue
        if image.get_area() >= SCAN_COVERAGE * rect.get_area():
            layout.scanned = True
        elif image.width >= MIN_FIGURE_SIDE and image.height >= MIN_FIGURE_SIDE:
            images.append(image)
    layout.figures = [tuple(i) for i in images]

    lines: list[_Line] = []
    for index, raw in enumerate(page.get_text("dict", textpage=textpage)["blocks"]):
        if raw["type"] != 0:
            continue
        for line in raw["lines"]:
            stats = _line_stats(line)
            if stats is None:
                continue
            dx, dy = line["dir"]
            if abs(dy) > 0.3 or dx < 0:  # rotated text: photo credits, axis titles
                layout.rotated_lines_dropped += 1
                continue
            lines.append(_Line(tuple(line["bbox"]), *stats, index))

    if not layout.scanned:
        lines = _attach_drop_caps(lines)
        layout.figures += _vector_figures(page, lines, images)
    labels = [ln for ln in lines if _TABLE_LABEL.match(ln.text)]
    # Rules also frame diagrams and sidebars; a real table has a 'TABLE n' caption close by.
    tables = [] if layout.scanned else [t for t in _find_tables(page, number) if any(
        min(abs(lb.y0 - t.y1), abs(t.y0 - lb.y1)) <= 90 or t.y0 <= lb.cy <= t.y1 for lb in labels)]
    free = [ln for ln in lines if not any(_in_rect(ln, t.bbox) for t in tables)]
    outside_figures = [ln for ln in free if not any(_in_rect(ln, f) for f in layout.figures)]
    tables += _text_tables(outside_figures, number)

    blocks: list[Block] = []
    current: list[_Line] = []

    def flush() -> None:
        if current:
            sizes = [c.size for c in current]
            blocks.append(Block(
                number, (min(c.x0 for c in current), min(c.y0 for c in current),
                         max(c.x1 for c in current), max(c.y1 for c in current)),
                "text", [c.text for c in current], sum(sizes) / len(sizes),
                sum(c.bold for c in current) / len(current) >= 0.9, lead_bold=current[0].lead_bold))
            current.clear()

    for ln in lines:
        if any(_in_rect(ln, t.bbox) for t in tables):
            continue
        if current and (ln.source != current[-1].source
                        # Font size is unreliable on OCR layers, so only split on size change for native text.
                        or (not layout.scanned and abs(ln.size - current[-1].size) >= 1.0)):
            flush()
        current.append(ln)
    flush()

    band = MARGIN_FRACTION * rect.height
    body = []
    for block in blocks + tables:
        if block.kind == "text" and block.y1 <= band:
            block.margin = "top"
            layout.margins.append(block)
        elif block.kind == "text" and block.y0 >= rect.height - band:
            block.margin = "bottom"
            layout.margins.append(block)
        else:
            body.append(block)
    layout.margins.sort(key=lambda b: (b.y0, b.x0))
    layout.blocks = reading_order(body)
    return layout
