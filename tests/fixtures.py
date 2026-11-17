"""Builders for hostile synthetic PDFs: each reproduces a layout failure seen in the real textbooks."""

from pathlib import Path

import pymupdf

FONT = pymupdf.Font("helv")
LETTER = pymupdf.paper_rect("letter")

_WORDS = ("reef coral fish algae lagoon larvae current plankton sponge urchin kelp tide shelf crest "
          "grazing spawning habitat predator sediment nutrient polyp tentacle shoal").split()


def filler(counter: list[int], sentences: int) -> str:
    """Sentences that carry a unique, increasing marker (tok0001, tok0002, ...) to prove reading order."""
    out = []
    for _ in range(sentences):
        counter[0] += 1
        a, b, c = (_WORDS[(counter[0] * k) % len(_WORDS)] for k in (3, 5, 7))
        out.append(f"Observation tok{counter[0]:04d} shows that {a} and {b} depend on {c} near the shelf.")
    return " ".join(out)


def write_line(page, point, text, size=10, font=FONT):
    writer = pymupdf.TextWriter(page.rect)
    writer.append(point, text, font=font, fontsize=size)
    writer.write_text(page)


def book_with_columns(path: Path, label_offset: int = 10) -> dict:
    """Eight pages: contents, two-column chapters with running header/page numbers, and an index."""
    counter = [0]
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER.width, height=LETTER.height)
    page.insert_text((72, 30), "THE MARINE BIOLOGY HANDBOOK", fontsize=9)
    page.insert_textbox(pymupdf.Rect(72, 100, 540, 400),
                        "Contents Chapter 1 Coral Reefs 12 Chapter 2 Reef Fishes 40 Index 77 contentsonlyword",
                        fontsize=10)
    page.insert_text((300, 770), str(1 + label_offset), fontsize=9)
    for number in range(2, 8):
        page = doc.new_page(width=LETTER.width, height=LETTER.height)
        page.insert_text((72, 30), "THE MARINE BIOLOGY HANDBOOK", fontsize=9)
        page.insert_text((300, 770), str(number + label_offset), fontsize=9)
        if number in (2, 5):
            title = "Reef Zonation" if number == 2 else "Reef Fish Feeding"
            page.insert_textbox(pymupdf.Rect(50, 60, 560, 95), title, fontsize=18, fontname="hebo")
        for x0 in (50, 320):  # left column text is generated first, so its markers are lower
            page.insert_textbox(pymupdf.Rect(x0, 110, x0 + 240, 740),
                                filler(counter, 8) + "\n\n" + filler(counter, 8) + "\n\n" + filler(counter, 8),
                                fontsize=10)
    page = doc.new_page(width=LETTER.width, height=LETTER.height)
    page.insert_text((72, 30), "THE MARINE BIOLOGY HANDBOOK", fontsize=9)
    page.insert_textbox(pymupdf.Rect(72, 100, 540, 400), "Index Algae, 12, 15 Anemones, 22 Corals, 3-9 indexonlyword",
                        fontsize=10)
    doc.set_toc([[1, "Contents", 1], [1, "Chapter 1 Coral Reefs", 2], [2, "Reef Zonation", 2],
                 [1, "Chapter 2 Reef Fishes", 5], [2, "Reef Fish Feeding", 5], [1, "Index", 8]])
    doc.save(path)
    doc.close()
    return {"markers": counter[0], "label_offset": label_offset}


def book_with_broken_words(path: Path) -> None:
    """Hyphenated line breaks, a real compound broken at its hyphen, a ligature, and a soft hyphen."""
    doc = pymupdf.open()
    for number in range(4):  # several pages so running-header logic has enough pages to work
        page = doc.new_page(width=LETTER.width, height=LETTER.height)
        page.insert_text((72, 30), "RUNNING HEADER TEXT", fontsize=9)
        lines = [
            "Parrot\ufb01sh scrape algae from the coral and the reef sys-",
            "tem depends on them for its health every single year. It is a well-",
            "known fact that deep-\xadsea fish live in cold water and that many",
            "shallow reef species are lost when the water warms up each summer.",
            "A well-known example is the crown-of-thorns starfish on warm reefs.",
        ]
        for i, line in enumerate(lines):
            write_line(page, (72, 100 + 12 * i), line)
    doc.save(path)
    doc.close()


def book_with_page_break(path: Path) -> None:
    """A sentence that is cut in half by a page break; both halves must end up in one chunk."""
    doc = pymupdf.open()
    for number in range(4):
        page = doc.new_page(width=LETTER.width, height=LETTER.height)
        page.insert_text((72, 30), "RUNNING HEADER TEXT", fontsize=9)
        if number == 1:
            page.insert_textbox(pymupdf.Rect(72, 600, 540, 700),
                                "Many reef species reproduce by broadcast spawning each year. The larvae of most "
                                "reef fishes drift in the plankton and eventually settle on", fontsize=10)
        elif number == 2:
            page.insert_textbox(pymupdf.Rect(72, 72, 540, 200),
                                "rocky substrates where they grow into juveniles and join the adult population "
                                "on the reef for many years.", fontsize=10)
        else:
            page.insert_textbox(pymupdf.Rect(72, 100, 540, 300),
                                "Kelp forests shelter many young fish from predators during winter storms.",
                                fontsize=10)
    doc.save(path)
    doc.close()


def book_with_tables(path: Path) -> None:
    """A ruled table (page 2) and a borderless table whose header is stacked (page 3), both captioned."""
    doc = pymupdf.open()
    for number in range(4):
        page = doc.new_page(width=LETTER.width, height=LETTER.height)
        page.insert_text((72, 30), "RUNNING HEADER TEXT", fontsize=9)
        page.insert_textbox(pymupdf.Rect(72, 60, 540, 100),
                            "Mean body mass differs among the reef fish species surveyed in the lagoon.", fontsize=10)
        if number == 1:
            page.insert_text((72, 130), "TABLE 2-1 Average mass of reef fish by species.", fontsize=9)
            rows = [["Species", "Mass (g)", "Length (cm)"], ["Parrotfish", "1200", "34"],
                    ["Damselfish", "45", "9"], ["Grouper", "5400", "62"], ["Wrasse", "310", "21"]]
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    cell_rect = pymupdf.Rect(72 + 120 * c, 145 + 22 * r, 192 + 120 * c, 167 + 22 * r)
                    page.draw_rect(cell_rect, color=(0, 0, 0), width=0.8)
                    page.insert_text((cell_rect.x0 + 4, cell_rect.y1 - 7), cell, fontsize=10)
            page.insert_textbox(pymupdf.Rect(72, 290, 540, 400),
                                "The table shows that groupers are by far the heaviest of the fishes we weighed.",
                                fontsize=10)
        if number == 2:
            page.insert_text((72, 130), "TABLE 3-1   Feeding rates of reef fishes in the lagoon.", fontsize=9)
            for x, text in ((72, "Species"), (232, "Bites per"), (352, "Algae")):
                page.insert_text((x, 160), text, fontsize=10)
            for x, text in ((232, "minute"), (352, "eaten (g)")):
                page.insert_text((x, 172), text, fontsize=10)
            data = [("Parrotfish", "14.2", "3.1"), ("Surgeonfish", "9.8", "2.2"),
                    ("Rabbitfish", "7.5", "1.9"), ("Damselfish", "21.0", "0.4")]
            for r, row in enumerate(data):
                for x, text in zip((72, 232, 352), row):
                    page.insert_text((x, 196 + 17 * r), text, fontsize=10)
            page.insert_textbox(pymupdf.Rect(72, 300, 540, 400),
                                "Parrotfish feed faster than any of the other herbivores that live on the reef.",
                                fontsize=10)
    doc.save(path)
    doc.close()


def book_with_figures(path: Path) -> None:
    """Embedded picture with inner labels, rotated credit, 'FIGURE' caption; plus a picture with only a credit line."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 200), False)
    pix.set_rect(pix.irect, (60, 120, 200))
    doc = pymupdf.open()
    for number in range(4):
        page = doc.new_page(width=LETTER.width, height=LETTER.height)
        page.insert_text((72, 30), "RUNNING HEADER TEXT", fontsize=9)
        page.insert_textbox(pymupdf.Rect(72, 60, 540, 130),
                            "Fringing reefs grow close to the shore and form a shallow lagoon behind the crest.",
                            fontsize=10)
        if number == 1:
            picture = pymupdf.Rect(150, 150, 450, 350)
            page.insert_image(picture, pixmap=pix)
            page.insert_text((200, 200), "Reef crest", fontsize=8, color=(1, 1, 1))
            page.insert_text((300, 300), "Lagoon", fontsize=8, color=(1, 1, 1))
            page.insert_text((470, 340), "Photo credit Stock Agency", fontsize=7, rotate=90)
            page.insert_textbox(pymupdf.Rect(150, 360, 450, 400),
                                "FIGURE 3.2 Cross-section of a fringing reef showing the crest and the lagoon.",
                                fontsize=9)
            page.insert_textbox(pymupdf.Rect(72, 420, 540, 520),
                                "The crest takes the full force of the waves while the lagoon stays calm.", fontsize=10)
        if number == 2:
            page.insert_image(pymupdf.Rect(150, 150, 450, 350), pixmap=pix)
            page.insert_textbox(pymupdf.Rect(150, 354, 450, 380), "\u201cParrotfish\u201d by Wikimedia [CC by 2.0]",
                                fontsize=7)
            page.insert_textbox(pymupdf.Rect(72, 420, 540, 520),
                                "Parrotfish bite coral to scrape the algae that grow on it.", fontsize=10)
    doc.save(path)
    doc.close()


def scanned_book(path: Path, layer_text: str, pages: int = 4, with_layer: bool = True) -> None:
    """Each page is a full-page bitmap of printed text, optionally with an invisible (OCR-style) text layer."""
    printed = pymupdf.open()
    sheet = printed.new_page(width=LETTER.width, height=LETTER.height)
    sheet.insert_textbox(pymupdf.Rect(72, 100, 540, 400),
                         "Coral reefs support many fish species and protect the shore from storm waves.",
                         fontsize=18)
    bitmap = sheet.get_pixmap(dpi=150)
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=LETTER.width, height=LETTER.height)
        page.insert_image(page.rect, pixmap=bitmap)
        if with_layer:
            page.insert_textbox(pymupdf.Rect(72, 100, 540, 400), layer_text, fontsize=10, render_mode=3)
    doc.save(path)
    doc.close()
    printed.close()


def book_with_vector_figures(path: Path) -> None:
    """A bar chart drawn with vector paths (no embedded picture) and a caption with nothing drawn above it."""
    doc = pymupdf.open()
    for number in range(4):
        page = doc.new_page(width=LETTER.width, height=LETTER.height)
        page.insert_text((72, 30), "RUNNING HEADER TEXT", fontsize=9)
        page.insert_textbox(pymupdf.Rect(72, 60, 540, 130),
                            "Grazing pressure differs between the reef crest and the lagoon floor.", fontsize=10)
        if number == 1:
            left, top, right, bottom = 150, 160, 450, 330
            page.draw_rect(pymupdf.Rect(left, top, right, bottom), color=(0, 0, 0), width=1)
            for i in range(5):
                page.draw_rect(pymupdf.Rect(left + 20 + i * 55, bottom - 20 - i * 25, left + 50 + i * 55, bottom),
                               color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
                page.insert_text((left + 22 + i * 55, bottom + 12), f"Zone{i}", fontsize=7)
            page.insert_text((200, 185), "Biomass chartlabel", fontsize=8)
            page.insert_textbox(pymupdf.Rect(150, 360, 450, 400),
                                "FIGURE 4.1 Algal biomass at five zones across the reef.", fontsize=9)
            page.insert_textbox(pymupdf.Rect(72, 420, 540, 520),
                                "Biomass peaks at the reef crest where grazing is lowest.", fontsize=10)
        if number == 2:
            page.insert_textbox(pymupdf.Rect(150, 300, 450, 340),
                                "FIGURE 5.1 A caption whose drawing is missing from the page.", fontsize=9)
            page.insert_textbox(pymupdf.Rect(72, 420, 540, 520),
                                "Another paragraph about parrotfish grazing on the reef.", fontsize=10)
    doc.save(path)
    doc.close()


def book_with_odd_pages(path: Path, normal: int = 20, odd: int = 12, odd_size: tuple[int, int] = (400, 520),
                        position: str = "end") -> None:
    """Letter-sized text pages plus a run of pages with a different size (start, middle or end)."""
    counter = [0]
    sizes = [(LETTER.width, LETTER.height)] * normal
    at = {"start": 0, "middle": normal // 2, "end": normal}[position]
    sizes[at:at] = [odd_size] * odd
    doc = pymupdf.open()
    for width, height in sizes:
        page = doc.new_page(width=width, height=height)
        page.insert_textbox(pymupdf.Rect(36, 50, width - 36, height - 50), filler(counter, 6), fontsize=10)
    doc.save(path)
    doc.close()


def book_with_short_first_chapter(path: Path, pages: int = 340) -> None:
    """No outline. Chapter 1 is short, so its running header repeats only 4 times in a long book."""
    doc = pymupdf.open()
    counter = [0]
    for number in range(1, pages + 1):
        chapter = 1 if number <= 8 else 2 + (number - 9) // 83
        title = {1: "Reef Basics", 2: "Reef Fishes", 3: "Reef Plants", 4: "Reef Threats", 5: "Reef Futures"}[chapter]
        page = doc.new_page(width=LETTER.width, height=LETTER.height)
        if number % 2:
            page.insert_text((72, 30), f"Chapter {chapter} {title} {number}", fontsize=9)
        else:
            page.insert_text((72, 30), f"{number} Part One The Reef Book", fontsize=9)
        page.insert_textbox(pymupdf.Rect(72, 100, 540, 700), filler(counter, 6), fontsize=10)
    doc.save(path)
    doc.close()
