"""Regression tests for defects found by inspecting the REAL ingested textbooks (not by synthetic cleanliness)."""

import re
import unittest

import pymupdf

import fixtures
from rag_mvp.chunking import Element, chunk_elements
from rag_mvp.ingest import _CREDIT
from rag_mvp.layout import _degraded, _regrid, extract_page
from test_layout_ingest import IngestCase, naive_pages


class VectorFigures(IngestCase):
    """Fishes: 166 of 188 figures were drawn as vector paths and had no image at all."""

    def test_vector_chart_is_rendered_linked_and_its_labels_stay_out_of_prose(self):
        _, _, chunks, _ = self.ingest(fixtures.book_with_vector_figures)
        chart = next(c for c in chunks if c["type"] == "figure" and c["page"] == 2)
        self.assertTrue(chart["text"].startswith("FIGURE 4.1 Algal biomass"))
        self.assertEqual(len(chart["images"]), 1)
        picture = pymupdf.Pixmap(str(self.root / "figs" / chart["images"][0]))
        self.assertGreater(picture.width, 200)  # whole chart, not a sliver
        self.assertGreater(picture.height, 120)
        prose = " ".join(c["text"] for c in chunks if c["type"] == "text")
        for label in ("chartlabel", "Zone0", "Zone4"):
            self.assertNotIn(label, prose)
        self.assertIn("Biomass peaks at the reef crest", prose)

    def test_caption_with_nothing_drawn_still_gets_a_figure_chunk_without_an_image(self):
        _, _, chunks, _ = self.ingest(fixtures.book_with_vector_figures)
        missing = next(c for c in chunks if c["type"] == "figure" and c["page"] == 3)
        self.assertEqual(missing["images"], [])

    def test_control_without_vector_detection_the_chart_has_no_figure_region(self):
        pdf, *_ = self.ingest(fixtures.book_with_vector_figures)
        with pymupdf.open(pdf) as doc:
            self.assertEqual(len(doc[1].get_image_info()), 0)  # nothing embedded: the old code could not see it
            self.assertEqual(len(extract_page(doc[1], 2).figures), 1)
            self.assertEqual(len(extract_page(doc[3], 4).figures), 0)  # no caption, no drawings: no phantom figure
        self.assertIn("chartlabel", naive_pages(pdf)[1])


class RunningHeaderChapters(IngestCase):
    """Intro book: chapter 1 got no chapter at all because its header repeated fewer times than the threshold."""

    def test_short_first_chapter_is_still_identified_in_a_long_book_without_outline(self):
        _, _, chunks, manifest = self.ingest(fixtures.book_with_short_first_chapter)
        self.assertEqual(manifest["pages"], 340)
        by_page = {}
        for c in chunks:
            for page in range(c["page"], c["page_end"] + 1):
                by_page.setdefault(page, c["chapter"])
        self.assertEqual(by_page[3], "Chapter 1 Reef Basics")
        self.assertEqual(by_page[8], "Chapter 1 Reef Basics")
        self.assertEqual(by_page[50], "Chapter 2 Reef Fishes")
        self.assertEqual([c for c in chunks if not c["chapter"]], [])
        self.assertFalse([c for c in chunks if re.search(r"Chapter \d Reef", c["text"])])  # headers removed


class TableRepair(unittest.TestCase):
    """12e: about half of the tables came out with whole rows fused into one 250-1,100 character cell."""

    def test_fused_cells_are_detected(self):
        fused = "| a | b |\n| --- | --- |\n| " + "Pacific 166.2 64.2 4,188 " * 12 + " | x |"
        self.assertTrue(_degraded(fused, 3))
        tidy = "| Species | Mass |\n| --- | --- |\n| Grouper | 5400 |\n| Wrasse | 310 |"
        self.assertFalse(_degraded(tidy, 3))

    def test_regrid_rebuilds_rows_and_columns_from_word_positions(self):
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), "Table 2.1 Average depths of the major ocean basins", fontsize=10)
        rows = [("Ocean", "Mean depth (m)", "Deepest place"), ("Pacific", "4,188", "Marianas Trench"),
                ("Atlantic", "3,736", "Puerto Rico Trench"), ("Indian", "3,872", "Java Trench"),
                ("Arctic", "1,330", "Molloy Deep")]
        for r, row in enumerate(rows):
            for x, text in zip((72, 200, 340), row):
                page.insert_text((x, 130 + 16 * r), text, fontsize=10)
        block = _regrid(page, (60, 85, 560, 230), 1)
        doc.close()
        self.assertIsNotNone(block)
        self.assertEqual(block.caption, "Table 2.1 Average depths of the major ocean basins")
        lines = block.markdown.splitlines()
        self.assertEqual(lines[0], "| Ocean | Mean depth (m) | Deepest place |")
        self.assertIn("| Atlantic | 3,736 | Puerto Rico Trench |", lines)
        self.assertEqual((block.rows, block.cols), (5, 3))

    def test_regrid_gives_up_on_prose_instead_of_inventing_a_table(self):
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(pymupdf.Rect(72, 100, 540, 300), "Coral reefs support many fish species and protect "
                            "the shore from storm waves each year. " * 6, fontsize=10)
        block = _regrid(page, (60, 90, 560, 320), 1)
        doc.close()
        self.assertIsNone(block)


class ChunkSizeCap(unittest.TestCase):
    """12e glossary: a chunk reached 480 words because the carried overlap was added on top of a full unit."""

    @staticmethod
    def paragraph(first: int, words: int) -> Element:
        sentences, used, index = [], 0, first
        while used < words:
            sentence = f"Observation tok{index:04d} shows reef fish depend on coral for shelter and food."
            sentences.append(sentence)
            used += len(sentence.split())
            index += 1
        return Element("paragraph", 1, 1, [" ".join(sentences)], toc_path=["Ch"], chapter="Ch")

    def test_no_chunk_exceeds_the_cap_and_no_sentence_is_lost(self):
        sizes = [150, 390, 170, 380, 100, 700, 60, 250]
        elements, index = [], 1
        for size in sizes:
            element = self.paragraph(index, size)
            index += len(re.findall(r"tok\d{4}", element.lines[0]))
            elements.append(element)
        chunks = chunk_elements(elements, None, "doc", "b.pdf")
        for chunk in chunks:
            self.assertLessEqual(len(chunk["text"].split()), 420)
        seen = {int(m) for c in chunks for m in re.findall(r"tok(\d{4})", c["text"])}
        self.assertEqual(seen, set(range(1, index)))

    def test_soft_hyphens_never_survive_into_any_chunk_type(self):
        table = Element("table", 1, 1, caption="TABLE 10-3 Stage-\xadSpecific mortality",
                        markdown="| a | b |\n| --- | --- |\n| 1 | 2 |", toc_path=["Ch"], chapter="Ch")
        chunks = chunk_elements([table], None, "doc", "b.pdf")
        self.assertEqual(chunks[0]["text"].count("\xad"), 0)
        self.assertIn("Stage-Specific", chunks[0]["text"])

    def test_degraded_tables_are_flagged_for_downstream_users(self):
        table = Element("table", 1, 1, markdown="| a | b |\n| --- | --- |\n| 1 | 2 |", degraded=True,
                        toc_path=["Ch"], chapter="Ch")
        self.assertTrue(chunk_elements([table], None, "doc", "b.pdf")[0]["table_structure_degraded"])

    def test_a_page_of_glossary_asides_is_split_at_the_size_cap(self):
        """12e back-matter glossary: a page made only of asides first became one 1,110-word chunk."""
        entry = "Term{} The definition of this glossary term takes about twenty words to explain it fully here."
        asides = [Element("aside", 1, 1, [entry.format(i)], toc_path=["Glossary"], chapter="G") for i in range(60)]
        chunks = chunk_elements(asides, None, "doc", "b.pdf")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c["text"].split()) <= 380 for c in chunks))
        self.assertEqual(sum(c["text"].count("Term") for c in chunks), 60)


class ChapterOpenings(IngestCase):
    """12e/Fishes/Student's Guide defects fixed in extractor version 4."""

    def chunks(self):
        pdf, _, chunks, _ = self.ingest(fixtures.book_with_chapter_opening)
        return pdf, chunks

    def test_drop_cap_initial_is_glued_back_onto_its_word(self):
        pdf, chunks = self.chunks()
        text = " ".join(c["text"] for c in chunks)
        self.assertIn("Fishes were the first vertebrates", text)
        self.assertFalse([c for c in chunks if c["text"].startswith("ishes")])
        self.assertNotIn("Fishes were", naive_pages(pdf)[0])  # control: the initial is a separate glyph

    def test_margin_glossary_gets_its_own_chunk_and_the_prose_flows_around_it(self):
        pdf, chunks = self.chunks()
        glossary = [c for c in chunks if "selectively permeable" in c["text"]]
        self.assertEqual(len(glossary), 1)
        self.assertTrue(glossary[0]["text"].startswith("Osmosis The diffusion"))
        self.assertIn("Diffusion The movement", glossary[0]["text"])
        self.assertNotIn("Bony fishes", glossary[0]["text"])
        self.assertTrue([c for c in chunks if "every single day and their kidneys excrete" in c["text"]])
        self.assertTrue([c for c in chunks if "From food and feces collection:" in c["text"]
                         and "Egestion equals" in c["text"]])  # a short run-in label stays with its prose
        page = naive_pages(pdf)[0]  # control: the glossary sits between the two halves of the sentence
        self.assertLess(page.index("every single day"), page.index("selectively permeable"))
        self.assertLess(page.index("selectively permeable"), page.index("and their kidneys"))

    def test_caption_between_picture_and_credit_line_is_kept_with_the_credit(self):
        _, chunks = self.chunks()
        figures = [c for c in chunks if c["type"] == "figure" and c["page"] == 2]
        self.assertEqual([c["text"] for c in figures], ["A reef shark photographed at Roatan, Honduras. CC BY 2.5"])
        self.assertTrue([c for c in chunks if c["type"] == "text" and c["text"].startswith("Photosynthesis by")])

    def test_credit_keywords_must_be_whole_words(self):
        for text in ("Photosynthesis by the algae", "Phototrophic bacteria", "Credits and acknowledgements"):
            self.assertIsNone(_CREDIT.search(text), text)
        for text in ("Photo: A. Smith", "Source: Bill Ober", "Source:NOAA", "Courtesy of the museum", "CC BY 2.5"):
            self.assertIsNotNone(_CREDIT.search(text), text)

    def test_video_placeholder_never_reaches_a_chunk(self):
        _, chunks = self.chunks()
        self.assertFalse([c for c in chunks if "has been excluded" in c["text"] or "view it online" in c["text"]])


if __name__ == "__main__":
    unittest.main()
