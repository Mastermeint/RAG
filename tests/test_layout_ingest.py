"""Ingestion tests on hostile synthetic PDFs. Each fixture also has a control proving the OLD naive
extraction (page.get_text(sort=True) + whitespace collapse) fails, so a pass here is not due to a clean PDF."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

import fixtures
from rag_mvp.ingest import _find_tesseract, ingest_pdf
from rag_mvp.layout import extract_page


def naive_pages(pdf: Path) -> list[str]:
    with pymupdf.open(pdf) as doc:
        return [re.sub(r"\s+", " ", page.get_text(sort=True)).strip() for page in doc]


def markers(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"tok(\d{4})", text)]


class IngestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def ingest(self, builder, *args, name="book", **kwargs):
        pdf = self.root / f"{name}.pdf"
        builder(pdf, *args) if args else builder(pdf)
        output = self.root / "out" / f"{name}.jsonl"
        result = ingest_pdf(pdf, output, figure_dir=self.root / "figs", **kwargs)
        chunks = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
        return pdf, result, chunks, manifest


class ColumnsHeadersAndStructure(IngestCase):
    def test_two_column_pages_are_read_column_by_column(self):
        pdf, _, chunks, _ = self.ingest(fixtures.book_with_columns)
        found = markers(" ".join(c["text"] for c in chunks))
        unique = list(dict.fromkeys(found))  # chunk overlap repeats a trailing sentence on purpose
        self.assertEqual(unique, sorted(unique))
        self.assertEqual(len(unique), 288)  # every sentence of every page survives

    def test_control_naive_extraction_interleaves_the_columns(self):
        pdf, *_ = self.ingest(fixtures.book_with_columns)
        naive = markers(" ".join(naive_pages(pdf)))
        self.assertNotEqual(naive, sorted(naive))

    def test_running_header_and_page_numbers_are_removed_and_become_labels(self):
        _, _, chunks, manifest = self.ingest(fixtures.book_with_columns, 10)
        self.assertFalse([c for c in chunks if "MARINE BIOLOGY HANDBOOK" in c["text"]])
        self.assertEqual(manifest["page_label_offset"], 10)
        self.assertEqual({(c["page"], c["page_label"]) for c in chunks if c["page"] == 3}, {(3, 13)})

    def test_control_naive_extraction_keeps_header(self):
        pdf, *_ = self.ingest(fixtures.book_with_columns)
        self.assertIn("THE MARINE BIOLOGY HANDBOOK", naive_pages(pdf)[3])

    def test_contents_and_index_pages_are_excluded_via_outline(self):
        _, _, chunks, manifest = self.ingest(fixtures.book_with_columns)
        text = " ".join(c["text"] for c in chunks)
        self.assertNotIn("contentsonlyword", text)
        self.assertNotIn("indexonlyword", text)
        self.assertEqual(set(manifest["excluded_pages"]), {"1", "8"})

    def test_chunks_carry_outline_section_path_and_never_cross_chapters(self):
        _, _, chunks, _ = self.ingest(fixtures.book_with_columns)
        for chunk in chunks:
            expected = "Chapter 1 Coral Reefs" if chunk["page_end"] <= 4 else "Chapter 2 Reef Fishes"
            self.assertEqual(chunk["section_path"][0], expected, chunk["page"])
            self.assertEqual(chunk["chapter"], expected)
        self.assertEqual({tuple(c["section_path"]) for c in chunks if c["page"] == 2},
                         {("Chapter 1 Coral Reefs", "Reef Zonation")})

    def test_chunk_sizes_are_bounded(self):
        _, _, chunks, _ = self.ingest(fixtures.book_with_columns)
        words = [len(c["text"].split()) for c in chunks]
        self.assertLessEqual(max(words), 420)
        self.assertGreaterEqual(min(words), 8)

    def test_manifest_records_hash_and_one_row_per_page(self):
        import hashlib
        pdf, _, _, manifest = self.ingest(fixtures.book_with_columns)
        self.assertEqual(manifest["sha256"], hashlib.sha256(pdf.read_bytes()).hexdigest())
        self.assertEqual([p["page"] for p in manifest["page_info"]], list(range(1, 9)))

    def test_page_range_does_not_break_page_numbers(self):
        pdf = self.root / "book.pdf"
        fixtures.book_with_columns(pdf)
        result = ingest_pdf(pdf, self.root / "part.jsonl", first_page=3, last_page=5)
        chunks = [json.loads(l) for l in (self.root / "part.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual((result.pages, min(c["page"] for c in chunks), max(c["page_end"] for c in chunks)), (3, 3, 5))
        with self.assertRaises(ValueError):
            ingest_pdf(pdf, self.root / "bad.jsonl", first_page=3, last_page=99)


class BrokenWords(IngestCase):
    def test_hyphenation_ligatures_and_soft_hyphens_are_repaired(self):
        _, _, chunks, _ = self.ingest(fixtures.book_with_broken_words)
        text = " ".join(c["text"] for c in chunks)
        self.assertIn("Parrotfish scrape", text)  # ligature
        self.assertIn("reef system depends", text)  # 'sys-' + 'tem' rejoined
        self.assertIn("well-known fact", text)  # real compound kept
        self.assertIn("deep-sea fish", text)  # hyphen + soft hyphen
        self.assertNotRegex(text, r"[a-z]- [a-z]|\xad|\ufb01")

    def test_control_naive_extraction_leaves_all_four_defects(self):
        pdf, *_ = self.ingest(fixtures.book_with_broken_words)
        naive = naive_pages(pdf)[0]
        self.assertIn("sys- tem", naive)
        self.assertIn("well- known", naive)
        self.assertTrue("\xad" in naive or "\ufb01" in naive)


class PageBreaks(IngestCase):
    def test_sentence_split_by_a_page_break_stays_in_one_chunk_with_both_pages(self):
        _, _, chunks, _ = self.ingest(fixtures.book_with_page_break)
        hits = [c for c in chunks if "settle on rocky substrates" in c["text"]]
        self.assertEqual(len(hits), 1)
        self.assertEqual((hits[0]["page"] <= 2, hits[0]["page_end"] >= 3), (True, True))

    def test_control_naive_pages_split_the_sentence(self):
        pdf, *_ = self.ingest(fixtures.book_with_page_break)
        self.assertFalse(any("settle on rocky" in page for page in naive_pages(pdf)))


class Tables(IngestCase):
    def test_ruled_and_borderless_tables_become_table_chunks_with_captions(self):
        _, result, chunks, _ = self.ingest(fixtures.book_with_tables)
        tables = [c for c in chunks if c["type"] == "table"]
        self.assertEqual(result.tables, 2)
        ruled = next(c for c in tables if c["page"] == 2)["text"]
        self.assertIn("TABLE 2-1 Average mass of reef fish by species.", ruled)
        self.assertIn("| Species | Mass (g) | Length (cm) |", ruled)
        self.assertIn("| Grouper | 5400 | 62 |", ruled)
        borderless = next(c for c in tables if c["page"] == 3)["text"]
        self.assertIn("| Species | Bites per minute | Algae eaten (g) |", borderless)  # stacked header merged
        self.assertIn("| Parrotfish | 14.2 | 3.1 |", borderless)
        self.assertIn("| Damselfish | 21.0 | 0.4 |", borderless)

    def test_table_cells_do_not_leak_into_prose_chunks(self):
        _, _, chunks, _ = self.ingest(fixtures.book_with_tables)
        prose = " ".join(c["text"] for c in chunks if c["type"] == "text")
        for leaked in ("5400", "14.2", "Surgeonfish", "Bites per"):
            self.assertNotIn(leaked, prose)
        self.assertIn("groupers are by far the heaviest", prose)

    def test_control_naive_extraction_flattens_the_table(self):
        pdf, *_ = self.ingest(fixtures.book_with_tables)
        self.assertNotIn("| Grouper | 5400", " ".join(naive_pages(pdf)))


class Figures(IngestCase):
    def test_captioned_picture_is_saved_and_linked_and_its_labels_stay_out_of_prose(self):
        _, result, chunks, _ = self.ingest(fixtures.book_with_figures)
        figures = [c for c in chunks if c["type"] == "figure"]
        self.assertEqual(result.figures, 2)
        explicit = next(c for c in figures if c["page"] == 2)
        self.assertTrue(explicit["text"].startswith("FIGURE 3.2 Cross-section of a fringing reef"))
        self.assertEqual(len(explicit["images"]), 1)
        self.assertTrue((self.root / "figs" / explicit["images"][0]).stat().st_size > 500)
        prose = " ".join(c["text"] for c in chunks if c["type"] == "text")
        for noise in ("Reef crest", "Lagoon", "Stock Agency", "Photo credit"):
            self.assertNotIn(noise, prose)

    def test_picture_with_only_a_credit_line_still_gets_a_figure_chunk(self):
        _, _, chunks, _ = self.ingest(fixtures.book_with_figures)
        credit = next(c for c in chunks if c["type"] == "figure" and c["page"] == 3)
        self.assertIn("Parrotfish", credit["text"])
        self.assertEqual(len(credit["images"]), 1)

    def test_control_naive_extraction_mixes_labels_into_text(self):
        pdf, *_ = self.ingest(fixtures.book_with_figures)
        naive = naive_pages(pdf)[1]
        self.assertIn("Reef crest", naive)
        self.assertIn("Stock", naive)


class Scans(IngestCase):
    DAMAGED = ("TI1e eggs remain in the sand until spaw11 Han1lets tllat fotmding Sho~ Answer "
               "and T11e usual shape of a reef front")

    def test_scanned_pages_with_damaged_text_layer_are_detected_and_flagged(self):
        _, result, _, manifest = self.ingest(fixtures.scanned_book, self.DAMAGED)
        self.assertEqual(result.scanned_pages, 4)
        self.assertEqual(manifest["poor_quality_pages"], [1, 2, 3, 4])

    def test_clean_text_layer_on_scan_is_not_flagged(self):
        clean = "Coral reefs support many fish species and protect the shore from storm waves each year."
        _, _, _, manifest = self.ingest(fixtures.scanned_book, clean)
        self.assertEqual((manifest["scanned_pages"], manifest["poor_quality_pages"]), (4, []))

    def _fake_ocr(self, text):
        source = pymupdf.open()
        sheet = source.new_page(width=612, height=792)
        sheet.insert_textbox(pymupdf.Rect(72, 100, 540, 400), text, fontsize=10)
        self.addCleanup(source.close)
        return lambda page, number: extract_page(sheet, number)

    def test_ocr_replaces_a_damaged_layer_only_when_it_reads_better(self):
        better = "Coral reefs support many fish species and protect the shore from storm waves each year."
        _, result, chunks, _ = self.ingest(fixtures.scanned_book, self.DAMAGED, ocr=True,
                                           ocr_page=self._fake_ocr(better))
        self.assertEqual(result.ocr_pages, [1, 2, 3, 4])
        self.assertIn("Coral reefs support many fish species", chunks[0]["text"])
        self.assertNotIn("TI1e", chunks[0]["text"])

    def test_ocr_that_is_not_better_is_discarded(self):
        worse = "x~ y| z^ q{ w} e\\ r_ t< u> i1o p2q l3m n4p ~~ ^^ || {{ }} " * 4
        _, result, chunks, _ = self.ingest(fixtures.scanned_book, self.DAMAGED, ocr=True,
                                           ocr_page=self._fake_ocr(worse))
        self.assertEqual(result.ocr_pages, [])
        self.assertIn("fotmding", chunks[0]["text"])

    def test_missing_tesseract_is_an_error_and_leaves_existing_output_untouched(self):
        output = self.root / "out" / "book.jsonl"
        output.parent.mkdir()
        output.write_text("existing\n", encoding="utf-8")
        pdf = self.root / "book.pdf"
        fixtures.scanned_book(pdf, self.DAMAGED)
        with patch("rag_mvp.ingest._find_tesseract", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Tesseract"):
                ingest_pdf(pdf, output, ocr=True)
        self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")

    @unittest.skipUnless(_find_tesseract(), "Tesseract is not installed")
    def test_real_tesseract_reads_a_scan_that_has_no_text_layer(self):
        _, result, chunks, _ = self.ingest(fixtures.scanned_book, "", 4, False, ocr=True)
        self.assertEqual(result.ocr_pages, [1, 2, 3, 4])
        self.assertIn("coral reefs support many fish species", " ".join(c["text"] for c in chunks).lower())


if __name__ == "__main__":
    unittest.main()
