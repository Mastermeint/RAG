import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import fixtures
from rag_mvp.cli import main
from rag_mvp.ingest import main as ingest_main


class SearchCli(unittest.TestCase):
    def setUp(self):
        # These tests cover word-matching output; keep them independent of whether the embedding model is installed.
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        previous = os.environ.get("RAG_MODEL_DIR")
        os.environ["RAG_MODEL_DIR"] = empty.name
        self.addCleanup(lambda: os.environ.pop("RAG_MODEL_DIR") if previous is None
                        else os.environ.__setitem__("RAG_MODEL_DIR", previous))

    def run_cli(self, *argv):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_result_shows_pages_printed_page_type_and_section(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.jsonl").write_text(json.dumps({
                "source": "book.pdf", "type": "table", "text": "| Parrotfish | 1200 |", "page": 42, "page_end": 43,
                "page_label": 30, "section_path": ["Chapter 2", "Fish"], "images": []}), encoding="utf-8")
            code, out, _ = self.run_cli("parrotfish", "--data-dir", str(root))
        self.assertEqual(code, 0)
        self.assertIn("[1] book.pdf (pages 42-43, printed p.30, table)", out)
        self.assertIn("Chapter 2 > Fish", out)

    def test_header_shows_quality_flags_and_a_printed_page_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"source": "a.pdf", "type": "text", "text": "reef alpha", "page": 5, "page_label": 30,
                 "page_label_end": 31, "low_quality": True},
                {"source": "a.pdf", "type": "table", "text": "reef beta", "page": 6, "page_label": 32,
                 "page_label_end": 32, "table_structure_degraded": True}]
            (root / "b.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            code, out, _ = self.run_cli("reef", "--data-dir", str(root), "--top-k", "2")
        self.assertEqual(code, 0)
        self.assertIn("(page 5, printed pp.30-31, text, low-quality text)", out)
        self.assertIn("(page 6, printed p.32, table, table structure degraded)", out)

    def test_plain_text_documents_still_work_without_an_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "notes.txt").write_text("Pricing depends on demand.", encoding="utf-8")
            code, out, _ = self.run_cli("pricing demand", "--data-dir", directory)
        self.assertEqual(code, 0)
        self.assertIn("[1] notes.txt", out)
        self.assertIn("Pricing depends on demand.", out)

    def test_type_filter_and_error_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "notes.txt").write_text("Pricing depends on demand.", encoding="utf-8")
            self.assertEqual(self.run_cli("pricing", "--data-dir", directory, "--type", "table")[0], 1)
            self.assertEqual(self.run_cli("unicorn", "--data-dir", directory)[0], 1)
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(self.run_cli("anything", "--data-dir", empty)[0], 1)
        with self.assertRaises(SystemExit):
            self.run_cli("x", "--top-k", "0")


class PipedOutput(unittest.TestCase):
    def test_non_cp1252_characters_survive_a_redirected_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "notes.txt").write_text("Urea \u2192 retained in caf\u00e9 sharks.", encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
            env["RAG_MODEL_DIR"] = directory  # no model there: plain word matching, no 500 MB load
            done = subprocess.run([sys.executable, "-m", "rag_mvp.cli", "urea sharks", "--data-dir", directory],
                                  capture_output=True, env=env)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("\u2192".encode("utf-8"), done.stdout)
        self.assertIn("caf\u00e9".encode("utf-8"), done.stdout)


class IngestCli(unittest.TestCase):
    def test_page_range_uses_a_distinct_output_name_and_writes_a_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "book.pdf"
            fixtures.book_with_columns(pdf)
            out = root / "processed"
            with redirect_stdout(StringIO()):
                code = ingest_main([str(pdf), "--output-dir", str(out), "--pages", "2", "5", "--no-figures"])
            self.assertEqual(code, 0)
            self.assertTrue((out / "book.p2-5.jsonl").exists())
            self.assertTrue((out / "book.p2-5.manifest.json").exists())
            self.assertFalse((out / "book.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
