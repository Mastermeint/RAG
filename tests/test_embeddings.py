"""Dense and hybrid retrieval, the on-disk vector store, and the CLI fallback when no model is installed."""

import os
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from rag_mvp.cli import main as cli_main
from rag_mvp.embeddings import (MODELS, DenseIndex, Embedder, HybridIndex, ModelSpec, VectorsMissing, VectorStore,
                                model_dir, passage_vectors, text_key)
from rag_mvp.retrieval import Index, Passage, RankingConfig

# Each concept is one vector dimension; synonyms share it, which is what a real embedding model learns.
CONCEPTS = [("shell", "chalky", "housing", "carbonate", "skeleton"), ("acid", "acidification", "sour", "ph"),
            ("aquarium", "tank", "pet"), ("death", "killing", "mortality", "die"), ("cyanide", "poison"),
            ("whale", "whales"), ("song", "sing", "call")]


class FakeEmbedder:
    def __init__(self):
        self.np, self.spec, self.fingerprint, self.calls = np, ModelSpec("fake", "", "", ""), "fake:1", 0

    def encode(self, texts, batch_size=16, progress=None):
        self.calls += len(texts)
        out = np.zeros((len(texts), len(CONCEPTS)), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                for dim, group in enumerate(CONCEPTS):
                    if word in group:
                        out[row, dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)

    def encode_query(self, question):
        return self.encode([question])[0]

    def passage_text(self, passage):
        return passage.search_text.strip()


PASSAGES = [
    Passage("a.pdf", 1, "Ocean acidification lowers carbonate so corals struggle to build a skeleton.", page=10),
    Passage("a.pdf", 2, "Cyanide fishing for the aquarium trade causes high mortality.", page=20),
    Passage("b.pdf", 3, "Humpback whales sing long songs during the breeding season.", page=30, type="figure"),
    Passage("b.pdf", 4, "The Bermuda fireworm glows a few nights after the full moon.", page=40),
]


class DenseAndHybrid(unittest.TestCase):
    def setUp(self):
        self.embedder = FakeEmbedder()
        self.vectors = self.embedder.encode([self.embedder.passage_text(p) for p in PASSAGES])
        self.dense = DenseIndex(PASSAGES, self.vectors, self.embedder)
        self.lexical = Index(PASSAGES)

    def test_dense_finds_a_paraphrase_that_shares_no_words_with_the_passage(self):
        question = "why do sour seas make chalky housings hard to grow"
        self.assertEqual(self.lexical.search(question, 3), [])  # control: no shared word at all
        self.assertIs(self.dense.search(question, 1)[0], PASSAGES[0])

    def test_hybrid_keeps_exact_word_matches_and_adds_meaning_matches(self):
        hybrid = HybridIndex(self.lexical, self.dense)
        self.assertIs(hybrid.search("Bermuda fireworm", 1)[0], PASSAGES[3])  # only BM25 can find this one
        self.assertIs(hybrid.search("pet tank poison deaths", 1)[0], PASSAGES[1])  # only dense can

    def test_reciprocal_rank_fusion_rewards_agreement(self):
        hybrid = HybridIndex(self.lexical, self.dense, k=60)
        scored = dict((p.index, s) for p, s in hybrid.search_scored("whales sing", 4))
        self.assertAlmostEqual(scored[3], 2 / 61)  # first in both rankings
        self.assertTrue(all(s < scored[3] for i, s in scored.items() if i != 3))

    def test_type_filter_and_zero_weight_apply_to_dense_and_hybrid(self):
        hybrid = HybridIndex(self.lexical, self.dense)
        for index in (self.dense, hybrid):
            self.assertEqual([p.type for p in index.search("whales sing", 4, types={"figure"})], ["figure"])
            hidden = index.search("whales sing", 4, ranking=RankingConfig(type_weights={"figure": 0}))
            self.assertNotIn(PASSAGES[2], hidden)


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / "store"
        self.embedder = FakeEmbedder()

    def test_vectors_survive_a_restart_and_a_new_model_starts_empty(self):
        store = VectorStore(self.folder, "fake:1")
        texts = [self.embedder.passage_text(p) for p in PASSAGES]
        store.add([text_key(t) for t in texts], self.embedder.encode(texts))
        store.save()
        again = VectorStore(self.folder, "fake:1")
        self.assertTrue(np.allclose(again.lookup([text_key(texts[1])]), self.embedder.encode([texts[1]])))
        other = VectorStore(self.folder, "other-model:2")
        self.assertEqual(other.keys, [])
        self.assertTrue(other.notes)

    def test_only_changed_passages_are_embedded_and_unused_vectors_can_be_pruned(self):
        store = VectorStore(self.folder, "fake:1")
        texts = [self.embedder.passage_text(p) for p in PASSAGES]
        store.add([text_key(t) for t in texts[:3]], self.embedder.encode(texts[:3]))
        store.save()
        self.embedder.calls = 0
        vectors = passage_vectors(PASSAGES, self.embedder, VectorStore(self.folder, "fake:1"))
        self.assertEqual(self.embedder.calls, 1)  # just the one passage without a saved vector
        self.assertEqual(vectors.shape, (4, len(CONCEPTS)))
        self.assertEqual(len(VectorStore(self.folder, "fake:1").keys), 3)  # ad-hoc searches never write
        self.assertEqual(store.prune({text_key(texts[0])}), 2)

    def test_many_missing_vectors_ask_for_a_build_instead_of_a_long_silent_wait(self):
        with self.assertRaises(VectorsMissing):
            passage_vectors(PASSAGES, self.embedder, VectorStore(self.folder, "fake:1"), max_on_the_fly=2)


class Download(unittest.TestCase):
    def test_files_with_a_wrong_hash_are_refused_and_nothing_is_installed(self):
        import hashlib
        import zipfile
        from unittest import mock

        from rag_mvp.embeddings import download

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "served.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("model.onnx", b"weights")
            served = {"model.zip": archive.read_bytes(), "tokenizer.json": b"{}"}
            good = ModelSpec("m", "https://example.invalid/", "model.zip", hashlib.sha256(served["model.zip"]).hexdigest(),
                             {"model.onnx": hashlib.sha256(b"weights").hexdigest(),
                              "tokenizer.json": hashlib.sha256(b"{}").hexdigest()})
            bad = ModelSpec(**{**good.__dict__, "files": {**good.files, "model.onnx": "0" * 64}})

            def fetch(url, target):
                Path(target).write_bytes(served[url.rsplit("/", 1)[1]])

            with mock.patch("rag_mvp.embeddings.urllib.request.urlretrieve", fetch):
                with self.assertRaises(ValueError):
                    download(bad, root / "bad")
                self.assertFalse((root / "bad" / "model.onnx").exists())
                download(good, root / "good")
            self.assertEqual((root / "good" / "model.onnx").read_bytes(), b"weights")


class CliFallback(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_without_a_model_the_default_falls_back_to_bm25_and_an_explicit_mode_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "notes.txt").write_text("Coral reefs need warm water.", encoding="utf-8")
            empty = Path(directory) / "no-model"
            empty.mkdir()
            previous = os.environ.get("RAG_MODEL_DIR")
            os.environ["RAG_MODEL_DIR"] = str(empty)
            try:
                code, out, err = self.run_cli("coral reefs", "--data-dir", directory)
                self.assertEqual(code, 0)
                self.assertIn("using bm25", err)
                self.assertIn("[1] notes.txt", out)
                with self.assertRaises(SystemExit):
                    self.run_cli("coral reefs", "--data-dir", directory, "--mode", "hybrid")
            finally:
                if previous is None:
                    os.environ.pop("RAG_MODEL_DIR")
                else:
                    os.environ["RAG_MODEL_DIR"] = previous


@unittest.skipUnless((model_dir() / "model.onnx").exists(), "embedding model not downloaded (uv run rag-embed download)")
class RealModel(unittest.TestCase):
    def test_the_installed_model_ranks_a_paraphrase_above_unrelated_text(self):
        embedder = Embedder(MODELS["gte-base-en-v1.5"])
        question = embedder.encode_query("Why is rising carbon dioxide making it harder for animals with chalky "
                                         "housings to grow?")
        passages = embedder.encode([
            "Ocean acidification lowers the carbonate ion concentration, so corals and molluscs struggle to build "
            "calcium carbonate shells.",
            "Two male tilapia in a tank will fight; one often ends up killing the other.",
            "Sharks maintain buoyancy using lipids stored in the liver."])
        similarity = passages @ question
        self.assertEqual(int(np.argmax(similarity)), 0)
        self.assertAlmostEqual(float(np.linalg.norm(passages[0])), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
