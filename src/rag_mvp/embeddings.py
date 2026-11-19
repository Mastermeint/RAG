"""Meaning-based (dense) retrieval with a local ONNX embedding model, a vector cache on disk, and hybrid search.

An embedding model turns a text into a vector (a list of numbers) so that texts with a similar meaning get similar
vectors, even when they share no words. `HybridIndex` merges the BM25 ranking with the embedding ranking using
Reciprocal Rank Fusion, so exact-word matches and meaning matches both count.

Run `uv run rag-embed --help` for the download/build/status commands."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from rag_mvp.console import use_utf8
from rag_mvp.retrieval import NEUTRAL, Index, Passage, RankingConfig, load_corpus

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
VECTORS_DIR = ROOT / "data" / "processed" / "embeddings"
MODES = ("bm25", "dense", "hybrid")
DEFAULT_MODE = "dense"  # dev verdict 2026-11-19: dense beat bm25 (+24/-3) and every hybrid weight; see README
RRF_K = 60  # standard Reciprocal Rank Fusion constant: a result at rank r contributes 1 / (RRF_K + r)
LEXICAL_WEIGHT = 1.0  # BM25's share in the fusion relative to dense (1.0 = standard RRF); dev sweep: lower is better,
# down to 0.15 where hybrid equals dense, so no weight made hybrid the default
FUSION_DEPTH = 100  # how many results of each ranking take part in the fusion
MAX_ON_THE_FLY = 50  # more missing vectors than this needs `rag-embed build` instead of a silent long wait


@dataclass(frozen=True)
class ModelSpec:
    """Where a model comes from, how to check it, and how to turn its token outputs into one vector."""

    name: str
    release: str  # base URL of the release assets
    archive: str  # zip asset that holds model.onnx
    archive_sha256: str
    files: dict[str, str] = field(default_factory=dict)  # file in the model folder -> sha256
    pooling: str = "cls"  # "cls": vector of the first token; "mean": average over tokens
    query_prefix: str = ""
    passage_prefix: str = ""
    max_tokens: int = 1024


MODELS = {
    "gte-base-en-v1.5": ModelSpec(
        name="gte-base-en-v1.5",
        release="https://github.com/magda-io/embedding_models/releases/download/gte-base-en-v1.5/",
        archive="model.zip",
        archive_sha256="55dfd0ae0b9140cf091a3ee26fe0542e00807c3827687ac2e66da27c12f01084",
        files={"model.onnx": "d908f84171c304a863061cde3eaf29f9f7ec9206068f9b3a5b97323fa7b2811e",
               "tokenizer.json": "cb374d6bc042c22455946f4e09a89d29882a199fdaf8fb25be00dc8b8857a448",
               "config.json": "238854682d429ce12c49ed3d1a31b6d1b9622d2afbcf50a17243da18926cbcab"},
        pooling="cls"),
}
DEFAULT_MODEL = "gte-base-en-v1.5"


class ModelMissing(RuntimeError):
    pass


class VectorsMissing(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_dir(name: str = DEFAULT_MODEL) -> Path:
    return Path(os.environ.get("RAG_MODEL_DIR") or MODELS_DIR / name)


class Embedder:
    """Runs the ONNX model on CPU. Vectors are L2-normalised, so a dot product is the cosine similarity."""

    def __init__(self, spec: ModelSpec, folder: Path | None = None, threads: int | None = None):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np, self.spec = np, spec
        folder = folder or model_dir(spec.name)
        missing = [f for f in ("model.onnx", "tokenizer.json") if not (folder / f).exists()]
        if missing:
            raise ModelMissing(f"model files {missing} not found in {folder}; run `uv run rag-embed download`")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(folder / "model.onnx"), options, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=spec.max_tokens)
        self.tokenizer.no_padding()
        # Identifies the exact weights for the vector cache without hashing 500 MB on every start.
        self.fingerprint = f"{spec.name}:{(folder / 'model.onnx').stat().st_size}:{spec.pooling}:{spec.max_tokens}"

    def _run(self, texts: list[str]):
        np = self.np
        encoded = self.tokenizer.encode_batch(texts)
        width = max(len(e.ids) for e in encoded)
        ids = np.zeros((len(encoded), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for row, e in enumerate(encoded):
            ids[row, :len(e.ids)] = e.ids
            mask[row, :len(e.ids)] = 1
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feeds)[0]
        if self.spec.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            weights = mask[..., None].astype(hidden.dtype)
            pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)

    def encode(self, texts: list[str], batch_size: int = 16, progress=None):
        """Embed texts in batches of similar length (less padding means less wasted work)."""
        np = self.np
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        result = np.zeros((len(texts), 0), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            vectors = self._run([texts[i] for i in chunk]).astype(np.float32)
            if not result.shape[1]:
                result = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
            result[chunk] = vectors
            if progress:
                progress(min(start + batch_size, len(order)), len(order), [texts[i] for i in chunk], vectors)
        return result

    def encode_query(self, question: str):
        return self.encode([self.spec.query_prefix + question])[0]

    def passage_text(self, passage: Passage) -> str:
        return self.spec.passage_prefix + passage.search_text.strip()


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class VectorStore:
    """Vectors saved on disk, addressed by a hash of the exact text that was embedded.

    Because the key is the text itself, a re-ingest only costs the chunks whose text changed, and a passage can never
    be paired with another passage's vector. One folder per model; a different model fingerprint starts empty."""

    def __init__(self, folder: Path, fingerprint: str):
        import numpy as np

        self.np, self.folder, self.fingerprint = np, Path(folder), fingerprint
        self.keys: list[str] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.notes: list[str] = []
        info_path = self.folder / "info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info.get("fingerprint") != fingerprint:
                self.notes.append(f"saved vectors were made with {info.get('fingerprint')!r}, not {fingerprint!r}; "
                                  f"ignoring them")
            else:
                self.keys = json.loads((self.folder / "keys.json").read_text(encoding="utf-8"))
                self.vectors = np.load(self.folder / "vectors.npy")
                if len(self.keys) != len(self.vectors):
                    self.notes.append("saved vectors and keys disagree in length; ignoring them")
                    self.keys, self.vectors = [], np.zeros((0, 0), dtype=np.float32)
        self.position = {k: i for i, k in enumerate(self.keys)}

    def __contains__(self, key: str) -> bool:
        return key in self.position

    def lookup(self, keys: list[str]):
        return self.vectors[[self.position[k] for k in keys]]

    def add(self, keys: list[str], vectors) -> None:
        np = self.np
        fresh = [(k, v) for k, v in zip(keys, vectors) if k not in self.position]
        if not fresh:
            return
        block = np.stack([v for _, v in fresh]).astype(np.float32)
        self.vectors = block if not len(self.keys) else np.concatenate([self.vectors, block])
        for k, _ in fresh:
            self.position[k] = len(self.keys)
            self.keys.append(k)

    def prune(self, keep: set[str]) -> int:
        """Drop vectors of texts that no longer exist in the corpus. Returns how many were dropped."""
        rows = [i for i, k in enumerate(self.keys) if k in keep]
        dropped = len(self.keys) - len(rows)
        if dropped:
            self.keys = [self.keys[i] for i in rows]
            self.vectors = self.vectors[rows]
            self.position = {k: i for i, k in enumerate(self.keys)}
        return dropped

    def save(self) -> None:
        """Write to temporary files first, then swap them in, so an interrupted save never leaves a broken store."""
        self.folder.mkdir(parents=True, exist_ok=True)
        dim = int(self.vectors.shape[1]) if self.vectors.size else 0
        staged = {}
        for name, write in (("vectors.npy", lambda h: self.np.save(h, self.vectors)),
                            ("keys.json", lambda h: h.write(json.dumps(self.keys).encode("utf-8"))),
                            ("info.json", lambda h: h.write(json.dumps(
                                {"fingerprint": self.fingerprint, "count": len(self.keys), "dim": dim},
                                indent=1).encode("utf-8")))):
            handle = tempfile.NamedTemporaryFile(dir=self.folder, delete=False, suffix=".tmp")
            with handle:
                write(handle)
            staged[name] = handle.name
        for name in ("vectors.npy", "keys.json", "info.json"):  # info last: it marks the store as complete
            os.replace(staged[name], self.folder / name)


def open_store(embedder: Embedder, root: Path = ROOT) -> VectorStore:
    return VectorStore(root / "data" / "processed" / "embeddings" / embedder.spec.name, embedder.fingerprint)


def passage_vectors(passages: list[Passage], embedder: Embedder, store: VectorStore,
                    max_on_the_fly: int = MAX_ON_THE_FLY):
    """One vector per passage from the store. A few missing ones (e.g. a new .txt note) are computed in memory and
    not saved, so ad-hoc searches never change the store; many missing ones need `rag-embed build`."""
    np = embedder.np
    if not passages:
        return np.zeros((0, 0), dtype=np.float32)
    texts = [embedder.passage_text(p) for p in passages]
    keys = [text_key(t) for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in store]
    if len(missing) > max_on_the_fly:
        raise VectorsMissing(f"{len(missing)} of {len(passages)} passages have no saved vector; "
                             f"run `uv run rag-embed build` (about {len(missing) * 0.45 / 60:.0f} min on a laptop CPU)")
    fresh = dict(zip(missing, embedder.encode([texts[i] for i in missing]))) if missing else {}
    present = [i for i in range(len(keys)) if i not in fresh]
    rows = store.lookup([keys[i] for i in present]) if present else None
    dim = rows.shape[1] if rows is not None else len(next(iter(fresh.values())))
    result = np.zeros((len(keys), dim), dtype=np.float32)
    if present:
        result[present] = rows
    for i, vector in fresh.items():
        result[i] = vector
    return result


class DenseIndex:
    """Ranks passages by cosine similarity between the question vector and each passage vector."""

    def __init__(self, passages: list[Passage], vectors, embedder: Embedder, ranking: RankingConfig | None = None):
        self.passages, self.vectors, self.embedder = passages, vectors, embedder
        self.ranking = ranking or RankingConfig()

    def search_scored(self, question: str, top_k: int = 3, types: set[str] | None = None,
                      ranking: RankingConfig | None = None) -> list[tuple[Passage, float]]:
        np = self.embedder.np
        ranking = ranking or self.ranking
        scores = self.vectors @ self.embedder.encode_query(question)
        weights = np.array([ranking.multiplier(p) for p in self.passages], dtype=np.float32)
        scores = np.where(weights > 0, scores * weights, -np.inf)
        hits = []
        for position in np.argsort(-scores, kind="stable"):
            passage = self.passages[position]
            if not np.isfinite(scores[position]):
                break
            if types is None or passage.type in types:
                hits.append((passage, float(scores[position])))
                if len(hits) >= top_k:
                    break
        return hits

    def search(self, question: str, top_k: int = 3, types: set[str] | None = None,
               ranking: RankingConfig | None = None) -> list[Passage]:
        return [p for p, _ in self.search_scored(question, top_k, types, ranking)]


class HybridIndex:
    """Reciprocal Rank Fusion of BM25 and dense rankings.

    Each ranking gives a passage weight / (k + rank) points; the points are added. Rank-based fusion avoids having to
    put BM25 scores (0 to ~40) and cosine similarities (about 0.3 to 0.9) on the same scale. `lexical_weight` below 1
    lets BM25 break ties and rescue exact-term questions without dragging its noise into meaning-only questions."""

    def __init__(self, lexical: Index, dense: DenseIndex, k: int = RRF_K, depth: int = FUSION_DEPTH,
                 ranking: RankingConfig | None = None, lexical_weight: float = LEXICAL_WEIGHT):
        self.lexical, self.dense, self.k, self.depth = lexical, dense, k, depth
        self.lexical_weight = lexical_weight
        self.passages = lexical.passages
        self.ranking = ranking or lexical.ranking

    def search_scored(self, question: str, top_k: int = 3, types: set[str] | None = None,
                      ranking: RankingConfig | None = None) -> list[tuple[Passage, float]]:
        ranking = ranking or self.ranking
        depth = max(self.depth, top_k)
        points: dict[int, float] = {}
        by_id: dict[int, Passage] = {}
        for weight, hits in ((self.lexical_weight, self.lexical.search_scored(question, depth, types, NEUTRAL)),
                             (1.0, self.dense.search_scored(question, depth, types, NEUTRAL))):
            for rank, (passage, _) in enumerate(hits, 1):
                by_id[id(passage)] = passage
                points[id(passage)] = points.get(id(passage), 0.0) + weight / (self.k + rank)
        scored = [(by_id[i], s * ranking.multiplier(by_id[i])) for i, s in points.items()]
        scored = [(p, s) for p, s in scored if s > 0]
        scored.sort(key=lambda item: -item[1])
        return scored[:top_k]

    def search(self, question: str, top_k: int = 3, types: set[str] | None = None,
               ranking: RankingConfig | None = None) -> list[Passage]:
        return [p for p, _ in self.search_scored(question, top_k, types, ranking)]


def build_index(passages: list[Passage], mode: str = DEFAULT_MODE, ranking: RankingConfig | None = None,
                model: str = DEFAULT_MODEL, root: Path = ROOT, max_on_the_fly: int = MAX_ON_THE_FLY,
                lexical_weight: float = LEXICAL_WEIGHT):
    """The searcher for a retrieval mode. Raises ModelMissing/VectorsMissing when embeddings are unavailable."""
    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}; use one of {', '.join(MODES)}")
    lexical = Index(passages, ranking=ranking)
    if mode == "bm25":
        return lexical
    embedder = Embedder(MODELS[model])
    dense = DenseIndex(passages, passage_vectors(passages, embedder, open_store(embedder, root), max_on_the_fly),
                       embedder, ranking)
    return dense if mode == "dense" else HybridIndex(lexical, dense, ranking=ranking, lexical_weight=lexical_weight)


def describe(index) -> dict:
    """What produced a ranking, for saved evaluation runs."""
    if isinstance(index, HybridIndex):
        return {"mode": "hybrid", "model": index.dense.embedder.fingerprint, "rrf_k": index.k, "depth": index.depth,
                "lexical_weight": index.lexical_weight}
    if isinstance(index, DenseIndex):
        return {"mode": "dense", "model": index.embedder.fingerprint}
    return {"mode": "bm25"}


# ---- command line ------------------------------------------------------------------------------------------------

def download(spec: ModelSpec, folder: Path) -> list[str]:
    """Fetch the model release, check every hash, and only then move the files into place."""
    log = []
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        tmp = Path(tmp)
        names = [spec.archive] + [f for f in spec.files if f != "model.onnx"]
        for name in names:
            log.append(f"downloading {spec.release}{name}")
            print(log[-1], flush=True)
            urllib.request.urlretrieve(spec.release + name, tmp / name)
        if sha256_file(tmp / spec.archive) != spec.archive_sha256:
            raise ValueError(f"{spec.archive}: SHA-256 does not match the pinned value; refusing to use it")
        with zipfile.ZipFile(tmp / spec.archive) as archive:
            archive.extract("model.onnx", tmp)
        for name, expected in spec.files.items():
            if sha256_file(tmp / name) != expected:
                raise ValueError(f"{name}: SHA-256 does not match the pinned value; refusing to use it")
        for name in spec.files:
            os.replace(tmp / name, folder / name)
    log.append(f"model ready in {folder}")
    return log


def main(argv: list[str] | None = None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="Manage the local embedding model and the saved passage vectors.")
    parser.add_argument("command", choices=("download", "build", "status"),
                        help="download: fetch and verify the model; build: embed every passage that has no saved "
                             "vector; status: report what is missing")
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, help="CPU threads for the model (default: all cores)")
    args = parser.parse_args(argv)
    spec = MODELS[args.model]

    if args.command == "download":
        try:
            download(spec, model_dir(spec.name))
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            print(f"download failed: {error}", file=sys.stderr)
            return 1
        return 0

    passages = load_corpus(base=ROOT)
    if not passages:
        parser.error("no corpus found; run rag-ingest first")
    try:
        embedder = Embedder(spec, threads=args.threads)
    except ModelMissing as error:
        parser.error(str(error))
    store = open_store(embedder)
    for note in store.notes:
        print("note:", note)
    texts = [embedder.passage_text(p) for p in passages]
    keys = [text_key(t) for t in texts]
    missing = sorted({k: i for i, k in enumerate(keys) if k not in store}.values())
    stale = len(set(store.keys) - set(keys))
    print(f"{len(passages)} passages, {len(passages) - len(missing)} with a saved vector, {len(missing)} missing, "
          f"{stale} saved vectors no longer used ({store.folder})")
    if args.command == "status":
        return 0 if not missing else 1

    started, last_save = time.time(), [time.time()]

    def progress(done, total, batch_texts, vectors):
        store.add([text_key(t) for t in batch_texts], vectors)
        elapsed = time.time() - started
        if time.time() - last_save[0] > 60 or done == total:
            store.save()  # an interrupted build keeps what it already computed
            last_save[0] = time.time()
            eta = elapsed / done * (total - done)
            print(f"  {done}/{total} embedded, {elapsed / 60:.1f} min elapsed, about {eta / 60:.1f} min left",
                  flush=True)

    if missing:
        embedder.encode([texts[i] for i in missing], args.batch_size, progress)
    dropped = store.prune(set(keys))
    store.save()
    print(f"done: {len(missing)} embedded, {dropped} unused vectors removed, {len(store.keys)} saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
