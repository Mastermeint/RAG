"""Per-book ingestion settings loaded from JSON, guarded by the PDF's SHA-256.

Anything specific to one PDF (pages to drop, thresholds) belongs in this file, never in the code."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

Range = tuple[int, int, str]


@dataclass(frozen=True)
class DetectConfig:
    size_tolerance: float = 0.03
    min_run_pages: int = 10
    max_reference_pages: int = 40
    max_excluded_share: float = 0.15

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"detect.{f.name} must be a positive number, got {value!r}")
        for name in ("min_run_pages", "max_reference_pages"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"detect.{name} must be a whole number")
        if self.max_excluded_share > 1:
            raise ValueError("detect.max_excluded_share must be at most 1")


@dataclass(frozen=True)
class BookConfig:
    name: str
    sha256: str | None = None
    exclude: tuple[Range, ...] = ()
    accept: tuple[Range, ...] = ()
    detect: DetectConfig = DetectConfig()


def _check_keys(where: str, data: dict, allowed: set[str]) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be an object")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}")


def _detect(where: str, data: dict, base: DetectConfig) -> DetectConfig:
    _check_keys(where, data, {f.name for f in fields(DetectConfig)})
    values = {f.name: getattr(base, f.name) for f in fields(DetectConfig)} | data
    try:
        return DetectConfig(**values)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def _ranges(where: str, items) -> tuple[Range, ...]:
    if not isinstance(items, list):
        raise ValueError(f"{where} must be a list")
    result = []
    for item in items:
        _check_keys(where, item, {"pages", "reason"})
        pages, reason = item.get("pages"), item.get("reason")
        if (not isinstance(pages, list) or len(pages) != 2 or not all(isinstance(p, int) and not isinstance(p, bool)
                                                                       for p in pages) or not 1 <= pages[0] <= pages[1]):
            raise ValueError(f"{where}: pages must be [first, last] with 1 <= first <= last, got {pages!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{where}: pages {pages} need a non-empty reason")
        result.append((pages[0], pages[1], reason.strip()))
    return tuple(result)


def load_config(path: Path) -> dict[str, BookConfig]:
    return load_settings(path)[1]


def load_settings(path: Path) -> tuple[DetectConfig, dict[str, BookConfig]]:
    """Default detector settings and the per-book entries keyed by PDF file name."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error})") from None
    _check_keys(str(path), data, {"defaults", "books"})
    defaults = data.get("defaults", {})
    _check_keys("defaults", defaults, {"detect"})
    base = _detect("defaults.detect", defaults.get("detect", {}), DetectConfig())
    books = {}
    for name, entry in data.get("books", {}).items():
        where = f"books[{name!r}]"
        _check_keys(where, entry, {"sha256", "exclude_pages", "accept_pages", "detect"})
        sha = entry.get("sha256")
        if sha is not None and not (isinstance(sha, str) and len(sha) == 64):
            raise ValueError(f"{where}: sha256 must be a 64-character hex string")
        books[name] = BookConfig(
            name, sha.lower() if sha else None, _ranges(f"{where}.exclude_pages", entry.get("exclude_pages", [])),
            _ranges(f"{where}.accept_pages", entry.get("accept_pages", [])),
            _detect(f"{where}.detect", entry.get("detect", {}), base))
    return base, books


def config_for(configs: dict[str, BookConfig], pdf_name: str, sha256: str) -> tuple[BookConfig | None, list[str]]:
    """The entry for this PDF, or None. An entry whose hash differs is ignored with a warning."""
    entry = configs.get(pdf_name)
    if entry is None:
        return None, []
    if entry.sha256 and entry.sha256 != sha256.lower():
        return None, [f"config entry for {pdf_name!r} ignored: its sha256 does not match this PDF "
                      f"(pages may differ in another edition); update ingest_config.json"]
    return entry, []
