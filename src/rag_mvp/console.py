"""Console setup shared by every command-line entry point."""

import sys


def use_utf8() -> None:
    """Stop piped output from crashing on characters outside the Windows code page."""
    for stream in (sys.stdout, sys.stderr):
        # StringIO (used by tests) has no reconfigure; skip streams that already encode UTF-8
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")
