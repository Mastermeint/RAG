# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["marker-pdf>=2,<3"]
# ///
"""Convert a small PDF sample to Markdown with Marker in CPU text-layer mode."""

import argparse
import time
from pathlib import Path

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    config = ConfigParser({"mode": "fast", "disable_ocr": True, "paginate_output": True})
    converter = PdfConverter(
        config=config.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config.get_processors(),
        renderer=config.get_renderer(),
        llm_service=config.get_llm_service(),
    )
    rendered = converter(str(args.pdf))
    text, _, images = text_from_rendered(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Output: {args.output}; conversion: {time.perf_counter() - started:.1f}s; "
          f"image references: {len(images)}")


if __name__ == "__main__":
    main()