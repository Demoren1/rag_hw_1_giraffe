#!/usr/bin/env python3
"""Extract a deterministic, page-based pilot corpus from the supplied PDF.

Usage:
    python tools/prepare_pilot_corpus.py --pdf PATH --output local_data/pilot

This script uses the PDF's embedded text layer through pypdf. It does not use
OCR, an LLM, external services, or any generated content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pypdf import PdfReader


EXTRACTION_MODE = "plain"
OUTPUT_FILENAMES = (
    "pilot_corpus.jsonl",
    "pages.jsonl",
    "extraction_report.json",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a page-level preliminary corpus from a PDF's embedded text."
        )
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the source PDF.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for the local pilot outputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_cyrillic_cp1251_letter(char: str) -> bool:
    """Check whether a raw byte-valued character maps to a CP1251 letter."""
    value = ord(char)
    return 0xC0 <= value <= 0xFF or value in (0xA8, 0xB8)


def is_unambiguous_math_operator(text: str, index: int) -> bool:
    """Preserve operators between non-Cyrillic operands."""
    char = text[index]
    if ord(char) not in (0xD7, 0xF7):
        return False
    if index == 0 or index + 1 >= len(text):
        return False

    previous = text[index - 1]
    following = text[index + 1]
    if previous.isspace() or following.isspace():
        return False
    return not (
        is_cyrillic_cp1251_letter(previous)
        or is_cyrillic_cp1251_letter(following)
    )


def recover_cp1251_text(raw_text: str) -> str:
    """Reverse the PDF's byte-valued Cyrillic extraction for readability."""
    output: list[str] = []
    for index, char in enumerate(raw_text):
        value = ord(char)
        if (
            0x80 <= value <= 0xFF
            and not is_unambiguous_math_operator(raw_text, index)
        ):
            output.append(bytes((value,)).decode("cp1251", errors="replace"))
        else:
            output.append(char)
    return "".join(output)


def normalize_page_text(decoded_text: str) -> str:
    """Apply layout-only cleanup while retaining formula line breaks."""
    text = decoded_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=[А-Яа-яЁё])-\n(?=[а-яё])", "", text)

    lines: list[str] = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSON records with stable keys and UTF-8 text."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")


def pypdf_version() -> str:
    """Return the installed pypdf version for provenance."""
    try:
        return version("pypdf")
    except PackageNotFoundError:
        return "unknown"


def main() -> int:
    """Extract pages, build pilot records, and write provenance reports."""
    args = parse_args()
    pdf_path = args.pdf.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not pdf_path.is_file():
        raise SystemExit(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Expected a .pdf file: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_digest = sha256_file(pdf_path)
    reader = PdfReader(pdf_path)
    metadata = reader.metadata or {}
    source_title = str(metadata.get("/Title", "")) or None
    source_author = str(metadata.get("/Author", "")) or None
    installed_version = pypdf_version()

    pages: list[dict[str, Any]] = []
    corpus: list[dict[str, Any]] = []
    empty_pages: list[int] = []
    raw_character_count = 0
    normalized_character_count = 0
    control_character_counts: Counter[str] = Counter()

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text(extraction_mode=EXTRACTION_MODE) or ""
        decoded_text = recover_cp1251_text(raw_text)
        normalized_text = normalize_page_text(decoded_text)

        raw_character_count += len(raw_text)
        normalized_character_count += len(normalized_text)
        control_character_counts.update(
            f"U+{ord(char):04X}"
            for char in raw_text
            if ord(char) < 32 and char not in "\n\r\t"
        )

        pages.append(
            {
                "page": page_number,
                "source_pdf_name": pdf_path.name,
                "source_pdf_sha256": pdf_digest,
                "extractor": f"pypdf=={installed_version}",
                "extraction_mode": EXTRACTION_MODE,
                "raw_text": raw_text,
                "decoded_text": decoded_text,
                "normalized_text": normalized_text,
            }
        )

        if not normalized_text:
            empty_pages.append(page_number)
            continue

        corpus.append(
            {
                "block_id": f"pilot_p{page_number:04d}_b0001",
                "page": page_number,
                "section": "",
                "text": normalized_text,
            }
        )

    corpus_path = output_dir / "pilot_corpus.jsonl"
    pages_path = output_dir / "pages.jsonl"
    report_path = output_dir / "extraction_report.json"
    jsonl_write(corpus_path, corpus)
    jsonl_write(pages_path, pages)

    report = {
        "status": "preliminary_pilot",
        "source": {
            "pdf_path": str(pdf_path),
            "pdf_name": pdf_path.name,
            "sha256": pdf_digest,
            "size_bytes": pdf_path.stat().st_size,
            "title": source_title,
            "author": source_author,
            "pdf_page_count": len(reader.pages),
            "page_numbering": "1-based physical PDF page index",
        },
        "extraction": {
            "library": "pypdf",
            "library_version": installed_version,
            "mode": EXTRACTION_MODE,
            "uses_embedded_text_layer": True,
            "uses_ocr": False,
            "uses_llm": False,
            "external_calls": False,
        },
        "corpus": {
            "records": len(corpus),
            "schema": ["block_id", "page", "section", "text"],
            "block_id_scheme": "pilot_p{page:04d}_b{block:04d}",
            "block_granularity": "one non-empty block per physical PDF page",
            "official_block_ids_available": False,
            "pilot_ids_are_local_only": True,
            "section_value": "empty string; no section labels inferred",
        },
        "text_handling": {
            "raw_text_is_preserved": True,
            "raw_text_location": "pages.jsonl",
            "encoding_recovery": (
                "Characters in U+0080..U+00FF are decoded as CP1251 bytes; "
                "multiplication/division glyphs are retained when directly "
                "between non-CP1251-letter operands."
            ),
            "normalization": [
                "line endings are normalized to LF",
                "line-end hyphenation across Cyrillic words is joined",
                "repeated horizontal whitespace is collapsed",
                "form feeds are treated as horizontal whitespace",
                (
                    "formula line breaks and non-whitespace control "
                    "characters are kept"
                ),
                "headers, footers, formulas, and page content are not removed",
            ],
            "no_manual_formula_repair": True,
        },
        "inventory": {
            "pages_with_extracted_text": len(corpus),
            "empty_or_whitespace_only_pages": empty_pages,
            "raw_character_count": raw_character_count,
            "normalized_character_count": normalized_character_count,
            "raw_control_character_counts": dict(
                sorted(control_character_counts.items())
            ),
        },
        "formula_assessment": {
            "quality": (
                "partial; exact two-dimensional formula recovery is unsafe"
            ),
            "visually_checked_physical_pdf_pages": [11, 12, 34],
            "observations": [
                (
                    "Page 11: Cauchy-Riemann partial derivatives survive as "
                    "stacked text lines, but numerator/denominator geometry "
                    "is not encoded."
                ),
                (
                    "Page 12: partial-derivative expressions are split across "
                    "lines; overbars and other spatial relationships may be "
                    "lost."
                ),
                (
                    "Page 34: integral signs are extracted as the letter Z "
                    "and contour labels detach from their integrals."
                ),
                (
                    "Control-code characters occur in the extracted text, "
                    "including near formulas; raw extraction is retained for "
                    "audit."
                ),
            ],
            "recommendation": (
                "Treat this as a preliminary text pilot. Verify formula "
                "citations against the source PDF or use the official corpus "
                "when available."
            ),
        },
        "outputs": {
            name: str(output_dir / name) for name in OUTPUT_FILENAMES
        },
    }
    with report_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")

    print(
        json.dumps(
            {
                "pages": len(reader.pages),
                "corpus_records": len(corpus),
                "empty_pages": empty_pages,
                "source_sha256": pdf_digest,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
