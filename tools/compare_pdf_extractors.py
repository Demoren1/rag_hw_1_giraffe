#!/usr/bin/env python3
"""Export several local PDF text-extraction views for manual comparison.

No OCR, language model, network service, or formula repair is used. Extractor
outputs are kept verbatim so that visual review can distinguish character
mapping problems from layout and reading-order problems.

Usage:
    python tools/compare_pdf_extractors.py --pdf PATH \
        --output local_data/extraction_comparison
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

CORE_REVIEW_PAGES = (
    11,
    12,
    20,
    26,
    28,
    30,
    31,
    33,
    34,
    35,
    36,
    37,
    38,
)

LATE_REVIEW_PAGES = (
    47,
    49,
    54,
    55,
    57,
    61,
    73,
    100,
    115,
    122,
    134,
    139,
    145,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export raw text views from installed PDF extractors for review."
        )
    )
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--render-pages",
        nargs="*",
        type=int,
        default=(*CORE_REVIEW_PAGES, *LATE_REVIEW_PAGES),
        help=(
            "Physical pages to render with PyMuPDF when installed. "
            "The default selects formula-heavy review pages."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for the source PDF."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(package: str) -> str | None:
    """Return an installed package version, if present."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write stable UTF-8 JSON lines."""
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


def pypdf_outputs(pdf_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Extract plain and layout views with pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    output: dict[str, list[dict[str, Any]]] = {
        "pypdf_plain": [],
        "pypdf_layout": [],
    }
    for page_number, page in enumerate(reader.pages, start=1):
        variants = (("pypdf_plain", "plain"), ("pypdf_layout", "layout"))
        for method, mode in variants:
            text = page.extract_text(extraction_mode=mode) or ""
            output[method].append(
                {"page": page_number, "method": method, "text": text}
            )
    return output


def pdfplumber_outputs(pdf_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Extract default reading-order and layout-grid views with pdfplumber."""
    import pdfplumber

    output: dict[str, list[dict[str, Any]]] = {
        "pdfplumber_text": [],
        "pdfplumber_layout": [],
    }
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            variants = (
                ("pdfplumber_text", False),
                ("pdfplumber_layout", True),
            )
            for method, layout in variants:
                text = page.extract_text(layout=layout) or ""
                output[method].append(
                    {"page": page_number, "method": method, "text": text}
                )
    return output


def pymupdf_outputs(pdf_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Extract text and blocks with MuPDF, in native and sorted order."""
    import pymupdf

    output: dict[str, list[dict[str, Any]]] = {
        "pymupdf_text": [],
        "pymupdf_text_sort": [],
        "pymupdf_blocks": [],
        "pymupdf_blocks_sort": [],
    }
    pdf = pymupdf.open(pdf_path)
    for page_number, page in enumerate(pdf, start=1):
        for method, sort in (
            ("pymupdf_text", False),
            ("pymupdf_text_sort", True),
        ):
            text = page.get_text("text", sort=sort)
            output[method].append(
                {"page": page_number, "method": method, "text": text}
            )

        for method, sort in (
            ("pymupdf_blocks", False),
            ("pymupdf_blocks_sort", True),
        ):
            blocks = page.get_text("blocks", sort=sort)
            structured_blocks = [
                {
                    "bbox": [round(float(value), 3) for value in block[:4]],
                    "text": str(block[4]),
                    "block_type": int(block[6]) if len(block) > 6 else None,
                }
                for block in blocks
            ]
            output[method].append(
                {
                    "page": page_number,
                    "method": method,
                    "text": "\n".join(
                        block["text"].rstrip() for block in structured_blocks
                    ),
                    "blocks": structured_blocks,
                }
            )

    return output


def poppler_outputs(pdf_path: Path) -> tuple[
    dict[str, list[dict[str, Any]]], str | None
]:
    """Extract raw and page-layout text with pdftotext if available."""
    executable = shutil.which("pdftotext")
    if executable is None:
        return {}, None

    version_result = subprocess.run(
        [executable, "-v"], capture_output=True, text=True, check=False
    )
    version_output = version_result.stderr or version_result.stdout
    version_text = version_output.splitlines()
    version = version_text[0].strip() if version_text else "available"
    output: dict[str, list[dict[str, Any]]] = {
        "poppler_raw": [],
        "poppler_layout": [],
    }
    variants = (("poppler_raw", "-raw"), ("poppler_layout", "-layout"))
    for method, mode in variants:
        result = subprocess.run(
            [executable, "-enc", "UTF-8", mode, str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pdftotext {mode} failed: {result.stderr.strip()}"
            )
        page_texts = result.stdout.split("\f")
        if page_texts and not page_texts[-1].strip():
            page_texts.pop()
        for page_number, text in enumerate(page_texts, start=1):
            output[method].append(
                {"page": page_number, "method": method, "text": text}
            )
    return output, version


def render_previews(
    pdf_path: Path, output_dir: Path, pages: list[int]
) -> str | None:
    """Render selected pages with PyMuPDF when the module is installed."""
    try:
        import pymupdf
    except ImportError:
        return None

    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    for page_number in pages:
        if not 1 <= page_number <= len(document):
            continue
        page = document[page_number - 1]
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(1.5, 1.5), alpha=False
        )
        pixmap.save(preview_dir / f"page-{page_number:03d}.png")
    return str(preview_dir)


def main() -> int:
    """Export all available extractor views and run provenance metadata."""
    args = parse_args()
    pdf_path = args.pdf.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Expected a .pdf file: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    methods_dir = output_dir / "methods"
    methods_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, list[dict[str, Any]]] = {}
    methods.update(pypdf_outputs(pdf_path))
    methods.update(pdfplumber_outputs(pdf_path))

    fitz_version: str | None = None
    if package_version("PyMuPDF") is not None:
        methods.update(pymupdf_outputs(pdf_path))
        fitz_version = package_version("PyMuPDF")

    poppler_views, poppler_version = poppler_outputs(pdf_path)
    methods.update(poppler_views)

    for method, rows in methods.items():
        jsonl_write(methods_dir / f"{method}.jsonl", rows)

    preview_path = render_previews(pdf_path, output_dir, args.render_pages)
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    report = {
        "source": {
            "path": str(pdf_path),
            "name": pdf_path.name,
            "sha256": sha256_file(pdf_path),
            "size_bytes": pdf_path.stat().st_size,
            "pdf_page_count": len(reader.pages),
            "page_indexing": "1-based physical PDF pages",
        },
        "scope": {
            "primary_review_pages": list(CORE_REVIEW_PAGES),
            "later_formula_review_pages": list(LATE_REVIEW_PAGES),
            "extractor_outputs_cover_all_pages": True,
        },
        "extractors": {
            "pypdf": package_version("pypdf"),
            "pdfplumber": package_version("pdfplumber"),
            "pymupdf": fitz_version,
            "poppler_pdftotext": poppler_version,
        },
        "methods": {
            method: {
                "pages_written": len(rows),
                "output": str(methods_dir / f"{method}.jsonl"),
            }
            for method, rows in sorted(methods.items())
        },
        "rendered_previews": preview_path,
        "constraints": {
            "ocr": False,
            "llm": False,
            "external_services": False,
            "manual_formula_repair": False,
            "extractor_text_outputs_are_verbatim": True,
        },
        "diagnostic_note": (
            "Character counts are not an accuracy score. The findings in "
            "comparison.md are based on visual inspection of the cited PDF "
            "pages and corresponding extractor outputs."
        ),
    }
    with (output_dir / "run_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")

    print(
        json.dumps(
            {
                "pages": len(reader.pages),
                "methods": sorted(methods),
                "pymupdf_available": fitz_version is not None,
                "poppler_available": poppler_version is not None,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
