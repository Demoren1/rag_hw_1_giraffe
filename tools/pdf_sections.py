"""Read section names and anchors directly from a PDF's outline."""

from __future__ import annotations

from typing import Any

from pypdf import PdfReader


def extract_section_anchors(reader: PdfReader) -> list[dict[str, Any]]:
    """Return source outline destinations in top-left page coordinates.

    Names come from PDF bookmarks, without inferring headings from prose or
    equations. Coordinates refer to physical pages, numbered from one.
    """
    anchors: list[dict[str, Any]] = []

    def visit(entries: list[Any], depth: int) -> None:
        for destination in entries:
            if isinstance(destination, list):
                visit(destination, depth + 1)
                continue
            title = " ".join(str(destination.title).split())
            try:
                page_index = reader.get_destination_page_number(destination)
            except (KeyError, TypeError, ValueError):
                continue
            if page_index is None or page_index < 0 or not title:
                continue
            page_top = float(reader.pages[page_index].mediabox.top)
            try:
                destination_top = float(destination.get("/Top"))
            except (TypeError, ValueError):
                destination_top = page_top
            anchors.append(
                {
                    "page": page_index + 1,
                    "top": round(page_top - destination_top, 4),
                    "title": title,
                    "depth": depth,
                    "source": "pdf_outline_destination",
                }
            )

    visit(reader.outline, 0)
    anchors.sort(key=lambda item: (item["page"], item["top"], item["depth"]))
    return anchors


def section_for_position(
    anchors: list[dict[str, Any]],
    page: int,
    top: float,
) -> str:
    """Find the last source bookmark preceding a line's page and baseline."""
    section = ""
    for anchor in anchors:
        if anchor["page"] > page:
            break
        if anchor["page"] == page and anchor["top"] > top:
            break
        section = str(anchor["title"])
    return section
