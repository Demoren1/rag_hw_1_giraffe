#!/usr/bin/env python3
"""Build a deterministic, geometry-aware corpus from decoded PDF glyphs.

The input geometry comes from a separate PDF font decoder. This script does
not call OCR, a language model, or an external service. The original glyph and
rule records are copied into the output so every rendered line remains
auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from math_layout import LayoutResult, layout_page
from pdf_sections import extract_section_anchors, section_for_position
import pdf_spatial_text


CYRILLIC_LOWER_RE = re.compile(r"^[а-яё]")
PAGE_FURNITURE_HEADER_TERMS = (
    "ЛЕКЦИИПОТФКП",
    "ГОРЯЙНОВ",
    "ПОЛОВИНКИН",
)
VISUAL_REVIEW_PAGES = (11, 12, 34, 115, 139)
VISUAL_REVIEW_NOTES = {
    11: "Cauchy-Riemann equations and derivative fractions in Theorem 2.1.",
    12: "The condition contains a barred z in the displayed equation for f.",
    34: (
        "Contour integrals over gamma_r, the derivative formula, and contour "
        "orientation."
    ),
    115: (
        "Limits, infinite products, integrals, and a piecewise expression "
        "with a brace."
    ),
    139: (
        "The barred z-prime, gradient notation, a large fraction, "
        "and a square root."
    ),
}
STATEMENT_MARKER_RE = re.compile(
    r"^\s*(?:ТЕОРЕМА|Теорема|ОПРЕДЕЛЕНИЕ|Определение|"
    r"ЛЕММА|Лемма|СЛЕДСТВИЕ|Следствие|ПРЕДЛОЖЕНИЕ|"
    r"Предложение)\b",
    re.IGNORECASE,
)
PROOF_MARKER_RE = re.compile(
    r"^\s*(?:ДОКАЗАТЕЛЬСТВО|Доказательство)\b",
    re.IGNORECASE,
)
INCOMPLETE_OPERATOR_BOUND_RE = re.compile(
    r"\\(?:int|sum|prod|coprod|bigcup|bigcap)(?:\\limits)?"
    r"((?:_\{[^{}]*\}|\^\{[^{}]*\}){1,2})"
)


def parse_args() -> argparse.Namespace:
    """Parse the geometry-aware corpus builder options."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a semantic PDF corpus from decoded glyph geometry."
        )
    )
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--glyphs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--old-corpus",
        type=Path,
        default=None,
        help=(
            "Optional previous pilot corpus to compare in the quality report."
        ),
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Mark this build as using an interim glyph mapping.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines with line-specific errors."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on {path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on {path}:{line_number}"
                )
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write stable UTF-8 JSON Lines."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")


def write_json(path: Path, record: dict[str, Any]) -> None:
    """Write an indented UTF-8 JSON report."""
    path.write_text(
        json.dumps(
            record, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )


def normalize_compact(text: str) -> str:
    """Normalize a short line for header/footer recognition only."""
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "", text).upper()


def is_page_furniture(
    line: dict[str, Any], page_record: dict[str, Any]
) -> str | None:
    """Recognize repeated running heads and isolated physical page numbers."""
    text = str(line.get("text", "")).strip()
    if not text:
        return None
    bbox = line.get("bbox", [0, 0, 0, 0])
    y0 = float(bbox[1])
    y1 = float(bbox[3])
    height = max(1.0, float(page_record.get("height", 842.0)))
    top = y0 / height
    bottom = y1 / height
    compact = normalize_compact(text)

    if top < 0.085 and len(text) < 160:
        if any(term in compact for term in PAGE_FURNITURE_HEADER_TERMS):
            return "repeated_running_header"
        if re.fullmatch(r"\d{1,3}", compact):
            return "physical_page_number"
    if bottom > 0.94 and re.fullmatch(r"\d{1,3}", compact):
        return "physical_page_number"
    return None


def looks_like_math(text: str) -> bool:
    """Detect lines whose spatial grouping is likely part of a formula."""
    if text.strip() in {"√", "∫", "∑", "∏", "∐", "⋃", "⋂", "∇"}:
        return True
    math_marks = sum(
        text.count(char)
        for char in "=±−+-*/∂∫∑∏√γΓζξαβθλπρσφω∇′"
    )
    return math_marks >= 2 or "\\frac" in text or "\\int" in text


def looks_like_prose(text: str) -> bool:
    """Distinguish Russian prose from compact equation rows."""
    cyrillic = sum("А" <= char.upper() <= "Я" or char in "Ёё" for char in text)
    letters = sum(char.isalpha() for char in text)
    return (
        letters > 0
        and cyrillic / letters >= 0.32
        and not looks_like_math(text)
    )


def join_block_lines(lines: list[dict[str, Any]]) -> str:
    """Retain layout line breaks and safely dehyphenate Cyrillic word wraps."""
    parts: list[str] = []
    for line in lines:
        text = str(line.get("text", ""))
        if line.get("preserve_whitespace"):
            text = text.strip()
        else:
            text = re.sub(r"[\t\f\v ]+", " ", text).strip()
        if not text:
            continue
        if parts and parts[-1].endswith("-") and CYRILLIC_LOWER_RE.match(text):
            previous = parts[-1]
            if previous and re.search(r"[А-Яа-яЁё]-$", previous):
                parts[-1] = previous[:-1] + text
                continue
        parts.append(text)
    return "\n".join(parts).strip()


def _bbox_intersects(first: list[float], second: list[float]) -> bool:
    """Return whether two top-left-origin page rectangles overlap."""
    return (
        first[0] <= second[2]
        and first[2] >= second[0]
        and first[1] <= second[3]
        and first[3] >= second[1]
    )


def _union_bbox(boxes: list[list[float]]) -> list[float]:
    """Return the bounding rectangle around nonempty rectangles."""
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _glyph_geometry_bbox(glyph: Any) -> list[float]:
    """Return the union of a glyph's advance and outline bounds."""
    boxes = [list(glyph.bbox)]
    if glyph.ink_bbox is not None:
        boxes.append(list(glyph.ink_bbox))
    return _union_bbox(boxes)


def _ambiguity_glyph_ids(ambiguity: dict[str, Any]) -> list[int]:
    """Extract source glyph IDs named by a geometry ambiguity record."""
    indexes: set[int] = set()
    for key in (
        "glyph_index",
        "glyph_indices",
        "near_above_glyphs",
        "near_below_glyphs",
    ):
        value = ambiguity.get(key)
        if isinstance(value, list):
            indexes.update(int(index) for index in value)
        elif value is not None:
            indexes.add(int(value))
    for relation in ambiguity.get("relations", []):
        for key in ("glyph_index", "script_parent"):
            if relation.get(key) is not None:
                indexes.add(int(relation[key]))
    return sorted(indexes)


def _add_incomplete_operator_bound_ambiguities(
    page_layout: LayoutResult,
    rendered_lines: list[dict[str, Any]],
) -> None:
    """Mark operator bounds whose rendered endpoint is an unfinished sign."""
    existing = {
        tuple(ambiguity.get("glyph_indices", []))
        for ambiguity in page_layout.ambiguities
        if ambiguity.get("type") == "incomplete_operator_bound"
    }
    for line in rendered_lines:
        text = str(line.get("text", ""))
        for match in INCOMPLETE_OPERATOR_BOUND_RE.finditer(text):
            bounds = re.findall(r"(?:_|\^)\{([^{}]*)\}", match.group(1))
            if not any(
                value.rstrip().endswith(("+", "−", "-"))
                for value in bounds
            ):
                continue
            glyph_indices = tuple(
                sorted(int(index) for index in line.get("glyph_indices", []))
            )
            if not glyph_indices or glyph_indices in existing:
                continue
            page_layout.ambiguities.append(
                {
                    "type": "incomplete_operator_bound",
                    "operator": match.group(0).split("\\")[1].split("\\")[0],
                    "glyph_indices": list(glyph_indices),
                    "bbox": list(line["bbox"]),
                    "reason": "an operator limit ends with an unpaired sign",
                }
            )
            existing.add(glyph_indices)


def _add_nearby_delimiter_ambiguities(
    page_layout: LayoutResult,
    rendered_lines: list[dict[str, Any]],
) -> None:
    """Keep detached extensible bars with the nearby formula as geometry."""
    glyph_by_index = {glyph.index: glyph for glyph in page_layout.glyphs}
    base_size = statistics.median(
        glyph.size for glyph in page_layout.glyphs if glyph.size > 0
    ) if any(glyph.size > 0 for glyph in page_layout.glyphs) else 10.0
    for line in rendered_lines:
        line_ids = [int(index) for index in line.get("glyph_indices", [])]
        members = [
            glyph_by_index[index]
            for index in line_ids
            if index in glyph_by_index
        ]
        if not members or not all(
            glyph.structural_kind == "delimiter_extender" for glyph in members
        ):
            continue
        nearby = []
        for other in rendered_lines:
            if other is line or not looks_like_math(
                str(other.get("text", ""))
            ):
                continue
            first = [float(value) for value in line["bbox"]]
            second = [float(value) for value in other["bbox"]]
            vertical_gap = max(
                0.0,
                second[1] - first[3],
                first[1] - second[3],
            )
            horizontal_overlap = max(
                0.0,
                min(first[2], second[2]) - max(first[0], second[0]),
            )
            if (
                vertical_gap <= base_size * 1.8
                and horizontal_overlap >= base_size * 0.3
            ):
                nearby.append((vertical_gap, other))
        if not nearby:
            continue
        nearest_gap = min(item[0] for item in nearby)
        related_lines = [
            other
            for gap, other in nearby
            if gap <= nearest_gap + base_size * 0.35
        ]
        related_lines.append(line)
        member_ids = sorted(
            {
                int(index)
                for related in related_lines
                for index in related.get("glyph_indices", [])
            }
        )
        bounds = [
            [float(value) for value in related["bbox"]]
            for related in related_lines
        ]
        bbox = _union_bbox(bounds)
        if any(
            ambiguity.get("type") == "nearby_detached_delimiter"
            and set(ambiguity.get("glyph_indices", [])) == set(member_ids)
            for ambiguity in page_layout.ambiguities
        ):
            continue
        page_layout.ambiguities.append(
            {
                "type": "nearby_detached_delimiter",
                "glyph_indices": member_ids,
                "bbox": bbox,
                "reason": (
                    "detached extensible delimiters lie beside a math row; "
                    "preserve both together in source geometry"
                ),
            }
        )


def _accent_base_relations(glyphs: list[Any]) -> list[set[int]]:
    """Pair source accent glyphs with the nearest glyph directly below."""
    accents = [
        glyph
        for glyph in glyphs
        if glyph.structural_kind == "accent"
        and glyph.figure_id is None
    ]
    groups: list[set[int]] = []
    for accent in accents:
        accent_box = accent.ink_bbox or accent.bbox
        accent_width = accent_box[2] - accent_box[0]
        candidates = []
        for base in glyphs:
            if (
                base.index == accent.index
                or base.figure_id != accent.figure_id
                or base.structural_kind is not None
                or not base.text.strip()
            ):
                continue
            base_box = base.ink_bbox or base.bbox
            vertical_gap = base_box[1] - accent_box[3]
            max_gap = max(accent.size, base.size) * 0.6
            if vertical_gap < -accent.size * 0.05 or vertical_gap > max_gap:
                continue
            overlap = max(
                0.0,
                min(accent_box[2], base_box[2])
                - max(accent_box[0], base_box[0]),
            )
            min_width = min(
                accent_width,
                base_box[2] - base_box[0],
            )
            if min_width <= 0 or overlap < min_width * 0.45:
                continue
            candidates.append((vertical_gap, base.baseline, base.index))

        if not candidates:
            continue
        nearest_gap, nearest_baseline, _ = min(candidates)
        baseline_tolerance = max(0.6, accent.size * 0.12)
        base_ids = {
            base_index
            for vertical_gap, baseline, base_index in candidates
            if vertical_gap <= nearest_gap + baseline_tolerance
            and abs(baseline - nearest_baseline) <= baseline_tolerance
        }
        if base_ids:
            groups.append({accent.index, *base_ids})
    return groups


def _relation_groups(page_layout: LayoutResult) -> list[set[int]]:
    """Collect glyph groups consumed together by inferred layout relations."""
    groups = _accent_base_relations(page_layout.glyphs)
    for relation in (page_layout.superscripts, page_layout.subscripts):
        for parent, children in relation.items():
            groups.append({int(parent), *(glyph.index for glyph in children)})
    for fraction in page_layout.fraction_tokens:
        members = set(
            int(index) for index in fraction.get("member_glyph_ids", [])
        )
        if members:
            groups.append(members)
    for overbar in page_layout.overbar_tokens:
        members = set(int(index) for index in overbar.get("glyph_indices", []))
        if members:
            groups.append(members)
    for operator, relation in page_layout.big_operator_limits.items():
        groups.append(
            {
                int(operator),
                *(
                    glyph.index
                    for members in relation.values()
                    for glyph in members
                ),
            }
        )
    for delimiter in page_layout.stretch_delimiters:
        groups.append(
            set(int(index) for index in delimiter.get("glyph_indices", []))
        )
    return groups


def _closed_fallback_glyphs(
    page_layout: LayoutResult,
    rendered_lines: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> tuple[set[int], list[dict[str, Any]], list[float]]:
    """Close fallback membership over rows, scripts, fractions, and geometry.
    """
    selected_ids = {
        int(index)
        for region in regions
        for line in region["lines"]
        for index in line.get("glyph_indices", [])
    }
    protected_ids = set(selected_ids)
    for region in regions:
        member_ids = set(
            int(index) for index in region.get("member_glyph_ids", [])
        )
        selected_ids.update(member_ids)
        protected_ids.update(member_ids)
    seed_boxes = [region["bbox"] for region in regions]
    seed_boxes.extend(
        [float(value) for value in line["bbox"]]
        for region in regions
        for line in region["lines"]
    )
    seed_bbox = _union_bbox(seed_boxes)
    glyph_by_index = {glyph.index: glyph for glyph in page_layout.glyphs}
    relation_groups = _relation_groups(page_layout)

    for glyph in page_layout.glyphs:
        glyph_box = _glyph_geometry_bbox(glyph)
        center = [
            (glyph_box[0] + glyph_box[2]) / 2,
            (glyph_box[1] + glyph_box[3]) / 2,
        ]
        if (
            glyph.figure_id is None
            and seed_bbox[0] <= center[0] <= seed_bbox[2]
            and seed_bbox[1] <= center[1] <= seed_bbox[3]
        ):
            selected_ids.add(glyph.index)

    # A line that only overlaps the math bbox should remain searchable as
    # normal prose; its glyphs must not be stretched out into the source grid.
    candidate_line_ids = {
        int(candidate["line_index"])
        for region in regions
        for candidate in region["lines"]
    }
    for line in rendered_lines:
        if int(line["line_index"]) in candidate_line_ids:
            continue
        if looks_like_prose(str(line.get("text", ""))):
            selected_ids.difference_update(
                set(int(index) for index in line.get("glyph_indices", []))
                - protected_ids
            )

    positive_sizes = [
        glyph.size for glyph in page_layout.glyphs if glyph.size > 0
    ]
    base_size = statistics.median(positive_sizes) if positive_sizes else 10.0
    structural_bbox = [
        max(0.0, seed_bbox[0] - base_size * 1.2),
        max(0.0, seed_bbox[1] - base_size * 1.2),
        seed_bbox[2] + base_size * 1.2,
        seed_bbox[3] + base_size * 1.2,
    ]
    for glyph in page_layout.glyphs:
        if glyph.figure_id is not None:
            continue
        if glyph.structural_kind not in {
            "accent",
            "delimiter_part",
            "delimiter_extender",
        }:
            continue
        glyph_box = _glyph_geometry_bbox(glyph)
        center = [
            (glyph_box[0] + glyph_box[2]) / 2,
            (glyph_box[1] + glyph_box[3]) / 2,
        ]
        if (
            structural_bbox[0] <= center[0] <= structural_bbox[2]
            and structural_bbox[1] <= center[1] <= structural_bbox[3]
        ):
            selected_ids.add(glyph.index)

    changed = True
    while changed:
        previous_count = len(selected_ids)
        for relation in relation_groups:
            if selected_ids.intersection(relation):
                selected_ids.update(relation)
        for line in rendered_lines:
            line_ids = set(
                int(index) for index in line.get("glyph_indices", [])
            )
            if selected_ids.intersection(line_ids):
                selected_ids.update(line_ids)
        changed = len(selected_ids) != previous_count

    affected_lines = []
    for line in rendered_lines:
        line_ids = set(int(index) for index in line.get("glyph_indices", []))
        if selected_ids.intersection(line_ids):
            affected_lines.append(line)
    selected_boxes = [
        _glyph_geometry_bbox(glyph_by_index[index])
        for index in selected_ids
        if index in glyph_by_index and glyph_by_index[index].figure_id is None
    ]
    selected_bbox = (
        _union_bbox(selected_boxes) if selected_boxes else seed_bbox
    )
    return selected_ids, affected_lines, selected_bbox


def _render_selected_geometry(
    page_record: dict[str, Any],
    selected_ids: set[int],
    selected_bbox: list[float],
) -> tuple[str, dict[str, Any]]:
    """Render only the closed glyph/rule set at its natural geometric bounds.
    """
    glyphs = [
        glyph
        for index, glyph in enumerate(page_record.get("glyphs", []))
        if index in selected_ids and not glyph.get("figure_id")
    ]
    rules = []
    for index, rule in enumerate(page_record.get("rules", [])):
        if rule.get("figure_id"):
            continue
        try:
            rule_bbox = [
                min(float(rule["x0"]), float(rule["x1"])),
                min(float(rule["y0"]), float(rule["y1"])),
                max(float(rule["x0"]), float(rule["x1"])),
                max(float(rule["y0"]), float(rule["y1"])),
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if _bbox_intersects(rule_bbox, selected_bbox):
            rules.append(rule)

    filtered_page = {
        "page": page_record.get("page"),
        "width": page_record.get("width"),
        "height": page_record.get("height"),
        "glyphs": glyphs,
        "rules": rules,
        "figures": [],
    }
    previous_max_columns = pdf_spatial_text.MAX_COLUMNS
    needed_columns = int(
        max(0.0, selected_bbox[2] - selected_bbox[0])
        / pdf_spatial_text.MIN_CELL_WIDTH
    ) + 3
    pdf_spatial_text.MAX_COLUMNS = max(previous_max_columns, needed_columns)
    try:
        return pdf_spatial_text.render_region_with_report(filtered_page)
    finally:
        pdf_spatial_text.MAX_COLUMNS = previous_max_columns


def _fallback_regions(page_layout: LayoutResult) -> list[dict[str, Any]]:
    """Build local source-grid crops for geometrically unresolved math."""
    base_size = sorted(
        glyph.size for glyph in page_layout.glyphs if glyph.size > 0
    )
    size = base_size[len(base_size) // 2] if base_size else 10.0
    regions: list[dict[str, Any]] = []
    relevant_types = {
        "unresolved_rule",
        "overbar_not_substituted",
        "unresolved_delimiter_part",
        "fraction_intersects_external_script",
        "overlapping_fraction_regions",
        "unresolved_text_operator_limits",
        "large_operator_without_resolved_limits",
        "incomplete_operator_bound",
        "unpaired_stretched_delimiter",
        "nearby_detached_delimiter",
    }
    for ambiguity in page_layout.ambiguities:
        bbox = ambiguity.get("bbox")
        if (
            ambiguity.get("type") not in relevant_types
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue
        if ambiguity.get("type") == "unresolved_delimiter_part":
            continue
        regions.append(
            {
                "bbox": [float(value) for value in bbox],
                "reasons": [str(ambiguity.get("type"))],
                "member_glyph_ids": _ambiguity_glyph_ids(ambiguity),
            }
        )

    delimiter_parts = [
        glyph
        for glyph in page_layout.glyphs
        if glyph.figure_id is None
        and glyph.structural_kind == "delimiter_part"
    ]
    delimiter_groups: list[list[Any]] = []
    for glyph in sorted(
        delimiter_parts,
        key=lambda item: (item.delimiter_base or "", item.center_x, item.y0),
    ):
        group = next(
            (
                candidate
                for candidate in delimiter_groups
                if candidate[0].delimiter_base == glyph.delimiter_base
                and abs(candidate[0].center_x - glyph.center_x) <= size * 0.9
            ),
            None,
        )
        if group is None:
            delimiter_groups.append([glyph])
        else:
            group.append(glyph)
    for group in delimiter_groups:
        if len(group) < 3:
            continue
        part_box = _union_bbox([list(glyph.bbox) for glyph in group])
        regions.append(
            {
                "bbox": [
                    part_box[0] - size * 0.8,
                    part_box[1] - size * 0.6,
                    part_box[2] + size * 0.8,
                    part_box[3] + size * 0.6,
                ],
                "reasons": ["composite_delimiter_parts"],
                "member_glyph_ids": sorted(
                    glyph.index for glyph in group
                ),
            }
        )

    regions.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))
    merged: list[dict[str, Any]] = []
    for region in regions:
        target = next(
            (
                current
                for current in merged
                if _bbox_intersects(
                    [
                        current["bbox"][0] - size * 0.6,
                        current["bbox"][1] - size * 0.6,
                        current["bbox"][2] + size * 0.6,
                        current["bbox"][3] + size * 0.6,
                    ],
                    region["bbox"],
                )
            ),
            None,
        )
        if target is None:
            merged.append(region)
        else:
            target["bbox"] = _union_bbox([target["bbox"], region["bbox"]])
            target["reasons"] = sorted(
                set(target["reasons"]) | set(region["reasons"])
            )
            target["member_glyph_ids"] = sorted(
                set(target.get("member_glyph_ids", []))
                | set(region.get("member_glyph_ids", []))
            )
    for index, region in enumerate(merged, start=1):
        region["region_id"] = f"spatial_{index:03d}"
    return merged


def apply_spatial_fallbacks(
    page_record: dict[str, Any],
    page_layout: LayoutResult,
    rendered_lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace ambiguous math AST text with a closed source-geometry grid."""
    _add_incomplete_operator_bound_ambiguities(page_layout, rendered_lines)
    _add_nearby_delimiter_ambiguities(page_layout, rendered_lines)
    page_size_values = [
        glyph.size for glyph in page_layout.glyphs if glyph.size > 0
    ]
    page_size = (
        sorted(page_size_values)[len(page_size_values) // 2]
        if page_size_values
        else 10.0
    )
    regions = _fallback_regions(page_layout)
    region_candidates: list[dict[str, Any]] = []
    for region in regions:
        bbox = region["bbox"]
        pad = page_size * 1.1
        candidates = [
            line
            for line in rendered_lines
            if line.get("figure_id") is None
            and _bbox_intersects(
                [
                    float(line["bbox"][0]) - pad,
                    float(line["bbox"][1]) - pad,
                    float(line["bbox"][2]) + pad,
                    float(line["bbox"][3]) + pad,
                ],
                bbox,
            )
            and (
                looks_like_math(str(line.get("text", "")))
                or line.get("stretch_delimiter_glyph_indices")
                or bool(
                    set(int(index) for index in line.get("glyph_indices", []))
                    & set(region.get("member_glyph_ids", []))
                )
                or any(
                    page_layout.glyphs[index].structural_kind
                    == "delimiter_part"
                    for index in line.get("glyph_indices", [])
                    if index < len(page_layout.glyphs)
                )
            )
        ]
        if not candidates:
            continue
        region_candidates.append({**region, "lines": candidates})

    merged_candidates: list[dict[str, Any]] = []
    for item in region_candidates:
        item_ids = {line["line_index"] for line in item["lines"]}
        target = next(
            (
                current
                for current in merged_candidates
                if item_ids
                & {line["line_index"] for line in current["lines"]}
            ),
            None,
        )
        if target is None:
            merged_candidates.append(item)
        else:
            target["bbox"] = _union_bbox([target["bbox"], item["bbox"]])
            target["reasons"] = sorted(
                set(target["reasons"]) | set(item["reasons"])
            )
            known = {line["line_index"] for line in target["lines"]}
            target["lines"].extend(
                line
                for line in item["lines"]
                if line["line_index"] not in known
            )

    fallback_records: list[dict[str, Any]] = []
    for region_index, region in enumerate(merged_candidates, start=1):
        region_id = f"spatial_{region_index:03d}"
        selected_ids, affected_lines, selected_bbox = _closed_fallback_glyphs(
            page_layout,
            rendered_lines,
            [region],
        )
        if not affected_lines or not selected_ids:
            continue
        affected_lines.sort(
            key=lambda line: (line["baseline"], line["line_index"])
        )
        grid, report = _render_selected_geometry(
            page_record,
            selected_ids,
            selected_bbox,
        )
        crop = report.get("bbox", selected_bbox)
        primary = affected_lines[0]
        original_text = str(primary.get("text", ""))
        primary.update(
            {
                "ast_text_suppressed": original_text,
                "bbox": [round(value, 3) for value in crop],
                "glyph_indices": sorted(selected_ids),
                "preserve_whitespace": True,
                "spatial_fallback": {
                    "region_id": region_id,
                    "bbox": [round(value, 3) for value in crop],
                    "reasons": region["reasons"],
                    "collision_cells": report.get("counts", {}).get(
                        "collisions", 0
                    ),
                    "clipped_glyphs": len(report.get("clipped_glyphs", [])),
                    "out_of_grid_glyphs": len(
                        report.get("out_of_grid_glyphs", [])
                    ),
                    "clipped_rules": len(report.get("clipped_rules", [])),
                },
                "text": (
                    "[формула неоднозначна; исходная геометрия ниже]\n" + grid
                ),
            }
        )
        for line in affected_lines:
            if line is primary:
                continue
            line["ast_text_suppressed"] = str(line.get("text", ""))
            line["text"] = ""
            line["spatial_fallback_region_id"] = region_id
        fallback_records.append(
            {
                "region_id": region_id,
                "bbox": [round(value, 3) for value in crop],
                "reasons": region["reasons"],
                "line_indices": sorted(
                    int(line["line_index"]) for line in affected_lines
                ),
                "glyph_indices": sorted(selected_ids),
                "collision_cells": report.get("counts", {}).get(
                    "collisions", 0
                ),
                "clipped_glyphs": len(report.get("clipped_glyphs", [])),
                "out_of_grid_glyphs": len(
                    report.get("out_of_grid_glyphs", [])
                ),
                "clipped_rules": len(report.get("clipped_rules", [])),
                "excluded_figure_ids": [
                    figure["id"]
                    for figure in report.get("excluded_figures", [])
                ],
            }
        )
    return fallback_records


def split_page_blocks(
    page_number: int,
    page_record: dict[str, Any],
    page_layout: LayoutResult,
    rendered_lines: list[dict[str, Any]],
    source_sha256: str,
    initial_section: str,
    section_anchors: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    """Create semantic page blocks, per-block provenance, and line audit."""
    kept: list[dict[str, Any]] = []
    figure_lines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    furniture: list[dict[str, Any]] = []
    for line in rendered_lines:
        reason = is_page_furniture(line, page_record)
        if reason:
            furniture.append({"reason": reason, **line})
        elif line.get("figure_id"):
            figure_lines[str(line["figure_id"])].append(line)
        else:
            kept.append(line)

    section = initial_section
    groups: list[
        tuple[str, list[dict[str, Any]], str, dict[str, Any] | None]
    ] = []
    current: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    typical_size = sorted(
        glyph.size for glyph in page_layout.glyphs if glyph.size > 0
    )
    page_size = typical_size[len(typical_size) // 2] if typical_size else 10.0
    page_width = max(1.0, float(page_record.get("width", 595.0)))
    long_line_left_edges = [
        float(line["bbox"][0])
        for line in kept
        if float(line["bbox"][2]) - float(line["bbox"][0]) >= page_width * 0.58
    ]
    left_margin = min(long_line_left_edges, default=43.0)
    theorem_statement_active = False

    def flush() -> None:
        nonlocal current
        if current:
            groups.append((section, current, "text", None))
            current = []

    for line in kept:
        text = str(line.get("text", "")).strip()
        line_section = section_for_position(
            section_anchors, page_number, float(line["baseline"])
        )
        if line_section != section:
            flush()
            section = line_section
            previous = None
            theorem_statement_active = False

        if STATEMENT_MARKER_RE.match(text):
            flush()
            current.append(line)
            previous = line
            theorem_statement_active = True
            continue
        if PROOF_MARKER_RE.match(text):
            flush()
            current.append(line)
            previous = line
            theorem_statement_active = False
            continue

        if previous is not None:
            # Baselines remain stable for display equations whose integral or
            # brace glyph has an unusually tall font bounding box.
            gap = float(line["baseline"]) - float(previous["baseline"])
            previous_text = str(previous.get("text", ""))
            math_continuation = (
                looks_like_math(text) or looks_like_math(previous_text)
            )
            threshold = max(page_size * 1.75, 18.0)
            if theorem_statement_active:
                threshold = max(page_size * 3.0, 31.0)
            elif math_continuation:
                threshold = max(page_size * 2.75, 27.0)
            indent_break = (
                current
                and looks_like_prose(text)
                and looks_like_prose(previous_text)
                and float(line["bbox"][0]) > left_margin + page_size * 0.8
                and float(previous["bbox"][0]) <= left_margin + page_size * 0.5
            )
            if gap > threshold or indent_break:
                flush()
        current.append(line)
        previous = line
    flush()

    figures = {
        str(figure.get("id")): figure
        for figure in page_record.get("figures", [])
        if figure.get("id") is not None
    }

    def figure_root(figure_id: str) -> tuple[str, dict[str, Any] | None]:
        current_id = figure_id
        seen: set[str] = set()
        while current_id not in seen:
            seen.add(current_id)
            figure = figures.get(current_id)
            parent = figure.get("parent_figure_id") if figure else None
            if not parent:
                return current_id, figure
            current_id = str(parent)
        return figure_id, figures.get(figure_id)

    figure_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    figure_info: dict[str, dict[str, Any] | None] = {}
    for figure_id, lines in figure_lines.items():
        root_id, root_record = figure_root(figure_id)
        figure_groups[root_id].extend(lines)
        figure_info[root_id] = root_record
    for figure_id, lines in figure_groups.items():
        lines.sort(key=lambda line: (line["baseline"], line["line_index"]))
        top = min(float(line["bbox"][1]) for line in lines)
        figure_section = section_for_position(
            section_anchors, page_number, top
        )
        groups.append(
            (
                figure_section,
                lines,
                "figure",
                {
                    "figure_id": figure_id,
                    "name": (figure_info[figure_id] or {}).get("name"),
                },
            )
        )
    groups.sort(
        key=lambda group: (
            min(float(line["baseline"]) for line in group[1]),
            0 if group[2] == "text" else 1,
        )
    )

    blocks: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for ordinal, (
        block_section,
        block_lines,
        group_kind,
        group_info,
    ) in enumerate(groups, start=1):
        text = join_block_lines(block_lines)
        if group_kind == "figure":
            figure_name = (group_info or {}).get("name") or "без имени"
            figure_id = (group_info or {}).get("figure_id", "unknown")
            text = (
                f"[Подписи внутри рисунка {figure_name} ({figure_id}); "
                "текст рисунка не реконструировался как формула]\n"
                + text
            ).strip()
        if not text:
            continue
        block_digest = hashlib.sha1(
            f"{source_sha256}:{page_number}:{ordinal}:{text}".encode("utf-8")
        ).hexdigest()[:8]
        block_id = f"pdfv2_p{page_number:04d}_b{ordinal:04d}_{block_digest}"
        blocks.append(
            {
                "block_id": block_id,
                "page": page_number,
                "section": block_section,
                "text": text,
            }
        )
        glyph_indices = sorted(
            {
                glyph_index
                for line in block_lines
                for glyph_index in line.get("glyph_indices", [])
            }
        )
        line_indices = [line["line_index"] for line in block_lines]
        boxes = [line["bbox"] for line in block_lines]
        bbox = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
        fraction_ids = sorted(
            {
                rule_id
                for line in block_lines
                for rule_id in line.get("fraction_rule_indices", [])
            }
        )
        overbar_ids = sorted(
            {
                rule_id
                for line in block_lines
                for rule_id in line.get("overbar_rule_indices", [])
            }
        )
        figure_ids = sorted(
            {
                str(line["figure_id"])
                for line in block_lines
                if line.get("figure_id")
            }
        )
        spatial_fallback_ids = sorted(
            {
                str(line["spatial_fallback"]["region_id"])
                for line in block_lines
                if line.get("spatial_fallback")
            }
        )
        page_ambiguities = []
        for ambiguity_index, ambiguity in enumerate(page_layout.ambiguities):
            ambiguity_bbox = ambiguity.get("bbox")
            if not ambiguity_bbox or len(ambiguity_bbox) != 4:
                continue
            if (
                ambiguity_bbox[0] <= bbox[2]
                and ambiguity_bbox[2] >= bbox[0]
                and ambiguity_bbox[1] <= bbox[3]
                and ambiguity_bbox[3] >= bbox[1]
            ):
                page_ambiguities.append(ambiguity_index)

        provenance.append(
            {
                "block_id": block_id,
                "source_pdf_sha256": source_sha256,
                "page": page_number,
                "bbox": [round(value, 3) for value in bbox],
                "line_indices": line_indices,
                "glyph_indices": glyph_indices,
                "fraction_rule_indices": fraction_ids,
                "overbar_rule_indices": overbar_ids,
                "figure_ids": figure_ids,
                "spatial_fallback_region_ids": spatial_fallback_ids,
                "ambiguity_indices": page_ambiguities,
                "id_origin": (
                    "local deterministic ID; not an official corpus block_id"
                ),
            }
        )

    diagnostics = {
        "page": page_number,
        "width": page_record.get("width"),
        "height": page_record.get("height"),
        "lines": rendered_lines,
        "suppressed_page_furniture": furniture,
        "fraction_tokens": page_layout.fraction_tokens,
        "overbar_tokens": page_layout.overbar_tokens,
        "unresolved_rules": page_layout.unresolved_rules,
        "big_operator_limit_glyphs": {
            str(glyph_index): {
                name: [glyph.index for glyph in glyphs]
                for name, glyphs in relation.items()
            }
            for glyph_index, relation
            in page_layout.big_operator_limits.items()
        },
        "ambiguities": page_layout.ambiguities,
        "spatial_fallbacks": [
            line["spatial_fallback"]
            for line in rendered_lines
            if line.get("spatial_fallback")
        ],
        "figure_regions": page_record.get("figures", []),
        "page_size_median": page_size,
    }
    return blocks, provenance, [diagnostics], section


def old_corpus_comparison(
    old_corpus_path: Path | None,
    page_text: dict[int, str],
) -> dict[str, Any]:
    """Compare important physical pages with the untouched pilot corpus."""
    if old_corpus_path is None or not old_corpus_path.is_file():
        return {"available": False}

    old_records = read_jsonl(old_corpus_path)
    old_by_page = {
        int(record["page"]): str(record.get("text", ""))
        for record in old_records
        if "page" in record
    }
    pages: dict[str, Any] = {}
    for page_number in VISUAL_REVIEW_PAGES:
        old_text = old_by_page.get(page_number, "")
        new_text = page_text.get(page_number, "")
        compact_old = normalize_compact(old_text)
        compact_new = normalize_compact(new_text)
        pages[str(page_number)] = {
            "old_record_present": bool(old_text),
            "new_record_present": bool(new_text),
            "old_integral_as_Z_candidate": bool(
                re.search(r"(?:^|\s)Z(?:\s|$)", old_text)
            ),
            "new_integral_symbol_count": new_text.count("∫")
            + new_text.count("\\int"),
            "old_overbar_candidates_present": (
                "Z̄" in old_text
                or "Z\u0305" in old_text
                or "DZ̄" in compact_old
            ),
            "new_overbar_rendered": "\\overline{" in new_text
            or "\\bar{" in new_text
            or "z̄" in new_text
            or "z\u0305" in new_text,
            "old_characters": len(old_text),
            "new_characters": len(new_text),
            "new_excerpt": new_text[:1200],
            "compact_old_sha1": hashlib.sha1(
                compact_old.encode("utf-8")
            ).hexdigest(),
            "compact_new_sha1": hashlib.sha1(
                compact_new.encode("utf-8")
            ).hexdigest(),
        }
    return {
        "available": True,
        "old_corpus_path": str(old_corpus_path.resolve()),
        "old_records": len(old_records),
        "pages": pages,
    }


def build_corpus(
    pdf_path: Path,
    glyph_path: Path,
    output_dir: Path,
    old_corpus_path: Path | None = None,
    draft: bool = False,
) -> dict[str, Any]:
    """Build corpus JSONL, provenance, lossless geometry, and quality report.
    """
    pdf_path = pdf_path.expanduser().resolve()
    glyph_path = glyph_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not glyph_path.is_file():
        raise FileNotFoundError(f"Glyph geometry not found: {glyph_path}")
    pilot_dir = pdf_path.parent / "local_data" / "pilot"
    if output_dir.name == "pilot" or output_dir.resolve() == pilot_dir:
        raise ValueError("The output must not overwrite the old pilot corpus.")

    glyph_pages = read_jsonl(glyph_path)
    if not glyph_pages:
        raise ValueError("The glyph geometry file contains no pages.")
    pages = sorted(glyph_pages, key=lambda record: int(record["page"]))
    page_numbers = [int(record["page"]) for record in pages]
    expected_pages = list(range(1, len(page_numbers) + 1))
    if page_numbers != expected_pages:
        raise ValueError(
            "Glyph pages must be consecutive, unique, and 1-based."
        )

    pdf_reader = PdfReader(pdf_path)
    if len(pdf_reader.pages) != len(pages):
        raise ValueError(
            f"PDF has {len(pdf_reader.pages)} pages, but glyph geometry has "
            f"{len(pages)} records."
        )
    section_anchors = extract_section_anchors(pdf_reader)
    source_sha256 = sha256_file(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    page_text: dict[int, str] = {}
    glyph_count = 0
    rule_count = 0
    mapping_unknown_count = 0
    fraction_count = 0
    overbar_count = 0
    unresolved_rule_count = 0
    ambiguity_count = 0
    spatial_fallback_count = 0
    fallback_clipped_glyph_count = 0
    fallback_out_of_grid_glyph_count = 0
    fallback_clipped_rule_count = 0
    fallback_collision_cell_count = 0
    fallback_regions_with_render_issues = 0
    figure_region_count = 0
    pages_with_ambiguous_math: set[int] = set()
    pages_with_blocks: set[int] = set()
    line_count = 0
    furniture_count = 0
    page_summaries: list[dict[str, Any]] = []
    current_section = ""

    for page_record in pages:
        page_number = int(page_record["page"])
        layout, rendered_lines = layout_page(page_record)
        apply_spatial_fallbacks(page_record, layout, rendered_lines)
        (
            blocks,
            page_provenance,
            page_diagnostics,
            current_section,
        ) = split_page_blocks(
            page_number,
            page_record,
            layout,
            rendered_lines,
            source_sha256,
            current_section,
            section_anchors,
        )
        corpus.extend(blocks)
        provenance.extend(page_provenance)
        diagnostics.extend(page_diagnostics)
        if blocks:
            pages_with_blocks.add(page_number)
            page_text[page_number] = "\n".join(
                block["text"] for block in blocks
            )

        page_glyph_count = len(page_record.get("glyphs", []))
        page_rule_count = len(page_record.get("rules", []))
        page_unknown_count = sum(
            1
            for glyph in page_record.get("glyphs", [])
            if glyph.get("unknown") is True
            or str(glyph.get("mapping", "")).lower()
            in {"unknown", "fallback_unknown"}
        )
        glyph_count += page_glyph_count
        rule_count += page_rule_count
        mapping_unknown_count += page_unknown_count
        fraction_count += len(layout.fraction_tokens)
        overbar_count += len(layout.overbar_tokens)
        unresolved_rule_count += len(layout.unresolved_rules)
        ambiguity_count += len(layout.ambiguities)
        page_fallbacks = page_diagnostics[0]["spatial_fallbacks"]
        spatial_fallback_count += len(page_fallbacks)
        fallback_clipped_glyph_count += sum(
            int(item.get("clipped_glyphs", 0)) for item in page_fallbacks
        )
        fallback_out_of_grid_glyph_count += sum(
            int(item.get("out_of_grid_glyphs", 0)) for item in page_fallbacks
        )
        fallback_clipped_rule_count += sum(
            int(item.get("clipped_rules", 0)) for item in page_fallbacks
        )
        fallback_collision_cell_count += sum(
            int(item.get("collision_cells", 0)) for item in page_fallbacks
        )
        fallback_regions_with_render_issues += sum(
            1
            for item in page_fallbacks
            if item.get("clipped_glyphs")
            or item.get("out_of_grid_glyphs")
            or item.get("clipped_rules")
            or item.get("collision_cells")
        )
        figure_region_count += len(page_record.get("figures", []))
        if layout.ambiguities:
            pages_with_ambiguous_math.add(page_number)
        line_count += len(rendered_lines)
        furniture_count += len(
            page_diagnostics[0]["suppressed_page_furniture"]
        )
        page_summaries.append(
            {
                "page": page_number,
                "glyphs": page_glyph_count,
                "rules": page_rule_count,
                "lines": len(rendered_lines),
                "blocks": len(blocks),
                "fraction_tokens": len(layout.fraction_tokens),
                "overbar_tokens": len(layout.overbar_tokens),
                "unresolved_rules": len(layout.unresolved_rules),
                "ambiguities": len(layout.ambiguities),
                "unknown_glyphs": page_unknown_count,
                "spatial_fallbacks": len(page_fallbacks),
                "fallback_clipped_glyphs": sum(
                    int(item.get("clipped_glyphs", 0))
                    for item in page_fallbacks
                ),
                "fallback_out_of_grid_glyphs": sum(
                    int(item.get("out_of_grid_glyphs", 0))
                    for item in page_fallbacks
                ),
                "fallback_clipped_rules": sum(
                    int(item.get("clipped_rules", 0))
                    for item in page_fallbacks
                ),
                "fallback_collision_cells": sum(
                    int(item.get("collision_cells", 0))
                    for item in page_fallbacks
                ),
                "figure_regions": len(page_record.get("figures", [])),
            }
        )

    corpus_path = output_dir / "corpus.jsonl"
    provenance_path = output_dir / "provenance.jsonl"
    geometry_path = output_dir / "source_geometry.jsonl"
    diagnostics_path = output_dir / "layout_diagnostics.jsonl"
    report_path = output_dir / "quality_report.json"
    write_jsonl(corpus_path, corpus)
    write_jsonl(provenance_path, provenance)
    write_jsonl(diagnostics_path, diagnostics)
    shutil.copyfile(glyph_path, geometry_path)

    old_path = old_corpus_path
    if old_path is None:
        candidate = (
            pdf_path.parent
            / "local_data"
            / "pilot"
            / "pilot_corpus.jsonl"
        )
        old_path = candidate if candidate.is_file() else None

    report = {
        "status": (
            "draft_interim_mapping"
            if draft
            else "geometry_aware_deterministic_corpus"
        ),
        "draft": draft,
        "source": {
            "pdf_path": str(pdf_path),
            "pdf_name": pdf_path.name,
            "pdf_sha256": source_sha256,
            "pdf_bytes": pdf_path.stat().st_size,
            "physical_page_count": len(pages),
            "page_numbering": "1-based physical PDF page index",
            "glyph_geometry_path": str(glyph_path),
            "glyph_geometry_sha256": sha256_file(glyph_path),
        },
        "method": {
            "uses_embedded_glyph_geometry": True,
            "uses_vector_rules": True,
            "uses_ocr": False,
            "uses_llm": False,
            "external_calls": False,
            "source_geometry_copied_losslessly": True,
            "math_policy": (
                "Emit LaTeX for conservative fractions, attached scripts, "
                "clear overbars, and large-operator limits. Keep uncertain "
                "relations in page geometry and diagnostics."
            ),
            "header_footer_policy": (
                "Suppress only short repeated running heads and isolated "
                "physical page numbers from retrieval blocks; preserve them "
                "in layout_diagnostics.jsonl and source_geometry.jsonl."
            ),
            "id_policy": (
                "New local deterministic IDs; no official block_id is "
                "inferred."
            ),
            "spatial_fallback_policy": (
                "When math geometry is unresolved or nested relations "
                "overlap, "
                "suppress the reconstructed formula text in that local region "
                "and include a deterministic source-faithful glyph/rule grid."
            ),
        },
        "coverage": {
            "pdf_pages": len(pages),
            "pages_with_glyph_geometry": sum(
                1 for record in pages if record.get("glyphs")
            ),
            "pages_with_corpus_blocks": len(pages_with_blocks),
            "page_coverage": round(len(pages_with_blocks) / len(pages), 6),
            "corpus_blocks": len(corpus),
            "rendered_lines": line_count,
            "glyphs": glyph_count,
            "vector_rules": rule_count,
            "suppressed_running_headers_or_page_numbers": furniture_count,
            "unknown_or_replacement_glyphs": mapping_unknown_count,
            "unknown_glyph_fraction": round(
                mapping_unknown_count / max(glyph_count, 1), 6
            ),
        },
        "math_layout": {
            "fraction_tokens": fraction_count,
            "overbar_tokens": overbar_count,
            "unresolved_horizontal_rules": unresolved_rule_count,
            "ambiguity_events": ambiguity_count,
            "pages_with_ambiguities": sorted(pages_with_ambiguous_math),
            "ambiguity_rate_per_rule": round(
                ambiguity_count / max(rule_count, 1), 6
            ),
            "ambiguities_are_not_silently_repaired": True,
            "source_grid_fallback_regions": spatial_fallback_count,
            "source_grid_fallback_pages": sorted(
                summary["page"]
                for summary in page_summaries
                if summary["spatial_fallbacks"]
            ),
            "source_grid_render_quality": {
                "regions_with_clipping_or_collisions": (
                    fallback_regions_with_render_issues
                ),
                "clipped_glyphs": fallback_clipped_glyph_count,
                "out_of_grid_glyphs": fallback_out_of_grid_glyph_count,
                "clipped_rules": fallback_clipped_rule_count,
                "collision_cells": fallback_collision_cell_count,
                "interpretation": (
                    "These are rendering/placement issues in the spatial "
                    "fallbacks; nonzero counts remain visible quality issues, "
                    "not evidence of a clean transcription."
                ),
            },
        },
        "figures": {
            "form_regions": figure_region_count,
            "policy": (
                "Keep text labels in marked figure blocks; do not interpret "
                "Form XObject content as mathematical prose."
            ),
        },
        "comparison_with_old_pilot": old_corpus_comparison(
            old_path, page_text
        ),
        "visually_reviewed_physical_pages": [
            {
                "page": page,
                "source_features_seen": VISUAL_REVIEW_NOTES[page],
                "review_scope": (
                    "The source page was visually inspected; this is a "
                    "targeted "
                    "review scope, not a claim that every formula was "
                    "verified."
                ),
            }
            for page in VISUAL_REVIEW_PAGES
        ],
        "outputs": {
            "corpus": str(corpus_path),
            "provenance": str(provenance_path),
            "source_geometry": str(geometry_path),
            "layout_diagnostics": str(diagnostics_path),
            "quality_report": str(report_path),
        },
        "pages": page_summaries,
    }
    write_json(report_path, report)
    return report


def main() -> int:
    """Run the complete deterministic corpus extraction."""
    args = parse_args()
    report = build_corpus(
        pdf_path=args.pdf,
        glyph_path=args.glyphs,
        output_dir=args.output,
        old_corpus_path=args.old_corpus,
        draft=args.draft,
    )
    summary = {
        "status": report["status"],
        "pages": report["coverage"]["pdf_pages"],
        "blocks": report["coverage"]["corpus_blocks"],
        "page_coverage": report["coverage"]["page_coverage"],
        "fraction_tokens": report["math_layout"]["fraction_tokens"],
        "overbar_tokens": report["math_layout"]["overbar_tokens"],
        "ambiguity_events": report["math_layout"]["ambiguity_events"],
        "output": report["outputs"]["corpus"],
        "quality_report": report["outputs"]["quality_report"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
