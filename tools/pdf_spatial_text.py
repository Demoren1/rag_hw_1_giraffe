"""Render PDF glyph records as a compact, coordinate-faithful text grid.

This module deliberately does not infer mathematical syntax. It places the
font-aware glyphs supplied by ``font_analysis/glyph_pages.jsonl`` according
to their PDF coordinates and draws vector rules as separate strokes. The
renderer is intended as a transparent fallback for inspecting difficult
formula regions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "local_data" / "font_analysis" / "glyph_pages.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "local_data" / "spatial_text"
MAX_COLUMNS = 120
MAX_ROWS = 160
MIN_CELL_WIDTH = 1.8
MIN_CELL_HEIGHT = 2.0
COLLISION_MARKER = "※"

# Unicode Mathematical Operators and Miscellaneous Technical pieces used to
# render embedded CMEX extensible delimiters. The matching glyph names come
# from the PDF's embedded Type 1 Encoding; code points follow the Unicode
# chart U+2300–U+23FF (https://www.unicode.org/charts/PDF/U2300.pdf).
DELIMITER_PART_GLYPHS = {
    "parenlefttp": "⎛",
    "parenleftmid": "⎜",
    "parenleftbt": "⎝",
    "parenleftex": "⎜",
    "parenrighttp": "⎞",
    "parenrightmid": "⎟",
    "parenrightbt": "⎠",
    "parenrightex": "⎟",
    "bracketlefttp": "⎡",
    "bracketleftmid": "⎢",
    "bracketleftbt": "⎣",
    "bracketleftex": "⎢",
    "bracketrighttp": "⎤",
    "bracketrightmid": "⎥",
    "bracketrightbt": "⎦",
    "bracketrightex": "⎥",
    "bracelefttp": "⎧",
    "braceleftmid": "⎨",
    "braceleftbt": "⎩",
    "braceex": "⎪",
    "bracerighttp": "⎫",
    "bracerightmid": "⎬",
    "bracerightbt": "⎭",
}
ACCENT_DISPLAY_GLYPHS = {
    "hatwide": ("\u0302", "\u02c6"),
    "tildewide": ("\u0303", "\u02dc"),
}

# These regions were selected from rendered page images and their glyph
# geometry. They are demonstration inputs only; render_region has no
# page-specific behavior.
DEMO_REGIONS = [
    {
        "page": 12,
        "label": "p12_zbar_derivative",
        "bbox": [197.0, 518.0, 243.5, 550.5],
        "selection": (
            "Region around the displayed derivative, selected from the "
            "rendered page."
        ),
    },
    {
        "page": 34,
        "label": "p34_theorem_5_3",
        "bbox": [111.0, 435.0, 410.0, 480.0],
        "selection": (
            "The equation line beneath Theorem 5.3, selected from the "
            "rendered page."
        ),
    },
    {
        "page": 38,
        "label": "p38_exponential_piecewise",
        "bbox": [184.0, 359.0, 300.0, 402.0],
        "selection": (
            "The piecewise example containing the exponential, "
            "selected from the rendered page."
        ),
    },
    {
        "page": 115,
        "label": "p115_gamma_product",
        "bbox": [138.0, 214.0, 310.0, 254.0],
        "selection": (
            "The displayed Gamma product, selected from the rendered page."
        ),
    },
    {
        "page": 145,
        "label": "p145_fractional_powers_root",
        "bbox": [130.0, 541.0, 320.0, 570.0],
        "selection": (
            "The displayed asymptotic formula with a radical and fractional "
            "powers, "
            "selected from the rendered page."
        ),
    },
]


def _as_bbox(value: Any) -> tuple[float, float, float, float] | None:
    """Return a finite, ordered bbox or None when the input is malformed."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x0, y0, x1, y1)):
        return None
    if x1 < x0 or y1 < y0:
        return None
    return x0, y0, x1, y1


def _glyph_bbox(
    glyph: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    """Prefer outline ink bounds, then the PDF text advance rectangle."""
    return (
        _as_bbox(glyph.get("ink_bbox"))
        or _as_bbox(glyph.get("bbox"))
        or _as_bbox(glyph.get("advance_bbox"))
    )


def _intersects(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return (
        first[0] <= second[2]
        and first[2] >= second[0]
        and first[1] <= second[3]
        and first[3] >= second[1]
    )


def _source_summary(glyph: dict[str, Any], index: int) -> dict[str, Any]:
    """Keep enough source data to audit a collision or a clipped glyph."""
    keys = (
        "text",
        "raw_text",
        "code",
        "code_hex",
        "font",
        "font_key",
        "font_resource",
        "glyph_name",
        "glyph_name_source",
        "structural_kind",
        "delimiter_base",
        "delimiter_part",
        "delimiter_scale",
        "mapping",
        "provenance",
        "ink_bbox_provenance",
        "tex_token",
        "confidence",
        "unknown",
        "bbox",
        "ink_bbox",
        "baseline",
        "figure_id",
    )
    result = {key: glyph.get(key) for key in keys if key in glyph}
    result["record_index"] = index
    return result


def _glyph_text(glyph: dict[str, Any]) -> str:
    """Return the decoded glyph text, with a visible raw-code fallback."""
    text = glyph.get("text")
    if isinstance(text, str) and text:
        return text
    code = glyph.get("code_hex") or glyph.get("code")
    return f"⟦{code if code is not None else 'empty-glyph'}⟧"


def _delimiter_part_text(glyph: dict[str, Any]) -> str | None:
    """Render a known CMEX assembly piece with its standard Unicode glyph."""
    if glyph.get("structural_kind") != "delimiter_part":
        return None
    return DELIMITER_PART_GLYPHS.get(str(glyph.get("glyph_name") or ""))


def _accent_display_text(glyph: dict[str, Any]) -> str | None:
    """Use a visible spacing form while retaining the source accent text."""
    if glyph.get("structural_kind") != "accent":
        return None
    mapping = ACCENT_DISPLAY_GLYPHS.get(str(glyph.get("glyph_name") or ""))
    if mapping and glyph.get("text") == mapping[0]:
        return mapping[1]
    return None


def _cell_width(text: str) -> int:
    """Approximate terminal monospace width without an external dependency."""
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return max(width, 1)


def _clip_line(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Clip a line segment to a rectangle with Liang-Barsky parameters."""
    left, top, right, bottom = bbox
    dx, dy = x1 - x0, y1 - y0
    p_values = (-dx, dx, -dy, dy)
    q_values = (x0 - left, right - x0, y0 - top, bottom - y0)
    start, end = 0.0, 1.0
    for p_value, q_value in zip(p_values, q_values):
        if abs(p_value) < 1e-12:
            if q_value < 0:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0:
            start = max(start, ratio)
        else:
            end = min(end, ratio)
        if start > end:
            return None
    return x0 + start * dx, y0 + start * dy, x0 + end * dx, y0 + end * dy


def _line_cells(
    segment: tuple[float, float, float, float],
    bbox: tuple[float, float, float, float],
    cell_width: float,
    cell_height: float,
) -> list[tuple[int, int]]:
    """Map a clipped rule segment to grid cells with integer Bresenham."""
    x0, y0, x1, y1 = segment
    col0 = round((x0 - bbox[0]) / cell_width)
    row0 = round((y0 - bbox[1]) / cell_height)
    col1 = round((x1 - bbox[0]) / cell_width)
    row1 = round((y1 - bbox[1]) / cell_height)
    dx, dy = abs(col1 - col0), abs(row1 - row0)
    sx = 1 if col0 < col1 else -1
    sy = 1 if row0 < row1 else -1
    error = dx - dy
    result = []
    while True:
        result.append((row0, col0))
        if col0 == col1 and row0 == row1:
            break
        twice_error = 2 * error
        if twice_error > -dy:
            error -= dy
            col0 += sx
        if twice_error < dx:
            error += dx
            row0 += sy
    return result


def _rule_character(segment: tuple[float, float, float, float]) -> str:
    """Choose a line-drawing character from its direction, not its meaning."""
    x0, y0, x1, y1 = segment
    dx, dy = x1 - x0, y1 - y0
    if abs(dy) <= max(0.3, abs(dx) * 0.025):
        return "─"
    if abs(dx) <= max(0.3, abs(dy) * 0.025):
        return "│"
    return "╲" if dx * dy > 0 else "╱"


def _in_figure_by_center(
    x: float,
    y: float,
    figures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    containing = []
    for figure in figures:
        figure_bbox = _as_bbox(figure.get("bbox"))
        if (
            figure_bbox
            and figure_bbox[0] <= x <= figure_bbox[2]
            and figure_bbox[1] <= y <= figure_bbox[3]
        ):
            area = (figure_bbox[2] - figure_bbox[0]) * (
                figure_bbox[3] - figure_bbox[1]
            )
            containing.append((area, figure))
    return min(containing, default=(None, None), key=lambda item: item[0])[1]


def render_region_with_report(
    page_record: dict[str, Any],
    bbox: Iterable[float] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render a page or crop and return its text plus transparent diagnostics.

    Coordinates are PDF points in the same top-left origin used by the glyph
    records. ``ink_bbox`` is used when available; its exact embedded-font
    outline bounds help place large math glyphs relative to their axis.
    """
    page_number = page_record.get("page")
    figures = page_record.get("figures") or []
    requested_bbox = _as_bbox(bbox) if bbox is not None else None
    if bbox is not None and requested_bbox is None:
        raise ValueError(
            "bbox must contain four finite values [x0, y0, x1, y1]"
        )

    glyph_entries = []
    excluded_figure_ids = {
        str(figure.get("id", figure.get("name", "unnamed")))
        for figure in figures
        if requested_bbox is None
        or (
            _as_bbox(figure.get("bbox"))
            and _intersects(_as_bbox(figure.get("bbox")), requested_bbox)
        )
    }
    for index, glyph in enumerate(page_record.get("glyphs") or []):
        glyph_bbox = _glyph_bbox(glyph)
        figure_id = glyph.get("figure_id")
        if figure_id is not None:
            if requested_bbox is None or (
                glyph_bbox and _intersects(glyph_bbox, requested_bbox)
            ):
                excluded_figure_ids.add(str(figure_id))
            continue
        if glyph_bbox is None:
            continue
        glyph_center = (
            (glyph_bbox[0] + glyph_bbox[2]) / 2,
            (glyph_bbox[1] + glyph_bbox[3]) / 2,
        )
        figure = _in_figure_by_center(*glyph_center, figures)
        if figure:
            identifier = figure.get("id", figure.get("name", "unnamed"))
            excluded_figure_ids.add(str(identifier))
            continue
        if requested_bbox is None or _intersects(glyph_bbox, requested_bbox):
            glyph_entries.append((index, glyph, glyph_bbox))

    rule_entries = []
    for index, rule in enumerate(page_record.get("rules") or []):
        if rule.get("figure_id") is not None:
            if requested_bbox is None:
                excluded_figure_ids.add(str(rule.get("figure_id")))
            continue
        try:
            x0 = float(rule["x0"])
            y0 = float(rule["y0"])
            x1 = float(rule["x1"])
            y1 = float(rule["y1"])
        except (KeyError, TypeError, ValueError):
            continue
        center_figure = _in_figure_by_center(
            (x0 + x1) / 2, (y0 + y1) / 2, figures
        )
        if center_figure:
            identifier = center_figure.get(
                "id", center_figure.get("name", "unnamed")
            )
            excluded_figure_ids.add(str(identifier))
            continue
        if requested_bbox is None or _clip_line(
            x0, y0, x1, y1, requested_bbox
        ):
            rule_entries.append((index, rule, (x0, y0, x1, y1)))

    if requested_bbox is None:
        bounds = [entry[2] for entry in glyph_entries]
        bounds.extend(
            (
                min(line[0], line[2]),
                min(line[1], line[3]),
                max(line[0], line[2]),
                max(line[1], line[3]),
            )
            for _, _, line in rule_entries
        )
        if bounds:
            crop = (
                min(item[0] for item in bounds),
                min(item[1] for item in bounds),
                max(item[2] for item in bounds),
                max(item[3] for item in bounds),
            )
        else:
            crop = (
                0.0,
                0.0,
                float(page_record.get("width") or 0),
                float(page_record.get("height") or 0),
            )
    else:
        crop = requested_bbox

    width = max(crop[2] - crop[0], MIN_CELL_WIDTH)
    height = max(crop[3] - crop[1], MIN_CELL_HEIGHT)
    cell_width = max(MIN_CELL_WIDTH, width / MAX_COLUMNS)
    cell_height = max(MIN_CELL_HEIGHT, height / MAX_ROWS)
    row_count = max(1, math.ceil(height / cell_height) + 1)
    column_count = max(1, math.ceil(width / cell_width) + 1)

    # Each occupied cell retains all source records. A collision is made
    # visible in the grid and fully described in the report below.
    cell_sources: dict[tuple[int, int], list[dict[str, Any]]] = {}
    clipped_glyphs = []
    out_of_grid_glyphs = []
    display_mappings = []

    for index, glyph, glyph_bbox in glyph_entries:
        visible = _intersects(glyph_bbox, crop)
        if not visible:
            continue
        clipped = not (
            glyph_bbox[0] >= crop[0]
            and glyph_bbox[1] >= crop[1]
            and glyph_bbox[2] <= crop[2]
            and glyph_bbox[3] <= crop[3]
        )
        if clipped:
            clipped_glyphs.append(_source_summary(glyph, index))
        ink_bbox = _as_bbox(glyph.get("ink_bbox"))
        advance_bbox = _as_bbox(glyph.get("bbox")) or _as_bbox(
            glyph.get("advance_bbox")
        )
        glyph_text = _glyph_text(glyph)
        is_combining_overlay = bool(glyph_text) and all(
            unicodedata.combining(char) for char in glyph_text
        )
        font_label = " ".join(
            str(glyph.get(key) or "") for key in ("font", "font_pdfminer_name")
        ).upper()
        is_cmex_outline = "CMEX" in font_label and ink_bbox is not None
        horizontal_bounds = (
            ink_bbox
            if ink_bbox is not None
            and (is_cmex_outline or is_combining_overlay)
            else (advance_bbox or glyph_bbox)
        )
        anchor_x = (horizontal_bounds[0] + horizontal_bounds[2]) / 2
        baseline = glyph.get("baseline")
        if is_cmex_outline:
            anchor_y = (ink_bbox[1] + ink_bbox[3]) / 2
            if glyph.get("structural_kind") == "operator":
                size = glyph.get("size")
                if isinstance(size, (int, float)) and math.isfinite(
                    float(size)
                ):
                    anchor_y += 0.25 * float(size)
        elif isinstance(baseline, (int, float)) and math.isfinite(
            float(baseline)
        ):
            anchor_y = float(baseline)
        else:
            anchor_y = (glyph_bbox[1] + glyph_bbox[3]) / 2
        col = round((anchor_x - crop[0]) / cell_width)
        row = round((anchor_y - crop[1]) / cell_height)
        edge_markers = []
        if col < 0:
            edge_markers.append("‹")
            col = 0
        elif col >= column_count:
            edge_markers.append("›")
            col = column_count - 1
        if row < 0:
            edge_markers.append("⌃")
            row = 0
        elif row >= row_count:
            edge_markers.append("⌄")
            row = row_count - 1
        source = _source_summary(glyph, index)
        source["edge_markers"] = edge_markers
        delimiter_part_text = _delimiter_part_text(glyph)
        accent_display_text = _accent_display_text(glyph)
        visible_text = delimiter_part_text or accent_display_text or glyph_text
        source["rendered_text"] = "".join(edge_markers) + visible_text
        display_mapping = None
        if delimiter_part_text:
            display_mapping = {
                "from_glyph_name": glyph.get("glyph_name"),
                "to_unicode": f"U+{ord(delimiter_part_text):04X}",
                "text": delimiter_part_text,
                "source": (
                    "Unicode chart U+2300-U+23FF; source glyph_name from the "
                    "PDF's embedded CMEX Type 1 Encoding"
                ),
            }
        elif accent_display_text:
            display_mapping = {
                "from_glyph_name": glyph.get("glyph_name"),
                "source_text": glyph_text,
                "to_unicode": f"U+{ord(accent_display_text):04X}",
                "text": accent_display_text,
                "source": (
                    "Display-only spacing form of the embedded CMEX wide "
                    "accent; the source glyph text and ink-bbox placement "
                    "are retained"
                ),
            }
        if display_mapping:
            source["display_mapping"] = display_mapping
            display_mappings.append({"record_index": index, **display_mapping})
        source["cell"] = [row, col]
        source["clipped"] = clipped
        cell_sources.setdefault((row, col), []).append(source)
        for extra_col in range(1, _cell_width(source["rendered_text"])):
            if col + extra_col < column_count:
                cell_sources.setdefault((row, col + extra_col), []).append(
                    {**source, "wide_glyph_continuation": True}
                )

    # Some PDF fonts emit a combining overlay as a separate glyph. NFC
    # composition keeps the native Unicode pieces visible as one grapheme;
    # source records for both glyphs remain in these diagnostics.
    overlays = []
    for cell, glyphs_here in list(cell_sources.items()):
        independent = [
            item
            for item in glyphs_here
            if not item.get("wide_glyph_continuation")
        ]
        bases = []
        marks = []
        for item in independent:
            rendered_text = item.get("rendered_text", "")
            if rendered_text and all(
                unicodedata.combining(char) for char in rendered_text
            ):
                marks.append(item)
            else:
                bases.append(item)
        if len(bases) != 1 or not marks:
            continue
        unnormalized = bases[0]["rendered_text"] + "".join(
            item["rendered_text"] for item in marks
        )
        normalized = unicodedata.normalize("NFC", unnormalized)
        if normalized == unnormalized:
            continue
        bases[0]["rendered_text"] = normalized
        mark_indexes = {item.get("record_index") for item in marks}
        cell_sources[cell] = [
            item
            for item in glyphs_here
            if item.get("record_index") not in mark_indexes
        ]
        overlays.append(
            {
                "cell": list(cell),
                "base_glyph": bases[0],
                "overlay_glyphs": marks,
                "composed_text": normalized,
                "normalization": "Unicode NFC of the source glyph texts",
            }
        )

    rule_sources: dict[tuple[int, int], list[dict[str, Any]]] = {}
    clipped_rules = []
    for index, rule, original_line in rule_entries:
        clipped_line = _clip_line(*original_line, crop)
        if clipped_line is None:
            continue
        if clipped_line != original_line:
            clipped_rules.append(
                {
                    "rule_index": index,
                    "original": original_line,
                    "visible_segment": clipped_line,
                    "width": rule.get("width"),
                    "figure_id": rule.get("figure_id"),
                }
            )
        char = _rule_character(clipped_line)
        for row, col in _line_cells(
            clipped_line, crop, cell_width, cell_height
        ):
            if 0 <= row < row_count and 0 <= col < column_count:
                rule_sources.setdefault((row, col), []).append(
                    {
                        "rule_index": index,
                        "character": char,
                        "original": original_line,
                        "width": rule.get("width"),
                    }
                )

    all_cells = set(cell_sources) | set(rule_sources)
    collisions = []
    for cell in sorted(all_cells):
        glyphs_here = cell_sources.get(cell, [])
        rules_here = rule_sources.get(cell, [])
        if (
            len(glyphs_here) > 1
            or (glyphs_here and rules_here)
            or len(rules_here) > 1
        ):
            # A continuation cell is part of its original wide glyph rather
            # than a second independent glyph for collision accounting.
            unique_glyphs = []
            seen_indexes = set()
            for item in glyphs_here:
                source_index = item.get("record_index")
                if source_index not in seen_indexes:
                    seen_indexes.add(source_index)
                    unique_glyphs.append(item)
            if (
                len(unique_glyphs) > 1
                or (unique_glyphs and rules_here)
                or len(rules_here) > 1
            ):
                collisions.append(
                    {
                        "cell": list(cell),
                        "glyphs": unique_glyphs,
                        "rules": rules_here,
                    }
                )

    occupied_rows = [row for row, _ in all_cells]
    occupied_columns = [col for _, col in all_cells]
    if occupied_rows and occupied_columns:
        top_row, bottom_row = min(occupied_rows), max(occupied_rows)
        left_col, right_col = min(occupied_columns), max(occupied_columns)
    else:
        top_row = bottom_row = left_col = right_col = 0

    display_lines = []
    for row in range(top_row, bottom_row + 1):
        line_chars = []
        col = left_col
        while col <= right_col:
            key = (row, col)
            glyphs_here = cell_sources.get(key, [])
            rules_here = rule_sources.get(key, [])
            collision_here = any(
                tuple(record["cell"]) == key for record in collisions
            )
            if collision_here:
                line_chars.append(COLLISION_MARKER)
            elif glyphs_here:
                primary = next(
                    (
                        item
                        for item in glyphs_here
                        if not item.get("wide_glyph_continuation")
                    ),
                    glyphs_here[0],
                )
                rendered_text = primary["rendered_text"]
                line_chars.append(rendered_text)
                col += _cell_width(rendered_text) - 1
            elif rules_here:
                line_chars.append(rules_here[0]["character"])
            else:
                line_chars.append(" ")
            col += 1
        display_lines.append("".join(line_chars).rstrip())
    while display_lines and not display_lines[0].strip():
        display_lines.pop(0)
        top_row += 1
    while display_lines and not display_lines[-1].strip():
        display_lines.pop()

    grid_origin = [
        crop[0] + left_col * cell_width,
        crop[1] + top_row * cell_height,
    ]
    header = (
        f"page={page_number} "
        f"bbox=[{crop[0]:.2f}, {crop[1]:.2f}, {crop[2]:.2f}, {crop[3]:.2f}] "
        f"origin=[{grid_origin[0]:.2f}, {grid_origin[1]:.2f}] "
        f"cell={cell_width:.2f}x{cell_height:.2f}pt (x,y; top-left)"
    )
    rendered_lines = [header, *display_lines]
    if excluded_figure_ids:
        figure_labels = []
        for figure in figures:
            identifier = str(figure.get("id", figure.get("name", "unnamed")))
            if identifier in excluded_figure_ids:
                figure_labels.append(
                    {
                        "id": identifier,
                        "name": figure.get("name"),
                        "bbox": figure.get("bbox"),
                    }
                )
        rendered_lines.append(
            "[FIGURE CONTENT OMITTED: "
            f"{len(figure_labels)} region(s); not interpreted as math]"
        )

    unknown_glyphs = [
        _source_summary(glyph, index)
        for index, glyph, _ in glyph_entries
        if (glyph.get("unknown") or not glyph.get("text"))
        and _delimiter_part_text(glyph) is None
    ]
    if collisions:
        rendered_lines.append(
            f"[{len(collisions)} collision(s) marked {COLLISION_MARKER}; "
            "see diagnostics]"
        )
    if clipped_glyphs or out_of_grid_glyphs or clipped_rules:
        rendered_lines.append(
            "[clipped/out-of-grid records: "
            f"glyphs={len(clipped_glyphs)}, "
            f"glyphs_out={len(out_of_grid_glyphs)}, "
            f"rules={len(clipped_rules)}; "
            "see diagnostics]"
        )
    if unknown_glyphs:
        rendered_lines.append(
            f"[unknown/empty glyph records={len(unknown_glyphs)}; "
            "visible raw-code fallbacks used]"
        )

    rendered_glyph_count = len(
        {
            item.get("record_index")
            for sources in cell_sources.values()
            for item in sources
            if not item.get("wide_glyph_continuation")
        }
    ) + sum(len(item["overlay_glyphs"]) for item in overlays)
    report = {
        "page": page_number,
        "requested_bbox": list(requested_bbox) if requested_bbox else None,
        "bbox": list(crop),
        "coordinate_system": (
            "PDF points; top-left origin; same absolute coordinates as source "
            "glyph records"
        ),
        "anchor_policy": (
            "ordinary glyphs use advance-bbox x-center and source baseline y; "
            "CMEX glyphs with embedded outlines use ink-bbox center, with "
            "operators shifted by 0.25em; malformed/missing baselines fall "
            "back to bbox center"
        ),
        "grid": {
            "cell_width_pt": cell_width,
            "cell_height_pt": cell_height,
            "rows": len(display_lines),
            "columns": right_col - left_col + 1,
            "origin": grid_origin,
            "trimmed_cell_offset": [left_col, top_row],
            "max_columns": MAX_COLUMNS,
            "max_rows": MAX_ROWS,
        },
        "counts": {
            "glyphs_considered": len(glyph_entries),
            "rules_considered": len(rule_entries),
            "glyphs_rendered": rendered_glyph_count,
            "rule_cells": len(rule_sources),
            "collisions": len(collisions),
            "clipped_glyphs": len(clipped_glyphs),
            "out_of_grid_glyphs": len(out_of_grid_glyphs),
            "clipped_rules": len(clipped_rules),
            "unknown_or_empty_glyphs": len(unknown_glyphs),
        },
        "excluded_figures": [
            {
                "id": str(figure.get("id", figure.get("name", "unnamed"))),
                "name": figure.get("name"),
                "bbox": figure.get("bbox"),
            }
            for figure in figures
            if str(figure.get("id", figure.get("name", "unnamed")))
            in excluded_figure_ids
        ],
        "collisions": collisions,
        "overlays": overlays,
        "glyph_display_mappings": display_mappings,
        "clipped_glyphs": clipped_glyphs,
        "out_of_grid_glyphs": out_of_grid_glyphs,
        "clipped_rules": clipped_rules,
        "unknown_or_empty_glyphs": unknown_glyphs,
    }
    return "\n".join(rendered_lines), report


def render_region(
    page_record: dict[str, Any], bbox: Iterable[float] | None = None
) -> str:
    """Render a full page or bbox crop as a compact monospaced spatial grid."""
    return render_region_with_report(page_record, bbox)[0]


def _read_pages(
    path: Path, required_pages: set[int]
) -> dict[int, dict[str, Any]]:
    """Read only requested page records from the JSONL stream."""
    pages: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            page_number = record.get("page")
            if page_number in required_pages:
                pages[page_number] = record
                if pages.keys() >= required_pages:
                    break
    missing = required_pages - pages.keys()
    if missing:
        raise ValueError(f"Pages not found in {path}: {sorted(missing)}")
    return pages


def _atomic_write_text(path: Path, content: str) -> None:
    """Write a complete artifact and atomically replace its destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def _demo(input_path: Path, output_dir: Path) -> None:
    """Create reproducible region examples and detailed render diagnostics."""
    pages = _read_pages(
        input_path, {int(item["page"]) for item in DEMO_REGIONS}
    )
    output_sections = []
    reports = []
    region_manifest = []
    for region in DEMO_REGIONS:
        record = pages[int(region["page"])]
        text, report = render_region_with_report(record, region["bbox"])
        report["label"] = region["label"]
        report["selection"] = region["selection"]
        reports.append(report)
        region_manifest.append(
            {
                "page": region["page"],
                "label": region["label"],
                "bbox": region["bbox"],
                "selection": region["selection"],
            }
        )
        output_sections.append(f"### {region['label']}\n{text}")
    _atomic_write_text(
        output_dir / "samples.txt", "\n\n".join(output_sections) + "\n"
    )
    _atomic_write_text(
        output_dir / "render_report.json",
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        output_dir / "demo_regions.json",
        json.dumps(region_manifest, ensure_ascii=False, indent=2) + "\n",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render font-aware PDF glyph records as a coordinate-faithful "
            "text grid."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Glyph-page JSONL input",
    )
    parser.add_argument(
        "--page", type=int, help="Physical page number to render"
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Optional crop in PDF points with a top-left origin",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Render the five visually selected formula regions to "
            "local_data/spatial_text"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional text output path; defaults to stdout",
    )
    parser.add_argument(
        "--report", type=Path, help="Optional diagnostics JSON output path"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Demo artifact directory",
    )
    args = parser.parse_args()
    if not args.demo and args.page is None:
        parser.error("--page is required unless --demo is specified")
    if args.demo and args.bbox:
        parser.error("--bbox cannot be combined with --demo")
    return args


def main() -> None:
    args = _parse_args()
    if args.demo:
        _demo(args.input, args.output_dir)
        return
    page_record = _read_pages(args.input, {args.page})[args.page]
    rendered, report = render_region_with_report(page_record, args.bbox)
    if args.output:
        _atomic_write_text(args.output, rendered + "\n")
    else:
        print(rendered)
    if args.report:
        _atomic_write_text(
            args.report,
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )


if __name__ == "__main__":
    main()
