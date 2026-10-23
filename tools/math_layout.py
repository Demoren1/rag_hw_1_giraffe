"""Deterministic geometry helpers for PDF text and mathematical notation.

The functions in this module use only decoded PDF glyphs and vector rules.
They never infer missing content. Uncertain spatial relations remain in the
lossless source geometry and are reported as ambiguities by the corpus builder.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable


MATH_OPERATORS = {"∫", "∑", "∏", "∐", "⋃", "⋂"}
OPERATOR_NAMES = {
    "∫": "int",
    "∑": "sum",
    "∏": "prod",
    "∐": "coprod",
    "⋃": "bigcup",
    "⋂": "bigcap",
}
OPERATOR_TOKEN_NAMES = {
    "int": "int",
    "sum": "sum",
    "prod": "prod",
    "coprod": "coprod",
    "bigcup": "bigcup",
    "bigcap": "bigcap",
}
SCRIPT_TOKEN = re.compile(
    r"^[A-Za-zА-Яа-яЁёΑ-ωϑϕϖϱϵϰ0-9′'(),;:+−/*∗→∞-]+$"
)
HORIZONTAL_TOLERANCE = 0.8


@dataclass(frozen=True)
class Glyph:
    """One decoded glyph with top-left-origin PDF coordinates."""

    index: int
    text: str
    raw_text: str
    font: str
    size: float
    bbox: tuple[float, float, float, float]
    baseline: float
    ink_bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    mapping: str | None = None
    tex_token: str | None = None
    font_resource: str | None = None
    code: int | None = None
    glyph_name: str | None = None
    figure_id: str | None = None
    structural_kind: str | None = None
    delimiter_base: str | None = None
    delimiter_scale: str | None = None
    delimiter_part: str | None = None

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Row:
    """Glyphs sharing an approximate baseline."""

    index: int
    baseline: float
    glyphs: list[Glyph] = field(default_factory=list)
    figure_id: str | None = None

    @property
    def x0(self) -> float:
        return min((glyph.x0 for glyph in self.glyphs), default=0.0)

    @property
    def x1(self) -> float:
        return max((glyph.x1 for glyph in self.glyphs), default=0.0)

    @property
    def y0(self) -> float:
        return min((glyph.y0 for glyph in self.glyphs), default=self.baseline)

    @property
    def y1(self) -> float:
        return max((glyph.y1 for glyph in self.glyphs), default=self.baseline)

    @property
    def median_size(self) -> float:
        return median_size(self.glyphs)


@dataclass(frozen=True)
class Rule:
    """One horizontal vector segment."""

    index: int
    x0: float
    y0: float
    x1: float
    y1: float
    stroke_width: float
    figure_id: str | None = None

    @property
    def length(self) -> float:
        """Return the segment's horizontal extent, independent of stroke."""
        return abs(self.x1 - self.x0)

    @property
    def x_min(self) -> float:
        return min(self.x0, self.x1)

    @property
    def x_max(self) -> float:
        return max(self.x0, self.x1)

    @property
    def y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class LayoutResult:
    """Reconstructed page rows and conservative math relations."""

    rows: list[Row]
    fraction_rules: list[dict[str, Any]]
    fraction_tokens: list[dict[str, Any]]
    overbar_rules: list[dict[str, Any]]
    overbar_tokens: list[dict[str, Any]]
    stretch_delimiters: list[dict[str, Any]]
    unresolved_rules: list[dict[str, Any]]
    superscripts: dict[int, list[Glyph]]
    subscripts: dict[int, list[Glyph]]
    big_operator_limits: dict[int, dict[str, list[Glyph]]]
    ambiguities: list[dict[str, Any]]
    glyphs: list[Glyph]
    figure_rule_count: int


def _bbox_contains(
    outer: tuple[float, float, float, float],
    x: float,
    y: float,
) -> bool:
    """Return whether a point lies inside a top-left-origin rectangle."""
    return outer[0] <= x <= outer[2] and outer[1] <= y <= outer[3]


def _innermost_figure_id(
    bbox: Iterable[float], figures: list[dict[str, Any]]
) -> str | None:
    """Find the smallest containing Form XObject region for a bbox."""
    coords = tuple(_float(value) for value in bbox)
    if len(coords) != 4:
        return None
    center_x = (coords[0] + coords[2]) / 2
    center_y = (coords[1] + coords[3]) / 2
    containing: list[tuple[float, str]] = []
    for figure in figures:
        figure_bbox = figure.get("bbox", ())
        if len(figure_bbox) != 4:
            continue
        rect = tuple(_float(value) for value in figure_bbox)
        if _bbox_contains(rect, center_x, center_y):
            area = max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])
            containing.append((area, str(figure.get("id", ""))))
    if not containing:
        return None
    return min(containing)[1]


def _float(value: Any, default: float = 0.0) -> float:
    """Convert an optional numeric field to a finite float."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def median_size(glyphs: Iterable[Glyph]) -> float:
    """Return the median positive glyph size, or a safe fallback."""
    values = [glyph.size for glyph in glyphs if glyph.size > 0]
    return statistics.median(values) if values else 10.0


def _is_cyrillic_letter(text: str) -> bool:
    """Return whether a text atom contains Cyrillic letters."""
    return any(
        char.isalpha() and ("А" <= char.upper() <= "Я" or char in "Ёё")
        for char in text
    )


def _glyph_space_threshold(
    previous: Glyph, current: Glyph, *, math_mode: bool = False
) -> float:
    """Use tight word-gap detection for the PDF's compact Cyrillic prose."""
    size = min(previous.size, current.size)
    if (
        not math_mode
        and (
            _is_cyrillic_letter(previous.text)
            or _is_cyrillic_letter(current.text)
        )
        and any(char.isalpha() or char == "∂" for char in previous.text)
        and any(char.isalpha() or char == "∂" for char in current.text)
    ):
        return max(0.65, size * 0.105)
    return max(2.0, size * (0.29 if math_mode else 0.25))


def parse_glyphs(page_record: dict[str, Any]) -> list[Glyph]:
    """Convert the font decoder's page schema to validated glyph objects."""
    result: list[Glyph] = []
    figures = page_record.get("figures", [])
    for index, item in enumerate(page_record.get("glyphs", [])):
        bbox = item.get("bbox", ())
        if len(bbox) != 4:
            continue
        coords = tuple(_float(value) for value in bbox)
        if coords[2] < coords[0] or coords[3] < coords[1]:
            continue
        text = str(item.get("text", ""))
        structural_kind = str(item.get("structural_kind") or "")
        glyph_name = (
            str(item["glyph_name"])
            if item.get("glyph_name") is not None
            else None
        )
        if not text and structural_kind == "delimiter_part":
            text = str(item.get("raw_text") or f"[{glyph_name or 'part'}]")
        if not text:
            continue
        size = _float(item.get("size"), max(1.0, coords[3] - coords[1]))
        baseline = _float(item.get("baseline"), coords[3])
        ink_bbox_value = item.get("ink_bbox")
        ink_bbox = None
        if (
            isinstance(ink_bbox_value, (list, tuple))
            and len(ink_bbox_value) == 4
        ):
            ink_bbox = tuple(_float(value) for value in ink_bbox_value)
        if (
            structural_kind == "delimiter"
            and str(item.get("font", "")).upper().startswith("CMEX")
            and ink_bbox is not None
        ):
            baseline = (ink_bbox[1] + ink_bbox[3]) / 2 + size * 0.25
        figure_id = item.get("figure_id") or _innermost_figure_id(
            coords, figures
        )
        result.append(
            Glyph(
                index=index,
                text=text,
                raw_text=str(item.get("raw_text", "")),
                font=str(item.get("font", "")),
                size=size,
                bbox=coords,
                baseline=baseline,
                ink_bbox=ink_bbox,
                confidence=(
                    _float(item["confidence"])
                    if item.get("confidence") is not None
                    else None
                ),
                mapping=(
                    str(item.get("mapping"))
                    if item.get("mapping") is not None
                    else None
                ),
                tex_token=(
                    str(item.get("tex_token"))
                    if item.get("tex_token") is not None
                    else None
                ),
                font_resource=(
                    str(item.get("font_resource"))
                    if item.get("font_resource") is not None
                    else None
                ),
                code=(
                    int(item["code"])
                    if item.get("code") is not None
                    else None
                ),
                glyph_name=glyph_name,
                figure_id=str(figure_id) if figure_id else None,
                structural_kind=structural_kind or None,
                delimiter_base=(
                    str(item["delimiter_base"])
                    if item.get("delimiter_base") is not None
                    else None
                ),
                delimiter_scale=(
                    str(item["delimiter_scale"])
                    if item.get("delimiter_scale") is not None
                    else None
                ),
                delimiter_part=(
                    str(item["delimiter_part"])
                    if item.get("delimiter_part") is not None
                    else None
                ),
            )
        )
    return result


def parse_rules(page_record: dict[str, Any]) -> list[Rule]:
    """Convert horizontal vector candidates to validated rule objects."""
    result: list[Rule] = []
    figures = page_record.get("figures", [])
    for index, item in enumerate(page_record.get("rules", [])):
        x0 = _float(item.get("x0"))
        y0 = _float(item.get("y0"))
        x1 = _float(item.get("x1"))
        y1 = _float(item.get("y1"))
        if abs(y1 - y0) > max(1.5, abs(x1 - x0) * 0.025):
            continue
        figure_id = item.get("figure_id") or _innermost_figure_id(
            (x0, y0, x1, y1), figures
        )
        result.append(
            Rule(
                index=index,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                stroke_width=max(0.0, _float(item.get("width"))),
                figure_id=str(figure_id) if figure_id else None,
            )
        )
    return result


def group_rows(glyphs: list[Glyph]) -> list[Row]:
    """Group glyphs by baseline without flattening raised or lowered text."""
    rows: list[Row] = []
    for glyph in sorted(glyphs, key=lambda item: (item.baseline, item.x0)):
        candidate: Row | None = None
        best_delta = math.inf
        for row in reversed(rows[-8:]):
            if row.figure_id != glyph.figure_id:
                continue
            delta = abs(row.baseline - glyph.baseline)
            tolerance = max(
                HORIZONTAL_TOLERANCE,
                min(row.median_size, glyph.size) * 0.09,
            )
            if delta <= tolerance and delta < best_delta:
                candidate = row
                best_delta = delta
        if candidate is None:
            rows.append(
                Row(
                    index=len(rows),
                    baseline=glyph.baseline,
                    figure_id=glyph.figure_id,
                )
            )
            candidate = rows[-1]
        candidate.glyphs.append(glyph)
        candidate.baseline = statistics.median(
            item.baseline for item in candidate.glyphs
        )

    for index, row in enumerate(rows):
        row.index = index
        row.glyphs.sort(key=lambda item: (item.x0, item.x1, item.index))
    return sorted(rows, key=lambda row: (row.baseline, row.x0))


def _text_for_glyphs(
    glyphs: Iterable[Glyph], *, math_mode: bool = False
) -> str:
    """Read glyphs from left to right and restore obvious word boundaries."""
    ordered = sorted(glyphs, key=lambda item: (item.x0, item.x1, item.index))
    pieces: list[str] = []
    previous: Glyph | None = None
    for glyph in ordered:
        text = glyph.text
        if text.isspace():
            if pieces and pieces[-1] != " ":
                pieces.append(" ")
            previous = glyph
            continue
        if previous is not None and not previous.text.isspace():
            gap = glyph.x0 - previous.x1
            threshold = _glyph_space_threshold(
                previous, glyph, math_mode=math_mode
            )
            if gap > threshold and pieces and pieces[-1] != " ":
                pieces.append(" ")
        pieces.append(text)
        previous = glyph
    text = "".join(pieces)
    text = re.sub(r"[\t\f\v ]+", " ", text)
    return text.strip()


def row_text(row: Row) -> str:
    """Return the conservative Unicode text for a row."""
    return _text_for_glyphs(row.glyphs)


def _span(glyphs: Iterable[Glyph]) -> tuple[float, float]:
    glyph_list = list(glyphs)
    if not glyph_list:
        return 0.0, 0.0
    return min(g.x0 for g in glyph_list), max(g.x1 for g in glyph_list)


def _line_clusters(
    glyphs: list[Glyph], tolerance: float = 1.4
) -> list[list[Glyph]]:
    """Cluster nearby glyphs with nearly equal baselines."""
    clusters: list[list[Glyph]] = []
    for glyph in sorted(glyphs, key=lambda item: (item.baseline, item.x0)):
        if clusters and abs(
            statistics.median(item.baseline for item in clusters[-1])
            - glyph.baseline
        ) <= max(tolerance, min(glyph.size, median_size(clusters[-1])) * 0.12):
            clusters[-1].append(glyph)
        else:
            clusters.append([glyph])
    return clusters


def _rule_side_glyphs(
    rule: Rule,
    glyphs: list[Glyph],
    base_size: float,
    *,
    side: str,
    excluded_ids: set[int] | None = None,
) -> list[Glyph]:
    """Find nearest text near one side of a rule's horizontal extent."""
    excluded_ids = excluded_ids or set()
    if rule.figure_id:
        return []
    margin = max(1.5, base_size * 0.08)
    x_min = rule.x_min - margin
    x_max = rule.x_max + margin
    candidates = [
        glyph
        for glyph in glyphs
        if glyph.figure_id is None
        and glyph.index not in excluded_ids
        and glyph.x1 >= x_min
        and glyph.x0 <= x_max
        and (
            glyph.baseline < rule.y - 0.2
            if side == "above"
            else glyph.baseline > rule.y + 0.2
        )
    ]
    if not candidates:
        return []

    def distance(glyph: Glyph) -> float:
        return abs(glyph.baseline - rule.y)

    nearest = min(distance(glyph) for glyph in candidates)
    near_limit = max(2.0, min(base_size * 1.65, nearest + base_size * 0.75))
    near = [glyph for glyph in candidates if distance(glyph) <= near_limit]
    clusters = _line_clusters(near)
    if not clusters:
        return []
    cluster = min(
        clusters,
        key=lambda group: (
            -_rule_overlap(rule, group),
            abs(
                statistics.median(glyph.baseline for glyph in group)
                - rule.y
            ),
        ),
    )
    if _rule_overlap(rule, cluster) < 0.15:
        return []
    return sorted(cluster, key=lambda glyph: (glyph.x0, glyph.index))


def _rule_overlap(rule: Rule, glyphs: list[Glyph]) -> float:
    if rule.length <= 0:
        return 0.0
    overlap = sum(
        max(
            0.0,
            min(glyph.x1, rule.x_max) - max(glyph.x0, rule.x_min),
        )
        for glyph in glyphs
    )
    return min(1.0, overlap / rule.length)


def _is_cyrillic_prose_context(
    cluster: list[Glyph], all_glyphs: list[Glyph]
) -> bool:
    """Recognize a Cyrillic fragment embedded in a nearby prose line."""
    letters = [
        char
        for glyph in cluster
        for char in glyph.text
        if "А" <= char.upper() <= "Я" or char in "Ёё"
    ]
    cluster_letters = [
        char for glyph in cluster for char in glyph.text if char.isalpha()
    ]
    if not letters or len(letters) != len(cluster_letters):
        return False
    baseline = statistics.median(glyph.baseline for glyph in cluster)
    context = [
        glyph
        for glyph in all_glyphs
        if abs(glyph.baseline - baseline)
        <= max(1.4, min(glyph.size, median_size(cluster)) * 0.14)
    ]
    context_letters = [
        1
        for glyph in context
        for char in glyph.text
        if char.isalpha()
    ]
    context_cyrillic = [
        char
        for glyph in context
        for char in glyph.text
        if "А" <= char.upper() <= "Я" or char in "Ёё"
    ]
    return (
        len(context_cyrillic) >= 4
        and len(context_cyrillic) / max(len(context_letters), 1) >= 0.62
    )


def _association_for_rule(
    rule: Rule,
    rules: list[Rule],
    glyphs: list[Glyph],
    base_size: float,
    excluded_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Conservatively classify a rule as fraction, overbar, or unresolved."""
    excluded_ids = excluded_ids or set()
    above = _rule_side_glyphs(
        rule,
        glyphs,
        base_size,
        side="above",
        excluded_ids=excluded_ids,
    )
    below = _rule_side_glyphs(
        rule,
        glyphs,
        base_size,
        side="below",
        excluded_ids=excluded_ids,
    )
    upper_overlap = _rule_overlap(rule, above)
    lower_overlap = _rule_overlap(rule, below)
    upper_baseline = (
        statistics.median(glyph.baseline for glyph in above)
        if above
        else math.inf
    )
    upper_is_cyrillic_prose = _is_cyrillic_prose_context(above, glyphs)
    parent_rule_between = any(
        candidate.index != rule.index
        and upper_baseline < candidate.y < rule.y
        and rule.length > 0
        and max(
            0.0,
            min(candidate.x_max, rule.x_max)
            - max(candidate.x_min, rule.x_min),
        )
        / rule.length
        >= 0.55
        and rule.y - candidate.y <= base_size * 1.1
        for candidate in rules
    )
    small_fraction_components = bool(above and below) and max(
        median_size(above), median_size(below)
    ) <= base_size * 0.62
    maximum_distance_ratio = 5.0 if small_fraction_components else 2.5
    if (
        above
        and below
        and not upper_is_cyrillic_prose
        and rule.stroke_width >= 0.05
        and upper_overlap >= 0.32
        and lower_overlap >= 0.32
    ):
        upper_distance = abs(
            statistics.median(glyph.baseline for glyph in above) - rule.y
        )
        lower_distance = abs(
            statistics.median(glyph.baseline for glyph in below) - rule.y
        )
        distance_ratio = max(upper_distance, lower_distance) / max(
            min(upper_distance, lower_distance), 0.1
        )
        if (
            not parent_rule_between
            and
            max(upper_distance, lower_distance) <= base_size * 1.55
            and distance_ratio <= maximum_distance_ratio
        ):
            return {
                "kind": "fraction",
                "rule_index": rule.index,
                "numerator_glyphs": above,
                "denominator_glyphs": below,
                "geometry_score": min(
                    0.95,
                    0.62 + 0.33 * min(upper_overlap, lower_overlap),
                ),
                "bbox": [
                    rule.x_min,
                    rule.y - base_size,
                    rule.x_max,
                    rule.y + base_size,
                ],
            }

    upper_distance = (
        abs(statistics.median(glyph.baseline for glyph in above) - rule.y)
        if above
        else math.inf
    )
    lower_distance = (
        abs(statistics.median(glyph.baseline for glyph in below) - rule.y)
        if below
        else math.inf
    )
    separated_upper_text = (
        above and upper_distance > max(base_size * 1.35, lower_distance * 2.3)
    )
    if (
        below
        and rule.stroke_width >= 0.05
        and (
            not above
            or separated_upper_text
            or parent_rule_between
            or upper_is_cyrillic_prose
        )
        and rule.length <= base_size * 3.4
    ):
        barred_glyphs = [
            glyph
            for glyph in below
            if rule.x_min - max(1.0, glyph.size * 0.1)
            <= glyph.center_x
            <= rule.x_max + max(1.0, glyph.size * 0.1)
        ]
        lower_text = _text_for_glyphs(barred_glyphs, math_mode=True)
        overlap = _rule_overlap(rule, barred_glyphs)
        radical_nearby = any(
            glyph.text.strip() in {"√", "\u221a"}
            and 0 <= rule.x_min - glyph.x1 <= max(2.0, glyph.size * 0.3)
            and abs(glyph.baseline - below[0].baseline) <= glyph.size * 0.35
            for glyph in glyphs
        )
        if lower_text and overlap >= 0.55 and not radical_nearby:
            return {
                "kind": "overbar",
                "rule_index": rule.index,
                "barred_glyphs": barred_glyphs,
                "geometry_score": min(0.93, 0.58 + 0.35 * overlap),
                "bbox": [
                    rule.x_min,
                    rule.y - base_size,
                    rule.x_max,
                    rule.y + base_size,
                ],
                "parent_rule_between": parent_rule_between,
                "upper_context_is_cyrillic_prose": upper_is_cyrillic_prose,
            }

    reason = (
        "rule has no unambiguous numerator/denominator or short overbar "
        "relation"
    )
    if not above and below and rule.length > base_size * 3.4:
        reason = "long line may be a radical vinculum or another construction"
    elif below and not above and rule.length <= base_size * 3.4:
        reason = "short line may be an overbar; relation score is too low"
    return {
        "kind": "unresolved",
        "rule_index": rule.index,
        "reason": reason,
        "near_above_glyphs": [glyph.index for glyph in above],
        "near_below_glyphs": [glyph.index for glyph in below],
        "bbox": [rule.x_min, rule.y, rule.x_max, rule.y],
    }


def _attach_scripts(
    glyphs: list[Glyph],
    base_size: float,
    reserved_owners: dict[int, int] | None = None,
) -> tuple[dict[int, list[Glyph]], dict[int, list[Glyph]], set[int]]:
    """Attach compact script runs, including multi-glyph forms like n+1."""
    reserved_owners = reserved_owners or {}
    superscripts: dict[int, list[Glyph]] = defaultdict(list)
    subscripts: dict[int, list[Glyph]] = defaultdict(list)
    consumed: set[int] = set()
    candidates = [
        glyph
        for glyph in glyphs
        if glyph.figure_id is None
        if glyph.size <= base_size * 0.78
        and glyph.text.strip()
        and SCRIPT_TOKEN.fullmatch(glyph.text.strip())
    ]
    base_glyphs = list(glyphs)

    script_clusters: list[list[Glyph]] = []
    current: list[Glyph] = []
    for glyph in sorted(candidates, key=lambda item: (item.baseline, item.x0)):
        if current:
            previous = current[-1]
            baseline_tolerance = max(
                1.0,
                min(previous.size, glyph.size) * 0.12,
            )
            size_ratio = max(previous.size, glyph.size) / max(
                min(previous.size, glyph.size), 0.1
            )
            horizontal_gap = glyph.x0 - previous.x1
            if (
                abs(glyph.baseline - previous.baseline) <= baseline_tolerance
                and size_ratio <= 1.25
                and horizontal_gap <= max(1.5, base_size * 0.45)
            ):
                current.append(glyph)
                continue
            script_clusters.append(current)
        current = [glyph]
    if current:
        script_clusters.append(current)

    for cluster in script_clusters:
        cluster_baseline = statistics.median(
            glyph.baseline for glyph in cluster
        )
        cluster_size = median_size(cluster)
        cluster_x0 = min(glyph.x0 for glyph in cluster)
        cluster_owners = {
            reserved_owners[glyph.index]
            for glyph in cluster
            if glyph.index in reserved_owners
        }
        cluster_is_reserved = bool(cluster_owners) and all(
            glyph.index in reserved_owners for glyph in cluster
        )
        possibilities: list[tuple[float, Glyph, str]] = []
        for base in base_glyphs:
            if base.index in {glyph.index for glyph in cluster}:
                continue
            if (
                base.figure_id is not None
                or base.figure_id != cluster[0].figure_id
            ):
                continue
            if base.index in reserved_owners:
                if not cluster_is_reserved:
                    continue
                if reserved_owners[base.index] not in cluster_owners:
                    continue
            elif cluster_is_reserved:
                continue
            if base.size < cluster_size * 1.18:
                continue
            baseline_delta = cluster_baseline - base.baseline
            if abs(baseline_delta) < max(0.6, base.size * 0.08):
                continue
            if abs(baseline_delta) > base.size * 0.92:
                continue
            gap = cluster_x0 - base.x1
            if gap < -base.size * 0.18 or gap > base.size * 0.52:
                continue
            kind = "sub" if baseline_delta > 0 else "super"
            score = (
                abs(baseline_delta) / base.size
                + abs(gap) / base.size
                + abs(cluster_size - base.size * 0.7) / base.size * 0.12
            )
            possibilities.append((score, base, kind))
        if not possibilities:
            continue
        _, base, kind = min(
            possibilities,
            key=lambda item: (item[0], item[1].index),
        )
        target = subscripts if kind == "sub" else superscripts
        target[base.index].extend(cluster)
        consumed.update(glyph.index for glyph in cluster)

    for relation in (superscripts, subscripts):
        for base_index in relation:
            relation[base_index].sort(key=lambda item: (item.x0, item.index))
    return superscripts, subscripts, consumed


def _operator_math_anchor(operator: Glyph) -> float:
    """Estimate a large operator's mathematical baseline from its outline."""
    if operator.font.upper().startswith("CMEX"):
        if operator.ink_bbox is not None:
            return (
                operator.ink_bbox[1] + operator.ink_bbox[3]
            ) / 2 + operator.size * 0.25
        return operator.baseline + operator.size * 1.35
    return operator.baseline


def _operator_limits(
    glyphs: list[Glyph],
    base_size: float,
) -> tuple[dict[int, dict[str, list[Glyph]]], set[int], list[dict[str, Any]]]:
    """Detect geometric limits around large integral and summation glyphs."""
    limits: dict[int, dict[str, list[Glyph]]] = {}
    consumed: set[int] = set()
    ambiguities: list[dict[str, Any]] = []
    for operator in glyphs:
        if operator.figure_id is not None:
            continue
        operator_height = max(operator.y1 - operator.y0, operator.size)
        tex_name = (operator.tex_token or "").lstrip("\\")
        operator_name = OPERATOR_NAMES.get(operator.text.strip())
        is_operator = (
            operator.text.strip() in MATH_OPERATORS
            or tex_name in set(OPERATOR_NAMES.values())
            or tex_name == operator_name
        )
        is_cmex_operator = (
            operator.font.upper().startswith("CMEX") and is_operator
        )
        geometry_indicates_display = (
            operator.size >= base_size * 1.18
            or operator_height >= base_size * 1.18
        )
        if (
            not is_operator
            or not (is_cmex_operator or geometry_indicates_display)
        ):
            continue
        if is_cmex_operator:
            horizontal_radius = base_size * (
                1.2 if operator.text.strip() == "∫" else 0.72
            )
            operator_anchor = _operator_math_anchor(operator)
            minimum_limit_offset = base_size * 0.35
            maximum_limit_offset = base_size * 2.4
        else:
            operator_anchor = operator.baseline
            horizontal_radius = max(4.0, operator.size * 0.68)
            minimum_limit_offset = max(1.3, operator_height * 0.2)
            maximum_limit_offset = operator_height * 1.55
        candidates = [
            glyph
            for glyph in glyphs
            if glyph.index != operator.index
            and glyph.figure_id is None
            and glyph.size <= max(base_size * 0.8, operator.size * 0.72)
            and abs(glyph.center_x - operator.center_x)
            <= horizontal_radius
            and abs(glyph.baseline - operator_anchor) >= minimum_limit_offset
            and abs(glyph.baseline - operator_anchor) <= maximum_limit_offset
        ]
        upper = [
            glyph
            for glyph in candidates
            if glyph.baseline < operator_anchor
        ]
        lower = [
            glyph
            for glyph in candidates
            if glyph.baseline > operator_anchor
        ]

        def nearest(group: list[Glyph]) -> list[Glyph]:
            if not group:
                return []
            closest = min(
                abs(glyph.baseline - operator_anchor) for glyph in group
            )
            selected = [
                glyph
                for glyph in group
                if abs(glyph.baseline - operator_anchor)
                <= closest + base_size * 0.55
            ]
            return sorted(selected, key=lambda glyph: (glyph.x0, glyph.index))

        upper = nearest(upper)
        lower = nearest(lower)
        if upper or lower:
            limits[operator.index] = {"upper": upper, "lower": lower}
            consumed.update(glyph.index for glyph in upper + lower)
        else:
            ambiguities.append(
                {
                    "type": "large_operator_without_resolved_limits",
                    "glyph_index": operator.index,
                    "operator": operator.text,
                    "bbox": list(operator.bbox),
                }
            )
    return limits, consumed, ambiguities


def _text_operator_limit_ambiguities(
    rows: list[Row], glyphs: list[Glyph], base_size: float
) -> list[dict[str, Any]]:
    """Flag word-form limits when bounds sit on a separate baseline."""
    ambiguities: list[dict[str, Any]] = []
    for row in rows:
        ordered = sorted(row.glyphs, key=lambda glyph: (glyph.x0, glyph.index))
        for start in range(max(0, len(ordered) - 2)):
            operator_glyphs = ordered[start:start + 3]
            if (
                "".join(glyph.text for glyph in operator_glyphs).lower()
                != "lim"
            ):
                continue
            if max(glyph.size for glyph in operator_glyphs) < base_size * 0.72:
                continue
            operator_x0 = min(glyph.x0 for glyph in operator_glyphs)
            operator_x1 = max(glyph.x1 for glyph in operator_glyphs)
            operator_baseline = statistics.median(
                glyph.baseline for glyph in operator_glyphs
            )
            operator_ids = {glyph.index for glyph in operator_glyphs}
            candidates = [
                glyph
                for glyph in glyphs
                if glyph.figure_id is None
                and glyph.index not in operator_ids
                and glyph.size <= base_size * 0.8
                and base_size * 0.35
                <= abs(glyph.baseline - operator_baseline)
                <= base_size * 1.35
                and glyph.x1 >= operator_x0 - base_size * 0.2
                and glyph.x0 <= operator_x1 + base_size * 0.35
            ]
            has_limit_syntax = any(
                glyph.text.strip() in {"→", "∞", "−", "≤", "<", ">"}
                or any(char.isdigit() for char in glyph.text)
                for glyph in candidates
            )
            if not candidates or not has_limit_syntax:
                continue
            members = operator_glyphs + candidates
            ambiguities.append(
                {
                    "type": "unresolved_text_operator_limits",
                    "operator": "lim",
                    "glyph_indices": sorted(glyph.index for glyph in members),
                    "bbox": [
                        min(glyph.x0 for glyph in members),
                        min(glyph.y0 for glyph in members),
                        max(glyph.x1 for glyph in members),
                        max(glyph.y1 for glyph in members),
                    ],
                }
            )
    return ambiguities


def _align_display_operators(
    rows: list[Row],
    glyphs: list[Glyph],
    limits: dict[int, dict[str, list[Glyph]]],
    base_size: float,
) -> None:
    """Place display operators on the body row indicated by their outline."""
    glyph_by_index = {glyph.index: glyph for glyph in glyphs}
    row_by_glyph = {
        glyph.index: row for row in rows for glyph in row.glyphs
    }
    for operator_index in limits:
        operator = glyph_by_index.get(operator_index)
        source = row_by_glyph.get(operator_index)
        if operator is None or source is None:
            continue
        anchor = _operator_math_anchor(operator)
        candidates = [
            row
            for row in rows
            if row is not source
            and row.glyphs
            and row.median_size >= base_size * 0.78
            and abs(row.baseline - anchor) <= base_size * 0.55
        ]
        if not candidates:
            continue
        target = min(
            candidates,
            key=lambda row: (
                abs(row.baseline - anchor),
                -row.median_size,
                row.x0,
            ),
        )
        source.glyphs.remove(operator)
        target.glyphs.append(operator)
        target.glyphs.sort(key=lambda glyph: (glyph.x0, glyph.index))
        row_by_glyph[operator_index] = target


def _compose_vertical_delimiters(
    rows: list[Row], glyphs: list[Glyph], base_size: float
) -> tuple[list[dict[str, Any]], set[int], list[dict[str, Any]]]:
    """Collapse repeated CMEX extender pieces into paired vertical bars."""
    candidates = [
        glyph
        for glyph in glyphs
        if glyph.figure_id is None and glyph.glyph_name == "vextendsingle"
    ]
    x_groups: list[list[Glyph]] = []
    for glyph in sorted(
        candidates, key=lambda item: (item.center_x, item.baseline)
    ):
        group = next(
            (
                candidate
                for candidate in x_groups
                if (
                    abs(
                        statistics.median(g.center_x for g in candidate)
                        - glyph.center_x
                    )
                    <= max(0.8, glyph.size * 0.08)
                )
            ),
            None,
        )
        if group is None:
            x_groups.append([glyph])
        else:
            group.append(glyph)

    runs: list[list[Glyph]] = []
    for group in x_groups:
        pieces: list[Glyph] = []
        for glyph in sorted(group, key=lambda item: item.baseline):
            if (
                pieces
                and glyph.baseline - pieces[-1].baseline > base_size * 0.8
            ):
                if len(pieces) >= 3:
                    runs.append(pieces)
                pieces = []
            pieces.append(glyph)
        if len(pieces) >= 3:
            runs.append(pieces)

    row_by_glyph = {
        glyph.index: row for row in rows for glyph in row.glyphs
    }
    run_records: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    for run in runs:
        y_min = min(glyph.baseline for glyph in run)
        y_max = max(glyph.baseline for glyph in run)
        center_y = (y_min + y_max) / 2
        center_x = statistics.median(glyph.center_x for glyph in run)
        hosts = [
            row
            for row in rows
            if row.figure_id is None
            and row.glyphs
            and row.median_size >= base_size * 0.78
            and y_min - base_size * 0.35
            <= row.baseline
            <= y_max + base_size * 0.35
            and row.x1 >= center_x - base_size * 0.2
            and row.x0 <= center_x + base_size * 0.2
        ]
        host = min(
            hosts,
            key=lambda row: (abs(row.baseline - center_y), row.index),
            default=None,
        )
        if host is None:
            continue
        run_records.append(
            {
                "host_row": host.index,
                "x0": min(glyph.x0 for glyph in run),
                "x1": max(glyph.x1 for glyph in run),
                "glyph_indices": sorted(glyph.index for glyph in run),
                "baseline_center": center_y,
            }
        )

    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in run_records:
        by_row[run["host_row"]].append(run)

    tokens: list[dict[str, Any]] = []
    consumed: set[int] = set()
    glyph_by_index = {glyph.index: glyph for glyph in glyphs}
    for row_index, row_runs in by_row.items():
        ordered = sorted(row_runs, key=lambda item: (item["x0"], item["x1"]))
        if len(ordered) % 2:
            ambiguities.append(
                {
                    "type": "unpaired_stretched_delimiter",
                    "row_index": row_index,
                    "glyph_indices": ordered[-1]["glyph_indices"],
                    "bbox": [
                        ordered[-1]["x0"],
                        min(
                            glyph_by_index[index].y0
                            for index in ordered[-1]["glyph_indices"]
                        ),
                        ordered[-1]["x1"],
                        max(
                            glyph_by_index[index].y1
                            for index in ordered[-1]["glyph_indices"]
                        ),
                    ],
                }
            )
            ordered = ordered[:-1]
        for index in range(0, len(ordered), 2):
            pair = ordered[index:index + 2]
            for side, run in zip(("open", "close"), pair):
                run_glyph_ids = run["glyph_indices"]
                run_glyphs = [glyph_by_index[item] for item in run_glyph_ids]
                token = {
                    **run,
                    "text": "\\left|" if side == "open" else "\\right|",
                    "side": side,
                    "bbox": [
                        run["x0"],
                        min(glyph.y0 for glyph in run_glyphs),
                        run["x1"],
                        max(glyph.y1 for glyph in run_glyphs),
                    ],
                }
                tokens.append(token)
                consumed.update(run_glyph_ids)

    if consumed:
        row_lookup = {row.index: row for row in rows}
        for row in rows:
            row.glyphs = [
                glyph for glyph in row.glyphs if glyph.index not in consumed
            ]
        for token in tokens:
            host = row_lookup[token["host_row"]]
            host.glyphs.extend(
                glyph_by_index[index] for index in token["glyph_indices"]
            )
            host.glyphs.sort(key=lambda glyph: (glyph.x0, glyph.index))
    return tokens, consumed, ambiguities


def _render_glyph_sequence(
    glyphs: Iterable[Glyph],
    superscripts: dict[int, list[Glyph]],
    subscripts: dict[int, list[Glyph]],
    limits: dict[int, dict[str, list[Glyph]]] | None = None,
    overbars: list[dict[str, Any]] | None = None,
    fraction_tokens: list[dict[str, Any]] | None = None,
    extra_atoms: list[dict[str, Any]] | None = None,
) -> str:
    """Render an ordered glyph sequence, including attached indices."""
    limits = limits or {}
    overbars = overbars or []
    fraction_tokens = fraction_tokens or []
    extra_atoms = extra_atoms or []
    ordered = sorted(glyphs, key=lambda item: (item.x0, item.index))
    available_ids = {glyph.index for glyph in ordered}
    consumed = {
        glyph.index
        for relation in (superscripts, subscripts)
        for base_index, attached in relation.items()
        if base_index in available_ids
        for glyph in attached
    }
    for glyph in ordered:
        if glyph.index in limits:
            consumed.update(
                item.index
                for group in limits[glyph.index].values()
                for item in group
            )
    fraction_atoms: list[tuple[float, float, str, set[int]]] = []
    for fraction in fraction_tokens:
        if fraction.get("script_parent") is not None:
            continue
        member_ids = set(fraction.get("member_glyph_ids", []))
        if member_ids and member_ids.issubset(available_ids):
            fraction_atoms.append(
                (
                    float(fraction["x0"]),
                    float(fraction["x1"]),
                    str(fraction["text"]),
                    member_ids,
                )
            )
            consumed.update(member_ids)
    overbar_atoms: list[tuple[float, float, str, set[int]]] = []
    for overbar in overbars:
        target_ids = set(overbar.get("glyph_indices", []))
        overlaps_fraction = any(
            target_ids.intersection(member_ids)
            for _, _, _, member_ids in fraction_atoms
        )
        if (
            target_ids
            and target_ids.issubset(available_ids)
            and not overlaps_fraction
        ):
            target_glyphs = [
                glyph for glyph in ordered if glyph.index in target_ids
            ]
            inner = _render_glyph_sequence(
                target_glyphs,
                superscripts,
                subscripts,
                limits,
                overbars=[],
                fraction_tokens=fraction_tokens,
            )
            if inner:
                overbar_atoms.append(
                    (
                        min(glyph.x0 for glyph in target_glyphs),
                        max(glyph.x1 for glyph in target_glyphs),
                        "\\overline{" + inner + "}",
                        target_ids,
                    )
                )
                consumed.update(target_ids)
    atoms: list[tuple[float, float, str]] = [
        (x0, x1, token)
        for x0, x1, token, _ in overbar_atoms
    ]
    atoms.extend(
        (x0, x1, token)
        for x0, x1, token, _ in fraction_atoms
    )
    atoms.extend(
        (
            float(extra["x0"]),
            float(extra["x1"]),
            str(extra["text"]),
        )
        for extra in extra_atoms
    )
    for glyph in ordered:
        if glyph.index in consumed:
            continue
        atoms.append(
            (
                glyph.x0,
                glyph.x1,
                _tex_atom_text(
                    glyph,
                    superscripts,
                    subscripts,
                    limits,
                    overbars,
                    fraction_tokens,
                ),
            )
        )
    atoms.sort(key=lambda atom: (atom[0], atom[1]))
    pieces = []
    previous_x1: float | None = None
    for x0, x1, token in atoms:
        if previous_x1 is not None and x0 - previous_x1 > 2.5:
            pieces.append(" ")
        pieces.append(token)
        previous_x1 = x1
    return "".join(pieces).strip()


def _tex_atom_text(
    glyph: Glyph,
    superscripts: dict[int, list[Glyph]],
    subscripts: dict[int, list[Glyph]],
    limits: dict[int, dict[str, list[Glyph]]],
    overbars: list[dict[str, Any]] | None = None,
    fraction_tokens: list[dict[str, Any]] | None = None,
) -> str:
    """Render a glyph with only geometrically supported script relations."""
    fraction_tokens = fraction_tokens or []
    if glyph.structural_kind == "delimiter_part":
        part = glyph.delimiter_part or glyph.glyph_name or "unknown"
        base = glyph.delimiter_base or glyph.text
        token = f"\\text{{[unresolved delimiter part: {base}/{part}]}}"
    elif glyph.structural_kind == "delimiter" and glyph.tex_token:
        token = glyph.tex_token
    else:
        token = glyph.text
    if glyph.index in limits:
        relation = limits[glyph.index]
        operator_name = OPERATOR_NAMES.get(glyph.text.strip())
        if operator_name is None:
            operator_name = OPERATOR_TOKEN_NAMES.get(
                (glyph.tex_token or "").lstrip("\\")
            )
        if operator_name is None:
            operator_name = "operatorname"
            token = "\\operatorname{" + glyph.text.strip() + "}"
        else:
            token = "\\" + operator_name + "\\limits"
        if relation["lower"]:
            lower = _render_glyph_sequence(
                relation["lower"],
                superscripts,
                subscripts,
                overbars=overbars,
                fraction_tokens=fraction_tokens,
            )
            token += "_{" + lower.replace(" ", "") + "}"
        if relation["upper"]:
            upper = _render_glyph_sequence(
                relation["upper"],
                superscripts,
                subscripts,
                overbars=overbars,
                fraction_tokens=fraction_tokens,
            )
            token += "^{" + upper.replace(" ", "") + "}"
        return token

    if glyph.index in subscripts:
        extra_atoms = [
            fraction
            for fraction in fraction_tokens
            if fraction.get("script_parent") == glyph.index
            and fraction.get("script_direction") == "sub"
        ]
        suffix = _render_glyph_sequence(
            subscripts[glyph.index],
            superscripts,
            subscripts,
            limits,
            overbars,
            fraction_tokens,
            extra_atoms,
        )
        token += "_{" + suffix + "}"
    if glyph.index in superscripts:
        extra_atoms = [
            fraction
            for fraction in fraction_tokens
            if fraction.get("script_parent") == glyph.index
            and fraction.get("script_direction") == "super"
        ]
        suffix = _render_glyph_sequence(
            superscripts[glyph.index],
            superscripts,
            subscripts,
            limits,
            overbars,
            fraction_tokens,
            extra_atoms,
        )
        token += "^{" + suffix + "}"
    return token


def _render_row(
    row: Row,
    *,
    superscripts: dict[int, list[Glyph]],
    subscripts: dict[int, list[Glyph]],
    limits: dict[int, dict[str, list[Glyph]]],
    consumed_scripts: set[int],
    consumed_limits: set[int],
    fraction_tokens: list[dict[str, Any]],
    consumed_fractions: set[int],
    overbar_tokens: list[dict[str, Any]],
    consumed_overbars: set[int],
    stretch_delimiters: list[dict[str, Any]],
    consumed_stretch_delimiters: set[int],
) -> str:
    """Render a baseline row, inserting fraction and script notation."""
    if row.figure_id is not None:
        return _text_for_glyphs(row.glyphs)
    atoms: list[tuple[float, float, str, Glyph | None]] = []
    for glyph in row.glyphs:
        if (
            glyph.index in consumed_scripts
            or glyph.index in consumed_limits
            or glyph.index in consumed_fractions
            or glyph.index in consumed_overbars
            or glyph.index in consumed_stretch_delimiters
        ):
            continue
        atoms.append(
            (
                glyph.x0,
                glyph.x1,
                _tex_atom_text(
                    glyph,
                    superscripts,
                    subscripts,
                    limits,
                    overbar_tokens,
                    fraction_tokens,
                ),
                glyph,
            )
        )
    for fraction in fraction_tokens:
        if fraction.get("host_row") == row.index:
            atoms.append(
                (fraction["x0"], fraction["x1"], fraction["text"], None)
            )
    for overbar in overbar_tokens:
        if (
            overbar.get("host_row") == row.index
            and not overbar.get("composed")
        ):
            atoms.append((overbar["x0"], overbar["x1"], overbar["text"], None))
    for delimiter in stretch_delimiters:
        if delimiter.get("host_row") == row.index:
            atoms.append(
                (
                    delimiter["x0"],
                    delimiter["x1"],
                    delimiter["text"],
                    None,
                )
            )
    atoms.sort(key=lambda atom: (atom[0], atom[1]))

    pieces: list[str] = []
    previous_x1: float | None = None
    previous_glyph: Glyph | None = None
    for x0, x1, text, glyph in atoms:
        if (
            previous_x1 is not None
            and glyph is not None
            and previous_glyph is not None
        ):
            gap = x0 - previous_x1
            threshold = _glyph_space_threshold(previous_glyph, glyph)
            if gap > threshold and pieces and pieces[-1] != " ":
                pieces.append(" ")
        elif previous_x1 is not None and x0 - previous_x1 > 3.0:
            if pieces and pieces[-1] != " ":
                pieces.append(" ")
        pieces.append(text)
        previous_x1 = x1
        previous_glyph = glyph
    return re.sub(r"[\t\f\v ]+", " ", "".join(pieces)).strip()


def analyze_page(page_record: dict[str, Any]) -> LayoutResult:
    """Reconstruct lines, conservative fractions/scripts, and ambiguities."""
    glyphs = parse_glyphs(page_record)
    rows = group_rows(glyphs)
    rules = parse_rules(page_record)
    base_size = median_size(glyphs)

    operator_limits, consumed_limits, op_ambiguities = _operator_limits(
        glyphs, base_size
    )
    reserved_owners = {
        glyph.index: operator_index
        for operator_index, relation in operator_limits.items()
        for glyphs_in_limit in relation.values()
        for glyph in glyphs_in_limit
    }
    _align_display_operators(rows, glyphs, operator_limits, base_size)
    stretch_delimiters, consumed_stretch_delimiters, delimiter_ambiguities = (
        _compose_vertical_delimiters(rows, glyphs, base_size)
    )

    horizontal_rules = [
        rule
        for rule in rules
        if rule.length >= max(3.0, base_size * 0.18)
        and rule.figure_id is None
    ]
    associations = [
        _association_for_rule(
            rule,
            horizontal_rules,
            glyphs,
            base_size,
            excluded_ids=consumed_limits | consumed_stretch_delimiters,
        )
        for rule in horizontal_rules
    ]
    fraction_associations = [
        item for item in associations if item["kind"] == "fraction"
    ]
    overbars = [item for item in associations if item["kind"] == "overbar"]
    unresolved = [
        item for item in associations if item["kind"] == "unresolved"
    ]

    superscripts, subscripts, consumed_scripts = _attach_scripts(
        glyphs, base_size, reserved_owners=reserved_owners
    )

    overbar_tokens: list[dict[str, Any]] = []
    for association in overbars:
        barred_glyphs = association["barred_glyphs"]
        if association["geometry_score"] < 0.82:
            unresolved.append(
                {
                    "kind": "unresolved",
                    "rule_index": association["rule_index"],
                    "reason": (
                        "overbar candidate does not meet the conservative "
                        "geometry-score threshold"
                    ),
                    "bbox": association["bbox"],
                }
            )
            continue
        host = min(
            rows,
            key=lambda row: abs(
                row.baseline
                - statistics.median(glyph.baseline for glyph in barred_glyphs)
            ),
            default=None,
        )
        if host is None:
            continue
        overbar_tokens.append(
            {
                "rule_index": association["rule_index"],
                "x0": association["bbox"][0],
                "x1": association["bbox"][2],
                "host_row": host.index,
                "glyph_indices": sorted(
                    glyph.index for glyph in barred_glyphs
                ),
                "geometry_score": association["geometry_score"],
                "bbox": association["bbox"],
                "composed": False,
            }
        )

    consumed_fractions: set[int] = set()
    fraction_tokens: list[dict[str, Any]] = []
    for association in fraction_associations:
        numerator = association["numerator_glyphs"]
        denominator = association["denominator_glyphs"]
        numerator_ids = {glyph.index for glyph in numerator}
        denominator_ids = {glyph.index for glyph in denominator}
        glyph_ids = numerator_ids | denominator_ids
        composed_overbars: list[dict[str, Any]] = []
        invalid_overbar_overlap = False
        for overbar in overbar_tokens:
            target_ids = set(overbar["glyph_indices"])
            if not target_ids.intersection(glyph_ids):
                continue
            if (
                target_ids.issubset(numerator_ids)
                or target_ids.issubset(denominator_ids)
            ):
                composed_overbars.append(overbar)
            else:
                invalid_overbar_overlap = True
                unresolved.append(
                    {
                        "kind": "unresolved",
                        "rule_index": association["rule_index"],
                        "reason": (
                            "fraction candidate cuts through an overbar "
                            "target; "
                            "keep both relations in geometry"
                        ),
                        "bbox": association["bbox"],
                    }
                )
                break
        if invalid_overbar_overlap:
            continue
        numerator_text = _render_glyph_sequence(
            numerator,
            superscripts,
            subscripts,
            operator_limits,
            overbars=composed_overbars,
        )
        denominator_text = _render_glyph_sequence(
            denominator,
            superscripts,
            subscripts,
            operator_limits,
            overbars=composed_overbars,
        )
        if not numerator_text or not denominator_text:
            unresolved.append(
                {
                    "kind": "unresolved",
                    "rule_index": association["rule_index"],
                    "reason": "fraction candidate has empty text on one side",
                    "bbox": association["bbox"],
                }
            )
            continue
        if consumed_fractions & glyph_ids:
            unresolved.append(
                {
                    "kind": "unresolved",
                    "rule_index": association["rule_index"],
                    "reason": (
                        "fraction candidates share numerator or denominator "
                        "glyphs"
                    ),
                    "bbox": association["bbox"],
                }
            )
            continue
        consumed_fractions.update(glyph_ids)
        for overbar in composed_overbars:
            overbar["composed"] = True
        lo = min(
            min(g.baseline for g in numerator),
            min(g.baseline for g in denominator),
        )
        hi = max(
            max(g.baseline for g in numerator),
            max(g.baseline for g in denominator),
        )
        rule_y = (association["bbox"][1] + association["bbox"][3]) / 2
        possible_hosts = [
            row for row in rows if lo - 0.5 <= row.baseline <= hi + 0.5
        ]
        body_hosts = [
            row
            for row in possible_hosts
            if row.median_size >= base_size * 0.78
        ]
        host_candidates = body_hosts or possible_hosts
        host = min(
            host_candidates,
            key=lambda row: (
                abs(row.baseline - (rule_y + base_size * 0.5)),
                -row.median_size,
            ),
            default=None,
        )
        if host is None:
            host = Row(index=len(rows), baseline=rule_y + base_size * 0.5)
            rows.append(host)
        fraction_tokens.append(
            {
                "rule_index": association["rule_index"],
                "text": (
                    "\\frac{"
                    + numerator_text.replace(" ", "")
                    + "}{"
                    + denominator_text.replace(" ", "")
                    + "}"
                ),
                "x0": association["bbox"][0],
                "x1": association["bbox"][2],
                "host_row": host.index,
                "baseline": host.baseline,
                "script_size": median_size(numerator + denominator),
                "numerator_glyphs": sorted(
                    glyph_ids & {glyph.index for glyph in numerator}
                ),
                "denominator_glyphs": sorted(
                    glyph_ids & {glyph.index for glyph in denominator}
                ),
                "member_glyph_ids": sorted(glyph_ids),
                "overbar_rule_indices": sorted(
                    token["rule_index"] for token in composed_overbars
                ),
                "geometry_score": association["geometry_score"],
                "bbox": association["bbox"],
            }
        )

    accepted_overbar_ids = {token["rule_index"] for token in overbar_tokens}
    overbar_glyphs = {glyph.index: glyph for glyph in glyphs}
    for token in overbar_tokens:
        target_glyphs = [
            overbar_glyphs[index]
            for index in token["glyph_indices"]
            if index in overbar_glyphs
        ]
        inner = _render_glyph_sequence(
            target_glyphs,
            superscripts,
            subscripts,
            operator_limits,
        ).replace(" ", "")
        token["text"] = "\\overline{" + inner + "}"
    ambiguities: list[dict[str, Any]] = list(op_ambiguities)
    ambiguities.extend(delimiter_ambiguities)
    ambiguities.extend(
        _text_operator_limit_ambiguities(rows, glyphs, base_size)
    )
    script_owners = {
        child.index: parent_index
        for relation in (superscripts, subscripts)
        for parent_index, children in relation.items()
        for child in children
    }
    for fraction in fraction_tokens:
        member_ids = set(fraction["member_glyph_ids"])
        cross_relations = [
            {
                "glyph_index": glyph_index,
                "script_parent": script_owners[glyph_index],
            }
            for glyph_index in sorted(member_ids & set(script_owners))
            if script_owners[glyph_index] not in member_ids
        ]
        if cross_relations:
            ambiguities.append(
                {
                    "type": "fraction_intersects_external_script",
                    "rule_index": fraction["rule_index"],
                    "relations": cross_relations,
                    "bbox": fraction["bbox"],
                }
            )
    for first_index, first in enumerate(fraction_tokens):
        first_members = set(first["member_glyph_ids"])
        for second in fraction_tokens[first_index + 1:]:
            second_members = set(second["member_glyph_ids"])
            if first_members.intersection(second_members):
                ambiguities.append(
                    {
                        "type": "overlapping_fraction_regions",
                        "rule_indices": [
                            first["rule_index"],
                            second["rule_index"],
                        ],
                        "bbox": [
                            min(first["bbox"][0], second["bbox"][0]),
                            min(first["bbox"][1], second["bbox"][1]),
                            max(first["bbox"][2], second["bbox"][2]),
                            max(first["bbox"][3], second["bbox"][3]),
                        ],
                    }
                )
    ambiguities.extend(
        {
            "type": "unresolved_delimiter_part",
            "glyph_index": glyph.index,
            "glyph_name": glyph.glyph_name,
            "delimiter_base": glyph.delimiter_base,
            "delimiter_part": glyph.delimiter_part,
            "bbox": list(glyph.bbox),
        }
        for glyph in glyphs
        if glyph.structural_kind == "delimiter_part"
    )
    ambiguities.extend(
        {
            "type": "unresolved_rule",
            **item,
        }
        for item in unresolved
    )
    ambiguities.extend(
        {
            "type": "overbar_not_substituted",
            "rule_index": association["rule_index"],
            "reason": "overbar candidate retained only in source geometry",
            "bbox": association["bbox"],
            "geometry_score": association["geometry_score"],
        }
        for association in overbars
        if association["rule_index"] not in accepted_overbar_ids
    )

    return LayoutResult(
        rows=sorted(rows, key=lambda row: (row.baseline, row.x0)),
        fraction_rules=fraction_tokens,
        fraction_tokens=fraction_tokens,
        overbar_rules=overbars,
        overbar_tokens=overbar_tokens,
        stretch_delimiters=stretch_delimiters,
        unresolved_rules=unresolved,
        superscripts=superscripts,
        subscripts=subscripts,
        big_operator_limits=operator_limits,
        ambiguities=ambiguities,
        glyphs=glyphs,
        figure_rule_count=sum(
            1 for rule in rules if rule.figure_id is not None
        ),
    )


def render_rows(layout: LayoutResult) -> list[dict[str, Any]]:
    """Render ordered line records while keeping glyph-to-line provenance."""
    consumed_scripts = {
        glyph.index
        for relation in (layout.superscripts, layout.subscripts)
        for glyphs in relation.values()
        for glyph in glyphs
    }
    consumed_limits = {
        glyph.index
        for relation in layout.big_operator_limits.values()
        for glyphs in relation.values()
        for glyph in glyphs
    }
    consumed_fractions = {
        glyph_index
        for fraction in layout.fraction_rules
        for key in ("numerator_glyphs", "denominator_glyphs")
        for glyph_index in fraction.get(key, [])
    }
    consumed_overbars = {
        glyph_index
        for overbar in layout.overbar_tokens
        for glyph_index in overbar["glyph_indices"]
    }
    consumed_stretch_delimiters = {
        glyph_index
        for delimiter in layout.stretch_delimiters
        for glyph_index in delimiter["glyph_indices"]
    }
    fraction_tokens = layout.fraction_tokens
    overbar_tokens = layout.overbar_tokens

    rendered: list[dict[str, Any]] = []
    for row in layout.rows:
        text = _render_row(
            row,
            superscripts=layout.superscripts,
            subscripts=layout.subscripts,
            limits=layout.big_operator_limits,
            consumed_scripts=consumed_scripts,
            consumed_limits=consumed_limits,
            fraction_tokens=fraction_tokens,
            consumed_fractions=consumed_fractions,
            overbar_tokens=overbar_tokens,
            consumed_overbars=consumed_overbars,
            stretch_delimiters=layout.stretch_delimiters,
            consumed_stretch_delimiters=consumed_stretch_delimiters,
        )
        if not text:
            continue
        row_glyphs = [glyph.index for glyph in row.glyphs]
        row_fraction_ids = [
            fraction["rule_index"]
            for fraction in fraction_tokens
            if fraction["host_row"] == row.index
        ]
        row_overbar_ids = [
            overbar["rule_index"]
            for overbar in overbar_tokens
            if overbar["host_row"] == row.index
        ]
        rendered.append(
            {
                "line_index": len(rendered),
                "baseline": round(row.baseline, 3),
                "bbox": [
                    round(row.x0, 3),
                    round(row.y0, 3),
                    round(row.x1, 3),
                    round(row.y1, 3),
                ],
                "text": text,
                "figure_id": row.figure_id,
                "glyph_indices": row_glyphs,
                "fraction_rule_indices": row_fraction_ids,
                "overbar_rule_indices": row_overbar_ids,
                "stretch_delimiter_glyph_indices": [
                    glyph_index
                    for delimiter in layout.stretch_delimiters
                    if delimiter["host_row"] == row.index
                    for glyph_index in delimiter["glyph_indices"]
                ],
            }
        )
    return rendered


def layout_page(
    page_record: dict[str, Any],
) -> tuple[LayoutResult, list[dict[str, Any]]]:
    """Convenience wrapper used by the corpus builder."""
    result = analyze_page(page_record)
    return result, render_rows(result)
