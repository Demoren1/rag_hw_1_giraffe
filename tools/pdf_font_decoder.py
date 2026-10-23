#!/usr/bin/env python3
"""Extract font-aware glyphs and drawing rules from a PDF.

The document uses TeX Type 1 math fonts with ToUnicode CMaps and Type 3
T2A Cyrillic fonts whose glyph names are opaque ``aNN`` names.  This module
uses embedded ToUnicode CMaps, glyph names from embedded Type 1 encodings,
and (when explicitly requested) the published CM-Super T2A vector for the
source PDF's Type 3 Cyrillic fonts. CMEX symbols and delimiter pieces are
decoded from the font's own Type 1 Encoding rather than hard-coded code slots.

Encoding references:

* CM-Super T2A encoding vector, generated from T2Auni.map and Adobe's glyph
  list:
  https://mirrors.ibiblio.org/CTAN/fonts/ps-type1/cm-super/dvips/cm-super-t2a.enc
* T2 package map descriptions (T2Auni.map maps T2A slots to Unicode):
  https://ctan.org/tex-archive/macros/latex/contrib/t2/enc-maps
* TeX font encoding guide (OML, OMS, OMX and Cyrillic encodings):
  https://tug.ctan.org/macros/latex/base/encguide.pdf
* CMEX Type 1 encoding table, used to cross-check embedded glyph semantics:
  https://www-archive.mozilla.org/projects/mathml/fonts/encoding/
  cmex-t1-encoding
* CMEX overview table:
  https://www-archive.mozilla.org/projects/mathml/fonts/encoding/cmex

The T2A vector is embedded below so extraction does not depend on a network
connection or a separately downloaded font package. Type 3 ``aNN`` names
alone are not treated as evidence of a Unicode mapping. The T2A profile is
opt-in because these Type 3 fonts lack ToUnicode and their metadata does not
prove their encoding; observed Cyrillic-slot coverage is only a
document-specific classification signal. Other Type 3 fonts remain
unresolved unless their glyph can be decoded by an explicit PDF encoding/CMap.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Iterator
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
import pdfminer
from fontTools import agl
from fontTools.pens.boundsPen import BoundsPen
from fontTools.t1Lib import T1Font
from pdfminer.converter import PDFPageAggregator
from pdfminer.cmapdb import CMapParser, FileUnicodeMap
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.utils import apply_matrix_rect
from pypdf import PdfReader


T2A_GLYPH_NAMES = """
grave acute circumflex tilde dieresis hungarumlaut ring caron breve macron
dotaccent cedilla ogonek uni04C0 angleleft angleright quotedblleft
quotedblright cyrflex dblgrave cyrbreve endash emdash.cyr afii61664
perthousandzero dotlessi dotlessj ff fi fl ffi ffl uni2423 exclam quotedbl
numbersign dollar percent ampersand quoteright parenleft parenright asterisk
plus comma hyphen period slash zero one two three four five six seven eight
nine colon semicolon less equal greater question at A B C D E F G H I J K L M
N O P Q R S T U V W X Y Z bracketleft backslash bracketright asciicircum
underscore quoteleft a b c d e f g h i j k l m n o p q r s t u v w x y z
braceleft bar braceright asciitilde hyphen.alt afii10050 uni0492 afii10051
afii10060 uni04BA uni0496 uni0498 afii10058 afii10056 uni049A uni04A0
uni049C uni04D4 uni04A2 uni04A4 afii10054 uni04E8 uni04AA afii10062 uni04AE
uni04B0 uni04B2 afii10145 uni04B8 uni04B6 afii10053 uni04D8 afii10059
afii10023 afii61352 currency section afii10098 uni0493 afii10099 afii10108
uni04BB uni0497 uni0499 afii10106 afii10104 uni049B uni04A1 uni049D uni04D5
uni04A3 uni04A5 afii10102 uni04E9 uni04AB afii10110 uni04AF uni04B1 uni04B3
afii10193 uni04B9 uni04B7 afii10101 afii10846 afii10107 afii10071
quotedblbase guillemotleft.cyr guillemotright.cyr afii10017 afii10018 afii10019
afii10020 afii10021 afii10022 afii10024 afii10025 afii10026 afii10027
afii10028 afii10029 afii10030 afii10031 afii10032 afii10033 afii10034
afii10035 afii10036 afii10037 afii10038 afii10039 afii10040 afii10041
afii10042 afii10043 afii10044 afii10045 afii10046 afii10047 afii10048
afii10049 afii10065 afii10066 afii10067 afii10068 afii10069 afii10070
afii10072 afii10073 afii10074 afii10075 afii10076 afii10077 afii10078
afii10079 afii10080 afii10081 afii10082 afii10083 afii10084 afii10085
afii10086 afii10087 afii10088 afii10089 afii10090 afii10091 afii10092
afii10093 afii10094 afii10095 afii10096 afii10097
""".split()

T2A_MAP = {
    code: agl.toUnicode(glyph_name)
    for code, glyph_name in enumerate(T2A_GLYPH_NAMES)
    if agl.toUnicode(glyph_name)
}

T2A_SPECIAL_TOKENS = {
    0x0D: r"\uni04C0",
    0x12: r"\cyrflex",
    0x13: r"\dblgrave",
    0x14: r"\cyrbreve",
    0x18: r"\perthousandzero",
}

PLACEHOLDER_RE = re.compile(r"^\(cid:\d+\)$")
HEX_RE = re.compile(r"<([0-9A-Fa-f]+)>")
TYPE1_ENCODING_RE = re.compile(
    r"/Encoding\s+256\s+array(?P<body>.*?)readonly\s+def", re.S
)
TYPE1_GLYPH_RE = re.compile(r"dup\s+(\d+)\s+/([A-Za-z0-9_.]+)\s+put")

CMEX_GLYPH_MAP = {
    "integraldisplay": ("∫", r"\int", "operator"),
    "integraltext": ("∫", r"\int", "operator"),
    "summationdisplay": ("∑", r"\sum", "operator"),
    "summationtext": ("∑", r"\sum", "operator"),
    "productdisplay": ("∏", r"\prod", "operator"),
    "producttext": ("∏", r"\prod", "operator"),
    "uniondisplay": ("⋃", r"\bigcup", "operator"),
    "intersectdisplay": ("⋂", r"\bigcap", "operator"),
    "radicalbig": ("√", r"\sqrt{}", "operator"),
    "radicalbigg": ("√", r"\sqrt{}", "operator"),
    "radicalBigg": ("√", r"\sqrt{}", "operator"),
    "hatwide": ("\u0302", r"\widehat{}", "accent"),
    "tildewide": ("\u0303", r"\widetilde{}", "accent"),
    "vextendsingle": ("|", r"\vert", "delimiter_extender"),
    "vextenddouble": ("‖", r"\Vert", "delimiter_extender"),
}

CMEX_DELIMITER_PREFIXES = (
    "parenleft",
    "parenright",
    "bracketleft",
    "bracketright",
    "braceleft",
    "braceright",
    "angleleft",
    "angleright",
    "floorleft",
    "floorright",
    "ceilingleft",
    "ceilingright",
    "slash",
    "backslash",
)

CMEX_DELIMITER_CHARACTERS = {
    "parenleft": "(",
    "parenright": ")",
    "bracketleft": "[",
    "bracketright": "]",
    "braceleft": "{",
    "braceright": "}",
    "angleleft": "⟨",
    "angleright": "⟩",
    "floorleft": "⌊",
    "floorright": "⌋",
    "ceilingleft": "⌈",
    "ceilingright": "⌉",
    "slash": "/",
    "backslash": "\\",
}

CMEX_DELIMITER_PARTS = {
    "tp": "top",
    "bt": "bottom",
    "mid": "middle",
    "ex": "extender",
}

SUPPLEMENTAL_GLYPH_ALIASES = {
    "mapsto": "↦",
    "rho1": "ϱ",
    "unionsqtext": "⊔",
    "intersectionsqtext": "⊓",
    "circledottext": "⊙",
    "circleplustext": "⊕",
    "circlemultiplytext": "⊗",
    "circleminustext": "⊖",
    "negationslash": "\u0338",
}


@dataclass
class FontInfo:
    key: str
    resources: set[str] = field(default_factory=set)
    pages: set[int] = field(default_factory=set)
    subtype: str = ""
    base_font: str = ""
    family: str = ""
    object_id: int | None = None
    object_generation: int | None = None
    differences: dict[int, str] = field(default_factory=dict)
    embedded_encoding: dict[int, str] = field(default_factory=dict)
    to_unicode: dict[int, str] = field(default_factory=dict)
    font_matrix: tuple[float, ...] | None = None
    outline_bounds: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    type1_parse_error: str = ""
    observed: Counter[int] = field(default_factory=Counter)
    map_kind: str = "unresolved"
    map_evidence: str = ""


def _pdf_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lstrip("/")


def _indirect_identity(
    reference: Any, obj: Any
) -> tuple[str, int | None, int | None]:
    object_id = getattr(reference, "idnum", None)
    generation = getattr(reference, "generation", None)
    if object_id is None:
        return f"direct:{id(obj)}", None, None
    return f"obj:{object_id}:{generation or 0}", object_id, generation or 0


def _page_resources(page: Any) -> Any:
    current = page
    while current is not None:
        try:
            resources = current.get("/Resources")
        except AttributeError:
            resources = None
        if resources is not None:
            return resources.get_object()
        parent = current.get("/Parent") if hasattr(current, "get") else None
        current = parent.get_object() if parent is not None else None
    return {}


def _parse_differences(font_dictionary: Any) -> dict[int, str]:
    encoding = font_dictionary.get("/Encoding")
    if encoding is None:
        return {}
    encoding = encoding.get_object()
    differences = encoding.get("/Differences", [])
    code: int | None = None
    result: dict[int, str] = {}
    for item in differences:
        if isinstance(item, int):
            code = int(item)
        elif code is not None:
            result[code] = _pdf_name(item)
            code += 1
    return result


def _decode_utf16_hex(value: str) -> str:
    raw = bytes.fromhex(value)
    if len(raw) % 2:
        raw = b"\x00" + raw
    try:
        return raw.decode("utf-16-be")
    except UnicodeDecodeError:
        return ""


def _parse_tounicode(font_dictionary: Any) -> dict[int, str]:
    stream = font_dictionary.get("/ToUnicode")
    if stream is None:
        return {}
    try:
        raw_content = stream.get_object().get_data()
    except Exception:
        return {}
    cmap = FileUnicodeMap()
    try:
        CMapParser(cmap, io.BytesIO(raw_content)).run()
        if cmap.cid2unichr:
            return dict(cmap.cid2unichr)
    except Exception:
        pass
    content = raw_content.decode("latin-1", errors="ignore")
    return _parse_tounicode_regex(content)


def _parse_tounicode_regex(content: str) -> dict[int, str]:
    """Parse common ToUnicode forms if the native CMap parser rejects one."""
    mapping: dict[int, str] = {}
    for block in re.findall(r"beginbfchar(.*?)endbfchar", content, re.S):
        for source, target in re.findall(
            r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block
        ):
            decoded = _decode_utf16_hex(target)
            if decoded:
                mapping[int(source, 16)] = decoded

    for block in re.findall(r"beginbfrange(.*?)endbfrange", content, re.S):
        groups = re.findall(
            r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*"
            r"(?:<([0-9A-Fa-f]+)>|\[(.*?)\])",
            block,
            re.S,
        )
        for start_hex, end_hex, target_hex, target_array in groups:
            start = int(start_hex, 16)
            end = int(end_hex, 16)
            if target_array:
                targets = HEX_RE.findall(target_array)
                for offset, target in enumerate(targets[: end - start + 1]):
                    decoded = _decode_utf16_hex(target)
                    if decoded:
                        mapping[start + offset] = decoded
                continue
            base_bytes = bytes.fromhex(target_hex)
            if len(base_bytes) % 2:
                base_bytes = b"\x00" + base_bytes
            base = int.from_bytes(base_bytes, "big")
            for offset, source in enumerate(range(start, end + 1)):
                try:
                    decoded = (
                        (base + offset)
                        .to_bytes(len(base_bytes), "big")
                        .decode("utf-16-be")
                    )
                except (OverflowError, UnicodeDecodeError):
                    continue
                mapping[source] = decoded
    return mapping


def _parse_type1_font(font_dictionary: Any) -> tuple[
    dict[int, str],
    tuple[float, ...] | None,
    dict[str, tuple[float, float, float, float]],
    str,
]:
    descriptor = font_dictionary.get("/FontDescriptor")
    if descriptor is None:
        return {}, None, {}, ""
    try:
        descriptor = descriptor.get_object()
        stream = descriptor.get("/FontFile")
        if stream is None:
            return {}, None, {}, ""
        data = stream.get_object().get_data()
    except Exception as error:
        return {}, None, {}, f"Cannot read embedded Type 1 FontFile: {error}"

    text = data.decode("latin-1", errors="ignore")
    match = TYPE1_ENCODING_RE.search(text)
    encoding = {
        int(code): name
        for code, name in TYPE1_GLYPH_RE.findall(match.group("body"))
        if name != ".notdef"
    } if match else {}

    font_matrix = None
    outlines: dict[str, tuple[float, float, float, float]] = {}
    parse_error = ""
    try:
        type1_font = T1Font.__new__(T1Font)
        type1_font.data = data
        type1_font.encoding = "ascii"
        type1_font.parse()
        matrix = type1_font["FontMatrix"]
        if len(matrix) == 6:
            font_matrix = tuple(float(value) for value in matrix)
        glyph_set = type1_font.getGlyphSet()
        for glyph_name, glyph in glyph_set.items():
            try:
                pen = BoundsPen(glyph_set)
                glyph.draw(pen)
                if pen.bounds is not None:
                    outlines[glyph_name] = tuple(
                        float(value) for value in pen.bounds
                    )
            except Exception:
                continue
    except Exception as error:
        parse_error = f"fontTools Type 1 outline parsing failed: {error}"
    return encoding, font_matrix, outlines, parse_error


def _make_font_info(
    reference: Any, font_dictionary: Any, resource_name: str, page_number: int
) -> FontInfo:
    key, object_id, generation = _indirect_identity(reference, font_dictionary)
    base_font = _pdf_name(font_dictionary.get("/BaseFont"))
    family = base_font.split("+")[-1] if base_font else ""
    encoding, font_matrix, outlines, parse_error = _parse_type1_font(
        font_dictionary
    )
    info = FontInfo(
        key=key,
        subtype=_pdf_name(font_dictionary.get("/Subtype")),
        base_font=base_font,
        family=family,
        object_id=object_id,
        object_generation=generation,
        differences=_parse_differences(font_dictionary),
        embedded_encoding=encoding,
        to_unicode=_parse_tounicode(font_dictionary),
        font_matrix=font_matrix,
        outline_bounds=outlines,
        type1_parse_error=parse_error,
    )
    if info.to_unicode and info.embedded_encoding:
        info.map_kind = "embedded_type1_encoding_and_tounicode_cmap"
    elif info.to_unicode:
        info.map_kind = "embedded_tounicode_cmap"
    elif info.embedded_encoding:
        info.map_kind = "embedded_type1_encoding"
    info.resources.add(resource_name)
    info.pages.add(page_number)
    return info


def _font_definitions(
    page: Any, page_number: int, cache: dict[str, FontInfo]
) -> dict[str, list[FontInfo]]:
    result: dict[str, list[FontInfo]] = defaultdict(list)
    visited_resources: set[int] = set()

    def collect(resources: Any) -> None:
        if resources is None:
            return
        resources = resources.get_object()
        identity = id(resources)
        if identity in visited_resources:
            return
        visited_resources.add(identity)

        font_resources = resources.get("/Font")
        if font_resources is not None:
            entries = font_resources.get_object().items()
            for resource_name, reference in entries:
                name = _pdf_name(resource_name)
                font_dictionary = reference.get_object()
                key, _, _ = _indirect_identity(reference, font_dictionary)
                info = cache.get(key)
                if info is None:
                    info = _make_font_info(
                        reference, font_dictionary, name, page_number
                    )
                    cache[key] = info
                else:
                    info.resources.add(name)
                    info.pages.add(page_number)
                if all(
                    candidate.key != info.key for candidate in result[name]
                ):
                    result[name].append(info)

        xobjects = resources.get("/XObject")
        if xobjects is None:
            return
        for reference in xobjects.get_object().values():
            xobject = reference.get_object()
            if _pdf_name(xobject.get("/Subtype")) == "Form":
                collect(xobject.get("/Resources"))

    collect(_page_resources(page))
    return dict(result)


def _match_font_info(
    candidates: list[FontInfo], runtime_name: str
) -> FontInfo | None:
    if not candidates:
        return None
    runtime_family = runtime_name.split("+")[-1]
    exact_matches = [
        info for info in candidates if info.base_font == runtime_name
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if exact_matches:
        return None

    family_matches = [
        info
        for info in candidates
        if info.family == runtime_family
        or (
            info.base_font
            and info.base_font.split("+")[-1] == runtime_family
        )
    ]
    if len(family_matches) == 1:
        return family_matches[0]
    if family_matches:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return None


def _ink_bbox(
    info: FontInfo | None,
    glyph_name: str | None,
    matrix: tuple[float, ...],
    fontsize: float,
    scaling: float,
    rise: float,
    page_height: float,
) -> list[float] | None:
    if (
        info is None
        or glyph_name is None
        or info.font_matrix is None
        or glyph_name not in info.outline_bounds
    ):
        return None

    local_bbox = apply_matrix_rect(
        info.font_matrix, info.outline_bounds[glyph_name]
    )
    local_bbox = (
        local_bbox[0] * fontsize * scaling,
        local_bbox[1] * fontsize + rise,
        local_bbox[2] * fontsize * scaling,
        local_bbox[3] * fontsize + rise,
    )
    user_bbox = apply_matrix_rect(matrix, local_bbox)
    return [
        _round(user_bbox[0]),
        _round(page_height - user_bbox[3]),
        _round(user_bbox[2]),
        _round(page_height - user_bbox[1]),
    ]


class GlyphCollector(PDFPageAggregator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.current_resource = ""
        self.page_height = 0.0
        self.page_fonts: dict[str, list[FontInfo]] = {}
        self.font_inventory: dict[str, FontInfo] = {}
        self.pages: list[dict[str, Any]] = []
        self.current_glyphs: list[dict[str, Any]] = []
        self.current_figures: list[dict[str, Any]] = []
        self.active_figure_ids: list[str] = []
        self.current_page_number = 0

    def start_page(self, page_number: int, page_height: float) -> None:
        self.current_page_number = page_number
        self.page_height = page_height
        self.current_glyphs = []
        self.current_figures = []
        self.active_figure_ids = []

    def begin_figure(self, name: str, bbox: Any, matrix: Any) -> None:
        super().begin_figure(name, bbox, matrix)
        transformed_bbox = self.cur_item.bbox
        figure_id = (
            f"p{self.current_page_number}:f{len(self.current_figures) + 1}"
        )
        figure = {
            "id": figure_id,
            "name": name,
            "bbox": [
                _round(transformed_bbox[0]),
                _round(self.page_height - transformed_bbox[3]),
                _round(transformed_bbox[2]),
                _round(self.page_height - transformed_bbox[1]),
            ],
        }
        if self.active_figure_ids:
            figure["parent_figure_id"] = self.active_figure_ids[-1]
        self.current_figures.append(figure)
        self.active_figure_ids.append(figure_id)

    def end_figure(self, name: str) -> None:
        super().end_figure(name)
        if self.active_figure_ids:
            self.active_figure_ids.pop()

    def finish_page(
        self, width: float, height: float, rules: list[dict[str, Any]]
    ) -> None:
        _tag_rules_with_figures(rules, self.current_figures)
        self.pages.append(
            {
                "page": self.current_page_number,
                "width": _round(width),
                "height": _round(height),
                "glyphs": self.current_glyphs,
                "rules": rules,
                "figures": self.current_figures,
            }
        )

    def render_char(
        self,
        matrix: tuple[float, ...],
        font: Any,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs: Any,
        graphicstate: Any,
    ) -> float:
        try:
            raw_text = font.to_unichr(cid)
        except Exception:
            raw_text = f"(cid:{cid})"

        item_count = len(getattr(self.cur_item, "_objs", []))
        advance = super().render_char(
            matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate
        )
        objects = getattr(self.cur_item, "_objs", [])
        item = objects[-1] if len(objects) > item_count else None
        if item is None or not hasattr(item, "bbox"):
            return advance

        bbox = item.bbox
        top_left_bbox = [
            bbox[0],
            self.page_height - bbox[3],
            bbox[2],
            self.page_height - bbox[1],
        ]
        resource = self.current_resource
        runtime_font_name = str(font.fontname or "unknown")
        info = _match_font_info(
            self.page_fonts.get(resource, []), runtime_font_name
        )
        if info is not None:
            inventory = self.font_inventory.setdefault(info.key, info)
            inventory.pages.add(self.current_page_number)
            inventory.resources.add(resource)
            inventory.observed[cid] += 1
            font_key = info.key
            font_name = info.family or runtime_font_name
        else:
            font_key = f"pdfminer:{runtime_font_name}:{resource or 'unknown'}"
            font_name = runtime_font_name
            info = self.font_inventory.setdefault(
                font_key,
                FontInfo(
                    key=font_key,
                    base_font=runtime_font_name,
                    family=runtime_font_name.split("+")[-1],
                    map_kind="pdfminer_fallback",
                    map_evidence=(
                        "No matching font dictionary in the page/Form "
                        "resource "
                        "stack; PDFMiner's font-specific decoding is retained."
                    ),
                ),
            )
            info.pages.add(self.current_page_number)
            info.resources.add(resource)
            info.observed[cid] += 1

        baseline = self.page_height - (float(matrix[5]) + float(rise))
        glyph_name = None
        glyph_name_source = None
        if info is not None:
            glyph_name = info.embedded_encoding.get(int(cid))
            glyph_name_source = (
                "embedded_type1_encoding" if glyph_name else None
            )
            if glyph_name is None:
                glyph_name = info.differences.get(int(cid))
                glyph_name_source = (
                    "pdf_encoding_differences" if glyph_name else None
                )
        ink_bbox = _ink_bbox(
            info,
            glyph_name,
            matrix,
            float(fontsize),
            float(scaling),
            float(rise),
            self.page_height,
        )
        self.current_glyphs.append(
            {
                "text": "",
                "raw_text": raw_text,
                "font": font_name,
                "font_pdfminer_name": runtime_font_name,
                "font_key": font_key,
                "font_resource": resource,
                "figure_id": (
                    self.active_figure_ids[-1]
                    if self.active_figure_ids
                    else None
                ),
                "font_subtype": info.subtype if info is not None else "",
                "code": int(cid),
                "code_hex": (
                    f"0x{int(cid):02X}"
                    if int(cid) <= 0xFF
                    else f"0x{int(cid):X}"
                ),
                "glyph_name": glyph_name,
                "glyph_name_source": glyph_name_source,
                "size": _round(float(fontsize)),
                "bbox": [_round(value) for value in top_left_bbox],
                "advance_bbox": [_round(value) for value in top_left_bbox],
                "ink_bbox": ink_bbox,
                "ink_bbox_provenance": (
                    "Embedded Type 1 CharString outline via fontTools "
                    "BoundsPen"
                    if ink_bbox is not None
                    else None
                ),
                "baseline": _round(baseline),
                "mapping": "pending",
                "provenance": "pending",
                "confidence": 0.0,
                "unknown": False,
            }
        )
        return advance


class ResourceTrackingInterpreter(PDFPageInterpreter):
    def do_Tf(self, fontid: Any, fontsize: float) -> None:
        self.device.current_resource = getattr(
            fontid, "name", str(fontid)
        ).lstrip("/")
        super().do_Tf(fontid, fontsize)


def _round(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(float(value), 4)


def _build_rules(pdf_page: pdfplumber.page.Page) -> list[dict[str, float]]:
    rules: list[dict[str, float]] = []
    for edge in pdf_page.edges:
        width = edge.get("linewidth")
        rules.append(
            {
                "x0": _round(float(edge.get("x0", 0.0))),
                "y0": _round(float(edge.get("top", 0.0))),
                "x1": _round(float(edge.get("x1", 0.0))),
                "y1": _round(float(edge.get("bottom", 0.0))),
                "width": _round(float(width or 0.0)),
            }
        )
    return rules


def _tag_rules_with_figures(
    rules: list[dict[str, Any]], figures: list[dict[str, Any]]
) -> None:
    for rule in rules:
        center_x = (rule["x0"] + rule["x1"]) / 2
        center_y = (rule["y0"] + rule["y1"]) / 2
        containing = []
        for figure in figures:
            x0, y0, x1, y1 = figure["bbox"]
            if x0 <= center_x <= x1 and y0 <= center_y <= y1:
                area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                containing.append((area, figure["id"]))
        if containing:
            rule["figure_id"] = min(containing)[1]


def _read_raw_pages(
    pdf_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, FontInfo]]:
    reader = PdfReader(str(pdf_path))
    with pdf_path.open("rb") as pdf_file:
        document = PDFDocument(PDFParser(pdf_file))
        resource_manager = PDFResourceManager()
        device = GlyphCollector(resource_manager, laparams=LAParams())
        interpreter = ResourceTrackingInterpreter(resource_manager, device)
        font_cache: dict[str, FontInfo] = {}

        try:
            with pdfplumber.open(str(pdf_path)) as plumber_pdf:
                page_streams = zip(
                    PDFPage.create_pages(document),
                    plumber_pdf.pages,
                    reader.pages,
                    strict=True,
                )
                for page_number, (pdfminer_page, plumber_page, pypdf_page) in (
                    enumerate(page_streams, start=1)
                ):
                    device.page_fonts = _font_definitions(
                        pypdf_page, page_number, font_cache
                    )
                    device.start_page(page_number, float(plumber_page.height))
                    interpreter.process_page(pdfminer_page)
                    device.finish_page(
                        float(plumber_page.width),
                        float(plumber_page.height),
                        _build_rules(plumber_page),
                    )
        finally:
            device.close()

    return device.pages, device.font_inventory


def _identify_type3_cyrillic_fonts(
    fonts: dict[str, FontInfo], profile: str
) -> None:
    if profile != "cm-super-t2a":
        return
    for info in fonts.values():
        if info.subtype != "Type3" or info.to_unicode:
            continue
        upper = {code for code in info.observed if 0xC0 <= code <= 0xDF}
        lower = {code for code in info.observed if 0xE0 <= code <= 0xFF}
        observed_cyrillic = upper | lower
        if len(observed_cyrillic) >= 3 and (upper or lower):
            info.map_kind = "t2a_source_profile"
            info.map_evidence = (
                "The explicit --type3-encoding=cm-super-t2a profile was "
                "selected for this PDF. This Type 3 font has no ToUnicode and "
                "its observed "
                "codes include at least three T2A letter slots in 0xC0--0xFF. "
                "That code coverage is a PDF-specific classification signal, "
                "not a universal proof of T2A; the document's Russian prose "
                "was "
                "visually checked against the published vector. Generic aNN "
                "glyph names alone are not treated as evidence."
            )


def _unknown_token(
    font_resource: str, code: int, glyph_name: str | None
) -> str:
    suffix = f"/{glyph_name}" if glyph_name else ""
    return f"\\glyph{{{font_resource or 'unknown'}}}{{0x{code:X}{suffix}}}"


def _cmex_glyph_mapping(glyph_name: str) -> dict[str, Any] | None:
    special = CMEX_GLYPH_MAP.get(glyph_name)
    if special is not None:
        text, tex_token, structural_kind = special
        return {
            "text": text,
            "tex_token": tex_token,
            "structural_kind": structural_kind,
        }

    if glyph_name == "braceex":
        return {
            "text": "",
            "tex_token": r"\delimiterpart{brace}{extender}",
            "structural_kind": "delimiter_part",
            "delimiter_base": "brace",
            "delimiter_part": "extender",
        }

    for prefix in CMEX_DELIMITER_PREFIXES:
        if not glyph_name.startswith(prefix):
            continue
        suffix = glyph_name[len(prefix) :]
        character = CMEX_DELIMITER_CHARACTERS[prefix]
        if suffix in CMEX_DELIMITER_PARTS:
            part = CMEX_DELIMITER_PARTS[suffix]
            return {
                "text": character,
                "tex_token": f"\\delimiterpart{{{character}}}{{{part}}}",
                "structural_kind": "delimiter_part",
                "delimiter_base": character,
                "delimiter_part": part,
            }
        if suffix in {"", "big", "bigg", "Big", "Bigg"}:
            token = None
            if suffix:
                side = "l" if prefix.endswith("left") else "r"
                tex_character = {
                    "{": r"\{",
                    "}": r"\}",
                    "⟨": r"\langle",
                    "⟩": r"\rangle",
                    "⌊": r"\lfloor",
                    "⌋": r"\rfloor",
                    "⌈": r"\lceil",
                    "⌉": r"\rceil",
                }.get(character, character)
                token = f"\\{suffix}{side}{tex_character}"
            return {
                "text": character,
                "tex_token": token,
                "structural_kind": "delimiter",
                "delimiter_base": character,
                "delimiter_scale": suffix or "normal",
            }
    return None


def _is_private_use(text: str) -> bool:
    return bool(text) and all(
        0xE000 <= ord(char) <= 0xF8FF
        or 0xF0000 <= ord(char) <= 0xFFFFD
        or 0x100000 <= ord(char) <= 0x10FFFD
        for char in text
    )


def _contains_private_use(text: str) -> bool:
    return any(
        0xE000 <= ord(char) <= 0xF8FF
        or 0xF0000 <= ord(char) <= 0xFFFFD
        or 0x100000 <= ord(char) <= 0x10FFFD
        for char in text
    )


def _is_usable_unicode_text(text: str) -> bool:
    """Reject missing, placeholder, and control-only extraction results."""
    return bool(text) and not PLACEHOLDER_RE.match(text) and all(
        char.isprintable() for char in text
    )


def _map_glyph(glyph: dict[str, Any], fonts: dict[str, FontInfo]) -> None:
    code = int(glyph["code"])
    info = fonts.get(glyph["font_key"])
    raw_text = glyph["raw_text"]

    glyph_name = (
        info.embedded_encoding.get(code) or info.differences.get(code)
        if info is not None
        else None
    )
    if glyph_name:
        glyph["glyph_name"] = glyph_name
        glyph["glyph_name_source"] = (
            "embedded_type1_encoding"
            if code in info.embedded_encoding
            else "pdf_encoding_differences"
        )

    cmex_mapping = (
        _cmex_glyph_mapping(glyph_name)
        if info is not None and info.family.startswith("CMEX") and glyph_name
        else None
    )
    if cmex_mapping is not None:
        if code in info.to_unicode and _is_private_use(info.to_unicode[code]):
            glyph["candidate_tounicode_text"] = info.to_unicode[code]
        glyph.update(
            **cmex_mapping,
            mapping="embedded_cmex_type1_encoding",
            provenance=(
                "Read-only parse of the embedded Type 1 /Encoding glyph name; "
                "semantic Unicode and delimiter roles derive from the glyph "
                "name."
            ),
            confidence=0.99,
        )
    elif (
        info is not None
        and code in info.to_unicode
        and _is_usable_unicode_text(info.to_unicode[code])
    ):
        text = info.to_unicode[code]
        if info.family.startswith("CMEX") and _is_private_use(text):
            glyph.update(
                text=text,
                tex_token=f"\\pua{{U+{ord(text[0]):04X}}}",
                mapping="pdf_tounicode_private_use_structural",
                provenance=(
                    "Embedded /ToUnicode maps to a Private Use Area "
                    "codepoint; "
                    "retained explicitly rather than treated as ordinary text."
                ),
                structural_kind="private_use_glyph",
                unknown=False,
                confidence=0.8,
            )
        else:
            glyph.update(
                text=text,
                mapping="pdf_tounicode_cmap",
                provenance=(
                    "Embedded PDF /ToUnicode CMap parsed by pdfminer "
                    "CMapParser"
                ),
                confidence=0.99,
            )
    elif (
        info is not None
        and info.map_kind == "t2a_source_profile"
        and code < len(T2A_GLYPH_NAMES)
    ):
        glyph_name = T2A_GLYPH_NAMES[code]
        mapped_text = T2A_MAP.get(code)
        glyph["glyph_name"] = glyph_name
        glyph["glyph_name_source"] = "cm_super_t2a_encoding_vector"
        if mapped_text:
            tex_token = None
            if code == 0x15:
                tex_token = r"\textendash"
            elif code == 0x16:
                tex_token = r"\cyrdash"
            elif 0xC0 <= code <= 0xFF:
                tex_token = f"\\cyr{{{mapped_text}}}"
            glyph.update(
                text=mapped_text,
                tex_token=tex_token,
                mapping="cm_super_t2a_encoding_vector",
                provenance=(
                    "Explicit source-specific CM-Super T2A profile; slot "
                    "mapping comes from the published T2AEncoding vector "
                    "and Adobe Glyph "
                    "List. PDF-specific classification evidence is recorded "
                    "in "
                    "mapping_report.json."
                ),
                confidence=0.91,
            )
        else:
            token = T2A_SPECIAL_TOKENS.get(code)
            if token:
                glyph.update(
                    text=token,
                    tex_token=token,
                    mapping="cm_super_t2a_tex_glyph_token",
                    provenance=(
                        "Published T2AEncoding slot has a known TeX glyph "
                        "name but "
                        "no unambiguous Unicode scalar value."
                    ),
                    confidence=0.85,
                )
            else:
                glyph.update(
                    text=_unknown_token(
                        glyph["font_resource"], code, glyph_name
                    ),
                    mapping="unresolved",
                    provenance=(
                        "T2A glyph name has no registered Unicode mapping."
                    ),
                    confidence=0.0,
                    unknown=True,
                )
    elif info is not None and info.embedded_encoding.get(code):
        embedded_name = info.embedded_encoding[code]
        mapped_text = SUPPLEMENTAL_GLYPH_ALIASES.get(
            embedded_name, agl.toUnicode(embedded_name)
        )
        if embedded_name == "negationslash":
            mapped_text = "\u0338"
            glyph["structural_kind"] = "overlay"
            glyph["tex_token"] = r"\not"
        if mapped_text:
            glyph.update(
                text=mapped_text,
                mapping="embedded_type1_encoding_glyph_name",
                provenance=(
                    "Read-only parse of the embedded Type 1 /Encoding glyph "
                    "name; "
                    "Unicode resolved through fontTools Adobe Glyph List."
                ),
                confidence=0.95,
            )
        else:
            glyph.update(
                text=_unknown_token(
                    glyph["font_resource"], code, embedded_name
                ),
                mapping="unresolved_glyph_name",
                provenance=(
                    "Embedded Type 1 Encoding provides a glyph name without a "
                    "verified Unicode mapping."
                ),
                confidence=0.0,
                unknown=True,
            )
    elif (
        _is_usable_unicode_text(raw_text)
        and not _contains_private_use(raw_text)
    ):
        glyph.update(
            text=raw_text,
            mapping="pdfminer_font_encoding_fallback",
            provenance=(
                "PDFMiner font-specific CMap/encoding fallback; no verified "
                "override was available."
            ),
            confidence=0.7,
        )
    else:
        glyph.update(
            text=_unknown_token(
                glyph["font_resource"], code, glyph.get("glyph_name")
            ),
            mapping="unresolved",
            provenance=(
                "No embedded ToUnicode mapping or verified encoding table"
            ),
            confidence=0.0,
            unknown=True,
        )


def _map_pages(
    pages: list[dict[str, Any]], fonts: dict[str, FontInfo]
) -> dict[str, Counter[int]]:
    unresolved_by_font: dict[str, Counter[int]] = defaultdict(Counter)
    for page in pages:
        for glyph in page["glyphs"]:
            _map_glyph(glyph, fonts)
            if glyph["unknown"]:
                unresolved_by_font[glyph["font_key"]][int(glyph["code"])] += 1
    return unresolved_by_font


def _font_report(fonts: dict[str, FontInfo]) -> list[dict[str, Any]]:
    reports = []
    for key, info in sorted(fonts.items()):
        reports.append(
            {
                "font_key": key,
                "resources": sorted(info.resources),
                "pages": sorted(info.pages),
                "subtype": info.subtype,
                "base_font": info.base_font,
                "family": info.family,
                "object_id": info.object_id,
                "object_generation": info.object_generation,
                "mapping_family": info.map_kind,
                "mapping_evidence": info.map_evidence,
                "has_tounicode": bool(info.to_unicode),
                "tounicode_entry_count": len(info.to_unicode),
                "has_embedded_type1_encoding": bool(info.embedded_encoding),
                "embedded_encoding_entry_count": len(info.embedded_encoding),
                "embedded_encoding_provenance": (
                    "Read-only regex parse of /Encoding ... readonly def in "
                    "the embedded Type 1 FontFile cleartext; no PostScript "
                    "execution."
                    if info.embedded_encoding
                    else None
                ),
                "font_matrix": (
                    list(info.font_matrix) if info.font_matrix else None
                ),
                "type1_outline_glyph_count": len(info.outline_bounds),
                "type1_outline_provenance": (
                    "fontTools T1Font parsing of embedded Type 1 CharStrings "
                    "and "
                    "BoundsPen; in-memory parsing, no PostScript execution."
                    if info.outline_bounds
                    else None
                ),
                "type1_parse_error": info.type1_parse_error or None,
                "observed_code_count": len(info.observed),
                "observed_glyph_count": sum(info.observed.values()),
                "observed_codes": [
                    {"code": code, "code_hex": f"0x{code:02X}", "count": count}
                    for code, count in sorted(info.observed.items())
                ],
                "mapping_sources": _font_sources(info),
            }
        )
    return reports


def _font_sources(info: FontInfo) -> list[str]:
    sources = []
    if info.to_unicode:
        sources.append(
            "Embedded PDF /ToUnicode CMap parsed by pdfminer CMapParser"
        )
    if info.embedded_encoding:
        sources.append(
            "Embedded Type 1 FontFile /Encoding cleartext; read-only parsing, "
            "glyph-name semantics from Adobe Glyph List"
        )
    if info.map_kind == "pdfminer_fallback":
        sources.append(
            "PDFMiner font-specific decoding fallback; no matching pypdf font "
            "dictionary was available in the resource stack"
        )
    if info.map_kind == "t2a_source_profile":
        sources.append(
            "CM-Super T2AEncoding vector: "
            "https://mirrors.ibiblio.org/CTAN/fonts/ps-type1/cm-super/"
            "dvips/cm-super-t2a.enc"
        )
    if info.family.startswith("CMEX"):
        sources.append(
            "Mozilla CMEX Type 1 encoding table: "
            "https://www-archive.mozilla.org/projects/mathml/fonts/encoding/"
            "cmex-t1-encoding"
        )
    if info.subtype == "Type3" and info.map_kind == "unresolved":
        sources.append(
            "No applicable standard mapping verified; preserve as unknown"
        )
    return sources


def _make_mapping_report(
    pdf_path: Path,
    pages: list[dict[str, Any]],
    fonts: dict[str, FontInfo],
    unresolved_by_font: dict[str, Counter[int]],
    type3_profile: str,
) -> dict[str, Any]:
    unresolved = []
    for font_key, codes in sorted(unresolved_by_font.items()):
        info = fonts.get(font_key)
        unresolved.append(
            {
                "font_key": font_key,
                "font": info.family if info else None,
                "font_resources": sorted(info.resources) if info else [],
                "codes": [
                    {
                        "code": code,
                        "code_hex": f"0x{code:02X}",
                        "count": count,
                        "glyph_name": (
                            info.embedded_encoding.get(code)
                            or info.differences.get(code)
                            if info
                            else None
                        ),
                    }
                    for code, count in sorted(codes.items())
                ],
            }
        )
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    glyphs = [glyph for page in pages for glyph in page["glyphs"]]
    return {
        "document": {
            "filename": pdf_path.name,
            "sha256": digest,
            "page_count": len(pages),
        },
        "extractor": {
            "name": "pdf_font_decoder.py",
            "pypdf": __import__("pypdf").__version__,
            "pdfminer_six": getattr(pdfminer, "__version__", "unknown"),
            "pdfplumber": pdfplumber.__version__,
            "fonttools": __import__("fontTools").version,
            "reading_order": "PDF content-stream order per physical page",
            "coordinates": (
                "top-left origin; points; baseline is text origin y"
            ),
            "glyph_boxes": (
                "bbox and advance_bbox are PDFMiner advance rectangles; "
                "ink_bbox is the transformed embedded Type 1 outline bounds "
                "when available"
            ),
            "figure_regions": (
                "Form/Image XObject bounds come from transformed PDFMiner "
                "LTFigure "
                "objects; glyphs and contained rule centers carry figure_id."
            ),
        },
        "mapping_policy": {
            "default": (
                "embedded ToUnicode CMap; when absent, embedded Type 1 "
                "Encoding "
                "glyph names and Adobe Glyph List"
            ),
            "cmex": (
                "CMEX glyph names are read from the embedded Type 1 "
                "/Encoding; "
                "Unicode symbols and delimiter part/size roles are retained"
            ),
            "type3_encoding_profile": type3_profile,
            "type3_cyrillic": (
                "The T2A map is applied only when explicitly selected with "
                "--type3-encoding=cm-super-t2a, and only to Type 3 font "
                "objects "
                "without ToUnicode that have at least three observed "
                "0xC0--0xFF "
                "slots. This is a PDF-specific inference supported by the "
                "visually checked Russian text; slot coverage and generic aNN "
                "names are not universal proof of T2A."
            ),
            "unknowns": (
                "Unmapped codes are preserved as explicit \\glyph tokens and "
                "listed below. Private Use Area mappings and structural CMEX "
                "delimiter pieces are reported separately."
            ),
            "no_contextual_repair": True,
            "no_ocr_or_language_model": True,
        },
        "sources": [
            {
                "name": "CM-Super T2AEncoding vector",
                "url": (
                    "https://mirrors.ibiblio.org/CTAN/fonts/ps-type1/cm-super/"
                    "dvips/cm-super-t2a.enc"
                ),
                "basis": (
                    "Published slot-to-glyph vector; glyph names mapped "
                    "through "
                    "fontTools Adobe Glyph List."
                ),
            },
            {
                "name": "CTAN T2 encoding map descriptions",
                "url": (
                    "https://ctan.org/tex-archive/"
                    "macros/latex/contrib/t2/enc-maps"
                ),
                "basis": (
                    "Identifies T2Auni.map as the mapping from T2A slots to "
                    "Unicode."
                ),
            },
            {
                "name": "LaTeX font encoding guide",
                "url": "https://tug.ctan.org/macros/latex/base/encguide.pdf",
                "basis": (
                    "Defines the standard TeX text and math font encoding "
                    "families."
                ),
            },
            {
                "name": "Mozilla CMEX Type 1 encoding table",
                "url": (
                    "https://www-archive.mozilla.org/projects/mathml/fonts/"
                    "encoding/cmex-t1-encoding"
                ),
                "basis": (
                    "Cross-check for CMEX glyph semantics and delimiter "
                    "assembly roles."
                ),
            },
            {
                "name": "FontTools Type 1 parser and BoundsPen",
                "url": "https://fonttools.readthedocs.io/en/latest/t1Lib.html",
                "basis": (
                    "Read embedded Type 1 cleartext encoding and CharString "
                    "outlines "
                    "in memory; PostScript is not executed."
                ),
            },
            {
                "name": "Unicode NamesList",
                "url": (
                    "https://www.unicode.org/Public/16.0.0/ucd/NamesList.txt"
                ),
                "basis": (
                    "Supplemental glyph-name aliases are Unicode symbol "
                    "names; "
                    "for example mapsto is U+21A6 RIGHTWARDS ARROW FROM BAR."
                ),
            },
        ],
        "fonts": _font_report(fonts),
        "unresolved_codes": unresolved,
        "summary": {
            "glyph_count": sum(len(page["glyphs"]) for page in pages),
            "unknown_glyph_count": sum(
                sum(codes.values()) for codes in unresolved_by_font.values()
            ),
            "unknown_font_count": len(unresolved_by_font),
            "private_use_structural_glyph_count": sum(
                glyph.get("mapping") == "pdf_tounicode_private_use_structural"
                for glyph in glyphs
            ),
            "structural_glyph_count": sum(
                bool(glyph.get("structural_kind")) for glyph in glyphs
            ),
            "glyphs_with_ink_bbox": sum(
                glyph.get("ink_bbox") is not None for glyph in glyphs
            ),
            "figure_region_count": sum(
                len(page.get("figures", [])) for page in pages
            ),
            "glyphs_with_figure_id": sum(
                glyph.get("figure_id") is not None for glyph in glyphs
            ),
            "rules_with_figure_id": sum(
                rule.get("figure_id") is not None
                for page in pages
                for rule in page["rules"]
            ),
        },
    }


def _write_jsonl(path: Path, pages: list[dict[str, Any]]) -> None:
    _atomic_write(path, _iter_jsonl(pages))


def _write_report(path: Path, report: dict[str, Any]) -> None:
    _atomic_write(
        path,
        [
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ],
    )


def _iter_jsonl(pages: list[dict[str, Any]]) -> Iterator[str]:
    for page in pages:
        yield json.dumps(
            page, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"


def _atomic_write(path: Path, chunks: Iterable[str]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as output:
            for chunk in chunks:
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def extract(
    pdf_path: Path,
    output_dir: Path,
    type3_encoding: str = "none",
) -> tuple[Path, Path]:
    pdf_path = pdf_path.expanduser().resolve()
    if type3_encoding not in {"none", "cm-super-t2a"}:
        raise ValueError(
            f"Unsupported Type 3 encoding profile: {type3_encoding}"
        )
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Input PDF is not a file: {pdf_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pages, fonts = _read_raw_pages(pdf_path)
    _identify_type3_cyrillic_fonts(fonts, type3_encoding)
    unresolved_by_font = _map_pages(pages, fonts)
    glyph_path = output_dir / "glyph_pages.jsonl"
    report_path = output_dir / "mapping_report.json"
    _write_jsonl(glyph_path, pages)
    _write_report(
        report_path,
        _make_mapping_report(
            pdf_path, pages, fonts, unresolved_by_font, type3_encoding
        ),
    )
    return glyph_path, report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract PDF glyphs with font-aware deterministic mappings."
        )
    )
    parser.add_argument(
        "--pdf", required=True, type=Path, help="Input PDF path"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local_data/font_analysis"),
        help="Directory for glyph_pages.jsonl and mapping_report.json",
    )
    parser.add_argument(
        "--type3-encoding",
        choices=("none", "cm-super-t2a"),
        default="none",
        help=(
            "Explicitly select the source-specific T2A profile for Type 3 "
            "fonts "
            "that lack ToUnicode; default leaves such fonts unmapped."
        ),
    )
    args = parser.parse_args(argv)
    glyph_path, report_path = extract(
        args.pdf, args.output_dir, type3_encoding=args.type3_encoding
    )
    print(f"Wrote {glyph_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
