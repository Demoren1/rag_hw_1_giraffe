"""Convert the 15 Markdown QA examples into page-only pilot JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "materials" / "tfkp_15_qa.md"
PILOT_DIR = PROJECT_ROOT / "local_data" / "pilot"
QUESTIONS_PATH = PILOT_DIR / "pilot_questions.jsonl"
GOLD_PATH = PILOT_DIR / "pilot_gold_pages.jsonl"

SECTION_PATTERN = re.compile(
    r"(?m)^## (?P<number>\d+)\.\s+(?P<title>.+?)\s*$"
)
TYPE_PATTERN = re.compile(
    r"(?m)^\*\*Тип:\*\*\s*(?P<category>[^.]+?)\.\s*"
    r"\*\*Сложность:\*\*"
)
QUESTION_PATTERN = re.compile(r"(?m)^\*\*Вопрос\.\*\*\s*(?P<text>.+?)\s*$")
OPTION_PATTERN = re.compile(
    r"(?m)^- \*\*(?P<id>[A-Z])\.\*\*\s*(?P<text>.+?)\s*$"
)
ANSWER_PATTERN = re.compile(r"(?m)^\*\*Ответ:\*\*\s*(?P<answer>.+?)\s*$")
SOURCE_PATTERN = re.compile(
    r"(?m)^\*\*Источник:\*\*\s*(?P<description>.+?)\s*$"
)
PAGE_PATTERN = re.compile(r"PDF с\.\s*(?P<pages>\d+(?:\s*,\s*\d+)*)")


def split_sections(markdown: str) -> list[tuple[int, str]]:
    """Return numbered QA sections without changing their Markdown content."""
    matches = list(SECTION_PATTERN.finditer(markdown))
    sections: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(markdown)
        sections.append(
            (int(match.group("number")), markdown[match.end():end])
        )
    return sections


def require_match(
    pattern: re.Pattern[str],
    text: str,
    field: str,
) -> re.Match[str]:
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Could not parse {field} from a QA section")
    return match


def parse_section(
    number: int,
    section: str,
    source_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one source section to question and provisional gold rows."""
    question_match = require_match(QUESTION_PATTERN, section, "question")
    category_match = require_match(TYPE_PATTERN, section, "category")
    answer_match = require_match(ANSWER_PATTERN, section, "answer")
    source_match = require_match(SOURCE_PATTERN, section, "source description")

    options = [
        {"id": match.group("id"), "text": match.group("text")}
        for match in OPTION_PATTERN.finditer(section)
    ]
    option_ids = [option["id"] for option in options]
    if not 4 <= len(options) <= 6 or len(option_ids) != len(set(option_ids)):
        raise ValueError(f"Question {number} must have 4–6 unique options")

    answer_text = answer_match.group("answer").rstrip(".")
    abstain = answer_text.strip().casefold() == "отказ"
    answer_ids = [] if abstain else [
        value.strip() for value in answer_text.split(",") if value.strip()
    ]
    if abstain != (not answer_ids):
        raise ValueError(
            f"Question {number} has an inconsistent abstention label"
        )
    if any(answer_id not in option_ids for answer_id in answer_ids):
        raise ValueError(
            f"Question {number} answer references an unknown option"
        )

    answer_end = answer_match.end()
    source_start = source_match.start()
    rationale = section[answer_end:source_start].strip()
    source_description = source_match.group("description").strip()
    page_match = require_match(
        PAGE_PATTERN,
        source_description,
        "PDF page numbers",
    )
    source_pdf_pages = [
        int(page.strip()) for page in page_match.group("pages").split(",")
    ]

    record_id = f"pilot_q_{number:03d}"
    question = {
        "id": record_id,
        "question": question_match.group("text"),
        "options": options,
    }
    gold = {
        "id": record_id,
        "category": category_match.group("category").strip(),
        "answer_ids": answer_ids,
        "abstain": abstain,
        "rationale": rationale,
        "source_pdf_pages": source_pdf_pages,
        "source_description": source_description,
        "schema_version": "pilot_gold_v0.1",
        "provenance": {
            "source_markdown": source_name,
            "source_question_number": number,
            "status": "page_only_pilot_no_official_citation_f1",
            "official_evaluator_schema": False,
            "official_block_ids_available": False,
            "official_citation_f1_claimed": False,
        },
    }
    return question, gold


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write deterministic UTF-8 JSONL with stable field ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            output_file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the 15 Markdown QA examples to pilot JSONL files."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=PILOT_DIR)
    parser.add_argument(
        "--dev-output", type=Path,
        help="Also export dev_questions.jsonl and page-only dev_gold.jsonl.",
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    output_dir = args.output.resolve()
    markdown = source_path.read_text(encoding="utf-8")
    sections = split_sections(markdown)
    expected_numbers = list(range(1, 16))
    actual_numbers = [number for number, _ in sections]
    if actual_numbers != expected_numbers:
        raise ValueError(
            f"Expected QA sections {expected_numbers}, found {actual_numbers}"
        )

    parsed = [
        parse_section(number, section, source_path.name)
        for number, section in sections
    ]
    questions = [question for question, _ in parsed]
    gold = [gold_record for _, gold_record in parsed]
    questions_path = output_dir / QUESTIONS_PATH.name
    gold_path = output_dir / GOLD_PATH.name
    write_jsonl(questions_path, questions)
    write_jsonl(gold_path, gold)
    print(f"Wrote {len(questions)} questions to {questions_path}")
    print(f"Wrote {len(gold)} page-only gold records to {gold_path}")
    if args.dev_output is not None:
        dev_dir = args.dev_output.resolve()
        write_jsonl(dev_dir / "dev_questions.jsonl", questions)
        write_jsonl(dev_dir / "dev_gold.jsonl", gold)
        print(f"Exported dev questions and page-only gold to {dev_dir}")


if __name__ == "__main__":
    main()
