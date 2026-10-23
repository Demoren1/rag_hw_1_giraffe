"""Evaluate local answer sets and page diagnostics, without official F1/Q."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from rag_baseline import (
    json_line,
    normalize_answer,
    question_records,
    read_jsonl,
    sha256_file,
    write_json,
)


def unique_by_id(path: Path) -> dict:
    rows = read_jsonl(path)
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate IDs: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "gold", "answers", "trace", "index", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    questions = question_records(args.questions)
    gold = unique_by_id(args.gold)
    answers = unique_by_id(args.answers)
    traces = unique_by_id(args.trace)
    expected = {row["id"] for row in questions}
    paired = (("gold", gold), ("answers", answers), ("trace", traces))
    for name, rows in paired:
        if set(rows) != expected:
            raise ValueError(f"{name} IDs do not match the questions")
    documents = read_jsonl(args.index / "documents.jsonl")
    by_block = {row["block_id"]: row for row in documents}
    categories = defaultdict(list)
    cases = []
    for question in questions:
        qid = question["id"]
        answer, reference, trace = answers[qid], gold[qid], traces[qid]
        errors = []
        try:
            no_context = trace.get("retrieval_mode") == "none"
            normalize_answer(
                json.dumps(answer), question,
                [] if no_context else documents,
                require_citations=not no_context,
            )
            if no_context and trace["retrieved_block_ids"]:
                errors.append("No-context run contains retrieved blocks")
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
        if trace.get("status") != "ok":
            errors.append("Inference trace has a technical error")
        if trace.get("limit_violation"):
            errors.append("Token budget exceeded")
        correct = (
            not errors
            and answer["abstain"] == reference["abstain"]
            and set(answer["answer_ids"]) == set(reference["answer_ids"])
        )
        categories[reference["category"]].append(correct)
        cited_pages = sorted({
            by_block[bid]["page"] for bid in answer["citations"]
            if bid in by_block
        })
        retrieved_pages = {
            by_block[bid]["page"]
            for bid in trace["retrieved_block_ids"] if bid in by_block
        }
        expected_pages = set(reference["source_pdf_pages"])
        cases.append({
            "id": qid, "category": reference["category"],
            "correct": bool(correct), "errors": errors,
            "answer_ids": answer["answer_ids"],
            "gold_answer_ids": reference["answer_ids"],
            "abstain": answer["abstain"],
            "gold_abstain": reference["abstain"],
            "rationale": answer["rationale"],
            "citations": answer["citations"], "cited_pages": cited_pages,
            "reference_pages": sorted(expected_pages),
            "retrieved_pages": sorted(retrieved_pages),
            "retrieved_reference_page_fraction": (
                len(retrieved_pages & expected_pages) / len(expected_pages)
                if expected_pages else None
            ),
            "any_cited_reference_page": bool(
                set(cited_pages) & expected_pages
            ),
        })

    usage_ok = all(
        row.get("usage_verified")
        and type(row.get("input_tokens")) is int
        and type(row.get("output_tokens")) is int
        for row in traces.values()
    )
    usage = {}
    if usage_ok:
        values = list(traces.values())
        calls = [call for row in values for call in row.get("calls", [row])]
        usage = {
            "mean_input_tokens": statistics.mean(
                row["input_tokens"] for row in values
            ),
            "mean_output_tokens": statistics.mean(
                row["output_tokens"] for row in values
            ),
            "mean_llm_calls": statistics.mean(
                row["llm_calls"] for row in values
            ),
            "mean_cost": statistics.mean(
                row["input_tokens"] + 4 * row["output_tokens"]
                + 256 * row["llm_calls"] for row in values
            ),
            "max_input_tokens": max(row["input_tokens"] for row in values),
            "max_output_tokens": max(row["output_tokens"] for row in values),
            "max_request_input_tokens": max(
                call["input_tokens"] for call in calls
            ),
            "max_request_output_tokens": max(
                call["output_tokens"] for call in calls
            ),
            "max_request_input_plus_allowed_output": max(
                call["input_tokens"] + call["request"]["max_tokens"]
                for call in calls
            ),
        }
    metrics = {
        "scope": "local_pilot_answer_sets_and_page_diagnostics",
        "citation_policy": (
            "empty_sources_for_no_context_answer_comparison"
            if all(row.get("retrieval_mode") == "none"
                   for row in traces.values())
            else "provided_source_ids_required_for_non_abstention"
        ),
        "questions": len(cases),
        "correct": sum(case["correct"] for case in cases),
        "answer_accuracy": statistics.mean(
            case["correct"] for case in cases
        ),
        "by_category": {
            name: {"count": len(values), "correct": sum(values),
                   "answer_accuracy": statistics.mean(values)}
            for name, values in categories.items()
        },
        "technical_or_format_errors": sum(bool(c["errors"]) for c in cases),
        "usage_verified": usage_ok,
        "limit_violations": sum(
            bool(row.get("limit_violation")) for row in traces.values()
        ),
        "median_latency_ms": statistics.median(
            row["elapsed_ms"] for row in traces.values()
        ),
        "mean_retrieved_reference_page_fraction": statistics.mean(
            case["retrieved_reference_page_fraction"] for case in cases
            if case["retrieved_reference_page_fraction"] is not None
        ),
        "official_citation_f1": None,
        "Q": None,
        "E": None,
        "limitations": (
            "Reference pages are diagnostic hints, not acceptable block-ID "
            "sets. Official corpus/gold/runtime/evaluator are unavailable. "
            "Q, citation F1, E and the final homework score are not computed."
        ),
        "files_sha256": {
            name: sha256_file(getattr(args, name))
            for name in ("questions", "gold", "answers", "trace")
        },
        **usage,
    }
    write_json(args.output, metrics)
    cases_path = args.output.with_name(args.output.stem + "_cases.jsonl")
    with cases_path.open("w", encoding="utf-8") as stream:
        stream.writelines(json_line(case) for case in cases)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
