"""Reapply response formatting rules to a saved run, without new LLM calls.

Original endpoint usage and timings are retained, not estimated anew.
This is deterministic postprocessing of each question's own response.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_baseline import (  # noqa: E402
    RESPONSE_PARSER_VERSION,
    json_line,
    normalize_answer,
    question_records,
    read_jsonl,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "index", "trace", "output", "output-trace"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.trace.resolve() in {
        args.output.resolve(), args.output_trace.resolve()
    } or args.output.resolve() == args.output_trace.resolve():
        raise ValueError("Keep the source trace and outputs separate")
    questions = question_records(args.questions)
    original = read_jsonl(args.trace)
    traces = {row["id"]: row for row in original}
    if len(traces) != len(original) or set(traces) != {
        question["id"] for question in questions
    }:
        raise ValueError("Saved trace IDs must match questions exactly")
    blocks = {
        row["block_id"]: row
        for row in read_jsonl(args.index / "documents.jsonl")
    }
    source_hash = sha256_file(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_trace.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as answers, (
        args.output_trace.open("w", encoding="utf-8")
    ) as output_trace:
        for question in questions:
            trace = dict(traces[question["id"]])
            trace["original_validation_status"] = trace["status"]
            trace["original_validation_error"] = trace.pop("error", None)
            trace["source_trace_sha256"] = source_hash
            trace["additional_llm_calls"] = 0
            trace["response_parser"] = RESPONSE_PARSER_VERSION
            repairs: list[str] = []
            try:
                if (
                    not trace.get("usage_verified")
                    or trace.get("limit_violation")
                    or trace.get("finish_reason") == "length"
                ):
                    raise ValueError("Original generation was not complete")
                documents = [
                    blocks[bid] for bid in trace["retrieved_block_ids"]
                ]
                if trace.get("pipeline") == "hybrid_evidence":
                    from hybrid_pipeline import (
                        WIRE_PARSER_VERSION,
                        parse_decision,
                    )

                    final = next(
                        call for call in reversed(trace["calls"])
                        if call["stage"] == "decision"
                    )
                    if final.get("finish_reason") == "length":
                        raise ValueError("Final generation was not complete")
                    answer = parse_decision(
                        final["response_text"], question, documents, repairs,
                        no_context=trace.get("retrieval_mode") == "none",
                    )
                    trace["response_parser"] = WIRE_PARSER_VERSION
                    trace["protocol_repairs"] = repairs
                else:
                    answer = normalize_answer(
                        trace["response_text"], question, documents,
                        canonicalize=True, repairs=repairs,
                    )
                trace["status"] = "ok"
            except (ValueError, KeyError, TypeError) as error:
                trace["status"] = "error"
                trace["error"] = f"{type(error).__name__}: {error}"[:1000]
                answer = {
                    "id": question["id"], "answer_ids": [], "abstain": True,
                    "rationale": (
                        "Технический отказ: корректный подтверждённый ответ "
                        "не получен. Подробности сохранены в trace."
                    ), "citations": [],
                }
            trace["format_repairs"] = repairs
            answers.write(json_line(answer))
            output_trace.write(json_line(trace))


if __name__ == "__main__":
    main()
