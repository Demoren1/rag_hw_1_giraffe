"""Export unchanged pilot answers and a compact, corpus-free trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_baseline import (  # noqa: E402
    json_line,
    normalize_answer,
    question_records,
    read_jsonl,
    sha256_file,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    questions = {q["id"]: q for q in question_records(args.questions)}
    published_questions = sha256_file(ROOT / "dev_questions.jsonl")
    if published_questions != sha256_file(args.questions):
        raise ValueError("Published dev questions differ from the run input")
    documents = read_jsonl(args.corpus)
    source_answers = args.run / "answers.jsonl"
    source_trace = args.run / "trace.jsonl"
    answers, traces = read_jsonl(source_answers), read_jsonl(source_trace)
    for records in (answers, traces):
        if len(records) != len(questions) or {
            row["id"] for row in records
        } != set(questions):
            raise ValueError("Source run IDs do not match the questions")
    config_hash = sha256_file(ROOT / "config.json")
    corpus_hash = sha256_file(args.corpus)
    for answer in answers:
        normalize_answer(json.dumps(answer), questions[answer["id"]],
                         documents)
    compact = []
    for trace in traces:
        if trace["config_sha256"] != config_hash:
            raise ValueError("Run does not match the submission config")
        if trace["corpus_sha256"] != corpus_hash:
            raise ValueError("Run does not match the source corpus")
        if not trace["usage_verified"] or trace["limit_violation"]:
            raise ValueError("Unverified or over-budget run")
        fields = (
            "id", "llm_calls", "input_tokens", "output_tokens",
            "elapsed_ms", "retrieved_block_ids", "context_pages",
            "source_aliases", "status", "pipeline", "config_sha256",
            "corpus_sha256", "submission_id", "model", "runtime_status",
            "response_parser", "usage_verified", "limit_violation",
            "protocol_repairs", "original_validation_status",
            "original_validation_error", "source_trace_sha256",
            "additional_llm_calls",
        )
        row = {key: trace[key] for key in fields if key in trace}
        row["dataset_kind"] = "local_pilot_not_official_dev"
        row["export_kind"] = "compact_no_source_text_or_request_messages"
        row["calls"] = []
        for call in trace["calls"]:
            fields = (
                "stage", "estimated_prompt_tokens", "request_sha256",
                "input_tokens", "output_tokens", "elapsed_ms", "usage",
                "prompt_count_difference", "finish_reason",
                "endpoint_response_id", "system_fingerprint",
            )
            item = {key: call[key] for key in fields if key in call}
            request = call["request"]
            item["generation"] = {
                key: request[key] for key in (
                    "temperature", "max_tokens", "chat_template_kwargs",
                ) if key in request
            }
            row["calls"].append(item)
        compact.append(row)
    destination = ROOT / "dev_answers.jsonl"
    destination.write_bytes(source_answers.read_bytes())
    trace_path = ROOT / "dev_trace.jsonl"
    trace_path.write_text("".join(json_line(row) for row in compact))
    manifest = {
        "schema_version": 1,
        "dataset_kind": "local_pilot_not_official_dev",
        "author": "Уденеев Александр Владимирович",
        "group": "Б05-602",
        "source_questions_sha256": sha256_file(args.questions),
        "source_corpus_sha256": corpus_hash,
        "source_answers_sha256": sha256_file(source_answers),
        "source_full_trace_sha256": sha256_file(source_trace),
        "answer_records": len(answers),
        "llm_calls": sum(row["llm_calls"] for row in compact),
        "answers_modified_by_audit": False,
        "trace_export": (
            "All original metered calls retained; request messages, source "
            "text and raw model responses remain in the ignored full trace."
        ),
        "data_preparation": "tools/prepare_data.py",
        "runtime_configuration": "config.json; runtime.json is not read",
        "gold_scope": "answer_sets_and_reference_pages_not_citation_f1",
        "audit_scope": "manual_source_and_math_review_not_official_scoring",
        "files_sha256": {
            name: sha256_file(ROOT / name) for name in (
                "config.json", "dev_answers.jsonl", "dev_trace.jsonl",
                "dev_questions.jsonl", "dev_gold.jsonl",
                "own_questions.jsonl", "own_gold.jsonl",
                "docs/citation_audit.jsonl", "docs/citation_audit.md",
                "report.md",
                "experiments.csv", "README.md", "requirements.txt",
                "requirements-pdf.txt", "tools/prepare_data.py",
                "tools/prepare_pilot_questions.py", "docs/reproduction.md",
                "docs/own_questions_notes.md",
            )
        },
    }
    write_json(ROOT / "submission_manifest.json", manifest)
    print(f"Exported {len(answers)} unchanged pilot answers and compact trace")


if __name__ == "__main__":
    main()
