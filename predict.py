"""Run the configured RAG pipeline with bounded, metered Qwen calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from rag_baseline import (
    RESPONSE_PARSER_VERSION,
    SparseSearch,
    json_line,
    load_config,
    normalize_answer,
    pack_context,
    question_records,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("pipeline") == "hybrid_evidence":
        from hybrid_pipeline import run_hybrid

        run_hybrid(args, config)
        return
    questions = question_records(args.questions)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        questions = questions[:args.limit]
    reserved = {args.questions.resolve(), args.config.resolve()}
    if args.output.resolve() == args.trace.resolve():
        raise ValueError("Answers and trace must be separate files")
    for path in (args.output, args.trace):
        if path.resolve() in reserved or path.resolve().is_relative_to(
            args.index.resolve()
        ):
            raise ValueError("Outputs must not overwrite inputs/index")
        path.parent.mkdir(parents=True, exist_ok=True)

    base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    model = os.environ.get("LLM_MODEL", "")
    submission = os.environ.get("SUBMISSION_ID", "")
    endpoint = urllib.parse.urlsplit(base)
    if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
        raise ValueError("Set an HTTP(S) LLM_BASE_URL")
    if endpoint.username or endpoint.password or endpoint.query:
        raise ValueError("Pass API credentials through LLM_API_KEY")
    if not model or not submission:
        raise ValueError("Set LLM_MODEL and SUBMISSION_ID")

    search = SparseSearch(args.index, config)
    if search.manifest["index_section_copies"] != config["retrieval"][
        "section_copies"
    ]:
        raise ValueError("Index settings changed; rebuild the index")
    if search.manifest["tokenizer"]["revision"] != config["tokenizer"][
        "revision"
    ]:
        raise ValueError("Tokenizer revision changed; rebuild the index")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.index / "tokenizer", local_files_only=True,
        trust_remote_code=False,
    )
    config_hash = sha256_file(args.config)
    with args.output.open("w", encoding="utf-8") as answers, args.trace.open(
        "w", encoding="utf-8"
    ) as traces:
        for question in questions:
            started = time.perf_counter()
            qid = question["id"]
            trace = {
                "id": qid, "llm_calls": 0, "input_tokens": 0,
                "output_tokens": 0, "retrieved_block_ids": [],
                "config_sha256": config_hash,
                "corpus_sha256": search.manifest["corpus_sha256"],
                "submission_id": submission, "model": model,
                "runtime_status": config["runtime_status"],
                "usage_verified": True, "status": "ok",
                "response_parser": RESPONSE_PARSER_VERSION,
            }
            try:
                candidates = search.search(question)
                trace["candidates"] = [
                    {"block_id": item["document"]["block_id"],
                     "page": item["document"]["page"],
                     "score": round(item["score"], 6)}
                    for item in candidates
                ]
                documents, messages, estimated = pack_context(
                    question, candidates, tokenizer, config
                )
                trace["retrieved_block_ids"] = [
                    doc["block_id"] for doc in documents
                ]
                trace["estimated_prompt_tokens"] = estimated
                trace["context_pages"] = [doc["page"] for doc in documents]
                if not documents:
                    raise ValueError("No source block fits the context budget")
                payload = {
                    "model": model, "messages": messages, "stream": False,
                    **config["generation"],
                }
                request_bytes = json.dumps(
                    payload, ensure_ascii=False
                ).encode()
                trace["request_sha256"] = hashlib.sha256(
                    request_bytes
                ).hexdigest()
                trace["request"] = payload
                request = urllib.request.Request(
                    base + "/chat/completions", data=request_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + os.environ.get(
                            "LLM_API_KEY", ""
                        ),
                        "X-Question-ID": qid,
                        "X-Submission-ID": submission,
                    }, method="POST",
                )
                # No automatic retries: every attempted call is accounted.
                trace["llm_calls"] = 1
                trace["usage_verified"] = False
                trace["input_tokens"] = None
                trace["output_tokens"] = None
                with urllib.request.urlopen(
                    request, timeout=config["http"]["timeout_seconds"]
                ) as response:
                    result = json.load(response)
                usage = result.get("usage", {})
                trace["usage"] = usage
                trace["endpoint_response_id"] = result.get("id")
                trace["system_fingerprint"] = result.get("system_fingerprint")
                for api_name, name in (
                    ("prompt_tokens", "input_tokens"),
                    ("completion_tokens", "output_tokens"),
                ):
                    value = usage.get(api_name)
                    if type(value) is not int or value < 0:
                        raise ValueError("Endpoint did not return token usage")
                    trace[name] = value
                trace["usage_verified"] = True
                trace["prompt_count_difference"] = (
                    trace["input_tokens"] - estimated
                )
                choice = result["choices"][0]
                trace["finish_reason"] = choice.get("finish_reason")
                text = choice["message"].get("content") or ""
                trace["response_text"] = text
                limits = config["limits"]
                violation = (
                    trace["input_tokens"] > limits["total_input_tokens"]
                    or trace["output_tokens"] > limits["total_output_tokens"]
                    or trace["input_tokens"] + config["generation"][
                        "max_tokens"
                    ] > limits["request_total_tokens"]
                    or trace["output_tokens"] > config["generation"][
                        "max_tokens"
                    ]
                )
                trace["limit_violation"] = violation
                if violation:
                    raise ValueError("Endpoint token usage exceeded a budget")
                if trace["finish_reason"] == "length":
                    raise ValueError("Generation exhausted its output budget")
                repairs: list[str] = []
                answer = normalize_answer(
                    text, question, documents,
                    canonicalize=True, repairs=repairs,
                )
                trace["format_repairs"] = repairs
            except (
                ValueError, KeyError, IndexError, TypeError, OSError,
                urllib.error.URLError,
            ) as error:
                trace["status"] = "error"
                trace["error"] = f"{type(error).__name__}: {error}"[:1000]
                answer = {
                    "id": qid, "answer_ids": [], "abstain": True,
                    "rationale": (
                        "Технический отказ: корректный подтверждённый ответ "
                        "не получен. Подробности сохранены в trace."
                    ), "citations": [],
                }
            trace["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            answers.write(json_line(answer))
            traces.write(json_line(trace))
            answers.flush()
            traces.flush()
            print(
                f"{qid}: {trace['status']}, {answer['answer_ids']}, "
                f"tokens={trace['input_tokens']}/{trace['output_tokens']}, "
                f"{trace['elapsed_ms'] / 1000:.1f}s", file=sys.stderr,
            )


if __name__ == "__main__":
    main()
