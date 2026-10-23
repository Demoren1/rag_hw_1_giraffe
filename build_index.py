"""Build an offline BM25 or hybrid index with the required model assets."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

from rag_baseline import (
    INDEX_VERSION,
    json_line,
    load_config,
    make_sparse_index,
    read_jsonl,
    sha256_file,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    output = args.output.resolve()
    corpus = args.corpus.resolve()
    if corpus == output or corpus.is_relative_to(output):
        raise ValueError("Source corpus must be outside the index directory")
    documents = []
    seen = set()
    for row in read_jsonl(corpus):
        block_id = row.get("block_id")
        if not isinstance(block_id, str) or not block_id or block_id in seen:
            raise ValueError("Corpus block IDs must be nonempty and unique")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"Empty text: {block_id}")
        if type(row.get("page")) is not int or row["page"] < 1:
            raise ValueError(f"Expected a 1-based PDF page: {block_id}")
        if not isinstance(row.get("section"), str):
            raise ValueError(f"Expected section text: {block_id}")
        seen.add(block_id)
        documents.append({
            key: row[key] for key in ("block_id", "page", "section", "text")
        })
    if not documents:
        raise ValueError("Corpus is empty")
    output.mkdir(parents=True, exist_ok=True)
    sparse = make_sparse_index(
        documents, config["retrieval"]["section_copies"]
    )
    with (output / "documents.jsonl").open("w", encoding="utf-8") as stream:
        stream.writelines(json_line(row) for row in documents)
    write_json(output / "bm25.json", sparse)

    # Tokenizer assets only: no model weights, embeddings, or LLM calls.
    from transformers import AutoTokenizer

    settings = config["tokenizer"]
    local = args.config.resolve().parent / settings["local_source"]
    if local.is_dir():
        tokenizer = AutoTokenizer.from_pretrained(
            local, local_files_only=True, trust_remote_code=False
        )
        manifest_path = local / "download_manifest.json"
        if manifest_path.is_file():
            local_manifest = json.loads(manifest_path.read_text())
            if local_manifest.get("revision") != settings["revision"]:
                raise ValueError("Local tokenizer model revision mismatch")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            settings["hub_id"], revision=settings["revision"],
            cache_dir=output / "download_cache", trust_remote_code=False,
        )
    tokenizer.save_pretrained(output / "tokenizer")
    dense_metadata = None
    if config.get("pipeline") == "hybrid_evidence":
        from hybrid_retrieval import build_dense_index

        dense_metadata = build_dense_index(
            output, documents, config, args.config
        )
    artifact_hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and (
            path.parent == output / "tokenizer"
            or path.name in {"bm25.json", "documents.jsonl"}
            or (
                dense_metadata is not None and ".cache" not in path.parts
                and (
                    path.is_relative_to(output / "encoder")
                    or path.is_relative_to(output / "lancedb")
                    or path.name == "dense_manifest.json"
                )
            )
        )
    }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest = {
        "index_version": INDEX_VERSION,
        "corpus_sha256": sha256_file(corpus),
        "blocks": len(documents),
        "pages": len({row["page"] for row in documents}),
        "index_section_copies": config["retrieval"]["section_copies"],
        "tokenizer": settings,
        "artifact_sha256": artifact_hashes,
        "build_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "retrieval_device": "cpu",
        "llm_calls": 0,
        "chunking": "one intact original block per document",
    }
    if dense_metadata is not None:
        manifest["dense"] = dense_metadata
    write_json(output / "manifest.json", manifest)
    manifest["index_bytes"] = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    write_json(output / "manifest.json", manifest)
    print(
        f"Built {len(documents)} blocks in {manifest['build_seconds']}s; "
        f"index: {output}", file=sys.stderr,
    )


if __name__ == "__main__":
    main()
