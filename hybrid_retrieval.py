"""CPU EmbeddingGemma, local LanceDB, rank fusion, and source neighbors."""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rag_baseline import SparseSearch, sha256_file, write_json


class GemmaEncoder:
    """Use the published sentence_embedding ONNX output on CPU only."""

    def __init__(self, path: Path, settings: dict[str, Any]):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if settings["provider"] != "CPUExecutionProvider":
            raise ValueError("Homework retrieval must run on CPU")
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings["threads"]
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path / settings["model_file"]), sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
        )
        self.settings = settings
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]
        if "sentence_embedding" not in self.output_names:
            raise ValueError("Expected the exported projection/pooling output")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed prompted strings with the pretrained dense projection."""
        batch = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.settings["max_length"], return_tensors="np",
        )
        inputs = {
            name: np.asarray(batch[name], dtype=np.int64)
            for name in self.inputs
        }
        vectors = self.session.run(["sentence_embedding"], inputs)[0]
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(texts), self.settings["dimensions"]):
            raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("Encoder returned nonfinite vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("Encoder returned a zero vector")
        return vectors / norms

    def document_windows(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Split only the embedding view; keep original cited blocks intact."""
        text = re.sub(r"\s+", " ", doc["text"]).strip()
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        length = self.settings["window_tokens"]
        stride = length - self.settings["overlap_tokens"]
        if not 0 < stride <= length < self.settings["max_length"] - 100:
            raise ValueError("Invalid embedding window/overlap")
        windows = []
        for start in range(0, len(ids), stride):
            chunk = ids[start:start + length]
            content = self.tokenizer.decode(chunk, skip_special_tokens=True)
            title = doc["section"] or "none"
            windows.append({
                "start_token": start, "end_token": start + len(chunk),
                "prompt": f"title: {title} | text: {content}",
            })
            if start + length >= len(ids):
                break
        return windows


def build_dense_index(
    output: Path, documents: list[dict[str, Any]],
    config: dict[str, Any], config_path: Path,
) -> dict[str, Any]:
    """Bundle the fixed encoder and build a small exact-search Lance table."""
    import lancedb
    import pyarrow as pa
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    settings = config["embedding"]
    source = config_path.resolve().parent / settings["local_source"]
    destination = output / "encoder"
    model_file = settings["model_file"]
    files = [
        model_file, model_file + "_data", "config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "README.md",
    ]
    if source.is_dir():
        source_manifest = json.loads(
            (source / "download_manifest.json").read_text()
        )
        if source_manifest["revision"] != settings["revision"]:
            raise ValueError("Embedding model revision mismatch")
        for name in files:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
    else:
        snapshot_download(
            settings["hub_id"], revision=settings["revision"],
            local_dir=destination, allow_patterns=files,
        )
    encoder = GemmaEncoder(destination, settings)
    windows = []
    for index, doc in enumerate(documents):
        for number, window in enumerate(encoder.document_windows(doc)):
            windows.append({
                **window, "doc_index": index, "chunk_index": number,
            })
    rows = []
    batch_size = settings["batch_size"]
    for start in range(0, len(windows), batch_size):
        batch = windows[start:start + batch_size]
        vectors = encoder.encode([item["prompt"] for item in batch])
        for window, vector in zip(batch, vectors, strict=True):
            doc = documents[window["doc_index"]]
            rows.append({
                "block_id": doc["block_id"], "page": doc["page"],
                "section": doc["section"], "text": doc["text"],
                "doc_index": window["doc_index"],
                "chunk_index": window["chunk_index"],
                "start_token": window["start_token"],
                "end_token": window["end_token"],
                "vector": vector.tolist(),
            })
        if start % (batch_size * 25) == 0:
            print(
                f"Embedding {min(start + batch_size, len(windows))}/"
                f"{len(windows)}", file=sys.stderr,
            )
    schema = pa.schema([
        pa.field("block_id", pa.string()), pa.field("page", pa.int32()),
        pa.field("section", pa.string()), pa.field("text", pa.string()),
        pa.field("doc_index", pa.int32()),
        pa.field("chunk_index", pa.int32()),
        pa.field("start_token", pa.int32()), pa.field("end_token", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), settings["dimensions"])),
    ])
    database = lancedb.connect(str(output / "lancedb"))
    table = database.create_table(
        "chunks", data=rows, schema=schema, mode="overwrite"
    )
    # Exact KNN is cheap at this corpus size and avoids ANN approximation.
    metadata = {
        "model": settings,
        "chunks": len(rows), "blocks": len(documents),
        "database_rows": table.count_rows(),
        "vector_search": "LanceDB exact dot product on normalized vectors",
        "seconds": round(time.perf_counter() - started, 3),
        "providers": encoder.session.get_providers(),
        "projection": "published ONNX sentence_embedding output",
    }
    write_json(output / "dense_manifest.json", metadata)
    return metadata


class HybridSearch:
    """Fuse sparse/dense ranks, then add nearby source blocks."""

    def __init__(
        self, path: Path, config: dict[str, Any],
        *, include_dense: bool = True,
    ):
        self.sparse = SparseSearch(path, config)
        self.documents = self.sparse.documents
        self.manifest = self.sparse.manifest
        self.include_dense = include_dense
        if include_dense:
            import lancedb

            dense = json.loads((path / "dense_manifest.json").read_text())
            if dense["model"] != config["embedding"]:
                raise ValueError("Encoder settings changed; rebuild index")
            self.encoder = GemmaEncoder(
                path / "encoder", config["embedding"],
            )
            self.database = lancedb.connect(str(path / "lancedb"))
            self.table = self.database.open_table("chunks")
            if self.table.count_rows() != dense["chunks"]:
                raise ValueError("LanceDB rows differ from the manifest")
        self.settings = config["retrieval"]
        self.by_id = {
            doc["block_id"]: index for index, doc in enumerate(self.documents)
        }

    def search(self, question: dict[str, Any]) -> list[dict[str, Any]]:
        sparse = self.sparse.search(question)
        channels = {
            "bm25": [item["document"]["block_id"] for item in sparse],
        }
        if self.include_dense:
            channels["dense"] = self.dense_order(question)
        return self.fuse_and_expand(channels)

    def dense_order(self, question: dict[str, Any]) -> list[str]:
        """Retrieve distinct source blocks from their embedding windows."""
        settings = self.settings
        query = question["question"] + "\n" + "\n".join(
            option["text"] for option in question["options"]
        )
        vector = self.encoder.encode([
            "task: search result | query: " + query
        ])[0]
        hits = self.table.search(vector.tolist()).distance_type("dot").limit(
            settings["dense_candidates"] * 3
        ).select(["block_id", "doc_index", "_distance"]).to_list()
        dense_order = []
        dense_seen = set()
        for hit in hits:
            bid = hit["block_id"]
            if bid not in dense_seen:
                dense_order.append(bid)
                dense_seen.add(bid)
            if len(dense_order) >= settings["dense_candidates"]:
                break
        return dense_order

    def fuse_and_expand(self, channels: dict[str, list[str]]) -> list[dict]:
        """Keep seed selection and source neighbors fixed across ablations."""
        settings = self.settings
        fusion: dict[str, float] = defaultdict(float)
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        for name, order in channels.items():
            for rank, bid in enumerate(order, 1):
                fusion[bid] += 1 / (settings["rrf_k"] + rank)
                ranks[bid][name] = rank
        ranked = sorted(fusion, key=lambda bid: (-fusion[bid], bid))
        page_counts: Counter = Counter()
        seeds = []
        for bid in ranked:
            doc = self.documents[self.by_id[bid]]
            if page_counts[doc["page"]] >= settings["max_seeds_per_page"]:
                continue
            seeds.append(bid)
            page_counts[doc["page"]] += 1
            if len(seeds) >= settings["seed_blocks"]:
                break
        ordered = list(seeds)
        reasons = {bid: "rrf_seed" for bid in seeds}
        if settings.get("neighbors", True):
            for bid in seeds:
                number = self.by_id[bid]
                doc = self.documents[number]
                for neighbor in (number - 1, number + 1):
                    if not 0 <= neighbor < len(self.documents):
                        continue
                    other = self.documents[neighbor]
                    if (
                        other["page"] == doc["page"]
                        and other["section"] == doc["section"]
                        and other["block_id"] not in reasons
                    ):
                        ordered.append(other["block_id"])
                        reasons[other["block_id"]] = "source_neighbor"
        for bid in ranked:
            if bid not in reasons:
                ordered.append(bid)
                reasons[bid] = "rrf_reserve"
        return [
            {"document": self.documents[self.by_id[bid]],
             "score": fusion.get(bid, 0), "ranks": ranks.get(bid, {}),
             "reason": reasons[bid]}
            for bid in ordered
        ]
