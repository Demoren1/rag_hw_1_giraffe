"""Download the pinned EmbeddingGemma ONNX assets to the ignored model dir."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import snapshot_download

from rag_baseline import load_config, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    settings = load_config(args.config)["embedding"]
    destination = (
        args.config.resolve().parent / settings["local_source"]
    )
    model = settings["model_file"]
    files = [
        model, model + "_data", "config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "README.md",
    ]
    snapshot_download(
        settings["hub_id"], revision=settings["revision"],
        local_dir=destination, allow_patterns=files,
    )
    write_json(destination / "download_manifest.json", {
        "hub_id": settings["hub_id"], "revision": settings["revision"],
        "files": {
            name: {"bytes": (destination / name).stat().st_size,
                   "sha256": sha256_file(destination / name)}
            for name in files
        },
    })
    print(f"Encoder ready: {destination}")


if __name__ == "__main__":
    main()
