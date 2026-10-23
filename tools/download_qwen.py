"""Download public Qwen weights into the ignored local model directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "models" / "Qwen3.5-4B",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    info = HfApi(token=False).model_info(
        MODEL_ID, revision=args.revision, files_metadata=True,
    )
    revision = info.sha
    if not revision:
        raise RuntimeError("Hugging Face did not return a model revision")
    size_bytes = sum(item.size or 0 for item in info.siblings or [])
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Revision: {revision}", flush=True)
    print(f"Repository size: {size_bytes / 1e9:.2f} GB", flush=True)
    print(f"Destination: {output}", flush=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=revision,
        local_dir=output,
        max_workers=4,
        token=False,
    )
    manifest = {
        "model_id": MODEL_ID,
        "revision": revision,
        "repository_size_bytes": size_bytes,
        "local_path": str(output),
        "source": f"https://huggingface.co/{MODEL_ID}/tree/{revision}",
    }
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Download complete.", flush=True)


if __name__ == "__main__":
    main()
