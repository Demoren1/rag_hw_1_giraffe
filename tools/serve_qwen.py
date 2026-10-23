"""Serve the downloaded Qwen model locally through MLX-LM."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--model", type=Path,
        default=PROJECT_ROOT / "models" / "Qwen3.5-4B",
    )
    args = parser.parse_args()
    model_path = args.model.resolve()
    if not (model_path / "download_manifest.json").is_file():
        raise SystemExit(
            "Download the model first with tools/download_qwen.py"
        )
    hf_home = PROJECT_ROOT / "local_data" / "huggingface"
    (hf_home / "hub").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(hf_home)
    command = [
        sys.executable, "-m", "mlx_lm.server",
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
