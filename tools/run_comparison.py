"""Run three fixed local-pilot comparisons sequentially on one endpoint."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_baseline import load_config, sha256_file, write_json  # noqa: E402


VARIANTS = (
    ("no_context", "config.no_context.json"),
    ("no_dense", "config.ablation_no_dense.json"),
    ("one_pass", "config.ablation_one_pass.json"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.index = args.index.resolve()
    args.questions = args.questions.resolve()
    args.gold = args.gold.resolve()
    if args.output.exists():
        raise ValueError("Choose a new output directory; keep existing runs")
    if any(
        path.is_relative_to(args.output)
        for path in (args.index, args.questions, args.gold)
    ) or args.output.is_relative_to(args.index):
        raise ValueError("Keep outputs separate from inputs and index")
    for _, filename in VARIANTS:
        load_config(ROOT / filename)
    args.output.mkdir(parents=True)
    record = {
        "status": "running",
        "questions_sha256": sha256_file(args.questions),
        "gold_sha256": sha256_file(args.gold),
        "index_manifest_sha256": sha256_file(args.index / "manifest.json"),
        "source_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "predict.py", "hybrid_pipeline.py", "hybrid_retrieval.py",
                "rag_baseline.py", "evaluate_pilot.py",
                "tools/run_comparison.py",
            )
        },
        "variants": [],
        "notes": (
            "Gold is passed only to the evaluator after generation. "
            "One sequential run per variant, no automatic retries. "
            "No-context answers deliberately have no source citations."
        ),
    }
    manifest = args.output / "comparison_manifest.json"
    write_json(manifest, record)
    for variant, filename in VARIANTS:
        config = ROOT / filename
        destination = args.output / variant
        destination.mkdir()
        submission = "rag-hw1-comparison-v1-" + variant
        env = {**os.environ, "SUBMISSION_ID": submission}
        entry = {
            "name": variant, "status": "running",
            "config": filename, "config_sha256": sha256_file(config),
            "submission_id": submission,
        }
        record["variants"].append(entry)
        write_json(manifest, record)
        print(f"Starting {variant}", flush=True)
        started = time.perf_counter()
        try:
            subprocess.run([
                sys.executable, str(ROOT / "predict.py"),
                "--index", str(args.index),
                "--questions", str(args.questions),
                "--output", str(destination / "answers.jsonl"),
                "--trace", str(destination / "trace.jsonl"),
                "--config", str(config),
            ], cwd=ROOT, env=env, check=True)
            entry["generation_seconds"] = round(
                time.perf_counter() - started, 3
            )
            subprocess.run([
                sys.executable, str(ROOT / "evaluate_pilot.py"),
                "--index", str(args.index),
                "--questions", str(args.questions),
                "--gold", str(args.gold),
                "--answers", str(destination / "answers.jsonl"),
                "--trace", str(destination / "trace.jsonl"),
                "--output", str(destination / "metrics.json"),
            ], cwd=ROOT, check=True)
        except subprocess.CalledProcessError:
            entry["status"] = "failed"
            record["status"] = "failed"
            write_json(manifest, record)
            raise
        entry["status"] = "complete"
        entry["metrics_sha256"] = sha256_file(destination / "metrics.json")
        write_json(manifest, record)
    record["status"] = "complete"
    write_json(manifest, record)


if __name__ == "__main__":
    main()
