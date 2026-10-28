"""Rebuild the local corpus and open questions from the supplied PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    """Hash an artifact without loading glyph geometry into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, default=ROOT)
    parser.add_argument("--check-reference", action="store_true")
    args = parser.parse_args()
    pdf = args.pdf.expanduser().resolve()
    output = args.output.expanduser().resolve()
    dev_output = args.dev_output.expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    source = ROOT / "materials" / "tfkp_15_qa.md"
    pilot = output / "pilot"
    glyphs = output / "font_analysis"
    corpus = output / "corpus_v2"
    steps = (
        ("prepare_pilot_corpus.py", [
            "--pdf", str(pdf), "--output", str(pilot),
        ]),
        ("prepare_pilot_questions.py", [
            "--source", str(source), "--output", str(pilot),
            "--dev-output", str(dev_output),
        ]),
        ("pdf_font_decoder.py", [
            "--pdf", str(pdf), "--type3-encoding", "cm-super-t2a",
            "--output-dir", str(glyphs),
        ]),
        ("build_pdf_corpus.py", [
            "--pdf", str(pdf), "--glyphs", str(glyphs / "glyph_pages.jsonl"),
            "--old-corpus", str(pilot / "pilot_corpus.jsonl"),
            "--output", str(corpus),
        ]),
    )
    started = time.perf_counter()
    for name, arguments in steps:
        print(f"Preparing data: {name}", flush=True)
        subprocess.run([
            sys.executable, str(ROOT / "tools" / name), *arguments,
        ], cwd=ROOT, check=True)
    paths = {
        "pdf": pdf, "corpus": corpus / "corpus.jsonl",
        "questions": dev_output / "dev_questions.jsonl",
        "gold": dev_output / "dev_gold.jsonl",
    }
    digests = {name: sha256_file(path) for name, path in paths.items()}
    reference_matches = None
    if args.check_reference:
        reference = json.loads((ROOT / "submission_manifest.json").read_text())
        reference_matches = {
            "corpus": digests["corpus"] == reference["source_corpus_sha256"],
            "questions": digests["questions"] == reference[
                "source_questions_sha256"
            ],
            "gold": digests["gold"] == reference["files_sha256"][
                "dev_gold.jsonl"
            ],
        }
    result = {
        "dataset_kind": "local_pilot_not_official_dev",
        "llm_calls": 0,
        "seconds": round(time.perf_counter() - started, 3),
        "files": {name: str(path) for name, path in paths.items()},
        "sha256": digests,
        "reference_matches": reference_matches,
        "gold_scope": "answer_sets_and_reference_pages_not_citation_f1",
    }
    record = output / "preparation_manifest.json"
    record.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if reference_matches is not None and not all(reference_matches.values()):
        raise ValueError(f"Artifacts differ from the reference: {record}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
