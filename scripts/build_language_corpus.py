#!/usr/bin/env python3
"""Materialize the controlled-English corpus and its quality checks."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from flystudy.language_corpus import (
    QA_FAMILIES,
    REPORT_FAMILIES,
    TEXT_FAMILIES,
    audit_corpus,
    event_channel,
    make_record,
)
from flystudy.language_audit import audit_files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload(row) -> dict:
    return {
        "id": row.record_id,
        "split": row.split,
        "mode": row.mode,
        "templateFamily": row.template_family,
        "variant": row.variant,
        "scene": [
            {
                "channel": event_channel(event),
                "location": event.location,
                "kind": event.kind,
                "level": event.level,
                "duration": event.duration,
                "speed": event.speed,
            }
            for event in row.scene.events
        ],
        "pairFamilies": [list(pair) for pair in row.pair_families],
        "promptTokens": list(row.prompt_words),
        "answerTokens": list(row.answer_words),
        "prompt": row.prompt,
        "answer": row.answer,
    }


def write_split(path: Path, *, split: str, examples: int, seed: int) -> dict:
    temporary = path.with_suffix(path.suffix + ".tmp")
    modes, templates = Counter(), Counter()
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for record_id in range(examples):
                    row = make_record(record_id, split=split, seed=seed)
                    modes[row.mode] += 1
                    templates[row.template_family] += 1
                    text.write(json.dumps(payload(row), ensure_ascii=False, sort_keys=True))
                    text.write("\n")
    os.replace(temporary, path)
    return {
        "path": path.name,
        "examples": examples,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "modeCounts": dict(sorted(modes.items())),
        "templateCounts": dict(sorted(templates.items())),
    }


def write_unique_bank(source: Path, destination: Path) -> dict:
    """Remove duplicate conditional records while preserving first occurrence."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    seen: set[bytes] = set()
    modes, templates = Counter(), Counter()
    rows = 0
    with gzip.open(source, "rt", encoding="utf-8") as incoming:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as outgoing:
                    for line in incoming:
                        row = json.loads(line)
                        scene = row["scene"] if row["mode"] != "text" else []
                        canonical = json.dumps(
                            [scene, row["promptTokens"], row["answerTokens"]],
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                        key = hashlib.blake2b(canonical, digest_size=16).digest()
                        if key in seen:
                            continue
                        seen.add(key)
                        modes[row["mode"]] += 1
                        templates[row["templateFamily"]] += 1
                        outgoing.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                        outgoing.write("\n")
                        rows += 1
    os.replace(temporary, destination)
    return {
        "path": destination.name,
        "examples": rows,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "modeCounts": dict(sorted(modes.items())),
        "templateCounts": dict(sorted(templates.items())),
        "deduplicationKey": "sensory scene + prompt + target; text records use prompt + target",
    }


def write_samples(path: Path, *, seed: int) -> None:
    wanted = {family: 3 for family in REPORT_FAMILIES + QA_FAMILIES + TEXT_FAMILIES}
    found = {family: [] for family in wanted}
    for record_id in range(200_000):
        row = make_record(record_id, seed=seed)
        bucket = found.get(row.template_family)
        if bucket is not None and len(bucket) < wanted[row.template_family]:
            bucket.append(row)
        if all(len(found[family]) == count for family, count in wanted.items()):
            break
    if not all(len(found[family]) == count for family, count in wanted.items()):
        raise RuntimeError("could not collect a representative sample for every family")
    lines = [
        "# Controlled-English sentence samples",
        "",
        "All examples use the fixed 100-word lexicon. Meanings such as taste=food and pressure=pain/danger are authored task conventions.",
        "",
    ]
    for family in REPORT_FAMILIES + QA_FAMILIES + TEXT_FAMILIES:
        lines.extend((f"## {family}", ""))
        for row in found[family]:
            lines.append(f"- **Input:** {row.prompt} **Target:** {row.answer}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/language-corpus"))
    parser.add_argument("--train-examples", type=int, default=200_000)
    parser.add_argument("--validation-examples", type=int, default=4_096)
    parser.add_argument("--test-examples", type=int, default=4_096)
    parser.add_argument("--audit-examples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.train_examples, args.validation_examples, args.test_examples,
           args.audit_examples) < 1:
        raise ValueError("all corpus sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.output_dir / "train.jsonl.gz",
        "trainUnique": args.output_dir / "train-unique.jsonl.gz",
        "validationId": args.output_dir / "validation-id.jsonl.gz",
        "validationComposition": args.output_dir / "validation-composition.jsonl.gz",
        "testComposition": args.output_dir / "test-composition.jsonl.gz",
        "audit": args.output_dir / "audit.json",
        "manifest": args.output_dir / "manifest.json",
        "samples": args.output_dir / "samples.md",
        "materializedAudit": args.output_dir / "materialized-audit.json",
    }
    expected = list(paths.values())
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite: {[str(path) for path in existing]}")

    files = {
        "train": write_split(paths["train"], split="train", examples=args.train_examples, seed=args.seed),
        "validationId": write_split(paths["validationId"], split="validation_id",
                                    examples=args.validation_examples, seed=args.seed),
        "validationComposition": write_split(paths["validationComposition"], split="validation_composition",
                                             examples=args.validation_examples, seed=args.seed),
        "testComposition": write_split(paths["testComposition"], split="test_composition",
                                       examples=args.test_examples, seed=args.seed),
    }
    files["trainUnique"] = write_unique_bank(paths["train"], paths["trainUnique"])
    audit = audit_corpus(examples=args.audit_examples, seed=args.seed)
    paths["audit"].write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                              encoding="utf-8")
    write_samples(paths["samples"], seed=args.seed)
    materialized = audit_files(args.output_dir)
    paths["materializedAudit"].write_text(json.dumps(materialized, indent=2, sort_keys=True) + "\n")
    builder = Path(__file__).resolve()
    project_root = builder.parents[1]
    source = project_root / "src/flystudy/language_corpus.py"
    readme = args.output_dir / "README.md"
    manifest = {
        "schema": 7,
        "revision": 2,
        "createdFor": "fixed-connectome grounded compositional language experiment",
        "seed": args.seed,
        "files": files,
        "audit": {"path": paths["audit"].name, "sha256": sha256(paths["audit"])},
        "materializedAudit": {"path": paths["materializedAudit"].name, "sha256": sha256(paths["materializedAudit"])},
        "samples": {"path": paths["samples"].name, "sha256": sha256(paths["samples"])},
        "generator": {"path": source.relative_to(project_root).as_posix(), "sha256": sha256(source)},
        "builder": {"path": builder.relative_to(project_root).as_posix(), "sha256": sha256(builder)},
        "claimScope": "controlled 100-word English, not unrestricted natural language",
    }
    if readme.exists():
        manifest["documentation"] = {"path": readme.name, "sha256": sha256(readme)}
    snapshot = args.output_dir / "source-snapshot"
    snapshot.mkdir(exist_ok=True)
    manifest["sourceSnapshot"] = {}
    for path in (source, builder, project_root / "src/flystudy/language_lexicon.py",
                 project_root / "src/flystudy/language_audit.py",
                 project_root / "src/flystudy/language_metrics.py",
                 project_root / "src/flystudy/language_runtime.py"):
        destination = snapshot / path.name
        shutil.copy2(path, destination)
        manifest["sourceSnapshot"][path.name] = sha256(destination)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
