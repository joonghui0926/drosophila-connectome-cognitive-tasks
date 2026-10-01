#!/usr/bin/env python3
"""Create or verify the twenty locked degree-preserving graph nulls."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from flystudy.language_support import atomic_json, utcnow
from flystudy.language_audit import audit_graphs
from flystudy.language_runtime import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/language_confirmatory.json"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--existing", type=Path, action="append", default=[],
                        help="reuse an audited graph for the corresponding first ensemble slot")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.workers not in (1, 2, 3, 4):
        raise ValueError("workers must be between one and four")
    config = json.loads(args.config.read_text())
    seeds = config.get("rewireSeeds", [])
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise ValueError("config must lock twenty unique rewire seeds")
    outputs = [args.output_root / f"rewire-{index:02d}" for index in range(1, 21)]
    if len(args.existing) > len(outputs):
        raise ValueError("more existing graphs were supplied than ensemble slots")
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    if file_hash(args.source / "graph.npz") != source_manifest.get("graph_sha256"):
        raise ValueError("source graph checksum does not match its manifest")
    source_nodes_sha = file_hash(args.source / "nodes.feather")

    def validate_rewire(path, seed):
        manifest_path = path / "manifest.json"
        graph_path = path / "graph.npz"
        if not manifest_path.is_file() or not graph_path.is_file():
            raise FileNotFoundError(f"rewire artifact is incomplete: {path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("sourceGraphSha256") != source_manifest["graph_sha256"]:
            raise ValueError(f"rewired graph has the wrong source: {path}")
        if manifest.get("control") != "directed-degree-preserving-rewire":
            raise ValueError(f"rewired graph has the wrong control type: {path}")
        rewiring = manifest.get("rewiring", {})
        if rewiring.get("seed") != seed or rewiring.get("swapsPerEdge") != 5:
            raise ValueError(f"rewired graph does not match its locked slot: {path}")
        if (manifest.get("neurons") != source_manifest.get("neurons")
                or manifest.get("edges") != source_manifest.get("edges")
                or not isinstance(rewiring.get("acceptedSwaps"), int)
                or rewiring["acceptedSwaps"] < 5 * source_manifest["edges"]):
            raise ValueError(f"rewired graph fails the locked 5E provenance: {path}")
        if file_hash(graph_path) != manifest.get("graph_sha256"):
            raise ValueError(f"rewired graph checksum mismatch: {path}")
        if file_hash(path / "nodes.feather") != source_nodes_sha:
            raise ValueError(f"rewired graph uses different neuron metadata: {path}")
        return manifest

    existing = []
    seen_hashes = set()
    for index, path in enumerate(args.existing):
        manifest = validate_rewire(path, seeds[index])
        if manifest.get("graph_sha256") in seen_hashes:
            raise ValueError("existing graph artifacts must be independent")
        seen_hashes.add(manifest.get("graph_sha256"))
        existing.append(str(path))
    plan = {
        "schema": 1,
        "createdAt": utcnow(),
        "source": str(args.source),
        "sourceManifestSha256": file_hash(args.source / "manifest.json"),
        "configSha256": file_hash(args.config),
        "swapsPerEdge": 5,
        "reusedGraphs": existing,
        "graphs": [{"index": index, "seed": seed, "output": str(output)}
                   for index, (seed, output) in enumerate(zip(seeds, outputs, strict=True), 1)],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "ensemble-plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        for key in ("sourceManifestSha256", "configSha256", "swapsPerEdge",
                    "reusedGraphs", "graphs"):
            if previous.get(key) != plan.get(key):
                raise ValueError("existing ensemble directory has a different locked plan")
    else:
        atomic_json(plan_path, plan)

    script = Path(__file__).with_name("prepare_language_rewire.py")

    for index, (path, output) in enumerate(zip(args.existing, outputs, strict=False)):
        if output.exists() or output.is_symlink():
            manifest = validate_rewire(output, seeds[index])
            if manifest["graph_sha256"] != json.loads(
                    (path / "manifest.json").read_text())["graph_sha256"]:
                raise FileExistsError(f"ensemble slot contains a different graph: {output}")
        else:
            staging = Path(tempfile.mkdtemp(
                prefix=output.name + ".building-", dir=args.output_root))
            try:
                for name in ("graph.npz", "nodes.feather", "manifest.json"):
                    shutil.copy2(path / name, staging / name)
                validate_rewire(staging, seeds[index])
                os.replace(staging, output)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)

    def create(seed, output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            validate_rewire(output, seed)
            return {"output": str(output), "status": "existing"}
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"incomplete graph directory requires manual inspection: {output}")
        subprocess.run([
            sys.executable, str(script),
            "--source", str(args.source),
            "--output", str(output),
            "--seed", str(seed),
            "--swaps-per-edge", "5",
        ], check=True)
        return {"output": str(output), "status": "created"}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(create, seed, output): output
                   for seed, output in zip(seeds[len(args.existing):],
                                           outputs[len(args.existing):], strict=True)}
        for future in as_completed(futures):
            results.append(future.result())
    results.extend({"output": str(output), "status": "reused"}
                   for output in outputs[:len(args.existing)])
    audit = audit_graphs(args.source, outputs)
    atomic_json(args.output_root / "ensemble-audit.json", audit)
    atomic_json(args.output_root / "ensemble-complete.json", {
        "at": utcnow(), "graphs": sorted(results, key=lambda row: row["output"]),
        "auditSha256": file_hash(args.output_root / "ensemble-audit.json"),
        "graphSha256s": [
            json.loads((output / "manifest.json").read_text())["graph_sha256"]
            for output in outputs
        ],
    })
    print(json.dumps({"graphs": len(results), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
