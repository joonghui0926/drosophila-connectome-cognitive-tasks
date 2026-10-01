"""Create a separately owned graph null without modifying the source dataset."""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from flystudy.graph import sha256_file
from flystudy.language_support import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/graph"))
    parser.add_argument("--output", type=Path, default=Path("data/graph-rewired-20260913"))
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--swaps-per-edge", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to replace an existing null graph")
    if args.swaps_per_edge < 1:
        raise ValueError("swaps per edge must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    if sha256_file(args.source / "graph.npz") != source_manifest["graph_sha256"]:
        raise ValueError("source graph checksum mismatch")
    with np.load(args.source / "graph.npz") as archive:
        original = dict(archive)
    rows, cols = original["rows"].astype(np.int32), original["cols"].astype(np.int32)
    n = source_manifest["neurons"]
    staging = Path(tempfile.mkdtemp(
        prefix=args.output.name + ".building-", dir=args.output.parent))
    try:
        with tempfile.TemporaryDirectory(prefix="flystudy-rewire-") as directory:
            build = Path(directory)
            shutil.copy2(Path(__file__).with_name("rewire_directed.cpp"), build)
            executable = "rewire.exe" if os.name == "nt" else "rewire"
            # Relative filenames also work with MinGW in non-ASCII directories.
            compiler_flags = ["-O3", "-std=c++17"]
            if os.name == "nt":
                compiler_flags.append("-save-temps=obj")
            subprocess.run(["g++", *compiler_flags, "rewire_directed.cpp",
                            "-o", executable], cwd=build, check=True)
            rows.tofile(build / "rows.bin")
            cols.tofile(build / "cols.bin")
            process = subprocess.run([
                str(build / executable), "rows.bin", "cols.bin", "rewired.bin",
                str(args.seed), str(args.swaps_per_edge),
            ], cwd=build, check=True, text=True, capture_output=True)
            swaps = json.loads(process.stdout)
            new_cols = np.fromfile(build / "rewired.bin", dtype=np.int32)
        if not np.array_equal(
                np.bincount(cols, minlength=n), np.bincount(new_cols, minlength=n)):
            raise ValueError("rewiring changed directed out-degrees")
        if not np.array_equal(rows[rows == cols], rows[rows == new_cols]):
            raise ValueError("rewiring changed the fixed self-loop set")
        matrix = csr_matrix((original["counts"], new_cols, original["crow"]), shape=(n, n))
        matrix.sort_indices()
        new_cols = matrix.indices.astype(cols.dtype)
        old_keys = (rows.astype(np.uint64) << 32) | cols.astype(np.uint64)
        new_keys = (rows.astype(np.uint64) << 32) | new_cols.astype(np.uint64)
        if len(np.unique(new_keys)) != len(new_keys):
            raise ValueError("rewiring created duplicate directed edges")
        if not np.array_equal(np.bincount(rows, minlength=n), np.diff(matrix.indptr)):
            raise ValueError("rewiring changed directed in-degrees")
        overlap = len(np.intersect1d(old_keys, new_keys, assume_unique=True)) / len(rows)
        original.update(cols=new_cols.astype(original["cols"].dtype), counts=matrix.data)
        np.savez_compressed(staging / "graph.npz", **original)
        shutil.copy2(args.source / "nodes.feather", staging / "nodes.feather")
        manifest = {
            **source_manifest,
            "graph_sha256": sha256_file(staging / "graph.npz"),
            "sourceGraphSha256": source_manifest["graph_sha256"],
            "control": "directed-degree-preserving-rewire",
            "rewiring": {
                **swaps,
                "seed": args.seed,
                "swapsPerEdge": args.swaps_per_edge,
                "edgeOverlapFraction": overlap,
                "inOutDegreesPreserved": True,
                "duplicates": 0,
                "selfLoops": int(np.sum(rows == cols)),
                "selfLoopPolicy": "existing self-loops fixed; no new self-loops",
                "mixingNote": (
                    "finite double-edge-swap null; not claimed to be a uniform random graph"
                ),
            },
        }
        atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, args.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(manifest["rewiring"]), flush=True)


if __name__ == "__main__":
    main()
