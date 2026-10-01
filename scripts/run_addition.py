#!/usr/bin/env python3
"""Run bounded-addition fits on one or more GPUs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


SEEDS = (11, 22, 33, 44)


def command(args, seed: int, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "flystudy.cli",
        "train-addition",
        "--graph-dir",
        str(args.graph_dir),
        "--dataset",
        str(args.dataset),
        "--config",
        str(args.config),
        "--run-dir",
        str(run_dir),
        "--device",
        "cuda",
        "--max-epochs",
        str(args.max_epochs),
        "--min-epochs",
        str(args.min_epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(seed),
    ]


def run(args) -> None:
    if not args.gpus or len(args.gpus) != len(set(args.gpus)) or min(args.gpus) < 0:
        raise ValueError("--gpus must contain unique nonnegative indices")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        raise ValueError("--seeds must contain unique nonnegative integers")
    for seed in args.seeds:
        run_dir = args.output / f"seed-{seed}"
        if args.resume:
            checkpoint = run_dir / "checkpoint-latest.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
        elif run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise FileExistsError(
                f"run directory is not empty: {run_dir}; use --resume or a new directory"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    pending = list(args.seeds)
    while pending:
        wave = pending[: len(args.gpus)]
        pending = pending[len(args.gpus) :]
        processes = []
        try:
            for gpu, seed in zip(args.gpus, wave, strict=False):
                run_dir = args.output / f"seed-{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                log = (run_dir / "train.log").open("a" if args.resume else "x", encoding="utf-8")
                try:
                    environment = os.environ.copy()
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    launch = command(args, seed, run_dir)
                    if args.resume:
                        launch.append("--resume")
                    process = subprocess.Popen(
                        launch,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log.close()
                    raise
                processes.append((seed, process, log))
            failures = []
            for seed, process, _ in processes:
                return_code = process.wait()
                if return_code:
                    failures.append((seed, return_code))
            if failures:
                raise RuntimeError(f"addition fits failed: {failures}")
        finally:
            for _, process, _ in processes:
                if process.poll() is None:
                    process.terminate()
            for _, process, log in processes:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                finally:
                    log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("data/processed/malecns-v1.0-annotated"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/bounded-arithmetic-split-v1.json"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/addition.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/addition-anatomical"))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
