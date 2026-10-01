from __future__ import annotations

import argparse
import json
from pathlib import Path

from flystudy.acquire import FILES, acquire_malecns
from flystudy.addition_train import run_addition_training
from flystudy.graph import prepare_graph


def _paths(raw_dir: Path) -> tuple[Path, Path]:
    return raw_dir / FILES["annotations"]["name"], raw_dir / FILES["connectivity"]["name"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="flystudy")
    subparsers = result.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download", help="download and verify the two required MaleCNS files")
    download.add_argument("--raw-dir", type=Path, default=Path("data/raw/malecns-v1.0"))

    prepare = subparsers.add_parser("prepare", help="build the selected whole-CNS sparse graph")
    prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw/malecns-v1.0"))
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed/malecns-v1.0-annotated"))
    prepare.add_argument("--node-policy", choices=("annotated", "traced"), default="annotated")

    addition = subparsers.add_parser(
        "train-addition", help="train exact without-replacement addition epochs with early stopping"
    )
    addition.add_argument("--graph-dir", type=Path, default=Path("data/processed/malecns-v1.0-annotated"))
    addition.add_argument("--dataset", type=Path, default=Path("data/bounded-arithmetic-split-v1.json"))
    addition.add_argument("--config", type=Path, default=Path("configs/addition.json"))
    addition.add_argument("--run-dir", type=Path, required=True)
    addition.add_argument("--max-epochs", type=int, default=100)
    addition.add_argument("--min-epochs", type=int, default=20)
    addition.add_argument("--patience", type=int, default=15)
    addition.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    addition.add_argument("--live-path", type=Path)
    addition.add_argument("--resume", action="store_true")
    addition.add_argument("--steps", type=int)
    addition.add_argument("--microbatch-size", type=int)
    addition.add_argument("--seed", type=int)
    addition.add_argument("--learning-rate", type=float)
    addition.add_argument("--ordinal-lambda", type=float)

    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "download":
        payload = acquire_malecns(args.raw_dir)
    elif args.command == "prepare":
        annotations, connectivity = _paths(args.raw_dir)
        payload = prepare_graph(annotations, connectivity, args.output_dir, node_policy=args.node_policy)
    else:
        overrides = {
            key: value
            for key, value in {
                "steps": args.steps,
                "readout_steps": min(4, args.steps) if args.steps else None,
                "microbatch_size": args.microbatch_size,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "ordinal_lambda": args.ordinal_lambda,
            }.items()
            if value is not None
        }
        payload = run_addition_training(
            graph_dir=args.graph_dir,
            dataset_path=args.dataset,
            config_path=args.config,
            run_dir=args.run_dir,
            max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,
            patience=args.patience,
            device_name=args.device,
            live_path=args.live_path,
            resume=args.resume,
            overrides=overrides,
        )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
