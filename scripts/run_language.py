#!/usr/bin/env python3
"""Run the 48-cell language campaign with fixed codebooks and token budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from flystudy.language_support import atomic_json, utcnow
from flystudy.language_runtime import file_hash


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import language_campaign as campaign  # noqa: E402


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema") != 1:
        raise ValueError("unsupported full-campaign config schema")
    campaign.configure_gpus(config.get("allowedPhysicalGpus", ()))
    if config.get("trainTokens") != 4_000_000:
        raise ValueError("the full confirmatory endpoint is locked at four million tokens")
    if config.get("warmupTokens") != 1_000_000:
        raise ValueError("the language curriculum uses a one-million-token report warmup")
    if len(config.get("interfaceSeeds", ())) != 4:
        raise ValueError("four interface placements are required")
    if config.get("canonicalInterfaceSeed") != config["interfaceSeeds"][0]:
        raise ValueError("the first interface seed must be the canonical placement")
    if config.get("rewireGraphs") != 20 or config.get("factorialRewireGraphs") != 4:
        raise ValueError("the locked design requires four factorial and twenty null rewires")
    if len(config.get("rewireSeeds", ())) != 20 or len(set(config["rewireSeeds"])) != 20:
        raise ValueError("twenty pre-registered independent rewire seeds are required")
    if not config.get("locking", {}).get("noPerformanceTuning"):
        raise ValueError("performance tuning must remain disabled")
    return config


def primary_cell(topology: str, interface_seed: int, graph: Path,
                 condition: str = "rewired") -> dict:
    return {
        "name": f"{topology}-i{interface_seed}-s11",
        "topology": topology,
        "condition": condition,
        "graph": graph,
        "interfaceSeed": int(interface_seed),
        "optimizationSeed": 11,
        "roles": [],
    }


def build_all_cells(real: Path, rewires: list[Path], config: dict) -> list[dict]:
    """Build the complete 48-cell design."""
    canonical = config["canonicalInterfaceSeed"]
    cells: list[dict] = []

    for interface_seed in config["interfaceSeeds"]:
        real_cell = primary_cell("real", interface_seed, real, condition="real")
        real_cell["roles"] = ["factorial"]
        if interface_seed == canonical:
            real_cell["roles"] += ["null-rank", "optimization-robustness"]
        cells.append(real_cell)
        for topology_index in range(1, 5):
            cell = primary_cell(
                f"rewire-{topology_index:02d}",
                interface_seed,
                rewires[topology_index - 1],
            )
            cell["roles"] = ["factorial"]
            if interface_seed == canonical:
                cell["roles"].append("null-rank")
                if topology_index <= 2:
                    cell["roles"].append("optimization-robustness")
            cells.append(cell)

    for topology_index in range(5, 21):
        cell = primary_cell(
            f"rewire-{topology_index:02d}",
            canonical,
            rewires[topology_index - 1],
        )
        cell["roles"] = ["null-rank"]
        cells.append(cell)

    for seed in (11, 22, 33, 44):
        cells.append({
            "name": f"dense-i{canonical}-s{seed}",
            "topology": "dense",
            "condition": "dense",
            "graph": real,
            "interfaceSeed": canonical,
            "optimizationSeed": seed,
            "roles": ["baseline"],
        })
    for seed in (11, 22):
        cells.append({
            "name": f"no-sensory-i{canonical}-s{seed}",
            "topology": "no-sensory",
            "condition": "no_sensory",
            "graph": real,
            "interfaceSeed": canonical,
            "optimizationSeed": seed,
            "roles": ["baseline"],
        })

    for topology, condition, graph in (
        ("real", "real", real),
        ("rewire-01", "rewired", rewires[0]),
        ("rewire-02", "rewired", rewires[1]),
    ):
        for seed in (22, config["additionalOptimizationSeed"]):
            cells.append({
                "name": f"{topology}-i{canonical}-s{seed}",
                "topology": topology,
                "condition": condition,
                "graph": graph,
                "interfaceSeed": canonical,
                "optimizationSeed": seed,
                "roles": ["optimization-robustness"],
            })

    names = [cell["name"] for cell in cells]
    if len(cells) != 48 or len(names) != len(set(names)):
        raise RuntimeError("the fresh campaign must contain 48 unique cells")
    return cells


def command_arguments(config: dict) -> list[str]:
    return campaign.model_arguments(config)


def training_command(cell: dict, run_dir: Path, corpus: Path, config: dict,
                     resume: bool) -> list[str]:
    command = [
        sys.executable, "-m", "flystudy.language_train",
        "--phase", "train",
        "--condition", cell["condition"],
        "--seed", str(cell["optimizationSeed"]),
        "--interface-seed", str(cell["interfaceSeed"]),
        "--graph-dir", str(cell["graph"]),
        "--corpus", str(corpus),
        "--run-dir", str(run_dir),
        "--train-tokens", str(config["trainTokens"]),
        "--warmup-tokens", str(config["warmupTokens"]),
        "--checkpoint-every-tokens", str(config["checkpointEveryTokens"]),
        "--pairs-per-family", str(config["evaluation"]["pairsPerFamily"]),
    ] + command_arguments(config)
    if resume:
        command.append("--resume")
    return command


def evaluation_command(cell: dict, run_dir: Path, corpus: Path,
                       config: dict) -> list[str]:
    evaluation = config["evaluation"]
    return [
        sys.executable, str(SCRIPT_DIR / "evaluate_language.py"),
        "--phase", "evaluate",
        "--condition", cell["condition"],
        "--seed", str(cell["optimizationSeed"]),
        "--interface-seed", str(cell["interfaceSeed"]),
        "--graph-dir", str(cell["graph"]),
        "--corpus", str(corpus),
        "--run-dir", str(run_dir),
        "--train-tokens", str(config["trainTokens"]),
        "--warmup-tokens", str(config["warmupTokens"]),
        "--checkpoint-every-tokens", str(config["checkpointEveryTokens"]),
        "--pairs-per-family", str(evaluation["pairsPerFamily"]),
        "--intervention-pairs-per-family", str(evaluation["interventionPairsPerFamily"]),
        "--minimal-per-family", str(evaluation["minimalPerFamily"]),
        "--orientation-permutations", str(evaluation["orientationPermutations"]),
    ] + command_arguments(config)


def wait_for_gpus(poll_seconds: int = 60):
    while True:
        try:
            return campaign.gpu_inventory()
        except RuntimeError as error:
            if "occupied" not in str(error):
                raise
            print(f"[{utcnow()}] approved GPU occupied; waiting without changing it", flush=True)
            time.sleep(poll_seconds)


def serialize_cell(cell: dict) -> dict:
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in cell.items()}


def stage_cell_metadata(output: Path, run_dir: Path, metadata: dict,
                        *, training_complete: bool) -> None:
    """Lock metadata without violating the trainer's new-directory contract."""
    cell_root = output / "cells"
    cell_root.mkdir(parents=True, exist_ok=True)
    campaign.write_locked_cell(cell_root / f"{metadata['name']}.json", metadata)
    if not training_complete:
        return
    if not run_dir.is_dir():
        raise FileNotFoundError(f"completed trainer did not create its run directory: {run_dir}")
    campaign.write_locked_cell(run_dir / "full-confirmatory-cell.json", metadata)


def write_plan(args, config: dict, cells: list[dict],
               real_provenance: dict, rewire_provenance: list[dict]) -> dict:
    sources = (
        Path(__file__),
        SCRIPT_DIR / "language_campaign.py",
        SCRIPT_DIR / "evaluate_language.py",
        SCRIPT_DIR / "analyze_language.py",
        Path(__file__).parents[1] / "src" / "flystudy" / "language_train.py",
        Path(__file__).parents[1] / "src" / "flystudy" / "language_evaluation.py",
    )
    plan = {
        "schema": 1,
        "createdAt": utcnow(),
        "configPath": str(args.config),
        "configSha256": file_hash(args.config),
        "sourceHashes": {path.name: file_hash(path) for path in sources},
        "corpusManifestSha256": file_hash(args.corpus / "manifest.json"),
        "realGraphProvenance": real_provenance,
        "rewireGraphProvenance": rewire_provenance,
        "trainTokens": config["trainTokens"],
        "warmupTokens": config["warmupTokens"],
        "cells": [serialize_cell(cell) for cell in cells],
        "locking": config["locking"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "campaign-plan.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        for key in ("configSha256", "sourceHashes", "corpusManifestSha256",
                    "realGraphProvenance", "rewireGraphProvenance",
                    "trainTokens", "warmupTokens", "cells", "locking"):
            if previous.get(key) != plan.get(key):
                raise ValueError("campaign resume differs from the locked full-budget plan")
        return previous
    atomic_json(destination, plan)
    return plan


def run_campaign(args, config: dict, cells: list[dict]):
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    total_waves = (len(cells) + len(campaign.ALLOWED_GPUS) - 1) // len(campaign.ALLOWED_GPUS)
    for start in range(0, len(cells), len(campaign.ALLOWED_GPUS)):
        wave_number = start // len(campaign.ALLOWED_GPUS) + 1
        wave = cells[start:start + len(campaign.ALLOWED_GPUS)]
        assigned = [{**cell, "gpu": gpu}
                    for gpu, cell in zip(campaign.ALLOWED_GPUS[:len(wave)], wave, strict=True)]
        atomic_json(args.output / "campaign-state.json", {
            "at": utcnow(), "status": "running", "wave": wave_number,
            "totalWaves": total_waves,
            "cells": [cell["name"] for cell in assigned],
        })
        inventory = wait_for_gpus()
        training = []
        for cell in assigned:
            run_dir = args.output / "runs" / cell["name"]
            metadata = {
                **serialize_cell(cell),
                "trainTokens": config["trainTokens"],
                "warmupTokens": config["warmupTokens"],
                "wave": wave_number,
            }
            stage_cell_metadata(
                args.output, run_dir, metadata, training_complete=False)
            if campaign.trained_state(run_dir, config["trainTokens"]):
                stage_cell_metadata(
                    args.output, run_dir, metadata, training_complete=True)
                continue
            training.append({
                **cell,
                "stage": "full",
                "phase": "train",
                "command": training_command(
                    cell, run_dir, args.corpus, config,
                    resume=(run_dir / "checkpoint-latest.pt").exists()),
            })
        if training:
            campaign.run_commands(training, inventory, logs)

        evaluation = []
        for cell in assigned:
            run_dir = args.output / "runs" / cell["name"]
            if not campaign.trained_state(run_dir, config["trainTokens"]):
                raise RuntimeError(f"full-budget training did not complete: {cell['name']}")
            metadata = {
                **serialize_cell(cell),
                "trainTokens": config["trainTokens"],
                "warmupTokens": config["warmupTokens"],
                "wave": wave_number,
            }
            stage_cell_metadata(
                args.output, run_dir, metadata, training_complete=True)
            if (run_dir / "confirmatory-evaluation.json").exists():
                continue
            evaluation.append({
                **cell,
                "stage": "full",
                "phase": "evaluate",
                "command": evaluation_command(cell, run_dir, args.corpus, config),
            })
        if evaluation:
            inventory = wait_for_gpus()
            campaign.run_commands(evaluation, inventory, logs)

    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "analyze_language.py"),
        "--campaign-root", str(args.output),
    ], check=True)
    atomic_json(args.output / "campaign-state.json", {
        "at": utcnow(), "status": "complete", "waves": total_waves,
    })
    atomic_json(args.output / "campaign-complete.json", {"at": utcnow()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--rewire-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/language_confirmatory.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    rewires = [args.rewire_root / f"rewire-{index:02d}" for index in range(1, 21)]
    real_provenance, rewire_provenance = campaign.validate_graph_ensemble(
        args.real, rewires, config)
    cells = build_all_cells(args.real, rewires, config)
    plan = write_plan(
        args, config, cells, real_provenance, rewire_provenance)
    if not args.execute:
        print(json.dumps({
            "output": str(args.output),
            "cells": len(plan["cells"]),
            "waves": (len(cells) + len(campaign.ALLOWED_GPUS) - 1) // len(campaign.ALLOWED_GPUS),
            "trainTokensPerNewCell": config["trainTokens"],
            "warmupTokens": config["warmupTokens"],
        }, indent=2))
        return
    run_campaign(args, config, cells)


if __name__ == "__main__":
    main()
