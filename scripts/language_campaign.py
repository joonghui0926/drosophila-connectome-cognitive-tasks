#!/usr/bin/env python3
"""Language campaign scheduling, GPU isolation and artifact validation."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from flystudy.language_support import atomic_json, utcnow
from flystudy.language_audit import audit_files, audit_graphs
from flystudy.language_runtime import file_hash


ALLOWED_GPUS = (0, 1, 2, 3)
FACTORIAL_LAYOUT = (
    ((1, 0, 0), (2, 1, 1), (3, 2, 2), (4, 3, 3)),
    ((0, 0, 0), (3, 1, 3), (4, 2, 1), (2, 3, 2)),
    ((3, 0, 1), (4, 1, 0), (0, 2, 3), (1, 3, 2)),
    ((4, 0, 2), (1, 1, 3), (2, 2, 0), (0, 3, 1)),
    ((2, 0, 3), (0, 1, 2), (1, 2, 1), (3, 3, 0)),
)


def factorial_cells(real: Path, rewires: list[Path], interface_seeds: list[int],
                    optimization_seed: int) -> list[dict]:
    if len(rewires) < 4 or len(interface_seeds) != 4 or len(set(interface_seeds)) != 4:
        raise ValueError("factorial study requires four rewires and four interface seeds")
    topologies = [("real", "real", real)] + [
        (f"rewire-{index:02d}", "rewired", path)
        for index, path in enumerate(rewires[:4], 1)
    ]
    cells = []
    # Each wave contains four distinct topologies, interfaces and GPU slots.
    # Across waves, every topology runs once per GPU and every interface visits
    # all four GPUs. This avoids both hardware and execution-order confounding.
    for wave_slot, wave in enumerate(FACTORIAL_LAYOUT):
        for topology_index, interface_index, hardware_slot in wave:
            topology, condition, graph = topologies[topology_index]
            interface_seed = interface_seeds[interface_index]
            cells.append({
                "stage": "factorial",
                "name": f"{topology}-i{interface_seed}",
                "topology": topology,
                "condition": condition,
                "graph": graph,
                "interfaceSeed": int(interface_seed),
                "optimizationSeed": int(optimization_seed),
                "hardwareSlot": hardware_slot,
                "waveSlot": wave_slot,
            })
    if len({(row["topology"], row["interfaceSeed"]) for row in cells}) != 20:
        raise RuntimeError("factorial construction did not produce a complete crossing")
    return cells


def null_rank_cells(real: Path, rewires: list[Path], canonical_interface: int,
                    optimization_seed: int, schedule_seed=20260923) -> list[dict]:
    if len(rewires) != 20 or len(set(map(str, rewires))) != 20:
        raise ValueError("null-rank study requires twenty independent graph paths")
    rows = [{
        "stage": "null-rank",
        "name": f"real-i{canonical_interface}",
        "topology": "real",
        "condition": "real",
        "graph": real,
        "interfaceSeed": int(canonical_interface),
        "optimizationSeed": int(optimization_seed),
    }]
    rows.extend({
        "stage": "null-rank",
        "name": f"rewire-{index:02d}-i{canonical_interface}",
        "topology": f"rewire-{index:02d}",
        "condition": "rewired",
        "graph": path,
        "interfaceSeed": int(canonical_interface),
        "optimizationSeed": int(optimization_seed),
    } for index, path in enumerate(rewires, 1))
    random.Random(schedule_seed).shuffle(rows)
    for order, row in enumerate(rows):
        row["scheduleOrder"] = order
    return rows


def configure_gpus(values) -> tuple[int, ...]:
    """Set the physical GPU indices used by the campaign launcher."""
    global ALLOWED_GPUS
    gpus = tuple(int(value) for value in values)
    if not gpus or len(gpus) > 8 or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("allowedPhysicalGpus must contain unique nonnegative indices")
    ALLOWED_GPUS = gpus
    return gpus


def schedule_waves(cells: list[dict], gpus=None) -> list[list[dict]]:
    gpus = ALLOWED_GPUS if gpus is None else tuple(gpus)
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU indices must be unique")
    if cells and all("hardwareSlot" in cell and "waveSlot" in cell for cell in cells):
        wave_ids = sorted({cell["waveSlot"] for cell in cells})
        waves = []
        for wave_id in wave_ids:
            selected = [cell for cell in cells if cell["waveSlot"] == wave_id]
            if len(selected) != len(gpus):
                raise ValueError("balanced wave must contain one job per approved GPU")
            by_slot = {cell["hardwareSlot"]: cell for cell in selected}
            if set(by_slot) != set(range(len(gpus))):
                raise ValueError("balanced wave must use every hardware slot once")
            waves.append([{**by_slot[slot], "gpu": int(gpus[slot])}
                          for slot in range(len(gpus))])
        return waves
    if cells and all("hardwareSlot" in cell for cell in cells):
        buckets = {slot: [] for slot in range(len(gpus))}
        for cell in cells:
            slot = cell["hardwareSlot"]
            if slot not in buckets:
                raise ValueError("hardware slot is outside the approved GPU set")
            buckets[slot].append(cell)
        sizes = {len(bucket) for bucket in buckets.values()}
        if len(sizes) != 1:
            raise ValueError("balanced hardware scheduling needs equal slot counts")
        for bucket in buckets.values():
            bucket.sort(key=lambda row: (row["topology"], row["interfaceSeed"]))
        return [
            [{**buckets[slot][wave], "gpu": int(gpus[slot])}
             for slot in range(len(gpus))]
            for wave in range(sizes.pop())
        ]
    waves = []
    for start in range(0, len(cells), len(gpus)):
        wave = []
        for gpu, cell in zip(gpus, cells[start:start + len(gpus)], strict=False):
            wave.append({**cell, "gpu": int(gpu)})
        waves.append(wave)
    return waves


def gpu_inventory(gpus=None) -> dict[int, dict]:
    gpus = ALLOWED_GPUS if gpus is None else tuple(gpus)
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True)
    entries = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    selected = {int(index): {"uuid": uuid.strip(), "name": name.strip()}
                for index, uuid, name in entries if int(index) in gpus}
    if set(selected) != set(gpus):
        raise RuntimeError(f"configured GPUs are not all visible: {gpus}")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    occupied = {row[0].strip() for row in csv.reader(io.StringIO(processes), skipinitialspace=True) if row}
    if occupied & {entry["uuid"] for entry in selected.values()}:
        raise RuntimeError("an approved GPU is occupied; no process was changed")
    return selected


def model_arguments(config: dict) -> list[str]:
    model = config["model"]
    pairs = (
        ("token-steps", model["tokenSteps"]),
        ("token-drive-mode", model["tokenDriveMode"]),
        ("learning-rate", model["learningRate"]),
        ("weight-decay", model["weightDecay"]),
        ("global-batch", model["globalBatch"]),
        ("microbatch", model["microbatch"]),
        ("encode-steps", model["encodeSteps"]),
        ("eval-batch", model["evalBatch"]),
        ("dimension", model["dimension"]),
        ("token-group", model["tokenGroup"]),
        ("output-group", model["outputGroup"]),
        ("leak", model["leak"]),
        ("token-gain", model["tokenGain"]),
        ("sensor-gain", model["sensorGain"]),
        ("temperature", model["temperature"]),
    )
    return ["--codebook-seed", str(config["codebookSeed"]),
            "--embedding", "orthogonal"] + [
        value for name, item in pairs for value in (f"--{name}", str(item))]






def run_commands(specs: list[dict], inventory: dict[int, dict], log_root: Path):
    running = []
    def interrupt(signum, _frame):
        raise RuntimeError(f"confirmatory launcher received signal {signum}")
    previous_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for spec in specs:
            env = dict(os.environ)
            env.update(
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                CUDA_VISIBLE_DEVICES=str(spec["gpu"]),
                FLY_EXPECTED_GPU_UUID=inventory[spec["gpu"]]["uuid"],
                OMP_NUM_THREADS="2",
                PYTHONUNBUFFERED="1",
            )
            log_path = log_root / f"{spec['stage']}-{spec['name']}-{spec['phase']}.log"
            stream = log_path.open("a")
            stream.write(json.dumps({"at": utcnow(), "command": spec["command"],
                                     "physicalGpu": spec["gpu"]}) + "\n")
            stream.flush()
            process = subprocess.Popen(
                spec["command"], env=env, stdout=stream, stderr=subprocess.STDOUT)
            running.append((process, stream, log_path, spec))
        while any(process.poll() is None for process, _, _, _ in running):
            if any(process.poll() not in (None, 0) for process, _, _, _ in running):
                raise RuntimeError("a confirmatory worker failed; inspect campaign logs")
            time.sleep(5)
        if any(process.returncode for process, _, _, _ in running):
            raise RuntimeError("a confirmatory worker failed; inspect campaign logs")
    finally:
        for process, stream, _, _ in running:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            stream.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def trained_state(run_dir: Path, tokens: int) -> bool:
    path = run_dir / "training-state.json"
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    if state.get("status") != "trained":
        return False
    required = ("budgetTokens", "processedTokens", "unusedTokens", "maxRecordTokens")
    if any(key not in state for key in required):
        raise ValueError(f"completed training state is incomplete: {run_dir}")
    valid = (
        state["budgetTokens"] == tokens
        and state["unusedTokens"] == tokens - state["processedTokens"]
        and 0 <= state["unusedTokens"] < state["maxRecordTokens"]
        and (run_dir / "checkpoint-latest.pt").is_file()
        and (run_dir / "config.json").is_file()
    )
    if not valid:
        raise ValueError(f"completed run does not match the locked token budget: {run_dir}")
    return True


def write_locked_cell(path: Path, payload: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"confirmatory cell metadata changed on resume: {path}")
        return
    atomic_json(path, payload)


def graph_provenance(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    graph_path = path / "graph.npz"
    nodes_path = path / "nodes.feather"
    if not manifest_path.is_file() or not graph_path.is_file() or not nodes_path.is_file():
        raise FileNotFoundError(f"graph artifact is incomplete: {path}")
    manifest = json.loads(manifest_path.read_text())
    graph_sha = file_hash(graph_path)
    if graph_sha != manifest.get("graph_sha256"):
        raise ValueError(f"graph checksum mismatch: {path}")
    return {
        "path": str(path),
        "manifestSha256": file_hash(manifest_path),
        "graphSha256": graph_sha,
        "nodesFeatherSha256": file_hash(nodes_path),
        "neurons": manifest.get("neurons"),
        "edges": manifest.get("edges"),
        "control": manifest.get("control"),
        "sourceGraphSha256": manifest.get("sourceGraphSha256"),
        "rewireSeed": manifest.get("rewiring", {}).get("seed"),
        "acceptedSwaps": manifest.get("rewiring", {}).get("acceptedSwaps"),
    }


def validate_graph_ensemble(real: Path, rewires: list[Path], config: dict):
    if len(rewires) != 20:
        raise ValueError("the locked ensemble requires twenty rewire paths")
    real_provenance = graph_provenance(real)
    rewire_provenance = [graph_provenance(path) for path in rewires]
    if not isinstance(real_provenance["neurons"], int) or real_provenance["neurons"] < 1:
        raise ValueError("real graph manifest has an invalid neuron count")
    if not isinstance(real_provenance["edges"], int) or real_provenance["edges"] < 1:
        raise ValueError("real graph manifest has an invalid edge count")
    if any(item["nodesFeatherSha256"] != real_provenance["nodesFeatherSha256"]
           for item in rewire_provenance):
        raise ValueError("all topology controls must use identical neuron metadata")
    if any(item["control"] != "directed-degree-preserving-rewire"
           or item["sourceGraphSha256"] != real_provenance["graphSha256"]
           or item["neurons"] != real_provenance["neurons"]
           or item["edges"] != real_provenance["edges"]
           or not isinstance(item["acceptedSwaps"], int)
           or item["acceptedSwaps"] < 5 * real_provenance["edges"]
           for item in rewire_provenance):
        raise ValueError("a topology control fails the locked rewiring provenance")
    if len({item["graphSha256"] for item in rewire_provenance}) != 20:
        raise ValueError("twenty independently materialized rewire graphs are required")
    if [item["rewireSeed"] for item in rewire_provenance] != config["rewireSeeds"]:
        raise ValueError("rewire graph order and seeds do not match the locked config")
    return real_provenance, rewire_provenance
