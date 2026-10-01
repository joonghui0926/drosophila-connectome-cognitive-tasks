from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

from flystudy.activity import summarize_activity
from flystudy.dataset import ArithmeticDataset
from flystudy.graph import sha256_file
from flystudy.losses import arithmetic_loss
from flystudy.metrics import classification_metrics, sample_mix
from flystudy.model import FixedTopologyArithmeticNet, Interface, select_interface


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested not in {"cpu", "mps", "cuda"} and not requested.startswith("cuda:"):
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N")
    return torch.device(requested)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _publish_live(live_path: Path, payload: dict, update: int) -> None:
    payload["availableSnapshots"] = list(range(update + 1))
    _atomic_json(live_path, payload)
    _atomic_json(live_path.parent / "updates" / f"update-{update:04d}.json", payload)


def _load_graph(graph_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    manifest = json.loads((graph_dir / "manifest.json").read_text())
    graph_path = graph_dir / "graph.npz"
    if sha256_file(graph_path) != manifest.get("graph_sha256"):
        raise ValueError("graph checksum does not match its manifest")
    with np.load(graph_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return arrays, manifest


def _diagnostic_rows(dataset: ArithmeticDataset) -> list[dict]:
    result: list[dict] = []
    for op in ("+", "-"):
        result.append(dataset.by_operation[op][0])
    for name in ("nontrivial", "times_zero", "times_one"):
        result.append(dataset.multiplication_pools[name][0])
    return result


def _validation_subset(dataset: ArithmeticDataset, limit: int) -> list[dict]:
    if limit <= 0 or limit >= len(dataset.rows):
        return list(dataset.rows)
    pools = [
        dataset.by_operation["+"],
        dataset.by_operation["-"],
        dataset.multiplication_pools["nontrivial"],
        dataset.multiplication_pools["times_zero"],
        dataset.multiplication_pools["times_one"],
    ]
    allocation = np.full(len(pools), limit // len(pools), dtype=int)
    allocation[: limit % len(pools)] += 1
    selected: list[dict] = []
    for pool, count in zip(pools, allocation, strict=True):
        if count == 0:
            continue
        indices = np.linspace(0, len(pool) - 1, num=count, dtype=int)
        selected.extend(pool[int(index)] for index in indices)
    return selected


def _model_from_graph(arrays: dict[str, np.ndarray], config: dict) -> tuple[FixedTopologyArithmeticNet, Interface]:
    interface = select_interface(
        arrays["rows"],
        arrays["cols"],
        n_nodes=len(arrays["body_ids"]),
        input_group_size=config["input_group_size"],
        output_group_size=config["output_group_size"],
        seed=config["interface_seed"],
    )
    model = FixedTopologyArithmeticNet(
        arrays["rows"],
        arrays["cols"],
        arrays["counts"],
        interface,
        n_nodes=len(arrays["body_ids"]),
        steps=config["steps"],
        readout_steps=config["readout_steps"],
        alpha=config["alpha"],
        gain=config["gain"],
        input_amplitude=config["input_amplitude"],
        temperature=config["temperature"],
        chunk_size=config.get("chunk_size", 1_048_576),
    )
    return model, interface


def _batch_tensors(rows: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded, targets = ArithmeticDataset.arrays(rows)
    return (
        torch.as_tensor(encoded, device=device),
        torch.as_tensor(targets, device=device),
    )


@torch.no_grad()
def evaluate(
    model: FixedTopologyArithmeticNet,
    rows: list[dict],
    *,
    device: torch.device,
    batch_size: int,
    ordinal_lambda: float,
) -> tuple[dict, list[dict]]:
    model.eval()
    predictions: list[int] = []
    recent: list[dict] = []
    loss_sums = {"loss": 0.0, "cross_entropy": 0.0, "ordinal": 0.0}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded, targets = _batch_tensors(batch, device)
        logits, _ = model(encoded)
        _, parts = arithmetic_loss(logits, targets, ordinal_lambda)
        batch_predictions = logits.argmax(dim=1).cpu().numpy()
        predictions.extend(int(value) for value in batch_predictions)
        for name, value in parts.items():
            loss_sums[name] += float(value.cpu()) * len(batch)
        for row, prediction in zip(batch, batch_predictions, strict=True):
            if len(recent) < 12:
                recent.append(
                    {
                        "expression": f"{row['a']} {row['op']} {row['b']}",
                        "target": int(row["y"]),
                        "prediction": int(prediction),
                        "correct": int(prediction) == int(row["y"]),
                    }
                )
    metrics = classification_metrics(rows, np.asarray(predictions))
    metrics["losses"] = {name: value / len(rows) for name, value in loss_sums.items()}
    return metrics, recent


@torch.no_grad()
def diagnostic_snapshot(
    model: FixedTopologyArithmeticNet,
    rows: list[dict],
    *,
    device: torch.device,
    nodes_path: Path,
    seed: int,
) -> dict:
    model.eval()
    encoded, targets = _batch_tensors(rows, device)
    logits, trace = model(encoded, capture_activity=True)
    probabilities = logits.softmax(dim=1)
    predictions = logits.argmax(dim=1)
    activity = summarize_activity(trace["final_state"], nodes_path, seed=seed)
    activity["timeline"] = [
        {"step": index + 1, "meanAbs": float(mean), "peakAbs": float(peak)}
        for index, (mean, peak) in enumerate(
            zip(trace["activity_mean"].cpu(), trace["activity_peak"].cpu(), strict=True)
        )
    ]
    first = rows[0]
    return {
        "activity": activity,
        "output": {
            "expression": f"{first['a']} {first['op']} {first['b']}",
            "target": int(targets[0].cpu()),
            "prediction": int(predictions[0].cpu()),
            "probabilities": [float(value) for value in probabilities[0].cpu()],
        },
        "diagnostics": [
            {
                "expression": f"{row['a']} {row['op']} {row['b']}",
                "target": int(target),
                "prediction": int(prediction),
            }
            for row, target, prediction in zip(rows, targets.cpu(), predictions.cpu(), strict=True)
        ],
    }


def _memory_snapshot(device: torch.device) -> dict:
    process = psutil.Process()
    result = {"processRssGiB": process.memory_info().rss / 2**30}
    if device.type == "mps":
        result.update(
            {
                "mpsAllocatedGiB": torch.mps.current_allocated_memory() / 2**30,
                "mpsDriverGiB": torch.mps.driver_allocated_memory() / 2**30,
            }
        )
    elif device.type == "cuda":
        result.update(
            {
                "cudaAllocatedGiB": torch.cuda.memory_allocated(device) / 2**30,
                "cudaReservedGiB": torch.cuda.memory_reserved(device) / 2**30,
                "cudaPeakAllocatedGiB": torch.cuda.max_memory_allocated(device) / 2**30,
                "cudaDevice": torch.cuda.get_device_name(device),
            }
        )
    return result


def run_training(
    *,
    graph_dir: str | Path,
    dataset_path: str | Path,
    config_path: str | Path,
    run_dir: str | Path,
    updates: int,
    device_name: str = "auto",
    validation_limit: int = 100,
    live_path: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    graph_dir, run_dir = Path(graph_dir), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text())
    config.update(overrides or {})
    if config.get("multiplication_mix") != {"nontrivial": 0.6, "times_zero": 0.2, "times_one": 0.2}:
        raise ValueError("this implementation locks multiplication sampling to 60/20/20")
    if config["examples_per_update"] % config["microbatch_size"]:
        raise ValueError("examples_per_update must be divisible by microbatch_size")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = choose_device(device_name)
    arrays, graph_manifest = _load_graph(graph_dir)
    model, interface = _model_from_graph(arrays, config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [model.theta],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    train_data = ArithmeticDataset.from_json(dataset_path, "train")
    validation_data = ArithmeticDataset.from_json(dataset_path, "validation")
    validation_rows = _validation_subset(validation_data, validation_limit)
    diagnostic_rows = _diagnostic_rows(validation_data)
    rng = np.random.default_rng(config["data_seed"])
    run_id = run_dir.name
    started_at = datetime.now(timezone.utc).isoformat()
    live_path = Path(live_path) if live_path else run_dir / "latest.json"
    history: list[dict] = []
    payload = {
        "schemaVersion": 1,
        "status": "initializing",
        "run": {
            "id": run_id,
            "update": 0,
            "totalUpdates": updates,
            "device": str(device),
            "startedAt": started_at,
        },
        "graph": graph_manifest,
        "config": config,
        "history": history,
        "provenance": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "dataset": str(Path(dataset_path).resolve()),
            "graph": str(graph_dir.resolve()),
        },
    }
    _atomic_json(run_dir / "interface.json", interface.to_dict())
    initial = diagnostic_snapshot(
        model,
        diagnostic_rows,
        device=device,
        nodes_path=graph_dir / "nodes.feather",
        seed=config["interface_seed"],
    )
    payload.update(initial)
    payload["memory"] = _memory_snapshot(device)
    _publish_live(live_path, payload, 0)
    _atomic_json(run_dir / "snapshots" / "update-0000.json", payload)

    log_path = run_dir / "metrics.jsonl"
    for update in range(1, updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_rows: list[dict] = []
        predictions: list[int] = []
        part_sums = {"loss": 0.0, "cross_entropy": 0.0, "ordinal": 0.0}
        started = time.perf_counter()
        microbatches = config["examples_per_update"] // config["microbatch_size"]
        for _ in range(microbatches):
            batch = train_data.sample_balanced(config["microbatch_size"], rng)
            encoded, targets = _batch_tensors(batch, device)
            logits, _ = model(encoded)
            loss, parts = arithmetic_loss(logits, targets, config["ordinal_lambda"])
            (loss / microbatches).backward()
            update_rows.extend(batch)
            predictions.extend(int(value) for value in logits.detach().argmax(dim=1).cpu())
            for name, value in parts.items():
                part_sums[name] += float(value.cpu()) / microbatches
        task_gradient_nonzero = int(torch.count_nonzero(model.theta.grad).cpu())
        task_gradient_finite = bool(torch.isfinite(model.theta.grad).all().cpu())
        gradient_norm = float(torch.nn.utils.clip_grad_norm_([model.theta], config["gradient_clip"]).cpu())
        optimizer.step()
        _synchronize(device)
        elapsed = time.perf_counter() - started
        train_metrics = classification_metrics(update_rows, np.asarray(predictions))
        train_metrics["losses"] = part_sums
        validation_metrics, recent = evaluate(
            model,
            validation_rows,
            device=device,
            batch_size=config["microbatch_size"],
            ordinal_lambda=config["ordinal_lambda"],
        )
        record = {
            "update": update,
            "elapsedSeconds": elapsed,
            "examplesPerSecond": config["examples_per_update"] / elapsed,
            "gradientNormBeforeClip": gradient_norm,
            "taskGradientNonzeroEdges": task_gradient_nonzero,
            "taskGradientFinite": task_gradient_finite,
            "train": train_metrics,
            "validation": validation_metrics,
            "sampleMix": sample_mix(update_rows),
            "memory": _memory_snapshot(device),
        }
        history.append(record)
        with log_path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        diagnostic = diagnostic_snapshot(
            model,
            diagnostic_rows,
            device=device,
            nodes_path=graph_dir / "nodes.feather",
            seed=config["interface_seed"],
        )
        payload.update(diagnostic)
        payload.update(
            {
                "status": "complete" if update == updates else "training",
                "run": {**payload["run"], "update": update},
                "history": history,
                "trainMetrics": train_metrics,
                "validationMetrics": validation_metrics,
                "sampleMix": record["sampleMix"],
                "recentPredictions": recent,
                "memory": record["memory"],
            }
        )
        _publish_live(live_path, payload, update)
        _atomic_json(run_dir / "snapshots" / f"update-{update:04d}.json", payload)
        checkpoint = {
            "theta": model.theta.detach().cpu(),
            "interface": interface.to_dict(),
            "config": config,
            "graph_sha256": graph_manifest["graph_sha256"],
            "updates": update,
        }
        _atomic_checkpoint(run_dir / "checkpoint.pt", checkpoint)
    _atomic_json(run_dir / "summary.json", payload)
    return payload


def benchmark_graph(
    *,
    graph_dir: str | Path,
    dataset_path: str | Path,
    config_path: str | Path,
    device_name: str,
    steps: int,
    batch_size: int,
) -> dict:
    config = json.loads(Path(config_path).read_text())
    config.update({"steps": steps, "readout_steps": min(4, steps)})
    torch.manual_seed(config["seed"])
    device = choose_device(device_name)
    arrays, manifest = _load_graph(Path(graph_dir))
    model, _ = _model_from_graph(arrays, config)
    model.to(device)
    dataset = ArithmeticDataset.from_json(dataset_path, "train")
    rows = dataset.sample_balanced(max(3, batch_size), np.random.default_rng(config["data_seed"]))[:batch_size]
    encoded, targets = _batch_tensors(rows, device)
    _synchronize(device)
    start = time.perf_counter()
    logits, _ = model(encoded)
    _synchronize(device)
    forward_seconds = time.perf_counter() - start
    loss, _ = arithmetic_loss(logits, targets, config["ordinal_lambda"])
    start = time.perf_counter()
    loss.backward()
    _synchronize(device)
    backward_seconds = time.perf_counter() - start
    return {
        "device": str(device),
        "steps": steps,
        "batchSize": batch_size,
        "neurons": manifest["neurons"],
        "edges": manifest["edges"],
        "loss": float(loss.detach().cpu()),
        "forwardSeconds": forward_seconds,
        "backwardSeconds": backward_seconds,
        "memory": _memory_snapshot(device),
        "thetaGradientFinite": bool(torch.isfinite(model.theta.grad).all().cpu()),
        "thetaGradientNonzero": int(torch.count_nonzero(model.theta.grad).cpu()),
    }
