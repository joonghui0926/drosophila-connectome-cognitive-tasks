from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from flystudy.activity import encode_activity_replay, spatial_metadata, stable_spatial_indices, summarize_activity
from flystudy.dataset import ArithmeticDataset
from flystudy.graph import sha256_file
from flystudy.losses import arithmetic_loss
from flystudy.metrics import classification_metrics, constant_answer_baseline
from flystudy.train import (
    _atomic_checkpoint,
    _atomic_json,
    _batch_tensors,
    _load_graph,
    _memory_snapshot,
    _model_from_graph,
    _synchronize,
    choose_device,
    evaluate,
)


def select_addition_probes(dataset: ArithmeticDataset) -> list[dict]:
    rows = sorted(dataset.by_operation["+"], key=lambda row: (int(row["y"]), int(row["a"]), int(row["b"])))
    criteria = [
        ("small_no_carry", lambda row: 0 < row["a"] < 10 and 0 < row["b"] < 10 and row["y"] < 10),
        ("one_digit_carry", lambda row: row["a"] < 10 and row["b"] < 10 and 10 <= row["y"] < 20),
        (
            "two_digit_no_carry",
            lambda row: row["a"] >= 10 and row["b"] >= 10 and row["y"] < 100 and row["a"] % 10 + row["b"] % 10 < 10,
        ),
        (
            "two_digit_carry",
            lambda row: row["a"] >= 10 and row["b"] >= 10 and row["y"] < 100 and row["a"] % 10 + row["b"] % 10 >= 10,
        ),
        ("sum_100", lambda row: row["y"] == 100 and row["a"] > 1 and row["b"] > 1),
    ]
    probes = []
    for name, predicate in criteria:
        candidates = [row for row in rows if predicate(row)]
        if not candidates:
            raise ValueError(f"validation split has no addition probe for {name}")
        center = candidates[len(candidates) // 2]
        probes.append({**center, "probe": name})
    return probes


def early_stop_reached(epoch: int, bad_epochs: int, *, min_epochs: int, patience: int) -> bool:
    return epoch >= min_epochs and bad_epochs >= patience


def _validate_run_directory(run_dir: Path, *, resume: bool) -> None:
    if resume:
        checkpoint = run_dir / "checkpoint-latest.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    elif run_dir.exists():
        # The GPU launcher opens train.log before starting the trainer.
        if not run_dir.is_dir() or any(path.name != "train.log" for path in run_dir.iterdir()):
            raise FileExistsError(
                f"run directory is not empty: {run_dir}; use --resume or a new directory"
            )


def _validate_resume(checkpoint: dict, config: dict, graph_sha256: str) -> None:
    if checkpoint["graph_sha256"] != graph_sha256:
        raise ValueError("resume checkpoint graph does not match this graph")
    saved = checkpoint.get("config", {})
    stopping_limits = {"max_epochs", "min_epochs", "patience"}
    changed = sorted(key for key in saved.keys() | config.keys()
                     if key not in stopping_limits and saved.get(key) != config.get(key))
    if changed:
        raise ValueError(f"resume checkpoint configuration differs: {', '.join(changed)}")
    if config["max_epochs"] < int(checkpoint["epoch"]):
        raise ValueError("max epochs cannot precede the resumed checkpoint")


def _publish_live(live_path: Path, payload: dict, *, completed_epoch: int, archive: bool = False) -> None:
    payload["availableEpochs"] = list(range(completed_epoch + 1))
    _atomic_json(live_path, payload)
    if archive:
        _atomic_json(live_path.parent / "epochs" / f"epoch-{completed_epoch:04d}.json", payload)


@torch.no_grad()
def addition_diagnostic_snapshot(
    model,
    probes: list[dict],
    *,
    device: torch.device,
    nodes_path: Path,
    capture_indices: np.ndarray,
    static_spatial: list[dict],
    prior_activity: dict | None = None,
    full_summary: bool = False,
) -> dict:
    model.eval()
    encoded, targets = _batch_tensors(probes, device)
    capture_nodes = torch.as_tensor(capture_indices, dtype=torch.int64, device=device)
    logits, trace = model(encoded, capture_activity=True, capture_nodes=capture_nodes)
    probabilities = logits.softmax(dim=1)
    predictions = logits.argmax(dim=1)
    replay = encode_activity_replay(trace["selected_activity"])
    if full_summary:
        activity = summarize_activity(
            trace["final_state"],
            nodes_path,
            spatial_indices=capture_indices,
        )
    else:
        activity = {
            "spatial": static_spatial,
            "regions": (prior_activity or {}).get("regions", []),
            "topNeurons": (prior_activity or {}).get("topNeurons", []),
            "spatialSampling": "stable deterministic soma sample shared by every replay",
        }
    activity.update(
        {
            "globalMeanAbs": float(trace["activity_mean"][-1].cpu()),
            "globalPeakAbs": float(trace["activity_peak"][-1].cpu()),
            "timeline": [
                {"step": index + 1, "meanAbs": float(mean), "peakAbs": float(peak)}
                for index, (mean, peak) in enumerate(
                    zip(trace["activity_mean"].cpu(), trace["activity_peak"].cpu(), strict=True)
                )
            ],
            "replay": replay,
        }
    )
    probe_outputs = []
    for row, target, prediction, probs in zip(
        probes, targets.cpu(), predictions.cpu(), probabilities.cpu(), strict=True
    ):
        probe_outputs.append(
            {
                "probe": row["probe"],
                "expression": f"{row['a']} + {row['b']}",
                "target": int(target),
                "prediction": int(prediction),
                "probabilities": [float(value) for value in probs],
            }
        )
    return {
        "activity": activity,
        "probeOutputs": probe_outputs,
        "output": probe_outputs[0],
        "diagnostics": [
            {
                "probe": item["probe"],
                "expression": item["expression"],
                "target": item["target"],
                "prediction": item["prediction"],
            }
            for item in probe_outputs
        ],
    }


def _latest_checkpoint(
    model,
    optimizer,
    *,
    config: dict,
    graph_sha256: str,
    epoch: int,
    updates: int,
    best_loss: float,
    best_epoch: int,
    bad_epochs: int,
    epoch_history: list[dict],
) -> dict:
    return {
        "theta": model.theta.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "graph_sha256": graph_sha256,
        "epoch": epoch,
        "updates": updates,
        "bestValidationLoss": best_loss,
        "bestEpoch": best_epoch,
        "badEpochs": bad_epochs,
        "epochHistory": epoch_history,
    }


def run_addition_smoke(
    *,
    graph_dir: str | Path,
    dataset_path: str | Path,
    config_path: str | Path,
    run_dir: str | Path,
    device_name: str = "auto",
    live_path: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """Run one real addition update plus the same replay capture used by long runs."""
    graph_dir, run_dir = Path(graph_dir), Path(run_dir)
    _validate_run_directory(run_dir, resume=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text())
    config.update(overrides or {})
    config["mode"] = "addition-smoke"
    batch_size = int(config["microbatch_size"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = choose_device(device_name)
    arrays, graph_manifest = _load_graph(graph_dir)
    model, interface = _model_from_graph(arrays, config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [model.theta], lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    train_data = ArithmeticDataset.from_json(dataset_path, "train")
    validation_data = ArithmeticDataset.from_json(dataset_path, "validation")
    train_rows = train_data.by_operation["+"]
    batches = train_data.operation_epoch_batches(
        "+", batch_size, epoch=1, seed=config["data_seed"]
    )
    batch = batches[0]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    encoded, targets = _batch_tensors(batch, device)
    started = time.perf_counter()
    logits, _ = model(encoded)
    loss, parts = arithmetic_loss(logits, targets, config["ordinal_lambda"])
    loss.backward()
    task_gradient_nonzero = int(torch.count_nonzero(model.theta.grad).cpu())
    task_gradient_finite = bool(torch.isfinite(model.theta.grad).all().cpu())
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_([model.theta], config["gradient_clip"]).cpu()
    )
    optimizer.step()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    predictions = logits.detach().argmax(dim=1).cpu().numpy()
    train_metrics = classification_metrics(batch, predictions)
    train_metrics["losses"] = {name: float(value.cpu()) for name, value in parts.items()}
    record = {
        "update": 1,
        "epoch": 1,
        "batch": 1,
        "batchesPerEpoch": len(batches),
        "elapsedSeconds": elapsed,
        "examplesPerSecond": len(batch) / elapsed,
        "gradientNormBeforeClip": gradient_norm,
        "taskGradientNonzeroEdges": task_gradient_nonzero,
        "taskGradientFinite": task_gradient_finite,
        "train": train_metrics,
        "memory": _memory_snapshot(device),
    }

    probes = select_addition_probes(validation_data)
    capture_indices = stable_spatial_indices(
        graph_dir / "nodes.feather", seed=config["interface_seed"], spatial_points=3_000
    )
    static_spatial = spatial_metadata(graph_dir / "nodes.feather", capture_indices)
    diagnostic = addition_diagnostic_snapshot(
        model,
        probes,
        device=device,
        nodes_path=graph_dir / "nodes.feather",
        capture_indices=capture_indices,
        static_spatial=static_spatial,
        full_summary=True,
    )
    train_constant = constant_answer_baseline(train_rows)
    validation_constant = constant_answer_baseline(
        validation_data.by_operation["+"], answer=train_constant["answer"]
    )
    payload = {
        "schemaVersion": 2,
        "status": "smoke_complete",
        "run": {
            "id": run_dir.name,
            "mode": "addition",
            "seed": config["seed"],
            "epoch": 1,
            "maxEpochs": 100,
            "batch": 1,
            "batchesPerEpoch": len(batches),
            "update": 1,
            "totalUpdates": len(batches) * 100,
            "device": str(device),
            "startedAt": datetime.now(timezone.utc).isoformat(),
        },
        "graph": graph_manifest,
        "config": config,
        "history": [record],
        "epochHistory": [],
        "trainMetrics": train_metrics,
        "baselines": {
            "trainConstant": train_constant,
            "validationConstant": validation_constant,
        },
        "memory": record["memory"],
        **diagnostic,
    }
    _atomic_json(run_dir / "interface.json", interface.to_dict())
    target = Path(live_path) if live_path else run_dir / "live" / "latest.json"
    _publish_live(target, payload, completed_epoch=0, archive=True)
    _atomic_json(run_dir / "summary.json", payload)
    return payload


def run_addition_training(
    *,
    graph_dir: str | Path,
    dataset_path: str | Path,
    config_path: str | Path,
    run_dir: str | Path,
    max_epochs: int = 100,
    min_epochs: int = 20,
    patience: int = 15,
    device_name: str = "auto",
    live_path: str | Path | None = None,
    resume: bool = False,
    overrides: dict | None = None,
) -> dict:
    if max_epochs < 1 or min_epochs < 1 or patience < 1:
        raise ValueError("max epochs, min epochs, and patience must be positive")
    graph_dir, run_dir = Path(graph_dir), Path(run_dir)
    _validate_run_directory(run_dir, resume=resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text())
    config.update(overrides or {})
    config.update({"mode": "addition", "max_epochs": max_epochs, "min_epochs": min_epochs, "patience": patience})
    config["dataset_sha256"] = sha256_file(Path(dataset_path))
    config["nodes_sha256"] = sha256_file(graph_dir / "nodes.feather")
    config["graph_file_sha256"] = sha256_file(graph_dir / "graph.npz")
    batch_size = int(config["microbatch_size"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = choose_device(device_name)
    arrays, graph_manifest = _load_graph(graph_dir)
    model, interface = _model_from_graph(arrays, config)
    model.to(device)
    optimizer = torch.optim.AdamW([model.theta], lr=config["learning_rate"], weight_decay=config["weight_decay"])

    train_data = ArithmeticDataset.from_json(dataset_path, "train")
    validation_data = ArithmeticDataset.from_json(dataset_path, "validation")
    test_data = ArithmeticDataset.from_json(dataset_path, "test")
    train_rows = train_data.by_operation["+"]
    validation_rows = validation_data.by_operation["+"]
    test_rows = test_data.by_operation["+"]
    if len(train_rows) % batch_size:
        raise ValueError("addition train rows must divide evenly by the batch size")
    batches_per_epoch = len(train_rows) // batch_size
    probes = select_addition_probes(validation_data)
    capture_indices = stable_spatial_indices(
        graph_dir / "nodes.feather", seed=config["interface_seed"], spatial_points=3_000
    )
    static_spatial = spatial_metadata(graph_dir / "nodes.feather", capture_indices)
    live_path = Path(live_path) if live_path else run_dir / "live" / "latest.json"
    latest_checkpoint_path = run_dir / "checkpoint-latest.pt"
    best_checkpoint_path = run_dir / "checkpoint-best.pt"
    start_epoch = 0
    updates = 0
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    epoch_history: list[dict] = []

    if resume:
        if not latest_checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        _validate_resume(checkpoint, config, graph_manifest["graph_sha256"])
        with torch.no_grad():
            model.theta.copy_(checkpoint["theta"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        updates = int(checkpoint["updates"])
        best_loss = float(checkpoint["bestValidationLoss"])
        best_epoch = int(checkpoint["bestEpoch"])
        bad_epochs = int(checkpoint["badEpochs"])
        epoch_history = list(checkpoint["epochHistory"])

    train_constant = constant_answer_baseline(train_rows)
    validation_constant = constant_answer_baseline(validation_rows, answer=train_constant["answer"])
    payload = {
        "schemaVersion": 2,
        "status": "initializing",
        "run": {
            "id": run_dir.name,
            "mode": "addition",
            "seed": config["seed"],
            "epoch": start_epoch,
            "maxEpochs": max_epochs,
            "batch": 0,
            "batchesPerEpoch": batches_per_epoch,
            "update": updates,
            "totalUpdates": max_epochs * batches_per_epoch,
            "device": str(device),
            "startedAt": datetime.now(timezone.utc).isoformat(),
            "resumed": bool(resume),
        },
        "graph": graph_manifest,
        "config": config,
        "history": [],
        "epochHistory": epoch_history,
        "baselines": {"trainConstant": train_constant, "validationConstant": validation_constant},
        "provenance": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "dataset": str(Path(dataset_path).resolve()),
            "graph": str(graph_dir.resolve()),
            "trainingRows": len(train_rows),
            "validationRows": len(validation_rows),
            "testRows": len(test_rows),
            "epochDefinition": "all addition training rows exactly once without replacement",
        },
    }
    _atomic_json(run_dir / "interface.json", interface.to_dict())
    diagnostic = addition_diagnostic_snapshot(
        model,
        probes,
        device=device,
        nodes_path=graph_dir / "nodes.feather",
        capture_indices=capture_indices,
        static_spatial=static_spatial,
        full_summary=True,
    )
    payload.update(diagnostic)
    payload["memory"] = _memory_snapshot(device)
    _publish_live(live_path, payload, completed_epoch=start_epoch, archive=start_epoch == 0)

    metrics_path = run_dir / "metrics.jsonl"
    terminal_status = "complete"
    for epoch in range(start_epoch + 1, max_epochs + 1):
        batches = train_data.operation_epoch_batches(
            "+", batch_size, epoch=epoch, seed=config["data_seed"]
        )
        epoch_rows: list[dict] = []
        epoch_predictions: list[int] = []
        epoch_losses = {"loss": 0.0, "cross_entropy": 0.0, "ordinal": 0.0}
        epoch_elapsed = 0.0
        latest_record = None
        for batch_number, batch in enumerate(batches, start=1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            encoded, targets = _batch_tensors(batch, device)
            started = time.perf_counter()
            logits, _ = model(encoded)
            loss, parts = arithmetic_loss(logits, targets, config["ordinal_lambda"])
            loss.backward()
            task_gradient_nonzero = int(torch.count_nonzero(model.theta.grad).cpu())
            task_gradient_finite = bool(torch.isfinite(model.theta.grad).all().cpu())
            gradient_norm = float(torch.nn.utils.clip_grad_norm_([model.theta], config["gradient_clip"]).cpu())
            optimizer.step()
            _synchronize(device)
            elapsed = time.perf_counter() - started
            updates += 1
            predictions = logits.detach().argmax(dim=1).cpu().numpy()
            train_metrics = classification_metrics(batch, predictions)
            train_metrics["losses"] = {name: float(value.cpu()) for name, value in parts.items()}
            latest_record = {
                "update": updates,
                "epoch": epoch,
                "batch": batch_number,
                "batchesPerEpoch": batches_per_epoch,
                "elapsedSeconds": elapsed,
                "examplesPerSecond": len(batch) / elapsed,
                "gradientNormBeforeClip": gradient_norm,
                "taskGradientNonzeroEdges": task_gradient_nonzero,
                "taskGradientFinite": task_gradient_finite,
                "train": train_metrics,
                "memory": _memory_snapshot(device),
            }
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(latest_record, allow_nan=False) + "\n")
            epoch_rows.extend(batch)
            epoch_predictions.extend(int(value) for value in predictions)
            epoch_elapsed += elapsed
            for name, value in parts.items():
                epoch_losses[name] += float(value.cpu()) / batches_per_epoch

            diagnostic = addition_diagnostic_snapshot(
                model,
                probes,
                device=device,
                nodes_path=graph_dir / "nodes.feather",
                capture_indices=capture_indices,
                static_spatial=static_spatial,
                prior_activity=payload.get("activity"),
                full_summary=batch_number == batches_per_epoch,
            )
            payload.update(diagnostic)
            payload.update(
                {
                    "status": "training",
                    "run": {
                        **payload["run"],
                        "epoch": epoch,
                        "batch": batch_number,
                        "update": updates,
                    },
                    "history": [latest_record],
                    "trainMetrics": train_metrics,
                    "memory": latest_record["memory"],
                }
            )
            _publish_live(live_path, payload, completed_epoch=epoch - 1)

        validation_metrics, recent = evaluate(
            model,
            validation_rows,
            device=device,
            batch_size=batch_size,
            ordinal_lambda=config["ordinal_lambda"],
        )
        train_epoch_metrics = classification_metrics(epoch_rows, np.asarray(epoch_predictions))
        train_epoch_metrics["losses"] = epoch_losses
        validation_loss = float(validation_metrics["losses"]["loss"])
        improved = validation_loss < best_loss - 1e-6
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            bad_epochs = 0
            _atomic_checkpoint(
                best_checkpoint_path,
                {
                    "theta": model.theta.detach().cpu(),
                    "config": config,
                    "graph_sha256": graph_manifest["graph_sha256"],
                    "epoch": epoch,
                    "validationLoss": validation_loss,
                },
            )
        else:
            bad_epochs += 1
        epoch_record = {
            "epoch": epoch,
            "updates": updates,
            "elapsedSeconds": epoch_elapsed,
            "examplesPerSecond": len(epoch_rows) / epoch_elapsed,
            "train": train_epoch_metrics,
            "validation": validation_metrics,
            "improved": improved,
            "bestEpoch": best_epoch,
            "bestValidationLoss": best_loss,
            "badEpochs": bad_epochs,
        }
        epoch_history.append(epoch_record)
        payload.update(
            {
                "epochHistory": epoch_history,
                "validationMetrics": validation_metrics,
                "recentPredictions": recent,
                "run": {**payload["run"], "epoch": epoch, "batch": batches_per_epoch},
            }
        )
        _atomic_checkpoint(
            latest_checkpoint_path,
            _latest_checkpoint(
                model,
                optimizer,
                config=config,
                graph_sha256=graph_manifest["graph_sha256"],
                epoch=epoch,
                updates=updates,
                best_loss=best_loss,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                epoch_history=epoch_history,
            ),
        )
        _publish_live(live_path, payload, completed_epoch=epoch, archive=True)
        if early_stop_reached(epoch, bad_epochs, min_epochs=min_epochs, patience=patience):
            terminal_status = "early_stopped"
            break

    if not best_checkpoint_path.exists():
        raise RuntimeError("training ended without a best validation checkpoint")
    selected = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    with torch.no_grad():
        model.theta.copy_(selected["theta"])
    test_metrics, test_recent = evaluate(
        model,
        test_rows,
        device=device,
        batch_size=batch_size,
        ordinal_lambda=config["ordinal_lambda"],
    )
    test_constant = constant_answer_baseline(test_rows, answer=train_constant["answer"])
    strong = test_metrics["overall"]["accuracy"] >= 0.9 and all(
        test_metrics[name]["accuracy"] >= 0.8
        for name in ("no_ones_carry", "ones_carry", "sum_100")
        if name in test_metrics
    )
    partial = test_metrics["overall"]["accuracy"] > test_constant["accuracy"]
    conclusion = "strong_generalization" if strong else ("partial_learning" if partial else "no_detectable_learning")
    payload.update(
        addition_diagnostic_snapshot(
            model,
            probes,
            device=device,
            nodes_path=graph_dir / "nodes.feather",
            capture_indices=capture_indices,
            static_spatial=static_spatial,
            full_summary=True,
        )
    )
    payload.update(
        {
            "status": terminal_status,
            "run": {
                **payload["run"],
                "selectedEpoch": int(selected["epoch"]),
                "completedAt": datetime.now(timezone.utc).isoformat(),
            },
            "testMetrics": test_metrics,
            "testPredictions": test_recent,
            "baselines": {**payload["baselines"], "testConstant": test_constant},
            "conclusion": conclusion,
            "memory": _memory_snapshot(device),
        }
    )
    _publish_live(live_path, payload, completed_epoch=int(payload["run"]["epoch"]))
    _atomic_json(run_dir / "summary.json", payload)
    return payload
