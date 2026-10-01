from pathlib import Path

import json

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flystudy.addition_train import (
    _validate_resume,
    _validate_run_directory,
    early_stop_reached,
    run_addition_training,
    select_addition_probes,
)
from flystudy.dataset import ArithmeticDataset
from flystudy.graph import sha256_file
from flystudy.train import _load_graph


DATA = Path(__file__).parents[1] / "data" / "bounded-arithmetic-split-v1.json"


def test_addition_probes_are_validation_only_and_cover_five_cases():
    validation = ArithmeticDataset.from_json(DATA, "validation")
    train = ArithmeticDataset.from_json(DATA, "train")
    probes = select_addition_probes(validation)
    assert [row["probe"] for row in probes] == [
        "small_no_carry",
        "one_digit_carry",
        "two_digit_no_carry",
        "two_digit_carry",
        "sum_100",
    ]
    assert all(row["op"] == "+" for row in probes)
    train_families = {row["family"] for row in train.rows}
    assert all(row["family"] not in train_families for row in probes)


def test_early_stopping_respects_minimum_epoch_and_patience():
    assert not early_stop_reached(19, 30, min_epochs=20, patience=15)
    assert not early_stop_reached(20, 14, min_epochs=20, patience=15)
    assert early_stop_reached(20, 15, min_epochs=20, patience=15)


@pytest.mark.parametrize("field", ["seed", "interface_seed", "steps", "learning_rate",
                                    "dataset_sha256", "nodes_sha256", "graph_file_sha256"])
def test_resume_rejects_changed_experiment_settings(field):
    config = {"max_epochs": 100, "min_epochs": 20, "patience": 15, field: "original"}
    checkpoint = {"graph_sha256": "graph", "config": config, "epoch": 10}
    with pytest.raises(ValueError, match=field):
        _validate_resume(checkpoint, {**config, field: "changed"}, "graph")


def test_resume_rejects_graph_drift_missing_hashes_and_shortened_budget():
    config = {"max_epochs": 100, "dataset_sha256": "data", "nodes_sha256": "nodes"}
    checkpoint = {"graph_sha256": "graph", "config": config, "epoch": 10}
    with pytest.raises(ValueError, match="graph"):
        _validate_resume(checkpoint, config, "different")
    with pytest.raises(ValueError, match="cannot precede"):
        _validate_resume(checkpoint, {**config, "max_epochs": 9}, "graph")
    legacy = {**checkpoint, "config": {"max_epochs": 100}}
    with pytest.raises(ValueError, match="dataset_sha256"):
        _validate_resume(legacy, config, "graph")


def test_addition_run_directory_requires_explicit_resume(tmp_path):
    run_dir = tmp_path / "run"
    _validate_run_directory(run_dir, resume=False)
    with pytest.raises(FileNotFoundError, match="resume checkpoint"):
        _validate_run_directory(run_dir, resume=True)
    assert not run_dir.exists()
    run_dir.mkdir()
    _validate_run_directory(run_dir, resume=False)
    (run_dir / "train.log").write_text("launcher output\n")
    _validate_run_directory(run_dir, resume=False)
    (run_dir / "summary.json").write_text("{}")
    with pytest.raises(FileExistsError, match="not empty"):
        _validate_run_directory(run_dir, resume=False)
    (run_dir / "checkpoint-latest.pt").write_bytes(b"checkpoint")
    _validate_run_directory(run_dir, resume=True)


def test_graph_loader_checks_file_against_manifest(tmp_path):
    graph = tmp_path / "graph.npz"
    np.savez_compressed(graph, rows=np.array([1]), cols=np.array([0]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"graph_sha256": sha256_file(graph)}))
    arrays, loaded = _load_graph(tmp_path)
    assert loaded["graph_sha256"] == sha256_file(graph)
    np.testing.assert_array_equal(arrays["rows"], [1])
    for invalid in ({"graph_sha256": "wrong"}, {}):
        manifest.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="checksum"):
            _load_graph(tmp_path)


def test_addition_training_runs_exact_epochs_and_resumes(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    nodes = 200
    rows = np.arange(nodes, dtype=np.int64)
    cols = np.roll(rows, 1)
    np.savez_compressed(
        graph_dir / "graph.npz",
        rows=rows,
        cols=cols,
        counts=np.ones(nodes, dtype=np.float32),
        body_ids=np.arange(10_000, 10_000 + nodes, dtype=np.int64),
    )
    (graph_dir / "manifest.json").write_text(
        json.dumps({"graph_sha256": sha256_file(graph_dir / "graph.npz"),
                    "neurons": nodes, "edges": nodes})
    )
    feather.write_feather(
        pa.table(
            {
                "bodyId": np.arange(10_000, 10_000 + nodes),
                "superclass": ["test"] * nodes,
                "somaLocation": [[index, index % 10, index % 5] for index in range(nodes)],
            }
        ),
        graph_dir / "nodes.feather",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 11,
                "data_seed": 20260912,
                "interface_seed": 20260912,
                "steps": 2,
                "readout_steps": 1,
                "alpha": 0.5,
                "gain": 1.0,
                "input_amplitude": 1.0,
                "temperature": 0.1,
                "input_group_size": 1,
                "output_group_size": 1,
                "ordinal_lambda": 1.0,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip": 1.0,
                "microbatch_size": 128,
            }
        )
    )
    run_dir = tmp_path / "run"
    first = run_addition_training(
        graph_dir=graph_dir,
        dataset_path=DATA,
        config_path=config_path,
        run_dir=run_dir,
        max_epochs=1,
        min_epochs=1,
        patience=1,
        device_name="cpu",
    )
    assert first["status"] == "complete"
    assert first["run"]["epoch"] == 1
    assert first["run"]["update"] == 28
    assert first["activity"]["replay"]["steps"] == 2
    assert first["activity"]["replay"]["probeCount"] == 5
    assert first["testMetrics"]["overall"]["count"] == 792
    assert (run_dir / "checkpoint-latest.pt").exists()
    assert (run_dir / "checkpoint-best.pt").exists()

    saved_files = {path.relative_to(run_dir): sha256_file(path)
                   for path in run_dir.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="not empty"):
        run_addition_training(
            graph_dir=graph_dir, dataset_path=DATA, config_path=config_path,
            run_dir=run_dir, max_epochs=1, min_epochs=1, patience=1,
            device_name="cpu", overrides={"seed": 22},
        )
    assert saved_files == {path.relative_to(run_dir): sha256_file(path)
                           for path in run_dir.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="interface_seed"):
        run_addition_training(
            graph_dir=graph_dir, dataset_path=DATA, config_path=config_path,
            run_dir=run_dir, max_epochs=2, min_epochs=1, patience=2,
            device_name="cpu", resume=True, overrides={"interface_seed": 20260913},
        )

    resumed = run_addition_training(
        graph_dir=graph_dir,
        dataset_path=DATA,
        config_path=config_path,
        run_dir=run_dir,
        max_epochs=2,
        min_epochs=1,
        patience=2,
        device_name="cpu",
        resume=True,
    )
    assert resumed["run"]["resumed"] is True
    assert resumed["run"]["epoch"] == 2
    assert resumed["run"]["update"] == 56
    assert len(resumed["epochHistory"]) == 2
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 56
