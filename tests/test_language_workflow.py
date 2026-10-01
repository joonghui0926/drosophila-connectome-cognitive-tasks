import importlib.util
import gzip
import json
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest

from flystudy.language_corpus import make_record
from flystudy.language_runtime import file_hash


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_corpus(directory):
    directory.mkdir()
    files = {}
    for name, split, size in (("train", "train", 1000),
                              ("validationId", "validation_id", 100),
                              ("validationComposition", "validation_composition", 1024),
                              ("testComposition", "test_composition", 2048)):
        path = directory / (name + ".jsonl.gz")
        with gzip.open(path, "wt") as stream:
            for i in range(size):
                row = make_record(i, split=split)
                payload = {"id": row.record_id, "split": row.split, "mode": row.mode,
                           "templateFamily": row.template_family, "variant": row.variant,
                           "promptTokens": row.prompt_words, "answerTokens": row.answer_words,
                           "scene": [vars(event) for event in row.scene.events]}
                stream.write(json.dumps(payload) + "\n")
        files[name] = {"path": path.name, "examples": size, "sha256": file_hash(path)}
    (directory / "manifest.json").write_text(json.dumps({"revision": 2, "files": files}))


def test_factorial_matrix_crosses_every_topology_and_interface_once():
    module = load_script("language_campaign.py")
    real = Path("real")
    rewires = [Path(f"r{i:02d}") for i in range(1, 21)]
    interfaces = [101, 202, 303, 404]
    cells = module.factorial_cells(real, rewires, interfaces, optimization_seed=11)
    assert len(cells) == 20
    assert len({cell["name"] for cell in cells}) == 20
    expected_topologies = {"real", "rewire-01", "rewire-02", "rewire-03", "rewire-04"}
    assert {cell["topology"] for cell in cells} == expected_topologies
    assert {(cell["topology"], cell["interfaceSeed"]) for cell in cells} == {
        (topology, interface) for topology in expected_topologies for interface in interfaces
    }
    waves = module.schedule_waves(cells)
    assert len(waves) == 5
    assert all({job["gpu"] for job in wave} == {0, 1, 2, 3} for wave in waves)
    assert all(len({job["topology"] for job in wave}) == 4 for wave in waves)
    assert all(len({job["interfaceSeed"] for job in wave}) == 4 for wave in waves)
    for interface in interfaces:
        used = {job["gpu"] for wave in waves for job in wave
                if job["interfaceSeed"] == interface}
        assert used == {0, 1, 2, 3}
    for topology in expected_topologies:
        used = {job["gpu"] for wave in waves for job in wave
                if job["topology"] == topology}
        assert used == {0, 1, 2, 3}
    assert sorted(job["name"] for wave in waves for job in wave) == sorted(
        cell["name"] for cell in cells)


def test_null_rank_plan_uses_twenty_independent_rewires_and_one_real_graph():
    module = load_script("language_campaign.py")
    rewires = [Path(f"r{i:02d}") for i in range(1, 21)]
    jobs = module.null_rank_cells(
        Path("real"), rewires, canonical_interface=101, optimization_seed=11)
    assert len(jobs) == 21
    assert sum(job["condition"] == "real" for job in jobs) == 1
    assert sum(job["condition"] == "rewired" for job in jobs) == 20
    assert len({job["graph"] for job in jobs if job["condition"] == "rewired"}) == 20
    waves = module.schedule_waves(jobs)
    assert sum(len(wave) for wave in waves) == 21
    assert all({job["gpu"] for job in wave} <= {0, 1, 2, 3} for wave in waves)


def test_analysis_uses_graph_level_rank_and_interface_paired_effects():
    module = load_script("analyze_language_design.py")
    rows = []
    for interface, real in ((101, .8), (202, .7)):
        rows.append({"stage": "factorial", "topology": "real",
                     "interfaceSeed": interface, "pairedNormal": real})
        rows.extend({"stage": "factorial", "topology": f"rewire-{i:02d}",
                     "interfaceSeed": interface, "pairedNormal": value}
                    for i, value in enumerate((.4, .5), 1))
    rows.append({"stage": "null-rank", "topology": "real",
                 "interfaceSeed": 101, "pairedNormal": .75})
    rows.extend({"stage": "null-rank", "topology": f"rewire-{i:02d}",
                 "interfaceSeed": 101, "pairedNormal": value}
                for i, value in enumerate((.2, .3, .8), 1))
    with pytest.raises(ValueError, match="all 20 factorial and 21 null-rank"):
        module.summarize(rows, bootstrap_samples=200, seed=5)
    result = module.summarize(
        rows, bootstrap_samples=200, seed=5, require_complete=False)
    assert result["factorial"]["realMean"] == .75
    assert result["factorial"]["rewiredMean"] == .45
    assert abs(result["factorial"]["pairedDifference"] - .30) < 1e-12
    assert result["nullRank"]["rewires"] == 3
    assert result["nullRank"]["oneSidedRankP"] == .5
    assert not result["completeDesign"]


def test_final_analysis_accepts_only_the_complete_locked_design():
    module = load_script("analyze_language_design.py")
    rows = []
    for interface in (101, 202, 303, 404):
        rows.append({"stage": "factorial", "topology": "real",
                     "interfaceSeed": interface, "pairedNormal": .8})
        rows.extend({"stage": "factorial", "topology": f"rewire-{index:02d}",
                     "interfaceSeed": interface, "pairedNormal": .5}
                    for index in range(1, 5))
    rows.append({"stage": "null-rank", "topology": "real",
                 "interfaceSeed": 101, "pairedNormal": .8})
    rows.extend({"stage": "null-rank", "topology": f"rewire-{index:02d}",
                 "interfaceSeed": 101, "pairedNormal": .4 + index / 100}
                for index in range(1, 21))
    result = module.summarize(rows, bootstrap_samples=50, seed=7)
    assert result["completeDesign"]
    assert result["factorial"]["interfaces"] == 4
    assert result["factorial"]["rewireGraphs"] == 4
    assert result["nullRank"]["rewires"] == 20


def test_completed_state_must_match_locked_budget_and_artifacts(tmp_path):
    module = load_script("language_campaign.py")
    run = tmp_path / "run"
    run.mkdir()
    state = {
        "status": "trained",
        "budgetTokens": 500,
        "processedTokens": 493,
        "unusedTokens": 7,
        "maxRecordTokens": 20,
    }
    (run / "training-state.json").write_text(json.dumps(state))
    (run / "checkpoint-latest.pt").write_bytes(b"checkpoint")
    (run / "config.json").write_text("{}")
    assert module.trained_state(run, 500)
    with pytest.raises(ValueError, match="locked token budget"):
        module.trained_state(run, 600)


def test_confirmatory_cell_metadata_is_immutable(tmp_path):
    module = load_script("language_campaign.py")
    path = tmp_path / "confirmatory-cell.json"
    module.write_locked_cell(path, {"name": "real-i1", "tokensPerRun": 500})
    module.write_locked_cell(path, {"name": "real-i1", "tokensPerRun": 500})
    with pytest.raises(ValueError, match="changed on resume"):
        module.write_locked_cell(path, {"name": "real-i1", "tokensPerRun": 250})


def test_graph_ensemble_is_locked_by_source_seed_metadata_and_unique_hash(tmp_path):
    module = load_script("language_campaign.py")
    real = tmp_path / "real"
    real.mkdir()
    (real / "graph.npz").write_bytes(b"real-graph")
    (real / "nodes.feather").write_bytes(b"same-neuron-metadata")
    real_sha = file_hash(real / "graph.npz")
    (real / "manifest.json").write_text(json.dumps({
        "graph_sha256": real_sha, "neurons": 3, "edges": 1,
    }))
    seeds = list(range(100, 120))
    rewires = []
    for index, seed in enumerate(seeds, 1):
        path = tmp_path / f"rewire-{index:02d}"
        path.mkdir()
        (path / "graph.npz").write_bytes(f"rewire-{index}".encode())
        (path / "nodes.feather").write_bytes(b"same-neuron-metadata")
        (path / "manifest.json").write_text(json.dumps({
            "graph_sha256": file_hash(path / "graph.npz"),
            "neurons": 3,
            "edges": 1,
            "control": "directed-degree-preserving-rewire",
            "sourceGraphSha256": real_sha,
            "rewiring": {"seed": seed, "acceptedSwaps": 5},
        }))
        rewires.append(path)
    real_provenance, controls = module.validate_graph_ensemble(
        real, rewires, {"rewireSeeds": seeds})
    assert real_provenance["graphSha256"] == real_sha
    assert len(controls) == 20
    (rewires[-1] / "graph.npz").write_bytes((rewires[0] / "graph.npz").read_bytes())
    duplicate = json.loads((rewires[-1] / "manifest.json").read_text())
    duplicate["graph_sha256"] = file_hash(rewires[-1] / "graph.npz")
    (rewires[-1] / "manifest.json").write_text(json.dumps(duplicate))
    with pytest.raises(ValueError, match="independently materialized"):
        module.validate_graph_ensemble(real, rewires, {"rewireSeeds": seeds})


def test_rewire_builder_publishes_only_a_complete_atomic_directory(tmp_path, monkeypatch):
    if shutil.which("g++") is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "뉴런-source"
    source.mkdir()
    n = 30
    rows = np.repeat(np.arange(n, dtype=np.int32), 3)
    cols = np.asarray([(i + offset) % n for i in range(n)
                       for offset in (0, 1, 4)], dtype=np.int32)
    crow = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=n)))).astype(np.int32)
    np.savez_compressed(
        source / "graph.npz",
        rows=rows,
        cols=cols,
        crow=crow,
        counts=np.ones(len(rows), dtype=np.int32),
        body_ids=np.arange(n, dtype=np.int64),
    )
    (source / "nodes.feather").write_bytes(b"test-neuron-metadata")
    (source / "manifest.json").write_text(json.dumps({
        "graph_sha256": file_hash(source / "graph.npz"),
        "neurons": n,
        "edges": len(rows),
    }))
    output = tmp_path / "재배선-rewired"
    compiler_temp = tmp_path / "컴파일-temp"
    compiler_temp.mkdir()
    for variable in ("TMP", "TEMP", "TMPDIR"):
        monkeypatch.setenv(variable, str(compiler_temp))
    script = Path(__file__).parents[1] / "scripts" / "prepare_language_rewire.py"
    subprocess.run([
        sys.executable, str(script),
        "--source", str(source),
        "--output", str(output),
        "--seed", "123",
        "--swaps-per-edge", "1",
    ], check=True, capture_output=True, text=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert file_hash(output / "graph.npz") == manifest["graph_sha256"]
    assert manifest["rewiring"]["acceptedSwaps"] >= len(rows)
    assert (output / "nodes.feather").read_bytes() == (source / "nodes.feather").read_bytes()
    assert not (output / "nodes.feather").is_symlink()
    assert not list(tmp_path.glob(output.name + ".building-*"))
    assert not list(compiler_temp.iterdir())
    with np.load(output / "graph.npz") as archive:
        new_rows, new_cols = archive["rows"], archive["cols"]
    np.testing.assert_array_equal(np.bincount(new_rows, minlength=n),
                                  np.bincount(rows, minlength=n))
    np.testing.assert_array_equal(np.bincount(new_cols, minlength=n),
                                  np.bincount(cols, minlength=n))
    np.testing.assert_array_equal(new_rows[new_rows == new_cols], rows[rows == cols])
    assert len(set(zip(new_rows.tolist(), new_cols.tolist()))) == len(rows)
    published_hash = file_hash(output / "graph.npz")
    second = subprocess.run([
        sys.executable, str(script), "--source", str(source), "--output", str(output),
    ], capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing to replace" in second.stderr
    assert file_hash(output / "graph.npz") == published_hash


def test_ensemble_reuses_graphs_without_symlink_permissions(tmp_path, monkeypatch):
    module = load_script("prepare_language_rewire_ensemble.py")
    source = tmp_path / "source"
    source.mkdir()
    (source / "graph.npz").write_bytes(b"source graph")
    (source / "nodes.feather").write_bytes(b"neuron metadata")
    source_sha = file_hash(source / "graph.npz")
    (source / "manifest.json").write_text(json.dumps({
        "graph_sha256": source_sha, "neurons": 3, "edges": 1,
    }))
    seeds = list(range(100, 120))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"rewireSeeds": seeds}))
    existing = []
    for index, seed in enumerate(seeds):
        graph = tmp_path / f"existing-{index}"
        graph.mkdir()
        (graph / "graph.npz").write_bytes(f"rewire {index}".encode())
        (graph / "nodes.feather").write_bytes(b"neuron metadata")
        (graph / "manifest.json").write_text(json.dumps({
            "graph_sha256": file_hash(graph / "graph.npz"),
            "sourceGraphSha256": source_sha,
            "neurons": 3, "edges": 1,
            "control": "directed-degree-preserving-rewire",
            "rewiring": {"seed": seed, "swapsPerEdge": 5, "acceptedSwaps": 5},
        }))
        existing.append(graph)
    output = tmp_path / "ensemble"
    arguments = ["prepare_language_rewire_ensemble.py", "--source", str(source),
                 "--output-root", str(output), "--config", str(config), "--execute"]
    for graph in existing:
        arguments.extend(["--existing", str(graph)])
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(module, "audit_graphs", lambda source, graphs: {"passed": True})

    def forbidden_symlink(*args, **kwargs):
        pytest.fail("reusing a graph must not require symlink permissions")

    monkeypatch.setattr(Path, "symlink_to", forbidden_symlink)
    module.main()
    module.main()
    for index, graph in enumerate(existing, 1):
        slot = output / f"rewire-{index:02d}"
        assert not slot.is_symlink()
        for name in ("graph.npz", "nodes.feather", "manifest.json"):
            assert (slot / name).read_bytes() == (graph / name).read_bytes()
    assert not list(output.glob("*.building-*"))
    assert (output / "ensemble-complete.json").is_file()


def test_frozen_checkpoint_runs_confirmatory_cli_without_config_drift(tmp_path):
    corpus = tmp_path / "corpus"
    run = tmp_path / "run"
    write_corpus(corpus)
    common = [
        "--tiny", "--corpus", str(corpus), "--run-dir", str(run),
        "--train-tokens", "400", "--warmup-tokens", "400",
        "--global-batch", "2", "--microbatch", "1",
        "--validation-per-family", "1", "--test-per-family", "1",
        "--pairs-per-family", "1", "--intervention-examples", "24",
        "--id-examples", "2", "--eval-batch", "4",
        "--checkpoint-every-tokens", "400",
    ]
    subprocess.run([
        sys.executable, "-m", "flystudy.language_train", "--phase", "train",
        "--codebook-seed", "20260914", *common,
    ], check=True, capture_output=True, text=True)
    script = Path(__file__).parents[1] / "scripts" / "evaluate_language.py"
    subprocess.run([
        sys.executable, str(script), "--phase", "evaluate",
        "--codebook-seed", "20260914", *common,
        "--minimal-per-family", "1", "--intervention-pairs-per-family", "1",
        "--orientation-permutations", "20",
    ], check=True, capture_output=True, text=True)
    result = json.loads((run / "confirmatory-evaluation.json").read_text())
    assert result["processedTokens"] <= 400
    assert result["primaryMetric"] == "pairedNormal.groupAccuracy"
    assert set(result["results"]) == {
        "pairedNormal", "pairedInterventionNormal", "pairedZero",
        "pairedShuffle", "minimalPairs", "orientationNull",
    }
