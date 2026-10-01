import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flystudy.language_training as engine


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    path = Path(__file__).parents[1] / "configs" / "language_confirmatory.json"
    return json.loads(path.read_text())


def test_evaluation_cli_defaults_match_the_paper_configuration():
    module = load_script("evaluate_language.py")
    args = module.parser().parse_args(["--run-dir", "runs/example"])
    evaluation = config()["evaluation"]
    assert args.pairs_per_family == evaluation["pairsPerFamily"]
    assert args.minimal_per_family == evaluation["minimalPerFamily"]
    assert args.intervention_pairs_per_family == evaluation["interventionPairsPerFamily"]
    assert args.orientation_permutations == evaluation["orientationPermutations"]


def test_trainer_accepts_any_single_visible_gpu(monkeypatch):
    selected = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("FLY_EXPECTED_GPU_UUID", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    device = engine.device_for(SimpleNamespace(tiny=False))
    assert device == torch.device("cuda:0")
    assert selected == [0]


def test_trainer_rejects_multiple_visible_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(RuntimeError, match="exactly one CUDA device"):
        engine.device_for(SimpleNamespace(tiny=False))


def test_campaign_plan_contains_every_unique_experimental_cell():
    module = load_script("run_language.py")
    rewires = [Path(f"r{index:02d}") for index in range(1, 21)]
    cells = module.build_all_cells(Path("real"), rewires, config())
    assert len(cells) == 48
    assert len({cell["name"] for cell in cells}) == 48
    assert sum("factorial" in cell["roles"] for cell in cells) == 20
    assert sum("null-rank" in cell["roles"] for cell in cells) == 21
    assert sum("baseline" in cell["roles"] for cell in cells) == 6
    assert sum("optimization-robustness" in cell["roles"] for cell in cells) == 9


@pytest.mark.parametrize("gpu_count", [1, 4, 5, 7, 8])
def test_full_campaign_schedules_partial_final_wave(tmp_path, monkeypatch, gpu_count):
    module = load_script("run_language.py")
    cfg = config()
    cells = module.build_all_cells(
        Path("real"), [Path(f"r{index:02d}") for index in range(1, 21)], cfg)
    monkeypatch.setattr(module.campaign, "ALLOWED_GPUS", tuple(range(gpu_count)))
    monkeypatch.setattr(module.campaign, "trained_state", lambda directory, budget: True)
    monkeypatch.setattr(module, "wait_for_gpus", lambda: [])
    states = []
    monkeypatch.setattr(module, "atomic_json", lambda path, value: states.append(value))
    analyses = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: analyses.append(command))
    for cell in cells:
        directory = tmp_path / "runs" / cell["name"]
        directory.mkdir(parents=True)
        (directory / "confirmatory-evaluation.json").write_text("{}")
    module.run_campaign(SimpleNamespace(output=tmp_path, corpus=Path("corpus")), cfg, cells)

    waves = [state for state in states if state.get("status") == "running"]
    assert len(waves) == (48 + gpu_count - 1) // gpu_count
    assert [name for state in waves for name in state["cells"]] == [cell["name"] for cell in cells]
    for index, cell in enumerate(cells):
        metadata = json.loads((tmp_path / "cells" / f"{cell['name']}.json").read_text())
        assert metadata["gpu"] == index % gpu_count
    assert len(analyses) == 1


def test_full_budget_commands_use_fixed_curriculum_and_codebook():
    module = load_script("run_language.py")
    cfg = config()
    cell = module.primary_cell("real", 20261031, Path("real"), condition="real")
    command = module.training_command(
        cell, Path("run"), Path("corpus"), cfg, resume=False)
    joined = " ".join(command)
    assert "--train-tokens 4000000" in joined
    assert "--warmup-tokens 1000000" in joined
    assert "--checkpoint-every-tokens 500000" in joined
    assert "--codebook-seed 20260914" in joined
    assert "--interface-seed 20261031" in joined
    assert "--token-drive-mode pulse" in joined


def test_pretraining_metadata_lock_does_not_create_trainer_run_directory(tmp_path):
    module = load_script("run_language.py")
    run_dir = tmp_path / "runs" / "real-i20261031-s11"
    metadata = {"name": "real-i20261031-s11", "trainTokens": 4_000_000}
    module.stage_cell_metadata(tmp_path, run_dir, metadata, training_complete=False)
    assert (tmp_path / "cells" / "real-i20261031-s11.json").is_file()
    assert not run_dir.exists()


def synthetic_row(topology, interface, value, seed=11):
    return {
        "name": f"{topology}-i{interface}-s{seed}",
        "origin": "test",
        "topology": topology,
        "condition": "real" if topology == "real" else "rewired",
        "graph": topology,
        "interfaceSeed": interface,
        "optimizationSeed": seed,
        "roles": [],
        "processedTokens": 4_000_000,
        "graphSha256": topology,
        "interfaceSha256": str(interface),
        "pairedNormal": value,
        "pairedInterventionNormal": value * .95,
        "pairedZero": 0.0,
        "pairedShuffle": value / 3,
        "pairedNormalGroups": 192,
        "pairedInterventionGroups": 48,
        "minimalReversalMacro": value / 2,
        "minimalStrictMacro": value / 4,
        "heldoutPairReversalMacro": value / 2,
        "heldoutPairStrictMacro": value / 4,
        "minimalPairs": {},
    }


def test_full_analysis_uses_interface_crossing_and_twenty_graph_rank():
    module = load_script("analyze_language.py")
    rows = []
    for interface in (20260914, 20261031, 20261117, 20261203):
        rows.append(synthetic_row("real", interface, .8))
        rows.extend(synthetic_row(f"rewire-{index:02d}", interface, .4 + index / 100)
                    for index in range(1, 5))
    rows.extend(synthetic_row(f"rewire-{index:02d}", 20260914, .3 + index / 200)
                for index in range(5, 21))
    summary = module.summarize({}, rows, bootstrap_samples=200, seed=7)
    assert summary["complete"]
    assert summary["factorial"]["realMean"] == .8
    assert abs(summary["factorial"]["rewiredMean"] - .425) < 1e-12
    assert "twoWayCrossedBootstrap95" in summary["factorial"]
    assert "hierarchicalBootstrap95" not in summary["factorial"]
    assert summary["nullRank"]["rewires"] == 20
    assert summary["nullRank"]["rewiresAtOrAboveReal"] == 0
    assert summary["nullRank"]["oneSidedRankP"] == 1 / 21


def test_analysis_report_uses_the_ensemble_rank_label():
    module = load_script("analyze_language.py")
    summary = {
        "generatedAt": "test",
        "complete": True,
        "factorial": None,
        "nullRank": {"real": .8, "rewiredMean": .4, "oneSidedRankP": 1 / 21},
    }
    report = module.markdown(summary, [])
    assert "Upper-tail ensemble rank fraction: 0.047619" in report
    assert "rank p:" not in report.lower()
