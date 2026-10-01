from pathlib import Path
import json
import re


ROOT = Path(__file__).parents[1]


def test_package_version_matches_release_metadata():
    from flystudy import __version__

    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', project, re.M).group(1) == __version__
    citation_path = ROOT / "CITATION.cff"
    if citation_path.is_file():
        citation = citation_path.read_text(encoding="utf-8")
        assert re.search(r'^version: (\S+)', citation, re.M).group(1) == __version__


def test_executable_sources_have_role_based_names():
    for directory in (ROOT / "src/flystudy", ROOT / "scripts", ROOT / "tests"):
        for path in directory.glob("*.py"):
            assert not re.search(r"_v\d+", path.name), path
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"\b[vV][4-9]\b|language_v\d+|language-v[4-9]", text), path


def test_language_entry_points_use_published_paths():
    from flystudy.language_train import parser

    args = parser().parse_args(["--run-dir", "runs/example"])
    assert args.corpus == Path("data/language-corpus")
    assert (ROOT / "scripts/evaluate_language.py").is_file()
    assert (ROOT / "scripts/run_language.py").is_file()


def test_language_cli_defaults_match_the_paper_configuration():
    from flystudy.language_train import parser

    args = parser().parse_args(["--run-dir", "runs/example"])
    config = json.loads((ROOT / "configs/language_confirmatory.json").read_text())
    assert args.train_tokens == config["trainTokens"]
    assert args.warmup_tokens == config["warmupTokens"]
    assert args.checkpoint_every_tokens == config["checkpointEveryTokens"]
    assert args.seed == config["optimizationSeed"]
    assert args.interface_seed == config["canonicalInterfaceSeed"]
    assert args.codebook_seed == config["codebookSeed"]
    for attribute, field in (
        ("token_steps", "tokenSteps"),
        ("token_drive_mode", "tokenDriveMode"),
        ("learning_rate", "learningRate"),
        ("weight_decay", "weightDecay"),
        ("global_batch", "globalBatch"),
        ("microbatch", "microbatch"),
        ("encode_steps", "encodeSteps"),
        ("eval_batch", "evalBatch"),
        ("dimension", "dimension"),
        ("token_group", "tokenGroup"),
        ("output_group", "outputGroup"),
        ("leak", "leak"),
        ("token_gain", "tokenGain"),
        ("sensor_gain", "sensorGain"),
        ("temperature", "temperature"),
    ):
        assert getattr(args, attribute) == config["model"][field]


def test_anatomical_labels_and_task_vocabulary_remain_distinct():
    from flystudy.language_data import KINDS as sensory_kinds
    from flystudy.language_lexicon import KINDS as task_kinds, LEXICON

    assert sensory_kinds == ["tactile", "gustatory", "chordotonal", "campaniform"]
    assert task_kinds == ("touch", "taste", "movement", "pressure")
    assert len(LEXICON) == len(set(LEXICON)) == 100
