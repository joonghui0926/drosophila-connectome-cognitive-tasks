"""Train language models with separate interface and fixed-codebook seeds."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from flystudy.language_model import fixed_codebook
from flystudy.language_runtime import file_hash
import flystudy.language_training as engine


_BASE_CONFIG_FOR = engine.config_for
SOURCE_FILES = ("language_train.py", "language_evaluation.py")


def apply_codebook_seed(model, codebook_seed: int) -> None:
    if codebook_seed < 0:
        raise ValueError("codebook seed must be nonnegative")
    input_values = fixed_codebook(
        "orthogonal", model.dimension, model.input_codebook.shape[1], codebook_seed + 1)
    output_values = fixed_codebook(
        "orthogonal", model.dimension, model.output_codebook.shape[1], codebook_seed + 2)
    with torch.no_grad():
        model.input_codebook.copy_(torch.from_numpy(input_values).to(model.input_codebook.device))
        model.output_codebook.copy_(torch.from_numpy(output_values).to(model.output_codebook.device))


def build(args, device):
    if args.embedding != "orthogonal":
        raise ValueError("the study requires orthogonal codebooks across all interfaces")
    model, manifest = engine.build(args, device)
    apply_codebook_seed(model, args.codebook_seed)
    return model, manifest


def source_hashes() -> dict:
    hashes = engine.source_hashes()
    root = Path(__file__).parent
    hashes.update({name: file_hash(root / name) for name in SOURCE_FILES})
    return hashes


def config_for(args, model, manifest):
    config = _BASE_CONFIG_FOR(args, model, manifest)
    config.update(
        schema=8,
        sourceHashes=source_hashes(),
        codebookSeed=args.codebook_seed,
        interfaceDefinition=(
            "interface_seed selects token input/output neurons; codebook_seed "
            f"independently fixes both {model.dimension}-dimensional codebooks"
        ),
    )
    return config


def parser():
    result = engine.parser()
    result.description = __doc__
    result.add_argument("--codebook-seed", type=int, default=20260914)
    return result


def _run_with_config(function, *args):
    original = engine.config_for
    engine.config_for = config_for
    try:
        return function(*args)
    finally:
        engine.config_for = original
        namespace = args[0]
        snapshot = namespace.run_dir / "source-snapshot"
        if snapshot.exists():
            source = Path(__file__)
            (snapshot / source.name).write_bytes(source.read_bytes())
            evaluation = source.with_name("language_evaluation.py")
            (snapshot / evaluation.name).write_bytes(evaluation.read_bytes())


def main():
    args = parser().parse_args()
    if args.codebook_seed < 0:
        raise ValueError("codebook seed must be nonnegative")
    torch.set_num_threads(1 if args.tiny else 2)
    torch.manual_seed(args.seed)
    device = engine.device_for(args)
    model, manifest = build(args, device)
    functions = {
        "train": engine.run_train,
        "benchmark": engine.run_benchmark,
        "diagnose": engine.run_diagnosis,
    }
    if args.phase not in functions:
        raise ValueError("final evaluation uses scripts/evaluate_language.py")
    result = _run_with_config(functions[args.phase], args, model, manifest, device)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
