#!/usr/bin/env python3
"""Evaluate a final language checkpoint with the specified study metrics."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from flystudy.language_support import atomic_json, utcnow
from flystudy.language_metrics import family_subset
from flystudy.language_runtime import file_hash, load_records
from flystudy.language_training import device_for, restore
from flystudy.language_evaluation import (
    CHANNEL_CHANGING_FACTORS,
    PAIR_PRESERVING_FACTORS,
    matched_normal_subset,
    minimal_pair_factors,
    paired_relations_intervened,
    score_orientation_null,
)
from flystudy.language_train import build, config_for, parser as training_parser


EVALUATION_SOURCES = ("language_train.py", "language_evaluation.py")


def parser():
    arguments = training_parser()
    arguments.description = __doc__
    arguments.add_argument("--minimal-per-family", type=int, default=2)
    arguments.add_argument("--intervention-pairs-per-family", type=int, default=2)
    arguments.add_argument("--orientation-permutations", type=int, default=2000)
    return arguments


def main():
    args = parser().parse_args()
    minimal_per_family = args.minimal_per_family
    intervention_pairs_per_family = args.intervention_pairs_per_family
    orientation_permutations = args.orientation_permutations
    delattr(args, "minimal_per_family")
    delattr(args, "intervention_pairs_per_family")
    delattr(args, "orientation_permutations")
    if min(minimal_per_family, intervention_pairs_per_family,
           orientation_permutations) < 1:
        raise ValueError("confirmatory evaluation budgets must be positive")
    torch.set_num_threads(1 if args.tiny else 2)
    torch.manual_seed(args.seed)
    device = device_for(args)
    model, manifest = build(args, device)
    config = config_for(args, model, manifest)
    state_path = args.run_dir / "training-state.json"
    if not state_path.exists() or json.loads(state_path.read_text()).get("status") != "trained":
        raise ValueError("confirmatory evaluation requires a completed common token budget")
    destination = args.run_dir / "confirmatory-evaluation.json"
    if destination.exists():
        raise FileExistsError("confirmatory evaluation already exists")
    saved = restore(args.run_dir / "checkpoint-latest.pt", model, None, config, None)
    rows = load_records(args.corpus, "testComposition")
    pairs = family_subset(
        rows, args.pairs_per_family, reports_only=True, order_invariant=True)
    intervention_pairs = family_subset(
        rows, intervention_pairs_per_family,
        reports_only=True, order_invariant=True)
    minimal = family_subset(
        rows, minimal_per_family, reports_only=True, order_invariant=True)
    metrics = {}
    predictions = {}
    no_sensory = args.condition == "no_sensory"
    for intervention in ("normal", "zero", "shuffle"):
        name = "paired" + intervention.title()
        subset = pairs if intervention == "normal" else intervention_pairs
        metrics[name], predictions[name] = paired_relations_intervened(
            model,
            subset,
            device=device,
            batch_size=max(1, args.eval_batch // 4),
            encode_steps=args.encode_steps,
            intervention=intervention,
            no_sensory=no_sensory,
            deadline=args.deadline,
        )
    metrics["pairedInterventionNormal"], predictions["pairedInterventionNormal"] = (
        matched_normal_subset(
            predictions["pairedNormal"], predictions["pairedShuffle"])
    )
    metrics["minimalPairs"], predictions["minimalPairs"] = minimal_pair_factors(
        model,
        minimal,
        device=device,
        batch_size=max(1, args.eval_batch // 4),
        encode_steps=args.encode_steps,
        no_sensory=no_sensory,
        deadline=args.deadline,
    )
    metrics["orientationNull"] = score_orientation_null(
        predictions["pairedNormal"],
        permutations=orientation_permutations,
        seed=20260923,
    )
    source_root = Path(__file__).parents[1] / "src" / "flystudy"
    result = {
        "schema": 1,
        "at": utcnow(),
        "checkpoint": "checkpoint-latest.pt",
        "processedTokens": saved["mixer"]["total"],
        "graphSha256": manifest["graph_sha256"],
        "interfaceSha256": config["interfaceSha256"],
        "interfaceSeed": args.interface_seed,
        "optimizationSeed": args.seed,
        "primaryMetric": "pairedNormal.groupAccuracy",
        "evaluationCounts": {
            "primaryGroups": len(pairs),
            "interventionGroups": len(intervention_pairs),
            "minimalGroupsPerFactor": len(minimal),
        },
        "minimalPairScope": {
            "heldoutPairPreservingFactors": list(PAIR_PRESERVING_FACTORS),
            "channelChangingFactors": list(CHANNEL_CHANGING_FACTORS),
        },
        "results": metrics,
        "sourceHashes": {
            **{name: file_hash(source_root / name) for name in EVALUATION_SOURCES},
            Path(__file__).name: file_hash(Path(__file__)),
        },
    }
    atomic_json(args.run_dir / "confirmatory-predictions.json", predictions)
    atomic_json(destination, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
