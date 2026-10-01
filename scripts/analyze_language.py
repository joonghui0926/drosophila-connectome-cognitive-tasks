#!/usr/bin/env python3
"""Aggregate the full-budget language campaign at graph and interface level."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from flystudy.language_support import atomic_json, utcnow


PAIR_PRESERVING = ("level", "duration", "speed", "order")


def _with_matched_intervention(evaluation: dict, predictions_path: Path) -> dict:
    """Attach the normal score on exactly the groups used by sensory shuffle."""
    results = evaluation["results"]
    if "pairedInterventionNormal" in results:
        return evaluation
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"predictions required for matched intervention score: {predictions_path}")
    predictions = json.loads(predictions_path.read_text())
    normal = {row["id"]: row for row in predictions["pairedNormal"]}
    if len(normal) != len(predictions["pairedNormal"]):
        raise ValueError(f"duplicate normal group IDs: {predictions_path}")
    ids = [row["id"] for row in predictions["pairedShuffle"]]
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(normal):
        raise ValueError(f"shuffle groups do not form a normal-score subset: {predictions_path}")
    selected = [normal[value] for value in ids]
    matched = {
        "groups": len(selected),
        "groupAccuracy": sum(row["correct"] for row in selected) / len(selected),
        "tiePolicy": "incorrect",
        "comparisonsPerGroup": 4,
        "intervention": "normal",
        "matchedTo": "pairedShuffle",
        "interpretation": (
            "normal-sensory score on exactly the groups used by the shuffle control"
        ),
    }
    return {
        **evaluation,
        "results": {**results, "pairedInterventionNormal": matched},
    }


def _evaluation_row(cell: dict, evaluation: dict, *, origin: str) -> dict:
    if evaluation.get("processedTokens", 0) < 3_999_800:
        raise ValueError(f"full-budget evaluation is below the locked endpoint: {cell['name']}")
    if evaluation.get("interfaceSeed") != cell["interfaceSeed"]:
        raise ValueError(f"interface mismatch: {cell['name']}")
    if evaluation.get("optimizationSeed") != cell["optimizationSeed"]:
        raise ValueError(f"optimization seed mismatch: {cell['name']}")
    results = evaluation["results"]
    factors = results["minimalPairs"]
    return {
        "name": cell["name"],
        "origin": origin,
        "topology": cell["topology"],
        "condition": cell["condition"],
        "graph": cell["graph"],
        "interfaceSeed": cell["interfaceSeed"],
        "optimizationSeed": cell["optimizationSeed"],
        "roles": cell["roles"],
        "processedTokens": evaluation["processedTokens"],
        "graphSha256": evaluation["graphSha256"],
        "interfaceSha256": evaluation["interfaceSha256"],
        "pairedNormal": results["pairedNormal"]["groupAccuracy"],
        "pairedInterventionNormal": results["pairedInterventionNormal"]["groupAccuracy"],
        "pairedZero": results["pairedZero"]["groupAccuracy"],
        "pairedShuffle": results["pairedShuffle"]["groupAccuracy"],
        "pairedNormalGroups": results["pairedNormal"]["groups"],
        "pairedInterventionGroups": results["pairedShuffle"]["groups"],
        "minimalReversalMacro": float(np.mean([
            value["preferenceReversalAccuracy"] for value in factors.values()])),
        "minimalStrictMacro": float(np.mean([
            value["strictGroupAccuracy"] for value in factors.values()])),
        "heldoutPairReversalMacro": float(np.mean([
            factors[name]["preferenceReversalAccuracy"] for name in PAIR_PRESERVING])),
        "heldoutPairStrictMacro": float(np.mean([
            factors[name]["strictGroupAccuracy"] for name in PAIR_PRESERVING])),
        "minimalPairs": {
            name: {
                "originalPreferenceAccuracy": value["originalPreferenceAccuracy"],
                "counterfactualPreferenceAccuracy": value["counterfactualPreferenceAccuracy"],
                "preferenceReversalAccuracy": value["preferenceReversalAccuracy"],
                "strictGroupAccuracy": value["strictGroupAccuracy"],
            }
            for name, value in factors.items()
        },
    }


def collect(campaign_root: Path, *, require_complete: bool = True) -> tuple[dict, list[dict]]:
    plan_path = campaign_root / "campaign-plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError("campaign-plan.json is required")
    plan = json.loads(plan_path.read_text())
    graph_hashes = {
        plan["realGraphProvenance"]["path"]: plan["realGraphProvenance"]["graphSha256"],
        **{item["path"]: item["graphSha256"]
           for item in plan["rewireGraphProvenance"]},
    }
    rows = []
    missing = []
    for cell in plan["cells"]:
        run_dir = campaign_root / "runs" / cell["name"]
        evaluation_path = run_dir / "confirmatory-evaluation.json"
        state_path = run_dir / "training-state.json"
        if not evaluation_path.is_file() or not state_path.is_file():
            missing.append(cell["name"])
            continue
        state = json.loads(state_path.read_text())
        if state.get("status") != "trained" or state.get("budgetTokens") != plan["trainTokens"]:
            raise ValueError(f"full-budget state is invalid: {cell['name']}")
        evaluation = _with_matched_intervention(
            json.loads(evaluation_path.read_text()),
            run_dir / "confirmatory-predictions.json",
        )
        if evaluation.get("graphSha256") != graph_hashes[cell["graph"]]:
            raise ValueError(f"graph mismatch: {cell['name']}")
        rows.append(_evaluation_row(cell, evaluation, origin="campaign"))
    if require_complete and missing:
        raise ValueError(f"full campaign has {len(missing)} incomplete cells")
    return plan, rows


def _mean(rows: list[dict], key: str) -> float | None:
    return float(np.mean([row[key] for row in rows])) if rows else None


def summarize(plan: dict, rows: list[dict], *, bootstrap_samples: int = 20_000,
              seed: int = 20260924, require_complete: bool = True) -> dict:
    canonical = 20260914
    primary = [row for row in rows
               if row["optimizationSeed"] == 11
               and row["topology"] in {"real", "rewire-01", "rewire-02", "rewire-03", "rewire-04"}]
    by_interface: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in primary:
        by_interface[row["interfaceSeed"]][row["topology"]] = row
    interfaces = sorted(by_interface)
    factorial_complete = len(interfaces) == 4 and all(
        set(by_interface[value]) == {"real", "rewire-01", "rewire-02", "rewire-03", "rewire-04"}
        for value in interfaces)
    if require_complete and not factorial_complete:
        raise ValueError("the full 5 by 4 topology-interface crossing is incomplete")

    factorial = None
    if factorial_complete:
        real = np.asarray([by_interface[i]["real"]["pairedNormal"] for i in interfaces])
        rewired = np.asarray([
            [by_interface[i][f"rewire-{j:02d}"]["pairedNormal"] for j in range(1, 5)]
            for i in interfaces])
        differences = real - rewired.mean(axis=1)
        rng = np.random.default_rng(seed)
        bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
        for index in range(bootstrap_samples):
            sampled_interfaces = rng.integers(0, 4, 4)
            sampled_graphs = rng.integers(0, 4, 4)
            bootstrap[index] = (
                real[sampled_interfaces].mean()
                - rewired[np.ix_(sampled_interfaces, sampled_graphs)].mean())
        factorial_rows = [row for value in by_interface.values() for row in value.values()]
        factorial = {
            "interfaces": interfaces,
            "realMean": float(real.mean()),
            "rewiredMean": float(rewired.mean()),
            "pairedDifference": float(differences.mean()),
            "differenceByInterface": {
                str(interface): float(value)
                for interface, value in zip(interfaces, differences, strict=True)},
            "twoWayCrossedBootstrap95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "bootstrapSamples": bootstrap_samples,
            "bootstrapMethod": (
                "independent resampling of the interface and rewire-identity axes"
            ),
            "realInterventionNormalMean": _mean(
                [row for row in factorial_rows if row["topology"] == "real"],
                "pairedInterventionNormal"),
            "rewiredInterventionNormalMean": _mean(
                [row for row in factorial_rows if row["topology"] != "real"],
                "pairedInterventionNormal"),
            "realShuffleMean": _mean(
                [row for row in factorial_rows if row["topology"] == "real"], "pairedShuffle"),
            "rewiredShuffleMean": _mean(
                [row for row in factorial_rows if row["topology"] != "real"], "pairedShuffle"),
            "realHeldoutReversalMean": _mean(
                [row for row in factorial_rows if row["topology"] == "real"],
                "heldoutPairReversalMacro"),
            "rewiredHeldoutReversalMean": _mean(
                [row for row in factorial_rows if row["topology"] != "real"],
                "heldoutPairReversalMacro"),
        }

    null_rows = [row for row in rows
                 if row["optimizationSeed"] == 11
                 and row["interfaceSeed"] == canonical
                 and (row["topology"] == "real" or row["topology"].startswith("rewire-"))]
    null_by_topology = {row["topology"]: row for row in null_rows}
    expected_null = {"real", *(f"rewire-{index:02d}" for index in range(1, 21))}
    null_complete = set(null_by_topology) == expected_null
    if require_complete and not null_complete:
        raise ValueError("the twenty-graph full-budget null distribution is incomplete")
    null_rank = None
    if "real" in null_by_topology:
        real_value = null_by_topology["real"]["pairedNormal"]
        null_values = [row["pairedNormal"] for name, row in null_by_topology.items()
                       if name != "real"]
        if null_values:
            exceed = sum(value >= real_value for value in null_values)
            null_rank = {
                "real": real_value,
                "rewires": len(null_values),
                "rewiredMean": float(np.mean(null_values)),
                "rewiredStandardDeviation": float(np.std(null_values, ddof=1)),
                "rewiredRange": [float(np.min(null_values)), float(np.max(null_values))],
                "rewiresAtOrAboveReal": exceed,
                "oneSidedRankP": (1 + exceed) / (1 + len(null_values)),
                "testUnit": "independently rewired graph",
            }

    baselines = {}
    for topology in ("dense", "no-sensory"):
        selected = [row for row in rows if row["topology"] == topology]
        baselines[topology] = {
            "runs": len(selected),
            "pairedNormalMean": _mean(selected, "pairedNormal"),
            "pairedShuffleMean": _mean(selected, "pairedShuffle"),
            "heldoutPairReversalMean": _mean(selected, "heldoutPairReversalMacro"),
        }

    robustness = {}
    for topology in ("real", "rewire-01", "rewire-02"):
        selected = [row for row in rows
                    if row["topology"] == topology
                    and row["interfaceSeed"] == canonical
                    and row["optimizationSeed"] in (11, 22, 33)]
        robustness[topology] = {
            "seeds": sorted(row["optimizationSeed"] for row in selected),
            "pairedNormalMean": _mean(selected, "pairedNormal"),
            "pairedNormalStandardDeviation": (
                float(np.std([row["pairedNormal"] for row in selected], ddof=1))
                if len(selected) > 1 else None),
        }

    return {
        "schema": 1,
        "generatedAt": utcnow(),
        "complete": bool(factorial_complete and null_complete),
        "runsCollected": len(rows),
        "factorial": factorial,
        "nullRank": null_rank,
        "baselines": baselines,
        "optimizationRobustness": robustness,
        "interpretationRule": (
            "A topology-associated inductive bias requires a positive topology-interface effect, "
            "a graph-level rank result, and normal sensory performance above shuffled sensory performance."
        ),
    }


def markdown(summary: dict, rows: list[dict]) -> str:
    lines = [
        "# Full-budget language results", "",
        f"Generated: {summary['generatedAt']}", "",
        f"Complete locked design: {summary['complete']}", "",
    ]
    if summary["factorial"]:
        value = summary["factorial"]
        lines += [
            "## Topology by interface", "",
            f"Real mean: {value['realMean']:.4f}", "",
            f"Rewired mean: {value['rewiredMean']:.4f}", "",
            f"Paired difference: {value['pairedDifference']:.4f}", "",
            f"Two-way crossed bootstrap 95%: {value['twoWayCrossedBootstrap95']}", "",
            f"Matched intervention groups per run: 48", "",
            f"Real matched normal -> shuffle: "
            f"{value['realInterventionNormalMean']:.4f} -> {value['realShuffleMean']:.4f}", "",
            f"Rewired matched normal -> shuffle: "
            f"{value['rewiredInterventionNormalMean']:.4f} -> {value['rewiredShuffleMean']:.4f}", "",
        ]
    if summary["nullRank"]:
        value = summary["nullRank"]
        lines += [
            "## Twenty-graph null", "",
            f"Real: {value['real']:.4f}", "",
            f"Rewired mean: {value['rewiredMean']:.4f}", "",
            f"Upper-tail ensemble rank fraction: {value['oneSidedRankP']:.6f}", "",
        ]
    lines += [
        "## Per-run results", "",
        "| Topology | Interface | Seed | Paired | Matched normal | Zero | Shuffle | Held-out reversal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (
            item["topology"], item["interfaceSeed"], item["optimizationSeed"])):
        lines.append(
            f"| {row['topology']} | {row['interfaceSeed']} | "
            f"{row['optimizationSeed']} | {row['pairedNormal']:.4f} | "
            f"{row['pairedInterventionNormal']:.4f} | {row['pairedZero']:.4f} | "
            f"{row['pairedShuffle']:.4f} | "
            f"{row['heldoutPairReversalMacro']:.4f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    plan, rows = collect(args.campaign_root, require_complete=not args.allow_incomplete)
    summary = summarize(
        plan, rows, bootstrap_samples=args.bootstrap_samples,
        require_complete=not args.allow_incomplete)
    output = args.output or args.campaign_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "runs.json", rows)
    atomic_json(output / "summary.json", summary)
    (output / "report.md").write_text(markdown(summary, rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
