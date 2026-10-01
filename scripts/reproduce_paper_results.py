#!/usr/bin/env python3
"""Recompute the paper's headline values from the compact run records."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def mean(rows: list[dict], key: str) -> float:
    if not rows:
        raise ValueError(f"cannot average an empty selection for {key}")
    return float(np.mean([row[key] for row in rows]))


def addition_results() -> dict:
    rows = read_json(ROOT / "results" / "addition" / "runs.json")
    anatomical = [row for row in rows if row["topology"] == "anatomical"]
    rewired = [row for row in rows if row["topology"].startswith("rewire-")]
    if len(anatomical) != 4 or len(rewired) != 8:
        raise ValueError("addition records must contain four anatomical and eight rewired runs")
    anatomical_mean = mean(
        [{"accuracy": row["testMetrics"]["overall"]["accuracy"]} for row in anatomical],
        "accuracy",
    )
    rewired_mean = mean(
        [{"accuracy": row["testMetrics"]["overall"]["accuracy"]} for row in rewired],
        "accuracy",
    )
    return {
        "anatomicalRuns": len(anatomical),
        "rewiredRuns": len(rewired),
        "anatomicalMean": anatomical_mean,
        "rewiredMean": rewired_mean,
        "difference": anatomical_mean - rewired_mean,
    }


def language_results(bootstrap_samples: int = 20_000, seed: int = 20260924) -> dict:
    rows = read_json(ROOT / "results" / "language" / "runs.json")
    primary_names = {"real", "rewire-01", "rewire-02", "rewire-03", "rewire-04"}
    primary = [
        row for row in rows
        if row["optimizationSeed"] == 11 and row["topology"] in primary_names
    ]
    by_interface: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in primary:
        by_interface[row["interfaceSeed"]][row["topology"]] = row
    interfaces = sorted(by_interface)
    if len(interfaces) != 4 or any(set(by_interface[value]) != primary_names for value in interfaces):
        raise ValueError("language records do not contain the complete 5 by 4 crossing")

    real = np.asarray([by_interface[value]["real"]["pairedNormal"] for value in interfaces])
    rewired = np.asarray([
        [by_interface[value][f"rewire-{index:02d}"]["pairedNormal"]
         for index in range(1, 5)]
        for value in interfaces
    ])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled_interfaces = rng.integers(0, 4, 4)
        sampled_graphs = rng.integers(0, 4, 4)
        bootstrap[index] = (
            real[sampled_interfaces].mean()
            - rewired[np.ix_(sampled_interfaces, sampled_graphs)].mean()
        )

    factorial_rows = [row for value in by_interface.values() for row in value.values()]
    matched_real = [row for row in factorial_rows if row["topology"] == "real"]
    matched_rewired = [row for row in factorial_rows if row["topology"] != "real"]

    canonical = 20260914
    null_rows = [
        row for row in rows
        if row["optimizationSeed"] == 11
        and row["interfaceSeed"] == canonical
        and (row["topology"] == "real" or row["topology"].startswith("rewire-"))
    ]
    null_by_topology = {row["topology"]: row for row in null_rows}
    expected = {"real", *(f"rewire-{index:02d}" for index in range(1, 21))}
    if set(null_by_topology) != expected:
        raise ValueError("language records do not contain the complete 21-graph comparison")
    real_null = null_by_topology["real"]["pairedNormal"]
    rewire_null = np.asarray([
        null_by_topology[f"rewire-{index:02d}"]["pairedNormal"]
        for index in range(1, 21)
    ])
    exceedances = int(np.sum(rewire_null >= real_null))

    return {
        "factorial": {
            "interfaces": interfaces,
            "anatomicalMean": float(real.mean()),
            "rewiredMean": float(rewired.mean()),
            "difference": float(real.mean() - rewired.mean()),
            "twoWayCrossedBootstrap95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "bootstrapSamples": bootstrap_samples,
            "matchedNormalAnatomical": mean(matched_real, "pairedInterventionNormal"),
            "matchedShuffleAnatomical": mean(matched_real, "pairedShuffle"),
            "matchedNormalRewired": mean(matched_rewired, "pairedInterventionNormal"),
            "matchedShuffleRewired": mean(matched_rewired, "pairedShuffle"),
        },
        "nullRank": {
            "anatomical": real_null,
            "rewiredMean": float(rewire_null.mean()),
            "rewiredStandardDeviation": float(rewire_null.std(ddof=1)),
            "rewiredRange": [float(rewire_null.min()), float(rewire_null.max())],
            "rewiresAtOrAboveAnatomical": exceedances,
            "oneSidedRankFraction": (1 + exceedances) / 21,
        },
    }


def verify_saved_summary(language: dict) -> None:
    saved = read_json(ROOT / "results" / "language" / "summary.json")
    comparisons = (
        (language["factorial"]["anatomicalMean"], saved["factorial"]["realMean"]),
        (language["factorial"]["rewiredMean"], saved["factorial"]["rewiredMean"]),
        (language["factorial"]["difference"], saved["factorial"]["pairedDifference"]),
        (language["factorial"]["matchedNormalAnatomical"],
         saved["factorial"]["realInterventionNormalMean"]),
        (language["factorial"]["matchedShuffleAnatomical"],
         saved["factorial"]["realShuffleMean"]),
        (language["nullRank"]["anatomical"], saved["nullRank"]["real"]),
        (language["nullRank"]["rewiredMean"], saved["nullRank"]["rewiredMean"]),
        (language["nullRank"]["oneSidedRankFraction"], saved["nullRank"]["oneSidedRankP"]),
    )
    if any(not np.isclose(left, right, rtol=0, atol=1e-12) for left, right in comparisons):
        raise AssertionError("recomputed language values do not match the canonical summary")
    if not np.allclose(
        language["factorial"]["twoWayCrossedBootstrap95"],
        saved["factorial"]["twoWayCrossedBootstrap95"],
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("recomputed bootstrap interval does not match the canonical summary")


def calculate() -> dict:
    addition = addition_results()
    language = language_results()
    verify_saved_summary(language)
    return {"addition": addition, "language": language}


def main() -> None:
    print(json.dumps(calculate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
