#!/usr/bin/env python3
"""Aggregate the topology-interface design and graph-null comparison."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def summarize(rows: list[dict], *, bootstrap_samples=10000, seed=20260923,
              require_complete=True) -> dict:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    factorial = [row for row in rows if row["stage"] == "factorial"]
    null_rows = [row for row in rows if row["stage"] == "null-rank"]
    result = {"completeDesign": False, "factorial": None, "nullRank": None}
    factorial_complete = False
    null_complete = False
    if require_complete and (len(factorial) != 20 or len(null_rows) != 21):
        raise ValueError(
            "final analysis requires all 20 factorial and 21 null-rank evaluations")
    if factorial:
        by_interface = defaultdict(dict)
        for row in factorial:
            if row["topology"] in by_interface[row["interfaceSeed"]]:
                raise ValueError("duplicate factorial cell")
            by_interface[row["interfaceSeed"]][row["topology"]] = row["pairedNormal"]
        interfaces = sorted(by_interface)
        if any("real" not in by_interface[interface] for interface in interfaces):
            raise ValueError("each factorial interface needs a real-topology result")
        rewire_names = sorted(set.intersection(*[
            {name for name in by_interface[interface] if name != "real"}
            for interface in interfaces
        ]))
        if not rewire_names:
            raise ValueError("no complete factorial rewire crossing")
        expected_rewires = [f"rewire-{index:02d}" for index in range(1, 5)]
        expected = {"real", *expected_rewires}
        factorial_complete = (
            len(factorial) == 20
            and len(interfaces) == 4
            and rewire_names == expected_rewires
            and all(set(by_interface[interface]) == expected for interface in interfaces)
        )
        if require_complete and not factorial_complete:
            raise ValueError("factorial analysis does not match the locked 4 by 5 design")
        real_values = np.asarray([by_interface[i]["real"] for i in interfaces])
        rewire_matrix = np.asarray([
            [by_interface[i][name] for name in rewire_names] for i in interfaces])
        differences = real_values - rewire_matrix.mean(1)
        rng = np.random.default_rng(seed)
        bootstrap = []
        for _ in range(bootstrap_samples):
            interface_indices = rng.integers(0, len(interfaces), len(interfaces))
            graph_indices = rng.integers(0, len(rewire_names), len(rewire_names))
            real_sample = real_values[interface_indices]
            null_sample = rewire_matrix[np.ix_(interface_indices, graph_indices)]
            bootstrap.append(float(real_sample.mean() - null_sample.mean()))
        result["factorial"] = {
            "interfaces": len(interfaces),
            "rewireGraphs": len(rewire_names),
            "realMean": float(real_values.mean()),
            "rewiredMean": float(rewire_matrix.mean()),
            "pairedDifference": float(differences.mean()),
            "differenceByInterface": {str(i): float(d) for i, d in zip(interfaces, differences)},
            "twoWayCrossedBootstrap95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "bootstrapSamples": bootstrap_samples,
            "bootstrapMethod": (
                "independent resampling of the interface and rewire-identity axes"
            ),
            "meanByPhysicalGpu": {
                str(gpu): float(np.mean([
                    row["pairedNormal"] for row in factorial
                    if row.get("physicalGpu") == gpu
                ]))
                for gpu in sorted({row.get("physicalGpu") for row in factorial
                                   if row.get("physicalGpu") is not None})
            },
        }
    if null_rows:
        topologies = [row["topology"] for row in null_rows]
        if len(topologies) != len(set(topologies)):
            raise ValueError("duplicate null-rank topology")
        real = [row["pairedNormal"] for row in null_rows if row["topology"] == "real"]
        rewired = [row["pairedNormal"] for row in null_rows if row["topology"] != "real"]
        if len(real) != 1 or not rewired:
            raise ValueError("null-rank analysis needs one real graph and at least one rewire")
        null_complete = set(topologies) == {
            "real", *(f"rewire-{index:02d}" for index in range(1, 21))}
        if require_complete and not null_complete:
            raise ValueError("null-rank analysis does not match the locked 20-graph null")
        exceed = sum(value >= real[0] for value in rewired)
        result["nullRank"] = {
            "real": real[0],
            "rewires": len(rewired),
            "rewiredMean": float(np.mean(rewired)),
            "rewiredRange": [float(np.min(rewired)), float(np.max(rewired))],
            "rewiresAtOrAboveReal": exceed,
            "oneSidedRankP": (1 + exceed) / (1 + len(rewired)),
            "testUnit": "independently rewired graph",
            "realPhysicalGpu": next((row.get("physicalGpu") for row in null_rows
                                     if row["topology"] == "real"), None),
            "rewiredMeanByPhysicalGpu": {
                str(gpu): float(np.mean([
                    row["pairedNormal"] for row in null_rows
                    if row["topology"] != "real" and row.get("physicalGpu") == gpu
                ]))
                for gpu in sorted({row.get("physicalGpu") for row in null_rows
                                   if row["topology"] != "real"
                                   and row.get("physicalGpu") is not None})
            },
        }
    result["completeDesign"] = bool(factorial_complete and null_complete)
    return result
