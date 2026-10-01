from __future__ import annotations

from collections import defaultdict

import numpy as np

from flystudy.dataset import multiplication_stratum


def addition_stratum(row: dict) -> str:
    if row.get("op") != "+":
        raise ValueError("addition stratum requested for a non-addition row")
    if int(row["y"]) == 100:
        return "sum_100"
    if int(row["a"]) % 10 + int(row["b"]) % 10 >= 10:
        return "ones_carry"
    return "no_ones_carry"


def classification_metrics(rows: list[dict], predictions: np.ndarray) -> dict:
    predictions = np.asarray(predictions, dtype=np.int64)
    if predictions.shape != (len(rows),):
        raise ValueError("one prediction is required for each row")
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        target = int(row["y"])
        buckets["overall"].append((target, int(prediction)))
        buckets[row["op"]].append((target, int(prediction)))
        if row["op"] == "+":
            buckets[addition_stratum(row)].append((target, int(prediction)))
        if row["op"] == "*":
            buckets[multiplication_stratum(row)].append((target, int(prediction)))
    result = {}
    for name, pairs in buckets.items():
        targets = np.asarray([pair[0] for pair in pairs])
        guesses = np.asarray([pair[1] for pair in pairs])
        result[name] = {
            "count": int(len(pairs)),
            "accuracy": float(np.mean(targets == guesses)),
            "mae": float(np.mean(np.abs(targets - guesses))),
        }
    return result


def constant_answer_baseline(rows: list[dict], *, answer: int | None = None) -> dict:
    if not rows:
        raise ValueError("constant baseline needs at least one row")
    targets = np.asarray([int(row["y"]) for row in rows], dtype=np.int64)
    counts = np.bincount(targets, minlength=101)
    answer = int(counts.argmax()) if answer is None else int(answer)
    if not 0 <= answer <= 100:
        raise ValueError("constant answer must be in 0..100")
    return {
        "answer": answer,
        "count": int(len(rows)),
        "accuracy": float(counts[answer] / len(rows)),
        "mae": float(np.mean(np.abs(targets - answer))),
    }


def sample_mix(rows: list[dict]) -> dict:
    counts = {"+": 0, "-": 0, "*": 0, "nontrivial": 0, "times_zero": 0, "times_one": 0}
    for row in rows:
        counts[row["op"]] += 1
        if row["op"] == "*":
            counts[multiplication_stratum(row)] += 1
    multiplication_total = counts["*"]
    return {
        "counts": counts,
        "multiplicationFractions": {
            name: (counts[name] / multiplication_total if multiplication_total else 0.0)
            for name in ("nontrivial", "times_zero", "times_one")
        },
    }
