import json
from collections import Counter
from pathlib import Path

import numpy as np

from flystudy.dataset import ArithmeticDataset, encode_expression, multiplication_stratum


DATA = Path(__file__).parents[1] / "data" / "bounded-arithmetic-split-v1.json"


def test_pinned_dataset_has_valid_arithmetic_and_no_family_leakage():
    payload = json.loads(DATA.read_text())
    rows = payload["examples"]
    assert len(rows) == 10_985
    assert Counter(row["op"] for row in rows) == {"+": 5_151, "-": 5_151, "*": 683}
    expected = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
    assert all(row["y"] == expected[row["op"]](row["a"], row["b"]) for row in rows)
    family_splits = {}
    for row in rows:
        family_splits.setdefault(row["family"], set()).add(row["split"])
    assert len(family_splits) == 2_948
    assert all(len(splits) == 1 for splits in family_splits.values())


def test_decimal_place_encoding_is_fixed_and_order_sensitive():
    encoded = encode_expression(100, 7, "-")
    assert encoded.shape == (47,)
    assert encoded.dtype == np.float32
    assert encoded.sum() == 7
    assert encoded[1] == 1  # A hundreds digit 1
    assert encoded[2] == 1  # A tens digit 0
    assert encoded[12] == 1  # A ones digit 0
    assert encoded[22] == 1  # B hundreds digit 0
    assert encoded[24] == 1  # B tens digit 0
    assert encoded[41] == 1  # B ones digit 7
    assert encoded[45] == 1  # subtraction
    assert not np.array_equal(encoded, encode_expression(7, 100, "-"))


def test_multiplication_strata_are_mutually_exclusive():
    assert multiplication_stratum({"a": 0, "b": 9, "op": "*"}) == "times_zero"
    assert multiplication_stratum({"a": 1, "b": 9, "op": "*"}) == "times_one"
    assert multiplication_stratum({"a": 8, "b": 9, "op": "*"}) == "nontrivial"


def test_training_sampler_does_not_concentrate_on_zero_or_one():
    dataset = ArithmeticDataset.from_json(DATA, split="train")
    rng = np.random.default_rng(20260912)
    observed = Counter()
    operation_counts = Counter()
    for _ in range(60):
        batch = dataset.sample_balanced(128, rng=rng)
        operation_counts.update(row["op"] for row in batch)
        observed.update(multiplication_stratum(row) for row in batch if row["op"] == "*")
    assert max(operation_counts.values()) - min(operation_counts.values()) <= 1
    multiplication_total = sum(observed.values())
    assert observed["nontrivial"] / multiplication_total >= 0.58
    assert observed["times_zero"] / multiplication_total <= 0.22
    assert observed["times_one"] / multiplication_total <= 0.22


def test_multiplication_mix_survives_microbatch_accumulation():
    dataset = ArithmeticDataset.from_json(DATA, split="train")
    rng = np.random.default_rng(20260912)
    rows = []
    for _ in range(32):
        rows.extend(dataset.sample_balanced(4, rng=rng))
    observed = Counter(multiplication_stratum(row) for row in rows if row["op"] == "*")
    assert observed == {"nontrivial": 26, "times_zero": 8, "times_one": 8}


def test_addition_epoch_is_complete_without_replacement_and_deterministic():
    dataset = ArithmeticDataset.from_json(DATA, split="train")
    first = dataset.operation_epoch_batches("+", 128, epoch=1, seed=20260912)
    repeated = dataset.operation_epoch_batches("+", 128, epoch=1, seed=20260912)
    second = dataset.operation_epoch_batches("+", 128, epoch=2, seed=20260912)
    flattened = [row for batch in first for row in batch]
    assert len(first) == 28
    assert all(len(batch) == 128 for batch in first)
    assert len(flattened) == 3_584
    assert all(row["op"] == "+" for row in flattened)
    assert len({(row["a"], row["b"], row["family"]) for row in flattened}) == 3_584
    assert first == repeated
    assert first != second
