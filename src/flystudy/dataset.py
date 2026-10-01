from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


OPERATIONS = ("+", "-", "*")
MULTIPLICATION_MIX = {
    "nontrivial": 0.60,
    "times_zero": 0.20,
    "times_one": 0.20,
}


def _number_code(value: int) -> np.ndarray:
    if not 0 <= value <= 100:
        raise ValueError(f"number outside 0..100: {value}")
    hundreds, remainder = divmod(value, 100)
    tens, ones = divmod(remainder, 10)
    code = np.zeros(22, dtype=np.float32)
    code[hundreds] = 1.0
    code[2 + tens] = 1.0
    code[12 + ones] = 1.0
    return code


def encode_expression(a: int, b: int, op: str) -> np.ndarray:
    """Fixed 47-channel code: A digits, B digits, and one operator."""
    if op not in OPERATIONS:
        raise ValueError(f"unsupported operation: {op}")
    operator = np.zeros(3, dtype=np.float32)
    operator[OPERATIONS.index(op)] = 1.0
    return np.concatenate((_number_code(a), _number_code(b), operator))


def multiplication_stratum(row: dict) -> str:
    if row.get("op") != "*":
        raise ValueError("multiplication stratum requested for a non-multiplication row")
    a, b = int(row["a"]), int(row["b"])
    if a == 0 or b == 0:
        return "times_zero"
    if a == 1 or b == 1:
        return "times_one"
    return "nontrivial"


def _allocate(total: int, weights: list[float], rotation: int = 0) -> list[int]:
    raw = np.asarray(weights, dtype=np.float64)
    raw = raw / raw.sum() * total
    result = np.floor(raw).astype(int)
    remainder = total - int(result.sum())
    fractions = raw - result
    order = sorted(
        range(len(weights)),
        key=lambda index: (-fractions[index], (index - rotation) % len(weights)),
    )
    for index in order[:remainder]:
        result[index] += 1
    return result.tolist()


@dataclass
class ArithmeticDataset:
    rows: list[dict]
    _sample_counter: int = field(default=0, init=False, repr=False)
    _multiplication_queue: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.by_operation: dict[str, list[dict]] = {
            op: [row for row in self.rows if row["op"] == op] for op in OPERATIONS
        }
        if any(not pool for pool in self.by_operation.values()):
            raise ValueError("each arithmetic operation needs at least one row")
        self.by_answer: dict[str, dict[int, list[dict]]] = {}
        for op, pool in self.by_operation.items():
            answers: dict[int, list[dict]] = defaultdict(list)
            for row in pool:
                answers[int(row["y"])].append(row)
            self.by_answer[op] = dict(answers)
        multiplication = self.by_operation["*"]
        self.multiplication_pools = {
            key: [row for row in multiplication if multiplication_stratum(row) == key]
            for key in MULTIPLICATION_MIX
        }
        if any(not pool for pool in self.multiplication_pools.values()):
            raise ValueError("all multiplication strata must be present")

    @classmethod
    def from_json(cls, path: str | Path, split: str) -> "ArithmeticDataset":
        normalized = "validation" if split == "val" else split
        payload = json.loads(Path(path).read_text())
        rows = [row for row in payload["examples"] if row["split"] == normalized]
        if not rows:
            raise ValueError(f"split has no rows: {split}")
        return cls(rows)

    @staticmethod
    def _choice(pool: list[dict], count: int, rng: np.random.Generator) -> list[dict]:
        indices = rng.integers(0, len(pool), size=count)
        return [pool[int(index)] for index in indices]

    def _sample_add_or_subtract(
        self, op: str, count: int, rng: np.random.Generator
    ) -> list[dict]:
        uniform_expression, uniform_answer = _allocate(count, [0.5, 0.5], self._sample_counter)
        selected = self._choice(self.by_operation[op], uniform_expression, rng)
        answer_keys = sorted(self.by_answer[op])
        chosen_answers = rng.choice(answer_keys, size=uniform_answer, replace=True)
        for answer in chosen_answers:
            selected.extend(self._choice(self.by_answer[op][int(answer)], 1, rng))
        return selected

    def _sample_multiplication(self, count: int, rng: np.random.Generator) -> list[dict]:
        selected: list[dict] = []
        for _ in range(count):
            if not self._multiplication_queue:
                block = ["nontrivial", "nontrivial", "nontrivial", "times_zero", "times_one"]
                rng.shuffle(block)
                self._multiplication_queue.extend(block)
            stratum = self._multiplication_queue.pop()
            selected.extend(self._choice(self.multiplication_pools[stratum], 1, rng))
        return selected

    def sample_balanced(self, batch_size: int, rng: np.random.Generator) -> list[dict]:
        if batch_size < 3:
            raise ValueError("batch size must be at least three")
        counts = _allocate(batch_size, [1, 1, 1], self._sample_counter)
        batch: list[dict] = []
        for op, count in zip(OPERATIONS, counts, strict=True):
            if op == "*":
                batch.extend(self._sample_multiplication(count, rng))
            else:
                batch.extend(self._sample_add_or_subtract(op, count, rng))
        rng.shuffle(batch)
        self._sample_counter += 1
        return batch

    def operation_epoch_batches(
        self,
        operation: str,
        batch_size: int,
        *,
        epoch: int,
        seed: int,
    ) -> list[list[dict]]:
        """Return one deterministic, without-replacement operation epoch."""
        if operation not in OPERATIONS:
            raise ValueError(f"unsupported operation: {operation}")
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if epoch < 1:
            raise ValueError("epoch numbers start at one")
        pool = self.by_operation[operation]
        generator = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
        order = generator.permutation(len(pool))
        return [
            [pool[int(index)] for index in order[start : start + batch_size]]
            for start in range(0, len(pool), batch_size)
        ]

    @staticmethod
    def arrays(rows: Iterable[dict]) -> tuple[np.ndarray, np.ndarray]:
        rows = list(rows)
        encoded = np.stack([encode_expression(row["a"], row["b"], row["op"]) for row in rows])
        targets = np.asarray([row["y"] for row in rows], dtype=np.int64)
        return encoded, targets
