"""Held-out grounded-relation evaluation and sensory interventions.

The functions in this module never update model parameters. They score fixed
checkpoints on held-out composition examples with explicit counterfactual and
sensory interventions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import time

import numpy as np
import torch
from torch.nn import functional as F

from flystudy.language_lexicon import (
    DURATIONS,
    KINDS,
    LOCATIONS,
    PAD,
    SPEEDS,
    Scene,
)
from flystudy.language_corpus import (
    REPORT_FAMILIES,
    pair_partition,
    render_record,
    swap_record,
)
from flystudy.language_metrics import group_correct
from flystudy.language_runtime import make_batch


FACTORS = ("location", "kind", "level", "duration", "speed", "order")
PAIR_PRESERVING_FACTORS = ("level", "duration", "speed", "order")
CHANNEL_CHANGING_FACTORS = ("location", "kind")
FACTOR_LABELS = {
    "location": "location",
    "kind": "stimulus kind",
    "level": "strength",
    "duration": "duration",
    "speed": "speed",
    "order": "event order",
}


def counterfactual_record(row, factor: str):
    """Change one latent factor while preserving the report question."""
    if factor not in FACTORS:
        raise ValueError(f"unknown factor: {factor}")
    if row.template_family not in REPORT_FAMILIES or len(row.scene.events) != 2:
        raise ValueError("factor counterfactuals require a two-event report")
    if factor == "order":
        changed = swap_record(row)
    else:
        first, second = row.scene.events
        if factor == "location":
            value = (first.location + 1) % len(LOCATIONS)
        elif factor == "kind":
            value = (first.kind + 1) % len(KINDS)
        elif factor == "level":
            value = 1 - first.level
        elif factor == "duration":
            value = (first.duration + 1) % len(DURATIONS)
        else:
            value = (first.speed + 1) % len(SPEEDS)
        event = replace(first, **{factor: value})
        scene = Scene((event, second))
        prompt, answer = render_record(scene, row.template_family, row.variant)
        changed = replace(row, scene=scene, prompt_words=prompt, answer_words=answer)
    if changed.prompt_words != row.prompt_words:
        raise ValueError("counterfactual changed the fixed report question")
    if changed.answer_words == row.answer_words:
        raise ValueError("counterfactual did not change the reference answer")
    return changed


@torch.no_grad()
def _score_candidates(model, conditions, candidates, *, device, encode_steps,
                      sensory_sources=None, zero_sensory=False,
                      normalize_by_length=False):
    if len(conditions) != len(candidates):
        raise ValueError("conditions and candidates must have equal length")
    batch = make_batch(
        conditions,
        device=device,
        encode_steps=encode_steps,
        no_sensory=zero_sensory,
        answer_override=candidates,
    )
    if sensory_sources is not None:
        if zero_sensory or len(sensory_sources) != len(conditions):
            raise ValueError("invalid sensory override")
        sensory = make_batch(
            sensory_sources,
            device=device,
            encode_steps=encode_steps,
        )
        if sensory.amplitudes.shape != batch.amplitudes.shape:
            raise ValueError("sensory intervention changed the event-time shape")
        batch.amplitudes = sensory.amplitudes
    logits = model(*batch.inputs())
    losses = F.cross_entropy(
        logits.flatten(0, 1),
        batch.targets.flatten(),
        ignore_index=PAD,
        reduction="none",
    ).reshape_as(batch.targets)
    scores = -losses.sum(1)
    if normalize_by_length:
        lengths = (batch.targets != PAD).sum(1).clamp_min(1)
        scores = scores / lengths
    return scores.cpu().numpy()


@torch.no_grad()
def minimal_pair_factors(model, rows, *, device, batch_size=12, encode_steps=8,
                         no_sensory=False, deadline=0):
    """Score original and one-factor counterfactual reports in both scenes."""
    rows = list(rows)
    if not rows:
        raise ValueError("minimal-pair evaluation needs examples")
    model.eval()
    outputs = []
    counts = {factor: {"original": 0, "counterfactual": 0,
                       "reversal": 0, "strict": 0, "groups": 0,
                       "margins": [], "counterfactualPartitions": Counter()}
              for factor in FACTORS}
    for factor in FACTORS:
        for start in range(0, len(rows), batch_size):
            if deadline and time.time() >= deadline:
                raise TimeoutError("minimal-pair evaluation deadline reached")
            block = rows[start:start + batch_size]
            conditions = []
            candidates = []
            pairs = []
            for row in block:
                changed = counterfactual_record(row, factor)
                conditions.extend((row, row, changed, changed))
                candidates.extend((row.answer_words, changed.answer_words,
                                   row.answer_words, changed.answer_words))
                pairs.append((row, changed))
            scores = _score_candidates(
                model, conditions, candidates, device=device,
                encode_steps=encode_steps,
                zero_sensory=no_sensory,
                normalize_by_length=True,
            ).reshape(-1, 2, 2)
            for (row, changed), matrix in zip(pairs, scores, strict=True):
                original = bool(matrix[0, 0] > matrix[0, 1])
                counterfactual = bool(matrix[1, 1] > matrix[1, 0])
                reversal = original and counterfactual
                strict = group_correct(matrix)
                margin = float(.5 * ((matrix[0, 0] - matrix[0, 1]) +
                                     (matrix[1, 1] - matrix[1, 0])))
                bucket = counts[factor]
                bucket["original"] += int(original)
                bucket["counterfactual"] += int(counterfactual)
                bucket["reversal"] += int(reversal)
                bucket["strict"] += int(strict)
                bucket["groups"] += 1
                bucket["margins"].append(margin)
                changed_partition = pair_partition(changed.pair_families[0])
                bucket["counterfactualPartitions"][changed_partition] += 1
                outputs.append({
                    "id": row.record_id,
                    "family": list(row.pair_families[0]),
                    "factor": factor,
                    "factorLabel": FACTOR_LABELS[factor],
                    "expected": list(row.answer_words),
                    "counterfactual": list(changed.answer_words),
                    "counterfactualPairPartition": changed_partition,
                    "scores": matrix.tolist(),
                    "originalCorrect": original,
                    "counterfactualCorrect": counterfactual,
                    "preferenceReversal": reversal,
                    "strictGroupCorrect": strict,
                    "meanCorrectMargin": margin,
                })
    metrics = {}
    for factor, bucket in counts.items():
        n = bucket["groups"]
        metrics[factor] = {
            "label": FACTOR_LABELS[factor],
            "groups": n,
            "originalPreferenceAccuracy": bucket["original"] / n,
            "counterfactualPreferenceAccuracy": bucket["counterfactual"] / n,
            "preferenceReversalAccuracy": bucket["reversal"] / n,
            "strictGroupAccuracy": bucket["strict"] / n,
            "meanCorrectMargin": float(np.mean(bucket["margins"])),
            "counterfactualPairPartitionCounts": dict(
                sorted(bucket["counterfactualPartitions"].items())),
            "binaryPreferenceChance": .5,
            "strictGroupIndependentScoreChance": 1 / 6,
            "scope": (
                "causal one-factor control; the counterfactual pair partition is "
                "reported and is not assumed to remain held out"
            ),
            "candidateScore": "mean teacher-forced log probability per target token",
        }
    return metrics, outputs


@torch.no_grad()
def paired_relations_intervened(model, rows, *, device, batch_size=12,
                                encode_steps=8, intervention="normal",
                                no_sensory=False, deadline=0):
    """Run the strict four-score relation test under sensory interventions."""
    if intervention not in ("normal", "zero", "shuffle"):
        raise ValueError("paired intervention must be normal, zero or shuffle")
    rows = list(rows)
    if not rows:
        raise ValueError("paired evaluation needs examples")
    if intervention == "shuffle" and len(rows) < 2:
        raise ValueError("sensory shuffle needs at least two groups")
    canonical = []
    for row in rows:
        if row.template_family not in REPORT_FAMILIES or len(row.scene.events) != 2:
            raise ValueError("paired relation evaluation requires two-event reports")
        prompt, answer = render_record(row.scene, row.template_family, 2)
        first = replace(row, variant=2, prompt_words=prompt, answer_words=answer)
        second = swap_record(first)
        if first.prompt_words != second.prompt_words or len(first.answer_words) != len(second.answer_words):
            raise ValueError("paired questions and candidate lengths must match")
        canonical.append((first, second))
    model.eval()
    outputs = []
    for start in range(0, len(canonical), batch_size):
        if deadline and time.time() >= deadline:
            raise TimeoutError("paired intervention deadline reached")
        stop = min(start + batch_size, len(canonical))
        conditions = []
        candidates = []
        sensory_sources = []
        sensory_source_ids = []
        for index in range(start, stop):
            first, second = canonical[index]
            conditions.extend((first, first, second, second))
            candidates.extend((first.answer_words, second.answer_words,
                               first.answer_words, second.answer_words))
            if no_sensory:
                sensory_source_ids.append(None)
            elif intervention == "shuffle":
                other_first, other_second = canonical[(index + 1) % len(canonical)]
                sensory_sources.extend((other_first, other_first, other_second, other_second))
                sensory_source_ids.append(other_first.record_id)
            elif intervention == "zero":
                sensory_source_ids.append(None)
            else:
                sensory_source_ids.append(first.record_id)
        scores = _score_candidates(
            model,
            conditions,
            candidates,
            device=device,
            encode_steps=encode_steps,
            sensory_sources=(
                sensory_sources
                if intervention == "shuffle" and not no_sensory else None
            ),
            zero_sensory=no_sensory or intervention == "zero",
        ).reshape(-1, 2, 2)
        for (first, _), matrix, sensory_source_id in zip(
                canonical[start:stop], scores, sensory_source_ids, strict=True):
            outputs.append({
                "id": first.record_id,
                "sensorySourceId": sensory_source_id,
                "family": list(first.pair_families[0]),
                "scores": matrix.tolist(),
                "correct": group_correct(matrix),
            })
    if no_sensory:
        interpretation = (
            "no-sensory baseline; the sensory tensor is zero during evaluation"
        )
    else:
        interpretation = {
            "normal": "primary held-out paired relation endpoint",
            "zero": (
                "structural negative control; identical paired sensory inputs must "
                "fail the strict four-comparison rule"
            ),
            "shuffle": "sensory-dependence control with a recorded mismatched source",
        }[intervention]
    return {
        "groups": len(outputs),
        "groupAccuracy": sum(row["correct"] for row in outputs) / len(outputs),
        "tiePolicy": "incorrect",
        "comparisonsPerGroup": 4,
        "intervention": intervention,
        "noSensory": bool(no_sensory),
        "interpretation": interpretation,
    }, outputs


def matched_normal_subset(normal_outputs, intervention_outputs):
    """Select normal-condition groups matching an intervention evaluation.

    Intervention runs can deliberately use fewer groups than the primary
    endpoint. This helper makes the before/after comparison use identical group
    IDs without repeating model inference.
    """
    normal_by_id = {row["id"]: row for row in normal_outputs}
    if len(normal_by_id) != len(normal_outputs):
        raise ValueError("normal outputs contain duplicate group IDs")
    ids = [row["id"] for row in intervention_outputs]
    if not ids:
        raise ValueError("intervention outputs must be nonempty")
    if len(set(ids)) != len(ids):
        raise ValueError("intervention outputs contain duplicate group IDs")
    missing = sorted(set(ids) - set(normal_by_id))
    if missing:
        raise ValueError(f"intervention groups missing from normal outputs: {missing[:3]}")
    selected = [normal_by_id[value] for value in ids]
    return {
        "groups": len(selected),
        "groupAccuracy": sum(row["correct"] for row in selected) / len(selected),
        "tiePolicy": "incorrect",
        "comparisonsPerGroup": 4,
        "intervention": "normal",
        "matchedTo": "pairedShuffle",
        "interpretation": (
            "normal-sensory score on exactly the groups used by the shuffle control"
        ),
    }, selected


def score_orientation_null(outputs, *, permutations=2000, seed=20260923):
    """Permute the four score positions within groups as a score-scale null.

    This is a calibration diagnostic. The independently rewired graph ensemble
    remains the inferential null for topology claims.
    """
    if permutations < 1 or not outputs:
        raise ValueError("positive permutations and nonempty outputs required")
    matrices = [np.asarray(row["scores"], dtype=np.float64) for row in outputs]
    if any(matrix.shape != (2, 2) or not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("each output needs four finite scores")
    rng = np.random.default_rng(seed)
    accuracies = []
    for _ in range(permutations):
        correct = 0
        for matrix in matrices:
            permuted = rng.permutation(matrix.reshape(-1)).reshape(2, 2)
            correct += int(group_correct(permuted))
        accuracies.append(correct / len(matrices))
    observed = sum(group_correct(matrix) for matrix in matrices) / len(matrices)
    return {
        "method": "within-group four-score position permutation",
        "scope": "calibration only; topology inference uses independent graph nulls",
        "groups": len(matrices),
        "permutations": permutations,
        "seed": seed,
        "observedAccuracy": observed,
        "meanAccuracy": float(np.mean(accuracies)),
        "accuracy95": np.quantile(accuracies, [.025, .975]).tolist(),
        "upperTailP": ((1 + sum(value >= observed for value in accuracies)) /
                       (1 + permutations)),
    }
