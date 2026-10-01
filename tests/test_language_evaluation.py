from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flystudy.language_lexicon import Event
from flystudy.language_corpus import make_record, render_record, swap_record
import flystudy.language_evaluation as evaluation
from flystudy.language_metrics import family_subset
from flystudy.language_runtime import tiny_model
from flystudy.language_model import ConnectomeLanguageModel
from flystudy.language_evaluation import (
    CHANNEL_CHANGING_FACTORS,
    FACTORS,
    PAIR_PRESERVING_FACTORS,
    counterfactual_record,
    matched_normal_subset,
    minimal_pair_factors,
    paired_relations_intervened,
    score_orientation_null,
)
from flystudy.language_train import apply_codebook_seed, build, parser


def two_event_reports(count=4096):
    rows = [make_record(i, split="test_composition") for i in range(count)]
    return [row for row in rows if row.mode == "report"]


def test_counterfactual_changes_only_the_requested_factor_and_keeps_prompt():
    row = two_event_reports()[0]
    names = [field.name for field in fields(Event)]
    for factor in FACTORS:
        changed = counterfactual_record(row, factor)
        assert changed.prompt_words == row.prompt_words
        assert changed.answer_words != row.answer_words
        if factor == "order":
            assert changed.scene.events == tuple(reversed(row.scene.events))
            continue
        before = row.scene.events[0]
        after = changed.scene.events[0]
        differences = [name for name in names if getattr(before, name) != getattr(after, name)]
        assert differences == [factor]
        assert changed.scene.events[1:] == row.scene.events[1:]


def test_minimal_pair_and_paired_intervention_evaluators_execute_on_tiny_model():
    torch.set_num_threads(1)
    model = tiny_model()
    rows = family_subset(two_event_reports(), 1, reports_only=True, order_invariant=True)
    factors, samples = minimal_pair_factors(
        model, rows[:4], device="cpu", batch_size=2, encode_steps=8)
    assert set(factors) == set(FACTORS)
    assert all(metric["groups"] == 4 for metric in factors.values())
    assert all(sum(metric["counterfactualPairPartitionCounts"].values()) == 4
               for metric in factors.values())
    assert all(metric["binaryPreferenceChance"] == .5 for metric in factors.values())
    assert len(samples) == len(FACTORS) * 4
    for sample in samples:
        assert sample["counterfactualPairPartition"] in {"train", "validation", "test"}
        if sample["factor"] in PAIR_PRESERVING_FACTORS:
            assert sample["counterfactualPairPartition"] == "test"
        else:
            assert sample["factor"] in CHANNEL_CHANGING_FACTORS
    for intervention in ("normal", "zero", "shuffle"):
        metric, scores = paired_relations_intervened(
            model, rows[:4], device="cpu", batch_size=2,
            encode_steps=8, intervention=intervention)
        assert metric["groups"] == len(scores) == 4
        assert metric["intervention"] == intervention
        assert metric["noSensory"] is False
        assert all(np.asarray(row["scores"]).shape == (2, 2) for row in scores)
        if intervention == "shuffle":
            assert all(row["sensorySourceId"] != row["id"] for row in scores)


def test_zero_sensory_pair_score_has_identical_rows_and_fails_strict_group_rule():
    torch.set_num_threads(1)
    model = tiny_model()
    rows = family_subset(two_event_reports(), 1, reports_only=True, order_invariant=True)[:4]
    metric, scores = paired_relations_intervened(
        model, rows, device="cpu", batch_size=2,
        encode_steps=8, intervention="zero")
    assert metric["groupAccuracy"] == 0
    for row in scores:
        matrix = np.asarray(row["scores"])
        np.testing.assert_allclose(matrix[0], matrix[1], atol=1e-6)


@pytest.mark.parametrize("intervention", ["normal", "zero", "shuffle"])
def test_primary_paired_candidates_use_the_fixed_relation_first_form(monkeypatch, intervention):
    records = two_event_reports()[:2]
    rows = []
    for variant, row in enumerate(records):
        prompt, answer = render_record(row.scene, row.template_family, variant)
        rows.append(replace(row, variant=variant, prompt_words=prompt, answer_words=answer))
    captured = []

    def score(model, conditions, candidates, **kwargs):
        captured.append((conditions, candidates))
        return torch.tensor([4., 1., 0., 3.]).repeat(len(conditions) // 4)

    monkeypatch.setattr(evaluation, "_score_candidates", score)
    metric, outputs = paired_relations_intervened(
        SimpleNamespace(eval=lambda: None), rows, device="cpu", batch_size=2,
        intervention=intervention)
    assert metric["groups"] == len(outputs) == 2
    conditions, candidates = captured[0]
    assert all(row.variant == 2 for row in conditions)
    for index, original in enumerate(rows):
        first = conditions[4 * index]
        prompt, answer = render_record(original.scene, original.template_family, 2)
        assert first.scene == original.scene
        assert first.prompt_words == prompt
        assert first.answer_words == answer
        second = swap_record(first)
        assert candidates[4 * index:4 * index + 4] == [
            first.answer_words, second.answer_words,
            first.answer_words, second.answer_words,
        ]


def test_no_sensory_condition_is_zero_during_normal_evaluation():
    torch.set_num_threads(1)
    model = tiny_model()
    rows = family_subset(two_event_reports(), 1, reports_only=True, order_invariant=True)[:4]
    metric, scores = paired_relations_intervened(
        model, rows, device="cpu", batch_size=2,
        encode_steps=8, intervention="normal", no_sensory=True)
    assert metric["groupAccuracy"] == 0
    assert metric["noSensory"] is True
    for row in scores:
        matrix = np.asarray(row["scores"])
        np.testing.assert_allclose(matrix[0], matrix[1], atol=1e-6)


def test_matched_normal_subset_uses_intervention_group_ids():
    normal = [
        {"id": 1, "correct": True},
        {"id": 2, "correct": False},
        {"id": 3, "correct": True},
    ]
    intervention = [{"id": 3}, {"id": 2}]
    metric, selected = matched_normal_subset(normal, intervention)
    assert [row["id"] for row in selected] == [3, 2]
    assert metric["groups"] == 2
    assert metric["groupAccuracy"] == .5


def test_orientation_null_is_deterministic_and_bounded():
    outputs = [
        {"scores": [[4.0, 1.0], [0.0, 3.0]]},
        {"scores": [[3.0, 2.0], [1.0, 5.0]]},
        {"scores": [[0.0, 2.0], [3.0, 1.0]]},
    ]
    first = score_orientation_null(outputs, permutations=200, seed=17)
    second = score_orientation_null(outputs, permutations=200, seed=17)
    assert first == second
    assert first["permutations"] == 200
    assert first["observedAccuracy"] == 2 / 3
    assert 0 <= first["meanAccuracy"] <= 1
    assert 0 < first["upperTailP"] <= 1
    assert len(first["accuracy95"]) == 2


def test_codebook_seed_can_be_fixed_independently_of_existing_model_seed():
    first = tiny_model(seed=11)
    second = tiny_model(seed=22)
    apply_codebook_seed(first, 777)
    apply_codebook_seed(second, 777)
    torch.testing.assert_close(first.input_codebook, second.input_codebook, atol=0, rtol=0)
    torch.testing.assert_close(first.output_codebook, second.output_codebook, atol=0, rtol=0)
    assert not torch.equal(first.theta, second.theta)


def test_interface_placement_changes_without_changing_codebooks_or_weights():
    nodes = 320
    rows = np.repeat(np.arange(nodes), 3)
    cols = np.asarray([(i + offset) % nodes for i in range(nodes)
                       for offset in (-1, 7, 31)])
    options = dict(
        n_nodes=nodes,
        sensory_groups=[np.array([i]) for i in range(24)],
        seed=11,
        dimension=109,
        token_group=1,
        output_group=1,
    )
    first = ConnectomeLanguageModel(rows, cols, interface_seed=101, **options)
    second = ConnectomeLanguageModel(rows, cols, interface_seed=202, **options)
    apply_codebook_seed(first, 777)
    apply_codebook_seed(second, 777)
    assert not torch.equal(first.token_nodes, second.token_nodes)
    assert not torch.equal(first.output_nodes, second.output_nodes)
    torch.testing.assert_close(first.input_codebook, second.input_codebook, atol=0, rtol=0)
    torch.testing.assert_close(first.output_codebook, second.output_codebook, atol=0, rtol=0)
    torch.testing.assert_close(first.theta, second.theta, atol=0, rtol=0)


def test_same_optimization_seed_gives_topologies_the_same_weight_multiset():
    nodes = 320
    rows = np.repeat(np.arange(nodes), 3)
    first_cols = np.asarray([(i + offset) % nodes for i in range(nodes)
                             for offset in (-1, 7, 31)])
    second_cols = np.asarray([(i + offset) % nodes for i in range(nodes)
                              for offset in (-1, 11, 37)])
    options = dict(
        n_nodes=nodes,
        sensory_groups=[np.array([i]) for i in range(24)],
        seed=11,
        interface_seed=101,
        dimension=109,
        token_group=1,
        output_group=1,
    )
    first = ConnectomeLanguageModel(rows, first_cols, **options)
    second = ConnectomeLanguageModel(rows, second_cols, **options)
    torch.testing.assert_close(first.theta, second.theta, atol=0, rtol=0)
    torch.testing.assert_close(first.token_nodes, second.token_nodes, atol=0, rtol=0)
    torch.testing.assert_close(first.output_nodes, second.output_nodes, atol=0, rtol=0)


def test_trainer_rejects_a_codebook_type_that_would_break_the_lock():
    args = parser().parse_args([
        "--tiny", "--embedding", "random", "--run-dir", "unused-test-run"])
    with pytest.raises(ValueError, match="requires orthogonal codebooks"):
        build(args, "cpu")
