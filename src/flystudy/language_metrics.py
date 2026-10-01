"""Post-generation controlled-English parsing and paired relation evaluation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import re
import time

import numpy as np
import torch
from torch.nn import functional as F

from flystudy.language_lexicon import EOS, PAD, VOCAB, LOCATIONS, KINDS, LEVELS, DURATIONS, SPEEDS
from flystudy.language_corpus import REPORT_FAMILIES, COMPOSITION_QA_FAMILIES, render_record, swap_record
from flystudy.language_runtime import make_batch


def noun(prefix, definite=False):
    article = "the" if definite else "(?:a )?"
    gap = " " if definite else ""
    return (rf"{article}{gap}(?P<{prefix}level>weak|strong) (?P<{prefix}kind>touch|taste|movement|pressure) "
            rf"on (?:the|my|its) (?P<{prefix}side>left|right) (?P<{prefix}part>front|middle|hind) leg")


ACTOR = r"(?:i felt|the fly detected)"


def timing(prefix, subject):
    return (rf"{subject} started (?P<{prefix}speed>slowly|quickly) and stopped after a "
            rf"(?P<{prefix}duration>short|long) time \.")


REPORT_PATTERNS = (
    ("single", re.compile(rf"{ACTOR} {noun('a')} \. {timing('a', 'the (?:sensation|signal)')}")),
    ("ordered", re.compile(rf"first , {ACTOR} {noun('a')} \. {timing('a', 'the (?:sensation|signal)')} "
                           rf"then , {ACTOR} {noun('b')} \. {timing('b', 'the (?:sensation|signal)')}")),
    ("ordered", re.compile(rf"{ACTOR} {noun('a')} first \. {timing('a', 'it')} "
                           rf"after that , {ACTOR} {noun('b')} \. {timing('b', 'it')}")),
    ("relation", re.compile(rf"{noun('a', True)} happened (?P<relation>before|after) {noun('b', True)} \. "
                            rf"{timing('first', 'the first signal')} {timing('last', 'the last signal')}")),
    ("preposed", re.compile(rf"before {noun('repeat', True)} happened , {ACTOR} {noun('a')} \. "
                            rf"{timing('a', 'the first sensation')} then , {ACTOR} {noun('b')} \. "
                            rf"{timing('b', 'the last sensation')}")),
)
RELATION = re.compile(rf"{noun('a', True)} happened (?P<relation>before|after) {noun('b', True)} \.")


def identity(values, prefix):
    return (LOCATIONS.index((values[prefix + "side"], values[prefix + "part"])),
            KINDS.index(values[prefix + "kind"]), LEVELS.index(values[prefix + "level"]))


def parse_report(words):
    """Parse only the preregistered report grammar, without seeing the stimulus.

    Unknown English forms receive no semantic credit and are retained for review.
    No parser or target is used during generation.
    """
    text = " ".join(words)
    for kind, pattern in REPORT_PATTERNS:
        match = pattern.fullmatch(text)
        if match is None:
            continue
        values = match.groupdict()
        if kind == "preposed" and identity(values, "repeat") != identity(values, "b"):
            return None
        names = ["a"] if kind == "single" else ["a", "b"]
        times = names
        if kind == "relation":
            if values["relation"] == "after":
                names = list(reversed(names))
            times = ["first", "last"]
        return tuple(identity(values, name) + (DURATIONS.index(values[t + "duration"]),
                                              SPEEDS.index(values[t + "speed"]))
                     for name, t in zip(names, times, strict=True))
    return None


def parse_relation(words):
    match = RELATION.fullmatch(" ".join(words))
    if match is None:
        return None
    values = match.groupdict()
    nodes = (identity(values, "a"), identity(values, "b"))
    return nodes if values["relation"] == "before" else nodes[::-1]


def semantic_correct(row, words):
    if row.template_family in (*REPORT_FAMILIES, "qa.feel", "qa.detect"):
        expected = tuple((e.location, e.kind, e.level, e.duration, e.speed) for e in row.scene.events)
        return parse_report(words) == expected
    if row.template_family in ("qa.first", "qa.after"):
        return parse_relation(words) == tuple((e.location, e.kind, e.level) for e in row.scene.events)
    return tuple(words) == row.answer_words


def group_correct(scores):
    s = np.asarray(scores)
    if s.shape != (2, 2) or not np.isfinite(s).all():
        raise ValueError("expected four finite scores")
    # All four comparisons must be strict. Ties are errors.
    return bool(s[0, 0] > s[0, 1] and s[1, 1] > s[1, 0] and
                s[0, 0] > s[1, 0] and s[1, 1] > s[0, 1])


def scene_key(row, *, order_invariant=False):
    values = tuple((e.location, e.kind, e.level, e.duration, e.speed) for e in row.scene.events)
    return tuple(sorted(values)) if order_invariant else values


def familiar_subset(rows, count):
    selected, seen = [], set()
    for row in rows:
        key = scene_key(row)
        if row.mode != "report" or key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError("not enough distinct familiar report scenes")


def family_subset(rows, per_family, *, reports_only=False, order_invariant=False):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row.mode == "text" or len(row.scene.events) != 2:
            continue
        if reports_only and row.template_family not in REPORT_FAMILIES:
            continue
        key = scene_key(row, order_invariant=order_invariant)
        if key in seen:
            continue
        seen.add(key)
        groups[row.pair_families[0]].append(row)
    if len(groups) != 24 or any(len(group) < per_family for group in groups.values()):
        raise ValueError("composition subset does not cover 24 families at requested count")
    return [row for family in sorted(groups) for row in groups[family][:per_family]]


@torch.no_grad()
def evaluate(model, rows, *, device, batch_size=24, encode_steps=8, max_tokens=128,
             intervention="normal", no_sensory=False, deadline=0):
    if intervention not in ("normal", "zero", "shuffle", "swap", "flip_strength"):
        raise ValueError("unknown intervention")
    rows = [r for r in rows if r.mode != "text"]
    if not rows:
        raise ValueError("generation evaluation needs grounded examples")
    originals = rows
    if intervention == "swap":
        rows = [swap_record(row) for row in rows]
    if intervention == "flip_strength":
        changed = []
        for row in rows:
            if row.template_family not in REPORT_FAMILIES:
                raise ValueError("strength intervention requires a report")
            first, *others = row.scene.events
            scene = replace(row.scene, events=(replace(first, level=1-first.level), *others))
            prompt, answer = render_record(scene, row.template_family, row.variant)
            if prompt != row.prompt_words:
                raise ValueError("counterfactual changed the question")
            changed.append(replace(row, scene=scene, answer_words=answer))
        rows = changed
    model.eval()
    samples = []
    groups = defaultdict(lambda: [0, 0])
    templates = defaultdict(lambda: [0, 0])
    total_ce = total_tokens = token_correct = 0
    for start in range(0, len(rows), batch_size):
        if deadline and time.time() >= deadline:
            raise TimeoutError("generation evaluation deadline reached")
        block = rows[start:start + batch_size]
        batch = make_batch(block, device=device, encode_steps=encode_steps,
                           no_sensory=no_sensory or intervention == "zero")
        if intervention == "shuffle" and not no_sensory:
            shuffled = [originals[(start + i + 1) % len(rows)] for i in range(len(block))]
            # Equal-count examples avoid changing the sensory delay with the intervention.
            if any(len(a.scene.events) != len(b.scene.events) for a, b in zip(block, shuffled)):
                raise ValueError("shuffle requires equal event counts")
            batch.amplitudes = make_batch(shuffled, device=device, encode_steps=encode_steps).amplitudes
        logits = model(*batch.inputs())
        total_ce += float(F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(),
                                         ignore_index=PAD, reduction="sum"))
        valid = batch.targets != PAD
        total_tokens += int(valid.sum())
        token_correct += int(((logits.argmax(-1) == batch.targets) & valid).sum())
        del logits
        generated = model.generate(batch.amplitudes, batch.prefixes, batch.lengths,
                                   batch.grounded, max_tokens=max_tokens).cpu().tolist()
        for row, tokens in zip(block, generated, strict=True):
            ended = EOS in tokens
            tokens = tokens[:tokens.index(EOS)] if ended else tokens
            words = tuple(VOCAB[t] for t in tokens)
            correct = ended and semantic_correct(row, words)
            literal = ended and words == row.answer_words
            key = ":".join(map(str, row.pair_families[0])) if len(row.scene.events) == 2 else "single"
            groups[key][0] += int(correct)
            groups[key][1] += 1
            templates[row.template_family][0] += int(correct)
            templates[row.template_family][1] += 1
            samples.append({"id": row.record_id, "family": key, "template": row.template_family,
                            "prompt": row.prompt, "expected": list(row.answer_words),
                            "generated": list(words), "semanticCorrect": bool(correct),
                            "literalExact": bool(literal), "eos": ended,
                            "reportParsed": parse_report(words) is not None,
                            "earlyEOSByReferenceLength": ended and len(words) < len(row.answer_words)})
    family_rates = [a / b for a, b in groups.values()]
    ci = np.quantile(np.random.default_rng(20260921).choice(family_rates,
                     (2000, len(family_rates)), replace=True).mean(1), [.025, .975]).tolist()
    n = len(samples)
    return {"count": n, "semanticExact": sum(s["semanticCorrect"] for s in samples) / n,
            "literalExact": sum(s["literalExact"] for s in samples) / n,
            "teacherForcedTokenAccuracy": token_correct / total_tokens,
            "crossEntropy": total_ce / total_tokens,
            "eosRate": sum(s["eos"] for s in samples) / n,
            "meanGeneratedTokens": sum(len(s["generated"]) for s in samples) / n,
            "familyMacroSemantic": float(np.mean(family_rates)), "familyBootstrap95": ci,
            "intervalScope": "descriptive family resampling, not seed uncertainty",
            "byFamily": {key: {"correct": a, "count": b} for key, (a, b) in groups.items()},
            "byTemplate": {key: {"correct": a, "count": b} for key, (a, b) in templates.items()},
            "intervention": intervention}, samples


@torch.no_grad()
def paired_relations(model, rows, *, device, batch_size=12, encode_steps=8, no_sensory=False, deadline=0):
    model.eval()
    outputs = []
    for start in range(0, len(rows), batch_size):
        if deadline and time.time() >= deadline:
            raise TimeoutError("paired evaluation deadline reached")
        block = []
        candidates = []
        ids = []
        for row in rows[start:start + batch_size]:
            if row.template_family not in REPORT_FAMILIES or len(row.scene.events) != 2:
                raise ValueError("paired relation evaluation requires two-event reports")
            prompt, answer = render_record(row.scene, row.template_family, 2)
            a = replace(row, variant=2, prompt_words=prompt, answer_words=answer)
            b = swap_record(a)
            if a.prompt_words != b.prompt_words or len(a.answer_words) != len(b.answer_words):
                raise ValueError("paired questions and candidate lengths must match")
            block.extend((a, a, b, b))
            candidates.extend((a.answer_words, b.answer_words, a.answer_words, b.answer_words))
            ids.append((row.record_id, row.pair_families[0]))
        batch = make_batch(block, device=device, encode_steps=encode_steps,
                           no_sensory=no_sensory, answer_override=candidates)
        logits = model(*batch.inputs())
        scores = -F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(),
                                  ignore_index=PAD, reduction="none").reshape_as(batch.targets).sum(1)
        scores = scores.cpu().numpy().reshape(-1, 2, 2)
        for (record_id, family), matrix in zip(ids, scores, strict=True):
            outputs.append({"id": record_id, "family": list(family), "scores": matrix.tolist(),
                            "correct": group_correct(matrix)})
    if not outputs:
        raise ValueError("empty paired evaluation")
    return {"groups": len(outputs), "groupAccuracy": sum(x["correct"] for x in outputs) / len(outputs),
            "tiePolicy": "incorrect", "comparisonsPerGroup": 4}, outputs


@torch.no_grad()
def text_lm_score(model, rows, *, device, batch_size=24, encode_steps=8, deadline=0):
    rows = list({r.prompt_words + r.answer_words: r for r in rows if r.mode == "text"}.values())
    if not rows:
        raise ValueError("no text documents")
    model.eval()
    ce = count = correct = 0
    for start in range(0, len(rows), batch_size):
        if deadline and time.time() >= deadline:
            raise TimeoutError("text evaluation deadline reached")
        batch = make_batch(rows[start:start+batch_size], device=device, encode_steps=encode_steps)
        logits = model(*batch.inputs())
        ce += float(F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=PAD, reduction="sum"))
        valid = batch.targets != PAD
        count += int(valid.sum())
        correct += int(((logits.argmax(-1) == batch.targets) & valid).sum())
    return {"documents": len(rows), "tokens": count, "crossEntropy": ce/count,
            "perplexity": float(np.exp(min(ce/count, 50))), "teacherForcedTokenAccuracy": correct/count,
            "scope": "whole-document causal prediction on reserved familiar-family text"}
