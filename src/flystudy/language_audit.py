"""Independent audits of materialized records and graph controls."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from flystudy.language_lexicon import LEXICON, PUNCTUATION
from flystudy.language_anatomy import anatomical_sensory_groups
from flystudy.language_corpus import pair_partition, scene_reserved, prompt_reserved
from flystudy.language_metrics import semantic_correct
from flystudy.language_runtime import from_payload, supervised_words, file_hash


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).digest()


def audit_files(directory: Path) -> dict:
    names = {"train": "train.jsonl.gz", "validationId": "validation-id.jsonl.gz",
             "validationComposition": "validation-composition.jsonl.gz", "testComposition": "test-composition.jsonl.gz"}
    profiles = {}
    scene_sets, input_sets, text_sets = {}, {}, {}
    allowed = set(LEXICON) | set(PUNCTUATION)
    for name, filename in names.items():
        scenes, inputs, prompts = set(), set(), set()
        words, targets, labels, events = Counter(), Counter(), defaultdict(Counter), Counter()
        text_answers = {}
        duplicates, n = 0, 0
        with gzip.open(directory / filename, "rt") as stream:
            for line in stream:
                raw = json.loads(line)
                row = from_payload(raw)
                n += 1
                if set(row.prompt_words + row.answer_words) - allowed:
                    raise ValueError("unknown vocabulary")
                partition = "validation" if name == "validationComposition" else "test" if name == "testComposition" else "train"
                if any(pair_partition(pair) != partition for pair in row.pair_families):
                    raise ValueError("semantic family leak")
                if name in ("train", "validationId"):
                    if scene_reserved(row.scene) != (name == "validationId"):
                        raise ValueError("scene reservation violated")
                    if row.mode == "text" and prompt_reserved(row.prompt_words) != (name == "validationId"):
                        raise ValueError("text prompt reservation violated")
                scene = sorted((e.location, e.kind, e.level, e.duration, e.speed) for e in row.scene.events)
                scenes.add(digest(scene))
                key = digest([raw["scene"] if row.mode != "text" else [], raw["promptTokens"], raw["answerTokens"]])
                duplicates += key in inputs
                inputs.add(key)
                words.update(supervised_words(row))
                targets.update(row.answer_words)
                events[len(row.scene.events)] += 1
                if row.mode == "text":
                    prompts.add(digest(row.prompt_words))
                    previous = text_answers.setdefault(row.prompt_words, row.answer_words)
                    if previous != row.answer_words:
                        raise ValueError("conflicting text continuation")
                elif not semantic_correct(row, row.answer_words):
                    raise ValueError("reference answer fails semantic evaluation")
                if row.template_family in ("qa.count", "qa.before", "qa.same_strength", "qa.same_side", "qa.same_leg", "qa.same_kind"):
                    labels[row.template_family][row.answer_words[0]] += 1
        scene_sets[name], input_sets[name], text_sets[name] = scenes, inputs, prompts
        profiles[name] = {"records": n, "uniqueConditionalRecords": len(inputs), "duplicateExposures": duplicates,
                          "lossBearingLexicalWords": len(set(words) & set(LEXICON)),
                          "answerLexicalWords": len(set(targets) & set(LEXICON)),
                          "sceneEventCounts": dict(events), "qaAnswerLabels": {k:dict(v) for k,v in labels.items()}}
    if profiles["train"]["lossBearingLexicalWords"] != 100:
        raise ValueError("not all 100 words receive supervision")
    overlaps = {name: {"conditionalRecordOverlap": len(input_sets["train"] & input_sets[name]),
                       "sceneOverlap": len(scene_sets["train"] & scene_sets[name]),
                       "textPromptOverlap": len(text_sets["train"] & text_sets[name])}
                for name in names if name != "train"}
    if any(value for group in overlaps.values() for value in group.values()):
        raise ValueError(f"cross-split overlap: {overlaps}")
    return {"revision": 2, "files": profiles, "trainEvaluationOverlaps": overlaps,
            "referenceSemantics": "all grounded reference answers pass independent evaluator",
            "splitScope": "scene attributes and text prompts held out within familiar channel families"}


def audit_graphs(real: Path, rewired: list[Path]) -> dict:
    def load(path):
        manifest = json.loads((path / "manifest.json").read_text())
        if file_hash(path / "graph.npz") != manifest["graph_sha256"]:
            raise ValueError("graph hash mismatch")
        with np.load(path / "graph.npz") as graph:
            arrays = {key: graph[key] for key in ("rows", "cols", "body_ids")}
        rows, cols = arrays["rows"], arrays["cols"]
        if len(rows) != manifest["edges"] or len(arrays["body_ids"]) != manifest["neurons"]:
            raise ValueError("graph manifest shape mismatch")
        if rows.min() < 0 or cols.min() < 0 or max(rows.max(), cols.max()) >= manifest["neurons"]:
            raise ValueError("edge endpoint outside graph")
        if np.any(rows[1:] < rows[:-1]):
            raise ValueError("rows must be in CSR order")
        return manifest, arrays
    rm, ra = load(real)
    n = rm["neurons"]
    oldkeys = ra["rows"].astype(np.int64) * n + ra["cols"]
    result = []
    seen = set()
    seeds = set()
    for directory in rewired:
        m, a = load(directory)
        if m.get("control") != "directed-degree-preserving-rewire" or m.get("sourceGraphSha256") != rm["graph_sha256"]:
            raise ValueError("invalid rewiring provenance")
        if m["graph_sha256"] in seen:
            raise ValueError("two independent rewiring artifacts are required")
        seen.add(m["graph_sha256"])
        rewiring = m.get("rewiring", {})
        if rewiring.get("acceptedSwaps", 0) < 5 * rm["edges"]:
            raise ValueError("rewiring has fewer than the preregistered 5E accepted swaps")
        if rewiring.get("seed") is None or rewiring["seed"] in seeds:
            raise ValueError("rewiring seeds must be recorded and independent")
        seeds.add(rewiring["seed"])
        if not np.array_equal(a["body_ids"], ra["body_ids"]):
            raise ValueError("neuron indexing differs")
        for key in ("rows", "cols"):
            if not np.array_equal(np.bincount(a[key], minlength=n), np.bincount(ra[key], minlength=n)):
                raise ValueError("directed degrees differ")
        keys = a["rows"].astype(np.int64) * n + a["cols"]
        if len(np.unique(keys)) != len(keys):
            raise ValueError("duplicate rewired edges")
        if not np.array_equal(np.sort(a["rows"][a["rows"] == a["cols"]]),
                              np.sort(ra["rows"][ra["rows"] == ra["cols"]])):
            raise ValueError("self-loop policy differs")
        result.append({"path": str(directory), "sha256": m["graph_sha256"],
                       "inOutDegreesPreserved": True, "edgeOverlap": len(np.intersect1d(oldkeys, keys))/len(keys),
                       "rewiring": m.get("rewiring")})
    return {"realSha256": rm["graph_sha256"], "neurons": n, "edges": len(oldkeys), "controls": result}


def audit_interface_reachability(directory: Path, *, interface_seed: int = 20260914,
                                 dimension: int = 128, token_group: int = 8,
                                 output_group: int = 8, max_hops: int = 8) -> dict:
    """Reconstruct the fixed interface and count directed input-to-output paths."""
    if min(dimension, token_group, output_group, max_hops) < 1:
        raise ValueError("interface dimensions and max_hops must be positive")
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_hash(directory / "graph.npz") != manifest["graph_sha256"]:
        raise ValueError("graph hash mismatch")
    with np.load(directory / "graph.npz") as graph:
        rows = graph["rows"].astype(np.int64, copy=False)
        cols = graph["cols"].astype(np.int64, copy=False)
        neurons = len(graph["body_ids"])
    sensory_groups, _ = anatomical_sensory_groups(directory)
    sensory = np.concatenate(sensory_groups)
    rng = np.random.default_rng(interface_seed)
    input_count = dimension * token_group
    candidates = np.setdiff1d(np.unique(cols), sensory)
    if len(candidates) < input_count:
        raise ValueError("not enough input-interface neurons")
    inputs = rng.choice(candidates, input_count, replace=False)
    output_count = dimension * output_group
    output_candidates = np.setdiff1d(np.unique(rows), np.concatenate((inputs, sensory)))
    if len(output_candidates) < output_count:
        raise ValueError("not enough output-interface neurons")
    outputs = rng.choice(output_candidates, output_count, replace=False)

    visited = np.zeros(neurons, dtype=bool)
    frontier = np.zeros(neurons, dtype=bool)
    visited[inputs] = True
    frontier[inputs] = True
    cumulative = {"0": int(visited[outputs].sum())}
    for hop in range(1, max_hops + 1):
        reached = np.zeros(neurons, dtype=bool)
        reached[rows[frontier[cols]]] = True
        frontier = reached & ~visited
        visited |= reached
        cumulative[str(hop)] = int(visited[outputs].sum())
        if not frontier.any():
            for remaining in range(hop + 1, max_hops + 1):
                cumulative[str(remaining)] = cumulative[str(hop)]
            break
    return {
        "path": str(directory), "graphSha256": manifest["graph_sha256"],
        "interfaceSeed": interface_seed, "inputNeurons": input_count,
        "outputNeurons": output_count,
        "inputOutputOverlap": int(np.intersect1d(inputs, outputs).size),
        "outputsReachableWithinHops": cumulative,
        "unreachableOutputsAfterMaxHops": output_count - cumulative[str(max_hops)],
        "maxHops": max_hops,
    }
