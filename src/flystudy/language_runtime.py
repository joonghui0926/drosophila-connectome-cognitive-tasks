"""Materialized data, causal batches, matched models and cross-entropy updates."""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flystudy.language_lexicon import EOS, PAD, SEP, TOKEN, VOCAB, Event, Scene
from flystudy.language_model import ConnectomeLanguageModel
from flystudy.language_corpus import CorpusRecord, SPLITS, event_channel


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def from_payload(raw: dict) -> CorpusRecord:
    events = tuple(Event(**{key: event[key] for key in
                           ("location", "kind", "level", "duration", "speed")})
                   for event in raw["scene"])
    return CorpusRecord(raw["id"], raw["split"], raw["mode"], raw["templateFamily"],
                        raw["variant"], Scene(events), tuple(raw["promptTokens"]),
                        tuple(raw["answerTokens"]))


def load_records(directory: Path, name: str) -> list[CorpusRecord]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("revision") != 2:
        raise ValueError("training requires corpus format revision 2")
    entry = manifest["files"][name]
    path = directory / entry["path"]
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"corpus checksum mismatch: {name}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [from_payload(json.loads(line)) for line in stream]
    if len(rows) != entry["examples"]:
        raise ValueError("corpus record count mismatch")
    return rows


def supervised_words(row: CorpusRecord) -> tuple[str, ...]:
    return row.prompt_words + row.answer_words if row.mode == "text" else row.answer_words


def processed_tokens(row: CorpusRecord) -> int:
    # Surface text plus EOS; grounded prefixes also contain SEP.
    return len(row.prompt_words) + len(row.answer_words) + 1 + (row.mode != "text")


@dataclass
class Batch:
    amplitudes: torch.Tensor
    prefixes: torch.Tensor
    lengths: torch.Tensor
    targets: torch.Tensor
    grounded: torch.Tensor

    def inputs(self):
        return self.amplitudes, self.prefixes, self.lengths, self.targets, self.grounded


def make_batch(rows, *, device, encode_steps=8, no_sensory=False,
               wave_seed=20260921, answer_override=None) -> Batch:
    if not rows or encode_steps < 6:
        raise ValueError("nonempty batch and at least six steps per event required")
    if answer_override is not None and len(answer_override) != len(rows):
        raise ValueError("answer override size mismatch")
    inputs, answers = [], []
    for i, row in enumerate(rows):
        # A text document is predicted from BOS. Its prompt never becomes a
        # conditioning prefix that could reveal the future text targets.
        inputs.append(tuple(TOKEN[w] for w in row.prompt_words) + (SEP,)
                      if row.mode != "text" else ())
        words = supervised_words(row) if answer_override is None else answer_override[i]
        answers.append(tuple(TOKEN[w] for w in words) + (EOS,))
    n = len(rows)
    max_events = max((len(r.scene.events) for r in rows if r.mode != "text"), default=1)
    amplitudes = np.zeros((max_events * encode_steps, 24, n), np.float32)
    prefixes = np.full((n, max(map(len, inputs), default=0)), PAD, np.int64)
    targets = np.full((n, max(map(len, answers))), PAD, np.int64)
    lengths = np.asarray(list(map(len, inputs)), dtype=np.int64)
    grounded = np.asarray([r.mode != "text" for r in rows], dtype=np.bool_)
    for col, row in enumerate(rows):
        prefixes[col, :len(inputs[col])] = inputs[col]
        targets[col, :len(answers[col])] = answers[col]
        if not grounded[col] or no_sensory:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([wave_seed, SPLITS.index(row.split), row.record_id]))
        offset = (max_events - len(row.scene.events)) * encode_steps
        for order, event in enumerate(row.scene.events):
            duration = 2 + 2 * event.duration
            envelope = np.linspace(.6, 1, duration) if event.speed == 0 else np.ones(duration)
            pulse = np.clip((.4 + .4 * event.level) * envelope + rng.normal(0, .003, duration), 0, 1)
            start = offset + order * encode_steps
            amplitudes[start:start + duration, event_channel(event), col] = pulse
    return Batch(*(torch.from_numpy(a).to(device) for a in
                   (amplitudes, prefixes, lengths, targets, grounded)))


class DenseLanguageModel(ConnectomeLanguageModel):
    """Parameter-matched recurrent control with fixed token/sensory interfaces.

    A temporary ring initializes the inherited fixed interface only. It is
    discarded before use. Every dense recurrent weight is trainable.
    """
    def __init__(self, width, **kwargs):
        indices = np.arange(width)
        super().__init__(indices, (indices + 1) % width, width,
                         sensory_groups=[np.array([i]) for i in range(24)], **kwargs)
        for name in ("rows", "cols", "crow", "tcrow", "tcols", "permutation"):
            delattr(self, name)
        self.edge_scale = torch.tensor(1 / math.sqrt(width), dtype=torch.float32)
        generator = torch.Generator().manual_seed(kwargs.get("seed", 11))
        self.theta = nn.Parameter(torch.randn(width, width, generator=generator))

    def _recurrent(self, state, values):
        return values @ state


def tiny_model(*, seed=11, dense=False, token_steps=1,
               token_drive_mode="pulse", leak=.5, token_gain=4.,
               sensor_gain=1., temperature=.5):
    options = dict(seed=seed, dimension=109, token_group=1, output_group=1,
                   backend="csr", temperature=temperature, token_steps=token_steps,
                   token_drive_mode=token_drive_mode, leak=leak,
                   token_gain=token_gain, sensor_gain=sensor_gain)
    if dense:
        return DenseLanguageModel(256, **options)
    nodes = 320
    rows = np.repeat(np.arange(nodes), 3)
    cols = np.asarray([(i + offset) % nodes for i in range(nodes) for offset in (-1, 7, 31)])
    return ConnectomeLanguageModel(rows, cols, nodes,
                                  sensory_groups=[np.array([i]) for i in range(24)], **options)


def update(model, optimizer, rows, *, microbatch, encode_steps, device,
           no_sensory=False, gradient_clip=1.0):
    """One optimizer step. CE is normalized by global loss-bearing tokens."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(len(supervised_words(r)) + 1 for r in rows)
    ordered = sorted(rows, key=lambda r: (r.mode != "text", len(r.scene.events),
                                         len(supervised_words(r)), len(r.prompt_words)))
    total_loss = correct = recurrent_steps = 0
    for start in range(0, len(ordered), microbatch):
        batch = make_batch(ordered[start:start + microbatch], device=device,
                           encode_steps=encode_steps, no_sensory=no_sensory)
        logits = model(*batch.inputs())
        loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(),
                               ignore_index=PAD, reduction="sum")
        (loss / denominator).backward()
        total_loss += float(loss.detach())
        correct += int(((logits.argmax(-1) == batch.targets) & (batch.targets != PAD)).sum())
        recurrent_steps += batch.targets.shape[0] * (batch.amplitudes.shape[0] + 1 +
                           model.token_steps * (batch.prefixes.shape[1] + batch.targets.shape[1]))
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    return {"crossEntropy": total_loss / denominator, "lossTokens": denominator,
            "tokenAccuracy": correct / denominator, "gradientNorm": float(gradient),
            "processedTokens": sum(processed_tokens(r) for r in rows),
            "paddedRecurrentSteps": recurrent_steps}


class TokenMixer:
    """Deterministic 60/20/20 processed-token mixture with resumable offsets."""
    weights = {"report": .6, "qa": .2, "text": .2}

    def __init__(self, rows, *, warmup_tokens=500_000):
        self.pools = {name: [r for r in rows if r.mode == name] for name in self.weights}
        if any(not pool for pool in self.pools.values()):
            raise ValueError("every training mode needs examples")
        self.warmup_tokens = warmup_tokens
        self.offsets = dict.fromkeys(self.weights, 0)
        self.exposure = dict.fromkeys(self.weights, 0)
        self.total = 0

    def state_dict(self):
        return {"offsets": dict(self.offsets), "exposure": dict(self.exposure), "total": self.total}

    def load_state_dict(self, state):
        self.offsets, self.exposure, self.total = dict(state["offsets"]), dict(state["exposure"]), state["total"]

    def take(self, count, remaining):
        rows = []
        for _ in range(count):
            mode = "report" if self.total < self.warmup_tokens else min(
                self.weights, key=lambda k: self.exposure[k] / self.weights[k])
            pool = self.pools[mode]
            row = pool[self.offsets[mode] % len(pool)]
            size = processed_tokens(row)
            if size > remaining:
                break
            self.offsets[mode] += 1
            if self.total >= self.warmup_tokens:
                self.exposure[mode] += size
            self.total += size
            remaining -= size
            rows.append(row)
        return rows
