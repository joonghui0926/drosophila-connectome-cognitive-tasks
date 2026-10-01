"""Prompt-conditioned whole-connectome language model.

The released directed connectome fixes every recurrent edge.  Token/sensory
interfaces and 128-D codebooks are fixed at initialization; one scalar per
connectome edge is the only trainable parameter.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flystudy.language_sparse import csr_edge_mm
from flystudy.language_lexicon import (BOS, EOS, MAX_TARGETS, PAD, SENSE, SEP,
                                       TEXT, VOCAB)
from flystudy.sparse_ops import edge_index_mm


def fixed_codebook(kind: str, dimension: int, vocabulary: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dimension, vocabulary))
    if kind == "orthogonal":
        if dimension < vocabulary:
            raise ValueError("orthogonal dimension must cover the vocabulary")
        matrix, triangular = np.linalg.qr(matrix, mode="reduced")
        matrix *= np.where(np.diag(triangular) < 0, -1, 1)[None, :]
    elif kind == "random":
        matrix /= np.linalg.norm(matrix, axis=0, keepdims=True)
    else:
        raise ValueError(f"unknown codebook {kind}")
    return matrix.astype(np.float32)


class ConnectomeLanguageModel(nn.Module):
    def __init__(self, rows, cols, n_nodes: int, *, sensory_groups, seed: int = 11,
                 interface_seed: int = 20260914, embedding: str = "orthogonal",
                 dimension: int = 128, token_group: int = 8, output_group: int = 8,
                 token_steps: int = 1, leak: float = 0.5, token_gain: float = 4.0,
                 sensor_gain: float = 1.0, temperature: float = 0.1,
                 backend: str = "csr", chunk_size: int = 262_144,
                 token_drive_mode: str = "sustained"):
        super().__init__()
        if token_steps < 1 or not 0 < leak <= 1 or temperature <= 0:
            raise ValueError("invalid recurrent configuration")
        if backend not in ("csr", "edge"):
            raise ValueError("backend must be csr or edge")
        if token_drive_mode not in ("pulse", "sustained"):
            raise ValueError("token_drive_mode must be pulse or sustained")
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if rows.ndim != 1 or rows.shape != cols.shape or not len(rows):
            raise ValueError("invalid graph vectors")
        if (np.any(rows[1:] < rows[:-1]) or min(rows.min(), cols.min()) < 0
                or max(rows.max(), cols.max()) >= n_nodes):
            raise ValueError("graph must be in valid postsynaptic CSR row order")
        if len(sensory_groups) != 24:
            raise ValueError("the model requires 24 anatomical sensory groups")

        self.n_nodes = int(n_nodes)
        self.dimension = int(dimension)
        self.token_steps = int(token_steps)
        self.leak = float(leak)
        self.token_gain = float(token_gain)
        self.sensor_gain = float(sensor_gain)
        self.temperature = float(temperature)
        self.backend = backend
        self.chunk_size = int(chunk_size)
        self.token_drive_mode = token_drive_mode

        indegree = np.bincount(rows, minlength=n_nodes)
        crow = np.concatenate(([0], np.cumsum(indegree)))
        permutation = np.argsort(cols, kind="stable")
        tcrow = np.concatenate(([0], np.cumsum(np.bincount(cols, minlength=n_nodes))))
        buffers = {
            "rows": rows.astype(np.int32),
            "cols": cols.astype(np.int32),
            "crow": crow.astype(np.int32),
            "tcrow": tcrow.astype(np.int32),
            "tcols": rows[permutation].astype(np.int32),
            "permutation": permutation.astype(np.int64),
            "edge_scale": (1 / np.sqrt(indegree[rows].clip(min=1))).astype(np.float32),
        }
        for name, value in buffers.items():
            self.register_buffer(name, torch.from_numpy(value))

        sensor_projection = np.zeros((n_nodes, 24), dtype=np.float32)
        sensor_nodes = []
        for group_index, values in enumerate(sensory_groups):
            group = np.asarray(values, dtype=np.int64)
            if group.ndim != 1 or not len(group) or group.min() < 0 or group.max() >= n_nodes:
                raise ValueError("invalid anatomical sensory group")
            sensor_projection[group, group_index] = 1 / np.sqrt(len(group))
            sensor_nodes.extend(group.tolist())
        if len(sensor_nodes) != len(set(sensor_nodes)):
            raise ValueError("anatomical sensory groups overlap")
        self.register_buffer("sensor_projection", torch.from_numpy(sensor_projection))

        rng = np.random.default_rng(interface_seed)
        input_count = dimension * token_group
        candidates = np.setdiff1d(np.unique(cols), np.asarray(sensor_nodes))
        if len(candidates) < input_count:
            raise ValueError("not enough input-interface neurons")
        inputs = rng.choice(candidates, input_count, replace=False)
        output_count = dimension * output_group
        output_candidates = np.setdiff1d(
            np.unique(rows), np.concatenate((inputs, np.asarray(sensor_nodes))))
        if len(output_candidates) < output_count:
            raise ValueError("not enough output-interface neurons")
        outputs = rng.choice(output_candidates, output_count, replace=False)
        self.register_buffer("token_nodes", torch.from_numpy(inputs.reshape(dimension, token_group)))
        self.register_buffer("output_nodes", torch.from_numpy(outputs.reshape(dimension, output_group)))
        self.register_buffer("input_codebook", torch.from_numpy(
            fixed_codebook(embedding, dimension, len(VOCAB), interface_seed + 1)))
        self.register_buffer("output_codebook", torch.from_numpy(
            fixed_codebook("orthogonal", dimension, len(VOCAB), interface_seed + 2)))

        generator = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(torch.randn(len(rows), generator=generator))

    def _recurrent(self, state: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if self.backend == "edge":
            return edge_index_mm(values, self.rows.long(), self.cols.long(), state,
                                 n_nodes=self.n_nodes, chunk_size=self.chunk_size)
        return csr_edge_mm(values, state, rows=self.rows, cols=self.cols, crow=self.crow,
                           tcrow=self.tcrow, tcols=self.tcols, permutation=self.permutation,
                           chunk_size=self.chunk_size)

    def _token_drive(self, tokens: torch.Tensor) -> torch.Tensor:
        drive = self.theta.new_zeros((self.n_nodes, len(tokens)))
        encoded = self.input_codebook.index_select(1, tokens) * self.token_gain
        encoded = encoded[:, None, :].expand(
            -1, self.token_nodes.shape[1], -1).reshape(-1, len(tokens))
        return drive.index_copy(0, self.token_nodes.flatten(), encoded)

    def _sensor_drive(self, group_amplitudes: torch.Tensor) -> torch.Tensor:
        return self.sensor_projection @ (group_amplitudes * self.sensor_gain)

    def _step(self, state: torch.Tensor, drive: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return (1 - self.leak) * state + self.leak * torch.tanh(
            self._recurrent(state, values) + drive)

    def _advance_token(self, state: torch.Tensor, tokens: torch.Tensor,
                       values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Apply one token pulse, followed by drive-free recurrent propagation.

        In ``pulse`` mode, only the first transition receives token input.
        In ``sustained`` mode, every transition receives the same input.
        """
        drive = self._token_drive(tokens)
        zero_drive = torch.zeros_like(drive)
        for internal in range(self.token_steps):
            applied = drive if internal == 0 or self.token_drive_mode == "sustained" else zero_drive
            updated = self._step(state, applied, values)
            state = torch.where(active[None, :], updated, state)
        return state

    def _readout(self, state: torch.Tensor) -> torch.Tensor:
        pooled = state.index_select(0, self.output_nodes.flatten())
        pooled = pooled.reshape(self.dimension, self.output_nodes.shape[1], -1).mean(1).T
        logits = F.normalize(pooled, dim=1, eps=1e-6) @ self.output_codebook / self.temperature
        mask = torch.zeros(len(VOCAB), dtype=torch.bool, device=state.device)
        mask[[BOS, PAD, SEP, SENSE, TEXT]] = True
        return logits.masked_fill(mask, -1e9)

    def _initial_state(self, amplitudes: torch.Tensor, grounded: torch.Tensor,
                       values: torch.Tensor) -> torch.Tensor:
        state = self.theta.new_zeros((self.n_nodes, amplitudes.shape[2]))
        for step in range(amplitudes.shape[0]):
            state = self._step(state, self._sensor_drive(amplitudes[step]), values)
        mode = torch.where(grounded, torch.full_like(grounded, SENSE, dtype=torch.long),
                           torch.full_like(grounded, TEXT, dtype=torch.long))
        return self._step(state, self._token_drive(mode), values)

    def _condition_prompt(self, state: torch.Tensor, prefixes: torch.Tensor,
                          prefix_lengths: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        for token_index in range(prefixes.shape[1]):
            active = token_index < prefix_lengths
            state = self._advance_token(state, prefixes[:, token_index], values, active)
        return state

    def forward(self, amplitudes: torch.Tensor, prefixes: torch.Tensor,
                prefix_lengths: torch.Tensor, targets: torch.Tensor,
                grounded: torch.Tensor) -> torch.Tensor:
        values = self.theta * self.edge_scale
        state = self._initial_state(amplitudes, grounded, values)
        state = self._condition_prompt(state, prefixes, prefix_lengths, values)
        previous = torch.full((targets.shape[0],), BOS, dtype=torch.long, device=targets.device)
        logits = []
        for token_index in range(targets.shape[1]):
            active = targets[:, token_index] != PAD
            state = self._advance_token(state, previous, values, active)
            logits.append(self._readout(state))
            previous = targets[:, token_index]
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, amplitudes: torch.Tensor, prefixes: torch.Tensor,
                 prefix_lengths: torch.Tensor, grounded: torch.Tensor,
                 *, max_tokens: int = MAX_TARGETS) -> torch.Tensor:
        values = self.theta * self.edge_scale
        state = self._initial_state(amplitudes, grounded, values)
        state = self._condition_prompt(state, prefixes, prefix_lengths, values)
        previous = torch.full((amplitudes.shape[2],), BOS, dtype=torch.long,
                              device=amplitudes.device)
        done = torch.zeros(amplitudes.shape[2], dtype=torch.bool, device=amplitudes.device)
        generated = []
        for _ in range(max_tokens):
            state = self._advance_token(state, previous, values, ~done)
            token = self._readout(state).argmax(-1)
            token = torch.where(done, torch.full_like(token, EOS), token)
            generated.append(token)
            done |= token == EOS
            previous = token
            if bool(done.all()):
                break
        return torch.stack(generated, dim=1)
