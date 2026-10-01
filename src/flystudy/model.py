from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from flystudy.sparse_ops import edge_index_mm


@dataclass(frozen=True)
class Interface:
    input_nodes: np.ndarray
    output_nodes: np.ndarray

    def to_dict(self) -> dict:
        return {
            "input_nodes": self.input_nodes.tolist(),
            "output_nodes": self.output_nodes.tolist(),
        }


def select_interface(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    n_nodes: int,
    input_group_size: int = 64,
    output_group_size: int = 32,
    seed: int = 20260912,
) -> Interface:
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    needed_inputs = 47 * input_group_size
    needed_outputs = 101 * output_group_size
    input_candidates = np.unique(cols)
    output_candidates = np.unique(rows)
    if len(input_candidates) < needed_inputs:
        raise ValueError("not enough neurons with outgoing edges for the input interface")
    rng = np.random.default_rng(seed)
    input_nodes = rng.choice(input_candidates, size=needed_inputs, replace=False)
    remaining_outputs = np.setdiff1d(output_candidates, input_nodes, assume_unique=False)
    if len(remaining_outputs) < needed_outputs:
        raise ValueError("not enough disjoint neurons with incoming edges for the output interface")
    output_nodes = rng.choice(remaining_outputs, size=needed_outputs, replace=False)
    return Interface(
        input_nodes=np.asarray(input_nodes, dtype=np.int64).reshape(47, input_group_size),
        output_nodes=np.asarray(output_nodes, dtype=np.int64).reshape(101, output_group_size),
    )


class FixedTopologyArithmeticNet(nn.Module):
    def __init__(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        counts: np.ndarray,
        interface: Interface,
        *,
        n_nodes: int | None = None,
        steps: int = 64,
        readout_steps: int = 4,
        alpha: float = 0.5,
        gain: float = 1.0,
        input_amplitude: float = 1.0,
        temperature: float = 0.1,
        chunk_size: int = 1_048_576,
    ) -> None:
        super().__init__()
        if steps < 1 or not 1 <= readout_steps <= steps:
            raise ValueError("readout_steps must be in 1..steps")
        rows_array = np.asarray(rows, dtype=np.int64)
        cols_array = np.asarray(cols, dtype=np.int64)
        counts_array = np.asarray(counts, dtype=np.int64)
        if not (rows_array.shape == cols_array.shape == counts_array.shape):
            raise ValueError("rows, cols, and counts must have equal shapes")
        inferred_nodes = int(max(rows_array.max(initial=0), cols_array.max(initial=0)) + 1)
        n_nodes = inferred_nodes if n_nodes is None else int(n_nodes)
        if n_nodes < inferred_nodes:
            raise ValueError("n_nodes is smaller than an edge endpoint")
        self.n_nodes = n_nodes
        self.steps = int(steps)
        self.readout_steps = int(readout_steps)
        self.alpha = float(alpha)
        self.gain = float(gain)
        self.input_amplitude = float(input_amplitude)
        self.temperature = float(temperature)
        self.chunk_size = int(chunk_size)
        self.register_buffer("rows", torch.as_tensor(rows_array, dtype=torch.int64))
        self.register_buffer("cols", torch.as_tensor(cols_array, dtype=torch.int64))
        self.register_buffer("input_nodes", torch.as_tensor(interface.input_nodes, dtype=torch.int64))
        self.register_buffer("output_nodes", torch.as_tensor(interface.output_nodes, dtype=torch.int64))
        indegree = np.bincount(rows_array, minlength=n_nodes).clip(min=1)
        scale = self.gain / np.sqrt(indegree[rows_array])
        self.register_buffer("edge_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.theta = nn.Parameter(torch.empty(len(rows_array), dtype=torch.float32))
        nn.init.normal_(self.theta, mean=0.0, std=1.0)

    def _drive(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 2 or encoded.shape[1] != 47:
            raise ValueError("encoded expressions must have shape [batch, 47]")
        batch_size = encoded.shape[0]
        drive = encoded.new_zeros((self.n_nodes, batch_size))
        group_size = self.input_nodes.shape[1]
        expanded = encoded.transpose(0, 1)[:, None, :].expand(47, group_size, batch_size)
        drive.index_copy_(0, self.input_nodes.reshape(-1), self.input_amplitude * expanded.reshape(-1, batch_size))
        return drive

    def _readout(self, state: torch.Tensor) -> torch.Tensor:
        group_size = self.output_nodes.shape[1]
        selected = state.index_select(0, self.output_nodes.reshape(-1))
        return selected.reshape(101, group_size, state.shape[1]).mean(dim=1).transpose(0, 1)

    def forward(
        self,
        encoded: torch.Tensor,
        *,
        capture_activity: bool = False,
        capture_nodes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = encoded.to(dtype=self.theta.dtype)
        drive = self._drive(encoded)
        state = encoded.new_zeros((self.n_nodes, encoded.shape[0]))
        edge_values = self.edge_scale * self.theta
        readouts: list[torch.Tensor] = []
        activity_mean: list[torch.Tensor] = []
        activity_peak: list[torch.Tensor] = []
        selected_activity: list[torch.Tensor] = []
        for step in range(self.steps):
            recurrent = edge_index_mm(
                edge_values,
                self.rows,
                self.cols,
                state,
                n_nodes=self.n_nodes,
                chunk_size=self.chunk_size,
            )
            state = (1.0 - self.alpha) * state + self.alpha * torch.tanh(drive + recurrent)
            if step >= self.steps - self.readout_steps:
                readouts.append(self._readout(state))
            if capture_activity:
                detached = state.detach().abs()
                activity_mean.append(detached.mean())
                activity_peak.append(detached.amax())
            if capture_nodes is not None:
                selected_activity.append(state.index_select(0, capture_nodes).detach().abs())
        logits = torch.stack(readouts).mean(dim=0) / self.temperature
        trace = {"final_state": state.detach()}
        if capture_activity:
            trace["activity_mean"] = torch.stack(activity_mean)
            trace["activity_peak"] = torch.stack(activity_peak)
        if capture_nodes is not None:
            trace["selected_activity"] = torch.stack(selected_activity)
        return logits, trace
