from __future__ import annotations

import torch


class _EdgeIndexMM(torch.autograd.Function):
    """Sparse matrix multiplication with gradients only for existing edges.

    Edge orientation is ``row <- col``. Chunking avoids materializing an
    ``edges x batch`` message tensor for the entire MaleCNS graph.
    """

    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        state: torch.Tensor,
        n_nodes: int,
        chunk_size: int,
    ) -> torch.Tensor:
        output = state.new_zeros((n_nodes, state.shape[1]))
        for start in range(0, values.numel(), chunk_size):
            end = min(start + chunk_size, values.numel())
            messages = values[start:end, None] * state.index_select(0, cols[start:end])
            output.index_add_(0, rows[start:end], messages)
        ctx.save_for_backward(values, rows, cols, state)
        ctx.n_nodes = n_nodes
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        values, rows, cols, state = ctx.saved_tensors
        value_grad = torch.empty_like(values) if ctx.needs_input_grad[0] else None
        state_grad = torch.zeros_like(state) if ctx.needs_input_grad[3] else None
        for start in range(0, values.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, values.numel())
            destination_grad = output_grad.index_select(0, rows[start:end])
            if value_grad is not None:
                source_state = state.index_select(0, cols[start:end])
                value_grad[start:end] = (destination_grad * source_state).sum(dim=1)
            if state_grad is not None:
                contribution = values[start:end, None] * destination_grad
                state_grad.index_add_(0, cols[start:end], contribution)
        return value_grad, None, None, state_grad, None, None


def edge_index_mm(
    values: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    state: torch.Tensor,
    *,
    n_nodes: int,
    chunk_size: int = 1_048_576,
) -> torch.Tensor:
    if values.ndim != 1 or rows.shape != values.shape or cols.shape != values.shape:
        raise ValueError("values, rows, and cols must be equal-length vectors")
    if state.ndim != 2 or state.shape[0] != n_nodes:
        raise ValueError("state must have shape [n_nodes, batch]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return _EdgeIndexMM.apply(values, rows, cols, state, n_nodes, chunk_size)

