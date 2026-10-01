"""CSR forward/state-backward, with edge-only weight gradients."""
from __future__ import annotations

import torch


class _CSRMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, state, crow, cols, rows, tcrow, tcols, permutation, chunk_size):
        n = state.shape[0]
        matrix = torch.sparse_csr_tensor(crow, cols, values, size=(n, n), device=values.device)
        output = torch.sparse.mm(matrix, state)
        ctx.save_for_backward(values, state, rows, cols, tcrow, tcols, permutation)
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    def backward(ctx, output_grad):
        values, state, rows, cols, tcrow, tcols, permutation = ctx.saved_tensors
        value_grad = torch.empty_like(values) if ctx.needs_input_grad[0] else None
        if value_grad is not None:
            for start in range(0, len(values), ctx.chunk_size):
                end = min(start + ctx.chunk_size, len(values))
                source = state.index_select(0, cols[start:end].long())
                destination = output_grad.index_select(0, rows[start:end].long())
                value_grad[start:end] = (source * destination).sum(dim=1)
        state_grad = None
        if ctx.needs_input_grad[1]:
            n = state.shape[0]
            transposed = torch.sparse_csr_tensor(tcrow, tcols, values.index_select(0, permutation), size=(n, n), device=values.device)
            state_grad = torch.sparse.mm(transposed, output_grad.contiguous())
        return value_grad, state_grad, None, None, None, None, None, None, None


def csr_edge_mm(values, state, *, crow, cols, rows, tcrow, tcols, permutation, chunk_size=262144):
    return _CSRMM.apply(values, state, crow, cols, rows, tcrow, tcols, permutation, chunk_size)
