import numpy as np
import pytest
import torch

from flystudy.sparse_ops import edge_index_mm


def test_edge_index_forward_and_gradients_match_dense_reference():
    # 0 -> 1 (0.7), 1 -> 2 (-0.4), 2 -> 0 (0.2), row=post, col=pre
    rows = torch.tensor([1, 2, 0], dtype=torch.int64)
    cols = torch.tensor([0, 1, 2], dtype=torch.int64)
    values = torch.tensor([0.7, -0.4, 0.2], dtype=torch.float64, requires_grad=True)
    state = torch.tensor([[0.3, -0.2], [-0.4, 0.6], [0.9, 0.1]], dtype=torch.float64, requires_grad=True)
    sparse = edge_index_mm(values, rows, cols, state, n_nodes=3, chunk_size=2)
    sparse.square().sum().backward()
    sparse_value_grad = values.grad.clone()
    sparse_state_grad = state.grad.clone()

    values.grad = None
    state.grad = None
    dense_matrix = torch.zeros((3, 3), dtype=torch.float64)
    dense_matrix[rows, cols] = values
    dense = dense_matrix @ state
    dense.square().sum().backward()

    torch.testing.assert_close(sparse, dense)
    torch.testing.assert_close(sparse_value_grad, values.grad)
    torch.testing.assert_close(sparse_state_grad, state.grad)


def test_sparse_operator_gradcheck():
    rows = torch.tensor([1, 2, 0], dtype=torch.int64)
    cols = torch.tensor([0, 1, 2], dtype=torch.int64)
    values = torch.tensor([0.7, -0.4, 0.2], dtype=torch.float64, requires_grad=True)
    state = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda v, s: edge_index_mm(v, rows, cols, s, n_nodes=3, chunk_size=2),
        (values, state),
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_edge_index_matches_cpu_values_and_gradients():
    generator = torch.Generator().manual_seed(17)
    rows = torch.tensor([0, 0, 1, 2, 3, 3, 4], dtype=torch.int64)
    cols = torch.tensor([1, 3, 2, 0, 1, 4, 3], dtype=torch.int64)
    values = torch.randn(7, generator=generator, requires_grad=True)
    state = torch.randn(5, 3, generator=generator, requires_grad=True)
    output = edge_index_mm(values, rows, cols, state, n_nodes=5, chunk_size=3)
    output.square().sum().backward()

    mps_values = values.detach().to("mps").requires_grad_()
    mps_state = state.detach().to("mps").requires_grad_()
    mps_output = edge_index_mm(
        mps_values,
        rows.to("mps"),
        cols.to("mps"),
        mps_state,
        n_nodes=5,
        chunk_size=3,
    )
    mps_output.square().sum().backward()
    torch.mps.synchronize()

    torch.testing.assert_close(output, mps_output.cpu(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(values.grad, mps_values.grad.cpu(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state.grad, mps_state.grad.cpu(), atol=1e-5, rtol=1e-5)
