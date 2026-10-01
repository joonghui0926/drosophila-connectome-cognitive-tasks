import numpy as np
import torch

from flystudy.model import FixedTopologyArithmeticNet, select_interface


def ring_graph(n: int = 200):
    rows = np.arange(n, dtype=np.int64)
    cols = np.roll(rows, 1)
    counts = np.ones(n, dtype=np.float32)
    return rows, cols, counts


def test_model_learns_only_existing_edge_weights_and_returns_101_logits():
    rows, cols, counts = ring_graph()
    interface = select_interface(rows, cols, n_nodes=200, input_group_size=1, output_group_size=1, seed=7)
    model = FixedTopologyArithmeticNet(
        rows,
        cols,
        counts,
        interface,
        steps=3,
        readout_steps=2,
        chunk_size=32,
    )
    encoded = torch.zeros((2, 47))
    encoded[:, [0, 2, 12, 22, 24, 34, 44]] = 1
    before_rows = model.rows.clone()
    before_cols = model.cols.clone()
    logits, trace = model(encoded, capture_activity=True)
    assert logits.shape == (2, 101)
    assert trace["final_state"].shape == (200, 2)
    assert trace["activity_mean"].shape == (3,)
    assert trace["activity_peak"].shape == (3,)
    assert [name for name, _ in model.named_parameters()] == ["theta"]
    loss = logits.square().mean()
    loss.backward()
    assert model.theta.grad is not None
    assert torch.isfinite(model.theta.grad).all()
    assert torch.equal(before_rows, model.rows)
    assert torch.equal(before_cols, model.cols)


def test_model_captures_selected_nodes_for_every_recurrent_step_and_example():
    rows, cols, counts = ring_graph()
    interface = select_interface(rows, cols, n_nodes=200, input_group_size=1, output_group_size=1, seed=7)
    model = FixedTopologyArithmeticNet(rows, cols, counts, interface, steps=4, readout_steps=2, chunk_size=32)
    encoded = torch.zeros((3, 47))
    selected = torch.tensor([1, 8, 21, 55])
    _, trace = model(encoded, capture_nodes=selected)
    assert trace["selected_activity"].shape == (4, 4, 3)
    assert not trace["selected_activity"].requires_grad


def test_interface_is_disjoint_and_reproducible():
    rows, cols, _ = ring_graph()
    first = select_interface(rows, cols, n_nodes=200, input_group_size=1, output_group_size=1, seed=99)
    second = select_interface(rows, cols, n_nodes=200, input_group_size=1, output_group_size=1, seed=99)
    assert np.array_equal(first.input_nodes, second.input_nodes)
    assert np.array_equal(first.output_nodes, second.output_nodes)
    assert not set(first.input_nodes.flat).intersection(first.output_nodes.flat)
