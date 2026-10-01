import torch

from flystudy.losses import arithmetic_loss, ordinal_cdf_loss


def logits_for_point(point: int) -> torch.Tensor:
    logits = torch.full((1, 101), -80.0)
    logits[0, point] = 80.0
    return logits


def test_ordinal_loss_respects_numeric_distance():
    target = torch.tensor([49])
    near = ordinal_cdf_loss(logits_for_point(48), target)
    far = ordinal_cdf_loss(logits_for_point(90), target)
    assert torch.isclose(near, torch.tensor(0.01), atol=1e-6)
    assert torch.isclose(far, torch.tensor(0.41), atol=1e-6)
    assert near < far


def test_combined_loss_is_finite_and_differentiable():
    logits = torch.randn(8, 101, requires_grad=True)
    target = torch.tensor([0, 1, 10, 25, 49, 50, 99, 100])
    loss, parts = arithmetic_loss(logits, target, ordinal_lambda=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"loss", "cross_entropy", "ordinal"}
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ce_only_switch_removes_ordinal_term_from_total():
    logits = torch.randn(4, 101)
    target = torch.tensor([1, 2, 3, 4])
    loss, parts = arithmetic_loss(logits, target, ordinal_lambda=0.0)
    assert torch.isclose(loss, parts["cross_entropy"])

