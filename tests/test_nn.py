import random

from autograd_from_scratch import MLP, Value


def test_parameter_count() -> None:
    mlp = MLP(3, [4, 4, 1], rng=random.Random(0))
    assert len(mlp.parameters()) == 3 * 4 + 4 + 4 * 4 + 4 + 4 * 1 + 1


def test_mlp_learns_tiny_dataset() -> None:
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    mlp = MLP(3, [4, 4, 1], rng=random.Random(1))

    def loss_value() -> Value:
        total = Value(0.0)
        for x, y in zip(xs, ys, strict=True):
            p = mlp(x)
            assert isinstance(p, Value)
            total = total + (p - y) ** 2
        return total

    first = loss_value().data
    for _ in range(50):
        loss = loss_value()
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= 0.05 * p.grad
    assert loss_value().data < first * 0.1
