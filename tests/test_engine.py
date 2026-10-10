import math
from collections.abc import Callable

import pytest

from autograd_from_scratch import Value


def numeric_grad(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


def test_lecture_expression() -> None:
    # The expression from the lecture: L = (a*b + c) * f
    a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
    loss = (a * b + c) * f
    loss.backward()
    assert loss.data == -8.0
    assert (a.grad, b.grad, c.grad, f.grad) == (6.0, -4.0, -2.0, 4.0)


def test_value_used_twice_accumulates() -> None:
    a = Value(3.0)
    out = a * a + a
    out.backward()
    assert out.grad == 1.0
    assert a.grad == pytest.approx(2 * 3.0 + 1.0)


def test_neuron_tanh() -> None:
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    b = Value(6.8813735870195432)
    out = (x1 * w1 + x2 * w2 + b).tanh()
    out.backward()
    assert out.data == pytest.approx(0.7071, abs=1e-4)
    assert x1.grad == pytest.approx(-1.5, abs=1e-4)
    assert w1.grad == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    ("op", "ref", "x"),
    [
        (lambda v: v.tanh(), math.tanh, 0.4),
        (lambda v: v.exp(), math.exp, 0.7),
        (lambda v: v.log(), math.log, 1.3),
        (lambda v: v**3, lambda x: x**3, 1.5),
        (lambda v: 1 / v, lambda x: 1 / x, 2.0),
        (lambda v: 2 - v, lambda x: 2 - x, 0.5),
        (lambda v: v / 4, lambda x: x / 4, 0.5),
        (lambda v: v.relu(), lambda x: max(x, 0.0), 0.9),
        (lambda v: v.relu(), lambda x: max(x, 0.0), -0.9),
    ],
)
def test_gradients_match_finite_differences(
    op: Callable[[Value], Value], ref: Callable[[float], float], x: float
) -> None:
    v = Value(x)
    out = op(v)
    out.backward()
    assert out.data == pytest.approx(ref(x))
    assert v.grad == pytest.approx(numeric_grad(ref, x), abs=1e-5)


def test_pow_rejects_value_exponent() -> None:
    with pytest.raises(TypeError):
        Value(2.0) ** Value(2.0)  # type: ignore[operator]


def test_backward_on_a_deep_graph_does_not_overflow_the_stack() -> None:
    x = Value(1.0)
    out = x
    for _ in range(5000):
        out = out + x
    out.backward()
    assert out.data == 5001.0
    assert x.grad == 5001.0
