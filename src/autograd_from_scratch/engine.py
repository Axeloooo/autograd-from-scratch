"""Scalar-valued reverse-mode automatic differentiation (lecture 1, micrograd)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable


class Value:
    """A scalar that remembers how it was computed so gradients can flow back."""

    def __init__(
        self,
        data: float,
        _children: Iterable[Value] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self.label = label
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    @staticmethod
    def _wrap(other: Value | float) -> Value:
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: Value | float) -> Value:
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Value:
        if not isinstance(exponent, (int, float)):
            raise TypeError("only int/float exponents are supported")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        return self + (-self._wrap(other))

    def __rsub__(self, other: Value | float) -> Value:
        return self._wrap(other) + (-self)

    def __radd__(self, other: Value | float) -> Value:
        return self + other

    def __rmul__(self, other: Value | float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * self._wrap(other) ** -1

    def __rtruediv__(self, other: Value | float) -> Value:
        return self._wrap(other) * self**-1

    def backward(self) -> None:
        """Backpropagate from this node: topologically sort, then apply the chain rule."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
