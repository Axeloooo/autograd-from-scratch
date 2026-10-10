"""Neurons, layers and an MLP on top of the autograd engine (lecture 1, micrograd)."""

from __future__ import annotations

import random
from collections.abc import Sequence

from autograd_from_scratch.engine import Value


class Module:
    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, nin: int, nonlin: str | None = "tanh", rng: random.Random | None = None):
        gen = rng or random.Random()
        self.w = [Value(gen.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: Sequence[Value | float]) -> Value:
        act = self.b
        for wi, xi in zip(self.w, x, strict=True):
            act = act + wi * xi
        if self.nonlin == "tanh":
            return act.tanh()
        if self.nonlin == "relu":
            return act.relu()
        return act

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class Layer(Module):
    def __init__(self, nin: int, nout: int, **kwargs: object) -> None:
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]  # type: ignore[arg-type]

    def __call__(self, x: Sequence[Value | float]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int], rng: random.Random | None = None) -> None:
        sizes = [nin, *nouts]
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                nonlin="tanh" if i < len(nouts) - 1 else None,
                rng=rng,
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x: Sequence[Value | float]) -> Value | list[Value]:
        out: Sequence[Value | float] = x
        for layer in self.layers:
            out = layer(out)
        return out[0] if len(out) == 1 else out  # type: ignore[return-value]

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
