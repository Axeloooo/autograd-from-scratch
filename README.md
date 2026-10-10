# autograd-from-scratch

A tiny scalar autograd engine and neural network library, built from scratch.
It follows Andrej Karpathy's lecture
[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0),
part of the Neural Networks: Zero to Hero series. Everything is plain Python with no dependencies.

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## What is here

| Lecture step                            | Code                                                          |
| --------------------------------------- | ------------------------------------------------------------- |
| `Value` node holding data and grad      | `Value` in `src/autograd_from_scratch/engine.py`              |
| Operators and their local derivatives   | `__add__`, `__mul__`, `__pow__`, `exp`, `log`, `tanh`, `relu` |
| Backpropagation over a topological sort | `Value.backward`                                              |
| Neuron, layer, multi-layer perceptron   | `Neuron`, `Layer`, `MLP` in `nn.py`                           |
| Training loop with gradient descent     | `tests/test_nn.py::test_mlp_learns_tiny_dataset`              |

## Example

```python
from autograd_from_scratch import Value

a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
loss = (a * b + c) * f
loss.backward()
print(loss.data, a.grad, b.grad)  # -8.0 6.0 -4.0
```

Train a small network:

```python
import random
from autograd_from_scratch import MLP

mlp = MLP(3, [4, 4, 1], rng=random.Random(1))
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
for _ in range(50):
    loss = sum((mlp(x) - y) ** 2 for x, y in zip(xs, ys))
    mlp.zero_grad()
    loss.backward()
    for p in mlp.parameters():
        p.data -= 0.05 * p.grad
```

## Verifying gradients

The tests compare every operator against central finite differences, and check the exact
numbers worked out in the lecture.

```
pytest -q
```

## Development

See `CLAUDE.md` for commands and conventions. Commits follow Conventional Commits; work lands on
`devel` through squash-merged pull requests and releases go to `main` from a `release/YYYY-MM-DD`
branch through a squash-merged pull request.

## License

MIT
