"""Render a computation graph as Graphviz DOT text (the lecture's `draw_dot`, no dependency)."""

from __future__ import annotations

from autograd_from_scratch.engine import Value


def trace(root: Value) -> tuple[list[Value], list[tuple[Value, Value]]]:
    """All nodes reachable from `root` and the (child, parent) edges between them."""
    nodes: list[Value] = []
    edges: list[tuple[Value, Value]] = []
    seen: set[Value] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        nodes.append(node)
        for child in node._prev:
            edges.append((child, node))
            stack.append(child)
    return nodes, edges


def to_dot(root: Value) -> str:
    """DOT source for the graph, left to right; paste it into any Graphviz viewer."""
    nodes, edges = trace(root)
    ids = {node: f"n{i}" for i, node in enumerate(nodes)}
    lines = ["digraph {", "  rankdir=LR;"]
    for node in nodes:
        label = f"{node.label} | " if node.label else ""
        text = f"{label}data {node.data:.4f} | grad {node.grad:.4f}"
        lines.append(f'  {ids[node]} [label="{{ {text} }}", shape=record];')
        if node._op:
            lines.append(f'  {ids[node]}_op [label="{node._op}"];')
            lines.append(f"  {ids[node]}_op -> {ids[node]};")
    for child, parent in edges:
        target = f"{ids[parent]}_op" if parent._op else ids[parent]
        lines.append(f"  {ids[child]} -> {target};")
    lines.append("}")
    return "\n".join(lines)
