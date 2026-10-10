from autograd_from_scratch import Value, to_dot, trace


def test_trace_counts_nodes_and_edges() -> None:
    a, b = Value(2.0, label="a"), Value(-3.0, label="b")
    c = a * b + a
    nodes, edges = trace(c)
    assert len(nodes) == 4  # a, b, a*b, a*b + a
    assert len(edges) == 4
    assert all(parent in nodes and child in nodes for child, parent in edges)


def test_to_dot_has_labels_ops_and_gradients() -> None:
    a, b = Value(2.0, label="a"), Value(-3.0, label="b")
    c = a * b
    c.label = "c"
    c.backward()
    dot = to_dot(c)
    assert dot.startswith("digraph {") and dot.endswith("}")
    assert "a | data 2.0000 | grad -3.0000" in dot
    assert "c | data -6.0000 | grad 1.0000" in dot
    assert 'label="*"' in dot


def test_to_dot_escapes_record_syntax_in_labels() -> None:
    x = Value(1.0, label='x|{"y"}<z>')
    assert 'x\\|\\{\\"y\\"\\}\\<z\\> | data' in to_dot(x)
