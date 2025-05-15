from graphviz import Digraph

def trace(root):
    # Build a set of nodes and edges in the graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    nodes, edges = trace(root)
    dot = Digraph(format = 'svg', graph_attr={'rankdir': 'LR'}) # LR: left to right

    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular node ('record')
        dot.node(name = uid, label = f"Label: {n.label} | Data: {n.data:.4f} | Grad: {n.grad:.4f}", shape = 'record')
        if n._op:
            # if this value is the result of an operation, create an op (circular) node
            dot.node(name = uid + n._op, label = n._op, shape = 'circle')
            # and connect the op node to the value node
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot