import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import minautograd
from minautograd import Variable, MLP
from plot import plot_graph


def test(func):
    def wrapper():
        node = func()
        graph = plot_graph(node)
        graph.render('graph', format="png", view=True, cleanup=True)
    return wrapper

@test
def test_1():
    a = Variable(0.24); a.label="A"
    b = Variable(0.7); b.label = "B"
    c = minautograd.mul(a, b); c.label = "C"
    e = minautograd.mul(c, a); e.label = "E"
    e.backward()
    return e
    
@test
def test_2():
    a = Variable(0.6); a.label="A"
    b = Variable(0.2); b.label = "B"
    c = minautograd.mul(a,b); c.label="C"
    d = Variable(0.1); d.label = "D";
    e = minautograd.add(c,d); e.label = "E"
    f = Variable(0.8); f.label = "F"
    g = minautograd.mul(e, f); g.label = "G"
    h = minautograd.tanh(g); h.label="H"
    h.backward()
    return h

@test
def test_3():
    a = Variable(0.2); a.label = "A"
    b = Variable(0.6); b.label = "B"
    c = minautograd.div(a,b); c.label = "C"
    d = minautograd.relu(c); d.label= "D"
    d.backward()
    return d


@test
def test_mlp():
    x = [Variable(random.uniform(0,1), label=f'x{i}') for i in range(3)]
    layer = MLP(num_inputs=3, layer_sizes=[4,1])
    out = layer(x)
    out.backward()
    return out

def main():
    # test_1()
    # test_2()
    # test_3()
    # test_neuron()
    test_mlp()

if __name__ == "__main__":
    main()