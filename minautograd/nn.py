from typing import List
import random
from .engine import Variable
from .function import mul, add, tanh

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, input_params: int, activation: bool = True) -> None:
        super().__init__()
        self.w = [Variable(random.uniform(-1,1), label=f"w") for i in range(input_params)]
        self.bias = Variable(0, label="b")
        self.activation = activation
    
    def parameters(self):
        return self.w + [self.bias]
    
    def __call__(self, x):
        wxs = [mul(wi,xi) for (wi,xi) in zip(self.w, x)]
        act = add(*wxs)
        return tanh(act) if self.activation else act
    
    def __repr__(self) -> str:
        return f"{'Tanh' if self.activation else 'Linear'}Neuron({len(self.w)})"
        

class Layer(Module):
    def __init__(self, input_size:int, output_size: int, **kwargs) -> None:
        super().__init__()
        self.neurons = [Neuron(input_size, **kwargs) for _ in range(output_size)]
        
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    

class MLP(Module):

    def __init__(self, num_inputs: int, layer_sizes: List[int]):
        sz = [num_inputs] + layer_sizes
        self.layers = [Layer(sz[i], sz[i+1], activation=i!=len(layer_sizes)-1) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
        