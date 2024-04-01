from abc import abstractmethod
from easydict import EasyDict
from typing import Any, Tuple, Union
import math
from .engine import Variable

class Function:
    def __init__(self) -> None:
        self.ctx = {}
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def backward(self, *args, **kwargs) -> Tuple:
        raise NotImplementedError()
    

class Multiply(Function):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, inp: Variable, other: Variable):
        out = inp * other
        return Variable(out, children=(inp, other), op="*")
    
    def backward(self, grad_output):
        inp_grad = grad_output * self.ctx.other
        other_grad = grad_output * self.ctx.inp
        return inp_grad, other_grad
        
    def __call__(self, inp, scalar):
        self.ctx = EasyDict({"inp": inp, "other": scalar})
        
        output = self.forward(inp, scalar)
        output._grad_func = self.backward
        
        return output

class Add(Function):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, inp: Variable):
        out = sum(x.data for x in inp)
        return Variable(out, children=(inp), op="+")
    
    def backward(self, grad_output):
        return (grad_output,) * len(self.ctx.inp)
        
    def __call__(self, *x):
        self.ctx = EasyDict({"inp": x})
        
        output = self.forward(x)
        output._grad_func = self.backward
        
        return output

class Exp(Function):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Variable):
        out = math.exp(x.data)
        return Variable(out, children=(x,), op="exp")

    def backward(self, grad_output):
        return (grad_output * self.ctx.result, )
        
    def __call__(self, x: Variable):
        self.ctx = EasyDict({"x": x})
        output = self.forward(x)
        self.ctx.result = output
        output._grad_func = self.backward
        return output

class Pow(Function):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Variable, p: Union[int, float]):
        out = math.pow(x.data, p)
        return Variable(out, children=(x, ), op=f"**{p}")

    def backward(self, grad_output):
        local_grad = self.ctx.p * (self.ctx.x.data ** (self.ctx.p - 1))
        return (grad_output * local_grad, )

    def __call__(self, x: Variable, p: Variable):
        self.ctx = EasyDict({"x": x, "p": p})
        output = self.forward(x, p)
        output._grad_func = self.backward
        return output
    
class Tanh(Function):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        res = math.tanh(x.data)
        return Variable(res, children=(x,), op="tanh")
    
    def backward(self, grad_output):
        local_grad = 1 - (self.ctx.result ** 2)
        return (grad_output * local_grad, )
    
    def __call__(self, x: Variable):
        self.ctx = EasyDict({"x": x})
        output = self.forward(x)
        self.ctx.result = output
        output._grad_func = self.backward
        return output

class ReLU(Function):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Variable):
        res = max(0, x.data)
        return Variable(res, children=(x, ), op="relu")
    
    def backward(self, grad_output):
        local_grad = 1 if self.ctx.x.data >= 0 else 0
        return (grad_output* local_grad, )
    
    def __call__(self, x: Variable):
        self.ctx = EasyDict({"x": x})
        output = self.forward(x)
        output._grad_func = self.backward
        return output
    
    
def mul(a,b):
    return Multiply()(a,b)

def add(*x):
    return Add()(*x)

def tanh(x):
    return Tanh()(x)

def div(a, b):
    return Multiply()(a, Pow()(b, -1))

def exp(x):
    return Exp()(x)

def power(x, p: Union[int, float]):
    return Pow()(x, p)

def relu(x):
    return ReLU()(x)

