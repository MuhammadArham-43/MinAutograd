

class Variable:
    def __init__(self, value, children=(), label: str = "", op: str = None):
        self.data = value
        self.grad = 0
        self._children = children
        self.label = label
        self._grad_func = lambda x: (x,)
        self._op = op
        
    def _backward(self):
        gradients = self._grad_func(self.grad)
        for child, gradient in zip(self._children, gradients):
            child.grad += gradient
        return gradients
        
    def __repr__(self) -> str:
        return f"Variable {self.label }(data: {self.data}, grad: {self.grad})"

    def __mul__(self, other):
        return self.data * other
    def __rmul__(self, other):
        return self.data * other
    def __add__(self, other):
        return self.data + other
    def __radd__(self, other):
        return self.data + other
    def __pow__(self, exponent):
        return self.data ** exponent
    def __sub__(self, other):
        return self.data + (-other)
    def __rsub__(self, other):
        return self.data + (-other)
    def __neg__(self):
        return self.data * -1
    def __rtruediv__(self, other):
        return Variable(self * other**-1, children=(self, other), op="/")
    def __truediv__(self, other):
        return Variable(self * other**-1, children=(self, other), op="/")
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1
        for node in reversed(topo):
            node._backward()