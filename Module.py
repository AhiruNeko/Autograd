from Tensor import Tensor
import numpy as np


class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        all_params = []
        for p in self._parameters.values():
            all_params.append(p)
        for m in self._modules.values():
            all_params.extend(m.parameters())
        return all_params

    def zero_grad(self):
        for p in self.parameters():
            p.clear_grad()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        limit = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(out_features, in_features) * limit)
        if bias:
            self.bias = Tensor(np.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight.T()

        if self.bias:
            out = out + self.bias
        return out

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features})"