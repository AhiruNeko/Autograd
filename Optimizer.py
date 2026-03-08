import numpy as np


class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters

    def zero_grad(self):
        for p in self.parameters:
            p.clear_grad()

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.lr * p.grad

class Momentum(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                self.v[i] = self.momentum * self.v[i] + (1 - self.momentum) * p.grad
                p.data -= self.lr * self.v[i]

class AdaGrad(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr
        self.sum_squared_grad = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                self.sum_squared_grad[i] += p.grad**2
                adaptive_lr = self.lr / (np.sqrt(self.sum_squared_grad[i]) + 1e-7)
                p.data -= adaptive_lr * p.grad
