import numpy as np

class Tensor:
    def __init__(self, data, _children=()):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += self._reduce_grad(out.grad, self.data.shape)
            other.grad += self._reduce_grad(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other))

        def _backward():
            self.grad += self._reduce_grad(out.grad, self.data.shape)
            other.grad += self._reduce_grad(-out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += self._reduce_grad(other.data * out.grad, self.data.shape)
            other.grad += self._reduce_grad(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float))
        out = Tensor(self.data ** n, (self,))

        def _backward():
            self.grad += self._reduce_grad(n * (self.data ** (n - 1)) * out.grad, self.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-7), (self,))

        def _backward():
            self.grad += (1.0 / (self.data + 1e-7)) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,))

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    @staticmethod
    def _reduce_grad(grad, shape):
        if not isinstance(grad, np.ndarray):
            grad = np.array(grad)
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))
        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def softmax(self):
        data_max = np.max(self.data, axis=1, keepdims=True)
        exps = np.exp(self.data - data_max)
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        out = Tensor(probs, (self,))

        def _backward():
            sum_grad_times_probs = np.sum(out.grad * probs, axis=1, keepdims=True)
            self.grad += probs * (out.grad - sum_grad_times_probs)

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), (self,))

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def clear_grad(self):
        self.grad = np.zeros_like(self.data)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if not (v in visited):
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        if isinstance(self.data, np.ndarray):
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.array(1.0)
        for node in reversed(topo):
            node._backward()

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *dims):
        out = Tensor(self.data.transpose(*dims), (self,))

        def _backward():
            undo_dims = np.argsort(dims)
            self.grad += out.grad.transpose(*undo_dims)

        out._backward = _backward
        return out

    def T(self):
        return self.transpose(1, 0)
