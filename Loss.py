from Tensor import Tensor
import numpy as np

class Loss:
    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        raise NotImplementedError

class MSELoss(Loss):
    def forward(self, pred, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)

        diff = target - pred
        return (diff ** 2).sum() * (1 / pred.data.size)


class BCELoss(Loss):
    def forward(self, pred, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)

        term1 = target * pred.log()
        term2 = (Tensor(1.0) - target) * (Tensor(1.0) - pred).log()

        res = (term1 + term2).sum() * (-1.0 / pred.data.size)
        return res


class CrossEntropyLoss(Loss):
    def forward(self, pred, target):
        exps = np.exp(pred.data - np.max(pred.data, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        batch_size = pred.data.shape[0]
        loss_data = -np.sum(target.data * np.log(probs + 1e-8)) / batch_size

        out = Tensor(np.array(loss_data), _children=(pred, target))

        def _backward():
            pred.grad += (probs - target.data) * out.grad / batch_size

        out._backward = _backward
        return out
