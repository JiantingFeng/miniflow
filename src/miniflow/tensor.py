# inspired by https://github.com/geohot/tinygrad


# Start from tensor class, which is the base

from typing import List, Optional
import numpy as np


class Tensor:
    '''
    A numpy array wrapper that supports autograd and related operations,
    only support CPU for now.

    Parameters
    ----------
    data : numpy.ndarray
    requires_grad : bool, default=False
    '''

    def __init__(self, data, requires_grad=False) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError('data must be a numpy.ndarray')
        self.data = data
        self.requires_grad = requires_grad
        # gradient tensor (Optional)
        self.grad: Optional[Tensor] = None
        # ctx can be seen as the context where the `Function` is called
        # In forward pass, ctx is used to store the intermediate results
        self.ctx: Optional[Function] = None

    def __repr__(self) -> str:
        return f"<Tensor: {self.data}, requires_grad={self.requires_grad}>"

    # Properties (read-only)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    # Classmethods: wrap some common numpy.ndarray methods
    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        # shape of randn donnot need to be wrapped by tuple
        return cls(np.random.randn(*shape), **kwargs)

    @classmethod
    def uniform(cls, *shape, **kwargs):
        return cls(np.random.uniform(*shape)*2-1, **kwargs)

    @classmethod
    def eye(cls, n, **kwargs):
        return cls(np.eye(n, dtype=np.float32), **kwargs)

    # Topological sort (bfs) and backpropagation
    def deepwalk(self) -> List['Tensor']:
        def bfs(node: Tensor, visited: set,
                nodes: list[Tensor]) -> list[Tensor]:
            visited.add(node)
            if node.ctx:
                [bfs(i, visited, nodes) for i in node.ctx.inputs
                 if i not in visited]
                nodes.append(node)
            return nodes
        return bfs(self, set(), [])

    # backpropagation
    # TODO: Not complete yet
    def backward(self):
        self.grad = Tensor.ones(self.shape, requires_grad=False)

        for node in self.deepwalk()[::-1]:
            if not any(x.requires_grad for x in node.ctx.inputs):
                continue
            grads = node.ctx.backward(node.grad)
            for i, grad in enumerate(grads):
                if grad is None:
                    continue
                if node.ctx.inputs[i].grad is None:
                    node.ctx.inputs[i].grad = grad
                else:
                    node.ctx.inputs[i].grad += grad
            del node.ctx


class Function:
    def __init__(self, *tensors: Tensor) -> None:
        self.inputs = tensors
        self.needs_input_grad = [t.requires_grad for t in self.inputs]
        self.needs_output_grad = any(self.needs_input_grad)
        self.saved_tensors: List[Tensor] = []

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def save_for_backward(self, *x) -> None:
        self.saved_tensors.extend(x)
