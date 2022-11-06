# inspired by https://github.com/geohot/tinygrad


# Start from tensor class, which is the base

from typing import Optional
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
