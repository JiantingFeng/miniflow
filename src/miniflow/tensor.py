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

    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            raise TypeError('data must be a numpy.ndarray')
        self.data = data
        self.requires_grad = requires_grad
        # gradient tensor (Optional)
        self.rgad: Optional[Tensor] = None
