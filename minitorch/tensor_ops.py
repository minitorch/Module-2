"""
"""

from .util import jit, prange
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index_to_position,
    shape_broadcast,
)


@jit
def get(index, stride, shape):
    # Change to index_to_position to begin.
    # return index_to_position(index, stride, shape)
    index_to_position
    return broadcast_index_to_position(index, stride, shape)


def tensor_map(fn):
    """
    Higher-order tensor map function.

    Args:
        fn: function mappings floats-to-floats to apply.
        out (`storage`): storage for out tensor.
        out_shape (tuple): shape for out tensor.
        out_strides (tuple): strides for out tensor.
        in_storage (`storage`): storage for in tensor.
        in_shape (tuple): shape for in tensor.
        in_strides (tuple): strides for in tensor.
    """

    @jit
    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        # TODO: Implement.
        raise NotImplementedError

    return _map


def map(fn):
    f = tensor_map(fn)

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    Higher-order tensor zipWith (or map2) function.

    Args:
        fn: function mappings two floats to float to apply.
        out (`storage`): storage for `out` tensor.
        out_shape (tuple): shape for `out` tensor.
        out_strides (tuple): strides for `out` tensor.
        a_storage (`storage`): storage for `a` tensor.
        a_shape (tuple): shape for `a` tensor.
        a_strides (tuple): strides for `a` tensor.
        b_storage (`storage`): storage for `b` tensor.
        b_shape (tuple): shape for `b` tensor.
        b_strides (tuple): strides for `b` tensor.
    """

    @jit
    def _zip(out, out_shape, out_strides, a, a_shape, a_strides, b, b_shape, b_strides):
        # TODO: Implement.
        raise NotImplementedError

    return _zip


def zip(fn):
    f = tensor_zip(fn)

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    Higher-order tensor reduce function.

    Args:
        fn: function mapping two floats to float for combine.
        out (`storage`): storage for `out` tensor.
        out_shape (tuple): shape for `out` tensor.
        out_strides (tuple): strides for `out` tensor.
        a_storage (`storage`): storage for `a` tensor.
        a_shape (tuple): shape for `a` tensor.
        a_strides (tuple): strides for `a` tensor.
        reduce_shape (tuple): shape of reduction (1 for dimension kept, shape value for dimensions summed out)
        reduce_size (int): size of reduce shape
    """

    @jit
    def _reduce(
        out, out_shape, out_strides, a, a_shape, a_strides, reduce_shape, reduce_size
    ):
        # TODO: Implement.
        raise NotImplementedError

    return _reduce


def reduce(fn):
    f = tensor_reduce(fn)

    def ret(a, dims=None, out=None):
        if out is None:
            out_shape = list(a.shape)
            for d in dims:
                out_shape[d] = 1
            # Other values when not sum.
            out = a.zeros(tuple(out_shape))

        diff = len(a.shape) - len(out.shape)

        reduce_shape = []
        reduce_size = 1
        for i, s in enumerate(a.shape):
            if i < diff or out.shape[i - diff] == 1:
                reduce_shape.append(s)
                reduce_size *= s
            else:
                reduce_shape.append(1)
        print(out.shape, a.shape, reduce_shape, reduce_size)
        assert len(out.shape) == len(a.shape)
        f(*out.tuple(), *a.tuple(), reduce_shape, reduce_size)
        return out

    return ret
