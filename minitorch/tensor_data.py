"""
Implementation of the internal tensordata object used for Tensor.


To make our code a bit simpler we will seperate out the main `tensor
data` object from the user-facing tensor itself.  Tensor data will
implement a core set of high-level operations and then we will wrap
these with easy user functions. The tensor data object is made up of
three parts. Raw data `storage`, tensor `shape` information, and a
tuple of `strides`.

Storage is where the actual data is kept. It is always a 1-D array of
numbers (for simplicity we will always use doubles for now), of length
`size`. There is really not more to it then that, no matter the
dimensionality or shape of the tensor, the storage is just a long
array. The information about the effective number of dimensions and
their individual size is kept in `shape` which is a tuple with exactly
this information.

In a standard multi-dimensional array this would be enough information
to implement indexing. If we wanted to find `tensor[i, j]` in a
matrix of shape (4, 3) we could simply look it up at position `3* i + j`.

.. image:: bigger.png


We refer to this as a `contiguous` tensor.

Things get more interesting if we transponse dimensions. If we first
transpose the tensor (remember, no copy!) and then lookup `tensor[j,
i]` in this way we would go to position `4 * j + i` which is now
wrong!

.. image:: bigger.png


This tells us that `shape` is not enough information to track positions.

Instead we track addition `stride` information on the tensor class. Whereas
`shape` is user-facing and tells them the semantic shape of tensor, `strides`
is internal and tells us, the implementers, where data is located.

Strides is a tuple where each value represents the distance between
values of a give dimension. For example, strides `(3, 1)` says that
moving one position in the first-dimension requires moving 3
positions in storage, but that moving one position in the
second-dimension requires moving only 1 position in storage.

.. image:: bigger.png


Given a lookup `tensor[i, j]` we can directly use strides to find
the storage position

`stide0 * i + stide1 * j `


`stide0 * index0 + stide1 * index1 + stide2 * index2 ... `


"""


import random
from .operators import prod
from numpy import array, float64, ndarray
from .util import jit, List


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


@jit
def index_to_position(index, strides):
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
       index (tuple): index tuple of ints
       strides (tuple): tensor strides

    Return:
        int : position in storage
    """

    raise NotImplementedError


@jit
def count(position, shape):
    """
    Convert a `position` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
       position (int): current position
       shape (tuple): tensor shape

    Returns:
       list : an index within shape

    """
    raise NotImplementedError


@jit
def broadcast_index_to_position(index, strides, shape):
    """
    Convert an index into a position (see `index_to_position`),
    when the index is from a broadcasted shape. In this case
    it may be larger or with more dimensions then the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removeed.

    Args:
       index (tuple): multidimensional index
       strides (tuple): tensor strides
       shape (tuple): tensor shape

    Returns:
       int: storage position after unbroadcasting and converting.

    """
    raise NotImplementedError


def shape_broadcast(shape1, shape2):
    """
    Broadcast two shapes to create a new union shape.

    Args:
       shape1 (tuple): first shape
       shape2 (tuple): second shape

    Returns:
       tuple: broadcasted shape

    """
    raise NotImplementedError


def strides_from_shape(shape):
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    def __init__(self, storage, shape, strides=None):
        if isinstance(storage, ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = List(strides)
        self._shape = List(shape)
        self.strides = tuple(strides)
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def is_contiguous(self):
        "Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions. "
        raise NotImplementedError

    @staticmethod
    def shape_broadcast(shape_a, shape_b):
        return shape_broadcast(shape_a, shape_b)

    def index(self, index, broadcast=False):
        if broadcast:
            return broadcast_index_to_position(
                List(index), self._strides, List(self.shape)
            )

        # Check for errors
        if len(index) != len(self.shape):
            raise IndexingError(f"Index {index} must be size of {self.shape}.")
        for i, ind in enumerate(index):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {index} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {index} not supported.")

        # Call fast indexing.
        return index_to_position(List(index), self._strides)

    def indices(self):
        lshape = List(self.shape)
        for i in range(self.size):
            yield tuple(count(i, lshape))

    def sample(self):
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key, broadcast=False):
        return self._storage[self.index(key, broadcast)]

    def set(self, key, val, broadcast=False):
        self._storage[self.index(key, broadcast)] = val

    def tuple(self):
        return (self._storage, self._shape, self._strides)

    def permute(self, *order):
        """
        Permute the dimensions of the tensor.

        Args:
           order (list): a permutation of the dimensions

        Returns:
           :class:`TensorData`: a new TensorData with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        raise NotImplementedError

    def to_string(self):
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
