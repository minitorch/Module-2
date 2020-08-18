"""
Implementation of the core Tensor object for autodifferentiation.

"""

from .autodiff import FunctionBase, Variable
from . import operators
import random
from . import tensor_ops
from .util import wrap_tuple, assert_close
from .tensor_data import TensorData


# Construction
def zeros(shape):
    return Tensor.make([0] * int(operators.prod(shape)), shape)


def rand(shape):
    """
    Produce a random tensor of size `shape`.

    Args:
       shape (tuple): shape of tensor.

    Returns:
       :class:`Tensor` : New tensor
    """

    # TODO: Implement.
    raise NotImplementedError


def tensor(ls, shape=None):
    if not shape:
        shape = (len(ls),)
    return Tensor.make(ls, shape)


def ensure_tensor(b):
    if isinstance(b, (int, float)):
        return tensor([b])
    return b


# Tensor class
class Tensor(Variable):
    def __init__(self, v, back=None, name=None):
        assert isinstance(v, TensorData)
        super().__init__(back, name=name)
        self._tensor = v

    @staticmethod
    def make(storage, shape, strides=None):
        return Tensor(TensorData(storage, shape, strides))

    # Properties
    @property
    def shape(self):
        return self._tensor.shape

    @property
    def dims(self):
        return self._tensor.dims

    def contiguous(self):
        return Copy.apply(self)

    # Functions
    def __add__(self, b):
        return Add.apply(self, ensure_tensor(b))

    def __sub__(self, b):
        return Add.apply(self, -ensure_tensor(b))

    def __mul__(self, b):
        return Mul.apply(self, ensure_tensor(b))

    def __truediv__(self, b):
        return Mul.apply(self, tensor([1 / b]))

    def __lt__(self, b):
        return LT.apply(self, ensure_tensor(b))

    def __gt__(self, b):
        return LT.apply(ensure_tensor(b), self)

    def __neg__(self):
        return Neg.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def relu(self):
        return ReLU.apply(self)

    def log(self):
        return Log.apply(self)

    def sum(self, dim=None):
        return Sum.apply(self, dim)

    def mean(self, dim=None):
        return Mean.apply(self, dim)

    def permute(self, *order):
        return Permute.apply(self, order)

    def view(self, *shape):
        return View.apply(self, shape)

    def __repr__(self):
        return self._tensor.to_string()

    def __getitem__(self, key):
        return self._tensor.get(wrap_tuple(key))

    def __setitem__(self, key, val):
        self._tensor.set(wrap_tuple(key), val)

    @property
    def grad(self):
        return self.derivative

    def expand(self, other):
        ""
        if self.shape == other.shape:
            return other

        shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = zeros(shape)
        Copy.op(other, out=buf)
        if self.shape == shape:
            return buf

        buf2 = zeros(self.shape)
        Sum.op(buf, out=buf2)
        return buf2

    # Internal
    def zeros(self, shape=None):
        if shape is None:
            return zeros(self.shape)
        else:
            return zeros(shape)

    def tuple(self):
        return self._tensor.tuple()

    # Extra
    def get_data(self):
        return Tensor(self._tensor)

    def backward(self, grad_output=None):
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = tensor([1.0])
        super().backward(grad_output)


# Constructors
class Function(FunctionBase):
    data_type = Tensor
    variable = Tensor

    @staticmethod
    def data(a):
        return a._tensor


class Neg(Function):
    op = tensor_ops.map(operators.neg)

    @staticmethod
    def forward(ctx, t1):
        return Neg.op(t1)

    @staticmethod
    def backward(ctx, grad_output):
        return Neg.op(grad_output)


class Add(Function):
    op = tensor_ops.zip(operators.add)

    @staticmethod
    def forward(ctx, t1, t2):
        return Add.op(t1, t2)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class Mul(Function):

    @staticmethod
    def forward(ctx, a, b):
        # TODO: Implement.
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement.
        raise NotImplementedError


class Sigmoid(Function):

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement.
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement.
        raise NotImplementedError


class ReLU(Function):

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement.
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement.
        raise NotImplementedError


class Log(Function):

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement.
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement.
        raise NotImplementedError


class Sum(Function):
    op = tensor_ops.reduce(operators.add)

    @staticmethod
    def forward(ctx, a, dim):
        ctx.save_for_backward(a.shape)
        if dim is not None:
            return Sum.op(a, [dim])
        else:
            return Sum.op(a, list(range(a.dims))).view(1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Mean(Function):
    @staticmethod
    def forward(ctx, a, dim):
        # TODO: Implement.
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement.
        raise NotImplementedError


class LT(Function):
    op = tensor_ops.zip(operators.lt)

    @staticmethod
    def forward(ctx, a, b):
        # TODO: Implement.
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement.
        raise NotImplementedError


class Permute(Function):
    @staticmethod
    def forward(ctx, a, order):
        ctx.save_for_backward(order)
        return Tensor(a._tensor.permute(*order))

    @staticmethod
    def backward(ctx, grad_output):
        order = ctx.saved_values
        order = [a[0] for a in sorted(enumerate(order), key=lambda a: a[1])]
        return Tensor(grad_output._tensor.permute(*order))


class View(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous, "Must be contiguous to view"
        t = Tensor.make(a._tensor._storage, shape)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        original = ctx.saved_values
        return Tensor.make(grad_output._tensor._storage, original)


class Copy(Function):
    op = tensor_ops.map(operators.id)

    @staticmethod
    def forward(ctx, a):
        return Copy.op(a)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def central_difference(f, *vals, arg=0, epsilon=1e-6, ind=None):
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f, *vals):
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = central_difference(f, *vals, arg=i, ind=ind)
        assert_close(x.grad[ind], check)
