"""
Implementation of the core Tensor object for autodifferentiation.
"""

from .autodiff import FunctionBase, Variable
from . import operators
import random
from .tensor_ops import TensorOps
from .tensor_data import TensorData
import numpy as np


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
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    return Tensor.make(vals, shape)


def tensor(ls, shape=None):
    if not shape:
        shape = (len(ls),)
    return Tensor.make(ls, shape)


def tensor_fromlist(ls):
    def shape(ls):
        if isinstance(ls, list):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls):
        if isinstance(ls, list):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape = shape(ls)
    return tensor(cur, tuple(shape))


# Tensor class
class Tensor(Variable):
    def __init__(self, v, back=None, name=None, backend=None):
        assert isinstance(v, TensorData)
        super().__init__(back, name=name)
        self._tensor = v
        self.tf = backend
        if backend is None:
            self.tf = TensorFunctions

    def _new(self, tensor_data):
        return Tensor(tensor_data, backend=self.tf)

    @staticmethod
    def make(storage, shape, strides=None, backend=None):
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def type_(self, tf):
        self.tf = tf
        if "Cuda" in str(tf._backend):
            self._tensor.to_cuda_()

    # Properties
    @property
    def shape(self):
        return self._tensor.shape

    @property
    def size(self):
        return self._tensor.size

    @property
    def dims(self):
        return self._tensor.dims

    def to_numpy(self):
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def contiguous(self):
        return self.tf.Copy.apply(self)

    def ensure_tensor(self, b):
        if isinstance(b, (int, float)):
            b = tensor([b])
        b.type_(self.tf)
        return b

    # Functions
    def __add__(self, b):
        return self.tf.Add.apply(self, self.ensure_tensor(b))

    def __sub__(self, b):
        return self.tf.Add.apply(self, -self.ensure_tensor(b))

    def __mul__(self, b):
        return self.tf.Mul.apply(self, self.ensure_tensor(b))

    def __truediv__(self, b):
        return self.tf.Mul.apply(self, self.tf.Inv.apply(self.ensure_tensor(b)))

    def __lt__(self, b):
        return self.tf.LT.apply(self, self.ensure_tensor(b))

    def __eq__(self, b):
        return self.tf.EQ.apply(self, self.ensure_tensor(b))

    def __gt__(self, b):
        return self.tf.LT.apply(self.ensure_tensor(b), self)

    def __neg__(self):
        return self.tf.Neg.apply(self)

    def sigmoid(self):
        return self.tf.Sigmoid.apply(self)

    def relu(self):
        return self.tf.ReLU.apply(self)

    def log(self):
        return self.tf.Log.apply(self)

    def exp(self):
        return self.tf.Exp.apply(self)

    def sum(self, dim=None):
        return self.tf.Sum.apply(self, dim)

    def mean(self, dim=None):
        return self.tf.Mean.apply(self, dim)

    def permute(self, *order):
        return self.tf.Permute.apply(self, order)

    def view(self, *shape):
        return self.tf.View.apply(self, shape)

    def __repr__(self):
        return self._tensor.to_string()

    def __getitem__(self, key):
        return self._tensor.get(key)

    def __setitem__(self, key, val):
        self._tensor.set(key, val)

    @property
    def grad(self):
        return self.derivative

    def expand(self, other):
        ""
        if self.shape == other.shape:
            return other

        shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = zeros(shape)
        self.tf._id_map(other, out=buf)
        if self.shape == shape:
            return buf

        buf2 = zeros(self.shape)
        self.tf._add_reduce(buf, out=buf2)
        return buf2

    # Internal
    def zeros(self, shape=None):

        if shape is None:
            out = zeros(self.shape)
        else:
            out = zeros(shape)
        out.type_(self.tf)
        return out

    def tuple(self):
        return self._tensor.tuple()

    # Extra
    def get_data(self):
        return Tensor(self._tensor, backend=self.tf)

    def backward(self, grad_output=None):
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = tensor([1.0])
            grad_output.tf = self.tf
        super().backward(grad_output)


# Constructors
class Function(FunctionBase):
    data_type = Tensor

    @staticmethod
    def variable(data, back):
        t = Tensor(data[0], back)
        t.type_(data[1])
        return t

    @staticmethod
    def data(a):
        return (a._tensor, a.tf)


def make_tensor_functions(backend):
    neg_map = backend.map(operators.neg)
    sigmoid_map = backend.map(operators.sigmoid)
    relu_map = backend.map(operators.relu)
    log_map = backend.map(operators.log)
    exp_map = backend.map(operators.exp)
    id_map = backend.map(operators.id)
    inv_map = backend.map(operators.inv)

    add_zip = backend.zip(operators.add)
    mul_zip = backend.zip(operators.mul)
    lt_zip = backend.zip(operators.lt)
    eq_zip = backend.zip(operators.eq)
    relu_back_zip = backend.zip(operators.relu_back)
    log_back_zip = backend.zip(operators.log_back)
    inv_back_zip = backend.zip(operators.inv_back)

    add_reduce = backend.reduce(operators.add)

    class TF:
        _add_reduce = add_reduce
        _id_map = id_map
        _backend = backend

        class Neg(Function):
            @staticmethod
            def forward(ctx, t1):
                return neg_map(t1)

            @staticmethod
            def backward(ctx, grad_output):
                return neg_map(grad_output)

        class Inv(Function):
            @staticmethod
            def forward(ctx, t1):
                ctx.save_for_backward(t1)
                return inv_map(t1)

            @staticmethod
            def backward(ctx, grad_output):
                t1 = ctx.saved_values
                return inv_back_zip(t1, grad_output)

        class Add(Function):
            @staticmethod
            def forward(ctx, t1, t2):
                return add_zip(t1, t2)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        class Mul(Function):
            @staticmethod
            def forward(ctx, a, b):
                # TODO: Implement for Task 2.2.
                raise NotImplementedError('Need to implement for Task 2.2')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.3.
                raise NotImplementedError('Need to implement for Task 2.3')

        class Sigmoid(Function):
            @staticmethod
            def forward(ctx, a):
                # TODO: Implement for Task 2.2.
                raise NotImplementedError('Need to implement for Task 2.2')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.3.
                raise NotImplementedError('Need to implement for Task 2.3')

        class ReLU(Function):
            @staticmethod
            def forward(ctx, a):
                # TODO: Implement for Task 2.2.
                raise NotImplementedError('Need to implement for Task 2.2')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.3.
                raise NotImplementedError('Need to implement for Task 2.3')

        class Log(Function):
            @staticmethod
            def forward(ctx, a):
                # TODO: Implement for Task 2.2.
                raise NotImplementedError('Need to implement for Task 2.2')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.3.
                raise NotImplementedError('Need to implement for Task 2.3')

        class Exp(Function):
            @staticmethod
            def forward(ctx, a):
                # TODO: Implement for Task 2.2.
                raise NotImplementedError('Need to implement for Task 2.2')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.3.
                raise NotImplementedError('Need to implement for Task 2.3')

        class Sum(Function):
            @staticmethod
            def forward(ctx, a, dim):
                ctx.save_for_backward(a.shape)
                if dim is not None:
                    return add_reduce(a, [dim])
                else:
                    return add_reduce(a, list(range(a.dims))).view(1)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        class Mean(Function):
            @staticmethod
            def forward(ctx, a, dim):
                # TODO: Implement for Task 2.2.
                raise NotImplementedError('Need to implement for Task 2.2')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.3.
                raise NotImplementedError('Need to implement for Task 2.3')

        class LT(Function):
            @staticmethod
            def forward(ctx, a, b):
                # TODO: Implement for Task 2.2.
                raise NotImplementedError('Need to implement for Task 2.2')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.3.
                raise NotImplementedError('Need to implement for Task 2.3')

        class EQ(Function):
            @staticmethod
            def forward(ctx, a, b):
                # TODO: Implement for Task 2.2.
                raise NotImplementedError('Need to implement for Task 2.2')

            @staticmethod
            def backward(ctx, grad_output):
                # TODO: Implement for Task 2.3.
                raise NotImplementedError('Need to implement for Task 2.3')

        class Permute(Function):
            @staticmethod
            def forward(ctx, a, order):
                ctx.save_for_backward(order)
                return a._new(a._tensor.permute(*order))

            @staticmethod
            def backward(ctx, grad_output):
                order = ctx.saved_values
                order = [a[0] for a in sorted(enumerate(order), key=lambda a: a[1])]
                return grad_output._new(grad_output._tensor.permute(*order))

        class View(Function):
            @staticmethod
            def forward(ctx, a, shape):
                ctx.save_for_backward(a.shape)
                assert a._tensor.is_contiguous, "Must be contiguous to view"
                t = Tensor.make(a._tensor._storage, shape)
                t.type_(a.tf)
                return t

            @staticmethod
            def backward(ctx, grad_output):
                original = ctx.saved_values
                ret = Tensor.make(grad_output._tensor._storage, original)
                ret.type_(grad_output.tf)
                return ret

        class Copy(Function):
            @staticmethod
            def forward(ctx, a):
                return id_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

    return TF


TensorFunctions = make_tensor_functions(TensorOps)

# Uncomment for Module 3
CudaTensorFunctions = None

# from cuda_ops import CudaOps
# CudaTensorFunctions = make_tensor_functions(CudaOps)


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
        np.testing.assert_allclose(x.grad[ind], check, 1e-2, 1e-2)
