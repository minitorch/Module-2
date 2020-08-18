import minitorch
import pytest
from hypothesis import given
from hypothesis.strategies import floats, lists
from .strategies import tensors, shaped_tensors, matmul_tensors

small_floats = floats(min_value=-100, max_value=100, allow_nan=False)


@given(lists(floats(allow_nan=False)))
def test_create(t1):
    t2 = minitorch.tensor(t1)
    for i in range(len(t1)):
        assert t1[i] == t2[i]


one_arg = [
    ("neg", lambda a: -a),
    ("addconstant", lambda a: a + 5),
    ("lt", lambda a: a < 5),
    ("gt", lambda a: a > 5),
    ("subconstant", lambda a: a - 5),
    ("mult", lambda a: 5 * a),
    ("div", lambda a: a / 5),
    ("sig", lambda a: a.sigmoid()),
    ("log", lambda a: (a + 100000).log()),
    ("relu", lambda a: (a + 2).relu()),
]


@given(tensors())
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn, t1):
    minitorch.grad_check(fn[1], t1)
    t2 = fn[1](t1)
    for ind in t2._tensor.indices():
        assert t2[ind] == fn[1](minitorch.Scalar(t1[ind])).data


reduce = [
    ("sum", lambda a: a.sum()),
    ("sum2", lambda a: a.sum(0)),
]


@given(tensors())
@pytest.mark.parametrize("fn", reduce)
def test_reduce(fn, t1):
    minitorch.grad_check(fn[1], t1)


two_arg = [("add", lambda a, b: a + b), ("mul", lambda a, b: a * b)]


@given(shaped_tensors(2))
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(fn, ts):
    t1, t2 = ts
    minitorch.grad_check(fn[1], t1, t2)

    t3 = fn[1](t1, t2)
    for ind in t3._tensor.indices():
        assert (
            t3[ind] == fn[1](minitorch.Scalar(t1[ind]), minitorch.Scalar(t2[ind])).data
        )

    # broadcast check
    minitorch.grad_check(fn[1], t1.sum(0), t2)
    minitorch.grad_check(fn[1], t1, t2.sum(0))


@given(matmul_tensors())
def test_matmul(ts):
    t1, t2 = ts
    minitorch.grad_check(minitorch.matmul, t1, t2)
