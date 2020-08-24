import minitorch
import pytest
from hypothesis import given
from hypothesis.strategies import floats, lists
from .strategies import tensors, shaped_tensors, assert_close

small_floats = floats(min_value=-100, max_value=100, allow_nan=False)

v = 4.524423
one_arg = [
    ("neg", lambda a: -a),
    ("addconstant", lambda a: a + v),
    ("lt", lambda a: a < v),
    ("subconstant", lambda a: a - v),
    ("mult", lambda a: 5 * a),
    ("div", lambda a: a / v),
    ("sig", lambda a: a.sigmoid()),
    ("log", lambda a: (a + 100000).log()),
    ("relu", lambda a: (a + 2).relu()),
    ("exp", lambda a: (a - 200).exp()),
]

reduce = [
    ("sum", lambda a: a.sum()),
    ("mean", lambda a: a.mean()),
    ("sum2", lambda a: a.sum(0)),
    ("mean2", lambda a: a.mean(0)),
]
two_arg = [
    ("add", lambda a, b: a + b),
    ("mul", lambda a, b: a * b),
    ("lt", lambda a, b: a < b + v),
]


@given(lists(floats(allow_nan=False)))
def test_create(t1):
    t2 = minitorch.tensor(t1)
    for i in range(len(t1)):
        assert t1[i] == t2[i]


@given(tensors())
@pytest.mark.task2_2
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn, t1):
    t2 = fn[1](t1)
    for ind in t2._tensor.indices():
        assert_close(t2[ind], fn[1](minitorch.Scalar(t1[ind])).data)


@given(shaped_tensors(2))
@pytest.mark.task2_2
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(fn, ts):
    t1, t2 = ts
    t3 = fn[1](t1, t2)
    for ind in t3._tensor.indices():
        assert (
            t3[ind] == fn[1](minitorch.Scalar(t1[ind]), minitorch.Scalar(t2[ind])).data
        )


@given(tensors())
@pytest.mark.task2_3
@pytest.mark.parametrize("fn", one_arg)
def test_one_derivative(fn, t1):
    minitorch.grad_check(fn[1], t1)


@given(tensors())
@pytest.mark.task2_3
@pytest.mark.parametrize("fn", reduce)
def test_reduce(fn, t1):
    minitorch.grad_check(fn[1], t1)


@given(shaped_tensors(2))
@pytest.mark.task2_3
@pytest.mark.parametrize("fn", two_arg)
def test_two_grad(fn, ts):
    t1, t2 = ts
    minitorch.grad_check(fn[1], t1, t2)


@given(shaped_tensors(2))
@pytest.mark.task2_4
@pytest.mark.parametrize("fn", two_arg)
def test_two_grad_broadcast(fn, ts):
    t1, t2 = ts
    minitorch.grad_check(fn[1], t1, t2)

    # broadcast check
    minitorch.grad_check(fn[1], t1.sum(0), t2)
    minitorch.grad_check(fn[1], t1, t2.sum(0))


def test_fromlist():
    t = minitorch.tensor_fromlist([[2, 3, 4], [4, 5, 7]])
    t.shape == (2, 3)
    t = minitorch.tensor_fromlist([[[2, 3, 4], [4, 5, 7]]])
    t.shape == (1, 2, 3)
