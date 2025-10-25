from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward
# Use what you defined in scalar_functions.py

_var_count = 0


class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        """Return the string representation of the scalar."""
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Multiply the scalar by another scalar."""
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Divide the scalar by another scalar."""
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Divide another scalar by this scalar."""
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        """Add the scalar to another scalar."""
        return Add.apply(self, b)

    def __bool__(self) -> bool:
        """Return the boolean representation of the scalar."""
        return bool(self.data)

    def __float__(self) -> float:
        """Return the float representation of the scalar."""
        return float(self.data)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Check if this scalar is less than another scalar."""
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Check if this scalar is greater than another scalar."""
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:  # type: ignore[override]
        """Check if this scalar equals another scalar."""
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtract another scalar from this scalar."""
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        """Negate the scalar."""
        return Neg.apply(self)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Add the scalar to another scalar."""
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Multiply the scalar by another scalar."""
        return self * b

    def log(self) -> Scalar:
        """Apply the log function to the scalar."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Apply the exp function to the scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Apply the sigmoid function to the scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Apply the relu function to the scalar."""
        return ReLU.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is a constant."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Return the parents of the scalar."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply chain rule to compute parent gradients."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        # call backward method of the function that created this scalar
        # this gives the gradients with respect to the inputs of the function
        gradients = h.last_fn.backward(h.ctx, d_output)

        # pair each gradient with the corresponding input
        return zip(h.inputs, gradients)

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.

    Asserts False if derivative is incorrect.

    Args:
        f: function from n-scalars to 1-scalar.
        *scalars: n input scalar values.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
    Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
    but was expecting derivative f'=%f from central difference.
    """
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
