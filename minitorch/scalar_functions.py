from __future__ import annotations

from typing import TYPE_CHECKING
from abc import abstractmethod

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    """Turn a singleton tuple into a value"""
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Perform the backward pass of the function."""
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Perform the forward pass of the function."""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        """Apply the function to the given values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)

    @staticmethod
    @abstractmethod
    def forward(ctx: Context, *args: float) -> float:
        """Computes the forward pass of the function.

        Args:
            ctx (Context): Context object to save information for backward computation.
            *args (float): Input values.

        Returns:
            float: The result of the forward computation.

        Raises:
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("Forward method not implemented.")

    @staticmethod
    @abstractmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass (derivative) of the function.

        Args:
            ctx (Context): Context object containing saved values from forward pass.
            d_output (float): Derivative of the output with respect to some scalar.

        Returns:
            Tuple[float, ...]: The gradients with respect to each input.

        Raises:
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("Backward method not implemented.")


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the function."""
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Perform the backward pass of the function."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the function."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Perform the backward pass of the function."""
        (a,) = ctx.saved_values
        return (operators.log_back(a, d_output),)


### To implement for Task 1.2 and 1.4 ###
# Look at the above classes for examples on how to implement the forward and backward functions
# Use the operators.py file from Module 0


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the function."""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform the backward pass of the function."""
        (a, b) = ctx.saved_values
        # apply chain rule to get partial derivatives with respect to a and b
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the function."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Perform the backward pass of the function."""
        (a,) = ctx.saved_values
        return (operators.inv_back(a, d_output),)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the function."""
        ctx.save_for_backward(a)
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Perform the backward pass of the function."""
        (a,) = ctx.saved_values
        return (operators.neg_back(a, d_output),)


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the function."""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Perform the backward pass of the function."""
        (a,) = ctx.saved_values
        return (operators.sigmoid_back(a, d_output),)


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the function."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Perform the backward pass of the function."""
        (a,) = ctx.saved_values
        return (operators.relu_back(a, d_output),)


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the function."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Perform the backward pass of the function."""
        (a,) = ctx.saved_values
        return (operators.exp_back(a, d_output),)


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the function."""
        ctx.save_for_backward(a, b)
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform the backward pass of the function."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the function."""
        ctx.save_for_backward(a, b)
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform the backward pass of the function."""
        return 0.0, 0.0
