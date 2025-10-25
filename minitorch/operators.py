"""Collection of the core mathematical operators used throughout the code base.

This module implements fundamental mathematical operations that serve as building blocks
for neural network computations in MiniTorch.

NOTE: The `task0_1` tests will not fully pass until you complete `task0_3`.
Some tests depend on higher-order functions implemented in the later task.
"""

import math
from typing import Callable, List

# =============================================================================
# Task 0.1: Mathematical Operators
# =============================================================================


# Implementation of elementary mathematical functions.

# FUNCTIONS TO IMPLEMENT:
#     Basic Operations:
#     - mul(x, y)     → Multiply two numbers
#     - id(x)         → Return input unchanged (identity function)
#     - add(x, y)     → Add two numbers
#     - neg(x)        → Negate a number

#     Comparison Operations:
#     - lt(x, y)      → Check if x < y
#     - eq(x, y)      → Check if x == y
#     - max(x, y)     → Return the larger of two numbers
#     - is_close(x, y) → Check if two numbers are approximately equal

#     Activation Functions:
#     - sigmoid(x)    → Apply sigmoid activation: 1/(1 + e^(-x))
#     - relu(x)       → Apply ReLU activation: max(0, x)

#     Mathematical Functions:
#     - log(x)        → Natural logarithm
#     - exp(x)        → Exponential function
#     - inv(x)        → Reciprocal (1/x)

#     Derivative Functions (for backpropagation):
#     - log_back(x, d)  → Derivative of log: d/x
#     - inv_back(x, d)  → Derivative of reciprocal: -d/(x²)
#     - relu_back(x, d) → Derivative of ReLU: d if x>0, else 0

# IMPORTANT IMPLEMENTATION NOTES:

# Numerically Stable Sigmoid:
#    To avoid numerical overflow, use different formulations based on input sign:

#    For x ≥ 0:  sigmoid(x) = 1/(1 + exp(-x))
#    For x < 0:  sigmoid(x) = exp(x)/(1 + exp(x))

#    Why? This prevents computing exp(large_positive_number) which causes overflow.

# is_close Function:
#    Use tolerance: |x - y| < 1e-2
#    This handles floating-point precision issues in comparisons.

# Derivative Functions (Backpropagation):
#    These compute: derivative_of_function(x) × upstream_gradient

#    - log_back(x, d):  d/dx[log(x)] = 1/x  →  return d/x
#    - inv_back(x, d):  d/dx[1/x] = -1/x**2   →  return -d/(x**2)
#    - relu_back(x, d): d/dx[relu(x)] = 1 if x>0 else 0  →  return d if x>0 else 0


# BASIC OPERATIONS
def mul(x: float, y: float) -> float:
    """Multiply two numbers

    Args:
        x: first number
        y: second number

    Returns:
        The product of the two numbers

    """
    return x * y


def id(x: float) -> float:
    """Return input unchanged

    Args:
        x: input number

    Returns:
        The input number

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers

    Args:
        x: first number
        y: second number

    Returns:
        The sum of the two numbers

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number

    Args:
        x: input number

    Returns:
        The negated number

    """
    return -x


# COMPARISON OPERATIONS
def lt(x: float, y: float) -> float:
    """Check if x < y

    Args:
        x: first number
        y: second number

    Returns:
        True if x < y, False otherwise

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x == y

    Args:
        x: first number
        y: second number

    Returns:
        True if x == y, False otherwise

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the larger of two numbers

    Args:
        x: first number
        y: second number

    Returns:
        the larger of the two numbers

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are approximately equal

    Args:
        x: first number
        y: second number

    Returns:
        True if x and y are approximately equal, False otherwise

    """
    return abs(x - y) < 1e-2


# ACTIVATION FUNCTIONS
def sigmoid(x: float) -> float:
    """Apply sigmoid activation: 1/(1 + e^(-x))

    Args:
        x: input number

    Returns:
        The sigmoid of the input number

    """
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    """Apply ReLU activation: max(0, x)

    Args:
        x: input number

    Returns:
        The ReLU of the input number

    """
    return max(0.0, x)


# MATHEMATICAL FUNCTIONS
def log(x: float) -> float:
    """Apply natural logarithm: log(x)

    Args:
        x: input number

    Returns:
        The natural logarithm of the input number

    """
    return math.log(x)


def exp(x: float) -> float:
    """Apply exponential function: e^x

    Args:
        x: input number

    Returns:
        The exponential of the input number

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Apply reciprocal: 1/x

    Args:
        x: input number

    Returns:
        The reciprocal of the input number

    """
    return 1 / x


# DERIVATIVE FUNCTIONS
def log_back(x: float, d: float) -> float:
    """Derivative of log: d/x

    Args:
        x: input number
        d: derivative of the input number

    Returns:
        The derivative of the log of the input number

    """
    return d / x


def inv_back(x: float, d: float) -> float:
    """Derivative of reciprocal: -d/(x²)

    Args:
        x: input number
        d: derivative of the input number

    Returns:
        The derivative of the reciprocal of the input number

    """
    return -d / (x**2)


def relu_back(x: float, d: float) -> float:
    """Derivative of ReLU: d if x>0, else 0

    Args:
        x: input number
        d: derivative of the input number

    """
    return d if x > 0 else 0


def neg_back(x: float, d: float) -> float:
    """Derivative of neg: -d

    Args:
        x: input number
        d: derivative of the input number

    """
    return -d


def exp_back(x: float, d: float) -> float:
    """Derivative of exp: d*e^x

    Args:
        x: input number
        d: derivative of the input number

    """
    return d * math.exp(x)


def sigmoid_back(x: float, d: float) -> float:
    """Derivative of sigmoid: d*sigmoid(x)*(1-sigmoid(x))

    Args:
        x: input number
        d: derivative of the input number

    """
    return d * sigmoid(x) * (1 - sigmoid(x))


# =============================================================================
# Task 0.3: Higher-Order Functions
# =============================================================================


# Implementation of functional programming concepts using higher-order functions.

# These functions work with other functions as arguments, enabling powerful
# abstractions for list operations.

# CORE HIGHER-ORDER FUNCTIONS TO IMPLEMENT:

#     map(fn, iterable):
#         Apply function `fn` to each element of `iterable`
#         Example: map(lambda x: x*2, [1,2,3]) → [2,4,6]

#     zipWith(fn, list1, list2):
#         Combine corresponding elements from two lists using function `fn`
#         Example: zipWith(add, [1,2,3], [4,5,6]) → [5,7,9]

#     reduce(fn, iterable, initial_value):
#         Reduce iterable to single value by repeatedly applying `fn`
#         Example: reduce(add, [1,2,3,4], 0) → 10

# FUNCTIONS TO BUILD USING THE ABOVE:

#     negList(lst):
#         Negate all elements in a list
#         Implementation hint: Use map with the neg function

#     addLists(lst1, lst2):
#         Add corresponding elements from two lists
#         Implementation hint: Use zipWith with the add function

#     sum(lst):
#         Sum all elements in a list
#         Implementation hint: Use reduce with add function and initial value 0

#     prod(lst):
#         Calculate product of all elements in a list
#         Implementation hint: Use reduce with mul function and initial value 1


def map(fn: Callable[[float], float], iterable: List[float]) -> List[float]:
    """Apply function `fn` to each element of `iterable`

    Args:
        fn: function to apply
        iterable: list of numbers

    Returns:
        List of the results of applying `fn` to each element of `iterable`

    """
    return [fn(x) for x in iterable]


def zipWith(
    fn: Callable[[float, float], float], list1: List[float], list2: List[float]
) -> List[float]:
    """Combine corresponding elements from two lists using function `fn`

    Args:
        fn: function to apply
        list1: first list of numbers
        list2: second list of numbers

    Returns:
        List of the results of applying `fn` to each pair of corresponding elements from `list1` and `list2`

    """
    return [fn(x, y) for x, y in zip(list1, list2)]


def reduce(
    fn: Callable[[float, float], float], iterable: List[float], initial_value: float
) -> float:
    """Reduce iterable to single value by repeatedly applying `fn`

    Args:
        fn: function to apply
        iterable: list of numbers
        initial_value: initial value

    Returns:
        The result of applying `fn` to the iterable

    """
    result = initial_value
    for x in iterable:
        result = fn(result, x)
    return result


def negList(lst: List[float]) -> List[float]:
    """Negate all elements in a list

    Args:
        lst: list of numbers

    Returns:
        List of the negated elements

    """
    return map(neg, lst)


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Add corresponding elements from two lists

    Args:
        lst1: first list of numbers
        lst2: second list of numbers

    Returns:
        List of the added elements

    """
    return zipWith(add, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Sum all elements in a list

    Args:
        lst: list of numbers

    Returns:
        The sum of the elements in the list

    """
    return reduce(add, lst, 0)


def prod(lst: List[float]) -> float:
    """Calculate product of all elements in a list

    Args:
        lst: list of numbers

    Returns:
        The product of the elements in the list

    """
    return reduce(mul, lst, 1)
