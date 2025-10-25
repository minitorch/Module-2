from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    Uses the central difference formula: f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant for finite difference approximation

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # convert vals to list  and copy
    vals_list = list(vals)

    # increment the arg by epsilon
    vals_plus = vals_list.copy()
    vals_plus[arg] += epsilon  #

    # decrement the arg by epsilon
    vals_minus = vals_list.copy()
    vals_minus[arg] -= epsilon

    # compute the forward difference
    forward_diff = f(*vals_plus) - f(*vals_minus)

    # compute the central difference
    central_diff = forward_diff / (2 * epsilon)

    return central_diff


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x: value to be accumulated

        """
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique id of this variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable is a leaf."""
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parents of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Apply the chain rule to compute gradients for parent variables."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.

    Hints:
        - Use depth-first search (DFS) to visit nodes
        - Track visited nodes to avoid cycles (use node.unique_id)
        - Return nodes in reverse order (dependencies first)

    """
    # create a set to track visited nodes
    visited = set()

    # create a list to store the topological order
    topological_order = []

    # define a helper function to perform DFS
    def dfs(node: Variable) -> None:
        """Perform DFS on the variable graph.

        Args:
            node: The current variable to visit

        """
        # Skip if already visited or if it's a constant
        if node.unique_id in visited or node.is_constant():
            return

        # add to visited set
        visited.add(node.unique_id)

        # recursively visit parents
        for parent in node.parents:
            dfs(parent)

        topological_order.append(node)

    # start DFS from the variable
    dfs(variable)

    # return the topological order in reverse order
    return reversed(topological_order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    Hints:
        - First get all nodes in topological order using topological_sort()
        - Create a dictionary to store derivatives for each node (keyed by unique_id)
        - Initialize the starting node's derivative to the input deriv
        - Process nodes in the topological order (which is already correct for backprop)
        - For leaf nodes: call node.accumulate_derivative(derivative)
        - For non-leaf nodes: call node.chain_rule(derivative) to get parent derivatives
        - Sum derivatives when the same parent appears multiple times

    """
    # get all nodes in topological order
    topological_order = topological_sort(variable)

    # create a dictionary to store derivatives for each node
    derivatives = {}

    # initialize the starting node's derivative
    derivatives[variable.unique_id] = deriv

    # process nodes in topological order
    for node in topological_order:
        # get the derivative for the current node
        derivative = derivatives[node.unique_id]

        # handle leaf vs. non-leaf nodes
        if node.is_leaf():
            # for leaf nodes: accumulate the derivative
            node.accumulate_derivative(derivative)
        else:
            # for non-leaf nodes: call chain_rule to get parent derivatives
            for parent, parent_derivative in node.chain_rule(derivative):
                # sum derivatives when the same parent appears multiple times
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = 0.0
                derivatives[parent.unique_id] += parent_derivative


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors."""
        return self.saved_values
