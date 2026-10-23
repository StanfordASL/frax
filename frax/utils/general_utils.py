"""Assorted utility functions"""

from typing import Callable
import functools
import types

import jax


def highest_matmul_precision_func(func: Callable) -> Callable:
    """Function decorator for full-precision matmuls"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Note: float32 here ONLY affects the matmul precision on float32 matrices
        # (which by default is TF32). This flag does not affect float64 inputs
        with jax.default_matmul_precision("float32"):
            return func(*args, **kwargs)

    return wrapper


def highest_matmul_precision_cls(cls: type) -> type:
    """Class decorator for full-precision matmuls"""
    for name, attr in list(vars(cls).items()):
        if isinstance(attr, types.FunctionType) and not name.startswith("__"):
            setattr(cls, name, highest_matmul_precision_func(attr))
    return cls


def tuplify(arr):
    """Recursively convert a nested structure (for instance, lists, tuples, arrays) to a tuple.
    If arr is not either of these, it returns the original value.

    For instance,
    ```
    tuplify([1, 2, [3, 4]]) == (1, 2, (3, 4))
    tuplify(np.array([[1, 2], [3, 4]])) == ((1, 2), (3, 4))
    tuplify(1) == 1
    tuplify("hello") == "hello"
    ```

    Args:
        arr (Any): A (possibly) nested structure of lists, tuples, arrays, etc.

    Returns:
        Any: All lists/tuples/arrays converted to tuples; other values unchanged
    """
    if isinstance(arr, (list, tuple)):
        return tuple(tuplify(a) for a in arr)
    elif hasattr(arr, "tolist") and callable(arr.tolist):
        # This handles numpy and jax arrays
        return tuplify(arr.tolist())
    else:
        return arr
