from typing import Tuple, Any
import numpy as np
import pandas as pd


def to_str_tuple(x) -> Tuple[Any, ...]:
    if isinstance(x, (str, bytes)):
        return (x,)
    elif isinstance(x, list):
        return tuple(x)
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, pd.Series):
        return tuple(x.values.tolist())
    elif isinstance(x, np.ndarray):
        return tuple(x.tolist())
    raise TypeError(f"Expected string or tuple of strings, got {type(x)}")
