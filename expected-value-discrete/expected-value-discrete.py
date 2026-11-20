import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    # Write code here
    x = np.asarray(x,dtype=float).copy()
    p = np.asarray(p,dtype=float).copy()
    return float(np.dot(x,p))