import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    rows = len(A)
    cols = len(A[0])
    
    transposed = [[A[i][j] for i in range(rows)] for j in range(cols)]
    return np.array(transposed)