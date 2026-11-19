import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    param = np.asarray(param, dtype=float).copy()
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float).copy()
    v = np.asarray(v, dtype=float).copy()

    m *= beta1
    m += (1.0 - beta1) * grad
    v *= beta2
    v += (1.0 - beta2) * (grad * grad)

    bias_c1 = 1.0 - beta1 ** t
    bias_c2 = 1.0 - beta2 ** t
    step_size = lr / bias_c1

    param -= step_size * m / (np.sqrt(v / bias_c2) + eps)
    return param, m, v