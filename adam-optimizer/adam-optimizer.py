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
    grad = np.asarray(grad, dtype=float).copy()
    m = np.asarray(m, dtype=float).copy()
    v = np.asarray(v, dtype=float).copy()
    
    m[:] = beta1*m + (1.0-beta1)*grad
    v[:] = beta2*v + (1.0-beta2)*(grad*grad)
    m_hat = m/(1.0-(beta1**t))
    v_hat = v/(1.0-(beta2**t))
    param -= (lr/(np.sqrt(v_hat)+eps))*m_hat
    return param,m,v