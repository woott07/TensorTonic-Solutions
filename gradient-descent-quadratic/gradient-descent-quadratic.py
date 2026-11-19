def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # Write code here
    target_x = -b/(2*a)
    for i in range(steps):
        grad = (2*a*x0) + b
        x0 = x0 - lr*grad
    return x0