import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    g, h = np.array(g_coeffs), np.array(h_coeffs)

    def dx(f:np.ndarray):
        if len(f) == 1:return np.array([0.0])
        return f[:-1] * np.arange(len(f) - 1, 0, -1)

    def val(f: np.ndarray, num:int):
        return np.dot(f, num ** np.arange(len(f) - 1, -1, -1))
    
    a = np.convolve(h, dx(g))
    b = np.convolve(g, dx(h))

    if len(a) < len(b):
        a = np.pad(a, (len(b) - len(a), 0))
    elif len(b) < len(a):
        b = np.pad(b, (len(a) - len(b), 0))
    num = val(a - b, x)

    den = val(h, x) ** 2
    return num / den
    