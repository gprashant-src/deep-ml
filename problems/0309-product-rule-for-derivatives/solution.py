import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    a, b = np.array(f_coeffs), np.array(g_coeffs)

    X = np.convolve(a, b)

    if len(X) == 1:
        return [0.0]
    return (X[1:] * np.arange(1, len(X), dtype=float)).tolist()
