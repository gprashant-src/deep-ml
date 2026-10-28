import random

def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

  Returns:
        float: The x-coordinate of the minimum point.
    """
    # Your code here
    def f(x):
        return x**4 - 3*x**3 + 2
        
    def grad(x):
        return 4 * x**3 - 9 * x**2

    def local_gradient_descent(cur_x):
        learning_rate = 1e-3
        tol = 1e-8
        iters = 10000
        for _ in range(iters):
            g = grad(cur_x)
            if abs(g) < tol:
                break
            cur_x = cur_x - learning_rate * g
        return cur_x

    best_x = None
    best_f = float('inf')
    
    for _ in range(20):
        rand_x = random.uniform(-5, 5) 
        candidate_x = local_gradient_descent(rand_x)
        candidate_f = f(candidate_x)
        
        if candidate_f < best_f:
            best_f = candidate_f
            best_x = candidate_x
            
    return best_x