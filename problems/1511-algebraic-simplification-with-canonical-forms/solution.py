def size(expr):
    if type(expr) != tuple: return 1
    return 1 + sum(size(x) for x in expr[1:])

def _node(e):
    if type(e) != tuple: return e
    op = e[0]
    num = (int, float)

    if op == "neg":
        x = e[1]
        if type(x) in num: return -x
        if type(x) == tuple and x[0] == "neg": return x[1]
        return ("neg", x)

    a, b = e[1], e[2]

    if op == "add":
        if type(a) in num and type(b) in num: return a + b
        if type(a) in num: a, b = b, a
        if type(b) in num and b == 0: return a

        if type(a) == tuple and a[0] == "add" and type(a[2]) in num:
            if type(b) in num: return ("add", a[1], a[2] + b)
            return ("add", ("add", a[1], b), a[2])

        if type(b) == tuple and b[0] == "add" and type(b[2]) in num:
            return ("add", ("add", a, b[1]), b[2])

        return ("add", a, b)

    if op == "mul":
        if type(a) in num and type(b) in num: return a * b  # Fixed here
        if type(a) in num: a, b = b, a
        if type(b) in num:
            if b == 0: return 0
            if b == 1: return a

        if type(a) == tuple and a[0] == "mul" and type(a[2]) in num and type(b) in num:
            return ("mul", a[1], a[2] * b)

        if type(a) == tuple and a[0] == "add" and type(b) in num:
            return ("add", _node(("mul", a[1], b)), _node(("mul", a[2], b)))

        return ("mul", a, b)

def _simplify_once(expr):
    if type(expr) != tuple: return expr
    if expr[0] == "neg":
        child = _simplify_once(expr[1])
        return _node(("neg", child))
    else:
        left = _simplify_once(expr[1])
        right = _simplify_once(expr[2])
        return _node((expr[0], left, right))

def simplify(expr):
    while True:
        new = _simplify_once(expr)
        if new == expr:
            return expr
        expr = new