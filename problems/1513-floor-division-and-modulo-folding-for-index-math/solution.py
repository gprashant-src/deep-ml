def term_bounds(terms, ranges):
    const = terms.get("", 0)
    M, m = const, const

    for v, c in terms.items():
        if v == "" or c == 0:
            continue
        
        lo, hi = ranges[v]
        hi -= 1

        if c > 0:
            m += c * lo
            M += c * hi
        elif c < 0:
            m += c * hi
            M += c * lo
    
    return m, M


def fold_div(terms, c, ranges):
    q = {"": 0}
    r = {"": 0}

    const = terms.get("", 0)
    if const % c == 0:
        q[""] = const // c
    else:
        r[""] = const
    
    for v, co in terms.items():
        if v == "" or c == 0:
            continue
        
        if co % c == 0:
            q[v] = co // c
        else:
            r[v] = co
    
    r_min, r_max = term_bounds(r, ranges)

    if 0 <= r_min and r_max < c:
        return q, None
    else:
        return q, r
    

def fold_mod(terms, c, ranges):
    r = {}
    const_drop = False

    for v, co in terms.items():
        if v == "":
            if co % c == 0:
                const_drop = True
            else:
                r[v] = co
        else:
            if co % c != 0:
                r[v] = co
    
    if const_drop or "" not in r:r[""] = 0

    r_min, r_max = term_bounds(r, ranges)

    return r, (0 <= r_min and r_max < c)
        


def unflatten(shape, flat_terms, ranges):
    n = len(shape)
    if n == 0:return []

    strides = [1] * n

    for j in range(n - 2, -1, -1):
        strides[j] = strides[j + 1] * shape[j + 1]
    
    res = []

    for j in range(n):
        s_j = strides[j]
        shape_j = shape[j]

        q, div_rem = fold_div(flat_terms, s_j, ranges)
        r, exact = fold_mod(q, shape_j, ranges)

        if div_rem is None and exact:
            res.append(r)
        else:
            res.append("div/mod")
    
    return res
