import numpy as np

def deterministic_hash(s):
    '''Converts a string to a deterministic integer.'''
    h = 0
    for c in str(s):
        h = (h * 31 + ord(c)) % (2**31)
    return h

def create_hv(dim, seed):
    '''Creates a bipolar hypervector of given dimension using the seed.'''
    np.random.seed(seed % (2**32 - 1))
    return np.random.choice([-1, 1], dim)

def create_row_hv(row, dim, random_seeds):
    '''Create composite hypervector for a dataset row.
    
    Hint: For each feature, the value seed should combine the base seed
    with the hashed value using modular arithmetic.
    '''
    res = np.zeros(dim)

    for name, val in row.items():
        base = random_seeds[name]
        value_seed = (base + deterministic_hash(val)) % (2 ** 32 - 1)

        name_v = create_hv(dim, base)
        val_v = create_hv(dim, value_seed)

        res = res + (name_v * val_v)
    
    return np.where(res >=0 , 1, -1)
    

