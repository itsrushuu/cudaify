"""Vector addition corpus: c[i] = a[i] + b[i]"""

def vec_add(a, b, c, n):
    for i in range(n):
        c[i] = a[i] + b[i]
    return c