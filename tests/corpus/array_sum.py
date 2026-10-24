"""Array sum reduction corpus: total = sum(a)"""

def array_sum(a, n):
    total = 0.0
    for i in range(n):
        total += a[i]
    return total