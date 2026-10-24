"""Square matrix multiply corpus: C = A * B"""

def mat_mul_square(A, B, C, n):
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += A[i * n + k] * B[k * n + j]
            C[i * n + j] = total
    return C