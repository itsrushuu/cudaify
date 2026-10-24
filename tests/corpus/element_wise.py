"""Element-wise with clamping corpus: out[i] = clamp((a[i] + b[i]) * scale, 0, 1)"""

def element_wise(a, b, scale, n):
    out = [0.0] * n
    for i in range(n):
        v = (a[i] + b[i]) * scale
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        out[i] = v
    return out