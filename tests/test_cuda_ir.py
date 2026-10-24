"""Tests for CUDA Intermediate Representation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.cuda_ir import build_ir, compare_ir_with_cuda, ir_summary


def test_build_ir_element_wise():
    source = "def vec_add(a, b, c, n):\n    for i in range(n):\n        c[i] = a[i] + b[i]\n"
    ir = build_ir(source)
    assert ir.ir_generated is True
    assert ir.function_name == "vec_add"
    assert ir.pattern is not None
    assert ir.pattern.pattern_type in ("element_wise", "reduction")
    assert len(ir.inputs) >= 1
    assert len(ir.outputs) >= 1


def test_build_ir_reduction():
    source = "def array_sum(a, n):\n    total = 0.0\n    for i in range(n):\n        total += a[i]\n    return total\n"
    ir = build_ir(source)
    assert ir.pattern is not None
    assert ir.pattern.pattern_type == "reduction"


def test_build_ir_stencil():
    source = "def stencil(a, out, n):\n    for i in range(1, n - 1):\n        out[i] = (a[i-1] + a[i+1]) / 2.0\n"
    ir = build_ir(source)
    assert ir.pattern is not None
    assert ir.pattern.pattern_type in ("stencil", "element_wise")


def test_build_ir_matrix_mul():
    source = "def mat_mul(A, B, C, n):\n    for i in range(n):\n        for j in range(n):\n            for k in range(n):\n                C[i*n+j] += A[i*n+k] * B[k*n+j]\n"
    ir = build_ir(source)
    assert ir.pattern is not None
    assert ir.pattern.pattern_type == "matrix_mul"


def test_build_ir_syntax_error():
    ir = build_ir("def broken(:\n    pass\n")
    assert ir.ir_cuda_notes == "Failed to parse Python source"


def test_compare_ir_with_cuda():
    ir = build_ir("def vec_add(a, b, c, n):\n    for i in range(n):\n        c[i] = a[i] + b[i]\n")
    cuda = "__global__ void vec_add(float* a, float* b, float* c, int n) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < n) c[idx] = a[idx] + b[idx];\n}"
    result = compare_ir_with_cuda(ir, cuda)
    assert "matches" in result
    assert "mismatches" in result
    assert "warnings" in result
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0


def test_ir_summary():
    ir = build_ir("def vec_add(a, b, c, n):\n    for i in range(n):\n        c[i] = a[i] + b[i]\n")
    s = ir_summary(ir)
    assert "Function: vec_add" in s
    assert "Pattern:" in s