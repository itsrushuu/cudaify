"""Tests for auto-tuning recommendations."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.cuda_ir import build_ir, CudaIR, ComputationPattern
from app.services.auto_tune import generate_tuning_recommendations


def test_generate_tuning_recommendations_element_wise():
    source = "def vec_add(a, b, c, n):\n    for i in range(n):\n        c[i] = a[i] + b[i]\n"
    ir = build_ir(source)
    cuda = "__global__ void kernel(float* a, float* b, float* c, int n) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < n) c[idx] = a[idx] + b[idx];\n}"
    recs = generate_tuning_recommendations(ir, cuda)
    assert "block_size" in recs
    assert "grid_size" in recs
    assert "occupancy" in recs
    assert "memory" in recs
    assert "compute" in recs
    assert "warnings" in recs
    assert "priority" in recs
    assert recs["block_size"] == 256


def test_generate_tuning_recommendations_stencil():
    ir = CudaIR()
    ir.pattern = ComputationPattern(pattern_type="stencil", confidence=0.8)
    cuda = "__global__ void kernel(float* a, float* out, int n) {}\n"
    recs = generate_tuning_recommendations(ir, cuda)
    assert recs["block_size"] == 16


def test_generate_tuning_recommendations_shared_memory_warning():
    ir = CudaIR()
    ir.pattern = ComputationPattern(pattern_type="element_wise", confidence=0.5)
    cuda = "__shared__ float s[256];\n__global__ void kernel() { s[0] = 1; }\n"
    recs = generate_tuning_recommendations(ir, cuda)
    assert any("__syncthreads()" in w for w in recs["warnings"])