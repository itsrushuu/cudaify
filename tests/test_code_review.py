"""Tests for static code review."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.code_review import review_cuda_code


def test_review_cuda_code_no_issues():
    cuda = """
__global__ void vec_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch(float* a, float* b, float* c, int n) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaGetLastError();
    cudaMalloc(&d_b, n * sizeof(float));
    cudaGetLastError();
    cudaMalloc(&d_c, n * sizeof(float));
    cudaGetLastError();
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaGetLastError();
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaGetLastError();
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    vec_add<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaGetLastError();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
"""
    review = review_cuda_code(cuda)
    assert "findings" in review
    assert "summary" in review
    assert review["ai_generated"] is True


def test_review_cuda_code_missing_bounds():
    cuda = """
__global__ void vec_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}
"""
    review = review_cuda_code(cuda)
    categories = [f["category"] for f in review["findings"]]
    assert "bounds_guarding" in categories


def test_review_cuda_code_empty():
    review = review_cuda_code("")
    assert "findings" in review
    assert review["findings"] == []