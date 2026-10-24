"""Tests for semantic equivalence framework."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.semantic_equiv import (
    TestInputGenerator,
    ReferenceExecutor,
    run_equivalence_tests,
    arrays_close,
    equivalence_result_to_dict,
)


def test_arrays_close_match():
    ok, msg = arrays_close([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert ok is True
    assert msg is None


def test_arrays_close_mismatch():
    ok, msg = arrays_close([1.0, 2.0], [1.0, 3.0])
    assert ok is False
    assert msg is not None


def test_arrays_close_length_mismatch():
    ok, msg = arrays_close([1.0], [1.0, 2.0])
    assert ok is False


def test_test_input_generator_all_cases():
    gen = TestInputGenerator()
    cases = gen.all_cases()
    assert len(cases) > 0
    workloads = {c.workload for c in cases}
    assert "vec_add" in workloads
    assert "element_wise" in workloads
    assert "array_sum" in workloads
    assert "mat_mul_square" in workloads
    assert "stencil_2d" in workloads


def test_reference_executor_vec_add():
    exec_ = ReferenceExecutor()
    result = exec_.run("vec_add", {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [0.0, 0.0], "n": [2.0]})
    assert result["c"] == [4.0, 6.0]


def test_reference_executor_array_sum():
    exec_ = ReferenceExecutor()
    result = exec_.run("array_sum", {"a": [1.0, 2.0, 3.0], "n": [3.0]})
    assert result["total"] == [6.0]


def test_run_equivalence_tests():
    result = run_equivalence_tests(cuda_available=False)
    assert result.total_cases > 0
    assert result.passed_cases > 0
    assert result.cuda_available is False
    assert result.overall_passed is False


def test_equivalence_result_to_dict():
    result = run_equivalence_tests(cuda_available=False)
    d = equivalence_result_to_dict(result)
    assert "total_cases" in d
    assert "passed_cases" in d
    assert "cases" in d
    assert len(d["cases"]) == result.total_cases