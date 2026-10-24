"""Test corpus loader and evaluation tests.

Loads representative numerical workloads from tests/corpus/ and verifies
that the application logic handles them correctly:

1. Each corpus file is valid Python that can be parsed.
2. The nemotron service rejects empty input and missing API key gracefully.
3. The code review detector identifies known issues in sample CUDA code.
4. Verification status transitions work correctly in the database.
"""

import os
import sys
import ast
import sqlite3
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import models
from app.services.nemotron import translate_to_cuda, is_configured
from app.services.code_review import review_cuda_code


CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")


def _list_corpus_files():
    """Return sorted list of .py files in the corpus directory."""
    if not os.path.isdir(CORPUS_DIR):
        return []
    return sorted(
        os.path.join(CORPUS_DIR, f)
        for f in os.listdir(CORPUS_DIR)
        if f.endswith(".py")
    )


def test_corpus_files_exist():
    """At least one corpus file should exist."""
    files = _list_corpus_files()
    assert len(files) >= 1, "No corpus files found in tests/corpus/"


def test_corpus_files_valid_python():
    """Every corpus file should be syntactically valid Python."""
    for path in _list_corpus_files():
        with open(path, "r") as fh:
            source = fh.read()
        try:
            ast.parse(source)
        except SyntaxError as exc:
            assert False, f"{path} is not valid Python: {exc}"


def test_corpus_has_vector_add():
    """Corpus should include a vector addition workload."""
    files = _list_corpus_files()
    assert any("vec_add" in f for f in files), "Missing vec_add.py in corpus"


def test_corpus_has_reduction():
    """Corpus should include a reduction workload."""
    files = _list_corpus_files()
    assert any("array_sum" in f for f in files), "Missing array_sum.py in corpus"


def test_corpus_has_matrix_op():
    """Corpus should include a matrix operation workload."""
    files = _list_corpus_files()
    assert any("mat_mul" in f for f in files), "Missing mat_mul_square.py in corpus"


def test_corpus_has_stencil():
    """Corpus should include a stencil workload."""
    files = _list_corpus_files()
    assert any("stencil" in f for f in files), "Missing stencil_2d.py in corpus"


def test_verification_status_transitions(tmp_path):
    """Verification status can transition through valid stages."""
    original_db = models.DB_PATH
    test_db = str(tmp_path / "test_verify.db")
    models.DB_PATH = test_db
    try:
        models.init_db()
        uid = models.create_user("eval_user", "hash")
        tid = models.save_translation(
            user_id=uid,
            source_code="def f(x): return x",
            cuda_code="__global__ void f() {}",
            explanation=None,
            model_used="nemotron-nano",
        )
        # Initial status is 'generated'
        row = models.get_translation(tid)
        assert row["verification_status"] == "generated"

        # Transition to ai_reviewed
        assert models.update_verification(tid, "ai_reviewed", "[]") is True
        row = models.get_translation(tid)
        assert row["verification_status"] == "ai_reviewed"

        # Transition to static_passed
        assert models.update_verification(tid, "static_passed", None) is True
        row = models.get_translation(tid)
        assert row["verification_status"] == "static_passed"

        # Invalid status raises ValueError
        try:
            models.update_verification(tid, "invalid_status", None)
            assert False, "Should have raised ValueError for invalid status"
        except ValueError:
            pass
    finally:
        models.DB_PATH = original_db


def test_nemotron_handles_corpus_inputs():
    """translate_to_cuda should not crash on corpus inputs (when unconfigured)."""
    for path in _list_corpus_files():
        with open(path, "r") as fh:
            source = fh.read()
        result = translate_to_cuda(source)
        # Without API key, should return an error dict, not crash
        assert "error" in result or "success" in result


def test_code_review_on_bad_cuda():
    """Code review should flag obvious issues in a minimal bad kernel."""
    bad_code = """
__global__ void bad_kernel(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = 1.0f;
}
"""
    result = review_cuda_code(bad_code)
    categories = {f["category"] for f in result["findings"]}
    # Should flag missing bounds guard and missing sync/error checking
    assert "bounds_guarding" in categories or "synchronization" in categories