"""Tests for database models."""

import sys
import os
import tempfile

# Ensure the cudaify package root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import models as m


def _use_temp_db(monkeypatch, tmp_path):
    """Helper to point models.DB_PATH at a temp file and re-init."""
    test_db = tmp_path / "test_cudaify.db"
    m.DB_PATH = str(test_db)
    m.init_db()
    return test_db


def test_init_db_creates_tables(tmp_path, monkeypatch):
    _use_temp_db(monkeypatch, tmp_path)
    assert m.get_conn() is not None


def test_create_and_get_user(tmp_path, monkeypatch):
    _use_temp_db(monkeypatch, tmp_path)
    uid = m.create_user("alice", "hash123")
    assert uid == 1
    user = m.get_user("alice")
    assert user is not None
    assert user["username"] == "alice"
    assert user["password_hash"] == "hash123"
    assert m.get_user("bob") is None


def test_save_and_get_translations(tmp_path, monkeypatch):
    _use_temp_db(monkeypatch, tmp_path)
    uid = m.create_user("bob", "hash456")
    tid = m.save_translation(
        uid,
        "def foo(x): return x + 1",
        "__global__ void cuda_foo(int* x) { *x += 1; }",
        "Simple element-wise increment",
        "nemotron-nano",
    )
    assert tid == 1
    rows = m.get_translations(uid)
    assert len(rows) == 1
    assert rows[0]["user_id"] == uid
    assert rows[0]["model_used"] == "nemotron-nano"
    assert "cuda" in rows[0]["cuda_code"].lower()


def test_verification_statuses_are_valid():
    assert "generated" in m.VERIFICATION_STATUSES
    assert "ai_reviewed" in m.VERIFICATION_STATUSES
    assert "static_passed" in m.VERIFICATION_STATUSES
    assert "compiled" in m.VERIFICATION_STATUSES
    assert "runtime_tested" in m.VERIFICATION_STATUSES
    assert "benchmarked" in m.VERIFICATION_STATUSES


def test_translation_to_dict(tmp_path, monkeypatch):
    _use_temp_db(monkeypatch, tmp_path)
    uid = m.create_user("carol", "hash789")
    tid = m.save_translation(
        uid,
        "src",
        "cuda",
        "explanation",
        "test-model",
        verification_status="ai_reviewed",
        pipeline_iterations=3,
        pipeline_converged=1,
    )
    row = m.get_translation(tid)
    d = m.translation_to_dict(row)
    assert d["id"] == tid
    assert d["user_id"] == uid
    assert d["verification_status"] == "ai_reviewed"
    assert d["pipeline_iterations"] == 3
    assert d["pipeline_converged"] is True