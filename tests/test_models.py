"""Tests for database models."""

import sys
import os
import tempfile

# Ensure the cudaify package root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import models as m
from app.routes.auth import hash_password, verify_password, PBKDF2_ITERATIONS


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


def test_password_hashing_uses_600k_iterations():
    """New passwords are hashed with PBKDF2-HMAC-SHA256 at 600k iterations."""
    pw_hash = hash_password("testpassword123")
    parts = pw_hash.split("$")
    assert parts[0] == "pbkdf2_sha256"
    assert int(parts[1]) == PBKDF2_ITERATIONS
    assert PBKDF2_ITERATIONS >= 600_000


def test_verify_password_valid_and_invalid():
    """verify_password returns (is_valid, needs_rehash) tuple."""
    pw_hash = hash_password("testpassword123")

    # Valid password
    is_valid, needs_rehash = verify_password("testpassword123", pw_hash)
    assert is_valid is True
    assert needs_rehash is False  # Fresh hash uses current iteration count

    # Invalid password
    is_valid, needs_rehash = verify_password("wrongpassword", pw_hash)
    assert is_valid is False
    assert needs_rehash is False


def test_verify_password_rehash_needed_for_old_hashes():
    """Old hashes with fewer iterations are verified but flagged for rehash."""
    # Simulate an old hash with 100,000 iterations
    old_hash = hash_password("oldpassword123", iterations=100_000)
    parts = old_hash.split("$")
    assert int(parts[1]) == 100_000  # Confirm old iteration count

    # Should verify correctly but flag for rehash
    is_valid, needs_rehash = verify_password("oldpassword123", old_hash)
    assert is_valid is True
    assert needs_rehash is True  # Needs rehash to current iteration count

    # Wrong password with old hash
    is_valid, needs_rehash = verify_password("wrongpassword", old_hash)
    assert is_valid is False
    assert needs_rehash is False