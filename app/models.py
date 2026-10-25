"""Database models and helpers using SQLite."""

import os
import sqlite3
import secrets
from datetime import datetime, UTC, timedelta
from typing import Optional, List, Dict, Any

DB_PATH = os.getenv("CUDAIFY_DB_PATH", "cudaify.db")

# Valid verification status values, in order of increasing confidence.
# NOTE: "generated", "ai_reviewed", and "static_passed" are AI-assisted or
# static-analysis statuses. They do NOT mean the code was compiled or
# executed. "compiled" requires successful nvcc compilation; "runtime_tested"
# requires execution against reference outputs; "benchmarked" requires
# performance measurement. The AI verifier's textual approval alone must
# never independently establish compilation success or runtime correctness.
VERIFICATION_STATUSES = (
    "generated",      # AI-generated, not independently verified
    "ai_reviewed",    # Passed AI code-review stage (AI-only, no compilation)
    "static_passed",  # Passed static analysis checks (not runtime execution)
    "compiled",       # Successfully compiled by nvcc (if available)
    "runtime_tested", # Passed runtime correctness tests
    "benchmarked",    # Performance benchmarked
)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent read/write safety.
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check whether a column exists in a table."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def init_db() -> None:
    """Initialize the database schema and migrate existing databases."""
    conn = get_conn()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                source_code TEXT NOT NULL,
                cuda_code TEXT NOT NULL,
                explanation TEXT,
                model_used TEXT,
                verification_status TEXT NOT NULL DEFAULT 'generated',
                verification_evidence TEXT,
                review_findings TEXT,
                sections_json TEXT,
                ir_json TEXT,
                ir_cuda_match_score REAL,
                pipeline_iterations INTEGER DEFAULT 0,
                agent_results_json TEXT,
                pipeline_converged INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )

        # Migration: add new columns to pre-existing tables.
        if not _column_exists(conn, "translations", "verification_status"):
            conn.execute(
                "ALTER TABLE translations ADD COLUMN verification_status TEXT NOT NULL DEFAULT 'generated'"
            )
        if not _column_exists(conn, "translations", "verification_evidence"):
            conn.execute(
                "ALTER TABLE translations ADD COLUMN verification_evidence TEXT"
            )
        if not _column_exists(conn, "translations", "review_findings"):
            conn.execute(
                "ALTER TABLE translations ADD COLUMN review_findings TEXT"
            )
        if not _column_exists(conn, "translations", "sections_json"):
            conn.execute(
                "ALTER TABLE translations ADD COLUMN sections_json TEXT"
            )
        if not _column_exists(conn, "translations", "pipeline_iterations"):
            conn.execute(
                "ALTER TABLE translations ADD COLUMN pipeline_iterations INTEGER DEFAULT 0"
            )
        if not _column_exists(conn, "translations", "agent_results_json"):
            conn.execute(
                "ALTER TABLE translations ADD COLUMN agent_results_json TEXT"
            )
        if not _column_exists(conn, "translations", "pipeline_converged"):
            conn.execute(
                "ALTER TABLE translations ADD COLUMN pipeline_converged INTEGER DEFAULT 0"
            )
        if not _column_exists(conn, "translations", "ir_json"):
            conn.execute(
                "ALTER TABLE translations ADD COLUMN ir_json TEXT"
            )
        if not _column_exists(conn, "translations", "ir_cuda_match_score"):
            conn.execute(
                "ALTER TABLE translations ADD COLUMN ir_cuda_match_score REAL"
            )

        conn.commit()
    finally:
        conn.close()


def create_user(username: str, password_hash: str) -> int:
    """Create a new user and return their id."""
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.now(UTC).isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_user(username: str) -> Optional[sqlite3.Row]:
    """Fetch a user by username."""
    conn = get_conn()
    try:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    """Fetch a user by id."""
    conn = get_conn()
    try:
        return conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    finally:
        conn.close()


def create_session(user_id: int, ttl_minutes: int = 60) -> str:
    """Create a session token for a user and return the token string."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(UTC) + timedelta(minutes=ttl_minutes)
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO sessions (user_id, token, expires_at, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                token,
                expires_at.isoformat(),
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
        return token
    finally:
        conn.close()


def get_session(token: str) -> Optional[sqlite3.Row]:
    """Fetch a session by token if it exists and has not expired."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM sessions WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            return None
        # Check expiry
        expires = row["expires_at"]
        if expires and datetime.now(UTC) > datetime.fromisoformat(expires):
            # Expired — delete it
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            return None
        return row
    finally:
        conn.close()


def delete_session(token: str) -> None:
    """Delete a session (logout)."""
    conn = get_conn()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


def cleanup_expired_sessions() -> None:
    """Remove all expired sessions from the database."""
    conn = get_conn()
    try:
        conn.execute(
            "DELETE FROM sessions WHERE expires_at < ?",
            (datetime.now(UTC).isoformat(),),
        )
        conn.commit()
    finally:
        conn.close()


def save_translation(
    user_id: int,
    source_code: str,
    cuda_code: str,
    explanation: Optional[str],
    model_used: str,
    verification_status: str = "generated",
    verification_evidence: Optional[str] = None,
    review_findings: Optional[str] = None,
    sections_json: Optional[str] = None,
    ir_json: Optional[str] = None,
    ir_cuda_match_score: Optional[float] = None,
    pipeline_iterations: int = 0,
    agent_results_json: Optional[str] = None,
    pipeline_converged: int = 0,
) -> int:
    """Save a translation record and return its id."""
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO translations
                (user_id, source_code, cuda_code, explanation, model_used,
                 verification_status, verification_evidence, review_findings, sections_json,
                 ir_json, ir_cuda_match_score,
                 pipeline_iterations, agent_results_json, pipeline_converged, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                source_code,
                cuda_code,
                explanation,
                model_used,
                verification_status,
                verification_evidence,
                review_findings,
                sections_json,
                ir_json,
                ir_cuda_match_score,
                pipeline_iterations,
                agent_results_json,
                pipeline_converged,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_pipeline(
    translation_id: int,
    pipeline_iterations: int,
    agent_results_json: str,
    pipeline_converged: int,
    verification_status: str,
    verification_evidence: Optional[str] = None,
    review_findings: Optional[str] = None,
    ir_json: Optional[str] = None,
    ir_cuda_match_score: Optional[float] = None,
) -> bool:
    """Update pipeline tracking fields on a translation record.

    Returns True if a row was updated, False if no matching row exists.
    """
    if verification_status not in VERIFICATION_STATUSES:
        raise ValueError(f"Invalid verification status: {verification_status}")

    conn = get_conn()
    try:
        cur = conn.execute(
            """
            UPDATE translations
            SET pipeline_iterations = ?,
                agent_results_json = ?,
                pipeline_converged = ?,
                verification_status = ?,
                verification_evidence = ?,
                review_findings = ?,
                ir_json = ?,
                ir_cuda_match_score = ?
            WHERE id = ?
            """,
            (
                pipeline_iterations,
                agent_results_json,
                pipeline_converged,
                verification_status,
                verification_evidence,
                review_findings,
                ir_json,
                ir_cuda_match_score,
                translation_id,
            ),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def update_verification(
    translation_id: int,
    verification_status: str,
    review_findings: Optional[str] = None,
    verification_evidence: Optional[str] = None,
) -> bool:
    """Update the verification status of an existing translation.

    Returns True if a row was updated, False if no matching row exists.
    """
    if verification_status not in VERIFICATION_STATUSES:
        raise ValueError(f"Invalid verification status: {verification_status}")

    conn = get_conn()
    try:
        cur = conn.execute(
            """
            UPDATE translations
            SET verification_status = ?, verification_evidence = ?, review_findings = ?
            WHERE id = ?
            """,
            (verification_status, verification_evidence, review_findings, translation_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_translations(user_id: int, limit: int = 20) -> List[sqlite3.Row]:
    """Fetch recent translations for a user."""
    conn = get_conn()
    try:
        return conn.execute(
            """
            SELECT * FROM translations
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    finally:
        conn.close()


def get_translation(translation_id: int) -> Optional[sqlite3.Row]:
    """Fetch a single translation by id."""
    conn = get_conn()
    try:
        return conn.execute(
            "SELECT * FROM translations WHERE id = ?", (translation_id,)
        ).fetchone()
    finally:
        conn.close()


def translation_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a translation row to a plain dict."""
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "source_code": row["source_code"],
        "cuda_code": row["cuda_code"],
        "explanation": row["explanation"],
        "model_used": row["model_used"],
        "verification_status": row["verification_status"],
        "verification_evidence": row["verification_evidence"],
        "review_findings": row["review_findings"],
        "sections_json": row["sections_json"],
        "ir_json": row["ir_json"],
        "ir_cuda_match_score": row["ir_cuda_match_score"],
        "pipeline_iterations": row["pipeline_iterations"],
        "agent_results_json": row["agent_results_json"],
        "pipeline_converged": bool(row["pipeline_converged"]),
        "created_at": row["created_at"],
    }