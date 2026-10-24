"""Authentication routes (signup/login/logout).

Uses PBKDF2-HMAC-SHA256 for password hashing (standard library, no
external dependencies) and server-issued session tokens instead of
user-controlled X-User-Id headers.
"""

import hashlib
import os
import sqlite3
from typing import Optional, Tuple
from fastapi import APIRouter, Form, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.models import (
    create_user,
    get_user,
    create_session,
    get_session,
    delete_session,
    cleanup_expired_sessions,
    get_user_by_id,
    get_conn,
)
from app.rate_limit import limiter

router = APIRouter()
security = HTTPBearer(auto_error=False)

# PBKDF2-HMAC-SHA256 iteration count. 100k is the OWASP 2021 minimum;
# 600k is the OWASP 2023 recommendation. Existing hashes with fewer
# iterations are still verified and rehashed on next successful login.
PBKDF2_ITERATIONS = 600_000


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 with a random salt."""
    salt = hashlib.sha256(os.urandom(16)).hexdigest()[:32]
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> tuple[bool, bool]:
    """Verify a password against a stored pbkdf2_sha256 hash.

    Returns (is_valid, needs_rehash). needs_rehash is True when the
    stored hash uses fewer than PBKDF2_ITERATIONS iterations, so the
    caller can rehash on next successful login.
    """
    try:
        parts = stored.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False, False
        _, iterations, salt, hash_hex = parts
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), int(iterations)
        )
        is_valid = dk.hex() == hash_hex
        needs_rehash = is_valid and int(iterations) < PBKDF2_ITERATIONS
        return is_valid, needs_rehash
    except (ValueError, IndexError):
        return False, False


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> sqlite3.Row:
    """Authenticate via Bearer token and return the user row."""
    cleanup_expired_sessions()

    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token.")

    session = get_session(credentials.credentials)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")

    user = get_user_by_id(session["user_id"])
    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


@router.post("/signup")
@limiter.limit("3/minute")
def signup(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Register a new user."""
    if not username or not username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty.")
    if not password or len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    if get_user(username.strip()):
        raise HTTPException(status_code=400, detail="User already exists.")

    create_user(username.strip(), hash_password(password))
    return JSONResponse({"ok": True, "message": "User created. Please log in."})


@router.post("/login")
@limiter.limit("5/minute")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Authenticate a user and create a session.

    Rotates session tokens: any existing sessions for the user are deleted
    before a new one is created, preventing session fixation and stale-token
    reuse.
    """
    user = get_user(username.strip())
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    is_valid, needs_rehash = verify_password(password, user["password_hash"])
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    # Rehash with current iteration count if the stored hash is outdated.
    if needs_rehash:
        conn = get_conn()
        try:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (hash_password(password), user["id"]),
            )
            conn.commit()
        finally:
            conn.close()

    # Rotate sessions: delete any existing sessions for this user first.
    conn = get_conn()
    try:
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user["id"],))
        conn.commit()
    finally:
        conn.close()

    token = create_session(user["id"], ttl_minutes=60)
    return JSONResponse(
        {
            "ok": True,
            "token": token,
            "user_id": user["id"],
            "username": user["username"],
            "expires_in_minutes": 60,
        }
    )


@router.post("/logout")
def logout(
    authorization: Optional[str] = Header(None),
    current_user: sqlite3.Row = Depends(get_current_user),
):
    """Delete the current session (logout)."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        delete_session(token)
    return JSONResponse({"ok": True, "message": "Logged out."})


@router.get("/me")
def get_me(current_user: sqlite3.Row = Depends(get_current_user)):
    """Return the current authenticated user's profile."""
    return {
        "user_id": current_user["id"],
        "username": current_user["username"],
    }