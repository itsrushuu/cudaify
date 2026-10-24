"""Tests for nemotron service (mocked)."""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services import nemotron


def test_model_for_translator_default(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "")
    model = nemotron.model_for("translator")
    assert model == nemotron.DEFAULT_MODEL


def test_model_for_role_env_override(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEMOTRON_REVIEWER_MODEL", "nvidia/custom-reviewer")
    model = nemotron.model_for("reviewer")
    assert model == "nvidia/custom-reviewer"


def test_is_configured_false(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    assert nemotron.is_configured() is False


def test_is_configured_true(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    assert nemotron.is_configured() is True


def test_translate_to_cuda_empty_source():
    result = nemotron.translate_to_cuda("")
    assert "error" in result


def test_translate_to_cuda_no_api_key(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    result = nemotron.translate_to_cuda("def f(x): return x")
    assert "error" in result


def test_chat_completion_mocked(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": "CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```\nHOST CODE:\n```cpp\nvoid launch() {}\n```\nEXPLANATION:\nSimple kernel."}}]
    })

    with patch("app.services.nemotron.requests.post", return_value=mock_resp):
        content = nemotron.chat_completion(
            "test-model",
            "system prompt",
            "user content",
        )
    assert "CUDA KERNEL" in content
    assert "__global__ void k()" in content


def test_chat_completion_401_not_retried(monkeypatch):
    """401 errors should raise immediately without retrying."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    call_count = {"n": 0}

    def failing_post(*args, **kwargs):
        call_count["n"] += 1
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = '{"error": {"message": "Unauthorized"}}'
        http_error = requests.exceptions.HTTPError("401 Unauthorized", response=mock_resp)
        raise http_error

    import requests
    with patch("app.services.nemotron.requests.post", side_effect=failing_post):
        try:
            nemotron.chat_completion(
                "test-model",
                "system prompt",
                "user content",
                max_retries=2,
            )
            assert False, "Should have raised NemotronError"
        except nemotron.NemotronError as exc:
            assert exc.status_code == 401
            assert exc.retryable is False
            assert call_count["n"] == 1  # No retries on 401


def test_chat_completion_429_retried(monkeypatch):
    """429 errors should be retried."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    call_count = {"n": 0}

    def flaky_post(*args, **kwargs):
        call_count["n"] += 1
        mock_resp = MagicMock()
        if call_count["n"] < 3:
            mock_resp.status_code = 429
            mock_resp.text = '{"error": {"message": "Rate limited"}}'
            http_error = requests.exceptions.HTTPError("429 Too Many Requests", response=mock_resp)
            raise http_error
        # Third call succeeds
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={
            "choices": [{"message": {"content": "CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```"}}]
        })
        return mock_resp

    import requests
    with patch("app.services.nemotron.requests.post", side_effect=flaky_post):
        with patch("app.services.nemotron.time.sleep"):  # Speed up test
            content = nemotron.chat_completion(
                "test-model",
                "system prompt",
                "user content",
                max_retries=2,
            )
    assert "CUDA KERNEL" in content
    assert call_count["n"] == 3  # Initial + 2 retries


def test_chat_completion_connection_error_retried(monkeypatch):
    """Connection errors should be retried."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    call_count = {"n": 0}

    def flaky_post(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise requests.exceptions.ConnectionError("Connection refused")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={
            "choices": [{"message": {"content": "CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```"}}]
        })
        return mock_resp

    import requests
    with patch("app.services.nemotron.requests.post", side_effect=flaky_post):
        with patch("app.services.nemotron.time.sleep"):
            content = nemotron.chat_completion(
                "test-model",
                "system prompt",
                "user content",
                max_retries=2,
            )
    assert "CUDA KERNEL" in content
    assert call_count["n"] == 2


def test_chat_completion_500_retried(monkeypatch):
    """500 errors should be retried."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    call_count = {"n": 0}

    def flaky_post(*args, **kwargs):
        call_count["n"] += 1
        mock_resp = MagicMock()
        if call_count["n"] < 2:
            mock_resp.status_code = 500
            mock_resp.text = '{"error": {"message": "Internal Server Error"}}'
            http_error = requests.exceptions.HTTPError("500 Internal Server Error", response=mock_resp)
            raise http_error
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={
            "choices": [{"message": {"content": "CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```"}}]
        })
        return mock_resp

    import requests
    with patch("app.services.nemotron.requests.post", side_effect=flaky_post):
        with patch("app.services.nemotron.time.sleep"):
            content = nemotron.chat_completion(
                "test-model",
                "system prompt",
                "user content",
                max_retries=2,
            )
    assert "CUDA KERNEL" in content
    assert call_count["n"] == 2