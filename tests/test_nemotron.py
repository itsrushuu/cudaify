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


def test_model_for_local_uses_local_model(monkeypatch):
    """In local mode, model_for() returns LOCAL_MODEL when set."""
    monkeypatch.setenv("AI_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_MODEL", "qwen2.5-coder-3b")
    model = nemotron.model_for("translator")
    assert model == "qwen2.5-coder-3b"


def test_model_for_local_falls_back_to_default(monkeypatch):
    """In local mode without LOCAL_MODEL, model_for() falls back to DEFAULT_MODEL."""
    monkeypatch.setenv("AI_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_MODEL", "")
    model = nemotron.model_for("translator")
    assert model == nemotron.DEFAULT_MODEL


def test_model_for_nebius_unchanged_in_local_mode(monkeypatch):
    """Nebius env vars are ignored when AI_PROVIDER=local."""
    monkeypatch.setenv("AI_PROVIDER", "local")
    monkeypatch.setenv("NEBIUS_MODEL", "nvidia/should-not-be-used")
    monkeypatch.setenv("NEMOTRON_TRANSLATOR_MODEL", "nvidia/should-not-be-used")
    monkeypatch.setenv("LOCAL_MODEL", "qwen2.5-coder-3b")
    model = nemotron.model_for("translator")
    assert model == "qwen2.5-coder-3b"


def test_is_configured_local_without_nebius_key(monkeypatch):
    """Local mode is configured even without NEBIUS_API_KEY when LOCAL_MODEL is set."""
    monkeypatch.setenv("AI_PROVIDER", "local")
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    monkeypatch.setenv("LOCAL_MODEL", "qwen2.5-coder-3b")
    assert nemotron.is_configured() is True


def test_is_configured_local_without_local_model(monkeypatch):
    """Local mode is NOT configured without LOCAL_MODEL."""
    monkeypatch.setenv("AI_PROVIDER", "local")
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    monkeypatch.setenv("LOCAL_MODEL", "")
    assert nemotron.is_configured() is False


def test_is_configured_nebius_requires_key(monkeypatch):
    """Nebius mode requires NEBIUS_API_KEY."""
    monkeypatch.setenv("AI_PROVIDER", "nebius")
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    assert nemotron.is_configured() is False


def test_is_configured_unknown_provider(monkeypatch):
    """Unknown provider is not configured (safe, no silent fallback)."""
    monkeypatch.setenv("AI_PROVIDER", "bogus")
    assert nemotron.is_configured() is False


def test_chat_completion_local_uses_local_base_url(monkeypatch):
    """Local mode sends requests to LOCAL_BASE_URL, not Nebius."""
    monkeypatch.setenv("AI_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:1235/v1")
    monkeypatch.setenv("LOCAL_MODEL", "qwen2.5-coder-3b")
    monkeypatch.setenv("NEBIUS_API_KEY", "")  # Must not be required

    captured = {}

    def capturing_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={
            "choices": [{"message": {"content": "CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```"}}]
        })
        return mock_resp

    import requests
    with patch("app.services.nemotron.requests.post", side_effect=capturing_post):
        content = nemotron.chat_completion(
            "qwen2.5-coder-3b",
            "system prompt",
            "user content",
        )
    assert "CUDA KERNEL" in content
    assert captured["url"] == "http://localhost:1235/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer lm-studio"
    assert captured["json"]["model"] == "qwen2.5-coder-3b"


def test_chat_completion_local_does_not_contact_nebius(monkeypatch):
    """Local mode never constructs a Nebius URL."""
    monkeypatch.setenv("AI_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:1235/v1")
    monkeypatch.setenv("LOCAL_MODEL", "qwen2.5-coder-3b")
    monkeypatch.setenv("NEBIUS_API_KEY", "")

    nebius_urls = []

    def tracking_post(url, headers=None, json=None, timeout=None):
        nebius_urls.append(url)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={
            "choices": [{"message": {"content": "CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```"}}]
        })
        return mock_resp

    import requests
    with patch("app.services.nemotron.requests.post", side_effect=tracking_post):
        nemotron.chat_completion(
            "qwen2.5-coder-3b",
            "system prompt",
            "user content",
        )
    assert not any("nebius" in u.lower() or "tokenfactory" in u.lower() for u in nebius_urls)


def test_chat_completion_default_provider_is_nebius(monkeypatch):
    """Unset AI_PROVIDER behaves exactly as before (Nebius)."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    assert nemotron._get_ai_provider() == "nebius"
    assert nemotron.is_configured() is True


def test_chat_completion_unknown_provider_raises(monkeypatch):
    """Unknown AI_PROVIDER produces a clear NemotronError."""
    monkeypatch.setenv("AI_PROVIDER", "bogus")
    try:
        nemotron.chat_completion(
            "test-model",
            "system prompt",
            "user content",
        )
        assert False, "Should have raised NemotronError"
    except nemotron.NemotronError as exc:
        assert exc.retryable is False
        assert "bogus" in str(exc)


def test_chat_completion_local_connection_error_sanitized(monkeypatch):
    """Local connection failures produce sanitized NemotronError, no secrets exposed."""
    monkeypatch.setenv("AI_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:1235/v1")
    monkeypatch.setenv("LOCAL_MODEL", "qwen2.5-coder-3b")
    monkeypatch.setenv("NEBIUS_API_KEY", "")

    def failing_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Connection refused")

    import requests
    with patch("app.services.nemotron.requests.post", side_effect=failing_post):
        try:
            nemotron.chat_completion(
                "qwen2.5-coder-3b",
                "system prompt",
                "user content",
                max_retries=1,
            )
            assert False, "Should have raised NemotronError"
        except nemotron.NemotronError as exc:
            assert exc.retryable is True
            # Error message must not contain the API key or authorization header
            assert "Bearer" not in str(exc)
            assert "lm-studio" not in str(exc)


# --- Phase 2E: Regression tests for validation ---


def test_validate_python_input_valid_function():
    """Valid Python function with array access should pass validation."""
    source = "def vec_add(a, b, c, n):\n    for i in range(n):\n        c[i] = a[i] + b[i]\n"
    result = nemotron.validate_python_input(source)
    assert result["supported"] is True
    assert result["syntax_valid"] is True
    assert "vec_add" in result["functions"]
    assert "a" in result["array_accesses"]
    assert "b" in result["array_accesses"]
    assert "c" in result["array_accesses"]


def test_validate_python_input_invalid_syntax():
    """Invalid Python syntax should be rejected."""
    source = "def vec_add(a, b, c, n):\n    for i in range(n):\n        c[i] = a[i] + b[i]\n    return c\n)  # extra paren"
    result = nemotron.validate_python_input(source)
    assert result["supported"] is False
    assert result["syntax_valid"] is False
    assert "invalid_python_syntax" in result["unsupported"]
    assert result["syntax_error"] is not None


def test_validate_python_input_unsupported_yield():
    """Generator with yield should be rejected."""
    source = "def gen(n):\n    for i in range(n):\n        yield i\n"
    result = nemotron.validate_python_input(source)
    assert result["supported"] is False
    assert "yield" in result["unsupported"]


def test_validate_python_input_unsupported_class():
    """Class definition should be rejected."""
    source = "class MyClass:\n    def method(self):\n        pass\n"
    result = nemotron.validate_python_input(source)
    assert result["supported"] is False
    assert "class_definition" in result["unsupported"]


def test_validate_python_input_unsupported_eval():
    """eval() call should be rejected."""
    source = "def f(x):\n    return eval(x)\n"
    result = nemotron.validate_python_input(source)
    assert result["supported"] is False
    assert "eval_exec" in result["unsupported"]


def test_validate_python_input_unsupported_try_except():
    """try/except should be rejected."""
    source = "def f(x):\n    try:\n        return x\n    except:\n        return 0\n"
    result = nemotron.validate_python_input(source)
    assert result["supported"] is False
    assert "try_except" in result["unsupported"]


def test_validate_python_input_no_function():
    """Code without function definition should be rejected."""
    source = "x = [1, 2, 3]\ny = [v * 2 for v in x]\n"
    result = nemotron.validate_python_input(source)
    assert result["supported"] is False
    assert "no_function_definition" in result["unsupported"]


def test_validate_python_input_warning_no_array_access():
    """Function without array access should produce warning but still pass."""
    source = "def f(x):\n    return x + 1\n"
    result = nemotron.validate_python_input(source)
    assert result["supported"] is True
    assert "no_array_access_detected" in result["warnings"]


def test_validate_python_input_decorator_rejected():
    """Decorator should be rejected."""
    source = "@decorator\ndef f(x):\n    return x\n"
    result = nemotron.validate_python_input(source)
    assert result["supported"] is False
    assert "decorator" in result["unsupported"]


def test_validate_cuda_output_complete():
    """Complete CUDA code should pass validation."""
    cuda = """
__global__ void vec_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
void host_launch(float* a, float* b, float* c, int n) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    vec_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaGetLastError();
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
"""
    result = nemotron.validate_cuda_output(cuda)
    assert result["valid"] is True
    assert result["has_kernel"] is True
    assert result["has_launch"] is True
    assert result["has_host"] is True
    assert result["has_memory_ops"] is True
    assert result["has_sync"] is True
    assert result["has_error_check"] is True
    assert result["has_bounds"] is True
    assert result["validation_level"] == "full"


def test_validate_cuda_output_empty():
    """Empty CUDA code should fail validation."""
    result = nemotron.validate_cuda_output("")
    assert result["valid"] is False
    assert result["validation_level"] == "none"
    assert "Empty CUDA code" in result["issues"]


def test_validate_cuda_output_missing_kernel():
    """CUDA code without __global__ kernel should fail."""
    cuda = "void host_func() { int x = 0; }"
    result = nemotron.validate_cuda_output(cuda)
    assert result["valid"] is False
    assert result["has_kernel"] is False
    assert "No __global__ kernel function found" in result["issues"]


def test_validate_cuda_output_missing_launch():
    """CUDA code without kernel launch should fail."""
    cuda = """
__global__ void kernel() { }
void host() { }
"""
    result = nemotron.validate_cuda_output(cuda)
    assert result["valid"] is False
    assert result["has_launch"] is False
    assert "No kernel launch configuration (<<< >>>) found" in result["issues"]


def test_validate_cuda_output_superficial_match_not_enough():
    """Code containing 'cuda' but no kernel should not pass as valid."""
    cuda = "void foo() { cudaVariable = 0; }"
    result = nemotron.validate_cuda_output(cuda)
    assert result["valid"] is False
    assert result["has_kernel"] is False


def test_translate_to_cuda_validation_included(monkeypatch):
    """translate_to_cuda should include validation and confidence in output."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "choices": [{"message": {
            "content": "CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```\nHOST CODE:\n```cpp\nvoid launch() {}\n```\nEXPLANATION:\nSimple.\n"
        }}]
    })

    import requests
    with patch("app.services.nemotron.requests.post", return_value=mock_resp):
        result = nemotron.translate_to_cuda("def f(x): return x")
    assert result["success"] is True
    assert "validation" in result
    assert "confidence" in result
    assert "input_validation" in result
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_translate_to_cuda_low_confidence_incomplete(monkeypatch):
    """Incomplete CUDA output should produce low confidence."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": "Some text without proper sections"}}]
    })

    import requests
    with patch("app.services.nemotron.requests.post", return_value=mock_resp):
        result = nemotron.translate_to_cuda("def f(x): return x")
    assert result["success"] is True
    assert result["confidence"] < 0.5
    assert result["validation"]["valid"] is False