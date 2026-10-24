"""Tests for the 'do you need custom CUDA?' advisor."""
from app.services import advisor

SAXPY = "import numpy as np\n\ndef saxpy(a, x, y):\n    out = np.empty_like(x)\n    for i in range(len(x)):\n        out[i] = a * x[i] + y[i]\n    return out\n"
VECTORIZED = "import numpy as np\n\ndef f(x, y):\n    return 2.0 * x + np.sin(y)\n"
MATMUL = "import numpy as np\n\ndef f(a, b):\n    return np.dot(a, b)\n"
STENCIL = "import numpy as np\n\ndef f(x):\n    out = np.zeros_like(x)\n    for i in range(1, len(x) - 1):\n        out[i] = x[i - 1] + x[i] + x[i + 1]\n    return out\n"
REDUCE = "def total(x):\n    s = 0.0\n    for i in range(len(x)):\n        s += x[i]\n    return s\n"


def names(adv):
    return [o.name for o in adv.options]


def test_elementwise_loop():
    adv = advisor.analyze(SAXPY)
    assert {"loops", "indexed-store"} <= set(adv.patterns)
    assert any("Numba" in n for n in names(adv))
    assert adv.cuda_verdict == "consider-alternatives"


def test_array_oriented_prefers_cupy():
    adv = advisor.analyze(VECTORIZED)
    assert "array-oriented" in adv.patterns
    assert adv.cuda_verdict == "likely-unnecessary"
    assert "cp." in adv.options[0].snippet and "import cupy as cp" in adv.options[0].snippet


def test_library_routine_detected():
    adv = advisor.analyze(MATMUL)
    assert "blas" in adv.patterns
    assert adv.options[0].fit == "best" and "cuBLAS" in " ".join(adv.options[0].why)


def test_stencil_recommends_custom_cuda():
    adv = advisor.analyze(STENCIL)
    assert "stencil" in adv.patterns
    assert adv.cuda_verdict == "recommended"


def test_scalar_accumulation_is_reduction():
    adv = advisor.analyze(REDUCE)
    assert "reduction" in adv.patterns
    assert any("CuPy reduction" in n for n in names(adv))


def test_invalid_python_does_not_crash():
    adv = advisor.analyze("def (:")
    assert adv.patterns == []
    assert adv.options and adv.options[0].name == "Profile first"


def test_to_cupy_swaps_import_and_prefix():
    out = advisor.to_cupy("import numpy as np\nx = np.zeros(3)\n")
    assert "import cupy as cp" in out and "cp.zeros" in out and "np." not in out


def test_to_dict_is_json_serializable():
    import json
    json.dumps(advisor.analyze(SAXPY).to_dict())
