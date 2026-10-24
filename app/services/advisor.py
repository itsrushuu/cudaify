"""'Do you actually need custom CUDA?' advisor.

Deterministic (AST-based, no LLM call, no network). It inspects the user's Python
and says whether a simpler GPU route (CuPy, Numba) is likely enough, so CUDAify
recommends CUDA only where it earns its complexity.

Why: library routines (cuBLAS, cuFFT via CuPy) usually beat hand-written kernels,
CuPy is a drop-in NumPy replacement, and Numba covers custom per-element kernels.
"""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set

BLAS_CALLS = {"dot", "matmul", "einsum", "tensordot", "inner", "outer", "vdot"}
REDUCTION_CALLS = {"sum", "mean", "max", "min", "prod", "cumsum", "argmax", "argmin", "std", "var"}
SORT_CALLS = {"sort", "argsort", "unique", "searchsorted"}

NUMBA_SKELETON = '''from numba import cuda

@cuda.jit
def kernel(x, y, out):          # adapt the arguments to your function
    i = cuda.grid(1)            # one thread per element
    if i < out.size:            # bounds guard
        out[i] = ...            # the body of your Python loop, using x[i], y[i]

threads = 256
blocks = (out.size + threads - 1) // threads
kernel[blocks, threads](x_gpu, y_gpu, out_gpu)
cuda.synchronize()'''


@dataclass
class Option:
    name: str
    fit: str                      # "best" | "good" | "possible" | "unnecessary"
    why: List[str]
    snippet: str = ""
    note: str = ""


@dataclass
class Advice:
    patterns: List[str]
    cuda_verdict: str             # "recommended" | "consider-alternatives" | "likely-unnecessary"
    summary: str
    options: List[Option] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _call_name(node: ast.Call) -> str:
    """'np.linalg.solve' -> 'linalg.solve', 'np.dot' -> 'dot' (module prefix dropped)."""
    parts: List[str] = []
    f = node.func
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.reverse()
        return ".".join(parts) if parts else f.id
    return ""


def detect_patterns(source: str) -> Set[str]:
    """Return a set of pattern tags found in the source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    patterns: Set[str] = set()
    loop_vars: Set[str] = set()
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]

    for n in loops:
        if isinstance(n, ast.For):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    loop_vars.add(t.id)

    def depth(node: ast.AST, d: int = 0) -> int:
        best = d
        for child in ast.iter_child_nodes(node):
            nd = d + 1 if isinstance(child, (ast.For, ast.While)) else d
            best = max(best, depth(child, nd))
        return best

    if loops:
        patterns.add("loops")
        if depth(tree) >= 2:
            patterns.add("nested-loops")

    for loop in loops:
        for n in ast.walk(loop):
            if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
                patterns.add("indexed-store")
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                patterns.add("reduction")
            # neighbour access such as x[i - 1] / x[i + 1] -> stencil
            if isinstance(n, ast.Subscript):
                for sub in ast.walk(n.slice):
                    if (
                        isinstance(sub, ast.BinOp)
                        and isinstance(sub.op, (ast.Add, ast.Sub))
                        and any(isinstance(s, ast.Name) and s.id in loop_vars for s in ast.walk(sub))
                        and any(isinstance(s, ast.Constant) and isinstance(s.value, int) for s in ast.walk(sub))
                    ):
                        patterns.add("stencil")

    uses_numpy = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in ("np", "numpy"):
            uses_numpy = True
        if isinstance(n, ast.Call):
            name = _call_name(n)
            last = name.split(".")[-1]
            if last in BLAS_CALLS:
                patterns.add("blas")
            if name.startswith("linalg."):
                patterns.add("lapack")
            if name.startswith("fft."):
                patterns.add("fft")
            if last in REDUCTION_CALLS:
                patterns.add("library-reduction")
            if last in SORT_CALLS:
                patterns.add("sort")
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            patterns.add("blas")
    if uses_numpy:
        patterns.add("numpy")
    if uses_numpy and not loops:
        patterns.add("array-oriented")
    return patterns


def to_cupy(source: str) -> str:
    """Best-effort NumPy -> CuPy starting point (swap the import and the np. prefix)."""
    out = re.sub(r"^\s*import numpy as np\s*$", "import cupy as cp", source, flags=re.MULTILINE)
    out = re.sub(r"\bnp\.", "cp.", out)
    if "import cupy as cp" not in out:
        out = "import cupy as cp\n" + out
    return out.rstrip() + (
        "\n\n# Move data to the GPU with cp.asarray(x) and back with cp.asnumpy(result)."
    )


def analyze(source: str) -> Advice:
    """Analyze Python source and recommend the right GPU route."""
    patterns = detect_patterns(source)
    p = patterns
    options: List[Option] = []

    library = p & {"blas", "lapack", "fft"}
    if library:
        routines = []
        if "blas" in p:
            routines.append("cuBLAS (matrix products / dot)")
        if "lapack" in p:
            routines.append("cuSOLVER (linear algebra)")
        if "fft" in p:
            routines.append("cuFFT")
        options.append(Option(
            "CuPy library routines", "best",
            [f"Your code calls {', '.join(routines)}-backed operations.",
             "Tuned vendor libraries usually beat hand-written kernels for these."],
            to_cupy(source),
            "Starting point only: review dtype and device-transfer costs.",
        ))

    if "array-oriented" in p and not library:
        options.append(Option(
            "CuPy (drop-in NumPy)", "best",
            ["The code is already array-oriented (no explicit Python loops).",
             "CuPy mirrors NumPy, so a custom CUDA kernel is probably not needed."],
            to_cupy(source),
            "Write custom CUDA only if you need to fuse several operations into one pass.",
        ))

    if "stencil" in p:
        options.append(Option(
            "Custom CUDA kernel", "best",
            ["Neighbour indexing (e.g. x[i-1], x[i+1]) is a stencil.",
             "Shared-memory tiling with halo cells is where hand-written CUDA pays off."]))
        options.append(Option(
            "Numba @cuda.jit", "good",
            ["Lets you express the same grid/block kernel in Python."], NUMBA_SKELETON))
    elif "reduction" in p:
        options.append(Option(
            "CuPy reduction (cp.sum / cp.max ...)", "good",
            ["Accumulating loops are reductions; CuPy ships tuned reduction kernels."],
            to_cupy(source), "Rewrite the loop as a single array reduction first."))
        options.append(Option(
            "Custom CUDA kernel", "possible",
            ["Worth it only for fused or unusual reductions (shared-memory tree reduction)."]))
    elif "indexed-store" in p:
        options.append(Option(
            "Numba @cuda.jit", "good",
            ["Independent per-element loop body maps one-to-one to a GPU thread.",
             "Stays in Python, no C++ toolchain needed."], NUMBA_SKELETON))
        options.append(Option(
            "Vectorize with NumPy, then CuPy", "good",
            ["If the loop body is plain arithmetic, an array expression is simpler and fast."]))
        options.append(Option(
            "Custom CUDA kernel (CUDAify)", "good",
            ["Gives full control and a standalone .cu file."]))
    elif "loops" in p:
        options.append(Option(
            "Custom CUDA kernel", "possible",
            ["Loops without array stores may be sequential or scalar; check for parallelism first."]))

    if "sort" in p:
        options.append(Option(
            "CuPy sort / CUB", "good",
            ["Sorting and unique-style operations are library-provided on the GPU."],
            to_cupy(source)))

    if not options:
        options.append(Option(
            "Profile first", "possible",
            ["No clear data-parallel pattern detected.",
             "Measure on the CPU (or try Numba @njit) before moving to the GPU."]))

    fits = [o.fit for o in options]
    cuda_names = [o for o in options if o.name.startswith("Custom CUDA")]
    if cuda_names and cuda_names[0].fit == "best":
        verdict = "recommended"
        summary = "A custom CUDA kernel looks worthwhile for this pattern."
    elif "best" in fits:
        verdict = "likely-unnecessary"
        summary = "A simpler GPU route is likely enough. Custom CUDA is optional here."
    elif cuda_names:
        verdict = "consider-alternatives"
        summary = "CUDA can work, but compare it with the simpler options below first."
    else:
        verdict = "consider-alternatives"
        summary = "No strong data-parallel pattern found. Profile before porting."

    return Advice(sorted(patterns), verdict, summary, options)
