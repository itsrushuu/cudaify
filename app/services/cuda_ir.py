"""CUDA Intermediate Representation (IR).

A structured, language-neutral representation of a numerical computation,
extracted from Python/NumPy source code. The IR sits between the Python
AST and the generated CUDA C++, enabling:

1. Independent verification of CUDA translations against the IR
2. Computation pattern classification (element-wise, reduction, stencil, matrix)
3. Auto-tuning recommendations (block size, grid size, memory layout)
4. Static correctness checking without compilation

This is a technical differentiator: most code-to-CUDA tools generate CUDA
directly from Python with no intermediate representation. CUDAify's IR
provides a verifiable, analyzable form of the computation.
"""

import ast
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Set


@dataclass
class ArrayRef:
    """Reference to an array variable in the computation."""
    name: str
    dtype: str = "float"
    ndim: int = 1
    shape: Optional[List[str]] = None
    access_pattern: str = "sequential"
    is_input: bool = True
    is_output: bool = False
    modified_in_place: bool = False


@dataclass
class ScalarRef:
    """Reference to a scalar variable."""
    name: str
    dtype: str = "float"
    is_input: bool = True
    is_loop_bound: bool = False


@dataclass
class LoopNest:
    """Description of loop structure in the computation."""
    depth: int = 1
    loop_vars: List[str] = field(default_factory=list)
    loop_bounds: List[str] = field(default_factory=list)
    is_perfectly_nested: bool = True
    has_dependencies: bool = False


@dataclass
class ComputationPattern:
    """Classification of the computation pattern."""
    pattern_type: str
    confidence: float = 0.0
    reduction_axis: Optional[int] = None
    stencil_radius: Optional[int] = None
    tileable: bool = False
    vectorizable: bool = False


@dataclass
class CudaIR:
    """Complete intermediate representation of a computation."""

    source_hash: str = ""
    function_name: str = ""

    inputs: List[ArrayRef] = field(default_factory=list)
    outputs: List[ArrayRef] = field(default_factory=list)
    scalars: List[ScalarRef] = field(default_factory=list)

    loops: Optional[LoopNest] = None

    pattern: Optional[ComputationPattern] = None

    total_elements: Optional[str] = None
    parallel_dimensions: int = 1
    suggested_block_size: int = 256
    suggested_grid_size: Optional[str] = None

    ir_generated: bool = False
    ir_cuda_match: Optional[bool] = None
    ir_cuda_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _infer_dtype(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int):
            return "int"
        if isinstance(node.value, float):
            return "float"
        if isinstance(node.value, complex):
            return "complex"
    return "float"


def _extract_array_accesses(node: ast.AST) -> List[str]:
    accesses: List[str] = []

    class SubscriptVisitor(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript):
            if isinstance(node.value, ast.Name):
                accesses.append(node.value.id)
            self.generic_visit(node)

    SubscriptVisitor().visit(node)
    return accesses


def _detect_stencil(node: ast.AST) -> bool:
    class StencilVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_BinOp(self, node: ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                left_is_name = isinstance(node.left, ast.Name)
                right_is_one = (
                    isinstance(node.right, ast.Constant)
                    and node.right.value == 1
                )
                right_is_name = isinstance(node.right, ast.Name)
                left_is_one = (
                    isinstance(node.left, ast.Constant)
                    and node.left.value == 1
                )
                if (left_is_name and right_is_one) or (right_is_name and left_is_one):
                    self.found = True
            self.generic_visit(node)

    visitor = StencilVisitor()
    visitor.visit(node)
    return visitor.found


def _is_reduction_pattern(node: ast.AST, target: str) -> bool:
    class ReductionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_AugAssign(self, node: ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == target:
                self.found = True
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == target for t in node.targets):
                if _references_name(node.value, target):
                    self.found = True
            self.generic_visit(node)

    visitor = ReductionVisitor()
    visitor.visit(node)
    return visitor.found


def _references_name(node: ast.AST, name: str) -> bool:
    class NameVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Name(self, node: ast.Name):
            if node.id == name:
                self.found = True

    NameVisitor().visit(node)
    return NameVisitor().found


def _extract_loop_depth(node: ast.AST) -> int:
    def _depth(n: ast.AST, current: int) -> int:
        max_d = current
        if isinstance(n, ast.For):
            current += 1
            max_d = max(max_d, current)
        for child in ast.iter_child_nodes(n):
            max_d = max(max_d, _depth(child, current))
        return max_d

    return _depth(node, 0)


def _collect_assigned_names(node: ast.AST) -> Set[str]:
    names: Set[str] = set()

    class AssignVisitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
            self.generic_visit(node)

    AssignVisitor().visit(node)
    return names


def build_ir(source_code: str) -> CudaIR:
    """Build a CUDA Intermediate Representation from Python source code."""
    ir = CudaIR()
    ir.ir_generated = True

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        ir.ir_cuda_notes = "Failed to parse Python source"
        return ir

    func_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            break

    if func_name:
        ir.function_name = func_name
        body = tree.body
    else:
        body = tree.body

    array_accesses = _extract_array_accesses(tree)
    unique_arrays = sorted(set(array_accesses))

    for i, arr_name in enumerate(unique_arrays):
        arr = ArrayRef(name=arr_name, dtype="float", ndim=1)
        if i == 0:
            arr.is_output = True
            arr.is_input = False
            ir.outputs.append(arr)
        else:
            arr.is_input = True
            ir.inputs.append(arr)

    assigned_names = _collect_assigned_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                assigned_names.add(node.target.id)
            if isinstance(node.iter, ast.Call):
                if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                    for arg in node.iter.args:
                        if isinstance(arg, ast.Name):
                            assigned_names.add(arg.id)

    array_names = {a.name for a in ir.inputs} | {a.name for a in ir.outputs}
    for name in sorted(assigned_names):
        if name not in array_names:
            ir.scalars.append(ScalarRef(name=name, dtype="float"))

    depth = _extract_loop_depth(tree)
    ir.loops = LoopNest(
        depth=max(depth, 1),
        loop_vars=[s.name for s in ir.scalars if s.is_loop_bound],
        is_perfectly_nested=depth <= 2,
    )

    pattern = ComputationPattern(pattern_type="other", confidence=0.3)

    for scalar in ir.scalars:
        if _is_reduction_pattern(tree, scalar.name):
            pattern.pattern_type = "reduction"
            pattern.confidence = 0.8
            pattern.reduction_axis = 0
            break

    if pattern.pattern_type == "other" and len(ir.inputs) >= 2 and ir.outputs:
        pattern.pattern_type = "element_wise"
        pattern.confidence = 0.7
        pattern.vectorizable = True
        pattern.tileable = True

    if _detect_stencil(tree) and pattern.pattern_type == "other":
        pattern.pattern_type = "stencil"
        pattern.confidence = 0.75
        pattern.stencil_radius = 1
        pattern.tileable = True

    if depth >= 3:
        pattern.pattern_type = "matrix_mul"
        pattern.confidence = 0.6
        pattern.tileable = True

    ir.pattern = pattern

    if pattern.pattern_type == "element_wise":
        ir.parallel_dimensions = 1
        ir.suggested_block_size = 256
    elif pattern.pattern_type == "reduction":
        ir.parallel_dimensions = 1
        ir.suggested_block_size = 256
    elif pattern.pattern_type == "stencil":
        ir.parallel_dimensions = 2
        ir.suggested_block_size = 16
    elif pattern.pattern_type == "matrix_mul":
        ir.parallel_dimensions = 2
        ir.suggested_block_size = 16

    if ir.outputs:
        ir.total_elements = "n"

    if ir.total_elements:
        ir.suggested_grid_size = "(total_elements + block_size - 1) / block_size"

    ir.ir_cuda_match = None
    return ir


def compare_ir_with_cuda(ir: CudaIR, cuda_code: str) -> Dict[str, Any]:
    """Compare a CUDA implementation against the IR for consistency."""
    matches: List[str] = []
    mismatches: List[str] = []
    warnings: List[str] = []
    score = 1.0

    ir_array_names = {a.name for a in ir.inputs} | {a.name for a in ir.outputs}
    cuda_lower = cuda_code.lower()

    for name in ir_array_names:
        if name.lower() in cuda_lower:
            matches.append(f"Array '{name}' referenced in CUDA code")
        else:
            mismatches.append(f"Array '{name}' from IR not found in CUDA code")
            score -= 0.1

    if ir.pattern and ir.pattern.pattern_type == "element_wise":
        if "if (" in cuda_lower and ("<" in cuda_lower or "<=" in cuda_lower):
            matches.append("Bounds guarding detected in CUDA kernel")
        else:
            warnings.append("Element-wise pattern but no obvious bounds guard in CUDA")
            score -= 0.15

    if "__shared__" in cuda_code and "__syncthreads()" not in cuda_code:
        warnings.append("Shared memory used without __syncthreads() barrier")
        score -= 0.2

    if "<<<" in cuda_code:
        matches.append("Kernel launch configuration present")
    else:
        mismatches.append("No kernel launch configuration (<<< >>>) found")
        score -= 0.2

    if "cudaDeviceSynchronize" in cuda_code:
        matches.append("cudaDeviceSynchronize present after kernel launch")
    else:
        warnings.append("No cudaDeviceSynchronize after kernel launch")
        score -= 0.1

    score = max(0.0, min(1.0, score))

    return {
        "matches": matches,
        "mismatches": mismatches,
        "warnings": warnings,
        "score": score,
    }


def ir_summary(ir: CudaIR) -> str:
    parts = [f"Function: {ir.function_name or 'unknown'}"]
    parts.append(f"Pattern: {ir.pattern.pattern_type if ir.pattern else 'unknown'} "
                 f"(confidence: {ir.pattern.confidence:.0%})")
    parts.append(f"Inputs: {', '.join(a.name for a in ir.inputs) or 'none'}")
    parts.append(f"Outputs: {', '.join(a.name for a in ir.outputs) or 'none'}")
    parts.append(f"Loop depth: {ir.loops.depth if ir.loops else 0}")
    parts.append(f"Parallel dims: {ir.parallel_dimensions}")
    parts.append(f"Suggested block: {ir.suggested_block_size}")
    return " | ".join(parts)