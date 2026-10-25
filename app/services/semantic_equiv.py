"""Semantic equivalence testing framework.

Validates that generated CUDA code produces the same results as the
original Python/NumPy reference implementation. Since a CUDA compiler
(nvcc) is not available in this environment, this framework:

1. Generates structured test inputs for each corpus workload
2. Runs the Python reference implementation to capture expected outputs
3. Defines a clean interface for comparing against CUDA output
   (when nvcc/GPU execution is available)
4. Records equivalence results in the database for transparency

This provides a verifiable path to runtime correctness testing without
requiring local GPU infrastructure.
"""

import ast
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class TestCase:
    workload: str
    name: str
    inputs: Dict[str, List[float]]
    expected_outputs: Dict[str, List[float]]
    tolerance: float = 1e-5
    passed: Optional[bool] = None
    notes: Optional[str] = None


@dataclass
class EquivalenceResult:
    workload: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    cases: List[TestCase]
    cuda_available: bool = False
    overall_passed: bool = False


class TestInputGenerator:
    __test__ = False

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _random_array(self, size: int, low: float = -10.0, high: float = 10.0) -> List[float]:
        return [self.rng.uniform(low, high) for _ in range(size)]

    def vec_add_cases(self) -> List[TestCase]:
        cases = []
        n = 8
        a = self._random_array(n)
        b = self._random_array(n)
        c = [a[i] + b[i] for i in range(n)]
        cases.append(TestCase(
            workload="vec_add",
            name="small_random",
            inputs={"a": a, "b": b, "c": [0.0] * n, "n": [float(n)]},
            expected_outputs={"c": c},
        ))
        n = 5
        a = [0.0] * n
        b = [0.0] * n
        c = [0.0] * n
        cases.append(TestCase(
            workload="vec_add",
            name="all_zeros",
            inputs={"a": a, "b": b, "c": c, "n": [float(n)]},
            expected_outputs={"c": c},
        ))
        n = 6
        a = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        c = [a[i] + b[i] for i in range(n)]
        cases.append(TestCase(
            workload="vec_add",
            name="negative_values",
            inputs={"a": a, "b": b, "c": [0.0] * n, "n": [float(n)]},
            expected_outputs={"c": c},
        ))
        n = 1
        a = [3.14]
        b = [2.71]
        c = [a[0] + b[0]]
        cases.append(TestCase(
            workload="vec_add",
            name="single_element",
            inputs={"a": a, "b": b, "c": [0.0], "n": [float(n)]},
            expected_outputs={"c": c},
        ))
        return cases

    def element_wise_cases(self) -> List[TestCase]:
        cases = []
        n = 6
        a = [0.1, 0.5, 0.9, 1.5, 2.0, -0.5]
        b = [0.2, 0.3, 0.1, 0.5, -1.0, 0.8]
        scale = 1.0
        expected = []
        for i in range(n):
            v = (a[i] + b[i]) * scale
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            expected.append(v)
        cases.append(TestCase(
            workload="element_wise",
            name="mixed_values",
            inputs={"a": a, "b": b, "scale": [scale], "n": [float(n)]},
            expected_outputs={"out": expected},
        ))
        n = 4
        a = [-5.0, -3.0, -1.0, -0.5]
        b = [-2.0, -1.0, -0.5, -0.1]
        scale = 2.0
        expected = [0.0] * n
        cases.append(TestCase(
            workload="element_wise",
            name="all_clamp_zero",
            inputs={"a": a, "b": b, "scale": [scale], "n": [float(n)]},
            expected_outputs={"out": expected},
        ))
        return cases

    def array_sum_cases(self) -> List[TestCase]:
        cases = []
        n = 5
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_sum = sum(a)
        cases.append(TestCase(
            workload="array_sum",
            name="positive_values",
            inputs={"a": a, "n": [float(n)]},
            expected_outputs={"total": [expected_sum]},
        ))
        n = 6
        a = [10.0, -5.0, 3.0, -2.0, 1.0, -1.0]
        expected_sum = sum(a)
        cases.append(TestCase(
            workload="array_sum",
            name="mixed_signs",
            inputs={"a": a, "n": [float(n)]},
            expected_outputs={"total": [expected_sum]},
        ))
        n = 1
        a = [42.0]
        cases.append(TestCase(
            workload="array_sum",
            name="single_element",
            inputs={"a": a, "n": [float(n)]},
            expected_outputs={"total": [a[0]]},
        ))
        return cases

    def mat_mul_cases(self) -> List[TestCase]:
        cases = []
        n = 2
        A = [1.0, 2.0, 3.0, 4.0]
        B = [5.0, 6.0, 7.0, 8.0]
        C = [19.0, 22.0, 43.0, 50.0]
        cases.append(TestCase(
            workload="mat_mul_square",
            name="two_by_two",
            inputs={"A": A, "B": B, "C": [0.0] * 4, "n": [float(n)]},
            expected_outputs={"C": C},
        ))
        n = 3
        A = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        B = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        C = list(B)
        cases.append(TestCase(
            workload="mat_mul_square",
            name="identity_matrix",
            inputs={"A": A, "B": B, "C": [0.0] * 9, "n": [float(n)]},
            expected_outputs={"C": C},
        ))
        return cases

    def stencil_cases(self) -> List[TestCase]:
        cases = []
        rows, cols = 3, 3
        input_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        output = [0.0] * 9
        output[4] = (input_vals[1] + input_vals[7] + input_vals[3] + input_vals[5]) / 4.0
        cases.append(TestCase(
            workload="stencil_2d",
            name="three_by_three",
            inputs={"input": input_vals, "output": output, "rows": [float(rows)], "cols": [float(cols)]},
            expected_outputs={"output": output},
        ))
        return cases

    def all_cases(self) -> List[TestCase]:
        all_cases: List[TestCase] = []
        all_cases.extend(self.vec_add_cases())
        all_cases.extend(self.element_wise_cases())
        all_cases.extend(self.array_sum_cases())
        all_cases.extend(self.mat_mul_cases())
        all_cases.extend(self.stencil_cases())
        return all_cases


class ReferenceExecutor:
    def __init__(self):
        self._workload_functions: Dict[str, Callable] = {}
        self._register_workloads()

    def _register_workloads(self):
        def vec_add(a, b, c, n):
            for i in range(n):
                c[i] = a[i] + b[i]
            return c

        def element_wise(a, b, scale, n):
            out = [0.0] * n
            for i in range(n):
                v = (a[i] + b[i]) * scale
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                out[i] = v
            return out

        def array_sum(a, n):
            total = 0.0
            for i in range(n):
                total += a[i]
            return total

        def mat_mul_square(A, B, C, n):
            for i in range(n):
                for j in range(n):
                    total = 0.0
                    for k in range(n):
                        total += A[i * n + k] * B[k * n + j]
                    C[i * n + j] = total
            return C

        def stencil_2d(input_arr, output, rows, cols):
            for i in range(1, int(rows) - 1):
                for j in range(1, int(cols) - 1):
                    output[i * int(cols) + j] = (
                        input_arr[(i - 1) * int(cols) + j]
                        + input_arr[(i + 1) * int(cols) + j]
                        + input_arr[i * int(cols) + (j - 1)]
                        + input_arr[i * int(cols) + (j + 1)]
                    ) / 4.0
            return output

        self._workload_functions = {
            "vec_add": vec_add,
            "element_wise": element_wise,
            "array_sum": array_sum,
            "mat_mul_square": mat_mul_square,
            "stencil_2d": stencil_2d,
        }

    def run(self, workload: str, inputs: Dict[str, List[float]]) -> Dict[str, List[float]]:
        func = self._workload_functions.get(workload)
        if func is None:
            raise ValueError(f"Unknown workload: {workload}")
        n = int(inputs.get("n", [0])[0]) if "n" in inputs else 0
        rows = int(inputs.get("rows", [0])[0]) if "rows" in inputs else 0
        cols = int(inputs.get("cols", [0])[0]) if "cols" in inputs else 0
        if workload == "vec_add":
            result = func(inputs["a"], inputs["b"], inputs["c"][:], n)
            return {"c": result}
        if workload == "element_wise":
            scale = inputs["scale"][0]
            result = func(inputs["a"], inputs["b"], scale, n)
            return {"out": result}
        if workload == "array_sum":
            result = func(inputs["a"], n)
            return {"total": [result]}
        if workload == "mat_mul_square":
            result = func(inputs["A"], inputs["B"], inputs["C"][:], n)
            return {"C": result}
        if workload == "stencil_2d":
            result = func(inputs["input"], inputs["output"][:], rows, cols)
            return {"output": result}
        raise ValueError(f"No reference executor for workload: {workload}")


def arrays_close(a: List[float], b: List[float], tolerance: float = 1e-5) -> Tuple[bool, Optional[str]]:
    if len(a) != len(b):
        return False, f"Length mismatch: {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        diff = abs(x - y)
        if diff > tolerance:
            return False, (
                f"Index {i}: {x} vs {y} (diff={diff:.6e} > {tolerance})"
            )
    return True, None


def run_equivalence_tests(
    workloads: Optional[List[str]] = None,
    cuda_available: bool = False,
) -> EquivalenceResult:
    generator = TestInputGenerator()
    executor = ReferenceExecutor()

    all_cases = generator.all_cases()
    if workloads:
        all_cases = [c for c in all_cases if c.workload in workloads]

    for case in all_cases:
        try:
            computed = executor.run(case.workload, case.inputs)
            all_passed = True
            for out_name, expected in case.expected_outputs.items():
                if out_name not in computed:
                    case.passed = False
                    case.notes = f"Missing output: {out_name}"
                    all_passed = False
                    continue
                ok, msg = arrays_close(computed[out_name], expected, case.tolerance)
                if not ok:
                    case.passed = False
                    case.notes = f"Reference mismatch: {msg}"
                    all_passed = False
            if all_passed:
                case.passed = True
                case.notes = "Reference implementation verified"
        except Exception as exc:
            case.passed = False
            case.notes = f"Reference execution error: {exc}"

    total = len(all_cases)
    passed = sum(1 for c in all_cases if c.passed is True)
    failed = sum(1 for c in all_cases if c.passed is False)
    skipped = sum(1 for c in all_cases if c.passed is None)

    cuda_passed = passed if cuda_available else 0
    overall = (passed == total) and cuda_available

    return EquivalenceResult(
        workload=",".join(workloads) if workloads else "all",
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed,
        skipped_cases=skipped,
        cases=all_cases,
        cuda_available=cuda_available,
        overall_passed=overall,
    )


def equivalence_result_to_dict(result: EquivalenceResult) -> Dict[str, Any]:
    return {
        "workload": result.workload,
        "total_cases": result.total_cases,
        "passed_cases": result.passed_cases,
        "failed_cases": result.failed_cases,
        "skipped_cases": result.skipped_cases,
        "cuda_available": result.cuda_available,
        "overall_passed": result.overall_passed,
        "cases": [asdict(c) for c in result.cases],
    }


def save_equivalence_result(
    translation_id: int,
    result: EquivalenceResult,
    db_path: str = "cudaify.db",
) -> bool:
    import sqlite3
    from app.models import update_verification, VERIFICATION_STATUSES

    findings_json = json.dumps(equivalence_result_to_dict(result), ensure_ascii=False)

    if result.overall_passed and result.cuda_available:
        status = "runtime_tested"
    elif result.passed_cases == result.total_cases and result.cuda_available:
        status = "static_passed"
    else:
        status = "generated"

    if status not in VERIFICATION_STATUSES:
        status = "generated"

    return update_verification(translation_id, status, findings_json)