"""Auto-tuning recommendations engine.

Analyzes the CUDA Intermediate Representation (IR) and generated CUDA code
to produce specific, actionable tuning recommendations.

All recommendations are heuristic-based and clearly labeled as suggestions
that should be validated through benchmarking on target hardware.
"""

from typing import List, Dict, Any
from app.services.cuda_ir import CudaIR, ComputationPattern


def _recommended_block_size(pattern: ComputationPattern) -> int:
    if pattern.pattern_type == "stencil":
        return 16
    if pattern.pattern_type == "matrix_mul":
        return 16
    if pattern.pattern_type == "reduction":
        return 256
    if pattern.pattern_type == "element_wise":
        return 256
    return 256


def _recommended_grid_size(ir: CudaIR, block_size: int) -> str:
    if ir.total_elements:
        return f"({ir.total_elements} + {block_size} - 1) / {block_size}"
    if ir.pattern and ir.pattern.pattern_type in ("stencil", "matrix_mul"):
        return f"ceil(rows / {block_size}), ceil(cols / {block_size})"
    return f"({ir.total_elements or 'N'} + {block_size} - 1) / {block_size}"


def _memory_recommendations(ir: CudaIR, cuda_code: str) -> List[str]:
    recommendations: List[str] = []
    pattern_type = ir.pattern.pattern_type if ir.pattern else "unknown"

    if pattern_type == "element_wise":
        recommendations.append(
            "Ensure contiguous memory access: threads should access consecutive "
            "elements (a[tid], not a[tid * stride]) for coalesced loads/stores."
        )

    if pattern_type in ("stencil", "matrix_mul"):
        recommendations.append(
            "Consider shared memory tiling: load data tiles into __shared__ "
            "memory to reduce global memory accesses and enable data reuse."
        )
        if "__shared__" not in cuda_code:
            recommendations.append(
                "No __shared__ usage detected — consider adding shared memory "
                "for data reuse in stencil/matrix operations."
            )
        if "__shared__" in cuda_code and "__syncthreads()" not in cuda_code:
            recommendations.append(
                "__syncthreads() required after shared memory writes to ensure "
                "all threads see consistent data."
            )

    if pattern_type == "element_wise" and ir.pattern and ir.pattern.vectorizable:
        recommendations.append(
            "Consider vectorized loads/stores (float4, float2) to reduce "
            "memory transaction count and improve bandwidth utilization."
        )

    if pattern_type == "reduction":
        recommendations.append(
            "Use parallel reduction with shared memory: each thread block "
            "reduces a chunk, then atomically adds to global result."
        )
        recommendations.append(
            "Be aware of floating-point non-associativity: reduction order "
            "affects numerical results vs. sequential Python."
        )

    return recommendations


def _occupancy_estimate(block_size: int, uses_shared_memory: bool, uses_registers_heavy: bool = False) -> Dict[str, Any]:
    max_threads_per_block = 1024
    max_blocks_per_sm = 32
    threads_per_sm = max_threads_per_block * max_blocks_per_sm

    threads_per_sm_by_block = block_size * max_blocks_per_sm
    base_occupancy = min(1.0, threads_per_sm_by_block / threads_per_sm)

    sm_penalty = 1.0
    if uses_shared_memory:
        sm_penalty = 0.8

    reg_penalty = 1.0
    if uses_registers_heavy:
        reg_penalty = 0.7

    estimated_occupancy = base_occupancy * sm_penalty * reg_penalty

    return {
        "block_size": block_size,
        "theoretical_occupancy_pct": round(estimated_occupancy * 100, 1),
        "threads_per_sm": threads_per_sm_by_block,
        "max_threads_per_sm": threads_per_sm,
        "note": (
            "Theoretical estimate only. Actual occupancy depends on register "
            "usage, shared memory per block, and target GPU architecture."
        ),
    }


def generate_tuning_recommendations(ir: CudaIR, cuda_code: str) -> Dict[str, Any]:
    pattern = ir.pattern
    block_size = _recommended_block_size(pattern) if pattern else 256
    grid_size = _recommended_grid_size(ir, block_size)

    uses_shared = "__shared__" in cuda_code
    uses_sync = "__syncthreads()" in cuda_code
    uses_atomic = "atomic" in cuda_code.lower()

    occupancy = _occupancy_estimate(block_size, uses_shared)
    memory_recs = _memory_recommendations(ir, cuda_code)

    compute_recs: List[str] = []
    if pattern:
        if pattern.pattern_type == "element_wise":
            compute_recs.append(
                "Use grid-stride loop: for (int i = tid; i < n; i += gridDim.x * blockDim.x)"
            )
            compute_recs.append(
                "Consider loop unrolling (#pragma unroll 4) for small, fixed-size arrays"
            )
        if pattern.pattern_type == "reduction":
            compute_recs.append(
                "Use warp-level reduction (warpReduce) before shared memory reduction"
            )
        if pattern.pattern_type == "stencil":
            compute_recs.append(
                "Load halo/ghost cells into shared memory to handle boundary conditions"
            )
        if pattern.pattern_type == "matrix_mul":
            compute_recs.append(
                "Use shared memory tiling (TILE_WIDTH x TILE_WIDTH) for matrix multiply"
            )
            compute_recs.append(
                "Consider tensor cores (wmma) for mixed-precision matrix multiply"
            )

    warnings: List[str] = []
    if uses_shared and not uses_sync:
        warnings.append(
            "Shared memory used without __syncthreads() — potential race condition"
        )
    if uses_atomic:
        warnings.append(
            "Atomic operations detected — may serialize threads and reduce parallelism"
        )
    if pattern and pattern.pattern_type == "reduction" and not uses_sync:
        warnings.append(
            "Reduction pattern without synchronization — results may be non-deterministic"
        )

    priority: List[str] = []
    if memory_recs:
        priority.append("Memory: " + memory_recs[0])
    if compute_recs:
        priority.append("Compute: " + compute_recs[0])
    if warnings:
        priority.append("Fix: " + warnings[0])
    priority.append(f"Occupancy: target {occupancy['theoretical_occupancy_pct']}% with block_size={block_size}")

    return {
        "block_size": block_size,
        "grid_size": grid_size,
        "occupancy": occupancy,
        "memory": memory_recs,
        "compute": compute_recs,
        "warnings": warnings,
        "priority": priority,
        "pattern": pattern.pattern_type if pattern else "unknown",
        "pattern_confidence": pattern.confidence if pattern else 0.0,
    }