from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numba import cuda

try:  # pragma: no cover
    import Rhino  # type: ignore
    _HAVE_RHINO = True
except Exception:  # pragma: no cover
    Rhino = None  # type: ignore
    _HAVE_RHINO = False

from .params import MatrixParams, SkyParams
from .utils.cpu_trace import (
    bin_tregenza_cpu,
    count_upward_misses_cpu,
    reduce_first_hits,
    trace_cpu_bvh_combined,
    trace_cpu_bvh_firsthit,
    trace_cpu_bvh_hitmask,
    trace_cpu_combined,
    trace_cpu_firsthit,
    trace_cpu_hitmask,
)
from .utils.cuda_trace import (
    kernel_accumulate_hits,
    kernel_accumulate_hits_stats,
    kernel_bin_tregenza,
    kernel_build_rays,
    kernel_count_upward_misses,
    kernel_reduce_hits,
    kernel_trace_bvh_combined,
    kernel_trace_bvh_count_upward,
    kernel_trace_bvh_firsthit,
    kernel_trace_bvh_tregenza,
    kernel_trace_combined,
    kernel_trace_count_upward,
    kernel_trace_firsthit,
    kernel_trace_tregenza,
    kernel_zero_i32,
    kernel_zero_i64,
)
from .utils.helpers import enforce_reciprocity_and_rowsum as _enforce_reciprocity_and_rowsum
from .utils.prepared import PreparedEmitter, PreparedSolver
from .utils.ray_builder import build_rays

_LOG_PROC = None
_BVH_AUTO_THRESHOLD = 512


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        return max(minimum, int(raw))
    except Exception:
        return max(minimum, int(default))


_GPU_BATCH_TARGET_ACTIVE_RAYS = _env_int("RAYSTRACK_GPU_BATCH_TARGET_ACTIVE_RAYS", 524288)
_GPU_BATCH_STREAMS_MAX = _env_int("RAYSTRACK_GPU_BATCH_STREAMS_MAX", 32)
_GPU_BATCH_MEMORY_BUDGET_MB = _env_int("RAYSTRACK_GPU_BATCH_MEMORY_BUDGET_MB", 512)
_GPU_SMALL_EMITTER_RAY_CAP = _env_int("RAYSTRACK_GPU_SMALL_EMITTER_RAY_CAP", 1048576)


def _open_log_console() -> None:
    global _LOG_PROC
    if _LOG_PROC is not None:
        return
    try:
        helper = [
            sys.executable,
            "-u",
            "-c",
            "import sys; [sys.stdout.write(l) for l in iter(sys.stdin.readline, '')]",
        ]
        if os.name == "nt":
            _LOG_PROC = subprocess.Popen(
                ["cmd.exe", "/k", *helper],
                stdin=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                text=True,
            )
        else:
            term = os.environ.get("TERM_WINDOW", "xterm")
            _LOG_PROC = subprocess.Popen([term, "-hold", "-e", *helper], stdin=subprocess.PIPE, text=True)
    except Exception:
        _LOG_PROC = None


def _log(msg: str) -> None:
    if _LOG_PROC is None:
        _open_log_console()
    if _LOG_PROC and _LOG_PROC.stdin:
        try:
            _LOG_PROC.stdin.write(msg + "\n")
            _LOG_PROC.stdin.flush()
            return
        except Exception:
            pass
    print(msg)


def _rhino_log(msg: str) -> None:
    if not _HAVE_RHINO:
        return
    try:
        Rhino.RhinoApp.WriteLine(msg)  # type: ignore[union-attr]
    except Exception:
        pass


def _compute_cuda_launch(n_items: int, preferred_threads: Optional[int], min_blocks: int = 4) -> Tuple[int, int]:
    n = int(max(0, n_items))
    if n <= 0:
        return 1, 1
    device = cuda.get_current_device()
    warp = 32
    max_threads = int(device.MAX_THREADS_PER_BLOCK)
    threads = int(max(1, min(preferred_threads or 256, max_threads)))
    if threads >= warp:
        threads = ((threads + warp - 1) // warp) * warp
    if n < threads:
        if n >= warp:
            target = max(warp, (n + min_blocks - 1) // min_blocks)
            threads = min(threads, ((target + warp - 1) // warp) * warp)
        else:
            threads = min(threads, max(1, min(n, (n + min_blocks - 1) // min_blocks)))
    threads = max(1, min(threads, max_threads))
    return max(1, (n + threads - 1) // threads), threads


def _select_bvh(bvh: str | None, total_faces: int) -> bool:
    mode = (bvh or "auto").lower()
    if mode not in ("auto", "off", "builtin"):
        raise ValueError(f"bvh must be 'auto', 'off', or 'builtin' (got {bvh!r})")
    if mode == "builtin":
        return True
    if mode == "off":
        return False
    return total_faces >= _BVH_AUTO_THRESHOLD


def _resolve_device(device: str | None) -> bool:
    have_cuda = cuda.is_available()
    dev = (device or "auto").lower()
    if dev not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be 'auto', 'gpu', or 'cpu' (got {device!r})")
    if dev == "auto":
        return have_cuda
    if dev == "gpu":
        if not have_cuda:
            raise RuntimeError("device='gpu' requested but CUDA is not available")
        return True
    return False


def _ensure_prepared(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    prepared: PreparedSolver | None,
) -> PreparedSolver:
    if prepared is None:
        return PreparedSolver(meshes)
    if not isinstance(prepared, PreparedSolver):
        raise TypeError("prepared must be a PreparedSolver instance")
    return prepared


def _convergence_checkpoint(iters_done: int, *, min_iters: int, interval: int, max_iters: int, needs_variance: bool = False) -> bool:
    if iters_done < max(1, int(min_iters)):
        return False
    if needs_variance and iters_done <= 1:
        return False
    if iters_done >= int(max_iters):
        return True
    span = max(1, int(interval))
    if span <= 1:
        return True
    start = max(1, int(min_iters))
    return ((iters_done - start) % span) == 0


def _host_ray_buffers(n_rays: int) -> Tuple[np.ndarray, np.ndarray]:
    orig = np.empty((n_rays, 3), dtype=np.float32)
    return orig, np.empty_like(orig)


def _build_rays_host(emitter: PreparedEmitter, rays: int, orig, dire, cp_grid, cp_dims) -> None:
    build_rays(
        emitter.u_grid,
        emitter.v_grid,
        emitter.halton_tri,
        emitter.halton_u,
        emitter.halton_v,
        emitter.halton_r1,
        emitter.halton_r2,
        emitter.cdf,
        emitter.tri_a,
        emitter.tri_e1,
        emitter.tri_e2,
        emitter.tri_u,
        emitter.tri_v,
        emitter.tri_n,
        emitter.tri_origin_eps,
        rays,
        orig,
        dire,
        cp_grid,
        cp_dims,
    )


@dataclass
class _MatrixGpuWorkspace:
    stream: Any
    ray_cap: int
    n_surf: int
    track_stats: bool
    d_orig: Any
    d_dirs: Any
    d_hit_sid: Any
    d_hit_front: Any
    d_hf_iter: Any
    d_hb_iter: Any
    d_hf_total: Any
    d_hb_total: Any
    d_sum_f: Any
    d_sumsq_f: Any
    d_sum_b: Any
    d_sumsq_b: Any
    h_hf_total: np.ndarray
    h_hb_total: np.ndarray
    h_sum_f: Optional[np.ndarray]
    h_sumsq_f: Optional[np.ndarray]
    h_sum_b: Optional[np.ndarray]
    h_sumsq_b: Optional[np.ndarray]
    h_orig: Optional[np.ndarray]
    h_dirs: Optional[np.ndarray]
    surf_blocks: int
    surf_threads: int


@dataclass
class _MatrixGpuEmitterState:
    idx_emit: int
    name_e: str
    receivers: List[int]
    recv_idx: np.ndarray
    emit_sid: int
    min_sid: int
    emitter: PreparedEmitter
    emitter_dev: Any
    n_rays_once: int
    blocks: int
    threads: int
    total_rays: int = 0
    iters_done: int = 0
    pending_chunk: int = 0
    prev_f: Optional[np.ndarray] = None
    prev_b: Optional[np.ndarray] = None
    started_at: float = 0.0


def _matrix_stats_required(tol_mode: str, return_stats: bool) -> bool:
    return bool(return_stats or tol_mode == "stderr")


def _matrix_can_batch_emitters(*, cuda_async: bool, gpu_raygen: bool) -> bool:
    return bool(cuda_async and gpu_raygen and _GPU_BATCH_STREAMS_MAX > 1)


def _matrix_can_chunk_iterations(*, cuda_async: bool, gpu_raygen: bool) -> bool:
    return not (cuda_async and not gpu_raygen)


def _estimate_matrix_gpu_workspace_bytes(
    ray_cap: int,
    n_surf: int,
    *,
    track_stats: bool,
    gpu_raygen: bool,
    cuda_async: bool,
) -> int:
    f32 = np.dtype(np.float32).itemsize
    f64 = np.dtype(np.float64).itemsize
    i32 = np.dtype(np.int32).itemsize
    i64 = np.dtype(np.int64).itemsize
    u8 = np.dtype(np.uint8).itemsize

    ray_bytes = int(ray_cap) * ((2 * 3 * f32) + i32 + u8)
    surf_device_bytes = int(n_surf) * (4 * i64)
    surf_host_bytes = int(n_surf) * (2 * i64)
    if track_stats:
        surf_device_bytes += int(n_surf) * (4 * f64)
        surf_host_bytes += int(n_surf) * (4 * f64)

    if cuda_async and not gpu_raygen:
        ray_bytes += int(ray_cap) * (2 * 3 * f32)

    return ray_bytes + surf_device_bytes + surf_host_bytes


def _matrix_batch_slot_count(
    ray_cap: int,
    group_size: int,
    n_surf: int,
    *,
    track_stats: bool,
    gpu_raygen: bool,
    cuda_async: bool,
) -> int:
    if group_size <= 1:
        return 1

    ray_cap_i = max(1, int(ray_cap))
    desired = max(2, (_GPU_BATCH_TARGET_ACTIVE_RAYS + ray_cap_i - 1) // ray_cap_i)
    device = cuda.get_current_device()
    sm_count = int(getattr(device, "MULTIPROCESSOR_COUNT", _GPU_BATCH_STREAMS_MAX))
    if ray_cap_i < 65536:
        workload_cap = min(_GPU_BATCH_STREAMS_MAX, 4)
    elif ray_cap_i < 262144:
        workload_cap = min(_GPU_BATCH_STREAMS_MAX, 8)
    else:
        workload_cap = _GPU_BATCH_STREAMS_MAX
    hard_cap = min(int(group_size), max(2, sm_count), workload_cap)

    ws_bytes = max(
        1,
        _estimate_matrix_gpu_workspace_bytes(
            ray_cap_i,
            n_surf,
            track_stats=track_stats,
            gpu_raygen=gpu_raygen,
            cuda_async=cuda_async,
        ),
    )
    budget_bytes = _GPU_BATCH_MEMORY_BUDGET_MB * 1024 * 1024
    budget_cap = max(1, budget_bytes // ws_bytes)
    return max(1, min(hard_cap, desired, int(budget_cap)))


def _iterations_to_next_checkpoint(
    iters_done: int,
    *,
    min_iters: int,
    interval: int,
    max_iters: int,
    needs_variance: bool,
) -> int:
    remaining = int(max_iters) - int(iters_done)
    if remaining <= 0:
        return 0

    start = max(1, int(min_iters))
    if needs_variance:
        start = max(start, 2)

    if iters_done < start:
        return min(start - iters_done, remaining)

    span = max(1, int(interval))
    if span <= 1:
        return 1

    rem = (iters_done - start) % span
    return min(span if rem == 0 else span - rem, remaining)


def _create_matrix_gpu_workspace(
    ray_cap: int,
    n_surf: int,
    *,
    cuda_async: bool,
    gpu_raygen: bool,
    track_stats: bool,
) -> _MatrixGpuWorkspace:
    stream = cuda.stream() if cuda_async else None
    d_orig = cuda.device_array((ray_cap, 3), dtype=np.float32)
    d_dirs = cuda.device_array((ray_cap, 3), dtype=np.float32)
    d_hit_sid = cuda.device_array(ray_cap, dtype=np.int32)
    d_hit_front = cuda.device_array(ray_cap, dtype=np.uint8)
    d_hf_iter = cuda.device_array(n_surf, dtype=np.int64)
    d_hb_iter = cuda.device_array(n_surf, dtype=np.int64)
    d_hf_total = cuda.device_array(n_surf, dtype=np.int64)
    d_hb_total = cuda.device_array(n_surf, dtype=np.int64)
    d_sum_f = cuda.device_array(n_surf, dtype=np.float64) if track_stats else None
    d_sumsq_f = cuda.device_array(n_surf, dtype=np.float64) if track_stats else None
    d_sum_b = cuda.device_array(n_surf, dtype=np.float64) if track_stats else None
    d_sumsq_b = cuda.device_array(n_surf, dtype=np.float64) if track_stats else None

    if cuda_async:
        h_hf_total = cuda.pinned_array(n_surf, dtype=np.int64)
        h_hb_total = cuda.pinned_array(n_surf, dtype=np.int64)
        h_sum_f = cuda.pinned_array(n_surf, dtype=np.float64) if track_stats else None
        h_sumsq_f = cuda.pinned_array(n_surf, dtype=np.float64) if track_stats else None
        h_sum_b = cuda.pinned_array(n_surf, dtype=np.float64) if track_stats else None
        h_sumsq_b = cuda.pinned_array(n_surf, dtype=np.float64) if track_stats else None
        if gpu_raygen:
            h_orig = h_dirs = None
        else:
            h_orig = cuda.pinned_array((ray_cap, 3), dtype=np.float32)
            h_dirs = cuda.pinned_array((ray_cap, 3), dtype=np.float32)
    else:
        h_hf_total = np.empty(n_surf, dtype=np.int64)
        h_hb_total = np.empty(n_surf, dtype=np.int64)
        h_sum_f = np.empty(n_surf, dtype=np.float64) if track_stats else None
        h_sumsq_f = np.empty(n_surf, dtype=np.float64) if track_stats else None
        h_sum_b = np.empty(n_surf, dtype=np.float64) if track_stats else None
        h_sumsq_b = np.empty(n_surf, dtype=np.float64) if track_stats else None
        if gpu_raygen:
            h_orig = h_dirs = None
        else:
            h_orig, h_dirs = _host_ray_buffers(ray_cap)

    surf_blocks, surf_threads = _compute_cuda_launch(n_surf, None)
    return _MatrixGpuWorkspace(
        stream=stream,
        ray_cap=ray_cap,
        n_surf=n_surf,
        track_stats=track_stats,
        d_orig=d_orig,
        d_dirs=d_dirs,
        d_hit_sid=d_hit_sid,
        d_hit_front=d_hit_front,
        d_hf_iter=d_hf_iter,
        d_hb_iter=d_hb_iter,
        d_hf_total=d_hf_total,
        d_hb_total=d_hb_total,
        d_sum_f=d_sum_f,
        d_sumsq_f=d_sumsq_f,
        d_sum_b=d_sum_b,
        d_sumsq_b=d_sumsq_b,
        h_hf_total=h_hf_total,
        h_hb_total=h_hb_total,
        h_sum_f=h_sum_f,
        h_sumsq_f=h_sumsq_f,
        h_sum_b=h_sum_b,
        h_sumsq_b=h_sumsq_b,
        h_orig=h_orig,
        h_dirs=h_dirs,
        surf_blocks=surf_blocks,
        surf_threads=surf_threads,
    )


def _reset_matrix_gpu_workspace(ws: _MatrixGpuWorkspace) -> None:
    zero = kernel_zero_i64[ws.surf_blocks, ws.surf_threads, ws.stream] if ws.stream is not None else kernel_zero_i64[ws.surf_blocks, ws.surf_threads]
    zero(ws.d_hf_total)
    zero(ws.d_hb_total)
    if ws.track_stats:
        zero(ws.d_sum_f)
        zero(ws.d_sumsq_f)
        zero(ws.d_sum_b)
        zero(ws.d_sumsq_b)


def _enqueue_matrix_gpu_summary_copy(ws: _MatrixGpuWorkspace) -> None:
    if ws.stream is not None:
        ws.d_hf_total.copy_to_host(ws.h_hf_total, stream=ws.stream)
        ws.d_hb_total.copy_to_host(ws.h_hb_total, stream=ws.stream)
        if ws.track_stats:
            ws.d_sum_f.copy_to_host(ws.h_sum_f, stream=ws.stream)
            ws.d_sumsq_f.copy_to_host(ws.h_sumsq_f, stream=ws.stream)
            ws.d_sum_b.copy_to_host(ws.h_sum_b, stream=ws.stream)
            ws.d_sumsq_b.copy_to_host(ws.h_sumsq_b, stream=ws.stream)


def _read_matrix_gpu_summary(ws: _MatrixGpuWorkspace) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if ws.stream is not None:
        ws.stream.synchronize()
        hits_f_total = np.asarray(ws.h_hf_total)
        hits_b_total = np.asarray(ws.h_hb_total)
        if ws.track_stats:
            return (
                hits_f_total,
                hits_b_total,
                np.asarray(ws.h_sum_f),
                np.asarray(ws.h_sumsq_f),
                np.asarray(ws.h_sum_b),
                np.asarray(ws.h_sumsq_b),
            )
        return hits_f_total, hits_b_total, None, None, None, None

    cuda.synchronize()
    hits_f_total = ws.d_hf_total.copy_to_host()
    hits_b_total = ws.d_hb_total.copy_to_host()
    if ws.track_stats:
        return (
            hits_f_total,
            hits_b_total,
            ws.d_sum_f.copy_to_host(),
            ws.d_sumsq_f.copy_to_host(),
            ws.d_sum_b.copy_to_host(),
            ws.d_sumsq_b.copy_to_host(),
        )
    return hits_f_total, hits_b_total, None, None, None, None


def _summary_to_stderr(sum_arr: np.ndarray, sumsq_arr: np.ndarray, iters_done: int) -> np.ndarray:
    if iters_done <= 1:
        return np.full(sum_arr.shape, np.inf, dtype=np.float64)
    n = float(iters_done)
    mean = sum_arr / n
    m2 = np.maximum(sumsq_arr - n * mean * mean, 0.0)
    return np.sqrt(m2 / float(iters_done - 1) / n)


def _build_matrix_gpu_state(
    idx_emit: int,
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    emitters: List[PreparedEmitter],
    prepared_solver: PreparedSolver,
    *,
    samples: int,
    rays: int,
    flip_faces: bool,
    reciprocity: bool,
) -> Optional[_MatrixGpuEmitterState]:
    name_e, _, _ = meshes[idx_emit]
    receivers = [j for j in range(idx_emit + 1, len(meshes))] if reciprocity else [j for j in range(len(meshes)) if j != idx_emit]
    if not receivers:
        return None

    recv_idx = np.asarray(receivers, dtype=np.int32)
    emit_sid, min_sid = _matrix_skip(idx_emit, reciprocity)
    emitter = emitters[idx_emit]
    n_rays_once = emitter.n_cells * rays
    blocks, threads = _compute_cuda_launch(n_rays_once, None)
    emitter_dev = prepared_solver.get_device_emitter(idx_emit, samples=samples, rays=rays, flip_faces=flip_faces)
    return _MatrixGpuEmitterState(
        idx_emit=idx_emit,
        name_e=name_e,
        receivers=receivers,
        recv_idx=recv_idx,
        emit_sid=emit_sid,
        min_sid=min_sid,
        emitter=emitter,
        emitter_dev=emitter_dev,
        n_rays_once=n_rays_once,
        blocks=blocks,
        threads=threads,
        started_at=time.time(),
    )


def _enqueue_matrix_gpu_chunk(
    ws: _MatrixGpuWorkspace,
    state: _MatrixGpuEmitterState,
    scene,
    d_scene,
    *,
    rays: int,
    seed: int,
    gpu_raygen: bool,
    cuda_async: bool,
) -> None:
    d_orig = ws.d_orig[: state.n_rays_once]
    d_dirs = ws.d_dirs[: state.n_rays_once]
    d_hit_sid = ws.d_hit_sid[: state.n_rays_once]
    d_hit_front = ws.d_hit_front[: state.n_rays_once]
    stream = ws.stream
    zero = kernel_zero_i64[ws.surf_blocks, ws.surf_threads, stream] if stream is not None else kernel_zero_i64[ws.surf_blocks, ws.surf_threads]

    offset = 0
    while offset < state.pending_chunk:  # type: ignore[attr-defined]
        itr = state.iters_done + offset
        rng = np.random.default_rng(seed + state.idx_emit + itr)
        cp_grid = rng.random(2, dtype=np.float32)
        cp_dims = rng.random(5, dtype=np.float32)

        if gpu_raygen:
            launch = kernel_build_rays[state.blocks, state.threads, stream] if stream is not None else kernel_build_rays[state.blocks, state.threads]
            launch(
                state.emitter_dev.u_grid, state.emitter_dev.v_grid,
                state.emitter_dev.halton_tri, state.emitter_dev.halton_u, state.emitter_dev.halton_v,
                state.emitter_dev.halton_r1, state.emitter_dev.halton_r2, state.emitter_dev.cdf,
                state.emitter_dev.tri_a, state.emitter_dev.tri_e1, state.emitter_dev.tri_e2,
                state.emitter_dev.tri_u, state.emitter_dev.tri_v, state.emitter_dev.tri_n, state.emitter_dev.tri_origin_eps,
                rays, d_orig, d_dirs, cp_grid, cp_dims,
            )
        else:
            h_orig = ws.h_orig[: state.n_rays_once]
            h_dirs = ws.h_dirs[: state.n_rays_once]
            _build_rays_host(state.emitter, rays, h_orig, h_dirs, cp_grid, cp_dims)
            if stream is not None:
                d_orig.copy_to_device(h_orig, stream=stream)
                d_dirs.copy_to_device(h_dirs, stream=stream)
            else:
                d_orig.copy_to_device(h_orig)
                d_dirs.copy_to_device(h_dirs)

        zero(ws.d_hf_iter)
        zero(ws.d_hb_iter)

        if scene.use_bvh:
            trace = kernel_trace_bvh_firsthit[state.blocks, state.threads, stream] if stream is not None else kernel_trace_bvh_firsthit[state.blocks, state.threads]
            trace(
                d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.normals, d_scene.sid,
                d_scene.bb_min, d_scene.bb_max, d_scene.left, d_scene.right, d_scene.start, d_scene.count,
                state.emit_sid, state.min_sid, d_hit_sid, d_hit_front,
            )
        else:
            trace = kernel_trace_firsthit[state.blocks, state.threads, stream] if stream is not None else kernel_trace_firsthit[state.blocks, state.threads]
            trace(
                d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.normals, d_scene.sid,
                state.emit_sid, state.min_sid, d_hit_sid, d_hit_front,
            )

        reduce_kernel = kernel_reduce_hits[state.blocks, state.threads, stream] if stream is not None else kernel_reduce_hits[state.blocks, state.threads]
        reduce_kernel(d_hit_sid, d_hit_front, ws.d_hf_iter, ws.d_hb_iter, ws.n_surf)

        if ws.track_stats:
            acc = kernel_accumulate_hits_stats[ws.surf_blocks, ws.surf_threads, stream] if stream is not None else kernel_accumulate_hits_stats[ws.surf_blocks, ws.surf_threads]
            acc(
                ws.d_hf_iter,
                ws.d_hb_iter,
                ws.d_hf_total,
                ws.d_hb_total,
                ws.d_sum_f,
                ws.d_sumsq_f,
                ws.d_sum_b,
                ws.d_sumsq_b,
                1.0 / float(state.n_rays_once),
            )
        else:
            acc = kernel_accumulate_hits[ws.surf_blocks, ws.surf_threads, stream] if stream is not None else kernel_accumulate_hits[ws.surf_blocks, ws.surf_threads]
            acc(ws.d_hf_iter, ws.d_hb_iter, ws.d_hf_total, ws.d_hb_total)

        offset += 1


def _finalize_matrix_gpu_state(
    state: _MatrixGpuEmitterState,
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    result: Dict[str, Dict[str, float]],
    stats_result: Optional[Dict[str, Dict[str, float]]],
    *,
    hits_f_total: np.ndarray,
    hits_b_total: np.ndarray,
    sum_f: Optional[np.ndarray],
    sumsq_f: Optional[np.ndarray],
    sum_b: Optional[np.ndarray],
    sumsq_b: Optional[np.ndarray],
    reciprocity: bool,
    areas: Optional[List[float]],
    return_stats: bool,
    use_bvh: bool,
) -> None:
    if state.iters_done > 1 and sum_f is not None and sumsq_f is not None and sum_b is not None and sumsq_b is not None:
        se_f = _summary_to_stderr(sum_f, sumsq_f, state.iters_done)
        se_b = _summary_to_stderr(sum_b, sumsq_b, state.iters_done)
    else:
        se_f = np.full((len(meshes),), np.inf, dtype=np.float64)
        se_b = np.full((len(meshes),), np.inf, dtype=np.float64)

    row: Dict[str, float] = {}
    stats_row: Dict[str, float] = {}
    for j in state.receivers:
        name_r, _, _ = meshes[j]
        f = hits_f_total[j] / float(state.total_rays)
        b = hits_b_total[j] / float(state.total_rays)
        if f > 0.0:
            row[f"{name_r}_front"] = f
            if reciprocity and areas is not None and areas[j] > 0.0:
                result[name_r][f"{state.name_e}_front"] = f * (areas[state.idx_emit] / areas[j])
            if return_stats:
                stats_row[f"{name_r}_front"] = float(se_f[j])
        if b > 0.0:
            row[f"{name_r}_back"] = b
            if return_stats:
                stats_row[f"{name_r}_back"] = float(se_b[j])
    result[state.name_e].update(row)
    if return_stats and stats_result is not None:
        stats_result[state.name_e] = stats_row

    msg = (
        f"({state.idx_emit+1}/{len(meshes)}) [{state.name_e}] {state.iters_done} iter, "
        f"{state.total_rays:,} rays -> {time.time() - state.started_at:0.3f}s  "
        f"(BVH={'builtin' if use_bvh else 'off'}, device=gpu)"
    )
    _log(msg)
    _rhino_log(msg)


def _evaluate_matrix_gpu_state(
    state: _MatrixGpuEmitterState,
    *,
    hits_f_total: np.ndarray,
    hits_b_total: np.ndarray,
    sum_f: Optional[np.ndarray],
    sumsq_f: Optional[np.ndarray],
    sum_b: Optional[np.ndarray],
    sumsq_b: Optional[np.ndarray],
    tol: float,
    tol_mode: str,
    min_iters: int,
    convergence_interval: int,
    max_iters: int,
) -> bool:
    done = state.iters_done >= int(max_iters)
    check_matrix = _convergence_checkpoint(
        state.iters_done,
        min_iters=min_iters,
        interval=convergence_interval,
        max_iters=max_iters,
        needs_variance=(tol_mode == "stderr"),
    )

    if tol_mode == "delta":
        curr_f = hits_f_total.astype(np.float64) / float(max(1, state.total_rays))
        curr_b = hits_b_total.astype(np.float64) / float(max(1, state.total_rays))
        if check_matrix and state.prev_f is not None:
            if np.all(np.abs(curr_f - state.prev_f) < tol) and np.all(np.abs(curr_b - state.prev_b) < tol):
                done = True
        if check_matrix:
            state.prev_f = curr_f.copy()
            state.prev_b = curr_b.copy()
    elif tol_mode == "stderr":
        if check_matrix and sum_f is not None and sumsq_f is not None and sum_b is not None and sumsq_b is not None:
            se_f = _summary_to_stderr(sum_f, sumsq_f, state.iters_done)
            se_b = _summary_to_stderr(sum_b, sumsq_b, state.iters_done)
            if np.all(se_f[state.recv_idx] <= tol) and np.all(se_b[state.recv_idx] <= tol):
                done = True
    else:
        raise ValueError(f"Unknown tol_mode: {tol_mode}")

    return done


def _run_matrix_gpu_serial(
    indices: List[int],
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    emitters: List[PreparedEmitter],
    prepared_solver: PreparedSolver,
    scene,
    d_scene,
    result: Dict[str, Dict[str, float]],
    stats_result: Optional[Dict[str, Dict[str, float]]],
    ws: _MatrixGpuWorkspace,
    *,
    samples: int,
    rays: int,
    seed: int,
    gpu_raygen: bool,
    cuda_async: bool,
    flip_faces: bool,
    reciprocity: bool,
    tol: float,
    tol_mode: str,
    min_iters: int,
    convergence_interval: int,
    max_iters: int,
    return_stats: bool,
    areas: Optional[List[float]],
    use_bvh: bool,
) -> None:
    allow_chunking = _matrix_can_chunk_iterations(cuda_async=cuda_async, gpu_raygen=gpu_raygen)
    interval = convergence_interval if allow_chunking else 1
    needs_variance = tol_mode == "stderr"

    for idx_emit in indices:
        state = _build_matrix_gpu_state(
            idx_emit,
            meshes,
            emitters,
            prepared_solver,
            samples=samples,
            rays=rays,
            flip_faces=flip_faces,
            reciprocity=reciprocity,
        )
        if state is None:
            name_e, _, _ = meshes[idx_emit]
            msg = f"({idx_emit+1}/{len(meshes)}) [{name_e}] 0 iter, 0 rays -> 0.000s  (BVH={'builtin' if use_bvh else 'off'}, device=gpu)"
            _log(msg)
            _rhino_log(msg)
            continue

        _reset_matrix_gpu_workspace(ws)

        while True:
            state.pending_chunk = _iterations_to_next_checkpoint(
                state.iters_done,
                min_iters=min_iters,
                interval=interval,
                max_iters=max_iters,
                needs_variance=needs_variance,
            )
            if state.pending_chunk <= 0:
                break

            _enqueue_matrix_gpu_chunk(
                ws,
                state,
                scene,
                d_scene,
                rays=rays,
                seed=seed,
                gpu_raygen=gpu_raygen,
                cuda_async=cuda_async,
            )
            _enqueue_matrix_gpu_summary_copy(ws)

            state.iters_done += state.pending_chunk
            state.total_rays += state.pending_chunk * state.n_rays_once

            hits_f_total, hits_b_total, sum_f, sumsq_f, sum_b, sumsq_b = _read_matrix_gpu_summary(ws)
            if _evaluate_matrix_gpu_state(
                state,
                hits_f_total=hits_f_total,
                hits_b_total=hits_b_total,
                sum_f=sum_f,
                sumsq_f=sumsq_f,
                sum_b=sum_b,
                sumsq_b=sumsq_b,
                tol=tol,
                tol_mode=tol_mode,
                min_iters=min_iters,
                convergence_interval=interval,
                max_iters=max_iters,
            ):
                _finalize_matrix_gpu_state(
                    state,
                    meshes,
                    result,
                    stats_result,
                    hits_f_total=hits_f_total,
                    hits_b_total=hits_b_total,
                    sum_f=sum_f,
                    sumsq_f=sumsq_f,
                    sum_b=sum_b,
                    sumsq_b=sumsq_b,
                    reciprocity=reciprocity,
                    areas=areas,
                    return_stats=return_stats,
                    use_bvh=use_bvh,
                )
                break


def _run_matrix_gpu_batched_group(
    indices: List[int],
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    emitters: List[PreparedEmitter],
    prepared_solver: PreparedSolver,
    scene,
    d_scene,
    result: Dict[str, Dict[str, float]],
    stats_result: Optional[Dict[str, Dict[str, float]]],
    workspaces: List[_MatrixGpuWorkspace],
    *,
    samples: int,
    rays: int,
    seed: int,
    gpu_raygen: bool,
    cuda_async: bool,
    flip_faces: bool,
    reciprocity: bool,
    tol: float,
    tol_mode: str,
    min_iters: int,
    convergence_interval: int,
    max_iters: int,
    return_stats: bool,
    areas: Optional[List[float]],
    use_bvh: bool,
) -> None:
    queue = list(indices)
    active: List[Optional[_MatrixGpuEmitterState]] = [None] * len(workspaces)
    needs_variance = tol_mode == "stderr"

    while queue or any(state is not None for state in active):
        for slot, ws in enumerate(workspaces):
            if active[slot] is not None:
                continue
            while queue:
                idx_emit = queue.pop(0)
                state = _build_matrix_gpu_state(
                    idx_emit,
                    meshes,
                    emitters,
                    prepared_solver,
                    samples=samples,
                    rays=rays,
                    flip_faces=flip_faces,
                    reciprocity=reciprocity,
                )
                if state is None:
                    name_e, _, _ = meshes[idx_emit]
                    msg = f"({idx_emit+1}/{len(meshes)}) [{name_e}] 0 iter, 0 rays -> 0.000s  (BVH={'builtin' if use_bvh else 'off'}, device=gpu)"
                    _log(msg)
                    _rhino_log(msg)
                    continue
                _reset_matrix_gpu_workspace(ws)
                active[slot] = state
                break

        if not any(state is not None for state in active):
            break

        for slot, state in enumerate(active):
            if state is None:
                continue
            state.pending_chunk = _iterations_to_next_checkpoint(
                state.iters_done,
                min_iters=min_iters,
                interval=convergence_interval,
                max_iters=max_iters,
                needs_variance=needs_variance,
            )
            if state.pending_chunk <= 0:
                state.pending_chunk = 1
            _enqueue_matrix_gpu_chunk(
                workspaces[slot],
                state,
                scene,
                d_scene,
                rays=rays,
                seed=seed,
                gpu_raygen=gpu_raygen,
                cuda_async=cuda_async,
            )
            _enqueue_matrix_gpu_summary_copy(workspaces[slot])

        for slot, state in enumerate(active):
            if state is None:
                continue
            state.iters_done += state.pending_chunk
            state.total_rays += state.pending_chunk * state.n_rays_once

            hits_f_total, hits_b_total, sum_f, sumsq_f, sum_b, sumsq_b = _read_matrix_gpu_summary(workspaces[slot])
            if _evaluate_matrix_gpu_state(
                state,
                hits_f_total=hits_f_total,
                hits_b_total=hits_b_total,
                sum_f=sum_f,
                sumsq_f=sumsq_f,
                sum_b=sum_b,
                sumsq_b=sumsq_b,
                tol=tol,
                tol_mode=tol_mode,
                min_iters=min_iters,
                convergence_interval=convergence_interval,
                max_iters=max_iters,
            ):
                _finalize_matrix_gpu_state(
                    state,
                    meshes,
                    result,
                    stats_result,
                    hits_f_total=hits_f_total,
                    hits_b_total=hits_b_total,
                    sum_f=sum_f,
                    sumsq_f=sumsq_f,
                    sum_b=sum_b,
                    sumsq_b=sumsq_b,
                    reciprocity=reciprocity,
                    areas=areas,
                    return_stats=return_stats,
                    use_bvh=use_bvh,
                )
                active[slot] = None


def _run_view_factor_matrix_gpu(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    emitters: List[PreparedEmitter],
    prepared_solver: PreparedSolver,
    scene,
    d_scene,
    result: Dict[str, Dict[str, float]],
    stats_result: Optional[Dict[str, Dict[str, float]]],
    *,
    samples: int,
    rays: int,
    seed: int,
    gpu_raygen: bool,
    cuda_async: bool,
    flip_faces: bool,
    reciprocity: bool,
    tol: float,
    tol_mode: str,
    min_iters: int,
    convergence_interval: int,
    max_iters: int,
    return_stats: bool,
    areas: Optional[List[float]],
    use_bvh: bool,
) -> None:
    n_surf = len(meshes)
    ray_counts = [emitter.n_cells * rays for emitter in emitters]
    max_rays = max(ray_counts) if ray_counts else 1
    track_stats = _matrix_stats_required(tol_mode, return_stats)
    serial_ws = _create_matrix_gpu_workspace(
        max_rays,
        n_surf,
        cuda_async=cuda_async,
        gpu_raygen=gpu_raygen,
        track_stats=track_stats,
    )

    batch_enabled = _matrix_can_batch_emitters(cuda_async=cuda_async, gpu_raygen=gpu_raygen)
    small_emitter_cap = max(1, min(max_rays, _GPU_SMALL_EMITTER_RAY_CAP))
    batch_workspaces: Optional[List[_MatrixGpuWorkspace]] = None

    idx_emit = 0
    while idx_emit < len(meshes):
        if batch_enabled and ray_counts[idx_emit] <= small_emitter_cap:
            group: List[int] = []
            while idx_emit < len(meshes) and ray_counts[idx_emit] <= small_emitter_cap:
                group.append(idx_emit)
                idx_emit += 1

            if len(group) > 1:
                group_ray_cap = max(ray_counts[g] for g in group)
                batch_slots = _matrix_batch_slot_count(
                    group_ray_cap,
                    len(group),
                    n_surf,
                    track_stats=track_stats,
                    gpu_raygen=gpu_raygen,
                    cuda_async=cuda_async,
                )
                if batch_slots > 1:
                    if batch_workspaces is None or len(batch_workspaces) < batch_slots or batch_workspaces[0].ray_cap < group_ray_cap:
                        batch_workspaces = [
                            _create_matrix_gpu_workspace(
                                group_ray_cap,
                                n_surf,
                                cuda_async=True,
                                gpu_raygen=True,
                                track_stats=track_stats,
                            )
                            for _ in range(batch_slots)
                        ]
                    active_workspaces = batch_workspaces[:batch_slots]
                    _run_matrix_gpu_batched_group(
                        group,
                        meshes,
                        emitters,
                        prepared_solver,
                        scene,
                        d_scene,
                        result,
                        stats_result,
                        active_workspaces,
                        samples=samples,
                        rays=rays,
                        seed=seed,
                        gpu_raygen=gpu_raygen,
                        cuda_async=cuda_async,
                        flip_faces=flip_faces,
                        reciprocity=reciprocity,
                        tol=tol,
                        tol_mode=tol_mode,
                        min_iters=min_iters,
                        convergence_interval=convergence_interval,
                        max_iters=max_iters,
                        return_stats=return_stats,
                        areas=areas,
                        use_bvh=use_bvh,
                    )
                    continue

            _run_matrix_gpu_serial(
                group,
                meshes,
                emitters,
                prepared_solver,
                scene,
                d_scene,
                result,
                stats_result,
                serial_ws,
                samples=samples,
                rays=rays,
                seed=seed,
                gpu_raygen=gpu_raygen,
                cuda_async=cuda_async,
                flip_faces=flip_faces,
                reciprocity=reciprocity,
                tol=tol,
                tol_mode=tol_mode,
                min_iters=min_iters,
                convergence_interval=convergence_interval,
                max_iters=max_iters,
                return_stats=return_stats,
                areas=areas,
                use_bvh=use_bvh,
            )
            continue

        _run_matrix_gpu_serial(
            [idx_emit],
            meshes,
            emitters,
            prepared_solver,
            scene,
            d_scene,
            result,
            stats_result,
            serial_ws,
            samples=samples,
            rays=rays,
            seed=seed,
            gpu_raygen=gpu_raygen,
            cuda_async=cuda_async,
            flip_faces=flip_faces,
            reciprocity=reciprocity,
            tol=tol,
            tol_mode=tol_mode,
            min_iters=min_iters,
            convergence_interval=convergence_interval,
            max_iters=max_iters,
            return_stats=return_stats,
            areas=areas,
            use_bvh=use_bvh,
        )
        idx_emit += 1

def _matrix_skip(idx_emit: int, reciprocity: bool) -> Tuple[int, int]:
    return (idx_emit, idx_emit + 1) if reciprocity else (idx_emit, 0)


def outside_workflow_shareable(matrix_params: MatrixParams, sky_params: SkyParams) -> bool:
    """Return True when outside-workflow can reuse one traced ray set.

    The shared outside-workflow path requires the matrix and sky solves to use
    identical ray-generation and execution settings so that a single set of
    rays is valid for both outputs. The following fields must match:

    - ``samples``
    - ``rays``
    - ``seed``
    - ``bvh``
    - ``device``
    - ``cuda_async``
    - ``gpu_raygen``

    The matrix solve must also keep ``flip_faces=False`` because the sky solve
    assumes outward emission.
    """
    shared_fields = ("samples", "rays", "seed", "bvh", "device", "cuda_async", "gpu_raygen")
    if bool(matrix_params.flip_faces):
        return False
    return all(getattr(matrix_params, key) == getattr(sky_params, key) for key in shared_fields)


def view_factor_matrix_and_sky(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    *,
    matrix_params: MatrixParams,
    sky_params: SkyParams,
    prepared: PreparedSolver | None = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Compute scene view factors and sky VF from a shared set of rays.

    This routine is intended for the outside-workflow case where the scene
    matrix and the sky visibility use the same sampling setup. Instead of
    tracing once for the matrix and then tracing a second time for sky, it:

    1. generates one ray set per emitter/iteration
    2. traces each ray once against the scene
    3. accumulates matrix hits from the first visible surface
    4. classifies rays that miss all geometry into merged or discrete sky bins

    The matrix and sky convergence checks remain independent. If one side
    converges earlier, tracing continues only to satisfy the other side, while
    still using the same per-iteration ray samples.

    For compatible parameter sets, this is intended to be numerically
    equivalent to calling :func:`view_factor_matrix` and
    :func:`view_factor_to_tregenza_sky` separately with the same shared
    sampling configuration, but without doubling the ray tracing work.

    When ``prepared`` is provided, the solver reuses cached prepared geometry,
    BVHs, ray tables and CUDA uploads across repeated calls on the same scene.
    """
    if not isinstance(matrix_params, MatrixParams):
        raise TypeError("matrix_params must be a MatrixParams instance")
    if not isinstance(sky_params, SkyParams):
        raise TypeError("sky_params must be a SkyParams instance")
    if not outside_workflow_shareable(matrix_params, sky_params):
        raise ValueError("matrix_params and sky_params are not compatible for shared tracing")

    mp = matrix_params.as_dict()
    sp = sky_params.as_dict()
    samples = mp["samples"]
    rays = mp["rays"]
    seed = mp["seed"]
    device = mp["device"]
    cuda_async = mp["cuda_async"]
    gpu_raygen = mp["gpu_raygen"]
    reciprocity = mp["reciprocity"]
    sky_discrete = sp["discrete"]
    matrix_interval = max(1, int(mp["convergence_interval"]))
    sky_interval = max(1, int(sp["convergence_interval"]))

    use_gpu = _resolve_device(device)
    prepared_solver = _ensure_prepared(meshes, prepared)
    use_bvh = _select_bvh(mp["bvh"], prepared_solver.total_faces)
    emitters = prepared_solver.get_emitters(samples=samples, rays=rays, flip_faces=False)
    scene = prepared_solver.get_scene(use_bvh=use_bvh)
    areas = [emitter.total_area for emitter in emitters] if reciprocity else None

    vf_scene: Dict[str, Dict[str, float]] = {name: {} for name, _, _ in meshes}
    if sky_discrete:
        sky_keys = [f"Sky_Patch_{i}" for i in range(1, 146)]
        sky_vf: Dict[str, Dict[str, float]] = {name: {key: 0.0 for key in sky_keys} for name, _, _ in meshes}
    else:
        sky_vf = {name: {"Sky": 0.0} for name, _, _ in meshes}

    d_scene = prepared_solver.get_device_scene(use_bvh=use_bvh) if use_gpu else None

    n_surf = len(meshes)
    for idx_emit, (name_e, _, _) in enumerate(meshes):
        t0 = time.time()
        receivers = [j for j in range(idx_emit + 1, len(meshes))] if reciprocity else [j for j in range(len(meshes)) if j != idx_emit]
        recv_idx = np.asarray(receivers, dtype=np.int32)
        emit_sid, matrix_min_sid = _matrix_skip(idx_emit, reciprocity)
        emitter = emitters[idx_emit]
        n_rays_once = emitter.n_cells * rays

        matrix_enabled = len(receivers) > 0
        sky_enabled = True
        matrix_done = not matrix_enabled
        sky_done = False

        hits_f = np.zeros(n_surf, np.int64)
        hits_b = np.zeros(n_surf, np.int64)
        mean_f = np.zeros(n_surf, np.float64)
        mean_b = np.zeros(n_surf, np.float64)
        M2_f = np.zeros(n_surf, np.float64)
        M2_b = np.zeros(n_surf, np.float64)
        prev_f = prev_b = None
        matrix_total_rays = 0
        matrix_iters_done = 0

        counts_total = np.zeros(145, np.int64) if sky_discrete else None
        mean_bins = np.zeros(145, np.float64) if sky_discrete else None
        M2_bins = np.zeros(145, np.float64) if sky_discrete else None
        upward_total = 0
        mean_sky = 0.0
        M2_sky = 0.0
        prev_frac = None
        sky_total_rays = 0
        sky_iters_done = 0

        trace_iters = 0
        total_traced_rays = 0
        max_trace_iters = max(int(mp["max_iters"]), int(sp["max_iters"]))

        if use_gpu:
            emitter_dev = prepared_solver.get_device_emitter(idx_emit, samples=samples, rays=rays, flip_faces=False)
            d_orig = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_dirs = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_hit_sid = cuda.device_array(n_rays_once, dtype=np.int32)
            d_hit_front = cuda.device_array(n_rays_once, dtype=np.uint8)
            d_any_hit = cuda.device_array(n_rays_once, dtype=np.uint8)
            d_hf = cuda.device_array(n_surf, dtype=np.int64)
            d_hb = cuda.device_array(n_surf, dtype=np.int64)
            d_counts = cuda.device_array(145, dtype=np.int32) if sky_discrete else None
            d_upward = cuda.device_array(1, dtype=np.int32) if not sky_discrete else None
            stream = cuda.stream() if cuda_async else None
            if cuda_async:
                h_hf = cuda.pinned_array(n_surf, dtype=np.int64)
                h_hb = cuda.pinned_array(n_surf, dtype=np.int64)
                h_counts = cuda.pinned_array(145, dtype=np.int32) if sky_discrete else None
                h_upward = cuda.pinned_array(1, dtype=np.int32) if not sky_discrete else None
                if not gpu_raygen:
                    h_orig = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
                    h_dirs = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
            else:
                h_hf = h_hb = h_counts = h_upward = None
                if not gpu_raygen:
                    h_orig, h_dirs = _host_ray_buffers(n_rays_once)
            blocks, threads = _compute_cuda_launch(n_rays_once, None)
            surf_blocks, surf_threads = _compute_cuda_launch(n_surf, threads)
            sky_blocks, sky_threads = _compute_cuda_launch(145 if sky_discrete else 1, threads)
        else:
            orig, dire = _host_ray_buffers(n_rays_once)
            hit_sid_iter = np.empty(n_rays_once, dtype=np.int32)
            front_iter = np.empty(n_rays_once, dtype=np.uint8)
            any_hit_iter = np.empty(n_rays_once, dtype=np.uint8)
            hits_f_iter = np.empty(n_surf, dtype=np.int64)
            hits_b_iter = np.empty(n_surf, dtype=np.int64)
            hitmask_iter = np.empty(n_rays_once, dtype=np.uint8)
            counts_iter = np.empty(145, dtype=np.int64) if sky_discrete else None

        for itr in range(max_trace_iters):
            if matrix_done and sky_done:
                break

            rng = np.random.default_rng(seed + idx_emit + itr)
            cp_grid = rng.random(2, dtype=np.float32)
            cp_dims = rng.random(5, dtype=np.float32)

            if use_gpu:
                if gpu_raygen:
                    launch = kernel_build_rays[blocks, threads, stream] if cuda_async else kernel_build_rays[blocks, threads]
                    launch(
                        emitter_dev.u_grid, emitter_dev.v_grid,
                        emitter_dev.halton_tri, emitter_dev.halton_u, emitter_dev.halton_v,
                        emitter_dev.halton_r1, emitter_dev.halton_r2, emitter_dev.cdf,
                        emitter_dev.tri_a, emitter_dev.tri_e1, emitter_dev.tri_e2,
                        emitter_dev.tri_u, emitter_dev.tri_v, emitter_dev.tri_n, emitter_dev.tri_origin_eps,
                        rays, d_orig, d_dirs, cp_grid, cp_dims,
                    )
                else:
                    _build_rays_host(emitter, rays, h_orig, h_dirs, cp_grid, cp_dims)
                    if cuda_async:
                        d_orig.copy_to_device(h_orig, stream=stream)
                        d_dirs.copy_to_device(h_dirs, stream=stream)
                    else:
                        d_orig.copy_to_device(h_orig)
                        d_dirs.copy_to_device(h_dirs)

                if not matrix_done and not sky_done:
                    if scene.use_bvh:
                        trace = kernel_trace_bvh_combined[blocks, threads, stream] if cuda_async else kernel_trace_bvh_combined[blocks, threads]
                        trace(
                            d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.normals, d_scene.sid,
                            d_scene.bb_min, d_scene.bb_max, d_scene.left, d_scene.right, d_scene.start, d_scene.count,
                            emit_sid, matrix_min_sid, d_hit_sid, d_hit_front, d_any_hit,
                        )
                    else:
                        trace = kernel_trace_combined[blocks, threads, stream] if cuda_async else kernel_trace_combined[blocks, threads]
                        trace(
                            d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.normals, d_scene.sid,
                            emit_sid, matrix_min_sid, d_hit_sid, d_hit_front, d_any_hit,
                        )

                    zero_hits = kernel_zero_i64[surf_blocks, surf_threads, stream] if cuda_async else kernel_zero_i64[surf_blocks, surf_threads]
                    zero_hits(d_hf)
                    zero_hits(d_hb)
                    reduce_kernel = kernel_reduce_hits[blocks, threads, stream] if cuda_async else kernel_reduce_hits[blocks, threads]
                    reduce_kernel(d_hit_sid, d_hit_front, d_hf, d_hb, n_surf)

                    if sky_discrete:
                        zero_sky = kernel_zero_i32[sky_blocks, sky_threads, stream] if cuda_async else kernel_zero_i32[sky_blocks, sky_threads]
                        zero_sky(d_counts)
                        sky_kernel = kernel_bin_tregenza[blocks, threads, stream] if cuda_async else kernel_bin_tregenza[blocks, threads]
                        sky_kernel(d_dirs, d_any_hit, d_counts)
                    else:
                        zero_sky = kernel_zero_i32[sky_blocks, sky_threads, stream] if cuda_async else kernel_zero_i32[sky_blocks, sky_threads]
                        zero_sky(d_upward)
                        sky_kernel = kernel_count_upward_misses[blocks, threads, stream] if cuda_async else kernel_count_upward_misses[blocks, threads]
                        sky_kernel(d_dirs, d_any_hit, d_upward)

                    if cuda_async:
                        d_hf.copy_to_host(h_hf, stream=stream)
                        d_hb.copy_to_host(h_hb, stream=stream)
                        if sky_discrete:
                            d_counts.copy_to_host(h_counts, stream=stream)
                        else:
                            d_upward.copy_to_host(h_upward, stream=stream)
                        stream.synchronize()
                        hits_f_iter = np.asarray(h_hf)
                        hits_b_iter = np.asarray(h_hb)
                        if sky_discrete:
                            counts_iter_arr = np.asarray(h_counts, dtype=np.int64)
                        else:
                            upward_iter = int(h_upward[0])
                    else:
                        cuda.synchronize()
                        hits_f_iter = d_hf.copy_to_host()
                        hits_b_iter = d_hb.copy_to_host()
                        if sky_discrete:
                            counts_iter_arr = d_counts.copy_to_host().astype(np.int64)
                        else:
                            upward_iter = int(d_upward.copy_to_host()[0])
                elif not matrix_done:
                    zero_hits = kernel_zero_i64[surf_blocks, surf_threads, stream] if cuda_async else kernel_zero_i64[surf_blocks, surf_threads]
                    zero_hits(d_hf)
                    zero_hits(d_hb)
                    if scene.use_bvh:
                        trace = kernel_trace_bvh_firsthit[blocks, threads, stream] if cuda_async else kernel_trace_bvh_firsthit[blocks, threads]
                        trace(
                            d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.normals, d_scene.sid,
                            d_scene.bb_min, d_scene.bb_max, d_scene.left, d_scene.right, d_scene.start, d_scene.count,
                            emit_sid, matrix_min_sid, d_hit_sid, d_hit_front,
                        )
                    else:
                        trace = kernel_trace_firsthit[blocks, threads, stream] if cuda_async else kernel_trace_firsthit[blocks, threads]
                        trace(d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.normals, d_scene.sid, emit_sid, matrix_min_sid, d_hit_sid, d_hit_front)
                    reduce_kernel = kernel_reduce_hits[blocks, threads, stream] if cuda_async else kernel_reduce_hits[blocks, threads]
                    reduce_kernel(d_hit_sid, d_hit_front, d_hf, d_hb, n_surf)

                    if cuda_async:
                        d_hf.copy_to_host(h_hf, stream=stream)
                        d_hb.copy_to_host(h_hb, stream=stream)
                        stream.synchronize()
                        hits_f_iter = np.asarray(h_hf)
                        hits_b_iter = np.asarray(h_hb)
                    else:
                        cuda.synchronize()
                        hits_f_iter = d_hf.copy_to_host()
                        hits_b_iter = d_hb.copy_to_host()
                else:
                    zero_sky = kernel_zero_i32[sky_blocks, sky_threads, stream] if cuda_async else kernel_zero_i32[sky_blocks, sky_threads]
                    if sky_discrete:
                        zero_sky(d_counts)
                        if scene.use_bvh:
                            trace = kernel_trace_bvh_tregenza[blocks, threads, stream] if cuda_async else kernel_trace_bvh_tregenza[blocks, threads]
                            trace(
                                d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.sid,
                                d_scene.bb_min, d_scene.bb_max, d_scene.left, d_scene.right, d_scene.start, d_scene.count,
                                idx_emit, 0, d_counts,
                            )
                        else:
                            trace = kernel_trace_tregenza[blocks, threads, stream] if cuda_async else kernel_trace_tregenza[blocks, threads]
                            trace(d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.sid, idx_emit, 0, d_counts)
                    else:
                        zero_sky(d_upward)
                        if scene.use_bvh:
                            trace = kernel_trace_bvh_count_upward[blocks, threads, stream] if cuda_async else kernel_trace_bvh_count_upward[blocks, threads]
                            trace(
                                d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.sid,
                                d_scene.bb_min, d_scene.bb_max, d_scene.left, d_scene.right, d_scene.start, d_scene.count,
                                idx_emit, 0, d_upward,
                            )
                        else:
                            trace = kernel_trace_count_upward[blocks, threads, stream] if cuda_async else kernel_trace_count_upward[blocks, threads]
                            trace(d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.sid, idx_emit, 0, d_upward)

                    if cuda_async:
                        if sky_discrete:
                            d_counts.copy_to_host(h_counts, stream=stream)
                        else:
                            d_upward.copy_to_host(h_upward, stream=stream)
                        stream.synchronize()
                        if sky_discrete:
                            counts_iter_arr = np.asarray(h_counts, dtype=np.int64)
                        else:
                            upward_iter = int(h_upward[0])
                    else:
                        cuda.synchronize()
                        if sky_discrete:
                            counts_iter_arr = d_counts.copy_to_host().astype(np.int64)
                        else:
                            upward_iter = int(d_upward.copy_to_host()[0])
            else:
                _build_rays_host(emitter, rays, orig, dire, cp_grid, cp_dims)
                if not matrix_done and not sky_done:
                    if scene.use_bvh:
                        trace_cpu_bvh_combined(
                            orig, dire, scene.v0, scene.e1, scene.e2, scene.normals, scene.sid,
                            scene.bb_min, scene.bb_max, scene.left, scene.right, scene.start, scene.count,
                            emit_sid, matrix_min_sid, hit_sid_iter, front_iter, any_hit_iter,
                        )
                    else:
                        trace_cpu_combined(
                            orig, dire, scene.v0, scene.e1, scene.e2, scene.normals, scene.sid,
                            emit_sid, matrix_min_sid, hit_sid_iter, front_iter, any_hit_iter,
                        )
                    reduce_first_hits(hit_sid_iter, front_iter, hits_f_iter, hits_b_iter)
                    if sky_discrete:
                        bin_tregenza_cpu(dire, any_hit_iter, counts_iter)
                        counts_iter_arr = counts_iter
                    else:
                        upward_iter = int(count_upward_misses_cpu(dire, any_hit_iter))
                elif not matrix_done:
                    if scene.use_bvh:
                        trace_cpu_bvh_firsthit(
                            orig, dire, scene.v0, scene.e1, scene.e2, scene.normals, scene.sid,
                            scene.bb_min, scene.bb_max, scene.left, scene.right, scene.start, scene.count,
                            emit_sid, matrix_min_sid, hit_sid_iter, front_iter,
                        )
                    else:
                        trace_cpu_firsthit(orig, dire, scene.v0, scene.e1, scene.e2, scene.normals, scene.sid, emit_sid, matrix_min_sid, hit_sid_iter, front_iter)
                    reduce_first_hits(hit_sid_iter, front_iter, hits_f_iter, hits_b_iter)
                else:
                    if scene.use_bvh:
                        trace_cpu_bvh_hitmask(
                            orig, dire, scene.v0, scene.e1, scene.e2, scene.sid,
                            scene.bb_min, scene.bb_max, scene.left, scene.right, scene.start, scene.count,
                            idx_emit, 0, hitmask_iter,
                        )
                    else:
                        trace_cpu_hitmask(orig, dire, scene.v0, scene.e1, scene.e2, scene.sid, idx_emit, 0, hitmask_iter)
                    if sky_discrete:
                        bin_tregenza_cpu(dire, hitmask_iter, counts_iter)
                        counts_iter_arr = counts_iter
                    else:
                        upward_iter = int(count_upward_misses_cpu(dire, hitmask_iter))

            trace_iters += 1
            total_traced_rays += n_rays_once

            if not matrix_done:
                hits_f += hits_f_iter
                hits_b += hits_b_iter
                matrix_total_rays += n_rays_once
                matrix_iters_done += 1

                f_iter = hits_f_iter.astype(np.float64) / float(n_rays_once)
                b_iter = hits_b_iter.astype(np.float64) / float(n_rays_once)
                delta_f = f_iter - mean_f
                mean_f += delta_f / matrix_iters_done
                M2_f += delta_f * (f_iter - mean_f)
                delta_b = b_iter - mean_b
                mean_b += delta_b / matrix_iters_done
                M2_b += delta_b * (b_iter - mean_b)

                check_matrix = _convergence_checkpoint(
                    matrix_iters_done,
                    min_iters=int(mp["min_iters"]),
                    interval=matrix_interval if use_gpu else 1,
                    max_iters=int(mp["max_iters"]),
                    needs_variance=(mp["tol_mode"] == "stderr"),
                )
                if mp["tol_mode"] == "delta":
                    curr_f = hits_f / float(matrix_total_rays)
                    curr_b = hits_b / float(matrix_total_rays)
                    if check_matrix and prev_f is not None:
                        if np.all(np.abs(curr_f - prev_f) < float(mp["tol"])) and np.all(np.abs(curr_b - prev_b) < float(mp["tol"])):
                            matrix_done = True
                    if check_matrix:
                        prev_f = curr_f.copy()
                        prev_b = curr_b.copy()
                elif mp["tol_mode"] == "stderr":
                    if check_matrix:
                        se_f = np.sqrt(np.maximum(M2_f / (matrix_iters_done - 1), 0.0) / matrix_iters_done)
                        se_b = np.sqrt(np.maximum(M2_b / (matrix_iters_done - 1), 0.0) / matrix_iters_done)
                        if np.all(se_f[recv_idx] <= float(mp["tol"])) and np.all(se_b[recv_idx] <= float(mp["tol"])):
                            matrix_done = True
                else:
                    raise ValueError(f"Unknown tol_mode: {mp['tol_mode']}")
                if matrix_iters_done >= int(mp["max_iters"]):
                    matrix_done = True

            if not sky_done:
                sky_total_rays += n_rays_once
                sky_iters_done += 1
                check_sky = _convergence_checkpoint(
                    sky_iters_done,
                    min_iters=int(sp["min_iters"]),
                    interval=sky_interval if use_gpu else 1,
                    max_iters=int(sp["max_iters"]),
                    needs_variance=(sp["tol_mode"] == "stderr"),
                )

                if sky_discrete:
                    counts_total += counts_iter_arr
                    frac_iter = counts_iter_arr.astype(np.float64) / float(n_rays_once)
                    sky_iter = float(frac_iter.sum())
                    delta = frac_iter - mean_bins
                    mean_bins += delta / sky_iters_done
                    M2_bins += delta * (frac_iter - mean_bins)
                    delta_sky = sky_iter - mean_sky
                    mean_sky += delta_sky / sky_iters_done
                    M2_sky += delta_sky * (sky_iter - mean_sky)

                    if sp["tol_mode"] == "delta":
                        if check_sky:
                            curr = counts_total.astype(np.float64) / float(sky_total_rays)
                            if prev_frac is not None and np.all(np.abs(curr - prev_frac) < float(sp["tol"])):
                                sky_done = True
                            prev_frac = curr.copy()
                    elif sp["tol_mode"] == "stderr":
                        if check_sky:
                            se_bins = np.sqrt(np.maximum(M2_bins / (sky_iters_done - 1), 0.0) / sky_iters_done)
                            if np.all(se_bins <= float(sp["tol"])):
                                sky_done = True
                    else:
                        raise ValueError(f"Unknown tol_mode: {sp['tol_mode']}")
                else:
                    upward_total += upward_iter
                    frac_iter = upward_iter / float(n_rays_once)
                    delta_sky = frac_iter - mean_sky
                    mean_sky += delta_sky / sky_iters_done
                    M2_sky += delta_sky * (frac_iter - mean_sky)

                    if sp["tol_mode"] == "delta":
                        if check_sky:
                            curr = upward_total / float(sky_total_rays)
                            if prev_frac is not None and abs(curr - prev_frac) < float(sp["tol"]):
                                sky_done = True
                            prev_frac = curr
                    elif sp["tol_mode"] == "stderr":
                        if check_sky:
                            se_sky = max(M2_sky / (sky_iters_done - 1), 0.0) ** 0.5 / sky_iters_done**0.5
                            if se_sky <= float(sp["tol"]):
                                sky_done = True
                    else:
                        raise ValueError(f"Unknown tol_mode: {sp['tol_mode']}")
                if sky_iters_done >= int(sp["max_iters"]):
                    sky_done = True

        if matrix_iters_done > 1:
            se_f = np.sqrt(np.maximum(M2_f / (matrix_iters_done - 1), 0.0) / matrix_iters_done)
            se_b = np.sqrt(np.maximum(M2_b / (matrix_iters_done - 1), 0.0) / matrix_iters_done)
        else:
            se_f = np.full_like(mean_f, np.inf, dtype=np.float64)
            se_b = np.full_like(mean_b, np.inf, dtype=np.float64)

        row: Dict[str, float] = {}
        for j in receivers:
            name_r, _, _ = meshes[j]
            f = hits_f[j] / float(matrix_total_rays) if matrix_total_rays > 0 else 0.0
            b = hits_b[j] / float(matrix_total_rays) if matrix_total_rays > 0 else 0.0
            if f > 0.0:
                row[f"{name_r}_front"] = f
                if reciprocity and areas is not None and areas[j] > 0.0:
                    vf_scene[name_r][f"{name_e}_front"] = f * (areas[idx_emit] / areas[j])
            if b > 0.0:
                row[f"{name_r}_back"] = b
        vf_scene[name_e].update(row)

        if sky_total_rays > 0:
            if sky_discrete:
                frac = counts_total.astype(np.float64) / float(sky_total_rays)
                sky_vf[name_e].update({f"Sky_Patch_{i+1}": float(frac[i]) for i in range(145)})
            else:
                sky_vf[name_e]["Sky"] = float(upward_total / float(sky_total_rays))

        msg = (
            f"({idx_emit+1}/{len(meshes)}) [{name_e}] traced {trace_iters} iter, {total_traced_rays:,} rays -> "
            f"{time.time() - t0:0.3f}s  (scene={matrix_iters_done} iter, sky={sky_iters_done} iter, "
            f"BVH={'builtin' if use_bvh else 'off'}, device={'gpu' if use_gpu else 'cpu'})"
        )
        _log(msg)
        _rhino_log(msg)

    return vf_scene, sky_vf


def view_factor_matrix(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    params: MatrixParams,
    *,
    prepared: PreparedSolver | None = None,
):
    if not isinstance(params, MatrixParams):
        raise TypeError("params must be a MatrixParams instance")

    p = params.as_dict()
    samples = p["samples"]
    rays = p["rays"]
    seed = p["seed"]
    device = p["device"]
    cuda_async = p["cuda_async"]
    gpu_raygen = p["gpu_raygen"]
    max_iters = p["max_iters"]
    tol = p["tol"]
    tol_mode = p["tol_mode"]
    min_iters = p["min_iters"]
    convergence_interval = max(1, int(p["convergence_interval"]))
    reciprocity = p["reciprocity"]
    enforce_reciprocity_rowsum = p["enforce_reciprocity_rowsum"]
    flip_faces = p["flip_faces"]
    return_stats = False

    use_gpu = _resolve_device(device)
    prepared_solver = _ensure_prepared(meshes, prepared)
    use_bvh = _select_bvh(p["bvh"], prepared_solver.total_faces)
    result: Dict[str, Dict[str, float]] = {name: {} for name, _, _ in meshes}
    stats_result: Dict[str, Dict[str, float]] = {} if return_stats else None  # type: ignore[assignment]
    emitters = prepared_solver.get_emitters(samples=samples, rays=rays, flip_faces=flip_faces)
    areas = [emitter.total_area for emitter in emitters] if reciprocity else None
    scene = prepared_solver.get_scene(use_bvh=use_bvh)

    d_scene = prepared_solver.get_device_scene(use_bvh=use_bvh) if use_gpu else None
    if use_gpu:
        _run_view_factor_matrix_gpu(
            meshes,
            emitters,
            prepared_solver,
            scene,
            d_scene,
            result,
            stats_result,
            samples=samples,
            rays=rays,
            seed=seed,
            gpu_raygen=gpu_raygen,
            cuda_async=cuda_async,
            flip_faces=flip_faces,
            reciprocity=reciprocity,
            tol=tol,
            tol_mode=tol_mode,
            min_iters=min_iters,
            convergence_interval=convergence_interval,
            max_iters=max_iters,
            return_stats=return_stats,
            areas=areas,
            use_bvh=use_bvh,
        )
        if enforce_reciprocity_rowsum:
            _enforce_reciprocity_and_rowsum(result, meshes, areas)
        if return_stats:
            return result, stats_result or {}
        return result

    n_surf = len(meshes)
    for idx_emit, (name_e, _, _) in enumerate(meshes):
        t_tot = time.time()
        receivers = [j for j in range(idx_emit + 1, len(meshes))] if reciprocity else [j for j in range(len(meshes)) if j != idx_emit]
        if not receivers:
            msg = f"({idx_emit+1}/{len(meshes)}) [{name_e}] 0 iter, 0 rays -> 0.000s  (BVH={'builtin' if use_bvh else 'off'}, device={'gpu' if use_gpu else 'cpu'})"
            _log(msg)
            _rhino_log(msg)
            continue

        recv_idx = np.asarray(receivers, dtype=np.int32)
        emit_sid, min_sid = _matrix_skip(idx_emit, reciprocity)
        emitter = emitters[idx_emit]
        n_rays_once = emitter.n_cells * rays
        hits_f = np.zeros(n_surf, np.int64)
        hits_b = np.zeros(n_surf, np.int64)
        mean_f = np.zeros(n_surf, np.float64)
        mean_b = np.zeros(n_surf, np.float64)
        M2_f = np.zeros(n_surf, np.float64)
        M2_b = np.zeros(n_surf, np.float64)
        prev_f = prev_b = None
        total_rays = 0
        iters_done = 0

        if use_gpu:
            emitter_dev = prepared_solver.get_device_emitter(idx_emit, samples=samples, rays=rays, flip_faces=flip_faces)
            d_orig = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_dirs = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_hit_sid = cuda.device_array(n_rays_once, dtype=np.int32)
            d_hit_front = cuda.device_array(n_rays_once, dtype=np.uint8)
            d_hf = cuda.device_array(n_surf, dtype=np.int64)
            d_hb = cuda.device_array(n_surf, dtype=np.int64)
            stream = cuda.stream() if cuda_async else None
            if cuda_async:
                h_hf = cuda.pinned_array(n_surf, dtype=np.int64)
                h_hb = cuda.pinned_array(n_surf, dtype=np.int64)
                if not gpu_raygen:
                    h_orig = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
                    h_dirs = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
            else:
                h_hf = h_hb = None
                if not gpu_raygen:
                    h_orig, h_dirs = _host_ray_buffers(n_rays_once)
            blocks, threads = _compute_cuda_launch(n_rays_once, None)
            zblocks, zthreads = _compute_cuda_launch(n_surf, threads)
        else:
            orig, dire = _host_ray_buffers(n_rays_once)
            hit_sid_iter = np.empty(n_rays_once, dtype=np.int32)
            front_iter = np.empty(n_rays_once, dtype=np.uint8)
            hits_f_iter = np.empty(n_surf, dtype=np.int64)
            hits_b_iter = np.empty(n_surf, dtype=np.int64)

        for itr in range(max_iters):
            rng = np.random.default_rng(seed + idx_emit + itr)
            cp_grid = rng.random(2, dtype=np.float32)
            cp_dims = rng.random(5, dtype=np.float32)

            if use_gpu:
                if gpu_raygen:
                    launch = kernel_build_rays[blocks, threads, stream] if cuda_async else kernel_build_rays[blocks, threads]
                    launch(
                        emitter_dev.u_grid, emitter_dev.v_grid,
                        emitter_dev.halton_tri, emitter_dev.halton_u, emitter_dev.halton_v,
                        emitter_dev.halton_r1, emitter_dev.halton_r2, emitter_dev.cdf,
                        emitter_dev.tri_a, emitter_dev.tri_e1, emitter_dev.tri_e2,
                        emitter_dev.tri_u, emitter_dev.tri_v, emitter_dev.tri_n, emitter_dev.tri_origin_eps,
                        rays, d_orig, d_dirs, cp_grid, cp_dims,
                    )
                else:
                    _build_rays_host(emitter, rays, h_orig, h_dirs, cp_grid, cp_dims)
                    if cuda_async:
                        d_orig.copy_to_device(h_orig, stream=stream)
                        d_dirs.copy_to_device(h_dirs, stream=stream)
                    else:
                        d_orig.copy_to_device(h_orig)
                        d_dirs.copy_to_device(h_dirs)

                zero = kernel_zero_i64[zblocks, zthreads, stream] if cuda_async else kernel_zero_i64[zblocks, zthreads]
                zero(d_hf)
                zero(d_hb)
                if scene.use_bvh:
                    trace = kernel_trace_bvh_firsthit[blocks, threads, stream] if cuda_async else kernel_trace_bvh_firsthit[blocks, threads]
                    trace(
                        d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.normals, d_scene.sid,
                        d_scene.bb_min, d_scene.bb_max, d_scene.left, d_scene.right, d_scene.start, d_scene.count,
                        emit_sid, min_sid, d_hit_sid, d_hit_front,
                    )
                else:
                    trace = kernel_trace_firsthit[blocks, threads, stream] if cuda_async else kernel_trace_firsthit[blocks, threads]
                    trace(d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.normals, d_scene.sid, emit_sid, min_sid, d_hit_sid, d_hit_front)
                reduce_kernel = kernel_reduce_hits[blocks, threads, stream] if cuda_async else kernel_reduce_hits[blocks, threads]
                reduce_kernel(d_hit_sid, d_hit_front, d_hf, d_hb, n_surf)

                if cuda_async:
                    d_hf.copy_to_host(h_hf, stream=stream)
                    d_hb.copy_to_host(h_hb, stream=stream)
                    stream.synchronize()
                    hits_f_iter = np.asarray(h_hf)
                    hits_b_iter = np.asarray(h_hb)
                else:
                    cuda.synchronize()
                    hits_f_iter = d_hf.copy_to_host()
                    hits_b_iter = d_hb.copy_to_host()
            else:
                _build_rays_host(emitter, rays, orig, dire, cp_grid, cp_dims)
                if scene.use_bvh:
                    trace_cpu_bvh_firsthit(
                        orig, dire, scene.v0, scene.e1, scene.e2, scene.normals, scene.sid,
                        scene.bb_min, scene.bb_max, scene.left, scene.right, scene.start, scene.count,
                        emit_sid, min_sid, hit_sid_iter, front_iter,
                    )
                else:
                    trace_cpu_firsthit(orig, dire, scene.v0, scene.e1, scene.e2, scene.normals, scene.sid, emit_sid, min_sid, hit_sid_iter, front_iter)
                reduce_first_hits(hit_sid_iter, front_iter, hits_f_iter, hits_b_iter)

            hits_f += hits_f_iter
            hits_b += hits_b_iter
            total_rays += n_rays_once
            iters_done += 1

            f_iter = hits_f_iter.astype(np.float64) / float(n_rays_once)
            b_iter = hits_b_iter.astype(np.float64) / float(n_rays_once)
            delta_f = f_iter - mean_f
            mean_f += delta_f / iters_done
            M2_f += delta_f * (f_iter - mean_f)
            delta_b = b_iter - mean_b
            mean_b += delta_b / iters_done
            M2_b += delta_b * (b_iter - mean_b)

            check_matrix = _convergence_checkpoint(
                iters_done,
                min_iters=min_iters,
                interval=convergence_interval if use_gpu else 1,
                max_iters=max_iters,
                needs_variance=(tol_mode == "stderr"),
            )
            if tol_mode == "delta":
                curr_f = hits_f / float(total_rays)
                curr_b = hits_b / float(total_rays)
                if check_matrix and prev_f is not None:
                    if np.all(np.abs(curr_f - prev_f) < tol) and np.all(np.abs(curr_b - prev_b) < tol):
                        break
                if check_matrix:
                    prev_f = curr_f.copy()
                    prev_b = curr_b.copy()
            elif tol_mode == "stderr":
                if check_matrix:
                    se_f = np.sqrt(np.maximum(M2_f / (iters_done - 1), 0.0) / iters_done)
                    se_b = np.sqrt(np.maximum(M2_b / (iters_done - 1), 0.0) / iters_done)
                    if np.all(se_f[recv_idx] <= tol) and np.all(se_b[recv_idx] <= tol):
                        break
            else:
                raise ValueError(f"Unknown tol_mode: {tol_mode}")

        if iters_done > 1:
            se_f = np.sqrt(np.maximum(M2_f / (iters_done - 1), 0.0) / iters_done)
            se_b = np.sqrt(np.maximum(M2_b / (iters_done - 1), 0.0) / iters_done)
        else:
            se_f = np.full_like(mean_f, np.inf, dtype=np.float64)
            se_b = np.full_like(mean_b, np.inf, dtype=np.float64)

        row: Dict[str, float] = {}
        stats_row: Dict[str, float] = {}
        for j in receivers:
            name_r, _, _ = meshes[j]
            f = hits_f[j] / float(total_rays)
            b = hits_b[j] / float(total_rays)
            if f > 0.0:
                row[f"{name_r}_front"] = f
                if reciprocity and areas is not None and areas[j] > 0.0:
                    result[name_r][f"{name_e}_front"] = f * (areas[idx_emit] / areas[j])
                if return_stats:
                    stats_row[f"{name_r}_front"] = float(se_f[j])
            if b > 0.0:
                row[f"{name_r}_back"] = b
                if return_stats:
                    stats_row[f"{name_r}_back"] = float(se_b[j])
        result[name_e].update(row)
        if return_stats:
            stats_result[name_e] = stats_row

        msg = f"({idx_emit+1}/{len(meshes)}) [{name_e}] {iters_done} iter, {total_rays:,} rays -> {time.time() - t_tot:0.3f}s  (BVH={'builtin' if use_bvh else 'off'}, device={'gpu' if use_gpu else 'cpu'})"
        _log(msg)
        _rhino_log(msg)

    if enforce_reciprocity_rowsum:
        _enforce_reciprocity_and_rowsum(result, meshes, areas)
    if return_stats:
        return result, stats_result or {}
    return result


def view_factor(sender, receiver, params: MatrixParams, *, prepared: PreparedSolver | None = None):
    senders = [sender] if isinstance(sender, tuple) else list(sender)
    receivers = [receiver] if isinstance(receiver, tuple) else list(receiver)
    meshes = senders + receivers
    vf_all = view_factor_matrix(meshes, params=params, prepared=prepared)
    sender_names = [s[0] for s in senders]
    return {name: vf_all.get(name, {}) for name in sender_names}


def view_factor_to_tregenza_sky(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    params: SkyParams,
    *,
    prepared: PreparedSolver | None = None,
):
    if not isinstance(params, SkyParams):
        raise TypeError("params must be a SkyParams instance")
    if len(meshes) == 0:
        raise ValueError("meshes must not be empty")

    p = params.as_dict()
    samples = p["samples"]
    rays = p["rays"]
    seed = p["seed"]
    device = p["device"]
    cuda_async = p["cuda_async"]
    gpu_raygen = p["gpu_raygen"]
    max_iters = p["max_iters"]
    tol = p["tol"]
    tol_mode = p["tol_mode"]
    min_iters = p["min_iters"]
    convergence_interval = max(1, int(p["convergence_interval"]))
    discrete = p["discrete"]

    use_gpu = _resolve_device(device)
    prepared_solver = _ensure_prepared(meshes, prepared)
    use_bvh = _select_bvh(p["bvh"], prepared_solver.total_faces)
    emitters = prepared_solver.get_emitters(samples=samples, rays=rays, flip_faces=False)
    scene = prepared_solver.get_scene(use_bvh=use_bvh)
    sky_name = "Sky"
    patch_prefix = "Sky_Patch_"
    sky_keys = [f"{patch_prefix}{i}" for i in range(1, 146)] if discrete else [sky_name]
    result: Dict[str, Dict[str, float]] = {name: {k: 0.0 for k in sky_keys} for name, _, _ in meshes}

    d_scene = None
    if use_gpu:
        d_scene = prepared_solver.get_device_scene(use_bvh=use_bvh)

    for idx_emit, (name_e, _, _) in enumerate(meshes):
        if len(meshes) <= 1:
            continue
        t0 = time.time()
        emitter = emitters[idx_emit]
        n_rays_once = emitter.n_cells * rays
        counts_total = np.zeros(145, np.int64) if discrete else None
        mean_bins = np.zeros(145, np.float64) if discrete else None
        M2_bins = np.zeros(145, np.float64) if discrete else None
        upward_total = 0
        mean_sky = 0.0
        M2_sky = 0.0
        prev_frac = None
        total_rays = 0
        iters_done = 0

        if use_gpu:
            emitter_dev = prepared_solver.get_device_emitter(idx_emit, samples=samples, rays=rays, flip_faces=False)
            d_orig = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_dirs = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_counts = cuda.device_array(145, dtype=np.int32) if discrete else None
            d_upward = cuda.device_array(1, dtype=np.int32) if not discrete else None
            stream = cuda.stream() if cuda_async else None
            if cuda_async:
                h_counts = cuda.pinned_array(145, dtype=np.int32) if discrete else None
                h_upward = cuda.pinned_array(1, dtype=np.int32) if not discrete else None
                if not gpu_raygen:
                    h_orig = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
                    h_dirs = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
            else:
                h_counts = h_upward = None
                if not gpu_raygen:
                    h_orig, h_dirs = _host_ray_buffers(n_rays_once)
            blocks, threads = _compute_cuda_launch(n_rays_once, None)
            zblocks, zthreads = _compute_cuda_launch(145 if discrete else 1, threads)
        else:
            orig, dire = _host_ray_buffers(n_rays_once)
            hitmask = np.empty(n_rays_once, dtype=np.uint8)
            counts_iter = np.empty(145, dtype=np.int64) if discrete else None

        for itr in range(max_iters):
            rng = np.random.default_rng(seed + idx_emit + itr)
            cp_grid = rng.random(2, dtype=np.float32)
            cp_dims = rng.random(5, dtype=np.float32)

            if use_gpu:
                if gpu_raygen:
                    launch = kernel_build_rays[blocks, threads, stream] if cuda_async else kernel_build_rays[blocks, threads]
                    launch(
                        emitter_dev.u_grid, emitter_dev.v_grid,
                        emitter_dev.halton_tri, emitter_dev.halton_u, emitter_dev.halton_v,
                        emitter_dev.halton_r1, emitter_dev.halton_r2, emitter_dev.cdf,
                        emitter_dev.tri_a, emitter_dev.tri_e1, emitter_dev.tri_e2,
                        emitter_dev.tri_u, emitter_dev.tri_v, emitter_dev.tri_n, emitter_dev.tri_origin_eps,
                        rays, d_orig, d_dirs, cp_grid, cp_dims,
                    )
                else:
                    _build_rays_host(emitter, rays, h_orig, h_dirs, cp_grid, cp_dims)
                    if cuda_async:
                        d_orig.copy_to_device(h_orig, stream=stream)
                        d_dirs.copy_to_device(h_dirs, stream=stream)
                    else:
                        d_orig.copy_to_device(h_orig)
                        d_dirs.copy_to_device(h_dirs)

                zero = kernel_zero_i32[zblocks, zthreads, stream] if cuda_async else kernel_zero_i32[zblocks, zthreads]
                if discrete:
                    zero(d_counts)
                    if scene.use_bvh:
                        trace = kernel_trace_bvh_tregenza[blocks, threads, stream] if cuda_async else kernel_trace_bvh_tregenza[blocks, threads]
                        trace(
                            d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.sid,
                            d_scene.bb_min, d_scene.bb_max, d_scene.left, d_scene.right, d_scene.start, d_scene.count,
                            idx_emit, 0, d_counts,
                        )
                    else:
                        trace = kernel_trace_tregenza[blocks, threads, stream] if cuda_async else kernel_trace_tregenza[blocks, threads]
                        trace(d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.sid, idx_emit, 0, d_counts)
                else:
                    zero(d_upward)
                    if scene.use_bvh:
                        trace = kernel_trace_bvh_count_upward[blocks, threads, stream] if cuda_async else kernel_trace_bvh_count_upward[blocks, threads]
                        trace(
                            d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.sid,
                            d_scene.bb_min, d_scene.bb_max, d_scene.left, d_scene.right, d_scene.start, d_scene.count,
                            idx_emit, 0, d_upward,
                        )
                    else:
                        trace = kernel_trace_count_upward[blocks, threads, stream] if cuda_async else kernel_trace_count_upward[blocks, threads]
                        trace(d_orig, d_dirs, d_scene.v0, d_scene.e1, d_scene.e2, d_scene.sid, idx_emit, 0, d_upward)

                if cuda_async:
                    if discrete:
                        d_counts.copy_to_host(h_counts, stream=stream)
                    else:
                        d_upward.copy_to_host(h_upward, stream=stream)
                    stream.synchronize()
                    if discrete:
                        counts_iter_arr = np.asarray(h_counts, dtype=np.int64)
                    else:
                        upward_iter = int(h_upward[0])
                else:
                    cuda.synchronize()
                    if discrete:
                        counts_iter_arr = d_counts.copy_to_host().astype(np.int64)
                    else:
                        upward_iter = int(d_upward.copy_to_host()[0])
            else:
                _build_rays_host(emitter, rays, orig, dire, cp_grid, cp_dims)
                if scene.use_bvh:
                    trace_cpu_bvh_hitmask(
                        orig, dire, scene.v0, scene.e1, scene.e2, scene.sid,
                        scene.bb_min, scene.bb_max, scene.left, scene.right, scene.start, scene.count,
                        idx_emit, 0, hitmask,
                    )
                else:
                    trace_cpu_hitmask(orig, dire, scene.v0, scene.e1, scene.e2, scene.sid, idx_emit, 0, hitmask)
                if discrete:
                    bin_tregenza_cpu(dire, hitmask, counts_iter)
                    counts_iter_arr = counts_iter
                else:
                    upward_iter = int(count_upward_misses_cpu(dire, hitmask))

            total_rays += n_rays_once
            iters_done += 1
            check_sky = _convergence_checkpoint(
                iters_done,
                min_iters=min_iters,
                interval=convergence_interval if use_gpu else 1,
                max_iters=max_iters,
                needs_variance=(tol_mode == "stderr"),
            )
            if discrete:
                counts_total += counts_iter_arr
                frac_iter = counts_iter_arr.astype(np.float64) / float(n_rays_once)
                sky_iter = float(frac_iter.sum())
                delta = frac_iter - mean_bins
                mean_bins += delta / iters_done
                M2_bins += delta * (frac_iter - mean_bins)
                delta_sky = sky_iter - mean_sky
                mean_sky += delta_sky / iters_done
                M2_sky += delta_sky * (sky_iter - mean_sky)

                if tol_mode == "delta":
                    if check_sky:
                        curr = counts_total.astype(np.float64) / float(total_rays)
                        if prev_frac is not None and np.all(np.abs(curr - prev_frac) < tol):
                            break
                        prev_frac = curr.copy()
                elif tol_mode == "stderr":
                    if check_sky:
                        se_bins = np.sqrt(np.maximum(M2_bins / (iters_done - 1), 0.0) / iters_done)
                        if np.all(se_bins <= tol):
                            break
                else:
                    raise ValueError(f"Unknown tol_mode: {tol_mode}")
            else:
                upward_total += upward_iter
                frac_iter = upward_iter / float(n_rays_once)
                delta_sky = frac_iter - mean_sky
                mean_sky += delta_sky / iters_done
                M2_sky += delta_sky * (frac_iter - mean_sky)

                if tol_mode == "delta":
                    if check_sky:
                        curr = upward_total / float(total_rays)
                        if prev_frac is not None and abs(curr - prev_frac) < tol:
                            break
                        prev_frac = curr
                elif tol_mode == "stderr":
                    if check_sky:
                        se_sky = max(M2_sky / (iters_done - 1), 0.0) ** 0.5 / iters_done**0.5
                        if se_sky <= tol:
                            break
                else:
                    raise ValueError(f"Unknown tol_mode: {tol_mode}")

        if discrete:
            frac = counts_total.astype(np.float64) / float(max(1, total_rays))
            result[name_e].update({f"{patch_prefix}{i+1}": float(frac[i]) for i in range(145)})
        else:
            result[name_e][sky_name] = float(upward_total / float(max(1, total_rays)))

        msg = f"({idx_emit+1}/{len(meshes)}) [{name_e}] {iters_done} iter, {total_rays:,} rays -> {time.time() - t0:0.3f}s  (BVH={'builtin' if use_bvh else 'off'}, device={'gpu' if use_gpu else 'cpu'})"
        _log(msg)
        _rhino_log(msg)

    return result


__all__ = [
    "outside_workflow_shareable",
    "view_factor_matrix",
    "view_factor_matrix_and_sky",
    "view_factor",
    "view_factor_to_tregenza_sky",
]
