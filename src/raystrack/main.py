from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

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
    reduce_first_hits,
    trace_cpu_bvh_combined,
    trace_cpu_bvh_firsthit,
    trace_cpu_bvh_hitmask,
    trace_cpu_combined,
    trace_cpu_firsthit,
    trace_cpu_hitmask,
)
from .utils.cuda_trace import (
    kernel_bin_tregenza,
    kernel_build_rays,
    kernel_reduce_hits,
    kernel_trace_bvh_combined,
    kernel_trace_bvh_firsthit,
    kernel_trace_bvh_tregenza,
    kernel_trace_combined,
    kernel_trace_firsthit,
    kernel_trace_tregenza,
    kernel_zero_i32,
    kernel_zero_i64,
)
from .utils.helpers import enforce_reciprocity_and_rowsum as _enforce_reciprocity_and_rowsum
from .utils.prepared import PreparedEmitter, prepare_emitters, prepare_scene
from .utils.ray_builder import build_rays

_LOG_PROC = None
_BVH_AUTO_THRESHOLD = 512


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


def _host_ray_buffers(n_rays: int) -> Tuple[np.ndarray, np.ndarray]:
    orig = np.empty((n_rays, 3), dtype=np.float32)
    return orig, np.empty_like(orig)


def _upload_emitter_data(emitter: PreparedEmitter):
    return {
        "u_grid": cuda.to_device(emitter.u_grid),
        "v_grid": cuda.to_device(emitter.v_grid),
        "halton_tri": cuda.to_device(emitter.halton_tri),
        "halton_u": cuda.to_device(emitter.halton_u),
        "halton_v": cuda.to_device(emitter.halton_v),
        "halton_r1": cuda.to_device(emitter.halton_r1),
        "halton_r2": cuda.to_device(emitter.halton_r2),
        "cdf": cuda.to_device(emitter.cdf),
        "tri_a": cuda.to_device(emitter.tri_a),
        "tri_e1": cuda.to_device(emitter.tri_e1),
        "tri_e2": cuda.to_device(emitter.tri_e2),
        "tri_u": cuda.to_device(emitter.tri_u),
        "tri_v": cuda.to_device(emitter.tri_v),
        "tri_n": cuda.to_device(emitter.tri_n),
    }


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
        rays,
        orig,
        dire,
        cp_grid,
        cp_dims,
    )


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

    have_cuda = cuda.is_available()
    dev = (device or "auto").lower()
    if dev not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be 'auto', 'gpu', or 'cpu' (got {device!r})")
    if dev == "auto":
        use_gpu = have_cuda
    elif dev == "gpu":
        if not have_cuda:
            raise RuntimeError("device='gpu' requested but CUDA is not available")
        use_gpu = True
    else:
        use_gpu = False

    use_bvh = _select_bvh(mp["bvh"], int(sum(F.shape[0] for _, _, F in meshes)))
    emitters = prepare_emitters(meshes, samples=samples, rays=rays, flip_faces=False)
    scene = prepare_scene(meshes, use_bvh=use_bvh)
    areas = [emitter.total_area for emitter in emitters] if reciprocity else None

    vf_scene: Dict[str, Dict[str, float]] = {name: {} for name, _, _ in meshes}
    if sky_discrete:
        sky_keys = [f"Sky_Patch_{i}" for i in range(1, 146)]
        sky_vf: Dict[str, Dict[str, float]] = {name: {key: 0.0 for key in sky_keys} for name, _, _ in meshes}
    else:
        sky_vf = {name: {"Sky": 0.0} for name, _, _ in meshes}

    d_scene = None
    if use_gpu:
        d_scene = {
            "v0": cuda.to_device(scene.v0),
            "e1": cuda.to_device(scene.e1),
            "e2": cuda.to_device(scene.e2),
            "normals": cuda.to_device(scene.normals),
            "sid": cuda.to_device(scene.sid),
        }
        if scene.use_bvh:
            d_scene["bb_min"] = cuda.to_device(scene.bb_min)
            d_scene["bb_max"] = cuda.to_device(scene.bb_max)
            d_scene["left"] = cuda.to_device(scene.left)
            d_scene["right"] = cuda.to_device(scene.right)
            d_scene["start"] = cuda.to_device(scene.start)
            d_scene["count"] = cuda.to_device(scene.count)

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

        counts_total = np.zeros(145, np.int64)
        mean_bins = np.zeros(145, np.float64)
        M2_bins = np.zeros(145, np.float64)
        mean_sky = 0.0
        M2_sky = 0.0
        prev_frac = None
        sky_total_rays = 0
        sky_iters_done = 0

        trace_iters = 0
        total_traced_rays = 0
        max_trace_iters = max(int(mp["max_iters"]), int(sp["max_iters"]))

        if use_gpu:
            emitter_dev = _upload_emitter_data(emitter)
            d_orig = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_dirs = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_hit_sid = cuda.device_array(n_rays_once, dtype=np.int32)
            d_hit_front = cuda.device_array(n_rays_once, dtype=np.uint8)
            d_any_hit = cuda.device_array(n_rays_once, dtype=np.uint8)
            d_hf = cuda.device_array(n_surf, dtype=np.int64)
            d_hb = cuda.device_array(n_surf, dtype=np.int64)
            d_counts = cuda.device_array(145, dtype=np.int32)
            stream = cuda.stream() if cuda_async else None
            if cuda_async:
                h_hf = cuda.pinned_array(n_surf, dtype=np.int64)
                h_hb = cuda.pinned_array(n_surf, dtype=np.int64)
                h_counts = cuda.pinned_array(145, dtype=np.int32)
                if not gpu_raygen:
                    h_orig = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
                    h_dirs = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
            else:
                h_hf = h_hb = h_counts = None
                if not gpu_raygen:
                    h_orig, h_dirs = _host_ray_buffers(n_rays_once)
            blocks, threads = _compute_cuda_launch(n_rays_once, None)
            surf_blocks, surf_threads = _compute_cuda_launch(n_surf, threads)
            count_blocks, count_threads = _compute_cuda_launch(145, threads)
        else:
            orig, dire = _host_ray_buffers(n_rays_once)
            hit_sid_iter = np.empty(n_rays_once, dtype=np.int32)
            front_iter = np.empty(n_rays_once, dtype=np.uint8)
            any_hit_iter = np.empty(n_rays_once, dtype=np.uint8)
            hits_f_iter = np.empty(n_surf, dtype=np.int64)
            hits_b_iter = np.empty(n_surf, dtype=np.int64)
            counts_iter = np.empty(145, dtype=np.int64)

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
                        emitter_dev["u_grid"], emitter_dev["v_grid"],
                        emitter_dev["halton_tri"], emitter_dev["halton_u"], emitter_dev["halton_v"],
                        emitter_dev["halton_r1"], emitter_dev["halton_r2"], emitter_dev["cdf"],
                        emitter_dev["tri_a"], emitter_dev["tri_e1"], emitter_dev["tri_e2"],
                        emitter_dev["tri_u"], emitter_dev["tri_v"], emitter_dev["tri_n"],
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

                if scene.use_bvh:
                    trace = kernel_trace_bvh_combined[blocks, threads, stream] if cuda_async else kernel_trace_bvh_combined[blocks, threads]
                    trace(
                        d_orig, d_dirs, d_scene["v0"], d_scene["e1"], d_scene["e2"], d_scene["normals"], d_scene["sid"],
                        d_scene["bb_min"], d_scene["bb_max"], d_scene["left"], d_scene["right"], d_scene["start"], d_scene["count"],
                        emit_sid, matrix_min_sid, d_hit_sid, d_hit_front, d_any_hit,
                    )
                else:
                    trace = kernel_trace_combined[blocks, threads, stream] if cuda_async else kernel_trace_combined[blocks, threads]
                    trace(
                        d_orig, d_dirs, d_scene["v0"], d_scene["e1"], d_scene["e2"], d_scene["normals"], d_scene["sid"],
                        emit_sid, matrix_min_sid, d_hit_sid, d_hit_front, d_any_hit,
                    )

                if not matrix_done:
                    zero_hits = kernel_zero_i64[surf_blocks, surf_threads, stream] if cuda_async else kernel_zero_i64[surf_blocks, surf_threads]
                    zero_hits(d_hf)
                    zero_hits(d_hb)
                    reduce_kernel = kernel_reduce_hits[blocks, threads, stream] if cuda_async else kernel_reduce_hits[blocks, threads]
                    reduce_kernel(d_hit_sid, d_hit_front, d_hf, d_hb, n_surf)

                if not sky_done:
                    zero_counts = kernel_zero_i32[count_blocks, count_threads, stream] if cuda_async else kernel_zero_i32[count_blocks, count_threads]
                    zero_counts(d_counts)
                    bin_kernel = kernel_bin_tregenza[blocks, threads, stream] if cuda_async else kernel_bin_tregenza[blocks, threads]
                    bin_kernel(d_dirs, d_any_hit, d_counts)

                if cuda_async:
                    if not matrix_done:
                        d_hf.copy_to_host(h_hf, stream=stream)
                        d_hb.copy_to_host(h_hb, stream=stream)
                    if not sky_done:
                        d_counts.copy_to_host(h_counts, stream=stream)
                    stream.synchronize()
                    if not matrix_done:
                        hits_f_iter = np.asarray(h_hf)
                        hits_b_iter = np.asarray(h_hb)
                    if not sky_done:
                        counts_iter_arr = np.asarray(h_counts, dtype=np.int64)
                else:
                    cuda.synchronize()
                    if not matrix_done:
                        hits_f_iter = d_hf.copy_to_host()
                        hits_b_iter = d_hb.copy_to_host()
                    if not sky_done:
                        counts_iter_arr = d_counts.copy_to_host().astype(np.int64)
            else:
                _build_rays_host(emitter, rays, orig, dire, cp_grid, cp_dims)
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
                if not matrix_done:
                    reduce_first_hits(hit_sid_iter, front_iter, hits_f_iter, hits_b_iter)
                if not sky_done:
                    bin_tregenza_cpu(dire, any_hit_iter, counts_iter)
                    counts_iter_arr = counts_iter

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

                if mp["tol_mode"] == "delta":
                    curr_f = hits_f / float(matrix_total_rays)
                    curr_b = hits_b / float(matrix_total_rays)
                    if prev_f is not None and matrix_iters_done >= max(1, int(mp["min_iters"])):
                        if np.all(np.abs(curr_f - prev_f) < float(mp["tol"])) and np.all(np.abs(curr_b - prev_b) < float(mp["tol"])):
                            matrix_done = True
                    prev_f = curr_f.copy()
                    prev_b = curr_b.copy()
                elif mp["tol_mode"] == "stderr":
                    if matrix_iters_done >= max(1, int(mp["min_iters"])) and matrix_iters_done > 1:
                        se_f = np.sqrt(np.maximum(M2_f / (matrix_iters_done - 1), 0.0) / matrix_iters_done)
                        se_b = np.sqrt(np.maximum(M2_b / (matrix_iters_done - 1), 0.0) / matrix_iters_done)
                        if np.all(se_f[recv_idx] <= float(mp["tol"])) and np.all(se_b[recv_idx] <= float(mp["tol"])):
                            matrix_done = True
                else:
                    raise ValueError(f"Unknown tol_mode: {mp['tol_mode']}")
                if matrix_iters_done >= int(mp["max_iters"]):
                    matrix_done = True

            if not sky_done:
                counts_total += counts_iter_arr
                sky_total_rays += n_rays_once
                sky_iters_done += 1
                frac_iter = counts_iter_arr.astype(np.float64) / float(n_rays_once)
                sky_iter = float(frac_iter.sum())
                delta = frac_iter - mean_bins
                mean_bins += delta / sky_iters_done
                M2_bins += delta * (frac_iter - mean_bins)
                delta_sky = sky_iter - mean_sky
                mean_sky += delta_sky / sky_iters_done
                M2_sky += delta_sky * (sky_iter - mean_sky)

                if sp["tol_mode"] == "delta":
                    if sky_iters_done >= max(1, int(sp["min_iters"])):
                        curr = counts_total.astype(np.float64) / float(sky_total_rays)
                        if prev_frac is not None and np.all(np.abs(curr - prev_frac) < float(sp["tol"])):
                            sky_done = True
                        prev_frac = curr.copy()
                elif sp["tol_mode"] == "stderr":
                    if sky_iters_done >= max(1, int(sp["min_iters"])) and sky_iters_done > 1:
                        se_bins = np.sqrt(np.maximum(M2_bins / (sky_iters_done - 1), 0.0) / sky_iters_done)
                        if sky_discrete:
                            if np.all(se_bins <= float(sp["tol"])):
                                sky_done = True
                        else:
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
            frac = counts_total.astype(np.float64) / float(sky_total_rays)
            if sky_discrete:
                sky_vf[name_e].update({f"Sky_Patch_{i+1}": float(frac[i]) for i in range(145)})
            else:
                sky_vf[name_e]["Sky"] = float(frac.sum())

        msg = (
            f"({idx_emit+1}/{len(meshes)}) [{name_e}] traced {trace_iters} iter, {total_traced_rays:,} rays -> "
            f"{time.time() - t0:0.3f}s  (scene={matrix_iters_done} iter, sky={sky_iters_done} iter, "
            f"BVH={'builtin' if use_bvh else 'off'}, device={'gpu' if use_gpu else 'cpu'})"
        )
        _log(msg)
        _rhino_log(msg)

    return vf_scene, sky_vf


def view_factor_matrix(meshes: List[Tuple[str, np.ndarray, np.ndarray]], params: MatrixParams):
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
    reciprocity = p["reciprocity"]
    enforce_reciprocity_rowsum = p["enforce_reciprocity_rowsum"]
    flip_faces = p["flip_faces"]
    return_stats = False

    have_cuda = cuda.is_available()
    dev = (device or "auto").lower()
    if dev not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be 'auto', 'gpu', or 'cpu' (got {device!r})")
    if dev == "auto":
        use_gpu = have_cuda
    elif dev == "gpu":
        if not have_cuda:
            raise RuntimeError("device='gpu' requested but CUDA is not available")
        use_gpu = True
    else:
        use_gpu = False

    use_bvh = _select_bvh(p["bvh"], int(sum(F.shape[0] for _, _, F in meshes)))
    result: Dict[str, Dict[str, float]] = {name: {} for name, _, _ in meshes}
    stats_result: Dict[str, Dict[str, float]] = {} if return_stats else None  # type: ignore[assignment]
    emitters = prepare_emitters(meshes, samples=samples, rays=rays, flip_faces=flip_faces)
    areas = [emitter.total_area for emitter in emitters] if reciprocity else None
    scene = prepare_scene(meshes, use_bvh=use_bvh)

    d_scene = None
    if use_gpu:
        d_scene = {
            "v0": cuda.to_device(scene.v0),
            "e1": cuda.to_device(scene.e1),
            "e2": cuda.to_device(scene.e2),
            "normals": cuda.to_device(scene.normals),
            "sid": cuda.to_device(scene.sid),
        }
        if scene.use_bvh:
            d_scene["bb_min"] = cuda.to_device(scene.bb_min)
            d_scene["bb_max"] = cuda.to_device(scene.bb_max)
            d_scene["left"] = cuda.to_device(scene.left)
            d_scene["right"] = cuda.to_device(scene.right)
            d_scene["start"] = cuda.to_device(scene.start)
            d_scene["count"] = cuda.to_device(scene.count)

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
            emitter_dev = _upload_emitter_data(emitter)
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
                        emitter_dev["u_grid"], emitter_dev["v_grid"],
                        emitter_dev["halton_tri"], emitter_dev["halton_u"], emitter_dev["halton_v"],
                        emitter_dev["halton_r1"], emitter_dev["halton_r2"], emitter_dev["cdf"],
                        emitter_dev["tri_a"], emitter_dev["tri_e1"], emitter_dev["tri_e2"],
                        emitter_dev["tri_u"], emitter_dev["tri_v"], emitter_dev["tri_n"],
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
                        d_orig, d_dirs, d_scene["v0"], d_scene["e1"], d_scene["e2"], d_scene["normals"], d_scene["sid"],
                        d_scene["bb_min"], d_scene["bb_max"], d_scene["left"], d_scene["right"], d_scene["start"], d_scene["count"],
                        emit_sid, min_sid, d_hit_sid, d_hit_front,
                    )
                else:
                    trace = kernel_trace_firsthit[blocks, threads, stream] if cuda_async else kernel_trace_firsthit[blocks, threads]
                    trace(d_orig, d_dirs, d_scene["v0"], d_scene["e1"], d_scene["e2"], d_scene["normals"], d_scene["sid"], emit_sid, min_sid, d_hit_sid, d_hit_front)
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

            if tol_mode == "delta":
                curr_f = hits_f / float(total_rays)
                curr_b = hits_b / float(total_rays)
                if prev_f is not None and iters_done >= max(1, min_iters):
                    if np.all(np.abs(curr_f - prev_f) < tol) and np.all(np.abs(curr_b - prev_b) < tol):
                        break
                prev_f = curr_f.copy()
                prev_b = curr_b.copy()
            elif tol_mode == "stderr":
                if iters_done >= max(1, min_iters) and iters_done > 1:
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


def view_factor(sender, receiver, params: MatrixParams):
    senders = [sender] if isinstance(sender, tuple) else list(sender)
    receivers = [receiver] if isinstance(receiver, tuple) else list(receiver)
    meshes = senders + receivers
    vf_all = view_factor_matrix(meshes, params=params)
    sender_names = [s[0] for s in senders]
    return {name: vf_all.get(name, {}) for name in sender_names}


def view_factor_to_tregenza_sky(meshes: List[Tuple[str, np.ndarray, np.ndarray]], params: SkyParams):
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
    discrete = p["discrete"]

    have_cuda = cuda.is_available()
    dev = (device or "auto").lower()
    if dev not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be 'auto', 'gpu', or 'cpu' (got {device!r})")
    if dev == "auto":
        use_gpu = have_cuda
    elif dev == "gpu":
        if not have_cuda:
            raise RuntimeError("device='gpu' requested but CUDA is not available")
        use_gpu = True
    else:
        use_gpu = False

    use_bvh = _select_bvh(p["bvh"], int(sum(F.shape[0] for _, _, F in meshes)))
    emitters = prepare_emitters(meshes, samples=samples, rays=rays, flip_faces=False)
    scene = prepare_scene(meshes, use_bvh=use_bvh)
    sky_name = "Sky"
    patch_prefix = "Sky_Patch_"
    sky_keys = [f"{patch_prefix}{i}" for i in range(1, 146)] if discrete else [sky_name]
    result: Dict[str, Dict[str, float]] = {name: {k: 0.0 for k in sky_keys} for name, _, _ in meshes}

    d_scene = None
    if use_gpu:
        d_scene = {"v0": cuda.to_device(scene.v0), "e1": cuda.to_device(scene.e1), "e2": cuda.to_device(scene.e2), "sid": cuda.to_device(scene.sid)}
        if scene.use_bvh:
            d_scene["bb_min"] = cuda.to_device(scene.bb_min)
            d_scene["bb_max"] = cuda.to_device(scene.bb_max)
            d_scene["left"] = cuda.to_device(scene.left)
            d_scene["right"] = cuda.to_device(scene.right)
            d_scene["start"] = cuda.to_device(scene.start)
            d_scene["count"] = cuda.to_device(scene.count)

    for idx_emit, (name_e, _, _) in enumerate(meshes):
        if len(meshes) <= 1:
            continue
        t0 = time.time()
        emitter = emitters[idx_emit]
        n_rays_once = emitter.n_cells * rays
        counts_total = np.zeros(145, np.int64)
        mean_bins = np.zeros(145, np.float64)
        M2_bins = np.zeros(145, np.float64)
        mean_sky = 0.0
        M2_sky = 0.0
        prev_frac = None
        total_rays = 0
        iters_done = 0

        if use_gpu:
            emitter_dev = _upload_emitter_data(emitter)
            d_orig = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_dirs = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_counts = cuda.device_array(145, dtype=np.int32)
            stream = cuda.stream() if cuda_async else None
            if cuda_async:
                h_counts = cuda.pinned_array(145, dtype=np.int32)
                if not gpu_raygen:
                    h_orig = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
                    h_dirs = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
            else:
                h_counts = None
                if not gpu_raygen:
                    h_orig, h_dirs = _host_ray_buffers(n_rays_once)
            blocks, threads = _compute_cuda_launch(n_rays_once, None)
            zblocks, zthreads = _compute_cuda_launch(145, threads)
        else:
            orig, dire = _host_ray_buffers(n_rays_once)
            hitmask = np.empty(n_rays_once, dtype=np.uint8)
            counts_iter = np.empty(145, dtype=np.int64)

        for itr in range(max_iters):
            rng = np.random.default_rng(seed + idx_emit + itr)
            cp_grid = rng.random(2, dtype=np.float32)
            cp_dims = rng.random(5, dtype=np.float32)

            if use_gpu:
                if gpu_raygen:
                    launch = kernel_build_rays[blocks, threads, stream] if cuda_async else kernel_build_rays[blocks, threads]
                    launch(
                        emitter_dev["u_grid"], emitter_dev["v_grid"],
                        emitter_dev["halton_tri"], emitter_dev["halton_u"], emitter_dev["halton_v"],
                        emitter_dev["halton_r1"], emitter_dev["halton_r2"], emitter_dev["cdf"],
                        emitter_dev["tri_a"], emitter_dev["tri_e1"], emitter_dev["tri_e2"],
                        emitter_dev["tri_u"], emitter_dev["tri_v"], emitter_dev["tri_n"],
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
                zero(d_counts)
                if scene.use_bvh:
                    trace = kernel_trace_bvh_tregenza[blocks, threads, stream] if cuda_async else kernel_trace_bvh_tregenza[blocks, threads]
                    trace(
                        d_orig, d_dirs, d_scene["v0"], d_scene["e1"], d_scene["e2"], d_scene["sid"],
                        d_scene["bb_min"], d_scene["bb_max"], d_scene["left"], d_scene["right"], d_scene["start"], d_scene["count"],
                        idx_emit, 0, d_counts,
                    )
                else:
                    trace = kernel_trace_tregenza[blocks, threads, stream] if cuda_async else kernel_trace_tregenza[blocks, threads]
                    trace(d_orig, d_dirs, d_scene["v0"], d_scene["e1"], d_scene["e2"], d_scene["sid"], idx_emit, 0, d_counts)

                if cuda_async:
                    d_counts.copy_to_host(h_counts, stream=stream)
                    stream.synchronize()
                    counts_iter_arr = np.asarray(h_counts, dtype=np.int64)
                else:
                    cuda.synchronize()
                    counts_iter_arr = d_counts.copy_to_host().astype(np.int64)
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
                bin_tregenza_cpu(dire, hitmask, counts_iter)
                counts_iter_arr = counts_iter

            counts_total += counts_iter_arr
            total_rays += n_rays_once
            iters_done += 1
            frac_iter = counts_iter_arr.astype(np.float64) / float(n_rays_once)
            sky_iter = float(frac_iter.sum())
            delta = frac_iter - mean_bins
            mean_bins += delta / iters_done
            M2_bins += delta * (frac_iter - mean_bins)
            delta_sky = sky_iter - mean_sky
            mean_sky += delta_sky / iters_done
            M2_sky += delta_sky * (sky_iter - mean_sky)

            if tol_mode == "delta":
                if iters_done >= max(1, min_iters):
                    curr = counts_total.astype(np.float64) / float(total_rays)
                    if prev_frac is not None and np.all(np.abs(curr - prev_frac) < tol):
                        break
                    prev_frac = curr.copy()
            elif tol_mode == "stderr":
                if iters_done >= max(1, min_iters) and iters_done > 1:
                    se_bins = np.sqrt(np.maximum(M2_bins / (iters_done - 1), 0.0) / iters_done)
                    if discrete:
                        if np.all(se_bins <= tol):
                            break
                    else:
                        se_sky = max(M2_sky / (iters_done - 1), 0.0) ** 0.5 / iters_done**0.5
                        if se_sky <= tol:
                            break
            else:
                raise ValueError(f"Unknown tol_mode: {tol_mode}")

        frac = counts_total.astype(np.float64) / float(max(1, total_rays))
        if discrete:
            result[name_e].update({f"{patch_prefix}{i+1}": float(frac[i]) for i in range(145)})
        else:
            result[name_e][sky_name] = float(frac.sum())

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
