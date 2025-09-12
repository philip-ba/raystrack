from __future__ import annotations
import os
import subprocess
import sys
import time
from typing import List, Tuple, Dict
import numpy as np
from numba import cuda, set_num_threads

# Try import if inside Rhino
try:
    import Rhino
except:
    pass


from .utils.halton import cached_halton
from .utils.geometry import flatten_receivers, flip_meshes
from .utils.ray_builder import build_rays
from .utils.cpu_trace import trace_cpu, trace_cpu_bvh
from .utils.cuda_trace import (
    kernel_trace,
    kernel_trace_bvh,
    kernel_zero_i64,
    kernel_build_rays,
)
from .utils.bvh import build_bvh

_LOG_PROC = None

def _open_log_console() -> None:
    """Open a new command shell window for log messages."""
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
            _LOG_PROC = subprocess.Popen(
                [term, "-hold", "-e", *helper],
                stdin=subprocess.PIPE,
                text=True,
            )
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


def _grid_from_density(area: float, density: float) -> int:
    """Return Halton grid size for a given surface area and sample density."""
    g = int(np.ceil(np.sqrt(max(area, 0.0) * density)))
    return max(g, 4)


def view_factor_matrix(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    samples: int = 256,
    rays: int = 256,
    seed: int = 0,
    gpu_threads=None,
    use_bvh: bool = False,
    flip_faces: bool = False,
    max_iters: int = 1,
    tol: float = 1e-5,
    reciprocity: bool = False,
    tol_mode: str = "delta",
    min_iters: int = 1,
    min_total_rays: int = 0,
    return_stats: bool = False,
    cuda_async: bool = True,
    gpu_raygen: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compute ``F(i→j)`` for every pair of surfaces in ``meshes``.

    Parameters
    ----------
    max_iters : int, optional
        Maximum number of Monte-Carlo iterations. ``1`` reproduces the
        previous behaviour.
    tol : float, optional
        Convergence tolerance. Interpretation depends on ``tol_mode``.
    tol_mode : {"delta", "stderr"}, optional
        - "delta": stop when successive cumulative estimates change by < tol.
        - "stderr": stop when per-iteration replicate standard error is <= tol.
    min_iters : int, optional
        Minimum number of Monte-Carlo iterations before a convergence check.
    min_total_rays : int, optional
        Minimum number of traced rays before a convergence check.
    reciprocity : bool, optional
        Also compute inverse view factors via reciprocity. Defaults to ``False``.
    return_stats : bool, optional
        When True, also return a per-entry standard error dictionary (same
        structure and keys as the result rows) computed from per-iteration
        estimates. Defaults to False.
    cuda_async : bool, optional
        Use pinned host memory and CUDA streams to overlap transfers and compute.
        Default True.
    gpu_raygen : bool, optional
        Generate rays on the GPU (avoids H2D for rays). Default False.
    """
    have_cuda = cuda.is_available()
    if not have_cuda:
        set_num_threads(1)
    result: Dict[str, Dict[str, float]] = {name: {} for name, _, _ in meshes}

    if flip_faces:
        meshes = flip_meshes(meshes)

    # Pre-compute surface areas only when reciprocity is requested
    areas = None
    if reciprocity:
        areas = []
        for _, V_a, F_a in meshes:
            A_a = 0.5 * np.linalg.norm(
                np.cross(
                    V_a[F_a[:, 1]] - V_a[F_a[:, 0]],
                    V_a[F_a[:, 2]] - V_a[F_a[:, 0]],
                ),
                axis=1,
            )
            areas.append(float(A_a.sum()))

    stats_result: Dict[str, Dict[str, float]] = {} if return_stats else None  # type: ignore[assignment]
    for idx_emit, (name_e, V_e, F_e) in enumerate(meshes):
        t_tot = time.time()

        # When reciprocity is enabled, skip receivers from previous surfaces
        # to avoid double work (we will fill them via reciprocity).
        # Otherwise, include all other surfaces (only skip the emitter itself).
        if reciprocity:
            skip = set(range(idx_emit + 1))
            receivers = [j for j in range(idx_emit + 1, len(meshes))]
        else:
            skip = set()
            receivers = [j for j in range(0, len(meshes)) if j != idx_emit]
        v0, e1, e2, sid, nrm = flatten_receivers(meshes, idx_emit, skip)
        n_surf = len(meshes)
        hits_f = np.zeros(n_surf, np.int64)
        hits_b = np.zeros_like(hits_f)
        # Running statistics for stderr mode (Welford). Always track to allow
        # returning stats even when tol_mode="delta".
        mean_f = np.zeros(n_surf, np.float64)
        mean_b = np.zeros(n_surf, np.float64)
        M2_f = np.zeros(n_surf, np.float64)
        M2_b = np.zeros(n_surf, np.float64)

        if len(v0) == 0:
            _log(
                f"({idx_emit+1}/{len(meshes)}) [{name_e}] 0 iter, 0 rays -> 0.000s  (BVH={'on' if use_bvh else 'off'})"
            )
            continue

        if use_bvh:
            bb_min, bb_max, left, right, start, cnt, perm = build_bvh(v0, e1, e2)
            v0 = v0[perm]
            e1 = e1[perm]
            e2 = e2[perm]
            sid = sid[perm]
            nrm = nrm[perm]
        else:
            bb_min = bb_max = left = right = start = cnt = None

        A = 0.5 * np.linalg.norm(
            np.cross(V_e[F_e[:, 1]] - V_e[F_e[:, 0]], V_e[F_e[:, 2]] - V_e[F_e[:, 0]]),
            axis=1,
        )
        cdf = np.cumsum(A) / A.sum()

        area = float(A.sum())
        g = _grid_from_density(area, samples)
        u_grid, v_grid = cached_halton(g)

        cells = g * g
        prev_f = prev_b = None
        total_rays = 0
        iters_done = 0

        # Pre-allocate persistent device and (optionally) pinned host buffers
        if have_cuda:
            d_v0 = cuda.to_device(v0)
            d_e1 = cuda.to_device(e1)
            d_e2 = cuda.to_device(e2)
            d_nrm = cuda.to_device(nrm)
            d_sid = cuda.to_device(sid)
            if use_bvh:
                d_bbmin = cuda.to_device(bb_min)
                d_bbmax = cuda.to_device(bb_max)
                d_left = cuda.to_device(left)
                d_right = cuda.to_device(right)
                d_start = cuda.to_device(start)
                d_cnt = cuda.to_device(cnt)
            n_rays_once = cells * rays
            d_orig = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_dirs = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_hf = cuda.device_array(hits_f.shape, dtype=hits_f.dtype)
            d_hb = cuda.device_array(hits_b.shape, dtype=hits_b.dtype)
            stream = cuda.stream() if cuda_async else None
            if cuda_async:
                h_orig = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
                h_dirs = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
                h_hf = cuda.pinned_array(hits_f.shape, dtype=hits_f.dtype)
                h_hb = cuda.pinned_array(hits_b.shape, dtype=hits_b.dtype)
            if gpu_raygen:
                # Upload constant per-emitter data to device for on-GPU raygen
                d_u_grid = cuda.to_device(u_grid)
                d_v_grid = cuda.to_device(v_grid)
                d_cdf = cuda.to_device(cdf.astype(np.float32))
                d_tri_a = cuda.to_device(V_e[F_e[:, 0]])
                d_tri_b = cuda.to_device(V_e[F_e[:, 1]])
                d_tri_c = cuda.to_device(V_e[F_e[:, 2]])

        for itr in range(max_iters):
            n_rays = cells * rays
            if have_cuda and cuda_async:
                orig = h_orig
                dire = h_dirs
            else:
                orig = np.empty((n_rays, 3), np.float32)
                dire = np.empty_like(orig)

            # Cranley–Patterson shifts for randomized QMC
            rng = np.random.default_rng(seed + idx_emit + itr)
            cp_grid = rng.random(2, dtype=np.float32)
            cp_dims = rng.random(5, dtype=np.float32)

            if have_cuda and gpu_raygen:
                threads_gen = gpu_threads or 256
                threads_gen = min(threads_gen, cuda.get_current_device().MAX_THREADS_PER_BLOCK)
                blocks_gen = (n_rays + threads_gen - 1) // threads_gen
                if cuda_async:
                    kernel_build_rays[blocks_gen, threads_gen, stream](
                        d_u_grid,
                        d_v_grid,
                        d_cdf,
                        d_tri_a,
                        d_tri_b,
                        d_tri_c,
                        g,
                        rays,
                        d_orig,
                        d_dirs,
                        cp_grid,
                        cp_dims,
                    )
                else:
                    kernel_build_rays[blocks_gen, threads_gen](
                        d_u_grid,
                        d_v_grid,
                        d_cdf,
                        d_tri_a,
                        d_tri_b,
                        d_tri_c,
                        g,
                        rays,
                        d_orig,
                        d_dirs,
                        cp_grid,
                        cp_dims,
                    )
            else:
                build_rays(
                    u_grid,
                    v_grid,
                    cdf.astype(np.float32),
                    V_e[F_e[:, 0]],
                    V_e[F_e[:, 1]],
                    V_e[F_e[:, 2]],
                    g,
                    rays,
                    orig,
                    dire,
                    cp_grid,
                    cp_dims,
                )

            if have_cuda and cuda_async:
                hits_f_iter = h_hf
                hits_b_iter = h_hb
            else:
                hits_f_iter = np.zeros_like(hits_f)
                hits_b_iter = np.zeros_like(hits_b)

            if have_cuda:
                # Copy rays into persistent device arrays
                if not gpu_raygen:
                    if cuda_async:
                        d_orig.copy_to_device(orig, stream=stream)
                        d_dirs.copy_to_device(dire, stream=stream)
                    else:
                        d_orig.copy_to_device(orig)
                        d_dirs.copy_to_device(dire)

                threads = gpu_threads or 256
                threads = min(threads, cuda.get_current_device().MAX_THREADS_PER_BLOCK)
                blocks = (n_rays + threads - 1) // threads
                # Zero device hit buffers
                zblocks = (hits_f_iter.size + threads - 1) // threads
                if cuda_async:
                    kernel_zero_i64[zblocks, threads, stream](d_hf)
                    kernel_zero_i64[zblocks, threads, stream](d_hb)
                else:
                    kernel_zero_i64[zblocks, threads](d_hf)
                    kernel_zero_i64[zblocks, threads](d_hb)

                if use_bvh:
                    if cuda_async:
                        kernel_trace_bvh[blocks, threads, stream](
                            d_orig,
                            d_dirs,
                            d_v0,
                            d_e1,
                            d_e2,
                            d_nrm,
                            d_sid,
                            d_bbmin,
                            d_bbmax,
                            d_left,
                            d_right,
                            d_start,
                            d_cnt,
                            d_hf,
                            d_hb,
                        )
                    else:
                        kernel_trace_bvh[blocks, threads](
                            d_orig,
                            d_dirs,
                            d_v0,
                            d_e1,
                            d_e2,
                            d_nrm,
                            d_sid,
                            d_bbmin,
                            d_bbmax,
                            d_left,
                            d_right,
                            d_start,
                            d_cnt,
                            d_hf,
                            d_hb,
                        )
                else:
                    if cuda_async:
                        kernel_trace[blocks, threads, stream](
                            d_orig, d_dirs, d_v0, d_e1, d_e2, d_nrm, d_sid, d_hf, d_hb
                        )
                    else:
                        kernel_trace[blocks, threads](
                            d_orig, d_dirs, d_v0, d_e1, d_e2, d_nrm, d_sid, d_hf, d_hb
                        )
                if cuda_async:
                    d_hf.copy_to_host(hits_f_iter, stream=stream)
                    d_hb.copy_to_host(hits_b_iter, stream=stream)
                    stream.synchronize()
                else:
                    cuda.synchronize()
                    hits_f_iter = d_hf.copy_to_host()
                    hits_b_iter = d_hb.copy_to_host()
            else:
                if use_bvh:
                    trace_cpu_bvh(
                        orig,
                        dire,
                        v0,
                        e1,
                        e2,
                        nrm,
                        sid,
                        bb_min,
                        bb_max,
                        left,
                        right,
                        start,
                        cnt,
                        hits_f_iter,
                        hits_b_iter,
                    )
                else:
                    trace_cpu(orig, dire, v0, e1, e2, nrm, sid, hits_f_iter, hits_b_iter)

            # Accumulate totals
            hits_f += hits_f_iter
            hits_b += hits_b_iter
            total_rays += n_rays
            iters_done += 1

            # Per-iteration estimates for stderr tracking
            f_iter = hits_f_iter.astype(np.float64) / float(n_rays)
            b_iter = hits_b_iter.astype(np.float64) / float(n_rays)

            # Welford updates
            delta_f = f_iter - mean_f
            mean_f += delta_f / iters_done
            M2_f += delta_f * (f_iter - mean_f)

            delta_b = b_iter - mean_b
            mean_b += delta_b / iters_done
            M2_b += delta_b * (b_iter - mean_b)

            # Convergence checks
            if tol_mode == "delta":
                curr_f = hits_f / float(total_rays)
                curr_b = hits_b / float(total_rays)
                if prev_f is not None:
                    if (
                        iters_done >= max(1, min_iters)
                        and total_rays >= max(0, min_total_rays)
                        and np.all(np.abs(curr_f - prev_f) < tol)
                        and np.all(np.abs(curr_b - prev_b) < tol)
                    ):
                        break
                prev_f = curr_f.copy()
                prev_b = curr_b.copy()
            elif tol_mode == "stderr":
                # Only check after min_iters and min_total_rays
                if iters_done >= max(1, min_iters) and total_rays >= max(0, min_total_rays):
                    if iters_done > 1:
                        var_f = M2_f / (iters_done - 1)
                        var_b = M2_b / (iters_done - 1)
                        se_f = np.sqrt(np.maximum(var_f, 0.0) / iters_done)
                        se_b = np.sqrt(np.maximum(var_b, 0.0) / iters_done)
                        # Check only receiver indices
                        if reciprocity:
                            recv_idx = np.array(receivers, dtype=np.int32)
                        else:
                            recv_idx = np.array(receivers, dtype=np.int32)
                        if (
                            np.all(se_f[recv_idx] <= tol)
                            and np.all(se_b[recv_idx] <= tol)
                        ):
                            break
            else:
                raise ValueError(f"Unknown tol_mode: {tol_mode}")

        row = {}
        stats_row = {}  # if requested, holds stderr values with same keys
        # Compute final stderr values from running stats
        if iters_done > 1:
            var_f = M2_f / (iters_done - 1)
            var_b = M2_b / (iters_done - 1)
            se_f = np.sqrt(np.maximum(var_f, 0.0) / iters_done)
            se_b = np.sqrt(np.maximum(var_b, 0.0) / iters_done)
        else:
            # Undefined with one iteration — report inf
            se_f = np.full_like(mean_f, np.inf, dtype=np.float64)
            se_b = np.full_like(mean_b, np.inf, dtype=np.float64)

        for j in receivers:
            name_r, _, _ = meshes[j]
            f = hits_f[j] / float(total_rays)
            b = hits_b[j] / float(total_rays)
            if f > 0:
                row[f"{name_r}_front"] = f
                if reciprocity:
                    result[name_r][f"{name_e}_front"] = f * (areas[idx_emit] / areas[j])
                if return_stats:
                    stats_row[f"{name_r}_front"] = float(se_f[j])
            if b > 0:
                row[f"{name_r}_back"] = b
                if return_stats:
                    stats_row[f"{name_r}_back"] = float(se_b[j])
        result[name_e].update(row)
        if return_stats:
            stats_result[name_e] = stats_row


        surf_total   = len(meshes)
        iter_count   = itr + 1             # how many Monte-Carlo iterations ran
        elapsed      = time.time() - t_tot

        msg = (f"({idx_emit+1}/{surf_total}) [{name_e}] "
               f"{iter_count} iter, {total_rays:,} rays -> "
               f"{elapsed:0.3f}s  (BVH={'on' if use_bvh else 'off'})")

        _log(msg)                          # console / VS Code
        try:
            import Rhino
            Rhino.RhinoApp.WriteLine(msg)  # live in Rhino / Grasshopper
        except Exception:
            pass

    # If the caller asked for stats, return the collected per-emitter rows.
    if return_stats:
        return result, stats_result or {}
    return result


def view_factor_matrix_brute(*args, **kw):
    kw["use_bvh"] = False
    return view_factor_matrix(*args, **kw)

def view_factor(sender, receiver, *args, reciprocity: bool = False, **kw):
    """Return F(sender→receiver) using the same Monte-Carlo algorithm.

    Parameters
    ----------
    sender : tuple or list[tuple]
        A single mesh tuple ``(name, V, F)`` or a list of such tuples
        describing the emitting surface(s).
    receiver : tuple or list[tuple]
        A single mesh tuple or a list of tuples describing the receiving
        surface(s).
    *args, **kw :
        Additional arguments forwarded to :func:`view_factor_matrix`.
    max_iters : int, optional
        Maximum number of Monte-Carlo iterations. Defaults to ``1``.
    tol : float, optional
        Stop iterating when all view factors change less than ``tol``.
    reciprocity : bool, optional
        Also compute inverse view factors via reciprocity. Defaults to ``False``.

    Returns
    -------
    dict
        A dictionary equivalent to the corresponding rows of
        :func:`view_factor_matrix`. By default only the sender entries are
        included; setting ``reciprocity=True`` adds entries for the receiver
        surfaces computed via reciprocity.
    """

    senders = [sender] if isinstance(sender, tuple) else list(sender)
    receivers = [receiver] if isinstance(receiver, tuple) else list(receiver)

    # ------------------------------------------------------------------
    # Gather parameters identical to :func:`view_factor_matrix`
    # ------------------------------------------------------------------
    samples = kw.get("samples", 256)
    rays = kw.get("rays", 256)
    seed = kw.get("seed", 0)
    gpu_threads = kw.get("gpu_threads", None)
    use_bvh = kw.get("use_bvh", False)
    flip_faces = kw.get("flip_faces", False)
    max_iters = kw.get("max_iters", 1)
    tol = kw.get("tol", 1e-5)
    tol_mode = kw.get("tol_mode", "delta")
    min_iters = kw.get("min_iters", 1)
    min_total_rays = kw.get("min_total_rays", 0)
    return_stats = kw.get("return_stats", False)

    meshes = senders + receivers
    have_cuda = cuda.is_available()
    if not have_cuda:
        set_num_threads(1)

    if flip_faces:
        meshes = flip_meshes(meshes)

    if reciprocity:
        result: Dict[str, Dict[str, float]] = {name: {} for name, _, _ in meshes}
        areas = []
        for _, V_a, F_a in meshes:
            A_a = 0.5 * np.linalg.norm(
                np.cross(
                    V_a[F_a[:, 1]] - V_a[F_a[:, 0]],
                    V_a[F_a[:, 2]] - V_a[F_a[:, 0]],
                ),
                axis=1,
            )
            areas.append(float(A_a.sum()))
    else:
        result: Dict[str, Dict[str, float]] = {}
        areas = None

    stats_result: Dict[str, Dict[str, float]] = {} if return_stats else None  # type: ignore[assignment]
    for idx_emit, (name_e, V_e, F_e) in enumerate(meshes[: len(senders)]):
        t_tot = time.time()

        # Only consider receivers, not other senders, to avoid wasted tracing.
        # flatten_receivers always excludes the emitter itself.
        skip_senders = set(range(len(senders)))
        v0, e1, e2, sid, nrm = flatten_receivers(meshes, idx_emit, skip_senders)
        n_surf = len(meshes)
        hits_f = np.zeros(n_surf, np.int64)
        hits_b = np.zeros_like(hits_f)
        mean_f = np.zeros(n_surf, np.float64)
        mean_b = np.zeros(n_surf, np.float64)
        M2_f = np.zeros(n_surf, np.float64)
        M2_b = np.zeros(n_surf, np.float64)

        if use_bvh:
            bb_min, bb_max, left, right, start, cnt, perm = build_bvh(v0, e1, e2)
            v0 = v0[perm]
            e1 = e1[perm]
            e2 = e2[perm]
            sid = sid[perm]
            nrm = nrm[perm]
        else:
            bb_min = bb_max = left = right = start = cnt = None

        A = 0.5 * np.linalg.norm(
            np.cross(V_e[F_e[:, 1]] - V_e[F_e[:, 0]], V_e[F_e[:, 2]] - V_e[F_e[:, 0]]),
            axis=1,
        )
        cdf = np.cumsum(A) / A.sum()

        area = float(A.sum())
        g = _grid_from_density(area, samples)
        u_grid, v_grid = cached_halton(g)

        cells = g * g
        prev_f = prev_b = None
        total_rays = 0
        iters_done = 0

        for itr in range(max_iters):
            n_rays = cells * rays
            orig = np.empty((n_rays, 3), np.float32)
            dire = np.empty_like(orig)
            rng = np.random.default_rng(seed + idx_emit + itr)
            cp_grid = rng.random(2, dtype=np.float32)
            cp_dims = rng.random(5, dtype=np.float32)

            build_rays(
                u_grid,
                v_grid,
                cdf.astype(np.float32),
                V_e[F_e[:, 0]],
                V_e[F_e[:, 1]],
                V_e[F_e[:, 2]],
                g,
                rays,
                orig,
                dire,
                cp_grid,
                cp_dims,
            )

            hits_f_iter = np.zeros_like(hits_f)
            hits_b_iter = np.zeros_like(hits_b)

            if have_cuda:
                d_orig.copy_to_device(orig)
                d_dirs.copy_to_device(dire)

                threads = gpu_threads or 256
                threads = min(threads, cuda.get_current_device().MAX_THREADS_PER_BLOCK)
                blocks = (n_rays + threads - 1) // threads
                zblocks = (hits_f_iter.size + threads - 1) // threads
                kernel_zero_i64[zblocks, threads](d_hf)
                kernel_zero_i64[zblocks, threads](d_hb)

                if use_bvh:
                    kernel_trace_bvh[
                        blocks,
                        threads,
                    ](
                        d_orig,
                        d_dirs,
                        d_v0,
                        d_e1,
                        d_e2,
                        d_nrm,
                        d_sid,
                        d_bbmin,
                        d_bbmax,
                        d_left,
                        d_right,
                        d_start,
                        d_cnt,
                        d_hf,
                        d_hb,
                    )
                else:
                    kernel_trace[
                        blocks,
                        threads,
                    ](d_orig, d_dirs, d_v0, d_e1, d_e2, d_nrm, d_sid, d_hf, d_hb)
                cuda.synchronize()
                hits_f_iter = d_hf.copy_to_host()
                hits_b_iter = d_hb.copy_to_host()
            else:
                if use_bvh:
                    trace_cpu_bvh(
                        orig,
                        dire,
                        v0,
                        e1,
                        e2,
                        nrm,
                        sid,
                        bb_min,
                        bb_max,
                        left,
                        right,
                        start,
                        cnt,
                        hits_f_iter,
                        hits_b_iter,
                    )
                else:
                    trace_cpu(orig, dire, v0, e1, e2, nrm, sid, hits_f_iter, hits_b_iter)

            hits_f += hits_f_iter
            hits_b += hits_b_iter
            total_rays += n_rays
            iters_done += 1

            # Per-iteration estimates and running stats
            f_iter = hits_f_iter.astype(np.float64) / float(n_rays)
            b_iter = hits_b_iter.astype(np.float64) / float(n_rays)
            delta_f = f_iter - mean_f
            mean_f += delta_f / iters_done
            M2_f += delta_f * (f_iter - mean_f)
            delta_b = b_iter - mean_b
            mean_b += delta_b / iters_done
            M2_b += delta_b * (b_iter - mean_b)

            if tol_mode == "delta":
                curr_f = hits_f / float(total_rays)
                curr_b = hits_b / float(total_rays)
                if prev_f is not None:
                    if (
                        iters_done >= max(1, min_iters)
                        and total_rays >= max(0, min_total_rays)
                        and np.all(np.abs(curr_f - prev_f) < tol)
                        and np.all(np.abs(curr_b - prev_b) < tol)
                    ):
                        break
                prev_f = curr_f.copy()
                prev_b = curr_b.copy()
            elif tol_mode == "stderr":
                if iters_done >= max(1, min_iters) and total_rays >= max(0, min_total_rays):
                    if iters_done > 1:
                        var_f = M2_f / (iters_done - 1)
                        var_b = M2_b / (iters_done - 1)
                        se_f = np.sqrt(np.maximum(var_f, 0.0) / iters_done)
                        se_b = np.sqrt(np.maximum(var_b, 0.0) / iters_done)
                        # receiver indices are j >= len(senders) and j != idx_emit
                        recv_idx = np.array([j for j in range(len(senders), n_surf) if j != idx_emit], dtype=np.int32)
                        if (
                            np.all(se_f[recv_idx] <= tol)
                            and np.all(se_b[recv_idx] <= tol)
                        ):
                            break
            else:
                raise ValueError(f"Unknown tol_mode: {tol_mode}")

        row = {}
        stats_row = {}
        if iters_done > 1:
            var_f = M2_f / (iters_done - 1)
            var_b = M2_b / (iters_done - 1)
            se_f = np.sqrt(np.maximum(var_f, 0.0) / iters_done)
            se_b = np.sqrt(np.maximum(var_b, 0.0) / iters_done)
        else:
            se_f = np.full(n_surf, np.inf, np.float64)
            se_b = np.full(n_surf, np.inf, np.float64)

        for j, (name_r, _, _) in enumerate(meshes):
            if j == idx_emit or j < len(senders):
                # Skip self and other emitters
                continue
            f = hits_f[j] / float(total_rays)
            b = hits_b[j] / float(total_rays)
            if f > 0:
                row[f"{name_r}_front"] = f
                if reciprocity:
                    result[name_r][f"{name_e}_front"] = f * (areas[idx_emit] / areas[j])
                if return_stats:
                    stats_row[f"{name_r}_front"] = float(se_f[j])
            if b > 0:
                row[f"{name_r}_back"] = b
                if return_stats:
                    stats_row[f"{name_r}_back"] = float(se_b[j])

        result[name_e] = row
        if return_stats:
            stats_result[name_e] = stats_row
        _log(
            f"[{name_e}] total {time.time() - t_tot:.3f}s  (BVH={'on' if use_bvh else 'off'})"
        )

        # Logging in Rhino
        try:
            Rhino.RhinoApp.WriteLine(
                    f"[{name_e}] total {time.time() - t_tot:.3f}s  (BVH={'on' if use_bvh else 'off'})"
                )
        except:
            pass

    if return_stats:
        return result, stats_result or {}
    return result

__all__ = ["view_factor_matrix", "view_factor_matrix_brute", "view_factor"]
