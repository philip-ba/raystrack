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


# Be resilient to namespace-package edge cases: prefer utils __init__ export,
# but fall back to explicit submodule import if needed.
try:  # pragma: no cover - import robustness
    from .utils import cached_halton  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    from .utils.halton import cached_halton
from .utils.geometry import flatten_receivers, flip_meshes
from .utils.ray_builder import build_rays
from .utils.cpu_trace import (
    trace_cpu,
    trace_cpu_bvh,
    trace_cpu_hitmask,
    trace_cpu_bvh_hitmask,
)
from .utils.cuda_trace import (
    kernel_trace,
    kernel_trace_bvh,
    kernel_zero_i64,
    kernel_zero_i32,
    kernel_zero_u8,
    kernel_build_rays,
    kernel_trace_hitmask,
    kernel_trace_bvh_hitmask,
    kernel_bin_tregenza,
)
from .utils.bvh import build_bvh
from .utils.helpers import (
    grid_from_density as _grid_from_density,
    enforce_reciprocity_and_rowsum as _enforce_reciprocity_and_rowsum,
)

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


"""Helper functions moved to raystrack.utils.helpers"""


def view_factor_matrix(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    samples: int = 256,
    rays: int = 256,
    seed: int = 0,
    gpu_threads=None,
    bvh: str = "auto",
    device: str = "auto",
    flip_faces: bool = False,
    max_iters: int = 1,
    tol: float = 1e-5,
    reciprocity: bool = False,
    tol_mode: str = "stderr",
    min_iters: int = 1,
    min_total_rays: int = 0,
    return_stats: bool = False,
    cuda_async: bool = True,
    gpu_raygen: bool = False,
    enforce_reciprocity_rowsum: bool = False,
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
    enforce_reciprocity_rowsum : bool, optional
        After computation, enforce reciprocity and make each row sum to 1 using
        symmetric diagonal scaling. Default False.
    """
    # Device and BVH selection
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
        set_num_threads(1)
    result: Dict[str, Dict[str, float]] = {name: {} for name, _, _ in meshes}

    # Note: flip only the current emitter during emission sampling.
    # Receivers remain unmodified so their normals/front-back stay consistent.

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

    # Decide BVH mode
    if isinstance(bvh, str):
        bvh_mode = bvh.lower()
    else:
        bvh_mode = "auto"
    if bvh_mode not in ("auto", "off", "builtin"):
        raise ValueError(f"bvh must be 'auto', 'off', or 'builtin' (got {bvh!r})")

    total_faces = int(sum(F.shape[0] for _, _, F in meshes))
    BVH_AUTO_THRESHOLD = 512
    use_bvh_flag = (
        True if bvh_mode == "builtin"
        else False if bvh_mode == "off"
        else total_faces >= BVH_AUTO_THRESHOLD
    )

    stats_result: Dict[str, Dict[str, float]] = {} if return_stats else None  # type: ignore[assignment]
    for idx_emit, (name_e, V_e, F_e) in enumerate(meshes):
        t_tot = time.time()

        # When reciprocity is enabled, skip receivers from previous surfaces
        # to avoid double work (we will fill them via reciprocity).
        # Otherwise, include all other surfaces (only skip the emitter itself).
        if reciprocity:
            # Skip all previous surfaces including the emitter; fill via reciprocity later
            skip = set(range(idx_emit + 1))
            receivers = [j for j in range(idx_emit + 1, len(meshes))]
        else:
            # Skip only the emitter itself; include all other surfaces
            skip = {idx_emit}
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
                f"({idx_emit+1}/{len(meshes)}) [{name_e}] 0 iter, 0 rays -> 0.000s  (BVH={'builtin' if use_bvh_flag else 'off'}, device={'gpu' if use_gpu else 'cpu'})"
            )
            continue

        if use_bvh_flag:
            bb_min, bb_max, left, right, start, cnt, perm = build_bvh(v0, e1, e2)
            v0 = v0[perm]
            e1 = e1[perm]
            e2 = e2[perm]
            sid = sid[perm]
            nrm = nrm[perm]
        else:
            bb_min = bb_max = left = right = start = cnt = None

        # Use flipped winding only for the emitter if requested
        F_emit = F_e[:, [0, 2, 1]] if flip_faces else F_e

        A = 0.5 * np.linalg.norm(
            np.cross(V_e[F_emit[:, 1]] - V_e[F_emit[:, 0]], V_e[F_emit[:, 2]] - V_e[F_emit[:, 0]]),
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
        if use_gpu:
            d_v0 = cuda.to_device(v0)
            d_e1 = cuda.to_device(e1)
            d_e2 = cuda.to_device(e2)
            d_nrm = cuda.to_device(nrm)
            d_sid = cuda.to_device(sid)
            if use_bvh_flag:
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
                d_tri_a = cuda.to_device(V_e[F_emit[:, 0]])
                d_tri_b = cuda.to_device(V_e[F_emit[:, 1]])
                d_tri_c = cuda.to_device(V_e[F_emit[:, 2]])

        for itr in range(max_iters):
            n_rays = cells * rays
            if use_gpu and cuda_async:
                orig = h_orig
                dire = h_dirs
            else:
                orig = np.empty((n_rays, 3), np.float32)
                dire = np.empty_like(orig)

            # Cranley–Patterson shifts for randomized QMC
            rng = np.random.default_rng(seed + idx_emit + itr)
            cp_grid = rng.random(2, dtype=np.float32)
            cp_dims = rng.random(5, dtype=np.float32)

            if use_gpu and gpu_raygen:
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
                    V_e[F_emit[:, 0]],
                    V_e[F_emit[:, 1]],
                    V_e[F_emit[:, 2]],
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

            if use_gpu:
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

                if use_bvh_flag:
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
                if use_bvh_flag:
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
               f"{elapsed:0.3f}s  (BVH={'builtin' if use_bvh_flag else 'off'}, device={'gpu' if use_gpu else 'cpu'})")

        _log(msg)                          # console / VS Code
        try:
            import Rhino
            Rhino.RhinoApp.WriteLine(msg)  # live in Rhino / Grasshopper
        except Exception:
            pass

    # Optional post-processing: enforce reciprocity and row-sum unity on totals
    if enforce_reciprocity_rowsum:
        _enforce_reciprocity_and_rowsum(result, meshes, areas)

    # If the caller asked for stats, return the collected per-emitter rows.
    if return_stats:
        return result, stats_result or {}
    return result


def view_factor(sender, receiver, *args, reciprocity: bool = False, **kw):
    """Return F(sender->receiver) using the same Monte-Carlo algorithm.

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

    # Delegated implementation to keep parity with view_factor_matrix
    senders = [sender] if isinstance(sender, tuple) else list(sender)
    receivers = [receiver] if isinstance(receiver, tuple) else list(receiver)
    meshes = senders + receivers
    vf_all = view_factor_matrix(
        meshes,
        samples=kw.get("samples", 256),
        rays=kw.get("rays", 256),
        seed=kw.get("seed", 0),
        gpu_threads=kw.get("gpu_threads", None),
        bvh=kw.get("bvh", "auto"),
        device=kw.get("device", "auto"),
        flip_faces=kw.get("flip_faces", False),
        max_iters=kw.get("max_iters", 1),
        tol=kw.get("tol", 1e-5),
        reciprocity=reciprocity,
        tol_mode=kw.get("tol_mode", "delta"),
        min_iters=kw.get("min_iters", 1),
        min_total_rays=kw.get("min_total_rays", 0),
        return_stats=False,
        cuda_async=kw.get("cuda_async", True),
        gpu_raygen=kw.get("gpu_raygen", False),
        enforce_reciprocity_rowsum=kw.get("enforce_reciprocity_rowsum", False),
    )
    sender_names = [s[0] for s in senders]
    out = {name: vf_all.get(name, {}) for name in sender_names}
    return out


def view_factor_to_tregenza_sky(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    samples: int = 256,
    rays: int = 256,
    seed: int = 0,
    device: str = "auto",
    bvh: str = "auto",
    gpu_threads=None,
    cuda_async: bool = True,
    gpu_raygen: bool = False,
    flip_faces: bool = False,
    discrete: bool = False,
    sky_name: str = "Sky",
    patch_prefix: str = "Sky_Patch_",
    # Monte Carlo control
    max_iters: int = 1,
    tol: float = 1e-5,
    tol_mode: str = "stderr",
    min_iters: int = 1,
    min_total_rays: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Directional Tregenza sky view factors (no finite sky geometry).

    For each emitter triangle, cast cosine-weighted rays. A ray is classified
    as "sky" if it misses all scene geometry and has global z > 0. Sky-visible
    rays are accumulated into the 145 Tregenza angular bins.

    Parameters
    ----------
    meshes : list[(str, V, F)]
        Scene meshes. ``V`` is ``(N,3)`` float32/float64, ``F`` is ``(M,3)`` int.
    samples : int
        Quasi-Monte Carlo grid per side used for emission per emitter.
    rays : int
        Rays per grid cell.
    seed : int
        Base RNG seed. Each emitter/iteration derives its own sub-seed.
    device : {"auto","gpu","cpu"}
        Visibility device. GPU uses CUDA hitmask kernels. CPU uses numba njit.
    bvh : {"auto","off","builtin"}
        Toggle a built-in BVH for faster visibility.
    gpu_threads : int, optional
        CUDA threads per block (leave None for a reasonable default).
    cuda_async : bool
        Use pinned memory + streams for overlap on CUDA.
    gpu_raygen : bool
        Generate rays on GPU to avoid H2D copies of rays.
    flip_faces : bool
        Flip emitter triangle winding during emission sampling.
    discrete : bool
        If False, return one merged key ``sky_name`` per emitter. If True,
        return 145 entries named ``f"{patch_prefix}{i}"``.
    sky_name : str
        Name used for the merged sky entry when ``discrete=False``.
    patch_prefix : str
        Prefix for discrete patch names when ``discrete=True``.
    max_iters, tol, tol_mode, min_iters, min_total_rays
        Convergence controls on sky fractions. ``tol_mode`` accepts
        "delta" or "stderr" (true replicate standard error).

    Returns
    -------
    dict[str, dict[str, float]]
        Per-emitter mapping to either a single merged sky entry or 145 patches.
    """
    # Directional Tregenza classification (no finite sky geometry)
    if len(meshes) == 0:
        raise ValueError("meshes must not be empty")

    # Tregenza rings and azimuths
    rings = [
        (0.0, 12.0, 30),
        (12.0, 24.0, 30),
        (24.0, 36.0, 24),
        (36.0, 48.0, 24),
        (48.0, 60.0, 18),
        (60.0, 72.0, 12),
        (72.0, 84.0, 6),
        (84.0, 90.0, 1),
    ]
    ring_start = []
    s = 0
    for _, _, n in rings:
        ring_start.append(s)
        s += n

    def ring_offset(n_az: int, r_idx: int) -> float:
        if n_az <= 1:
            return 0.0
        return 180.0 / n_az if (r_idx % 2 == 1) else 0.0

    def dir_to_patch_id(d: np.ndarray) -> int:
        # Global up is +Z; horizon at z=0
        z = float(d[2])
        if z <= 0.0:
            return -1  # below horizon -> no sky
        # elevation from horizon in degrees
        el = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
        # azimuth 0..360, X=0°, Y=90°
        az = (np.rad2deg(np.arctan2(d[1], d[0])) + 360.0) % 360.0
        # Find ring
        ridx = -1
        for j, (e0, e1, n_az) in enumerate(rings):
            if el < e1 or (j == len(rings) - 1 and abs(el - e1) < 1e-6):
                ridx = j
                break
        if ridx < 0:
            return -1
        e0, e1, n_az = rings[ridx]
        base = ring_start[ridx]
        if n_az == 1:
            return base  # zenith cap
        w = 360.0 / n_az
        off = ring_offset(n_az, ridx)
        t = (az - off) % 360.0
        aidx = int(t // w)
        return base + aidx

    # Prepare outputs
    if discrete:
        sky_keys = [f"{patch_prefix}{i}" for i in range(1, 145)]
    else:
        sky_keys = [sky_name]
    result: Dict[str, Dict[str, float]] = {name: {k: 0.0 for k in sky_keys} for name, _, _ in meshes}

    # Build receivers (scene excluding emitter) once per emitter and trace hit masks
    have_cuda = cuda.is_available()
    dev = (device or "auto").lower()
    if dev not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be 'auto', 'gpu', or 'cpu' (got {device!r})")
    use_gpu = (dev == "gpu") or (dev == "auto" and have_cuda)
    use_bvh = True if (bvh or "auto").lower() != "off" else False

    for idx_emit, (name_e, V_e, F_e) in enumerate(meshes):
        t0 = time.time()
        # Receivers: all other meshes
        skip = {idx_emit}
        v0, e1, e2, sid, nrm = flatten_receivers(meshes, idx_emit, skip)
        if v0.shape[0] == 0:
            continue
        if use_bvh:
            bb_min, bb_max, left, right, start, cnt, perm = build_bvh(v0, e1, e2)
            v0 = v0[perm]; e1 = e1[perm]; e2 = e2[perm]; sid = sid[perm]; nrm = nrm[perm]
        else:
            bb_min = bb_max = left = right = start = cnt = None

        # Emission sampling from emitter triangles
        F_emit = F_e[:, [0, 2, 1]] if flip_faces else F_e
        A = 0.5 * np.linalg.norm(
            np.cross(V_e[F_emit[:, 1]] - V_e[F_emit[:, 0]], V_e[F_emit[:, 2]] - V_e[F_emit[:, 0]]),
            axis=1,
        )
        cdf = np.cumsum(A) / A.sum()
        area = float(A.sum())
        g = _grid_from_density(area, samples)
        u_grid, v_grid = cached_halton(g)
        n_rays_once = g * g * rays

        counts_total = np.zeros(145, np.int64)
        prev_frac = None
        total_rays = 0
        # Welford running stats for stderr convergence
        iters_done = 0
        mean_bins = np.zeros(145, np.float64)
        M2_bins = np.zeros(145, np.float64)
        mean_sky = 0.0
        M2_sky = 0.0

        # Optional GPU persistent buffers
        if use_gpu:
            d_v0 = cuda.to_device(v0)
            d_e1 = cuda.to_device(e1)
            d_e2 = cuda.to_device(e2)
            d_nrm = cuda.to_device(nrm)
            d_sid = cuda.to_device(sid)
            if use_bvh:
                d_bbmin = cuda.to_device(bb_min)
                d_bbmax = cuda.to_device(bb_max)
                d_left  = cuda.to_device(left)
                d_right = cuda.to_device(right)
                d_start = cuda.to_device(start)
                d_cnt   = cuda.to_device(cnt)
            d_orig = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_dirs = cuda.device_array((n_rays_once, 3), dtype=np.float32)
            d_hit = cuda.device_array(n_rays_once, dtype=np.uint8)
            d_counts = cuda.device_array(145, dtype=np.int32)
            stream = cuda.stream() if cuda_async else None
            if cuda_async:
                h_hit = cuda.pinned_array(n_rays_once, dtype=np.uint8)
                h_orig = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
                h_dirs = cuda.pinned_array((n_rays_once, 3), dtype=np.float32)
                h_counts = cuda.pinned_array(145, dtype=np.int32)

            d_u = d_v = d_cdf = d_tri_a = d_tri_b = d_tri_c = None
            if gpu_raygen:
                d_u = cuda.to_device(u_grid)
                d_v = cuda.to_device(v_grid)
                d_cdf = cuda.to_device(cdf.astype(np.float32))
                d_tri_a = cuda.to_device(V_e[F_emit[:, 0]])
                d_tri_b = cuda.to_device(V_e[F_emit[:, 1]])
                d_tri_c = cuda.to_device(V_e[F_emit[:, 2]])

            threads = gpu_threads or 256
            threads = min(threads, cuda.get_current_device().MAX_THREADS_PER_BLOCK)
            blocks = (n_rays_once + threads - 1) // threads

        for itr in range(max_iters):
            rng = np.random.default_rng(seed + idx_emit + itr)
            cp_grid = rng.random(2, dtype=np.float32)
            cp_dims = rng.random(5, dtype=np.float32)

            if use_gpu:
                # Ray generation
                if gpu_raygen:
                    if cuda_async:
                        kernel_build_rays[blocks, threads, stream](
                            d_u, d_v, d_cdf, d_tri_a, d_tri_b, d_tri_c,
                            g, rays, d_orig, d_dirs, cp_grid, cp_dims
                        )
                    else:
                        kernel_build_rays[blocks, threads](
                            d_u, d_v, d_cdf, d_tri_a, d_tri_b, d_tri_c,
                            g, rays, d_orig, d_dirs, cp_grid, cp_dims
                        )
                else:
                    # Build on CPU then upload
                    orig = h_orig if cuda_async else np.empty((n_rays_once, 3), np.float32)
                    dire = h_dirs if cuda_async else np.empty((n_rays_once, 3), np.float32)
                    build_rays(
                        u_grid, v_grid, cdf.astype(np.float32),
                        V_e[F_emit[:, 0]], V_e[F_emit[:, 1]], V_e[F_emit[:, 2]],
                        g, rays, orig, dire, cp_grid, cp_dims
                    )
                    if cuda_async:
                        d_orig.copy_to_device(orig, stream=stream)
                        d_dirs.copy_to_device(dire, stream=stream)
                    else:
                        d_orig.copy_to_device(orig)
                        d_dirs.copy_to_device(dire)

                # Zero hitmask and trace
                if cuda_async:
                    kernel_zero_u8[blocks, threads, stream](d_hit)
                else:
                    kernel_zero_u8[blocks, threads](d_hit)
                if use_bvh:
                    if cuda_async:
                        kernel_trace_bvh_hitmask[blocks, threads, stream](
                            d_orig, d_dirs, d_v0, d_e1, d_e2, d_nrm, d_sid,
                            d_bbmin, d_bbmax, d_left, d_right, d_start, d_cnt,
                            d_hit,
                        )
                    else:
                        kernel_trace_bvh_hitmask[blocks, threads](
                            d_orig, d_dirs, d_v0, d_e1, d_e2, d_nrm, d_sid,
                            d_bbmin, d_bbmax, d_left, d_right, d_start, d_cnt,
                            d_hit,
                        )
                else:
                    if cuda_async:
                        kernel_trace_hitmask[blocks, threads, stream](
                            d_orig, d_dirs, d_v0, d_e1, d_e2, d_nrm, d_sid, d_hit
                        )
                    else:
                        kernel_trace_hitmask[blocks, threads](
                            d_orig, d_dirs, d_v0, d_e1, d_e2, d_nrm, d_sid, d_hit
                        )

                # Download hitmask and directions if needed for classification
                # Bin on device to avoid host copies of directions
                # Zero counts then classify sky-visible rays into 145 bins
                if cuda_async:
                    kernel_zero_i32[(145 + threads - 1) // threads, threads, stream](d_counts)
                else:
                    kernel_zero_i32[(145 + threads - 1) // threads, threads](d_counts)
                if cuda_async:
                    kernel_bin_tregenza[blocks, threads, stream](d_dirs, d_hit, d_counts)
                else:
                    kernel_bin_tregenza[blocks, threads](d_dirs, d_hit, d_counts)
                # Download counts for this iteration
                if cuda_async:
                    d_counts.copy_to_host(h_counts, stream=stream)
                    stream.synchronize()
                    counts_iter = h_counts.astype(np.int64)
                else:
                    cuda.synchronize()
                    counts_iter = d_counts.copy_to_host().astype(np.int64)
                # Accumulate and compute per-iteration fractions for stderr
                counts_total += counts_iter
                total_rays += n_rays_once
                iters_done += 1
                frac_iter = counts_iter.astype(np.float64) / float(n_rays_once)
                sky_iter = float(frac_iter.sum())
                delta = frac_iter - mean_bins
                mean_bins += delta / iters_done
                M2_bins += delta * (frac_iter - mean_bins)
                delta_sky = sky_iter - mean_sky
                mean_sky += delta_sky / iters_done
                M2_sky += delta_sky * (sky_iter - mean_sky)

                # Convergence checks
                if tol_mode == "delta":
                    if iters_done >= max(1, min_iters) and total_rays >= max(0, min_total_rays):
                        curr = counts_total.astype(np.float64) / float(total_rays)
                        if prev_frac is not None and np.all(np.abs(curr - prev_frac) < tol):
                            break
                        prev_frac = curr.copy()
                elif tol_mode == "stderr":
                    if iters_done >= max(1, min_iters) and total_rays >= max(0, min_total_rays):
                        if iters_done > 1:
                            var_bins = M2_bins / (iters_done - 1)
                            se_bins = np.sqrt(np.maximum(var_bins, 0.0) / iters_done)
                            if discrete:
                                if np.all(se_bins <= tol):
                                    break
                            else:
                                var_sky = M2_sky / (iters_done - 1)
                                se_sky = (var_sky if var_sky > 0 else 0.0) ** 0.5 / iters_done**0.5
                                if se_sky <= tol:
                                    break
                else:
                    raise ValueError(f"Unknown tol_mode: {tol_mode}")
            else:
                # CPU path
                orig = np.empty((n_rays_once, 3), np.float32)
                dire_host = np.empty_like(orig)
                build_rays(
                    u_grid, v_grid, cdf.astype(np.float32),
                    V_e[F_emit[:, 0]], V_e[F_emit[:, 1]], V_e[F_emit[:, 2]],
                    g, rays, orig, dire_host, cp_grid, cp_dims
                )
                hitmask = np.zeros(n_rays_once, np.int8)
                if use_bvh:
                    trace_cpu_bvh_hitmask(
                        orig, dire_host, v0, e1, e2, nrm, sid,
                        bb_min, bb_max, left, right, start, cnt,
                        hitmask,
                    )
                else:
                    trace_cpu_hitmask(orig, dire_host, v0, e1, e2, nrm, sid, hitmask)

                # Classify and accumulate on CPU
                counts_iter = np.zeros(145, np.int64)
                for k in range(n_rays_once):
                    if hitmask[k] != 0:
                        continue
                    pid = dir_to_patch_id(dire_host[k])
                    if pid >= 0:
                        counts_iter[pid] += 1

                counts_total += counts_iter
                total_rays += n_rays_once
                iters_done += 1

                frac_iter = counts_iter.astype(np.float64) / float(n_rays_once)
                sky_iter = float(frac_iter.sum())

                delta = frac_iter - mean_bins
                mean_bins += delta / iters_done
                M2_bins += delta * (frac_iter - mean_bins)
                delta_sky = sky_iter - mean_sky
                mean_sky += delta_sky / iters_done
                M2_sky += delta_sky * (sky_iter - mean_sky)

                if tol_mode == "delta":
                    if iters_done >= max(1, min_iters) and total_rays >= max(0, min_total_rays):
                        curr = counts_total.astype(np.float64) / float(total_rays)
                        if prev_frac is not None and np.all(np.abs(curr - prev_frac) < tol):
                            break
                        prev_frac = curr.copy()
                elif tol_mode == "stderr":
                    if iters_done >= max(1, min_iters) and total_rays >= max(0, min_total_rays):
                        if iters_done > 1:
                            var_bins = M2_bins / (iters_done - 1)
                            se_bins = np.sqrt(np.maximum(var_bins, 0.0) / iters_done)
                            if discrete:
                                if np.all(se_bins <= tol):
                                    break
                            else:
                                var_sky = M2_sky / (iters_done - 1)
                                se_sky = (var_sky if var_sky > 0 else 0.0) ** 0.5 / iters_done**0.5
                                if se_sky <= tol:
                                    break
                else:
                    raise ValueError(f"Unknown tol_mode: {tol_mode}")

        # Normalize by total rays
        if discrete:
            row = {}
            frac = counts_total.astype(np.float64) / float(max(1, total_rays))
            for i in range(145):
                row[f"{patch_prefix}{i+1}"] = float(frac[i])
            result[name_e].update(row)
        else:
            result[name_e][sky_name] = float(counts_total.sum()) / float(max(1, total_rays))

        # Log in the same style as view_factor_matrix
        elapsed = time.time() - t0
        # Derive iteration count from rays accumulated
        rays_per_iter = max(1, n_rays_once)
        iter_count = max(1, total_rays // rays_per_iter)
        bvh_str = 'builtin' if use_bvh else 'off'
        dev_str = 'gpu' if use_gpu else 'cpu'
        msg = (
            f"({idx_emit+1}/{len(meshes)}) [{name_e}] "
            f"{iter_count} iter, {total_rays:,} rays -> {elapsed:0.3f}s  "
            f"(BVH={bvh_str}, device={dev_str})"
        )
        _log(msg)
        try:
            import Rhino
            Rhino.RhinoApp.WriteLine(msg)
        except Exception:
            pass

    return result


__all__ = [
    "view_factor_matrix",
    "view_factor",
    "view_factor_to_tregenza_sky",
]
