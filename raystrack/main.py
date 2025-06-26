from __future__ import annotations
import time
from typing import List, Tuple, Dict
import numpy as np
from numba import cuda

from .utils.halton import cached_halton
from .utils.geometry import flatten_receivers, flip_meshes
from .utils.ray_builder import build_rays
from .utils.cpu_trace import trace_cpu, trace_cpu_bvh
from .utils.cuda_trace import kernel_trace, kernel_trace_bvh
from .utils.bvh import build_bvh


def view_factor_matrix(meshes: List[Tuple[str, np.ndarray, np.ndarray]],
                       samples: int = 256, rays: int = 256,
                       seed: int = 0, gpu_threads=None,
                       use_bvh: bool = False, flip_faces: bool = False) -> Dict[str, Dict[str, float]]:
    """Compute F(i→j) for every pair of surfaces in *meshes*."""
    have_cuda = cuda.is_available()
    u_grid, v_grid = cached_halton(samples)
    result: Dict[str, Dict[str, float]] = {}

    if flip_faces:
        meshes = flip_meshes(meshes)

    for idx_emit, (name_e, V_e, F_e) in enumerate(meshes):
        t_tot = time.time()

        v0, e1, e2, sid, nrm = flatten_receivers(meshes, idx_emit)
        n_surf = len(meshes)
        hits_f = np.zeros(n_surf, np.int64)
        hits_b = np.zeros_like(hits_f)

        if use_bvh:
            bb_min, bb_max, left, right, start, cnt, perm = build_bvh(v0, e1, e2)
            v0 = v0[perm]
            e1 = e1[perm]
            e2 = e2[perm]
            sid = sid[perm]
            nrm = nrm[perm]
        else:
            bb_min = bb_max = left = right = start = cnt = None

        A = 0.5 * np.linalg.norm(np.cross(V_e[F_e[:, 1]] - V_e[F_e[:, 0]],
                                          V_e[F_e[:, 2]] - V_e[F_e[:, 0]]), axis=1)
        cdf = np.cumsum(A) / A.sum()

        g = samples
        cells = g * g
        n_rays = cells * rays
        orig = np.empty((n_rays, 3), np.float32)
        dire = np.empty_like(orig)
        rng_uni = np.random.default_rng(seed + idx_emit).random(n_rays * 2 + cells, dtype=np.float32)

        build_rays(u_grid, v_grid,
                   cdf.astype(np.float32),
                   V_e[F_e[:, 0]], V_e[F_e[:, 1]], V_e[F_e[:, 2]],
                   rng_uni, samples, rays, orig, dire)

        if have_cuda:
            d_orig, d_dirs = cuda.to_device(orig), cuda.to_device(dire)
            d_v0, d_e1, d_e2 = map(cuda.to_device, (v0, e1, e2))
            d_nrm, d_sid = cuda.to_device(nrm), cuda.to_device(sid)
            d_hf, d_hb = cuda.to_device(hits_f), cuda.to_device(hits_b)

            threads = gpu_threads or 256
            threads = min(threads, cuda.get_current_device().MAX_THREADS_PER_BLOCK)
            blocks = (n_rays + threads - 1) // threads

            if use_bvh:
                d_bbmin = cuda.to_device(bb_min)
                d_bbmax = cuda.to_device(bb_max)
                d_left = cuda.to_device(left)
                d_right = cuda.to_device(right)
                d_start = cuda.to_device(start)
                d_cnt = cuda.to_device(cnt)
                kernel_trace_bvh[blocks, threads](d_orig, d_dirs,
                                                  d_v0, d_e1, d_e2, d_nrm, d_sid,
                                                  d_bbmin, d_bbmax, d_left, d_right, d_start, d_cnt,
                                                  d_hf, d_hb)
            else:
                kernel_trace[blocks, threads](d_orig, d_dirs,
                                              d_v0, d_e1, d_e2, d_nrm, d_sid,
                                              d_hf, d_hb)
            cuda.synchronize()
            hits_f = d_hf.copy_to_host()
            hits_b = d_hb.copy_to_host()
        else:
            if use_bvh:
                trace_cpu_bvh(orig, dire,
                              v0, e1, e2, nrm, sid,
                              bb_min, bb_max, left, right, start, cnt,
                              hits_f, hits_b)
            else:
                trace_cpu(orig, dire, v0, e1, e2, nrm, sid, hits_f, hits_b)

        row = {}
        n_rays_f = float(n_rays)
        for j, (name_r, _, _) in enumerate(meshes):
            if j == idx_emit:
                continue
            f = hits_f[j] / n_rays_f
            b = hits_b[j] / n_rays_f
            if f > 0:
                row[f"{name_r}_front"] = f
            if b > 0:
                row[f"{name_r}_back"] = b
        result[name_e] = row
        print(f"[{name_e}] total {time.time() - t_tot:.3f}s  (BVH={'on' if use_bvh else 'off'})")

    return result


def view_factor_matrix_brute(*args, **kw):
    kw["use_bvh"] = False
    return view_factor_matrix(*args, **kw)

def view_factor(sender, receiver, *args, **kw):
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

    Returns
    -------
    dict
        A dictionary equivalent to the corresponding rows of
        :func:`view_factor_matrix` but only containing entries for the
        specified receiver surfaces.
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

    meshes = senders + receivers
    have_cuda = cuda.is_available()
    u_grid, v_grid = cached_halton(samples)

    if flip_faces:
        meshes = flip_meshes(meshes)

    result: Dict[str, Dict[str, float]] = {}

    for idx_emit, (name_e, V_e, F_e) in enumerate(meshes[: len(senders)]):
        t_tot = time.time()

        v0, e1, e2, sid, nrm = flatten_receivers(meshes, idx_emit)
        n_surf = len(meshes)
        hits_f = np.zeros(n_surf, np.int64)
        hits_b = np.zeros_like(hits_f)

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

        g = samples
        cells = g * g
        n_rays = cells * rays
        orig = np.empty((n_rays, 3), np.float32)
        dire = np.empty_like(orig)
        rng_uni = np.random.default_rng(seed + idx_emit).random(
            n_rays * 2 + cells, dtype=np.float32
        )

        build_rays(
            u_grid,
            v_grid,
            cdf.astype(np.float32),
            V_e[F_e[:, 0]],
            V_e[F_e[:, 1]],
            V_e[F_e[:, 2]],
            rng_uni,
            samples,
            rays,
            orig,
            dire,
        )

        if have_cuda:
            d_orig, d_dirs = cuda.to_device(orig), cuda.to_device(dire)
            d_v0, d_e1, d_e2 = map(cuda.to_device, (v0, e1, e2))
            d_nrm, d_sid = cuda.to_device(nrm), cuda.to_device(sid)
            d_hf, d_hb = cuda.to_device(hits_f), cuda.to_device(hits_b)

            threads = gpu_threads or 256
            threads = min(threads, cuda.get_current_device().MAX_THREADS_PER_BLOCK)
            blocks = (n_rays + threads - 1) // threads

            if use_bvh:
                d_bbmin = cuda.to_device(bb_min)
                d_bbmax = cuda.to_device(bb_max)
                d_left = cuda.to_device(left)
                d_right = cuda.to_device(right)
                d_start = cuda.to_device(start)
                d_cnt = cuda.to_device(cnt)
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
            hits_f = d_hf.copy_to_host()
            hits_b = d_hb.copy_to_host()
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
                    hits_f,
                    hits_b,
                )
            else:
                trace_cpu(orig, dire, v0, e1, e2, nrm, sid, hits_f, hits_b)

        row = {}
        n_rays_f = float(n_rays)
        for j, (name_r, _, _) in enumerate(meshes):
            if j == idx_emit or j < len(senders):
                # Skip self and other emitters
                continue
            f = hits_f[j] / n_rays_f
            b = hits_b[j] / n_rays_f
            if f > 0:
                row[f"{name_r}_front"] = f
            if b > 0:
                row[f"{name_r}_back"] = b

        result[name_e] = row
        print(
            f"[{name_e}] total {time.time() - t_tot:.3f}s  (BVH={'on' if use_bvh else 'off'})"
        )

    return result

__all__ = ["view_factor_matrix", "view_factor_matrix_brute", "view_factor"]
