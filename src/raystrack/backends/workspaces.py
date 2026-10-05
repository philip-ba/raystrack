"""Reusable backend buffers behind a common trace interface."""
from __future__ import annotations
import numpy as np
from ..utils.ray_builder import build_rays
from ..utils.cpu_trace import (trace_cpu_combined, trace_cpu_bvh_combined, reduce_first_hits, bin_tregenza_cpu, count_upward_misses_cpu)

def _build(emitter, rays, orig, dirs, cp_grid, cp_dims, offset=0):
    build_rays(emitter.u_grid, emitter.v_grid, emitter.halton_tri,
               emitter.halton_u, emitter.halton_v, emitter.halton_r1,
               emitter.halton_r2, emitter.cdf, emitter.tri_a, emitter.tri_e1,
               emitter.tri_e2, emitter.tri_u, emitter.tri_v, emitter.tri_n,
               emitter.tri_origin_eps, rays, orig, dirs, cp_grid, cp_dims, offset)


class _CpuWorkspace:
    def __init__(self):
        self.capacity = 0

    def trace(self, scene, emitter, n_surf, *, rays, cp_grid, cp_dims,
              surf_active, emit_sid, discrete, include_sky, ray_count,
              ray_offset=0, **unused):
        if ray_count > self.capacity or self.capacity == 0:
            self.capacity = max(1, ray_count)
            self.orig = np.empty((self.capacity, 3), np.float32)
            self.dirs = np.empty_like(self.orig)
            self.hit = np.empty(self.capacity, np.int32)
            self.front = np.empty(self.capacity, np.uint8)
            self.mask = np.empty(self.capacity, np.uint8)
        orig, dirs = self.orig[:ray_count], self.dirs[:ray_count]
        hit, front, mask = self.hit[:ray_count], self.front[:ray_count], self.mask[:ray_count]
        _build(emitter, rays, orig, dirs, cp_grid, cp_dims, ray_offset)
        if getattr(scene, "instanced", False):
            from ..utils.cpu_trace import trace_cpu_instanced_combined
            trace_cpu_instanced_combined(orig, dirs, scene, surf_active, emit_sid, 0, hit, front, mask)
        elif scene.use_bvh:
            args = (orig, dirs, scene.v0, scene.e1, scene.e2, scene.normals, scene.sid, surf_active)
            trace_cpu_bvh_combined(*args, scene.bb_min, scene.bb_max, scene.left,
                                   scene.right, scene.start, scene.count,
                                   emit_sid, 0, hit, front, mask)
        else:
            args = (orig, dirs, scene.v0, scene.e1, scene.e2, scene.normals, scene.sid, surf_active)
            trace_cpu_combined(*args, emit_sid, 0, hit, front, mask)
        hf, hb = np.zeros(n_surf, np.int64), np.zeros(n_surf, np.int64)
        reduce_first_hits(hit, front, hf, hb)
        sky = np.zeros(145 if discrete else 1, np.int64)
        if include_sky:
            if discrete:
                bin_tregenza_cpu(dirs, mask, sky)
            else:
                sky[0] = count_upward_misses_cpu(dirs, mask)
        return hf, hb, sky

    def trace_many(self, prepared, emitter_index, scene, emitter, n_surf, specs, **options):
        """Use the common replicate interface with reusable CPU ray buffers."""
        return [self.trace(scene, emitter, n_surf, cp_grid=grid, cp_dims=dims,
                           ray_count=count, ray_offset=offset, **options)
                for grid, dims, count, offset in specs]


class _CudaWorkspace:
    """Persistent buffers; multiple replicates read back in one compact copy."""
    def __init__(self, async_):
        from numba import cuda
        self.stream = cuda.stream() if async_ else 0
        self.capacity = self.summary_capacity = self.columns = 0

    def trace_many(self, prepared, emitter_index, scene, emitter, n_surf, specs, **options):
        from numba import cuda
        from ..utils.cuda_trace import kernel_build_rays, kernel_zero_i64
        from .cuda_fused import kernel_fused_flat, kernel_fused_bvh, kernel_fused_instanced
        from .launch import _compute_cuda_launch
        if not specs:
            return []
        stream = self.stream
        capacity = max(1, max(spec[2] for spec in specs))
        columns = 2 * n_surf + (145 if options["discrete"] else 1)
        if capacity > self.capacity:
            self.capacity = capacity
            self.orig = cuda.device_array((capacity, 3), np.float32)
            self.dirs = cuda.device_array((capacity, 3), np.float32)
            self.host_orig = np.empty((capacity, 3), np.float32)
            self.host_dirs = np.empty_like(self.host_orig)
        if columns != self.columns or len(specs) > self.summary_capacity:
            self.columns, self.summary_capacity = columns, len(specs)
            self.summary = cuda.device_array((len(specs), columns), np.int64)
            self.active = cuda.device_array(n_surf, np.uint8)
        self.active.copy_to_device(options["surf_active"], stream=stream)
        ds = (prepared.get_device_instanced_scene() if getattr(scene, "instanced", False)
              else prepared.get_device_scene(use_bvh=scene.use_bvh))
        de = prepared.get_device_emitter(emitter_index, samples=options["samples"],
                    rays=options["rays"], flip_faces=options["flip_faces"])
        # Geometry uploads may use the default stream. Finish them before the
        # workspace stream traverses updated buffers.
        if getattr(self, "_scene_version", None) != prepared.version:
            cuda.synchronize()
            self._scene_version = prepared.version
        for slot, (cp_grid, cp_dims, ray_count, ray_offset) in enumerate(specs):
            orig, dirs = self.orig[:ray_count], self.dirs[:ray_count]
            blocks, threads = _compute_cuda_launch(ray_count, None)
            if options["gpu_raygen"]:
                kernel_build_rays[blocks, threads, stream](de.u_grid, de.v_grid,
                    de.halton_tri, de.halton_u, de.halton_v, de.halton_r1, de.halton_r2,
                    de.cdf, de.tri_a, de.tri_e1, de.tri_e2, de.tri_u, de.tri_v,
                    de.tri_n, de.tri_origin_eps, options["rays"], orig, dirs,
                    cp_grid, cp_dims, ray_offset)
            else:
                ho, hd = self.host_orig[:ray_count], self.host_dirs[:ray_count]
                _build(emitter, options["rays"], ho, hd, cp_grid, cp_dims, ray_offset)
                # The host buffers are shared: synchronous copies prevent
                # overwriting them while an earlier upload is pending.
                orig.copy_to_device(ho)
                dirs.copy_to_device(hd)
            row = self.summary[slot]
            zblocks, zthreads = _compute_cuda_launch(columns, None)
            kernel_zero_i64[zblocks, zthreads, stream](row)
            if getattr(scene, "instanced", False):
                kernel_fused_instanced[blocks, threads, stream](
                    orig, dirs, *ds, self.active, emitter_index, n_surf,
                    options["discrete"], options["include_sky"], row)
            elif scene.use_bvh:
                args = (orig, dirs, ds.v0, ds.e1, ds.e2, ds.normals, ds.sid, self.active)
                kernel_fused_bvh[blocks, threads, stream](*args,
                    ds.bb_min, ds.bb_max, ds.left, ds.right, ds.start, ds.count,
                    emitter_index, n_surf, options["discrete"], options["include_sky"], row)
            else:
                args = (orig, dirs, ds.v0, ds.e1, ds.e2, ds.normals, ds.sid, self.active)
                kernel_fused_flat[blocks, threads, stream](*args,
                    emitter_index, n_surf, options["discrete"], options["include_sky"], row)
        host = self.summary[:len(specs)].copy_to_host(stream=stream)
        if stream:
            stream.synchronize()
        else:
            cuda.synchronize()
        return [(r[:n_surf], r[n_surf:2*n_surf], r[2*n_surf:]) for r in host]


class _PortableWorkspace:
    """Adapt portable kernels to the same CPU/CUDA replicate interface."""

    def __init__(self, arch):
        from ..utils.taichi_trace import get_taichi_backend
        self.backend = get_taichi_backend(arch=arch)

    def trace_many(self, prepared, emitter_index, scene, emitter, n_surf, specs, **options):
        if not specs:
            return []
        # These controls belong to emitter preparation, not portable launches.
        portable = {key: value for key, value in options.items()
                    if key not in ("samples", "flip_faces")}
        first = specs[0]
        if len(specs) > 1 and all(spec[2:] == first[2:] for spec in specs[1:]):
            front, back, sky = self.backend.trace_iterations(
                scene, emitter,
                cp_grids=np.asarray([spec[0] for spec in specs], np.float32),
                cp_dims=np.asarray([spec[1] for spec in specs], np.float32),
                ray_count=first[2], ray_offset=first[3], **portable)
            return list(zip(front, back, sky))
        return [self.backend.trace_batch(scene, emitter, cp_grid=grid, cp_dims=dims,
                    ray_count=count, ray_offset=offset, **portable)
                for grid, dims, count, offset in specs]


def workspace_for(prepared, backend, cfg):
    """Return a cached backend workspace, including CUDA context ownership."""
    cache = getattr(prepared, "_execution_workspace_cache", None)
    if cache is None:
        cache = prepared._execution_workspace_cache = {}
    if backend == "cpu":
        key, factory = "cpu", _CpuWorkspace
    elif backend == "cuda":
        from numba import cuda
        key = ("cuda", *prepared._cuda_key(cuda), cfg.cuda_async)
        factory = lambda: _CudaWorkspace(cfg.cuda_async)
    elif backend in ("taichi", "vulkan", "metal"):
        arch = "auto" if backend == "taichi" else backend
        key = ("portable", arch)
        factory = lambda: _PortableWorkspace(arch)
    else:
        raise ValueError(f"Unknown backend workspace: {backend}")
    if key not in cache:
        cache[key] = factory()
    return cache[key]
