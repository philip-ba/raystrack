"""Optional portable FP32 GPU tracing through Taichi Vulkan/Metal.

No Taichi import or runtime initialization occurs until a backend is requested.
Taichi owns one runtime per process: an existing compatible runtime is reused,
and an incompatible runtime is rejected instead of resetting someone else's data.
The raystrack backend deliberately never falls back to CPU.
"""

import importlib
import os
import sys
import threading

import numpy as np

_LOCK = threading.RLock()
_BACKEND = None
_MAX_BATCH_RAYS = 1 << 20
_INT32_MAX = np.iinfo(np.int32).max


class TaichiUnavailableError(RuntimeError):
    """The optional dependency, requested device, or runtime is unavailable."""


def _load_taichi():
    try:
        return importlib.import_module("taichi")
    except (ImportError, OSError) as exc:
        raise TaichiUnavailableError(
            "Portable GPU tracing requires Taichi; install raystrack[portable-gpu] "
            "in a Python version supported by Taichi."
        ) from exc


def _arch_name(ti, arch):
    for name in ("vulkan", "metal", "cuda", "cpu"):
        if arch == getattr(ti, name, None):
            return name
    return str(arch)


def probe_taichi():
    """Return dependency and GPU capability details without initializing a runtime.

    The Taichi capability probes load device drivers, but do not call ``ti.init``.
    Adapter selection, when needed, uses TI_VISIBLE_DEVICE before first use.
    """
    report = {"installed": False, "available": False, "architectures": [],
              "version": None, "runtime_arch": None,
              "adapter": os.environ.get("TI_VISIBLE_DEVICE")}
    try:
        ti = _load_taichi()
        report["installed"] = True
        report["version"] = getattr(ti, "__version__", None)
        core = ti._lib.core
        for name, check in (("vulkan", "with_vulkan"), ("metal", "with_metal"),
                            ("cuda", "with_cuda")):
            try:
                if bool(getattr(core, check)()):
                    report["architectures"].append(name)
            except (AttributeError, RuntimeError):
                pass
        runtime = ti.lang.impl.get_runtime()
        if runtime.prog is not None:
            report["runtime_arch"] = _arch_name(ti, ti.lang.impl.current_cfg().arch)
        report["available"] = bool(report["architectures"])
        if runtime.prog is not None:
            config = ti.lang.impl.current_cfg()
            if report["runtime_arch"] not in ("vulkan", "metal", "cuda"):
                report["available"] = False
                report["reason"] = ("An incompatible Taichi runtime is already active (%s); "
                                    "use a separate process for GPU tracing." % report["runtime_arch"])
            elif config.default_fp != ti.f32 or config.default_ip != ti.i32:
                report["available"] = False
                report["reason"] = "The active Taichi runtime must use default_fp=f32 and default_ip=i32."
    except TaichiUnavailableError as exc:
        report["reason"] = str(exc)
    return report


def _escape_links(scene):
    """Next node after each subtree, allowing traversal without a local stack."""
    if not scene.use_bvh:
        return np.full(1, -1, np.int32)
    n = len(scene.count)
    escapes = np.full(n, -1, np.int32)
    visited = set()
    pending = [(0, -1)]
    while pending:
        node, successor = pending.pop()
        if node < 0 or node >= n or node in visited:
            raise ValueError("Invalid or cyclic BVH topology")
        visited.add(node)
        escapes[node] = successor
        if scene.count[node] > 0:
            if scene.start[node] < 0 or scene.start[node] + scene.count[node] > len(scene.sid):
                raise ValueError("BVH leaf refers to triangles outside the scene")
        else:
            left, right = int(scene.left[node]), int(scene.right[node])
            pending.append((right, successor))
            pending.append((left, right))
    if len(visited) != n:
        raise ValueError("BVH contains unreachable nodes")
    return escapes


def _make_kernels(ti):
    # Keep evaluated annotations here: Taichi inspects these Python definitions.
    f2 = ti.types.ndarray(dtype=ti.f32, ndim=2)
    f1 = ti.types.ndarray(dtype=ti.f32, ndim=1)
    i2 = ti.types.ndarray(dtype=ti.i32, ndim=2)
    i1 = ti.types.ndarray(dtype=ti.i32, ndim=1)

    @ti.func
    def intersect(o, d, geom: ti.template(), tri):
        a = ti.Vector([geom[tri, 0], geom[tri, 1], geom[tri, 2]])
        e1 = ti.Vector([geom[tri, 3], geom[tri, 4], geom[tri, 5]])
        e2 = ti.Vector([geom[tri, 6], geom[tri, 7], geom[tri, 8]])
        p = d.cross(e2)
        det = e1.dot(p)
        result = 1.0e20
        if ti.abs(det) >= 1.0e-7:
            inv = 1.0 / det
            delta = o - a
            u = delta.dot(p) * inv
            q = delta.cross(e1)
            v = d.dot(q) * inv
            distance = e2.dot(q) * inv
            if u >= 0.0 and u <= 1.0 and v >= 0.0 and u + v <= 1.0 and distance > 1.0e-6:
                result = distance
        return result

    @ti.func
    def aabb(o, d, bounds: ti.template(), node):
        near = 0.0
        far = 1.0e20
        valid = 1
        for axis in ti.static(range(3)):
            # Explicit parallel slabs avoid 0 * infinity and signed-zero errors.
            if ti.abs(d[axis]) <= 1.0e-9:
                if o[axis] < bounds[node, axis] or o[axis] > bounds[node, axis + 3]:
                    valid = 0
            else:
                t0 = (bounds[node, axis] - o[axis]) / d[axis]
                t1 = (bounds[node, axis + 3] - o[axis]) / d[axis]
                near = ti.max(near, ti.min(t0, t1))
                far = ti.min(far, ti.max(t0, t1))
        result = 1.0e20
        if valid and near <= far:
            result = near
        return result

    @ti.func
    def patch(d):
        thresholds = ti.Vector([0.2079116908, 0.4067366431, 0.5877852523,
                                0.7431448255, 0.8660254038, 0.9510565163,
                                0.9945218954, 1.0])
        counts = ti.Vector([30, 30, 24, 24, 18, 12, 6, 1])
        starts = ti.Vector([0, 30, 60, 84, 108, 126, 138, 144])
        ring = 7
        for j in ti.static(range(7)):
            if ring == 7 and d[2] < thresholds[j]:
                ring = j
        az = ti.atan2(d[1], d[0]) * 57.2957795131
        if az < 0.0:
            az += 360.0
        n = counts[ring]
        off = 0.0
        if ring % 2 == 1:
            off = 180.0 / n
        angle = (az - off + 360.0) % 360.0
        result = starts[ring] + ti.min(ti.cast(angle / (360.0 / n), ti.i32), n - 1)
        if d[2] <= 0.0:
            result = -1
        return result

    @ti.kernel
    def generate(grid: f2, halton: f2, emitter: f2, cdf: f1, shifts: f2,
                 output: f2, rays_per_cell: ti.i32, offset: ti.i32, n: ti.i32,
                 shift_index: ti.i32):
        for k in range(n):
            idx = k + offset
            cell = idx // rays_per_cell
            ug = (grid[cell, 0] + shifts[shift_index, 0]) % 1.0
            vg = (grid[cell, 1] + shifts[shift_index, 1]) % 1.0
            choice = (halton[idx, 0] + shifts[shift_index, 2]) % 1.0
            low, high = 0, cdf.shape[0] - 1
            while low <= high:
                mid = (low + high) // 2
                if cdf[mid] < choice:
                    low = mid + 1
                else:
                    high = mid - 1
            tri = ti.min(low, cdf.shape[0] - 1)
            ur = (halton[idx, 1] + shifts[shift_index, 3] + ug) % 1.0
            vr = (halton[idx, 2] + shifts[shift_index, 4] + vg) % 1.0
            root = ti.sqrt(ur)
            b, c = root * vr, root * (1.0 - vr)
            r1 = (halton[idx, 3] + shifts[shift_index, 5]) % 1.0
            r2 = (halton[idx, 4] + shifts[shift_index, 6]) % 1.0
            sine = ti.sqrt(1.0 - r1)
            phi = 6.28318530718 * r2
            x, y, z = sine * ti.cos(phi), sine * ti.sin(phi), ti.sqrt(r1)
            for axis in ti.static(range(3)):
                output[k, axis] = (emitter[tri, axis] + b * emitter[tri, axis + 3]
                                   + c * emitter[tri, axis + 6]
                                   + emitter[tri, 18] * emitter[tri, axis + 15])
                output[k, axis + 3] = (x * emitter[tri, axis + 9]
                                       + y * emitter[tri, axis + 12]
                                       + z * emitter[tri, axis + 15])

    @ti.kernel
    def trace(raydata: f2, geom: f2, sid: i1, active: i1, bounds: f2,
              nodes: i2, escapes: i1, result: i2, sky: i1, n: ti.i32,
              n_tri: ti.i32, use_bvh: ti.i32, emit_sid: ti.i32, min_sid: ti.i32,
              matrix: ti.i32, include_sky: ti.i32, discrete: ti.i32,
              result_offset: ti.i32, sky_offset: ti.i32):
        for k in range(n):
            o = ti.Vector([raydata[k, 0], raydata[k, 1], raydata[k, 2]])
            d = ti.Vector([raydata[k, 3], raydata[k, 4], raydata[k, 5]])
            best, hit, front, any_hit = 1.0e20, -1, 0, 0
            node = 0
            # The same loop supports a leafless brute-force scene and a BVH.
            while node >= 0:
                start, count = 0, n_tri
                visit_leaf = 1
                successor = -1
                if use_bvh:
                    successor = escapes[node]
                    visit_leaf = 0
                    if aabb(o, d, bounds, node) < best:
                        if nodes[node, 3] > 0:
                            start, count = nodes[node, 2], nodes[node, 3]
                            visit_leaf = 1
                        else:
                            successor = nodes[node, 0]
                if visit_leaf:
                    for local in range(count):
                        tri = start + local
                        surface = sid[tri]
                        if surface != emit_sid and active[surface] != 0:
                            t = intersect(o, d, geom, tri)
                            if t < 1.0e20:
                                any_hit = 1
                                if matrix and surface >= min_sid and t < best:
                                    best, hit = t, surface
                                    norm = ti.Vector([geom[tri, 9], geom[tri, 10], geom[tri, 11]])
                                    front = ti.cast(-d.dot(norm) > 0.0, ti.i32)
                node = successor
            if matrix and hit >= 0:
                ti.atomic_add(result[result_offset + hit, 1 - front], 1)
            if include_sky and any_hit == 0 and d[2] > 0.0:
                bin_id = 0
                if discrete:
                    bin_id = patch(d)
                if bin_id >= 0:
                    ti.atomic_add(sky[sky_offset + bin_id], 1)
    @ti.kernel
    def trace_instanced(raydata: f2, geom: f2, active: i1, blas_bounds: f2,
                        blas_nodes: i2, blas_escape: i1, roots: i1, transforms: f2,
                        tlas_bounds: f2, tlas_nodes: i2, tlas_escape: i1,
                        instance_order: i1, result: i2, sky: i1, n: ti.i32,
                        n_tlas: ti.i32, emit_sid: ti.i32, min_sid: ti.i32,
                        matrix: ti.i32, include_sky: ti.i32, discrete: ti.i32,
                        result_offset: ti.i32, sky_offset: ti.i32):
        for k in range(n):
            o = ti.Vector([raydata[k, 0], raydata[k, 1], raydata[k, 2]])
            d = ti.Vector([raydata[k, 3], raydata[k, 4], raydata[k, 5]])
            best, hit, front, any_hit = 1.0e20, -1, 0, 0
            node = -1
            if n_tlas > 0:
                node = 0
            while node >= 0:
                successor = tlas_escape[node]
                if aabb(o, d, tlas_bounds, node) < best:
                    if tlas_nodes[node, 3] == 0:
                        successor = tlas_nodes[node, 0]
                    else:
                        for position in range(tlas_nodes[node, 2], tlas_nodes[node, 2] + tlas_nodes[node, 3]):
                            surface = instance_order[position]
                            if surface != emit_sid and active[surface] != 0:
                                local_o, local_d = ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
                                for axis in ti.static(range(3)):
                                    for row in ti.static(range(3)):
                                        local_o[axis] += transforms[surface, row * 3 + axis] * (o[row] - transforms[surface, 9 + row])
                                        local_d[axis] += transforms[surface, row * 3 + axis] * d[row]
                                blas_node = roots[surface]
                                while blas_node >= 0:
                                    next_blas = blas_escape[blas_node]
                                    if aabb(local_o, local_d, blas_bounds, blas_node) < best:
                                        if blas_nodes[blas_node, 3] == 0:
                                            next_blas = blas_nodes[blas_node, 0]
                                        else:
                                            for tri in range(blas_nodes[blas_node, 2], blas_nodes[blas_node, 2] + blas_nodes[blas_node, 3]):
                                                distance = intersect(local_o, local_d, geom, tri)
                                                if distance < 1.0e20:
                                                    any_hit = 1
                                                    if matrix and surface >= min_sid and distance < best:
                                                        best, hit = distance, surface
                                                        normal = ti.Vector([geom[tri, 9], geom[tri, 10], geom[tri, 11]])
                                                        front = ti.cast(-local_d.dot(normal) > 0, ti.i32)
                                    blas_node = next_blas
                node = successor
            if matrix and hit >= 0:
                ti.atomic_add(result[result_offset + hit, 1 - front], 1)
            if include_sky and any_hit == 0 and d[2] > 0:
                bin_id = 0
                if discrete:
                    bin_id = patch(d)
                if bin_id >= 0:
                    ti.atomic_add(sky[sky_offset + bin_id], 1)
    return generate, trace, trace_instanced


class TaichiBackend:
    """Persistent device geometry, ray buffers, and bounded i32 reduction bins."""

    def __init__(self, arch="auto", *, max_batch_rays=_MAX_BATCH_RAYS):
        if arch not in ("auto", "vulkan", "metal", "cuda"):
            raise ValueError("Taichi arch must be auto, vulkan, metal, or cuda")
        if isinstance(max_batch_rays, bool) or not 0 < int(max_batch_rays) <= _INT32_MAX:
            raise ValueError("max_batch_rays must fit a positive int32")
        self.max_batch_rays = int(max_batch_rays)
        self._buffers = {}
        self._scene = None
        self._instanced_blas = None
        self._emitter = None
        self._host_orig = self._host_dirs = self._host_packed = None
        with _LOCK:
            self.ti = ti = _load_taichi()
            runtime = ti.lang.impl.get_runtime()
            if runtime.prog is not None:
                actual = _arch_name(ti, ti.lang.impl.current_cfg().arch)
                if actual not in ("vulkan", "metal", "cuda") or (arch != "auto" and actual != arch):
                    raise TaichiUnavailableError(
                        "An incompatible Taichi runtime is already active (%s). "
                        "Use a separate process for the requested GPU backend." % actual)
                config = ti.lang.impl.current_cfg()
                if config.default_fp != ti.f32 or config.default_ip != ti.i32:
                    raise TaichiUnavailableError(
                        "Existing Taichi runtime must use default_fp=ti.f32 and "
                        "default_ip=ti.i32 for portable tracing. Use a separate process.")
                self.arch = actual
            else:
                candidates = (["metal", "vulkan", "cuda"] if sys.platform == "darwin"
                              else ["vulkan", "cuda"])
                if arch != "auto":
                    candidates = [arch]
                failures = []
                for name in candidates:
                    try:
                        ti.init(arch=getattr(ti, name), enable_fallback=False,
                                default_fp=ti.f32, default_ip=ti.i32, fast_math=False)
                        actual = _arch_name(ti, ti.lang.impl.current_cfg().arch)
                        if actual != name:
                            raise TaichiUnavailableError("Requested %s but Taichi selected %s" % (name, actual))
                        self.arch = actual
                        break
                    except Exception as exc:
                        failures.append("%s: %s" % (name, exc))
                else:
                    raise TaichiUnavailableError("No supported GPU backend: " + "; ".join(failures))
            self._runtime_program = ti.lang.impl.get_runtime().prog
            self._generate, self._trace, self._trace_instanced = _make_kernels(ti)

    @property
    def details(self):
        return {"backend": "taichi", "arch": self.arch,
                "version": getattr(self.ti, "__version__", None),
                "adapter": os.environ.get("TI_VISIBLE_DEVICE"),
                "max_batch_rays": self.max_batch_rays}

    def _check_runtime(self):
        if self.ti.lang.impl.get_runtime().prog is not self._runtime_program:
            raise TaichiUnavailableError("Taichi runtime was reset externally; create a new backend")

    def _buffer(self, name, shape, dtype, data=None):
        current = self._buffers.get(name)
        if current is None or current.shape != shape:
            current = self.ti.ndarray(dtype=dtype, shape=shape)
            self._buffers[name] = current
        if data is not None:
            current.from_numpy(np.ascontiguousarray(data))
        return current

    def _upload_scene(self, scene):
        if getattr(scene, "instanced", False):
            self._upload_instanced_scene(scene)
            return
        if scene is self._scene:
            return
        n = len(scene.sid)
        geom = np.column_stack((scene.v0, scene.e1, scene.e2, scene.normals)).astype(np.float32)
        if n == 0:
            geom = np.zeros((1, 12), np.float32)
        self._buffer("geom", geom.shape, self.ti.f32, geom)
        sids = np.asarray(scene.sid, np.int32) if n else np.zeros(1, np.int32)
        self._buffer("sid", sids.shape, self.ti.i32, sids)
        escapes = _escape_links(scene)
        bounds = (np.column_stack((scene.bb_min, scene.bb_max)).astype(np.float32)
                  if scene.use_bvh else np.zeros((1, 6), np.float32))
        nodes = (np.column_stack((scene.left, scene.right, scene.start, scene.count)).astype(np.int32)
                 if scene.use_bvh else np.zeros((1, 4), np.int32))
        self._buffer("bounds", bounds.shape, self.ti.f32, bounds)
        self._buffer("nodes", nodes.shape, self.ti.i32, nodes)
        self._buffer("escape", escapes.shape, self.ti.i32, escapes)
        self._scene = scene

    def _upload_instanced_scene(self, scene):
        if scene is self._scene:
            return
        def padded(array, width=None):
            return array if len(array) else np.zeros((1, width) if width else (1,), array.dtype)
        blas = scene.blas
        if blas is not self._instanced_blas:
            self._buffer("inst_geom", padded(blas.geom, 12).shape, self.ti.f32, padded(blas.geom, 12))
            bounds = padded(np.column_stack((blas.bb_min, blas.bb_max)).astype(np.float32), 6)
            nodes = padded(np.column_stack((blas.left, blas.right, blas.start, blas.count)).astype(np.int32), 4)
            self._buffer("inst_blas_bounds", bounds.shape, self.ti.f32, bounds)
            self._buffer("inst_blas_nodes", nodes.shape, self.ti.i32, nodes)
            escape = padded(blas.escape)
            self._buffer("inst_blas_escape", escape.shape, self.ti.i32, escape)
            self._instanced_blas = blas
        transforms = padded(np.column_stack((scene.rotations.reshape(-1, 9), scene.translations)).astype(np.float32), 12)
        roots = padded(scene.roots)
        bounds = padded(np.column_stack((scene.bb_min, scene.bb_max)).astype(np.float32), 6)
        nodes = padded(np.column_stack((scene.left, scene.right, scene.start, scene.count)).astype(np.int32), 4)
        self._buffer("inst_transforms", transforms.shape, self.ti.f32, transforms)
        self._buffer("inst_roots", roots.shape, self.ti.i32, roots)
        self._buffer("inst_tlas_bounds", bounds.shape, self.ti.f32, bounds)
        self._buffer("inst_tlas_nodes", nodes.shape, self.ti.i32, nodes)
        for name, array in (("inst_tlas_escape", scene.escape), ("inst_order", scene.instance_order)):
            array = padded(array)
            self._buffer(name, array.shape, self.ti.i32, array)
        self._scene = scene

    def _upload_emitter(self, emitter):
        if emitter is self._emitter:
            return
        if not len(emitter.cdf):
            raise ValueError("Cannot generate rays from an empty emitter")
        packed = np.column_stack((emitter.tri_a, emitter.tri_e1, emitter.tri_e2,
                                  emitter.tri_u, emitter.tri_v, emitter.tri_n,
                                  emitter.tri_origin_eps)).astype(np.float32)
        grid = np.column_stack((emitter.u_grid, emitter.v_grid)).astype(np.float32)
        halton = np.column_stack((emitter.halton_tri, emitter.halton_u, emitter.halton_v,
                                  emitter.halton_r1, emitter.halton_r2)).astype(np.float32)
        self._buffer("emitter", packed.shape, self.ti.f32, packed)
        self._buffer("grid", grid.shape, self.ti.f32, grid)
        self._buffer("halton", halton.shape, self.ti.f32, halton)
        cdf = np.asarray(emitter.cdf, np.float32)
        self._buffer("cdf", cdf.shape, self.ti.f32, cdf)
        self._emitter = emitter

    def _prepare(self, scene, surf_active, capacity):
        self._check_runtime()
        self._upload_scene(scene)
        active = np.asarray(surf_active, np.int32)
        if active.ndim != 1 or not len(active):
            raise ValueError("surf_active must be a nonempty one-dimensional array")
        if len(scene.sid) and (np.min(scene.sid) < 0 or np.max(scene.sid) >= len(active)):
            raise ValueError("Scene surface ids exceed surf_active")
        self._buffer("active", active.shape, self.ti.i32, active)
        current = self._buffers.get("raydata")
        if current is not None:
            capacity = max(capacity, current.shape[0])
        self._buffer("raydata", (max(1, capacity), 6), self.ti.f32)
        self._buffer("result", (len(active), 2), self.ti.i32)
        self._buffer("sky", (145,), self.ti.i32)

    def _run_trace(self, scene, n, emit_sid, min_sid, matrix, sky, discrete):
        b = self._buffers
        b["result"].fill(0)
        b["sky"].fill(0)
        self._launch_trace(scene, b["result"], b["sky"], n, emit_sid, min_sid,
                           matrix, sky, discrete, 0, 0)
        hits = b["result"].to_numpy().astype(np.int64)
        counts = b["sky"].to_numpy().astype(np.int64)
        return hits, counts

    def _launch_trace(self, scene, result, sky_result, n, emit_sid, min_sid,
                      matrix, sky, discrete, result_offset, sky_offset):
        b = self._buffers
        if getattr(scene, "instanced", False):
            self._trace_instanced(b["raydata"], b["inst_geom"], b["active"],
                b["inst_blas_bounds"], b["inst_blas_nodes"], b["inst_blas_escape"],
                b["inst_roots"], b["inst_transforms"], b["inst_tlas_bounds"],
                b["inst_tlas_nodes"], b["inst_tlas_escape"], b["inst_order"],
                result, sky_result, n, len(scene.count), emit_sid, min_sid,
                int(matrix), int(sky), int(discrete), result_offset, sky_offset)
        else:
            self._trace(b["raydata"], b["geom"], b["sid"], b["active"], b["bounds"],
                b["nodes"], b["escape"], result, sky_result, n, len(scene.sid),
                int(scene.use_bvh), emit_sid, min_sid, int(matrix), int(sky),
                int(discrete), result_offset, sky_offset)

    def trace_batch(self, scene, emitter, *, rays, cp_grid, cp_dims, surf_active,
                    emit_sid, min_sid=-1, include_matrix=True, include_sky=False,
                    discrete=False, gpu_raygen=True, ray_count=None, ray_offset=0):
        """Trace ``rays`` per emitter cell, returning front/back and sky counts.

        Oversized iterations are split into bounded GPU batches; only compact
        counts are read back and accumulated in host int64. Changing geometry
        must produce a new PreparedScene/PreparedEmitter (the prepared solver
        update API does this) so device cache identity invalidation is explicit.
        """
        rays = int(rays)
        if rays <= 0 or rays > _INT32_MAX:
            raise ValueError("rays per cell must be a positive int32")
        available = int(emitter.n_cells) * rays
        ray_offset = int(ray_offset)
        total = available - ray_offset if ray_count is None else int(ray_count)
        if ray_offset < 0 or total < 0 or ray_offset + total > available:
            raise ValueError("Requested ray slice exceeds the prepared iteration")
        if ray_offset + total > _INT32_MAX:
            raise ValueError("A prepared iteration exceeds the int32 sample index limit")
        if ray_offset + total > len(emitter.halton_tri):
            raise ValueError("Emitter Halton data has fewer samples than requested rays")
        shifts = np.concatenate((np.asarray(cp_grid, np.float32), np.asarray(cp_dims, np.float32)))
        if np.asarray(cp_grid).shape != (2,) or np.asarray(cp_dims).shape != (5,) or not np.isfinite(shifts).all():
            raise ValueError("cp_grid/cp_dims must contain 2/5 finite shifts")
        with _LOCK:
            self._prepare(scene, surf_active, min(total, self.max_batch_rays))
            self._upload_emitter(emitter)
            self._buffer("shifts", (1, 7), self.ti.f32, shifts.reshape(1, 7))
            sums = np.zeros((len(surf_active), 2), np.int64)
            sky_sums = np.zeros(145, np.int64)
            if not gpu_raygen:
                from .ray_builder import build_rays
                capacity = self._buffers["raydata"].shape[0]
                if self._host_orig is None or len(self._host_orig) < capacity:
                    self._host_orig = np.empty((capacity, 3), np.float32)
                    self._host_dirs = np.empty((capacity, 3), np.float32)
                    self._host_packed = np.empty((capacity, 6), np.float32)
            for offset in range(0, total, self.max_batch_rays):
                n = min(self.max_batch_rays, total - offset)
                b = self._buffers
                if gpu_raygen:
                    self._generate(b["grid"], b["halton"], b["emitter"], b["cdf"],
                                   b["shifts"], b["raydata"], rays, ray_offset + offset, n, 0)
                else:
                    build_rays(emitter.u_grid, emitter.v_grid, emitter.halton_tri,
                               emitter.halton_u, emitter.halton_v, emitter.halton_r1,
                               emitter.halton_r2, emitter.cdf, emitter.tri_a, emitter.tri_e1,
                               emitter.tri_e2, emitter.tri_u, emitter.tri_v, emitter.tri_n,
                               emitter.tri_origin_eps, rays, self._host_orig[:n],
                               self._host_dirs[:n], np.asarray(cp_grid, np.float32),
                               np.asarray(cp_dims, np.float32), ray_offset + offset)
                    self._host_packed[:n, :3] = self._host_orig[:n]
                    self._host_packed[:n, 3:] = self._host_dirs[:n]
                    b["raydata"].from_numpy(self._host_packed)
                hits, counts = self._run_trace(scene, n, emit_sid, min_sid,
                                               include_matrix, include_sky, discrete)
                sums += hits
                sky_sums += counts
            return sums[:, 0], sums[:, 1], sky_sums if discrete else sky_sums[:1]

    def trace_iterations(self, scene, emitter, *, rays, cp_grids, cp_dims,
                         surf_active, emit_sid, min_sid=-1, include_matrix=True,
                         include_sky=False, discrete=False, gpu_raygen=True,
                         ray_count=None, ray_offset=0):
        """Trace several randomized iterations with a single count readback.

        Each iteration has its own bounded i32 bins; results return host int64
        arrays shaped ``(iterations, surfaces)`` and ``(iterations, sky_bins)``.
        GPU ray generation uploads all shifts together, avoiding per-iteration
        host uploads and synchronizations. Use small groups for responsive
        cancellation and convergence checks.
        """
        grids, dims = np.asarray(cp_grids, np.float32), np.asarray(cp_dims, np.float32)
        if grids.ndim != 2 or grids.shape[1:] != (2,) or dims.shape != (len(grids), 5):
            raise ValueError("cp_grids/cp_dims must have matching (n,2)/(n,5) shapes")
        if not np.isfinite(grids).all() or not np.isfinite(dims).all():
            raise ValueError("Iteration shifts must be finite")
        iterations, n_surfaces = len(grids), len(surf_active)
        if not iterations:
            return (np.zeros((0, n_surfaces), np.int64), np.zeros((0, n_surfaces), np.int64),
                    np.zeros((0, 145 if discrete else 1), np.int64))
        rays, ray_offset = int(rays), int(ray_offset)
        if rays <= 0 or rays > _INT32_MAX:
            raise ValueError("rays per cell must be a positive int32")
        available = int(emitter.n_cells) * rays
        total = available - ray_offset if ray_count is None else int(ray_count)
        if ray_offset < 0 or total < 0 or ray_offset + total > available:
            raise ValueError("Requested ray slice exceeds the prepared iteration")
        if ray_offset + total > _INT32_MAX or iterations * max(n_surfaces, 145) > _INT32_MAX:
            raise ValueError("Iteration counts or indices exceed the int32 limit")
        if ray_offset + total > len(emitter.halton_tri):
            raise ValueError("Emitter Halton data has fewer samples than requested rays")
        # The host raygen path remains supported; grouping GPU raygen avoids
        # device uploads and readbacks between individual iterations.
        if not gpu_raygen:
            values = [self.trace_batch(scene, emitter, rays=rays, cp_grid=grid,
                                      cp_dims=dim, surf_active=surf_active, emit_sid=emit_sid,
                                      min_sid=min_sid, include_matrix=include_matrix,
                                      include_sky=include_sky, discrete=discrete,
                                      gpu_raygen=False, ray_count=total, ray_offset=ray_offset)
                      for grid, dim in zip(grids, dims)]
            return tuple(np.stack([value[i] for value in values]) for i in range(3))
        with _LOCK:
            self._prepare(scene, surf_active, min(total, self.max_batch_rays))
            self._upload_emitter(emitter)
            shifts = np.column_stack((grids, dims)).astype(np.float32)
            self._buffer("iteration_shifts", shifts.shape, self.ti.f32, shifts)
            results = self._buffer("iteration_results", (iterations * n_surfaces, 2), self.ti.i32)
            sky_results = self._buffer("iteration_sky", (iterations * 145,), self.ti.i32)
            results.fill(0)
            sky_results.fill(0)
            b = self._buffers
            for iteration in range(iterations):
                for offset in range(0, total, self.max_batch_rays):
                    n = min(self.max_batch_rays, total - offset)
                    self._generate(b["grid"], b["halton"], b["emitter"], b["cdf"],
                                   b["iteration_shifts"], b["raydata"], rays,
                                   ray_offset + offset, n, iteration)
                    self._launch_trace(scene, results, sky_results, n, emit_sid, min_sid,
                                       include_matrix, include_sky, discrete,
                                       iteration * n_surfaces, iteration * 145)
            host = results.to_numpy().astype(np.int64).reshape(iterations, n_surfaces, 2)
            sky_host = sky_results.to_numpy().astype(np.int64).reshape(iterations, 145)
            return host[:, :, 0], host[:, :, 1], sky_host if discrete else sky_host[:, :1]

    def trace_rays(self, scene, orig, directions, *, surf_active, emit_sid=-1,
                   min_sid=-1, include_matrix=True, include_sky=False, discrete=False):
        """Trace caller-supplied rays; useful for integrations and parity checks."""
        orig, directions = np.asarray(orig, np.float32), np.asarray(directions, np.float32)
        if orig.ndim != 2 or orig.shape[1:] != (3,) or directions.shape != orig.shape:
            raise ValueError("orig and directions must have matching (n, 3) shapes")
        if not np.isfinite(orig).all() or not np.isfinite(directions).all():
            raise ValueError("Rays must be finite")
        total = len(orig)
        with _LOCK:
            self._prepare(scene, surf_active, min(total, self.max_batch_rays))
            sums = np.zeros((len(surf_active), 2), np.int64)
            sky_sums = np.zeros(145, np.int64)
            for offset in range(0, total, self.max_batch_rays):
                n = min(self.max_batch_rays, total - offset)
                packed = np.zeros(self._buffers["raydata"].shape, np.float32)
                packed[:n, :3] = orig[offset:offset + n]
                packed[:n, 3:] = directions[offset:offset + n]
                self._buffers["raydata"].from_numpy(packed)
                hits, counts = self._run_trace(scene, n, emit_sid, min_sid,
                                               include_matrix, include_sky, discrete)
                sums += hits
                sky_sums += counts
            return sums[:, 0], sums[:, 1], sky_sums if discrete else sky_sums[:1]


def get_taichi_backend(arch="auto"):
    """Return the shared portable backend; never reset another Taichi runtime."""
    global _BACKEND
    with _LOCK:
        if _BACKEND is not None:
            if arch != "auto" and _BACKEND.arch != arch:
                raise TaichiUnavailableError("The process already uses Taichi %s" % _BACKEND.arch)
            _BACKEND._check_runtime()
            return _BACKEND
        _BACKEND = TaichiBackend(arch)
        return _BACKEND


__all__ = ["TaichiBackend", "TaichiUnavailableError", "get_taichi_backend", "probe_taichi"]
