"""Flatten shared public options once for scheduling and backend hot paths."""
from __future__ import annotations
from types import SimpleNamespace


def controls(solver, query, options, budget):
    senders, receivers = query.resolve(solver.scene.surface_ids)
    s, a = options.sampling, options.accuracy
    return SimpleNamespace(
        samples=s.density, rays=s.rays_per_cell, seed=s.seed,
        sampling_mode=s.mode, flip_faces=s.flip_faces, strategy=s.strategy,
        sequence=s.sequence, pair_samples=s.pair_samples,
        max_iters=a.max_replicates, min_iters=a.min_replicates,
        tol=a.tolerance, tol_mode=a.mode, convergence_interval=a.check_interval,
        min_total_rays=a.min_rays, emitter_names=senders, receiver_names=receivers,
        receiver_sides=query.receiver_sides, include_matrix=query.scene,
        include_sky=query.sky_mode is not None, discrete=query.sky_mode == "tregenza145",
        ray_batch_size=options.batch_size, max_total_rays=budget.rays, max_time_ms=budget.time_ms,
        device=solver.device, bvh=solver.bvh, cuda_async=solver.cuda_async,
        gpu_raygen=solver.gpu_raygen, auto_tune=solver.auto_tune,
        tune_budget_ms=solver.tune_budget_ms,
        signature=(s, query, solver.device, solver.bvh, solver.cuda_async, solver.gpu_raygen),
    )
