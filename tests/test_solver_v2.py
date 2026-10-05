"""Direct v2 contracts over one physical scene and one seeded Run pipeline."""
from __future__ import annotations

from dataclasses import replace
from threading import Barrier, Event, Thread
from unittest.mock import patch

import numpy as np
import pytest

from raystrack import (Accuracy, Budget, CapabilityError, Channel, Mesh,
                       Postprocessing, Query, Result, Sampling, Scene,
                       SolveOptions, Solver, SparseValues, Surface)


def square(z=0, *, down=False, radius=0.5, x=0):
    vertices = np.asarray([[x-radius, -radius, z], [x+radius, -radius, z],
                           [x+radius, radius, z], [x-radius, radius, z]], np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], np.int32)
    return Mesh(vertices, faces[:, ::-1] if down else faces)


def pair_scene(*, blocked=False, receiver_down=True):
    meshes = {"A": square(), "B": square(1, down=receiver_down)}
    if blocked:
        meshes["blocker"] = square(0.5, radius=10)
    return Scene.from_meshes(meshes)


def fixed_options(*, replicas=4, rays=7, batch=19, seed=19, mode="fair"):
    # The preserved sampling grid has at least 4x4 cells. Zero delta tolerance
    # prevents early convergence and gives every requested row an exact cap.
    return SolveOptions(sampling=Sampling(density=1, rays_per_cell=rays, seed=seed, mode=mode),
                        accuracy=Accuracy(max_replicates=replicas, min_replicates=1,
                                          tolerance=0, mode="delta"), batch_size=batch)


def row_cap(options):
    return 16 * options.sampling.rays_per_cell * options.accuracy.max_replicates


def assert_same_snapshot(actual, expected):
    assert actual.sender_ids == expected.sender_ids
    assert actual.channels == expected.channels
    np.testing.assert_array_equal(actual.coverage, expected.coverage)
    np.testing.assert_array_equal(actual.dense(), expected.dense())
    np.testing.assert_array_equal(actual.data.errors, expected.data.errors)
    assert actual.cumulative_rays == expected.cumulative_rays
    assert actual.statistics["emitters"] == expected.statistics["emitters"]


@pytest.mark.parametrize("acceleration", ["flat", "instanced"])
def test_pair_row_matrix_share_identical_complete_emitter_prefixes(acceleration):
    scene = Scene.from_meshes({"A": square(), "B": square(1, down=True), "C": square(2, x=4)})
    options = fixed_options()
    per_row = row_cap(options)
    with Solver(scene, device="cpu", acceleration=acceleration, auto_tune=False) as solver:
        pair = solver.solve(Query.pair("A", "B"), options, Budget(rays=per_row))
        row = solver.solve(Query.row("A"), options, Budget(rays=per_row))
        matrix = solver.solve(Query.matrix(), options, Budget(rays=3 * per_row))
    for side in ("front", "back"):
        channel = Channel("surface", "B", side)
        assert pair.value("A", channel) == row.value("A", channel) == matrix.value("A", channel)
        assert pair.error("A", channel) == row.error("A", channel) == matrix.error("A", channel)
    for result in (pair, row, matrix):
        assert result.statistics["emitters"]["A"]["rays"] == per_row
        assert result.statistics["emitters"]["A"]["replicates"] == 4
        assert result.statistics["emitters"]["A"]["ray_offset"] == 0
    assert matrix.rays_used == 3 * per_row


@pytest.mark.parametrize("sky_mode", ["merged", "tregenza145"])
def test_shared_scene_sky_equals_separate_queries_for_same_sample_prefix(sky_mode):
    scene = pair_scene()
    options = fixed_options()
    budget = Budget(rays=row_cap(options))
    with Solver(scene, device="cpu", auto_tune=False) as solver:
        shared = solver.solve(Query.row("A", sky=sky_mode), options, budget)
        matrix = solver.solve(Query.row("A"), options, budget)
        sky = solver.solve(Query.sky(("A",), discrete=sky_mode == "tregenza145"), options, budget)
    for channel in shared.channels:
        if channel.kind == "surface":
            assert shared.value("A", channel) == matrix.value("A", channel)
            assert shared.error("A", channel) == matrix.error("A", channel)
        elif channel.kind == "sky":
            assert shared.value("A", channel) == sky.value("A", channel)
            assert shared.error("A", channel) == sky.error("A", channel)
    assert shared.rays_used == matrix.rays_used == sky.rays_used == row_cap(options)
    assert sum(shared.row("A").values()) == pytest.approx(1)


@pytest.mark.parametrize("acceleration", ["flat", "instanced"])
def test_pair_filter_keeps_blocker_and_accounts_for_unrequested_hits(acceleration):
    options = fixed_options(rays=16)
    blocked = pair_scene(blocked=True)
    channel = Channel("surface", "B", "front")
    with Solver(blocked, device="cpu", acceleration=acceleration, auto_tune=False) as solver:
        pair = solver.solve(Query.pair("A", "B"), options, Budget(rays=row_cap(options)))
        row = solver.solve(Query.row("A", ("B",), sky="merged"), options, Budget(rays=row_cap(options)))
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        visible = solver.solve(Query.pair("A", "B"), options, Budget(rays=row_cap(options)))
    assert pair.value("A", channel) == row.value("A", channel) == 0
    assert visible.value("A", channel) > 0.1
    assert pair.value("A", Channel("unrequested")) > 0.99
    assert sum(pair.row("A").values()) == pytest.approx(1)
    assert sum(row.row("A").values()) == pytest.approx(1)


def test_receiver_side_filter_records_hits_on_unrequested_back_side():
    scene = pair_scene(receiver_down=False)
    options = fixed_options()
    with Solver(scene, device="cpu", auto_tune=False) as solver:
        front = solver.solve(Query.pair("A", "B", receiver_sides=("front",)),
                             options, Budget(rays=row_cap(options)))
        both = solver.solve(Query.pair("A", "B"), options, Budget(rays=row_cap(options)))
    assert front.value("A", Channel("surface", "B", "front")) == 0
    assert front.value("A", Channel("unrequested")) == both.value("A", Channel("surface", "B", "back"))
    assert front.value("A", Channel("unrequested")) > 0
    assert sum(front.row("A").values()) == pytest.approx(1)


def test_sky_only_keeps_accounting_for_all_unrequested_surface_hits():
    options = fixed_options()
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        result = solver.solve(Query.sky(("A",)), options, Budget(rays=row_cap(options)))
        scene = solver.solve(Query.pair("A", "B"), options, Budget(rays=row_cap(options)))
    assert result.value("A", Channel("unrequested")) == scene.value("A", Channel("surface", "B", "front"))
    assert result.value("A", Channel("unrequested")) > 0
    assert sum(result.row("A").values()) == pytest.approx(1)


@pytest.mark.parametrize("budget", [Budget(rays=0), Budget(rays=100, time_ms=0)])
def test_zero_budget_is_uncomputed_and_allocates_no_backend(budget):
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        run = solver.start(Query.row("A", sky="merged"), fixed_options())
        with patch.object(solver, "_sync_scene", side_effect=AssertionError("zero budget prepared geometry")):
            result = run.advance(budget)
        assert result.rays_used == result.cumulative_rays == 0
        assert result.coverage.tolist() == [0]
        assert all(value is None for value in result.row("A").values())
        assert all(result.error("A", channel) is None for channel in result.channels)
        assert np.isnan(result.dense()).all()
        assert result.execution == {}
        sampled = run.advance(Budget(rays=1))
        assert sampled.rays_used == sampled.cumulative_rays == 1
        assert sampled.coverage.tolist() == [1]
        assert sampled.value("A", Channel("surface", "B", "back")) == 0
        assert sampled.error("A", Channel("surface", "B", "back")) is None


def test_only_complete_independent_replicates_establish_standard_errors():
    options = fixed_options(replicas=4)
    one_replica = 16 * options.sampling.rays_per_cell
    channel = Channel("surface", "B", "back")
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        run = solver.start(Query.row("A", sky="merged"), options)
        partial = run.advance(Budget(rays=one_replica - 1))
        first = run.advance(Budget(rays=1))
        second = run.advance(Budget(rays=one_replica))
    assert partial.statistics["emitters"]["A"]["matrix"]["replicates"] == 0
    assert first.statistics["emitters"]["A"]["matrix"]["replicates"] == 1
    assert first.error("A", channel) is None
    assert first.statistics["emitters"]["A"]["matrix"]["stderr"] is None
    assert second.statistics["emitters"]["A"]["matrix"]["replicates"] == 2
    assert second.value("A", channel) == second.error("A", channel) == 0
    assert first.cumulative_rays == one_replica
    with pytest.raises(TypeError):
        first.statistics["emitters"]["A"]["rays"] = 0


def test_partial_new_replicate_marks_positive_and_zero_channel_errors_unknown_until_complete():
    options = fixed_options(replicas=4, rays=16)
    replica_rays = 16 * options.sampling.rays_per_cell
    positive, zero = Channel("surface", "B", "front"), Channel("surface", "B", "back")
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        run = solver.start(Query.pair("A", "B"), options)
        complete = run.advance(Budget(rays=2 * replica_rays))
        partial = run.advance(Budget(rays=1))
        finished = run.advance(Budget(rays=replica_rays - 1))
    assert complete.value("A", positive) > 0
    assert complete.error("A", positive) is not None
    assert complete.error("A", zero) == 0
    assert partial.value("A", positive) > 0 and partial.value("A", zero) == 0
    assert partial.error("A", positive) is partial.error("A", zero) is None
    assert partial.coverage.tolist() == [1]
    partial_stats = partial.statistics["emitters"]["A"]["matrix"]
    assert partial_stats["replicates"] == 2 and partial_stats["pending_rays"] == 1
    assert partial_stats["completed_rays"] == 2 * replica_rays
    assert partial_stats["stderr"] is not None
    assert partial_stats["stderr_basis"] == "completed_replicates"
    assert finished.cumulative_rays == 3 * replica_rays
    assert finished.statistics["emitters"]["A"]["matrix"]["pending_rays"] == 0
    assert finished.statistics["emitters"]["A"]["matrix"]["replicates"] == 3
    assert finished.error("A", positive) is not None
    assert finished.error("A", zero) == 0


def test_reported_sampling_error_matches_independent_single_replicate_results():
    options = fixed_options(replicas=4, rays=16)
    channel = Channel("surface", "B", "front")
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        aggregate = solver.solve(Query.pair("A", "B"), options, Budget(rays=row_cap(options)))
        samples = [solver.solve(Query.pair("A", "B"),
                               fixed_options(replicas=1, rays=16, seed=options.sampling.seed + i),
                               Budget(rays=16 * 16)).value("A", channel) for i in range(4)]
    expected_error = np.std(samples, ddof=1) / np.sqrt(len(samples))
    assert expected_error > 0
    assert aggregate.value("A", channel) == np.mean(samples)
    assert aggregate.error("A", channel) == pytest.approx(expected_error, abs=1e-14)
    assert aggregate.statistics["emitters"]["A"]["matrix"]["stderr"] == pytest.approx(expected_error, abs=1e-14)


@pytest.mark.parametrize("mode", ["fair", "adaptive"])
def test_resume_exactly_preserves_scene_and_sky_with_new_batch_sizes(mode):
    options = fixed_options(batch=31, mode=mode)
    smaller = replace(options, batch_size=5)
    query = Query.row("A", sky="tregenza145")
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        run = solver.start(query, options)
        first = run.advance(Budget(rays=37))
        second = run.advance(Budget(rays=91), options=smaller)
        chunks = []
        third = run.advance(Budget(rays=128), progress=chunks.append)
        expected = solver.solve(query, smaller, Budget(rays=256))
    assert (first.rays_used, second.rays_used, third.rays_used) == (37, 91, 128)
    assert (first.cumulative_rays, second.cumulative_rays, third.cumulative_rays) == (37, 128, 256)
    assert sum(chunks) == 128 and max(chunks) <= 5
    assert_same_snapshot(third, expected)
    assert first.statistics["emitters"]["A"]["rays"] == 37


def test_async_cancellation_keeps_valid_completed_prefix_and_spends_no_more_rays():
    options = fixed_options(batch=7)
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        run = solver.start(Query.row("A", sky="merged"), options)
        result = run.submit(Budget(rays=100), progress=lambda _: run.cancel()).result(timeout=30)
        stopped = run.advance(Budget(rays=100))
        expected = solver.solve(Query.row("A", sky="merged"), options, Budget(rays=7))
    assert result.status == stopped.status == "cancelled"
    assert result.rays_used == result.cumulative_rays == 7
    assert stopped.rays_used == 0 and stopped.cumulative_rays == 7
    np.testing.assert_array_equal(result.dense(), expected.dense())
    np.testing.assert_array_equal(stopped.dense(), result.dense())


def test_solver_can_close_from_its_async_progress_callback_without_joining_itself():
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        run = solver.start(Query.row("A"), fixed_options(batch=7))
        result = run.submit(Budget(rays=100), progress=lambda _: solver.close()).result(timeout=30)
        assert result.status == "cancelled"
        assert result.rays_used == result.cumulative_rays == 7
        assert run.result is result
        with pytest.raises(RuntimeError, match="Solver is closed"):
            solver.start(Query.row("A"))
        with pytest.raises(RuntimeError, match="Solver is closed"):
            run.advance(Budget(rays=0))
        solver.close()


def test_concurrent_run_creation_and_geometry_updates_keep_registry_consistent():
    scene = pair_scene()
    rendezvous = Barrier(2)
    runs, errors = [], []
    with Solver(scene, device="cpu", auto_tune=False) as solver:

        def create_runs():
            try:
                for _ in range(30):
                    rendezvous.wait(timeout=5)
                    for _ in range(20):
                        runs.append(solver.start(Query.pair("A", "B"), fixed_options()))
                    rendezvous.wait(timeout=5)
            except BaseException as error:
                errors.append(error)
                rendezvous.abort()

        def mutate_scene():
            try:
                for index in range(30):
                    rendezvous.wait(timeout=5)
                    transform = np.eye(4)
                    transform[0, 3] = index / 1000
                    scene.update_transform("B", transform)
                    rendezvous.wait(timeout=5)
            except BaseException as error:
                errors.append(error)
                rendezvous.abort()

        workers = [Thread(target=create_runs), Thread(target=mutate_scene)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
            assert not worker.is_alive()
        assert not errors
        assert len(runs) == 600 and scene.revision == 30
        stale = 0
        for run in runs:
            if run.scene_revision != scene.revision or run.status == "invalidated":
                stale += 1
                with pytest.raises(RuntimeError, match="geometry changed"):
                    run.advance(Budget(rays=0))
            else:
                assert run.advance(Budget(rays=0)).cumulative_rays == 0
        assert stale >= 580


@pytest.mark.parametrize("mutation", ["transform", "add", "remove"])
def test_geometry_mutation_inside_progress_returns_old_revision_snapshot_and_invalidates_run(mutation):
    scene = pair_scene()
    options = fixed_options(batch=7)
    with Solver(scene, device="cpu", auto_tune=False) as solver:
        run = solver.start(Query.matrix(), options)

        def change_scene(_):
            if mutation == "transform":
                transform = np.eye(4)
                transform[0, 3] = 2
                scene.update_transform("B", transform)
            elif mutation == "add":
                scene.add(Surface("C", square(2)))
            else:
                scene.remove("B")

        result = run.advance(Budget(rays=100), progress=change_scene)
        assert result.status == "invalidated"
        assert result.scene_revision == 0 and scene.revision == 1
        assert result.sender_ids == ("A", "B")
        assert result.rays_used == 1
        assert Channel("surface", "B", "front") in result.channels
        with pytest.raises(RuntimeError, match="geometry changed"):
            run.advance(Budget(rays=1))
        fresh = solver.solve(Query.matrix(), options, Budget(rays=3))
        assert fresh.scene_revision == 1 and fresh.sender_ids == scene.surface_ids
        assert fresh.cumulative_rays == 3


def test_external_geometry_update_cancels_without_waiting_for_executing_scene_lock():
    scene = pair_scene()
    notified = Event()
    scene._subscribe(notified.set)
    workers = []
    with Solver(scene, device="cpu", auto_tune=False) as solver:
        run = solver.start(Query.row("A"), fixed_options(batch=7))

        def on_progress(_):
            worker = Thread(target=scene.update_transform, args=("B", np.eye(4)))
            workers.append(worker)
            worker.start()
            assert notified.wait(5)

        result = run.advance(Budget(rays=100), progress=on_progress)
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
        assert result.status == "invalidated" and result.rays_used == 7
        assert scene.revision == 1
        with pytest.raises(RuntimeError, match="geometry changed"):
            run.advance(Budget(rays=1))


def test_display_labels_do_not_change_scene_ids_or_invalidate_accumulation():
    scene = pair_scene()
    options = fixed_options()
    with Solver(scene, device="cpu", auto_tune=False) as solver:
        run = solver.start(Query.pair("A", "B"), options)
        first = run.advance(Budget(rays=37))
        scene.set_label("A", "Same display label")
        scene.set_label("B", "Same display label")
        resumed = run.advance(Budget(rays=91))
        reference = solver.solve(Query.pair("A", "B"), options, Budget(rays=128))
    assert scene.surface_ids == ("A", "B") and scene.revision == 0
    assert first.scene_revision == resumed.scene_revision == 0
    assert_same_snapshot(resumed, reference)


def test_incompatible_sampling_rejection_preserves_run_for_valid_refinement():
    options = fixed_options()
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        run = solver.start(Query.pair("A", "B"), options)
        run.advance(Budget(rays=37))
        with pytest.raises(ValueError, match="Sampling changed"):
            run.advance(Budget(rays=91), options=replace(options, sampling=replace(options.sampling, seed=20)))
        assert run.cumulative_rays == 37
        resumed = run.advance(Budget(rays=91))
        expected = solver.solve(Query.pair("A", "B"), options, Budget(rays=128))
    assert_same_snapshot(resumed, expected)


def test_reciprocity_rejects_partial_query_and_preview_then_retains_raw_counts():
    options = replace(fixed_options(), postprocessing=Postprocessing("bidirectional"))
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        with pytest.raises(ValueError, match="all sender"):
            solver.start(Query.pair("A", "B"), options)
        run = solver.start(Query.matrix(), options)
        with pytest.raises(ValueError, match="complete solve"):
            run.advance(Budget(rays=1))
        assert run.cumulative_rays == 1
        raw_options = replace(options, postprocessing=Postprocessing())
        raw = run.advance(Budget(rays=127), options=raw_options)
        expected = solver.solve(Query.matrix(), raw_options, Budget(rays=128))
    assert_same_snapshot(raw, expected)
    assert raw.provenance == ()


def test_bidirectional_reciprocity_has_provenance_and_unknown_transformed_errors():
    options = fixed_options()
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        raw = solver.solve(Query.matrix(), options, Budget(rays=2 * row_cap(options)))
        reciprocal = solver.solve(Query.matrix(), replace(options, postprocessing=Postprocessing("bidirectional")),
                                  Budget(rays=2 * row_cap(options)))
    ab, ba = Channel("surface", "B", "front"), Channel("surface", "A", "front")
    expected = 0.5 * (raw.value("A", ab) + raw.value("B", ba))
    assert reciprocal.value("A", ab) == reciprocal.value("B", ba) == expected
    assert reciprocal.error("A", ab) is reciprocal.error("B", ba) is None
    assert reciprocal.provenance[0]["algorithm"] == "bidirectional"
    assert reciprocal.provenance[0]["uncertainty"] == "unknown after postprocessing"


def enclosure():
    vertices = [
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],
        [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
        [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
    ]
    return Scene.from_meshes({str(i): Mesh(v, [[0, 1, 2], [0, 2, 3]]) for i, v in enumerate(vertices)})


def test_rowsum_reciprocity_rejects_open_scene_and_sky_queries():
    options = replace(fixed_options(), postprocessing=Postprocessing("rowsum"))
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        with pytest.raises(ValueError, match="sky"):
            solver.start(Query.matrix(sky="merged"), options)
        with pytest.raises(ValueError, match="closed scene"):
            solver.solve(Query.matrix(), options, Budget(rays=2 * row_cap(options)))


def test_rowsum_reciprocity_on_complete_closed_enclosure_preserves_energy():
    scene = enclosure()
    options = replace(fixed_options(rays=16), postprocessing=Postprocessing("rowsum"))
    with Solver(scene, device="cpu", auto_tune=False) as solver:
        result = solver.solve(Query.matrix(), options, Budget(rays=len(scene) * row_cap(options)))
    assert np.all(result.coverage == 1)
    for sender in result.sender_ids:
        assert sum(result.row(sender).values()) == pytest.approx(1, abs=1e-6)
        assert result.value(sender, Channel("rest")) == 0
        for receiver in result.sender_ids:
            front = Channel("surface", receiver, "front")
            reverse = Channel("surface", sender, "front")
            assert result.value(sender, front) == pytest.approx(result.value(receiver, reverse), abs=1e-12)
            assert result.error(sender, front) is None
    assert result.provenance[0]["algorithm"] == "rowsum"


@pytest.mark.parametrize("device", ["gpu", "cuda", "vulkan", "metal"])
def test_area_pair_explicit_gpu_is_capability_error_without_loading_device(device):
    options = replace(fixed_options(), sampling=Sampling(strategy="area_pair", pair_samples=16))
    with Solver(pair_scene(), device=device) as solver:
        with pytest.raises(CapabilityError, match="CPU only"):
            solver.start(Query.pair("A", "B"), options)


@pytest.mark.parametrize("receiver_down", [True, False])
def test_area_pair_matches_unit_parallel_square_reference_and_structured_side(receiver_down):
    options = SolveOptions(sampling=Sampling(strategy="area_pair", pair_samples=16384, seed=19),
                           accuracy=Accuracy(max_replicates=4, min_replicates=1, mode="delta", tolerance=0))
    with Solver(pair_scene(receiver_down=receiver_down), device="cpu", auto_tune=False) as solver:
        result = solver.solve(Query.pair("A", "B"), options, Budget(rays=4 * 16384))
    side = "front" if receiver_down else "back"
    opposite = "back" if receiver_down else "front"
    # Closed-form parallel unit-square view factor at unit separation, also a
    # preserved analytical validation case in the repository.
    assert result.value("A", Channel("surface", "B", side)) == pytest.approx(0.199824895698387, abs=0.001)
    assert result.value("A", Channel("surface", "B", opposite)) == 0
    assert result.rays_used == result.cumulative_rays == 65536
    assert result.execution["backend"] == "cpu"
    assert result.statistics["emitters"]["A"]["matrix"]["replicates"] == 4
    assert result.error("A", Channel("surface", "B", side)) is not None


def test_area_pair_visibility_retains_unrequested_blocker():
    options = SolveOptions(sampling=Sampling(strategy="area_pair", pair_samples=64),
                           accuracy=Accuracy(max_replicates=2, min_replicates=1, mode="delta", tolerance=0))
    with Solver(pair_scene(blocked=True), device="cpu", auto_tune=False) as solver:
        result = solver.solve(Query.pair("A", "B"), options, Budget(rays=128))
    assert result.value("A", Channel("surface", "B", "front")) == 0
    assert result.error("A", Channel("surface", "B", "front")) == 0


def test_area_pair_weighted_prefix_is_bitwise_independent_of_chunk_sizes():
    options = SolveOptions(sampling=Sampling(strategy="area_pair", pair_samples=1024, seed=19),
                           accuracy=Accuracy(max_replicates=10, min_replicates=1, mode="delta", tolerance=0),
                           batch_size=512)
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        run = solver.start(Query.pair("A", "B"), options)
        first = run.advance(Budget(rays=333))
        resumed = run.advance(Budget(rays=1667), options=replace(options, batch_size=7))
        same_batch = solver.solve(Query.pair("A", "B"), replace(options, batch_size=7), Budget(rays=2000))
        larger_batch = solver.solve(Query.pair("A", "B"), replace(options, batch_size=777), Budget(rays=2000))
    assert first.rays_used == first.cumulative_rays == 333
    assert resumed.rays_used == 1667 and resumed.cumulative_rays == 2000
    assert_same_snapshot(resumed, same_batch)
    assert_same_snapshot(resumed, larger_batch)


def test_import_unknown_errors_and_explicit_zero_remain_distinct_from_uncomputed():
    channel = Channel("surface", "receiver", "back")
    result = Result(sender_ids=("known", "unknown", "uncomputed"), channels=(channel,),
                    data=SparseValues(np.asarray([0, 1]), np.asarray([0, 0]),
                                      np.asarray([0.0, 0.2]), np.asarray([np.nan, np.nan])),
                    coverage=np.asarray([1, -1, 0], np.int8),
                    statistics={"emitters": {"known": {"matrix": {"replicates": 4}}}},
                    rays_used=None, cumulative_rays=None, converged=None)
    assert result.value("known", channel) == 0
    assert result.value("unknown", channel) == 0.2
    assert result.value("uncomputed", channel) is None
    assert all(result.error(sender, channel) is None for sender in result.sender_ids)
    np.testing.assert_array_equal(result.dense(), [[0.0], [0.2], [np.nan]])


def test_empty_sender_selection_is_completed_without_tracing():
    with Solver(pair_scene(), device="cpu", auto_tune=False) as solver:
        result = solver.solve(Query.matrix(senders=()), fixed_options(), Budget(rays=100))
    assert result.sender_ids == ()
    assert result.coverage.size == 0
    assert result.rays_used == result.cumulative_rays == 0
    assert result.converged and result.status == "converged"
