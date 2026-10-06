# Changelog

## 2.0.0

Raystrack 2 introduces a reusable scene and solver API and a compiled Rhino 8
Grasshopper integration.

- Use `Mesh`, `Surface`, `Scene`, `Solver`, `Query`, and `SolveOptions` to select
  senders, receivers, sky channels, sampling, and convergence independently.
- Keep immutable result snapshots and resume a `Run` with an additional ray
  budget. CPU, CUDA, and portable GPU tracing share the execution pipeline.
- Include other faces of the emitting mesh in self-viewing and occlusion.
  Same-ID pair queries work with cosine tracing and the CPU area-pair estimator.
- Save and load geometry, transforms, settings, results, and metadata in the
  versioned store; import supported v1 data through isolated readers.
- Install 15 compiled RS components with an isolated bundled Python runtime
  through Rhino Package Manager on Windows, Rhino 8.35 or later.
- Configure sky channels with RS Sky and preview or bake native spherical
  patches and numbered labels. Convert stored surfaces or scenes with RS To Brep.
- Read runtime capabilities as explained text and use the refreshed component
  icons. The boxes example covers exterior and interior emission.

This release replaces the v1 calculation functions and parameter classes.
Read [the migration guide](docs/v2-migration.md) before upgrading. Previous
Grasshopper user objects need to be rebuilt with the compiled RS components.

Physical Vulkan has been validated locally. CUDA kernel behavior has been
checked in simulation; physical CUDA and Metal validation remain outstanding.
