# Releasing Raystrack

The Python package and compiled Grasshopper package share the release version.
Version 2 is built from this repository; `raystrack_grasshopper` contains the
previous v1 user-object integration.

## Prepare and verify

Update `pyproject.toml`, `grasshopper/manifest.yml`, the C# assembly info and
project, and `tools/grasshopper/build.py` together. Update the installation
instructions and changelog. Versions already uploaded to PyPI or Yak must not
be reused.

Run the regression suite from the repository root:

```powershell
$env:PYTHONPATH = 'src'
$env:NUMBA_NUM_THREADS = '4'
$env:RAYSTRACK_TEST_NATIVE_GH = '1'
python -m pytest -q
```

The native checks require a local Rhino 8 installation. Device-specific tests
can skip when a backend is absent. Record the backend and simulation scope.

Build and check the Python distributions:

```powershell
python -m pip install build twine
python -m build --outdir dist/python
python -m twine check dist/python/*
```

Build the Windows Grasshopper bundle on a machine with Rhino 8 installed:

```powershell
python tools/grasshopper/build.py --gpu-smoke
```

Use `--offline` if the verified Python archive and binary wheel cache is already
populated. This produces a GHA with an adjacent runtime, a standalone ZIP, and
a Yak package. The build verifies its bundled interpreter and records source
and payload hashes. Rhino's assembly references are never redistributed.

Install and verify the Yak package locally before publication. Include the
boxes definition and guides in the package. Saved Run inputs are disarmed when
opened; switch false, then true to solve.

## Commit, push, and publish Python

Commit the release sources, docs, examples, and validation evidence. Push the
branch and wait for the publishing workflow's Python 3.9, 3.12, and 3.13 tests
and distribution checks to pass. Fast-forward `main` without rewriting history.

An annotated `v2.0.0` tag matching `pyproject.toml` triggers
`.github/workflows/publish.yml`. Its test and build jobs must pass before the
PyPI job uses the configured `pypi` environment and trusted publishing. A manual
workflow dispatch publishes to TestPyPI instead. No PyPI password or token is
stored in this repository.

Check the workflow result and the published version on PyPI before reporting
the release as published. Create a GitHub release for the tag and attach the
verified Python distributions, Windows ZIP, and Yak file with their SHA-256
checksums.

## Publish Yak

Complete the Rhino Account login yourself if the cached session has expired:

```powershell
& 'C:\Program Files\Rhino 8\System\Yak.exe' login
```

Push the verified package to the test server, check that exact version, then
push it to the public server:

```powershell
& 'C:\Program Files\Rhino 8\System\Yak.exe' push --source https://test.yak.rhino3d.com dist/grasshopper/raystrack-2.0.0-rh8_35-win.yak
& 'C:\Program Files\Rhino 8\System\Yak.exe' search --source https://test.yak.rhino3d.com --all raystrack
& 'C:\Program Files\Rhino 8\System\Yak.exe' push dist/grasshopper/raystrack-2.0.0-rh8_35-win.yak
& 'C:\Program Files\Rhino 8\System\Yak.exe' search --all raystrack
```

Yak may infer a different Rhino target from the installed build references;
use the actual filename reported by the builder. Publication is complete only
after the public registry lists the new version. GitHub runners do not have the
licensed local Rhino references needed by this compiled build, so Yak is built
and published from the verified Windows artifact.
