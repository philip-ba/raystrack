"""Build the Rhino 8 Windows GHA, standalone bundle and local Yak package.

The build downloads official CPython and pinned binary wheels. Rhino uses the
bundled interpreter without first-use package installation. Only --install installs
the package locally; this tool never publishes packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[2]
GH = ROOT / "grasshopper"
PYTHON_VERSION = "3.12.10"
PYTHON_ARCHIVE = f"python-{PYTHON_VERSION}-embed-amd64.zip"
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/{PYTHON_ARCHIVE}"
# Digest published in Python.org's release Sigstore bundle for this exact file.
PYTHON_SHA256 = "4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3"
PYTHON_RELEASE = "https://www.python.org/downloads/release/python-31210/"
VERSION = "2.0.0"
ASSEMBLY = "Raystrack.Components.gha"


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path = ROOT, env=None, label=None) -> None:
    """Run a build subprocess and propagate failures with its output visible.

    ``label`` replaces the printed command for embedded smoke-test source; it
    does not change the executed argument list or the subprocess environment.
    """
    print("+ " + (label or subprocess.list2cmdline(command)), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_logged(command: list[str], *, cwd: Path, log: Path) -> None:
    """Keep Yak's several-thousand-line file tree in a build log."""
    print("+ " + subprocess.list2cmdline(command), flush=True)
    with log.open("wb") as output:
        completed = subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
    if completed.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        raise RuntimeError(f"Command failed; full output in {log}\n{tail}")
    print(f"Completed; full output: {log}", flush=True)


def reset_directory(path: Path, parent: Path) -> None:
    """Delete only an explicitly checked build child, never its output parent."""
    target = path.resolve()
    boundary = parent.resolve()
    if target == boundary or boundary not in target.parents:
        raise ValueError(f"Build directory is outside the output directory: {target}")
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink build directory: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)


def download_python(cache: Path, offline=False) -> Path:
    """Return the cached official CPython archive after verifying its release hash.

    Online builds download a missing archive to a temporary filename before
    accepting it. Offline builds require an existing, valid cache entry.
    """
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / PYTHON_ARCHIVE
    if not archive.exists():
        if offline:
            raise FileNotFoundError(f"Offline Python archive is missing: {archive}")
        partial = archive.with_suffix(".partial")
        print(f"Downloading {PYTHON_URL}", flush=True)
        urllib.request.urlretrieve(PYTHON_URL, partial)
        if sha256(partial) != PYTHON_SHA256:
            partial.unlink()
            raise ValueError("Official Python archive SHA-256 does not match the release digest")
        partial.replace(archive)
    if sha256(archive) != PYTHON_SHA256:
        raise ValueError(f"Python archive SHA-256 mismatch: {archive}")
    return archive


def extract_zip(archive: Path, destination: Path) -> None:
    """Extract a build archive while rejecting traversal, links and drive paths."""
    base = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or ":" in name or ".." in name.split("/"):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink is not supported: {member.filename}")
            target = (destination / name).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Archive member escapes extraction directory: {member.filename}")
        bundle.extractall(destination)


def write_python_path(runtime: Path) -> None:
    """Isolate the embedded interpreter to its own relative library directories.

    The explicit site-packages path supports bundled dependency DLL loading,
    while CPython's ``_pth`` isolation excludes user and system Python imports.
    """
    # _pth turns on CPython isolation. All paths are relative to the shipped runtime,
    # so an unrelated system Python, cwd or PYTHONPATH cannot supply dependencies.
    (runtime / "python312._pth").write_text(
        "python312.zip\n.\nLib/site-packages\nimport site\n", encoding="utf-8"
    )


def dependency_wheels(cache: Path, offline=False) -> list[Path]:
    """Prepare pinned Windows CPython 3.12 binary wheels and return cached files.

    Downloads use the build interpreter's pip; the shipped interpreter never
    invokes pip. Offline installation later resolves requirements from this
    cache and fails if a required wheel is unavailable.
    """
    wheels = cache / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    if not offline:
        run([
            sys.executable, "-m", "pip", "download", "--only-binary=:all:",
            "--platform", "win_amd64", "--python-version", "3.12",
            "--implementation", "cp", "--abi", "cp312", "--dest", str(wheels),
            "--requirement", str(GH / "runtime-requirements.txt"),
        ])
    available = sorted(wheels.glob("*.whl"))
    if not available:
        raise FileNotFoundError(f"No binary dependency wheels in {wheels}")
    return available


def local_wheel(destination: Path) -> Path:
    """Build the current numerical package from a fresh source-only temporary tree.

    A fresh tree prevents setuptools from reusing stale ignored ``build/lib``
    modules left over from another API version. ``destination`` must contain no
    older Raystrack wheel, so the installed wheel is unambiguous.
    """
    # Build in a fresh tree: setuptools otherwise reuses ignored build/lib content.
    with tempfile.TemporaryDirectory(prefix="raystrack-gh-source-") as temporary:
        source = Path(temporary)
        shutil.copytree(ROOT / "src" / "raystrack", source / "src" / "raystrack",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.nbc", "*.nbi"))
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(ROOT / name, source / name)
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
             "--wheel-dir", str(destination), str(source)])
    matches = sorted(destination.glob("raystrack-*.whl"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one fresh local raystrack wheel, found {len(matches)}")
    return matches[0]


def copy_licenses(runtime: Path, site_packages: Path) -> None:
    """Collect CPython, Raystrack and wheel license notices in the bundled runtime.

    Dependency directory names and nested license paths are retained to identify
    each license's distribution. An absent CPython license rejects the build.
    """
    licenses = runtime / "licenses"
    licenses.mkdir()
    python_license = runtime / "LICENSE.txt"
    if not python_license.is_file():
        raise FileNotFoundError("Official embeddable Python license is missing")
    shutil.copy2(python_license, licenses / "CPython-LICENSE.txt")
    for distribution in sorted(site_packages.glob("*.dist-info")):
        for path in sorted(distribution.rglob("*")):
            if path.is_file() and ("license" in path.name.lower() or "notice" in path.name.lower()):
                target = licenses / distribution.name / path.relative_to(distribution)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    shutil.copy2(ROOT / "LICENSE", licenses / "Raystrack-LICENSE.txt")


def build_runtime(runtime: Path, cache: Path, offline=False) -> None:
    """Populate an empty runtime with CPython, dependencies and current Raystrack.

    Install binary wheels using the host pip, remove host-specific launchers and
    installation URLs, and write provenance plus hashes of runtime payload
    files. The resulting directory runs without a separate Python installation
    or first-launch downloads.
    """
    python_archive = download_python(cache, offline)
    extract_zip(python_archive, runtime)
    write_python_path(runtime)
    wheels = dependency_wheels(cache, offline)
    site_packages = runtime / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="raystrack-gh-wheel-") as temporary:
        package_wheel = local_wheel(Path(temporary))
        run([
            sys.executable, "-m", "pip", "install", "--no-index", "--no-compile", "--no-warn-conflicts",
            "--only-binary=:all:", "--platform", "win_amd64", "--python-version", "3.12",
            "--implementation", "cp", "--abi", "cp312", "--target", str(site_packages),
            "--find-links", str(cache / "wheels"), "--requirement",
            str(GH / "runtime-requirements.txt"), str(package_wheel),
        ])
        provenance = {
            "format_version": 1,
            "platform": "win_amd64",
            "python": {"version": PYTHON_VERSION, "url": PYTHON_URL,
                       "release": PYTHON_RELEASE, "sha256": PYTHON_SHA256},
            "wheels": [{"filename": path.name, "sha256": sha256(path)} for path in wheels],
            "raystrack": {"filename": package_wheel.name, "sha256": sha256(package_wheel)},
        }
    # pip's local direct_url.json otherwise leaks the temporary build machine path.
    for direct_url in site_packages.glob("*.dist-info/direct_url.json"):
        direct_url.unlink()
    # Console entry points created by the host pip embed its absolute interpreter
    # path. The worker uses modules directly and needs none of these launchers.
    for name in ("bin", "Scripts"):
        launchers = site_packages / name
        if launchers.is_dir():
            shutil.rmtree(launchers)
    copy_licenses(runtime, site_packages)
    provenance["files"] = {
        str(path.relative_to(runtime)).replace("\\", "/"): sha256(path)
        for path in sorted(runtime.rglob("*")) if path.is_file()
    }
    (runtime / "runtime-manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def reference_paths(rhino: Path) -> list[Path]:
    """Find required Rhino 8 build references without copying them into the bundle."""
    system = rhino / "System"
    grasshopper = rhino / "Plug-ins" / "Grasshopper"
    paths = [system / "RhinoCommon.dll", system / "Newtonsoft.Json.dll",
             grasshopper / "Grasshopper.dll", grasshopper / "GH_IO.dll"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Rhino 8 build reference is missing: {path}")
    return paths


def build_assembly(stage: Path, rhino: Path, compiler: Path) -> Path:
    """Compile the component sources and original icon resources into a staged GHA.

    Rhino supplies the external assembly references. A temporary response file
    handles spaces in paths and is removed after compilation. The compiler must
    support the project's C# language version; no Rhino DLLs are redistributed.
    """
    sources = sorted(GH.glob("*.cs"))
    if not sources:
        raise FileNotFoundError("No Grasshopper C# component sources found")
    if not compiler.is_file():
        raise FileNotFoundError(f"C# compiler not found: {compiler}; use --compiler")
    assembly = stage / ASSEMBLY
    # .NET Framework csc.exe is available without a dotnet SDK and compiles C# 5.
    documentation = stage / "Raystrack.Components.xml"
    options = ["/nologo", "/target:library", "/optimize+", "/platform:anycpu",
               '/out:"' + str(assembly) + '"', '/doc:"' + str(documentation) + '"']
    for reference in reference_paths(rhino):
        options.append('/reference:"' + str(reference) + '"')
    options += ["/reference:System.dll", "/reference:System.Core.dll",
                "/reference:System.Drawing.dll", "/reference:System.Windows.Forms.dll"]
    for icon in sorted((GH / "icons").glob("*.png")):
        options.append('/resource:"' + str(icon) + '",Raystrack.Icons.' + icon.name)
    options.extend('"' + str(source) + '"' for source in sources)
    with tempfile.TemporaryDirectory(prefix="raystrack-gh-csc-") as temporary:
        response = Path(temporary) / "build.rsp"
        response.write_text("\n".join(options) + "\n", encoding="utf-8-sig")
        run([str(compiler), "@" + str(response)])
    return assembly


def smoke_runtime(runtime: Path, *, gpu=False) -> None:
    """Verify exact resumed sampling using the shipped, isolated interpreter.

    The child process runs outside the source checkout and keeps numerical
    caches in a temporary directory. ``gpu=True`` explicitly requires Vulkan;
    GPU failure cannot silently pass through a CPU fallback.
    """
    script = r'''
import sys, json
from importlib.metadata import version
import numpy as np
import numba
from raystrack import Scene, Mesh, Solver, Query, SolveOptions, Sampling, Accuracy, Budget, Channel
assert version("raystrack") == RELEASE_VERSION
vertices = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
faces = [[0,1,2],[0,2,3]]
upper = [[x,y,1] for x,y,z in vertices]
scene = Scene.from_meshes({"A": Mesh(vertices,faces), "B": Mesh(upper,[[0,2,1],[0,3,2]])})
options = SolveOptions(sampling=Sampling(seed=11))
with Solver(scene, device=DEVICE) as solver:
    run = solver.start(Query.pair("A","B"), options)
    run.advance(Budget(rays=19))
    result = run.advance(Budget(rays=45))
    fresh = solver.start(Query.pair("A","B"), options).advance(Budget(rays=64))
    assert result.cumulative_rays == fresh.cumulative_rays == 64
    assert np.array_equal(result.data.estimates, fresh.data.estimates)
box_vertices = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]
box_faces = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
             [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
box = Scene.from_meshes({"box": Mesh(box_vertices,box_faces)})
inside = SolveOptions(Sampling(density=2,rays_per_cell=16,flip_faces=True),
                      Accuracy(max_replicates=2,min_replicates=2,tolerance=0))
with Solver(box,device=DEVICE,acceleration="instanced",auto_tune=False) as solver:
    self_view = solver.solve(Query.pair("box","box"),inside)
    assert self_view.value("box",Channel("surface","box","back")) == 1
    assert self_view.value("box",Channel("surface","box","front")) == 0
    assert self_view.value("box",Channel("rest")) == 0
print(json.dumps({"python": sys.version.split()[0], "raystrack":version("raystrack"), "numpy": np.__version__, "numba": numba.__version__, "device": DEVICE, "resume": "exact", "closed_box_self_view":1,"closed_box_escape":0}))
'''
    script = script.replace("DEVICE", repr("vulkan" if gpu else "cpu"))
    script = script.replace("RELEASE_VERSION", repr(VERSION))
    with tempfile.TemporaryDirectory(prefix="raystrack-gh-relocated-") as temporary:
        temporary_path = Path(temporary)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["NUMBA_NUM_THREADS"] = "2"
        env["NUMBA_CACHE_DIR"] = str(temporary_path / "numba-cache")
        env["TI_OFFLINE_CACHE_FILE_PATH"] = str(temporary_path / "taichi-cache")
        run([str(runtime / "python.exe"), "-I", "-B", "-c", script], cwd=temporary_path, env=env,
            label="Bundled runtime correctness check (" + ("Vulkan" if gpu else "CPU") + ")")


def archive_bundle(stage: Path, destination: Path) -> None:
    """Write a relocatable ZIP whose GHA and runtime stay adjacent after extraction.

    Archive member paths are relative to the stage root. Existing Yak artifacts
    are excluded so a package cannot recursively contain an earlier package.
    """
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.suffix != ".yak":
                bundle.write(path, str(path.relative_to(stage)).replace("\\", "/"))


def write_package_provenance(stage: Path) -> None:
    """Record the current build's source identity and non-runtime payload hashes.

    C# source, numerical Python source and original embedded icons have separate
    hashes. Runtime payload integrity is covered by the hashed runtime manifest.
    The package manifest itself is excluded to avoid a self-referential hash;
    Yak-generated distribution metadata is identified separately from payload.
    Earlier native-host reports remain historical evidence, not new test claims.
    """
    sources = sorted(GH.glob("*.cs")) + sorted((ROOT / "src" / "raystrack").rglob("*.py"))
    sources += [path for path in (GH / "manifest.yml", GH / "runtime-requirements.txt",
                                  GH / "Raystrack.Components.csproj", ROOT / "pyproject.toml")
                if path.is_file()]
    icons = sorted((GH / "icons").glob("*.png"))
    provenance = {
        "format_version": 1,
        "grasshopper_version": VERSION,
        "python_runtime_version": PYTHON_VERSION,
        "platform": "win_amd64",
        # Yak regenerates manifest.yml with platform/assembly GUID fields. Its
        # source is hashed above; the immutable payload stays identical in ZIP/Yak.
        "generated_distribution_metadata": ["manifest.yml"],
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sources},
        "original_icon_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in icons},
        "files": {
            path.relative_to(stage).as_posix(): sha256(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file() and "runtime" not in path.relative_to(stage).parts
            and path.name not in ("package-manifest.json", "manifest.yml") and path.suffix != ".yak"
        },
    }
    runtime_manifest = stage / "runtime" / "runtime-manifest.json"
    if runtime_manifest.is_file():
        provenance["runtime_manifest_sha256"] = sha256(runtime_manifest)
    (stage / "package-manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def main(argv=None) -> int:
    """Build selected Windows artifacts and optionally install the local Yak file.

    Complete builds include the isolated runtime, guides and portable examples.
    Runtime-only and assembly-only modes support development independently;
    installation requires a complete Yak package and never publishes it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "grasshopper",
                        help="Directory for generated GHA, standalone ZIP and Yak artifacts")
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "raystrack-gh-downloads",
                        help="Build download cache for the verified CPython archive and binary wheels")
    parser.add_argument("--rhino", type=Path, default=Path("C:/Program Files/Rhino 8"),
                        help="Rhino 8 installation directory supplying build references and Yak")
    parser.add_argument("--compiler", type=Path,
                        default=Path(os.environ.get("WINDIR", "C:/Windows")) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe",
                        help="C# compiler executable; defaults to the Windows .NET Framework compiler")
    parser.add_argument("--yak", type=Path, help="Override Rhino's Yak executable for package building and local installation")
    parser.add_argument("--skip-yak", action="store_true", help="Build the standalone bundle only")
    parser.add_argument("--gha-only", action="store_true", help="Build assembly only; requires an adjacent compatible runtime/")
    parser.add_argument("--runtime-only", action="store_true", help="Prepare the shipped Python runtime without compiling components")
    parser.add_argument("--offline", action="store_true", help="Use an already populated Python/wheel download cache")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip shipped-runtime CPU correctness check")
    parser.add_argument("--gpu-smoke", action="store_true", help="Also check a real Vulkan device using the shipped runtime")
    parser.add_argument("--install", action="store_true", help="Install the built local Yak package into Rhino 8; never publish")
    args = parser.parse_args(argv)
    if platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64"):
        parser.error("The shipped runtime targets Windows x64 Rhino 8 only")
    if args.gha_only and args.runtime_only:
        parser.error("--gha-only and --runtime-only are mutually exclusive")
    if args.install and (args.skip_yak or args.gha_only or args.runtime_only):
        parser.error("--install requires the complete Yak build")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    stage = args.output / f"raystrack-{VERSION}{'-gha-only' if args.gha_only else ''}"
    reset_directory(stage, args.output)
    if not args.runtime_only:
        # Fail before downloading a runtime if the installed Rhino references or
        # component sources cannot compile on the build machine.
        build_assembly(stage, args.rhino, args.compiler)
    if not args.gha_only:
        runtime = stage / "runtime"
        runtime.mkdir()
        build_runtime(runtime, args.cache.resolve(), args.offline)
        if not args.skip_smoke:
            smoke_runtime(runtime)
            if args.gpu_smoke:
                smoke_runtime(runtime, gpu=True)
    if args.runtime_only:
        print(f"Prepared runtime: {stage / 'runtime'}", flush=True)
        return 0
    for source, name in ((GH / "manifest.yml", "manifest.yml"),
                         (GH / "README.md", "README.md"),
                         (ROOT / "CHANGELOG.md", "CHANGELOG.md"),
                         (ROOT / "LICENSE", "LICENSE")):
        if source.exists():
            shutil.copy2(source, stage / name)
    guide = stage / "docs"
    guide.mkdir()
    for name in ("grasshopper.md", "v2-migration.md", "releasing.md"):
        source = ROOT / "docs" / name
        if source.exists():
            shutil.copy2(source, guide / name)
    examples = ROOT / "examples" / "grasshopper"
    if examples.is_dir():
        target = stage / "examples" / "grasshopper"
        target.mkdir(parents=True)
        for pattern in ("*.gh", "*.ghx", "README.md"):
            for source in sorted(examples.glob(pattern)):
                shutil.copy2(source, target / source.name)
    for name in ("grasshopper_v2_acceptance.json", "grasshopper_dev3_managed.json", "grasshopper_dev4_native.json", "grasshopper_dev5_self_viewing.json", "grasshopper_boxes.json", "grasshopper_v2_release.json"):
        report = ROOT / "validation" / "results" / name
        if report.exists():
            target = stage / "validation" / "results"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report, target / name)
    readme = stage / "README.md"
    if readme.exists():
        readme.write_text(readme.read_text(encoding="utf-8").replace(
            "(../docs/grasshopper.md)", "(docs/grasshopper.md)"), encoding="utf-8")
    icon = GH / "icons" / "raystrack_icon.png"
    if icon.exists():
        shutil.copy2(icon, stage / "icon.png")
    if args.gha_only:
        (stage / "RUNTIME-REQUIRED.txt").write_text(
            "This GHA needs the matching Raystrack bundled runtime/ folder beside it.\n"
            "Use the complete Windows x64 zip or Yak package for installation.\n", encoding="utf-8")
    write_package_provenance(stage)
    archive = args.output / f"raystrack-{VERSION}-win-x64{'-gha-only' if args.gha_only else ''}.zip"
    archive_bundle(stage, archive)
    artifacts = {"assembly": str(stage / ASSEMBLY), "standalone_zip": str(archive)}
    if not args.skip_yak and not args.gha_only:
        yak = args.yak or args.rhino / "System" / "Yak.exe"
        if not yak.is_file():
            raise FileNotFoundError(f"Yak executable is missing: {yak}; use --skip-yak for standalone only")
        run_logged([str(yak), "build", "--platform", "win"], cwd=stage,
                   log=args.output / "yak-build.log")
        built = sorted(stage.glob("*.yak"))
        if len(built) != 1:
            raise RuntimeError("Yak did not produce exactly one local package")
        package = args.output / built[0].name
        shutil.move(str(built[0]), str(package))
        artifacts["yak"] = str(package)
        if args.install:
            run_logged([str(yak), "install", str(package)], cwd=ROOT,
                       log=args.output / "yak-install.log")
    print(json.dumps(artifacts, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Build failed: {error}", file=sys.stderr)
        raise SystemExit(1)
