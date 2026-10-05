"""Portable distribution contracts without downloading or installing in Rhino."""
import importlib.util
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("raystrack_gh_build", ROOT / "tools/grasshopper/build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_runtime_path_is_relative_and_independent_of_host(tmp_path):
    runtime = tmp_path / "renamed installation" / "runtime"
    runtime.mkdir(parents=True)
    build.write_python_path(runtime)
    entries = (runtime / "python312._pth").read_text().splitlines()
    assert entries == ["python312.zip", ".", "Lib/site-packages", "import site"]
    assert all(not Path(entry).is_absolute() for entry in entries[:-1])
    assert "src" not in entries


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "C:/outside.txt", "folder\\..\\outside.txt"])
def test_runtime_extraction_rejects_path_escape(tmp_path, name):
    archive = tmp_path / "untrusted.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(name, "unexpected")
    with pytest.raises(ValueError, match="Unsafe"):
        build.extract_zip(archive, tmp_path / "runtime")
    assert not (tmp_path / "outside.txt").exists()


def test_archive_can_relocate_and_keeps_runtime_next_to_gha(tmp_path):
    stage = tmp_path / "build machine" / "package"
    (stage / "runtime/Lib/site-packages/raystrack").mkdir(parents=True)
    (stage / build.ASSEMBLY).write_bytes(b"compiled assembly")
    (stage / "runtime/python.exe").write_bytes(b"python")
    (stage / "runtime/Lib/site-packages/raystrack/__init__.py").write_text("# local package\n")
    build.write_python_path(stage / "runtime")
    # Yak files are artifacts, not a dependency recursively shipped in the zip.
    (stage / "old.yak").write_bytes(b"not part of installation")
    archive = tmp_path / "standalone.zip"
    build.archive_bundle(stage, archive)
    destination = tmp_path / "a completely different location"
    build.extract_zip(archive, destination)
    assert (destination / build.ASSEMBLY).is_file()
    assert (destination / "runtime/python.exe").is_file()
    assert not (destination / "old.yak").exists()
    with zipfile.ZipFile(archive) as bundle:
        assert all(not name.startswith(("/", "C:")) for name in bundle.namelist())


def test_cpython_download_rejects_modified_cache_without_network(tmp_path):
    (tmp_path / build.PYTHON_ARCHIVE).write_bytes(b"modified interpreter")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build.download_python(tmp_path, offline=True)


def test_build_cleanup_cannot_target_parent_or_outside(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    protected = tmp_path / "unrelated"
    protected.mkdir()
    (protected / "keep").write_text("user data")
    for path in (output, protected):
        with pytest.raises(ValueError, match="outside"):
            build.reset_directory(path, output)
    assert (protected / "keep").read_text() == "user data"


def test_dependency_versions_and_rhino_references_are_distribution_contracts():
    lines = (ROOT / "grasshopper/runtime-requirements.txt").read_text().splitlines()
    requirements = [line for line in lines if line and not line.startswith("#")]
    assert all("==" in line and ">" not in line for line in requirements)
    assert "taichi==1.7.4" in requirements
    project = ET.parse(ROOT / "grasshopper/Raystrack.Components.csproj").getroot()
    assert project.findtext(".//TargetFramework") == "net48"
    assert all(reference.findtext("Private") == "false"
               for reference in project.findall(".//Reference") if reference.find("HintPath") is not None)
    manifest = (ROOT / "grasshopper/manifest.yml").read_text()
    assert f"version: {build.VERSION}" in manifest
    assert project.findtext(".//Version") == build.VERSION
    assert "rhino8" in manifest


def test_dependency_licenses_keep_distribution_names(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    site = runtime / "Lib/site-packages"
    (site / "numpy-1.26.4.dist-info/licenses").mkdir(parents=True)
    (runtime / "LICENSE.txt").write_text("Python license")
    (site / "numpy-1.26.4.dist-info/licenses/LICENSE.txt").write_text("NumPy license")
    build.copy_licenses(runtime, site)
    assert (runtime / "licenses/CPython-LICENSE.txt").read_text() == "Python license"
    assert (runtime / "licenses/numpy-1.26.4.dist-info/licenses/LICENSE.txt").read_text() == "NumPy license"
    assert (runtime / "licenses/Raystrack-LICENSE.txt").is_file()


def test_package_provenance_identifies_source_icons_and_documentation(tmp_path, monkeypatch):
    repository = tmp_path / "source"
    grasshopper = repository / "grasshopper"
    (grasshopper / "icons").mkdir(parents=True)
    python_source = repository / "src/raystrack"
    python_source.mkdir(parents=True)
    (grasshopper / "Components.cs").write_text("// component source\n")
    (grasshopper / "manifest.yml").write_text("name: raystrack\n")
    (grasshopper / "icons/original.png").write_bytes(b"original image")
    (python_source / "worker.py").write_text("# Python source\n")
    stage = tmp_path / "relocatable package"
    (stage / "runtime").mkdir(parents=True)
    (stage / build.ASSEMBLY).write_bytes(b"compiled component")
    (stage / "Raystrack.Components.xml").write_text("<doc />\n")
    (stage / "manifest.yml").write_text("name: raystrack\n")
    (stage / "runtime/runtime-manifest.json").write_text('{"files": {}}\n')
    monkeypatch.setattr(build, "ROOT", repository)
    monkeypatch.setattr(build, "GH", grasshopper)
    build.write_package_provenance(stage)
    provenance = json.loads((stage / "package-manifest.json").read_text())
    assert provenance["source_sha256"]["grasshopper/Components.cs"] == build.sha256(grasshopper / "Components.cs")
    assert provenance["source_sha256"]["src/raystrack/worker.py"] == build.sha256(python_source / "worker.py")
    assert provenance["original_icon_sha256"]["grasshopper/icons/original.png"] == build.sha256(grasshopper / "icons/original.png")
    assert provenance["files"]["Raystrack.Components.xml"] == build.sha256(stage / "Raystrack.Components.xml")
    assert provenance["runtime_manifest_sha256"] == build.sha256(stage / "runtime/runtime-manifest.json")
    assert "package-manifest.json" not in provenance["files"]
    assert "manifest.yml" not in provenance["files"]
    assert provenance["source_sha256"]["grasshopper/manifest.yml"] == build.sha256(grasshopper / "manifest.yml")
    assert provenance["generated_distribution_metadata"] == ["manifest.yml"]
    assert not any("runtime/" in name or ":" in name for name in provenance["files"])
    (stage / "Raystrack.Components.xml").write_text("<doc>changed</doc>\n")
    assert provenance["files"]["Raystrack.Components.xml"] != build.sha256(stage / "Raystrack.Components.xml")
