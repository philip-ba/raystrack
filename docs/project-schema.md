# File-driven shortwave schema, version 1

Project JSON uses `schema: "raystrack.shortwave-project"`, `version: 1`,
`units: {"length":"m","irradiance":"W/m2"}` and `coordinates: "ENU"`.
All asset paths use `/`, are relative to the project JSON directory, and must
remain inside that directory after resolving links. No download is performed.
Only numeric NPY arrays are accepted; pickled/object arrays are rejected.

| Field | Contract |
| --- | --- |
| `prototypes` | List of `{id, vertices, faces}`; float 32 `(N,3)` vertices and int 32 `(M,3)` faces in NPY files. Each prototype is loaded once. |
| `instances` | List of `{id, prototype, material, transform?}`; IDs unique, transform absolute proper rigid 4x4 matrix from prototype local coordinates into world. Defaults to identity. |
| `materials` | Map ID to `opaque` (`front`, optional `back`), `diffuse_sheet` (`front_reflectance`, `back_reflectance`, `transmittance`), `clear_glazing` (`refractive_index`) or `klems_xml` (`path`, `band:"solar"`). Coefficients dimensionless scalar or three neutral channels. |
| `sensors` | CSV path or `{path}`. Exact header `id,group,x,y,z,nx,ny,nz`; metres, normals normalized, unique stable IDs. |
| `site` | `latitude_deg` north, `longitude_deg` east, optional `north_deg` counterclockwise world rotation, `utc_offset_minutes` fixed standard-time offset east of UTC. No DST. |
| `weather` | `type:"synthetic"`, `year`, `interval_minutes`, optional `steps` and explicit `start_utc` interval start. Deterministic forcing without downloads. Or `type:"csv", path, units:"W/m2"`. |
| Weather CSV | Exact header `timestamp_utc,duration_hours,dni,dhi`. UTC timestamps ending Z are **interval midpoints**, DNI beam-normal and DHI diffuse-horizontal, W/m2. Positive actual interval duration in hours; no overlap. |
| `states` | Increasing list of explicit `timestamp_utc` instants matching interval **starts**; optional `transforms`, `materials`, `vertices`, `meshes` maps keyed by instance ID. Hold sampled poses until next replacement, no interpolation. |
| `state_tracks` | Optional `{path:"poses.csv", interpolation:"hold"}`. CSV header `timestamp_utc,instance_id,x,y,z,yaw_deg,material`; metres and counterclockwise yaw degrees about world +Z. Absolute transform is translation times Rz(yaw). A row may leave all four pose fields empty for a material-only assignment. |
| State replacements | `materials` values are material IDs; `vertices` values float 32 NPY paths interpreted in world coordinates (transform resets); `meshes` values `{vertices,faces}` replaces world geometry/topology and resets transform. A state may not also transform the same replaced instance. |
| `settings` | Runner-owned options, included with input hashes in resume configuration. |

`rough_transmission` is an additional explicitly synthetic finite Full Klems
material: scalar `transmittance`, `exponent` and optional `reflectance`.
It represents a reciprocal, globally energy-bounded directional finite lobe;
it does not claim a measured microfacet model or a delta refractive path.

Pose CSV timestamps must not decrease; each timestamp/instance pair is unique.
CSV poses merge with inline geometry/optical edits at the same boundary;
duplicate assignments and transform/replacement conflicts fail explicitly.
Inline 4x4 matrices remain available for arbitrary proper rigid rotations.

Original WINDOW XML stays unchanged. Solar import is distinct from Radiance's
Visible selection. Supplying any arbitrary XML does not prove measured provenance.
The known CGDB fixture retains its original attribution and license.

Result directories have `manifest.json`, `schema: "raystrack.shortwave-results"`,
`version:1`. `axes` lists sensors, components and bands in array order, alongside
sensor groups. Components default to `direct,sky,reflected,transmitted`; the
three neutral RGB channels represent copies of broadband irradiance, not
independently specified spectral intervals. Array axes are `(time,sensor,component,band)`.
Values and replicate standard errors are W/m2. Explicit UTC midpoint timestamps
and duration hours are separate numeric arrays. Exposure is Wh/m2 from actual
durations; integrated uncertainty is a conservative linear sum of standard
errors because repeated transport samples may correlate intervals.

Each completed block records scene/material revisions, time range and SHA 256
of every numeric file. Files are written and fsynced before an atomic manifest
replacement publishes them; incomplete orphan blocks are ignored on resume.
Concurrent writers to one directory are unsupported. Resume compares the full
configuration, axes, retention and storage then verifies all committed hashes.
It recomputes integrated exposure and uncertainty from one completed block at a
time and rejects modified aggregate summaries. Timestamp selection validates
the small time/duration axes against block time bounds before filtering, so
incorrect manifest bounds cannot silently exclude samples.
Source and native binary hashes belong in `configuration`.

`storage="npy"` (default) retains separate memory-mappable arrays per block;
`storage="npz"` compresses each block. `retention="exposure"` keeps times and
per-block exposure summaries, and explicitly rejects hourly reads.
`hourly` retention preserves every input interval, including subhourly samples;
it performs no resampling. Existing
v2 snapshot stores are unchanged. ResultReader integer windows are half-open:
`read(100,124,sensor_ids=["sheltered"],components=["sky"])`. `read_times()` accepts
explicit UTC strings and selects midpoint timestamps in a half-open window.
`export_csv()` exports the same selection. An empty selection produces empty
arrays. With verification enabled, the selected block's payload is streamed for
checksum verification; RAM remains bounded but all selected-file bytes are read.
NPZ necessarily decompresses entire selected-block members. Consumers should
select windows explicitly; requesting the entire store allocates that request.

Reproduce format comparison in separate idle processes:

```powershell
$env:PYTHONPATH = 'src;.'
.local/release-venv/Scripts/python.exe -m validation.benchmark_project_store --output build/project-storage --storage npy
.local/release-venv/Scripts/python.exe -m validation.benchmark_project_store --output build/project-storage --storage npz
```

The benchmark records equal deterministic numeric inputs, bytes, write time,
verified and unchecked selected-read time and process peak resident memory.
Peak memory is the OS lifetime high-water mark, not an incremental allocation.

Three sequential fresh-process trials per format used equal 168-hour,
256-sensor numeric inputs in 24-hour blocks (8,257,536 value/error bytes).
Median NPY/NPZ writes were 0.490 / 0.593 s; verified six-interval, two-sensor
selected reads were 8.80 / 11.18 ms, and unchecked reads 3.98 / 7.36 ms.
Stored sizes were 8,497,607 / 6,073,532 bytes. Both median process peaks were
77,643,776 bytes: this small benchmark establishes no RAM saving. NPY is the
default for measured selected-read and write benefits; NPZ remains an explicit
space-saving choice. Verification necessarily streams selected-file bytes;
memory mapping does not bypass integrity checks. Raw trials and the qualified
host-load conditions are linked from the [evidence report](../validation/evidence/dynamic-district-20261111.json).
