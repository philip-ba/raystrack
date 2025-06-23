import sys
import Rhino.Geometry as rg
import scriptcontext as sc
import System
import numpy as np
from Grasshopper.Kernel import GH_RuntimeMessageLevel as RML
from raystrack.main import view_factor_matrix

def error(msg):
    ghenv.Component.AddRuntimeMessage(RML.Error, msg)
    raise ValueError(msg)

def warn(msg):
    ghenv.Component.AddRuntimeMessage(RML.Warning, msg)

def convert_guids_to_breps(face_geos):
    new_brep = []
    for geoId in face_geos:
        rhino_obj = sc.doc.Objects.FindId(geoId)
        if rhino_obj is None:
            print("Could not find object with ID:", geoId)
            continue
        geometry = rhino_obj.Geometry
        if isinstance(geometry, rg.Brep):
            new_brep.append(geometry)
        else:
            try:
                if hasattr(geometry, 'ToBrep'):
                    brep = geometry.ToBrep()
                    if brep is not None:
                        new_brep.append(brep)
            except:
                print("Could not convert geometry", geoId, "to Brep")
    return new_brep

def geometry_to_mesh(geo, mp=None):
    mp = mp or rg.MeshingParameters.Default
    if isinstance(geo, rg.Mesh):
        msh = geo.DuplicateMesh()
    elif isinstance(geo, rg.Brep):
        mlist = rg.Mesh.CreateFromBrep(geo, mp)
        if not mlist:
            error("Meshing failed for Brep.")
        msh = rg.Mesh()
        for m in mlist:
            msh.Append(m)
    elif isinstance(geo, (rg.Surface, rg.Extrusion)):
        mlist = rg.Mesh.CreateFromBrep(geo.ToBrep(), mp)
        if not mlist:
            error("Meshing failed for Surface/Extrusion.")
        msh = rg.Mesh()
        for m in mlist:
            msh.Append(m)
    elif hasattr(rg, "SubD") and isinstance(geo, rg.SubD):
        msh = rg.SubD.CreateSubDMesh(geo, mp)
    else:
        error("Unsupported geometry type.")
    msh.Faces.ConvertQuadsToTriangles()
    msh.UnifyNormals()
    msh.Normals.ComputeNormals()
    msh.Compact()
    return msh

def rhino_mesh_to_arrays(msh):
    if not msh.IsValid or msh.Vertices.Count == 0 or msh.Faces.Count == 0:
        error("Empty or invalid mesh.")
    verts = np.asarray([[v.X, v.Y, v.Z] for v in msh.Vertices], dtype=np.float32)
    faces = np.asarray([[f.A, f.B, f.C] for f in msh.Faces], dtype=np.int32)
    return verts, faces

if not _breps or len(_breps) == 0:
    error("No geometry in _breps.")
if _names is None:
    error("_names is required.")
if len(_names) != len(_breps):
    error("_names length mismatch.")

_seen = set()
for n in _names:
    if not isinstance(n, str) or n.strip() == "":
        error("_names entries must be non-empty strings.")
    if n in _seen:
        error("Duplicate name in _names.")
    _seen.add(n)

_samples = int(_samples) if _samples is not None else 256
if _samples <= 0:
    error("_samples must be positive.")
_rays = int(_rays) if _rays is not None else 256
if _rays <= 0:
    error("_rays must be positive.")
_seed = int(_seed) if _seed is not None else 0

if _gpu_threads is not None:
    try:
        _gpu_threads = int(_gpu_threads)
        if _gpu_threads <= 0:
            error("_gpu_threads must be positive.")
    except:
        error("_gpu_threads must be an integer.")
_use_bvh = bool(_use_bvh) if _use_bvh is not None else False

_flip_faces = bool(_flip_faces) if _flip_faces is not None else False

meshes = []
meshes_np = []
for obj, name in zip(_breps, _names):
    if obj is None:
        warn("Null geometry skipped.")
        continue
    if isinstance(obj, (System.Guid, str)):
        try:
            gid = System.Guid(obj) if isinstance(obj, str) else obj
        except:
            error("Invalid GUID input.")
        breps = convert_guids_to_breps([gid])
        if not breps:
            error("GUID conversion returned no Brep.")
        msh = geometry_to_mesh(breps[0])
    else:
        msh = geometry_to_mesh(obj)
    V, F = rhino_mesh_to_arrays(msh)
    meshes.append(msh)
    meshes_np.append((name, V, F))

if len(meshes_np) < 2:
    warn("Need at least two surfaces.")

if run:
    vf_result = view_factor_matrix(
        meshes_np,
        samples=_samples,
        rays=_rays,
        seed=_seed,
        gpu_threads=_gpu_threads,
        use_bvh=_use_bvh,
        flip_faces=_flip_faces
    )
    vf_matrix = [vf_result]

