# Raystrack: A Plugin for Radiative View Factors (AGPL)
# This file is part of Raystrack.

# Copyright (c) 2025, Philip Balizki.
# You should have received a copy of the GNU Affero General Public License
# along with Raystrack; If not, see <http://www.gnu.org/licenses/>.

# @license AGPL-3.0-or-later <https://spdx.org/licenses/AGPL-3.0-or-later>

"""
Compute view factors from sender meshes to receiver meshes.

-

    Args:
        _sender: List of Breps/Meshes/Guids for sender surfaces.
        _sender_name: List of unique names for sender surfaces.
        _receiver: List of Breps/Meshes/Guids for receiver surfaces.
        _receiver_name: List of unique names for receiver surfaces.
        _samples: Integer sample count per face (default 256).
        _rays: Integer rays per sample (default 256).
        _seed: Integer RNG seed (default 0).
        _gpu_threads: Optional integer GPU thread hint.
        _use_bvh: Boolean; accelerate with BVH (default False).
        _flip_faces: Boolean; flip mesh face normals (default False).
        _max_iters: Integer max iterations (default 1).
        _tol: Float tolerance for convergence (default 1e-5).
        run: Boolean to trigger the computation.

    Returns:
        vf_matrix: Dictionary of the computed view-factor matrix -> {sender: {receiver -> value}}.
"""

import sys
import Rhino.Geometry as rg
import scriptcontext as sc
import System
import numpy as np
from Grasshopper.Kernel import GH_RuntimeMessageLevel as RML
from raystrack.main import view_factor

ghenv.Component.Name = 'RaystrackComputeVF'
ghenv.Component.NickName = 'RaystrackComputeVF'
ghenv.Component.Message = '1.0.0'
ghenv.Component.Category = 'Raystrack'
ghenv.Component.SubCategory = '1 :: View Factors'
ghenv.Component.AdditionalHelpFromDocStrings = '2'

try:
    input_help = {
        '_sender': 'Sender Breps/Meshes or Guids list.',
        '_sender_name': 'Unique names for sender surfaces (list of strings).',
        '_receiver': 'Receiver Breps/Meshes or Guids list.',
        '_receiver_name': 'Unique names for receiver surfaces (list of strings).',
        '_samples': 'Samples per face (int, default 256).',
        '_rays': 'Rays per sample (int, default 256).',
        '_seed': 'Random seed (int, default 0).',
        '_gpu_threads': 'Optional GPU thread hint (int).',
        '_use_bvh': 'Use BVH acceleration (bool).',
        '_flip_faces': 'Flip mesh face normals (bool).',
        '_max_iters': 'Max iterations (int).',
        '_tol': 'Tolerance for convergence (float).',
        'run': 'Trigger the computation (bool).'
    }
    for p in ghenv.Component.Params.Input:
        name = getattr(p, 'Name', '') or getattr(p, 'NickName', '')
        key = name if name in input_help else f'_{name}'
        if key in input_help:
            p.Description = input_help[key]
except Exception:
    pass



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

if not _sender or len(_sender) == 0:
    error("No geometry in _sender.")
if _sender_name is None:
    error("_sender_name is required.")
if len(_sender_name) != len(_sender):
    error("_sender_name length mismatch.")

_seen = set()
for n in _sender_name:
    if not isinstance(n, str) or n.strip() == "":
        error("_sender_name entries must be non-empty strings.")
    if n in _seen:
        error("Duplicate name in _sender_name.")
    _seen.add(n)

if not _receiver or len(_receiver) == 0:
    error("No geometry in _receiver.")
if _receiver_name is None:
    error("_receiver_name is required.")
if len(_receiver_name) != len(_receiver):
    error("_receiver_name length mismatch.")

_seen = set()
for n in _receiver_name:
    if not isinstance(n, str) or n.strip() == "":
        error("_receiver_name entries must be non-empty strings.")
    if n in _seen:
        error("Duplicate name in _receiver_name.")
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

_max_iters = int(_max_iters) if _max_iters is not None else 1

_tol = int(_tol) if _tol is not None else 1e-5

sender_meshes = []
sender_meshes_np = []
for obj, name in zip(_sender, _sender_name):
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
    sender_meshes.append(msh)
    sender_meshes_np.append((name, V, F))

if len(sender_meshes_np) < 1:
    warn("Need at least one surface.")

receiver_meshes = []
receiver_meshes_np = []
for obj, name in zip(_receiver, _receiver_name):
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
    receiver_meshes.append(msh)
    receiver_meshes_np.append((name, V, F))

if len(receiver_meshes_np) < 1:
    warn("Need at least one surface.")

if run:
    vf_result = view_factor(
        sender=sender_meshes_np,
        receiver=receiver_meshes_np,
        samples=_samples,
        rays=_rays,
        seed=_seed,
        gpu_threads=_gpu_threads,
        use_bvh=_use_bvh,
        flip_faces=_flip_faces,
        max_iters=_max_iters,
        tol=_tol
    )
    vf_matrix = [vf_result]

