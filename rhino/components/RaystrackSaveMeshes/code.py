# Raystrack: A Plugin for Radiative View Factors (AGPL)
# This file is part of Raystrack.

# Copyright (c) 2025, Philip Balizki.
# You should have received a copy of the GNU Affero General Public License
# along with Raystrack; If not, see <http://www.gnu.org/licenses/>.

# @license AGPL-3.0-or-later <https://spdx.org/licenses/AGPL-3.0-or-later>

"""
Mesh and save input geometry to a JSON file compatible with Raystrack.

-

    Args:
        _breps: List of Breps/Meshes/Guids to save (triangulated on export).
        _names: List of unique names matching each geometry.
        _path: Destination file path (.json appended if missing).
        run: Boolean to trigger saving.

    Returns:
        path: Absolute path to the written JSON file.
"""

import System
import Rhino.Geometry as rg
import scriptcontext as sc
import numpy as np
from Grasshopper.Kernel import GH_RuntimeMessageLevel as RML
from raystrack.io import save_meshes_json


ghenv.Component.Name = 'RaystrackSaveMeshes'
ghenv.Component.NickName = 'RaystrackSaveMeshes'
ghenv.Component.Message = '1.0.0'
ghenv.Component.Category = 'Raystrack'
ghenv.Component.SubCategory = '0 :: IO'
ghenv.Component.AdditionalHelpFromDocStrings = '2'

try:
    input_help = {
        '_breps': 'Breps/Meshes or Guids list of surfaces (triangulated).',
        '_names': 'Unique names matching each geometry (strings).',
        '_path': 'Destination file path (.json appended if missing).',
        'run': 'Trigger the save (bool).'
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

path = None

if run:
    if not _breps or len(_breps) == 0:
        error("No geometry in _breps.")
    if _names is None or len(_names) != len(_breps):
        error("_names must be provided and match _breps length.")

    seen = set()
    for n in _names:
        if not isinstance(n, str) or n.strip() == "":
            error("_names entries must be non-empty strings.")
        if n in seen:
            error("Duplicate name in _names.")
        seen.add(n)

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
        meshes_np.append((name, V, F))

    if not _path:
        error("_path is required.")
    path = save_meshes_json(meshes_np, _path)
    ghenv.Component.AddRuntimeMessage(RML.Remark, f"Saved: {path}")

