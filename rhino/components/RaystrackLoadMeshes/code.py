# Raystrack: A Plugin for Radiative View Factors (AGPL)
# This file is part of Raystrack.

# Copyright (c) 2025, Philip Balizki.
# You should have received a copy of the GNU Affero General Public License
# along with Raystrack; If not, see <http://www.gnu.org/licenses/>.

# @license AGPL-3.0-or-later <https://spdx.org/licenses/AGPL-3.0-or-later>

"""
Load meshes from a Raystrack JSON file and output Rhino meshes and names.

-

    Args:
        _path: Path to the JSON file to load.
        run: Boolean to trigger loading.

    Returns:
        meshes: List of Rhino.Geometry.Mesh.
        names: List of mesh names.
"""

import Rhino.Geometry as rg
import numpy as np
from Grasshopper.Kernel import GH_RuntimeMessageLevel as RML
from raystrack.io import load_meshes_json


ghenv.Component.Name = 'RaystrackLoadMeshes'
ghenv.Component.NickName = 'RaystrackLoadMeshes'
ghenv.Component.Message = '1.0.0'
ghenv.Component.Category = 'Raystrack'
ghenv.Component.SubCategory = '0 :: IO'
ghenv.Component.AdditionalHelpFromDocStrings = '2'

try:
    input_help = {
        '_path': 'Path to the Raystrack mesh JSON file to load.',
        'run': 'Trigger the load (bool).'
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


def arrays_to_rhino_mesh(V, F):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    if V.ndim != 2 or V.shape[1] != 3:
        error("Vertices must have shape (N,3)")
    if F.ndim != 2 or F.shape[1] != 3:
        error("Faces must have shape (M,3) of triangles")
    m = rg.Mesh()
    for x, y, z in V:
        m.Vertices.Add(x, y, z)
    for a, b, c in F:
        m.Faces.AddFace(int(a), int(b), int(c))
    m.Faces.ConvertQuadsToTriangles()
    m.UnifyNormals()
    m.Normals.ComputeNormals()
    m.Compact()
    return m


meshes = []
names = []

if run:
    if not _path:
        error("_path is required.")
    try:
        data = load_meshes_json(_path)
    except Exception as e:
        error(str(e))

    for name, V, F in data:
        names.append(name)
        meshes.append(arrays_to_rhino_mesh(V, F))
    ghenv.Component.AddRuntimeMessage(RML.Remark, f"Loaded {len(meshes)} mesh(es)")

