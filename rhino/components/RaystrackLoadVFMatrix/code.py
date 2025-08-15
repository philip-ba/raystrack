# Raystrack: A Plugin for Radiative View Factors (AGPL)
# This file is part of Raystrack.

# Copyright (c) 2025, Philip Balizki.
# You should have received a copy of the GNU Affero General Public License
# along with Raystrack; If not, see <http://www.gnu.org/licenses/>.

# @license AGPL-3.0-or-later <https://spdx.org/licenses/AGPL-3.0-or-later>

"""
Load a previously saved Raystrack view-factor matrix from a JSON file.

-

    Args:
        _path: Path to the JSON file containing the matrix.
        run: Boolean to trigger loading.

    Returns:
        vf_matrix: Dictionary {sender: {receiver -> value}}.
"""

from Grasshopper.Kernel import GH_RuntimeMessageLevel as RML
from raystrack.io import load_vf_matrix_json


ghenv.Component.Name = 'RaystrackLoadVFMatrix'
ghenv.Component.NickName = 'RaystrackLoadVFMatrix'
ghenv.Component.Message = '1.0.0'
ghenv.Component.Category = 'Raystrack'
ghenv.Component.SubCategory = '1 :: View Factors'
ghenv.Component.AdditionalHelpFromDocStrings = '2'

try:
    input_help = {
        '_path': 'Path to the JSON file to load.',
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


vf_matrix = None
if run:
    if not _path:
        error("_path is required.")
    try:
        vf_matrix = load_vf_matrix_json(_path)
        ghenv.Component.AddRuntimeMessage(RML.Remark, "Loaded view-factor matrix")
    except Exception as e:
        error(str(e))

