# Raystrack: A Plugin for Radiative View Factors (AGPL)
# This file is part of Raystrack.

# Copyright (c) 2025, Philip Balizki.
# You should have received a copy of the GNU Affero General Public License
# along with Raystrack; If not, see <http://www.gnu.org/licenses/>.

# @license AGPL-3.0-or-later <https://spdx.org/licenses/AGPL-3.0-or-later>

"""
Save a Raystrack view-factor matrix to a JSON file.

-

    Args:
        _vf_matrix: Dict or list of dicts {sender: {receiver -> value}}.
        _path: Destination file path (.json added if missing).
        run: Boolean to trigger saving.

    Returns:
        path: Absolute path to the written JSON file.
"""

from Grasshopper.Kernel import GH_RuntimeMessageLevel as RML
from raystrack.io import save_vf_matrix_json


ghenv.Component.Name = 'RaystrackSaveVFMatrix'
ghenv.Component.NickName = 'RaystrackSaveVFMatrix'
ghenv.Component.Message = '1.0.0'
ghenv.Component.Category = 'Raystrack'
ghenv.Component.SubCategory = '1 :: View Factors'
ghenv.Component.AdditionalHelpFromDocStrings = '2'

try:
    input_help = {
        '_vf_matrix': 'Dict or list-of-dicts {sender: {receiver: value}}.',
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


path = None
if run:
    if _vf_matrix is None:
        error("_vf_matrix is required.")
    if not _path:
        error("_path is required.")
    try:
        path = save_vf_matrix_json(_vf_matrix, _path)
        ghenv.Component.AddRuntimeMessage(RML.Remark, f"Saved: {path}")
    except Exception as e:
        error(str(e))

