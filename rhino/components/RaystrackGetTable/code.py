# Raystrack: A Plugin for Radiative View Factors (AGPL)
# This file is part of Raystrack.

# Copyright (c) 2025, Philip Balizki.
# You should have received a copy of the GNU Affero General Public License
# along with Raystrack; If not, see <http://www.gnu.org/licenses/>.

# @license AGPL-3.0-or-later <https://spdx.org/licenses/AGPL-3.0-or-later>

"""
Create a Grasshopper table (DataTree) from a Raystrack view-factor matrix.

-

    Args:
        _vf_matrix: Dictionary of the computed view-factor matrix -> {sender: {receiver -> value}}.

    Returns:
        senders: Ordered list of sender names.
        receivers: Ordered list of receiver names.
        vf_tree: DataTree[float] with one branch per sender, items ordered by receivers.
"""

import Rhino, ghpythonlib
import Grasshopper as gh
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path

ghenv.Component.Name = 'RaystrackGetTable'
ghenv.Component.NickName = 'RaystrackGetTable'
ghenv.Component.Message = '1.0.0'
ghenv.Component.Category = 'Raystrack'
ghenv.Component.SubCategory = '1 :: View Factors'
ghenv.Component.AdditionalHelpFromDocStrings = '2'

try:
    input_help = {
        '_vf_matrix': 'Dict: {sender: {receiver: view_factor}}.'
    }
    for p in ghenv.Component.Params.Input:
        name = getattr(p, 'Name', '') or getattr(p, 'NickName', '')
        key = name if name in input_help else f'_{name}'
        if key in input_help:
            p.Description = input_help[key]
except Exception:
    pass

def vf_to_tree(matrix):
    # Flatten list-of-dicts or accept dict directly
    if isinstance(matrix, list):
        flat = {}
        for d in matrix:
            if not isinstance(d, dict):
                raise TypeError("All elements of _vf_matrix list must be dicts")
            flat.update(d)  # expects {sender: {receiver: value}, ...}
    elif isinstance(matrix, dict):
        flat = matrix
    else:
        raise TypeError("_vf_matrix must be a dict or list of dicts")

    # Validate rows
    for k, v in flat.items():
        if not isinstance(v, dict):
            raise TypeError(f"Row for '{k}' must be a dict mapping receiver->value")

    # Collect senders (in insertion order) and all receivers (first-seen order)
    senders = list(flat.keys())
    receivers = []
    for row in flat.values():
        for r in row.keys():
            if r not in receivers:
                receivers.append(r)

    # Build DataTree with one branch per sender, items ordered by receivers
    tree = DataTree[float]()
    for i, s in enumerate(senders):
        path = GH_Path(i)
        row = flat[s]
        for r in receivers:
            val = row.get(r, 0.0)
            try:
                val = float(val)
            except Exception:
                val = float('nan')
            tree.Add(val, path)

    return senders, receivers, tree

senders, receivers, vf_tree = vf_to_tree(_vf_matrix)
