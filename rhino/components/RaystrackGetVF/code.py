# Raystrack: A Plugin for Radiative View Factors (AGPL)
# This file is part of Raystrack.

# Copyright (c) 2025, Philip Balizki.
# You should have received a copy of the GNU Affero General Public License
# along with Raystrack; If not, see <http://www.gnu.org/licenses/>.

# @license AGPL-3.0-or-later <https://spdx.org/licenses/AGPL-3.0-or-later>

"""
Query a Raystrack view-factor matrix for a single value or a row of values.

-

    Args:
        _vf_matrix: Dictionary of the computed view-factor matrix -> {sender: {receiver -> value}}.
        _sender: Sender name to query (string, must exist in matrix keys).
        _receiver: Optional receiver substring to find a single value.

    Returns:
        view_factor: Float if _receiver specified; otherwise list of floats.
"""

import Rhino, ghpythonlib

ghenv.Component.Name = 'RaystrackGetVF'
ghenv.Component.NickName = 'RaystrackGetVF'
ghenv.Component.Message = '1.0.0'
ghenv.Component.Category = 'Raystrack'
ghenv.Component.SubCategory = '1 :: View Factors'
ghenv.Component.AdditionalHelpFromDocStrings = '2'

try:
    input_help = {
        '_vf_matrix': 'Dict: {sender: {receiver: view_factor}}.',
        '_sender': 'Sender name to query (string).',
        '_receiver': 'Optional receiver substring filter (string).'
    }
    for p in ghenv.Component.Params.Input:
        name = getattr(p, 'Name', '') or getattr(p, 'NickName', '')
        key = name if name in input_help else f'_{name}'
        if key in input_help:
            p.Description = input_help[key]
except Exception:
    pass

def vf_lookup(matrix, sender, receiver=None):
    # 1) flatten list-of-dicts if needed
    if isinstance(matrix, list):
        flat = {}
        for d in matrix:
            if not isinstance(d, dict):
                raise TypeError("All elements of _vf_matrix list must be dicts")
            flat.update(d)
    elif isinstance(matrix, dict):
        flat = matrix
    else:
        raise TypeError("_vf_matrix must be a dict or list of dicts")

    # 2) get the row for sender
    try:
        row = flat[sender]
    except KeyError:
        raise KeyError(f"Sender '{sender}' not found in the VF matrix")

    # 3) pick the right key
    if receiver:
        for key, val in row.items():
            # match if receiver substring is in the key
            if receiver in key:
                print(f"{key}: {float(val)}")
                return float(val)
        raise KeyError(f"No entry containing '{receiver}' in row for '{sender}'")
    else:
        # Return all values (view factors) as a list
        vf_list = list(row.values())
        print(row)
        return vf_list

view_factor = vf_lookup(_vf_matrix, _sender, _receiver)
