import Rhino, ghpythonlib

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
        row_dict = [row]
        print(row)
        return row_dict

view_factor = vf_lookup(_vf_matrix, _sender, _receiver)