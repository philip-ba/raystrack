#!/usr/bin/env python3
from __future__ import annotations

import importlib


VALIDATIONS = [
    "validate_01_parallel_equal_square",
    "validate_02_parallel_equal_rectangle",
    "validate_03_equal_coaxial_discs",
    "validate_04_patch_to_disc",
    "validate_05_perpendicular_square_rectangle",
    "validate_06_canyon_view3d_compare",
]


def main() -> None:
    for module_name in VALIDATIONS:
        print(f"running {module_name}...")
        module = importlib.import_module(module_name)
        module.main()


if __name__ == "__main__":
    main()
