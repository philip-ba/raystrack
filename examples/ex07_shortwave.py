"""Synthetic shortwave and annual exposure with a mutable Raystrack scene."""
import argparse

import numpy as np

from _support import Mesh, Scene
from raystrack.shortwave import Opaque, ShortwaveSolver, SolarSite, SolarTimeline, ThinDiffuse


def panel(x, z, half_width):
    w = half_width
    return Mesh([[x-w, -w, z], [x+w, -w, z], [x+w, w, z], [x-w, w, z]],
                [[0, 1, 2], [0, 2, 3]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("embree", "vulkan-rt", "native-builtin"), default="embree")
    args = parser.parse_args()
    scene = Scene.from_meshes({
        "shade": panel(0, 1, 0.7),
        "panel": panel(3, 1, 1),
        "reflector": panel(3, -0.5, 1.2),
    })
    materials = {"panel": ThinDiffuse(0.3, 0.4), "reflector": Opaque(0.6, 0.2)}
    points = [[0, 0.1, 0], [3, 0.1, 0], [3, 0.1, 0.5]]
    normals = [[0, 0, 1], [0, 0, 1], [0, 0.6, -0.8]]
    timeline = SolarTimeline.for_year(2026, SolarSite(52.52, 13.405), utc_offset_minutes=60)
    selected = np.array([np.argmax(timeline.directions[:, 2]), len(timeline.timestamps) // 2])

    with ShortwaveSolver(scene, backend=args.backend, materials=materials) as solver:
        annual = solver.solve(points, normals, timeline, dni=800, samples=0, bounces=0,
                              forcing_id="synthetic constant DNI at sun-up")
        print("Annual direct exposure per sensor [kWh/m²]", annual.exposure()[:, 0, 0] / 1000)
        options = dict(dni=800, dhi=100, samples=2048, replicates=4, bounces=2, seed=13,
                       forcing_id="synthetic representative steps")
        before = solver.solve(points, normals, timeline.directions[selected], **options)
        moved = np.eye(4)
        moved[0, 3] = 1.5
        scene.update_transform("shade", moved)
        solver.set_material("panel", ThinDiffuse(0.2, 0.6))
        after = solver.solve(points, normals, timeline.directions[selected], **options)
        print("Components", before.components)
        for i, instant in enumerate(timeline.timestamps[selected]):
            print(instant, "UTC. Before / after [W/m² per sensor and component]")
            print(np.round(before.irradiance[i, :, :, 0], 2))
            print(np.round(after.irradiance[i, :, :, 0], 2))
        print("Geometry/material revisions", (before.geometry_revision, before.material_revision),
              (after.geometry_revision, after.material_revision))


if __name__ == "__main__":
    main()
