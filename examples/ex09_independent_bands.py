"""Synthetic two-band radiometric example; no broadband weather decomposition."""
from raystrack import Mesh, Scene
from raystrack.shortwave import BandInput, ThinDiffuse, WavelengthBand, solve_independent_bands


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="embree", choices=("native-builtin", "embree", "vulkan-rt"))
    args = parser.parse_args()
    scene = Scene.from_meshes({"sheet": Mesh(
        [[-10000, -10000, 1], [10000, -10000, 1], [10000, 10000, 1], [-10000, 10000, 1]],
        [[0, 1, 2], [0, 2, 3]])})
    bands = [
        BandInput(WavelengthBand("400–700 nm", 400, 700), {"sheet": ThinDiffuse(0, .2)},
                  100, 20, "synthetic constant tau=.2 across 400–700 nm",
                  "synthetic DNI=100, DHI=20 W/m² integrated over 400–700 nm"),
        BandInput(WavelengthBand("700–2500 nm", 700, 2500), {"sheet": ThinDiffuse(0, .8)},
                  300, 40, "synthetic constant tau=.8 across 700–2500 nm",
                  "synthetic DNI=300, DHI=40 W/m² integrated over 700–2500 nm"),
    ]
    result = solve_independent_bands(scene, [.123, .234, 0], [0, 0, 1], [0, 0, 1], bands,
                                     backend=args.backend, samples=4096, bounces=1,
                                     seed=37, durations_hours=[2])
    print("Band transmitted irradiance, W/m²:", result.irradiance[:, 0, 0, 3])
    print("Total over supplied bands, W/m²:", result.total[0, 0])
    print("Two-hour transmitted exposure, Wh/m²:", result.exposure()[0, 3])


if __name__ == "__main__":
    main()
