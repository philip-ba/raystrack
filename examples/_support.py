"""Shared setup for runnable v2 examples from a source checkout."""
from pathlib import Path
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from raystrack import Mesh, Scene, SolveOptions, Sampling, Accuracy
from raystrack.io import load, save


def canyon():
    stored=ROOT/"examples"/"street_canyon.raystrack"
    if stored.exists():
        return load(stored).scene
    from ex00_street_canyon_geometry import build_street_canyon
    return Scene.from_meshes({sid:Mesh(v,f) for sid,v,f in build_street_canyon()})


def options(seed=1,flip_faces=False):
    return SolveOptions(Sampling(density=4,rays_per_cell=32,seed=seed,flip_faces=flip_faces),
                        Accuracy(max_replicates=20,min_replicates=5,tolerance=1e-3))


def save_snapshot(stem,scene,result=None):
    path=ROOT/"examples"/f"{stem}-{uuid4().hex[:8]}.raystrack"
    return save(path,scene,result)
