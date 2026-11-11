"""Regenerate the editable courtyard, busy subhourly day or repeated district."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np


def pose(x=0., y=0., z=0., angle=0.):
    matrix = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    matrix[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    matrix[:3, 3] = [x, y, z]
    return matrix.tolist()


def create_project(output, *, tiles=1, busy=False, inline_states=False):
    """Write small typed local inputs with no download or measured weather claim."""
    output = Path(output)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    prototypes = []

    def mesh(name, vertices, faces):
        np.save(assets / f"{name}-vertices.npy", np.asarray(vertices, dtype=np.float32), allow_pickle=False)
        np.save(assets / f"{name}-faces.npy", np.asarray(faces, dtype=np.int32), allow_pickle=False)
        prototypes.append(dict(id=name, vertices=f"assets/{name}-vertices.npy", faces=f"assets/{name}-faces.npy"))

    def quad(name, corners):
        mesh(name, corners, [[0, 1, 2], [0, 2, 3]])

    quad("pavement", [[0,0,0], [12,0,0], [12,12,0], [0,12,0]])
    quad("facade", [[0,0,0], [12,0,0], [12,0,5], [0,0,5]])
    quad("canopy", [[-1.2,-1.5,0], [1.2,-1.5,0], [1.2,1.5,0], [-1.2,1.5,0]])
    quad("slat", [[-.3,-1.5,0], [.3,-1.5,0], [.3,1.5,0], [-.3,1.5,0]])
    vertices = [[x,y,z] for z in (0,1.75) for y in (-.2,.2) for x in (-.25,.25)]
    faces = [[0,2,3], [0,3,1], [4,5,7], [4,7,6], [0,1,5], [0,5,4],
             [2,6,7], [2,7,3], [0,4,6], [0,6,2], [1,3,7], [1,7,5]]
    mesh("person", vertices, faces)
    source = Path(__file__).resolve().parents[1] / "tests/fixtures/bsdf"
    for item in source.iterdir():
        if item.is_file() and (item.name == "2011-SA1.XML" or item.suffix.lower() in (".md", ".txt")):
            shutil.copyfile(item, assets / item.name)
    materials = dict(pavement=dict(type="opaque", front=.2), facade=dict(type="opaque", front=.5),
        person=dict(type="opaque", front=.15), shade=dict(type="opaque", front=.3),
        glass=dict(type="clear_glazing", refractive_index=1.52),
        sheet=dict(type="diffuse_sheet", front_reflectance=.15, back_reflectance=.25, transmittance=.45),
        sheet_open=dict(type="diffuse_sheet", front_reflectance=.1, back_reflectance=.1, transmittance=.7),
        measured=dict(type="klems_xml", path="assets/2011-SA1.XML", band="solar"))
    instances, sensors = [], []
    width = int(np.ceil(np.sqrt(tiles)))
    for tile in range(tiles):
        dx, dy = 14*(tile%width), 14*(tile//width)

        def add(name, prototype, material, x=0, y=0, z=0, angle=0):
            instances.append(dict(id=f"t{tile}/{name}", prototype=prototype, material=material,
                                  transform=pose(dx+x, dy+y, z, angle)))
        add("ground", "pavement", "pavement")
        add("north", "facade", "facade", y=11)
        add("west", "facade", "facade", angle=np.pi/2)
        for x, name in ((2,"glass"), (5,"sheet"), (8,"measured")):
            add(name, "canopy", name, x=x, y=3.5, z=2.8)
        for j in range(8):
            add(f"shade{j}", "slat", "shade", x=1.3+j*1.3, y=7.7, z=3.2)
        for j in range(18):
            add(f"person{j}", "person", "person", x=1+(j%9)*1.1, y=5.4+(j//9)*3.9)
        for name, group, x, y, z in (("glass","sheltered",2.13,3.37,.01),
                ("sheet","sheltered",5.13,3.37,.01), ("measured","sheltered",8.13,3.37,.01),
                ("shade","shade",4,7.7,.01), ("person","pedestrian",1,5.4,.01),
                ("street","outdoor",10.8,.8,.01), ("open","reference",11,1,7)):
            sensors.append([f"t{tile}/{name}",group,dx+x,dy+y,z,0,0,1])
    with (output / "sensors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id","group","x","y","z","nx","ny","nz"])
        writer.writerows(sensors)
    start = np.datetime64("2025-06-21T00:00:00" if busy else "2024-12-31T23:00:00", "s")
    minutes, steps = (15,96) if busy else (60,8760)
    indices = range(steps) if busy else range(0,8640,720)
    states = []
    for ordinal, index in enumerate(indices):
        event = dict(timestamp_utc=str(start+np.timedelta64(index*minutes,"m"))+"Z", interpolation="hold", transforms={})
        for tile in range(tiles):
            dx, dy = 14*(tile%width), 14*(tile//width)
            for j in range(18):
                event["transforms"][f"t{tile}/person{j}"] = pose(dx+1+(j%9)*1.1,
                    dy+5.4+(j//9)*3.9+.8*np.sin(index*.31+j*.7), angle=.2*np.sin(index*.1+j))
            for j in range(8):
                event["transforms"][f"t{tile}/shade{j}"] = pose(dx+1.3+j*1.3, dy+7.7+1.5*(ordinal%2),3.2, angle=.12*(ordinal%3))
        states.append(event)
    for index, category in (((32 if busy else 4320),"materials"),
                             ((48 if busy else 6480),"vertices"), ((64 if busy else 7200),"meshes")):
        stamp = str(start+np.timedelta64(index*minutes,"m"))+"Z"
        event = next((state for state in states if state["timestamp_utc"] == stamp),None)
        if event is None:
            event = dict(timestamp_utc=stamp)
            states.append(event)
        event[category] = {}
        for tile in range(tiles):
            if category == "materials":
                event[category][f"t{tile}/sheet"] = "sheet_open"
            elif category == "vertices":
                world = np.array([[0,0,0],[12,0,0],[12,0,3.5],[0,0,3.5]],np.float32)
                world += np.array([14*(tile%width),14*(tile//width)+11,0],np.float32)
                name = f"assets/facade-deformed-t{tile}.npy"
                np.save(output/name,world,allow_pickle=False)
                event[category][f"t{tile}/north"] = name
            else:
                event.get("transforms",{}).pop(f"t{tile}/shade7",None)
                old = next(item for item in instances if item["id"] == f"t{tile}/shade7")
                world = np.load(assets/"slat-vertices.npy") + np.array(old["transform"],np.float32)[:3,3]
                name = f"assets/slat-topology-t{tile}.npy"
                np.save(output/name,world,allow_pickle=False)
                np.save(assets/"slat-one-face.npy",np.array([[0,1,2]],np.int32),allow_pickle=False)
                event[category][f"t{tile}/shade7"] = dict(vertices=name,faces="assets/slat-one-face.npy")
                for later in states:
                    if later["timestamp_utc"] > stamp:
                        later.get("transforms", {}).pop(f"t{tile}/shade7", None)
    states = sorted(states,key=lambda item:item["timestamp_utc"])
    if not inline_states:
        with (output/"poses.csv").open("w",newline="",encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp_utc","instance_id","x","y","z","yaw_deg","material"])
            for event in states:
                for key, matrix in event.pop("transforms",{}).items():
                    yaw = float(np.degrees(np.arctan2(matrix[1][0],matrix[0][0])))
                    writer.writerow([event["timestamp_utc"],key,*[matrix[i][3] for i in range(3)],yaw,""])
        states = [event for event in states if any(event.get(category) for category in ("materials","vertices","meshes"))]
    manifest = dict(schema="raystrack.shortwave-project",version=1,
        description="Synthetic moving courtyard; original licensed CGDB Solar; workflow execution, not annual accuracy validation",
        units=dict(length="m",irradiance="W/m2"),coordinates="ENU",
        site=dict(latitude_deg=52.52,longitude_deg=13.405,north_deg=0,utc_offset_minutes=60),
        weather=dict(type="synthetic",year=2025,interval_minutes=minutes,steps=steps,start_utc=str(start)+"Z"),
        prototypes=prototypes,instances=instances,materials=materials,sensors="sensors.csv",
        states=sorted(states,key=lambda item:item["timestamp_utc"]),
        settings=dict(backend="embree",samples=64,replicates=2,bounces=3,seed=17,
                      block_steps=168,storage="npy",retention="hourly",max_sensors=4096))
    if not inline_states:
        manifest["state_tracks"] = dict(path="poses.csv",interpolation="hold")
    (output/"project.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return output/"project.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="examples/dynamic_district")
    parser.add_argument("--tiles",type=int,default=1)
    parser.add_argument("--busy",action="store_true")
    parser.add_argument("--inline-states",action="store_true",help="reproduce the original inline timing inputs")
    args = parser.parse_args()
    if args.tiles < 1:
        parser.error("tiles must be positive")
    print(create_project(args.output,tiles=args.tiles,busy=args.busy,inline_states=args.inline_states))


if __name__ == "__main__":
    main()
