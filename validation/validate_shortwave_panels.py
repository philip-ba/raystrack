"""Controlled equivalent-coefficient panel checks; empirically probe BSDF band use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from annual_radiance import Radiance, cube_fixture, polygon_text
from raystrack import Mesh, Scene
from raystrack.shortwave import KlemsBSDF, ShortwaveSolver, ThinDiffuse
from raystrack.shortwave.klems import (KLEMS_AZIMUTH_COUNTS, KLEMS_POLAR_BOUNDS_DEG,
                                      KLEMS_PROJECTED_SOLID_ANGLES)


def write_window(path, solar, visible):
    root = ET.Element("WindowElement", xmlns="http://windows.lbl.gov")
    ET.SubElement(root, "WindowElementType").text = "System"
    ET.SubElement(root, "FileType").text = "BSDF"
    layer = ET.SubElement(ET.SubElement(root, "Optical"), "Layer")
    material = ET.SubElement(layer, "Material")
    for tag, value in (("Name", "Controlled coefficients"), ("Manufacturer", "Raystrack validation"),
                       ("DeviceType", "Other")):
        ET.SubElement(material, tag).text = value
    ET.SubElement(layer, "Geometry")
    definition = ET.SubElement(layer, "DataDefinition")
    ET.SubElement(definition, "IncidentDataStructure").text = "Columns"
    basis = ET.SubElement(definition, "AngleBasis")
    ET.SubElement(basis, "AngleBasisName").text = "LBNL/Klems Full"
    for i, count in enumerate(KLEMS_AZIMUTH_COUNTS):
        ring = ET.SubElement(basis, "AngleBasisBlock")
        ET.SubElement(ring, "Theta").text = str((KLEMS_POLAR_BOUNDS_DEG[i]+KLEMS_POLAR_BOUNDS_DEG[i+1])/2)
        ET.SubElement(ring, "nPhis").text = str(count)
        bounds = ET.SubElement(ring, "ThetaBounds")
        ET.SubElement(bounds, "LowerTheta").text = str(KLEMS_POLAR_BOUNDS_DEG[i])
        ET.SubElement(bounds, "UpperTheta").text = str(KLEMS_POLAR_BOUNDS_DEG[i+1])
    names = ("Reflection Front", "Reflection Back", "Transmission Front", "Transmission Back")
    for band, matrices in (("Solar", solar), ("Visible", visible)):
        data = ET.SubElement(layer, "WavelengthData")
        ET.SubElement(data, "LayerNumber").text = "System"
        ET.SubElement(data, "Wavelength", unit="Integral").text = band
        ET.SubElement(data, "SourceSpectrum").text = "D65"
        ET.SubElement(data, "DetectorSpectrum").text = "CIE 1931 2-deg"
        for label, matrix in zip(names, matrices):
            block = ET.SubElement(data, "WavelengthDataBlock")
            ET.SubElement(block, "WavelengthDataDirection").text = label
            for axis in ("ColumnAngleBasis", "RowAngleBasis"):
                ET.SubElement(block, axis).text = "LBNL/Klems Full"
            ET.SubElement(block, "ScatteringDataType").text = "BTDF"
            ET.SubElement(block, "ScatteringData").text = ",\n".join(f"{v:.12g}" for v in matrix.T.ravel())
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def controlled_matrices(tau=.35):
    # Half-turn invariant sector modulation gives reciprocal directional BTDFs.
    features = []
    for ring, count in enumerate(KLEMS_AZIMUTH_COUNTS):
        theta = np.radians((KLEMS_POLAR_BOUNDS_DEG[ring]+KLEMS_POLAR_BOUNDS_DEG[ring+1])/2)
        features.extend(np.sin(theta)**2*np.cos(4*np.pi*np.arange(count)/count))
    feature = np.asarray(features)
    values = np.empty((4,145,145))
    values[:2] = .1/np.pi
    values[2:] = tau/np.pi*(1+.6*feature[:,None]*feature[None,:])
    return values


def reference(rad, polygon, material, directions, points, normals, tag, samples, surroundings=None):
    sensors = "".join(" ".join(map(str,row))+"\n" for row in np.c_[points,normals]).encode()
    result = []
    for i, sun in enumerate(directions):
        diameter = .01
        brightness = 800/(np.pi*np.sin(np.radians(diameter/2))**2)
        geometry = ""
        if surroundings:
            geometry = "void plastic neutral\n0\n0\n5 .2 .2 .2 0 0\n" + "".join(
                polygon_text(name, vertices, "neutral") for name, vertices in surroundings.items())
        source = (geometry+material+polygon_text("sheet",polygon,"sheet_mat")+
                  f"void light solar\n0\n0\n3 {brightness:.12g} {brightness:.12g} {brightness:.12g}\n"
                  "solar source sun\n0\n0\n4 "+" ".join(map(str,sun))+f" {diameter}\n")
        name = f"{tag}_{i}"
        (rad.output/f"{name}.rad").write_text(source,encoding="ascii")
        rad.run("oconv",[f"{name}.rad"],target=f"{name}.oct")
        values = rad.run("rtrace",["-I+","-h","-u-","-ab",1,"-aa",0,"-ad",samples,
            "-as",0,"-av",0,0,0,"-lw","1e-8","-lr",8,"-dt",0,"-dc",1,
            "-dj",0,"-ds",0,f"{name}.oct"],data=sensors)
        result.append(np.fromstring(values.decode(),sep=" ").reshape(-1,3))
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("build/annual-validation/panels"))
    parser.add_argument("--radiance-bin",default=r"C:\Radiance\bin")
    parser.add_argument("--samples",type=int,default=32768)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rad = Radiance(args.radiance_bin,args.output)
    version = rad.run("rtrace",["-version"]).decode().strip()
    polygon = np.array([[-1000,-1000,1],[1000,-1000,1],[1000,1000,1],[-1000,1000,1]])
    points = np.array([[.12,.07,0],[.12,.07,2],[2,.07,0]])
    normals = np.array([[0,0,1],[0,0,-1],[.6,0,.8]])
    directions = np.array([[0,0,1],[.5,0,np.sqrt(.75)],[0,np.sqrt(.75),.5]])
    mesh = Mesh(polygon,[[0,1,2],[0,2,3]])
    solar,visible = controlled_matrices(.35),controlled_matrices(.65)
    write_window(rad.output/"equivalent.xml",solar,solar)
    write_window(rad.output/"distinct.xml",solar,visible)
    asymmetric = solar.copy()
    asymmetric[1] *= 2
    # Unequal front/back reflection is reciprocal. Keep transmission reciprocal:
    # Radiance's adjoint tracer reverses incident/outgoing vectors and therefore
    # is not an independent reference for arbitrary nonreciprocal BTDFs.
    write_window(rad.output/"front_back.xml",asymmetric,asymmetric)
    bsdf = "void BSDF sheet_mat\n6 0 equivalent.xml 0 1 0 .\n0\n0\n"
    thin_ref = reference(rad,polygon,
        "void trans sheet_mat\n0\n0\n7 .45 .45 .45 0 0 .777777777778 0\n",
        directions,points,normals,"thin",args.samples)
    klems_ref = reference(rad,polygon,bsdf,directions,points,normals,"klems",args.samples)
    distinct_ref = reference(rad,polygon,bsdf.replace("equivalent.xml","distinct.xml"),
        directions,points,normals,"distinct",args.samples)
    references = {"thin":thin_ref,"klems_equivalent_coefficients":klems_ref}
    references["klems_front_back"] = reference(rad,polygon,
        bsdf.replace("equivalent.xml","front_back.xml"),directions,points,normals,
        "front_back",args.samples)
    references["klems_reversed_winding"] = reference(rad,polygon[::-1],
        bsdf.replace("equivalent.xml","front_back.xml"),directions,points,normals,
        "reversed_winding",args.samples)
    report = dict(radiance_version=version,units="W/m2 neutral channels",samples=args.samples,
        seed=71,replicates=4,model="controlled equivalent coefficients; finite-width scattering",
        orientation_test="unequal front/back reflection with reciprocal transmission",
        projected_weight_sum=float(KLEMS_PROJECTED_SOLID_ANGLES.sum()),backends={})
    materials = {"thin":ThinDiffuse(.1,.35),"klems_equivalent_coefficients":
        KlemsBSDF.from_xml(rad.output/"equivalent.xml",window_front="back")}
    materials["klems_front_back"] = KlemsBSDF.from_xml(rad.output/"front_back.xml",window_front="back")
    materials["klems_reversed_winding"] = materials["klems_front_back"]
    for backend in ("embree","vulkan-rt"):
        results = {}
        for name,material in materials.items():
            case_mesh = (Mesh(polygon,[[2,1,0],[3,2,0]]) if name=="klems_reversed_winding" else mesh)
            with ShortwaveSolver(Scene.from_meshes({"sheet":case_mesh}),backend=backend,
                                 materials={"sheet":material}) as solver:
                result = solver.solve(points,normals,directions,dni=800,samples=args.samples,
                                      bounces=1,replicates=4,seed=71)
                handle = solver._native
                solver.set_material("sheet",ThinDiffuse(.1,.2))
                edited = solver.solve(points,normals,directions,dni=800,samples=4096,bounces=1,seed=71)
                assert handle is solver._native
                np.testing.assert_allclose(edited.total[:,:2,0],
                    directions[:,2,None]*[160,80],atol=.2,rtol=0)
                assert edited.material_revision==result.material_revision+1
                assert edited.geometry_revision==result.geometry_revision
            actual = result.total
            error = actual-references[name]
            results[name] = dict(max_abs_error_w_m2=float(abs(error).max()),
                max_relative_error=float((abs(error)/np.maximum(references[name],1)).max()),
                actual=actual.tolist(),radiance=references[name].tolist(),
                component_standard_error=result.standard_error.tolist(),
                material_edit_retained_geometry=True,edited=edited.total.tolist())
            assert np.all(abs(error)<=2+.03*references[name]),results[name]
        report["backends"][backend] = results
    # Extend the annual cube with a finite overhead sheet at selected sun angles.
    _, surrounding, cube_points, cube_normals, labels = cube_fixture()
    # Avoid placing the ground sensor y=1 exactly on a shadow silhouette.
    # Point-sun edge ownership is backend dependent, and a finite disk has a
    # penumbra there; neither is a useful isolated scattering comparison.
    overhead = np.array([[-1.1,-1.1,1.5],[1.1,-1.1,1.5],[1.1,1.1,1.5],[-1.1,1.1,1.5]])
    cube_meshes = {name: Mesh(vertices,[[0,1,2],[0,2,3]])
                   for name,vertices in surrounding.items()}
    cube_meshes["sheet"] = Mesh(overhead,[[0,1,2],[0,2,3]])
    report["cube_overhead"] = dict(labels=labels,panel_half_extent_m=1.1,
        shadow_boundary_policy="sensors avoid exact silhouettes; point versus .01 degree disk",backends={})
    for name,material in materials.items():
        if name=="klems_reversed_winding":
            continue  # The isolated asymmetric case already checks winding.
        text = ("void trans sheet_mat\n0\n0\n7 .45 .45 .45 0 0 .777777777778 0\n"
                if name=="thin" else bsdf.replace("equivalent.xml","front_back.xml")
                if name=="klems_front_back" else bsdf)
        expected = reference(rad,overhead,text,directions,cube_points,cube_normals,
                             "cube_"+name,args.samples,surrounding)
        for backend in ("embree","vulkan-rt"):
            optics = {sid: ThinDiffuse(.2,0) for sid in surrounding}
            optics["sheet"] = material
            with ShortwaveSolver(Scene.from_meshes(cube_meshes),backend=backend,materials=optics) as solver:
                actual = solver.solve(cube_points,cube_normals,directions,dni=800,
                    samples=args.samples,replicates=4,seed=71,bounces=1).total
            difference = abs(actual-expected)
            report["cube_overhead"]["backends"].setdefault(backend,{})[name] = dict(
                max_abs_error_w_m2=float(difference.max()),actual=actual.tolist(),radiance=expected.tolist())
            np.savez(rad.output/f"cube_{backend}_{name}.npz",actual=actual,expected=expected,
                     points=cube_points,normals=cube_normals,directions=directions)
            assert np.all(difference<=2+.03*expected), (backend,name,float(difference.max()))
    ratio = distinct_ref[:,0,0]/klems_ref[:,0,0]
    report["band_probe"] = dict(solar_transmittance=.35,visible_transmittance=.65,
        transmitted_ratios=ratio.tolist(),visible_expected_ratio=.65/.35,
        selected="Visible" if np.allclose(ratio,.65/.35,rtol=.03) else "unresolved")
    assert report["band_probe"]["selected"]=="Visible",report["band_probe"]
    np.savez(rad.output/"panels.npz",directions=directions,points=points,normals=normals,
        thin_reference=thin_ref,klems_reference=klems_ref,distinct_reference=distinct_ref)
    (rad.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
