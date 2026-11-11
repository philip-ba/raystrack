"""Independent analytic, Radiance and native delta thin-glazing gates.

Unpolarized, nonabsorbing, smooth parallel plate in air, zero geometric
thickness, incoherent internal reflections. No roughness, interference, tint,
dispersion, or refraction displacement. Delta events have no finite BSDF/PDF
value in sr^-1 and must remain separate from Klems scattering.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
from pathlib import Path

import numpy as np

from raystrack import Mesh, Scene
from raystrack.shortwave import ClearGlazing, ShortwaveSolver, SolarSite, SolarTimeline

try:
    from .annual_radiance import Radiance, polygon_text
except ImportError:
    from annual_radiance import Radiance, polygon_text


def plate_fresnel(cosine, refractive_index=1.52):
    """Return unpolarized plate (R,T), including all internal reflections."""
    cosine = np.asarray(cosine,dtype=float)
    n = float(refractive_index)
    if not np.isfinite(n) or n<1 or not np.isfinite(cosine).all() or np.any((cosine<0)|(cosine>1)):
        raise ValueError("finite n >= 1 and incidence cosine in [0,1] required")
    if n==1:
        return np.zeros_like(cosine),np.ones_like(cosine)
    inside = np.sqrt(1-(1-cosine**2)/n**2)
    rs = ((cosine-n*inside)/(cosine+n*inside))**2
    rp = ((n*cosine-inside)/(n*cosine+inside))**2
    transmission = .5*((1-rs)/(1+rs)+(1-rp)/(1+rp))
    return 1-transmission,transmission


def plate_fixture():
    """Large horizontal plate and sensors seeing its opposite hemispheres."""
    vertices = [[-10000,-10000,1],[10000,-10000,1],[10000,10000,1],[-10000,10000,1]]
    scene = Scene.from_meshes({"plate": Mesh(vertices, [[0,1,2],[0,2,3]])})
    return scene, np.array([[0,.1,0],[0,.1,2]]), np.array([[0,0,1],[0,0,-1]])


def native_checks(angles, measured, *, backends, samples, output):
    """Validate beam connections and independently integrated diffuse energy."""
    scene, points, normals = plate_fixture()
    cosine = np.cos(np.radians(angles))
    directions = np.column_stack((np.sin(np.radians(angles)), np.zeros(len(angles)), cosine))
    reflection, transmission = plate_fresnel(cosine)
    expected = 800*cosine[:,None]*np.column_stack((transmission,reflection))
    # Deterministic quadrature is independent of renderer transport sampling.
    nodes, weights = np.polynomial.legendre.leggauss(256)
    diffuse_cosine = (nodes+1)/2
    diffuse_r, diffuse_t = plate_fresnel(diffuse_cosine)
    diffuse_expected = 100*np.array([np.dot(weights*diffuse_cosine,diffuse_t),
                                     np.dot(weights*diffuse_cosine,diffuse_r)])
    timeline = SolarTimeline.for_year(2025,SolarSite(52.52,13.405),step_minutes=60,
                                     utc_offset_minutes=60)
    indices = np.array([np.flatnonzero(timeline.above_horizon)[i] for i in (200,1200,2200,3200)])
    annual_sun = timeline.directions[indices]
    annual_r, annual_t = plate_fresnel(annual_sun[:,2])
    annual_expected = 800*annual_sun[:,2,None]*np.column_stack((annual_t,annual_r))
    annual_cosines = np.maximum(timeline.directions[:,2],0)
    full_r, full_t = plate_fresnel(annual_cosines)
    full_expected = 800*annual_cosines[:,None]*np.column_stack((full_t,full_r))
    expected_exposure = np.einsum("t,ts->s",timeline.durations_hours,full_expected)
    reports = {}
    for backend in backends:
        with ShortwaveSolver(scene,backend=backend,materials={"plate":ClearGlazing(1.52)}) as solver:
            beam = solver.solve(points,normals,directions,dni=800,samples=1,bounces=1,seed=71)
            np.testing.assert_allclose(beam.total[:,:,0],expected,rtol=2e-5,atol=.002)
            np.testing.assert_allclose(beam.total[:,:,0],measured,rtol=2e-5,atol=.002)
            np.testing.assert_allclose(beam.irradiance[:,:,0],0,atol=1e-8)
            np.testing.assert_allclose(beam.irradiance[:,0,3,0],expected[:,0],rtol=2e-5,atol=.002)
            np.testing.assert_allclose(beam.irradiance[:,1,2,0],expected[:,1],rtol=2e-5,atol=.002)
            common = dict(samples=samples,bounces=1,replicates=4,seed=71)
            mixed = solver.solve(points,normals,directions,dni=800,dhi=100,**common)
            diffuse = solver.solve(points,normals,directions,dni=0,dhi=100,**common)
            np.testing.assert_allclose(mixed.total,beam.total+diffuse.total,atol=.002,rtol=2e-5)
            # Complete-replicate uncertainty per component; sum is a conservative
            # bound without assuming independence between component estimates.
            diffuse_se = diffuse.standard_error.sum(axis=2)[:,:,0]
            diffuse_error = abs(diffuse.total[:,:,0]-diffuse_expected)
            assert np.all(diffuse_error <= 2+4*diffuse_se), (backend,diffuse_error)
            annual = solver.solve(points,normals,annual_sun,dni=800,samples=1,bounces=1,seed=71)
            np.testing.assert_allclose(annual.total[:,:,0],annual_expected,rtol=2e-5,atol=.002)
            full = solver.solve(points,normals,timeline,dni=800,samples=1,bounces=1,seed=71,
                forcing_id="synthetic DNI800 during geometric sun-up; direct clear-plate analytic gate")
            np.testing.assert_allclose(full.total[:,:,0],full_expected,rtol=2e-5,atol=.002)
            np.testing.assert_array_equal(full.total[~timeline.above_horizon],0)
            full_error = full.total[:,:,0]-full_expected
            actual_exposure = full.exposure().sum(axis=1)[:,0]
            exposure_relative_error = abs(actual_exposure-expected_exposure)/np.maximum(expected_exposure,1)
            assert np.all(exposure_relative_error<=.02), (backend,exposure_relative_error)
            worst_step,worst_sensor = np.unravel_index(abs(full_error).argmax(),full_error.shape)
            annual_file = f"annual-direct-{backend}.npz"
            np.savez(output/annual_file,sun=timeline.directions,timestamps=timeline.timestamps,
                durations_hours=timeline.durations_hours,dni=np.where(timeline.above_horizon,800,0),
                points=points,normals=normals,actual=full.total,expected=full_expected,
                irradiance_components=full.irradiance)
        reports[backend] = dict(actual=beam.total.tolist(),
            max_analytic_error_w_m2=float(abs(beam.total[:,:,0]-expected).max()),
            max_radiance_error_w_m2=float(abs(beam.total[:,:,0]-measured).max()),
            mixed_actual=mixed.total.tolist(),diffuse_actual=diffuse.total.tolist(),
            diffuse_expected=diffuse_expected.tolist(),diffuse_standard_error=diffuse_se.tolist(),
            diffuse_95_percent_interval_lower=(diffuse.total[:,:,0]-3.182446305*diffuse_se).tolist(),
            diffuse_95_percent_interval_upper=(diffuse.total[:,:,0]+3.182446305*diffuse_se).tolist(),
            uncertainty="Student-t df=3, component-SE sum; sampling only, no universal coverage claim",
            max_diffuse_error_w_m2=float(diffuse_error.max()),
            linearity_max_error_w_m2=float(abs(mixed.total-beam.total-diffuse.total).max()),
            annual_indices=indices.tolist(),annual_sun=annual_sun.tolist(),
            annual_timestamps=[str(t) for t in timeline.timestamps[indices]],
            annual_expected=annual_expected.tolist(),annual_actual=annual.total.tolist(),
            annual_max_analytic_error_w_m2=float(abs(annual.total[:,:,0]-annual_expected).max()),
            full_annual_direct=dict(steps=len(timeline.timestamps),sun_up_steps=int(timeline.above_horizon.sum()),
                source="independent Fresnel plate analytic reference; no annual Radiance material claim",
                saved_arrays=annual_file,panel_half_extent_m=10000,
                expected_exposure_wh_m2=expected_exposure.tolist(),
                actual_exposure_wh_m2=actual_exposure.tolist(),
                exposure_relative_error=exposure_relative_error.tolist(),
                max_abs_error_w_m2=float(abs(full_error).max()),
                worst_timestamp=str(timeline.timestamps[worst_step]),worst_sensor=int(worst_sensor),
                worst_error_w_m2=float(full_error[worst_step,worst_sensor])),
            transport_location="host; backend selects traversal only")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("build/bsdf-milestone-20261111/clear-glazing"))
    parser.add_argument("--radiance-bin",default=r"C:\Radiance\bin")
    parser.add_argument("--backends",nargs="+",default=["native-builtin","embree","vulkan-rt"])
    parser.add_argument("--samples",type=int,default=8192)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rad = Radiance(args.radiance_bin,args.output)
    version = rad.run("rtrace",["-version"]).decode().strip()
    angles = np.array([0,20,45,60,75,85],dtype=float)
    cosines = np.cos(np.radians(angles))
    reflectance,transmittance = plate_fresnel(cosines)
    projected_omega = np.pi*np.sin(np.radians(.01/2))**2
    brightness = 800/projected_omega
    values = []
    for i,(angle,z) in enumerate(zip(angles,cosines)):
        x = np.sin(np.radians(angle))
        text = ("void glass glazing\n0\n0\n4 1 1 1 1.52\n"+
            polygon_text("plate",[[-1000,-1000,1],[1000,-1000,1],
                                  [1000,1000,1],[-1000,1000,1]],"glazing")+
            f"void light solar\n0\n0\n3 {brightness:.12g} {brightness:.12g} {brightness:.12g}\n"
            f"solar source sun\n0\n0\n4 {x:.12g} 0 {z:.12g} .01\n")
        (rad.output/f"glazing-{i}.rad").write_text(text)
        rad.run("oconv",[f"glazing-{i}.rad"],target=f"glazing-{i}.oct")
        rays=f"0 .1 0 {x:.12g} 0 {z:.12g}\n0 .1 2 {x:.12g} 0 {-z:.12g}\n".encode()
        data = rad.run("rtrace",["-h","-ov","-ab",0,"-lr",12,"-lw","1e-10",
                                 f"glazing-{i}.oct"],data=rays,target=f"beams-{i}.txt")
        radiance = np.fromstring(data.decode(),sep=" ").reshape(2,3)
        values.append(radiance[:,0]*projected_omega*z)
    measured = np.asarray(values)
    expected = 800*cosines[:,None]*np.column_stack((transmittance,reflectance))
    report = dict(model=__doc__,radiance_version=version,refractive_index=1.52,
        transmissivity=1,angles_degrees=angles.tolist(),irradiance_units="W/m2",
        radiance_to_irradiance="radiance times pi*sin(source_radius)^2 times cos(incidence)",
        events="delta reflection and straight-through delta transmission; no finite BSDF PDF",
        reflectance=reflectance.tolist(),transmittance=transmittance.tolist(),
        expected=expected.tolist(),radiance=measured.tolist(),
        max_abs_error_w_m2=float(abs(measured-expected).max()),
        max_energy_residual=float(abs(reflectance+transmittance-1).max()),
        renderer_integration="ClearGlazing native host transport; delta events separate from finite BSDF PDF",
        reference_sources=["https://radsite.lbl.gov/radiance/refer/ray.html#Glass",
                           "https://radsite.lbl.gov/radiance/refer/materials.pdf"],
        acceptance=dict(beam_relative_tolerance=2e-5,beam_absolute_tolerance_w_m2=.002,
            diffuse_absolute_tolerance_w_m2=2,diffuse_standard_error_multiplier=4),
        samples=args.samples,replicates=4,seed=71)
    report["backends"] = native_checks(angles,measured,backends=args.backends,samples=args.samples,output=rad.output)
    library = os.environ.get("RAYSTRACK_NATIVE_LIBRARY")
    report["native_library_sha256"] = (hashlib.sha256(Path(library).read_bytes()).hexdigest()
                                         if library else None)
    (rad.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    np.testing.assert_allclose(measured,expected,rtol=2e-5,atol=.002)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
