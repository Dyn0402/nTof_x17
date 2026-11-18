"""Hero render for the poster: a generalised X17 search, four TPCs boxed around a target.

A 3D version of Paper I Fig. 1(a) (papers/paper1_v2/figs/make_experiment_layout.py): entrance windows 200 mm
from the target, a 30 mm drift gap, a thin and a thick scintillator layer behind each chamber, one e+e- pair
with the ionisation it leaves in the gas. Idealised, not the n_TOF setup. Beam along +z of the scene.

    ../.venv/bin/python make_hero.py [--size 2400 2000] [--out assets/hero_x17_box.png]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))

R_WIN = 200.0      # target to entrance window [mm]
GAP = 30.0         # drift gap
ACTIVE = 400.0     # active area side
C_GAS, C_PCB, C_STRIP = '#9fd3e6', '#3f9c84', '#f0cf7a'
C_S1, C_S2 = '#3b8fd0', '#f2a04a'
C_EM, C_EP, C_ION = '#d94f1e', '#7a3fb0', '#0b3d91'


def frame(phi_deg):
    """Chamber frame at azimuth phi: outward normal n, tangent t, beam axis b."""
    p = np.radians(phi_deg)
    return (np.array([np.cos(p), np.sin(p), 0.0]), np.array([-np.sin(p), np.cos(p), 0.0]),
            np.array([0.0, 0.0, 1.0]))


def slab(phi, r0, r1, wt, wb):
    """Box between radii r0 and r1 along the normal, wt wide along t, wb along the beam."""
    n, t, b = frame(phi)
    box = pv.Cube(x_length=r1 - r0, y_length=wt, z_length=wb)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = n, t, b
    m[:3, 3] = n * 0.5 * (r0 + r1)
    return box.transform(m, inplace=False)


def strips(phi, r, n_lines=40):
    """A coarse grid on the readout face (drawn every ~10 mm; the real pitch is 0.78 mm)."""
    n, t, b = frame(phi)
    pts, lines = [], []
    for u in np.linspace(-ACTIVE / 2, ACTIVE / 2, n_lines + 1):
        for a, c in ((t, b), (b, t)):
            i = len(pts)
            pts += [n * r + a * u - c * ACTIVE / 2, n * r + a * u + c * ACTIVE / 2]
            lines += [2, i, i + 1]
    return pv.PolyData(np.array(pts), lines=np.array(lines))


def add_arm(p, phi, near):
    """Far arms solid; the two arms nearest the camera as outlines, so the inside of the box shows."""
    parts = [(slab(phi, R_WIN, R_WIN + GAP, ACTIVE, ACTIVE), C_GAS, 0.16, '#4a90a8'),
             (slab(phi, R_WIN + GAP, R_WIN + GAP + 4, ACTIVE + 30, ACTIVE + 30), C_PCB, 1.0, C_PCB),
             (slab(phi, 262, 266, 500, 470), C_S1, 0.8, C_S1),
             (slab(phi, 292, 322, 560, 500), C_S2, 1.0, '#c9772a')]
    for mesh, col, op, edge in parts:
        if near:
            p.add_mesh(mesh, color=col, opacity=0.035, lighting=False)
            p.add_mesh(mesh.extract_feature_edges(), color=edge, line_width=2.0, opacity=0.55)
        else:
            p.add_mesh(mesh, color=col, opacity=op, ambient=0.45, diffuse=0.7, specular=0.1)
    if not near:
        p.add_mesh(slab(phi, R_WIN, R_WIN + GAP, ACTIVE, ACTIVE).extract_feature_edges(), color='#4a90a8',
                   line_width=2.0)
        p.add_mesh(strips(phi, R_WIN + GAP - 0.3), color=C_STRIP, line_width=1.0, opacity=0.8)


def add_track(p, d, phi, color):
    """Lepton from the origin along d through the chamber at azimuth phi; dots and drift lines in the gas."""
    n, _, _ = frame(phi)
    d = np.asarray(d, float) / np.linalg.norm(d)
    s_of = lambda r: r / d.dot(n)
    end = d * s_of(292.0)
    p.add_mesh(pv.Tube(pointa=(0, 0, 0), pointb=end, radius=4.0), color=color, smooth_shading=True, ambient=0.5)
    p.add_mesh(pv.Cone(center=end + d * 9, direction=d, height=36, radius=12, resolution=40), color=color, ambient=0.5)
    for r in np.linspace(R_WIN + 3, R_WIN + GAP - 3, 6):
        q = d * s_of(r)
        p.add_mesh(pv.Sphere(radius=6.5, center=q), color=C_ION, smooth_shading=True, ambient=0.4)
        foot = q + n * (R_WIN + GAP - r)
        p.add_mesh(pv.Line(q, foot), color=C_ION, line_width=2.5, opacity=0.8)


def build(size, scene='neutron'):
    """scene 'neutron': a gas cell in a beam, one wide pair (the poster's top-left figure).
    scene 'proton': a beam pipe ending on a thin foil target, one pair at 140 deg, and a cosmic muon that
    crosses two chambers without pointing at the target (the ATOMKI-repeat panel)."""
    p = pv.Plotter(off_screen=True, window_size=list(size))
    p.enable_depth_peeling(number_of_peels=16, occlusion_ratio=0.0)
    p.enable_anti_aliasing('ssaa')
    for phi in (0, 90, 180, 270):
        add_arm(p, phi, near=phi in (180, 270))
    if scene == 'neutron':
        p.add_mesh(pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=16, height=90, resolution=60),
                   color='#c9ced6', opacity=0.55, smooth_shading=True)
        p.add_mesh(pv.Sphere(radius=7, center=(0, 0, 0)), color='#1a1f2b')
        p.add_mesh(pv.Tube(pointa=(0, 0, -330), pointb=(0, 0, 250), radius=3.0), color='#6b7280', opacity=0.55)
        p.add_mesh(pv.Cone(center=(0, 0, 265), direction=(0, 0, 1), height=40, radius=11, resolution=40),
                   color='#6b7280')
        add_track(p, (1.0, 0.33, 0.20), 0, C_EM)
        add_track(p, (-0.28, 1.0, -0.14), 90, C_EP)
    else:
        # thin-walled beam pipe from below, ending in a cap that carries the target film
        pipe = pv.Cylinder(center=(0, 0, -190), direction=(0, 0, 1), radius=20, height=380, resolution=60,
                           capping=False)
        p.add_mesh(pipe, color='#aab2bd', opacity=0.45, smooth_shading=True)
        p.add_mesh(pv.Tube(pointa=(0, 0, -380), pointb=(0, 0, -6), radius=4.0), color='#e0457b', opacity=0.9,
                   ambient=0.5)
        p.add_mesh(pv.Cone(center=(0, 0, -18), direction=(0, 0, 1), height=30, radius=10, resolution=40),
                   color='#e0457b', ambient=0.5)
        p.add_mesh(pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=20, height=3, resolution=60),
                   color='#1a1f2b')
        c, s_ = np.cos(np.radians(25)), np.sin(np.radians(25))
        add_track(p, (c, -s_, 0.10), 0, C_EM)            # 140 deg apart, into two neighbouring arms
        add_track(p, (-s_, c, -0.06), 90, C_EP)
        # a cosmic muon: straight through two arms, missing the target by ~9 cm
        a_, b_ = np.array([330.0, 60.0, 330.0]), np.array([-40.0, 330.0, -300.0])
        p.add_mesh(pv.Tube(pointa=a_, pointb=b_, radius=2.6), color='#6b7280', ambient=0.5)
    p.set_background('white')
    p.camera.position = (-1150, -1420, 1050)
    p.camera.focal_point = (30, 30, -20)
    p.camera.up = (0, 0, 1)
    p.camera.view_angle = 28
    p.renderer.reset_camera_clipping_range()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', nargs=2, type=int, default=[2400, 2000])
    ap.add_argument('--scene', default='neutron', choices=['neutron', 'proton'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(HERE, 'assets', 'hero_x17_box.png' if a.scene == 'neutron' else 'hero_proton_box.png')
    p = build(a.size, a.scene)
    p.screenshot(a.out, transparent_background=True)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
