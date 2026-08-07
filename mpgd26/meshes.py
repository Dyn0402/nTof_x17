#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meshes.py -- geometry -> PyVista meshes.

Pure builders; nothing here knows about colour, lighting or which scene it is
being used in.  All of them take and return millimetres in the scene frame, and
every builder takes an explicit ``normal`` axis so the same chamber code serves
the vertical cosmic bench ('z') and the horizontal SPS rail ('y'-up, beam 'z').
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

AXES = {'x': 0, 'y': 1, 'z': 2}


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def slab(center, u_size, v_size, thickness, normal='z'):
    """An axis-aligned box of ``thickness`` along ``normal``."""
    n = AXES[normal]
    size = [0.0, 0.0, 0.0]
    uv = [i for i in range(3) if i != n]
    size[uv[0]], size[uv[1]], size[n] = u_size, v_size, thickness
    c = np.asarray(center, float)
    b = []
    for i in range(3):
        b += [c[i] - size[i] / 2, c[i] + size[i] / 2]
    return pv.Box(bounds=b)


def frame_ring(center, outer_u, outer_v, inner_u, inner_v, thickness,
               normal='z'):
    """A rectangular picture-frame (4 bars) -- the chamber's mechanical frame."""
    n = AXES[normal]
    uv = [i for i in range(3) if i != n]
    c = np.asarray(center, float)
    bu, bv = (outer_u - inner_u) / 2, (outer_v - inner_v) / 2
    parts = []
    for du, dv, su, sv in (
            (0, +(inner_v + bv) / 2, outer_u, bv),
            (0, -(inner_v + bv) / 2, outer_u, bv),
            (+(inner_u + bu) / 2, 0, bu, inner_v),
            (-(inner_u + bu) / 2, 0, bu, inner_v)):
        cc = c.copy()
        cc[uv[0]] += du
        cc[uv[1]] += dv
        parts.append(slab(cc, su, sv, thickness, normal=normal))
    return parts[0].merge(parts[1:])


def polygon_prism(poly2d, offset, thickness, plane='xy', normal_axis='z'):
    """Extrude a closed 2-D polygon into a solid prism.

    ``poly2d`` is (n, 2) in the plane's own (u, v); ``offset`` is the coordinate
    of the *back* face along ``normal_axis``.  The polygon may be concave (an
    annulus sector is), so it is triangulated by VTK's triangle filter rather
    than by Delaunay, which would fill the hole.
    """
    n = len(poly2d)
    pts3 = np.zeros((n, 3))
    uv = [i for i in range(3) if i != AXES[normal_axis]]
    pts3[:, uv[0]] = poly2d[:, 0]
    pts3[:, uv[1]] = poly2d[:, 1]
    pts3[:, AXES[normal_axis]] = offset
    face = np.hstack([[n], np.arange(n)])
    cap = pv.PolyData(pts3, faces=face).triangulate()
    vec = np.zeros(3)
    vec[AXES[normal_axis]] = thickness
    return cap.extrude(vec, capping=True).clean()


def band_prism(inner_xy, outer_xy, offset, thickness, normal_axis='z'):
    """Extrude the quad strip between two matching polylines (an annulus sector).

    More robust than triangulating a concave outline, and it keeps the arc
    tessellation regular, which matters for the specular highlight on the
    aluminium fan frames.
    """
    m = len(inner_xy)
    assert len(outer_xy) == m
    uv = [i for i in range(3) if i != AXES[normal_axis]]
    na = AXES[normal_axis]

    pts = np.zeros((2 * m, 3))
    pts[0::2, uv[0]], pts[0::2, uv[1]] = inner_xy[:, 0], inner_xy[:, 1]
    pts[1::2, uv[0]], pts[1::2, uv[1]] = outer_xy[:, 0], outer_xy[:, 1]
    pts[:, na] = offset

    faces = []
    for i in range(m - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2
        faces.append([4, a, b, c, d])
    cap = pv.PolyData(pts, faces=np.hstack(faces))
    vec = np.zeros(3)
    vec[na] = thickness
    return cap.extrude(vec, capping=True).clean()


def quads_mesh(polys, offset, normal_axis='z', scalars=None):
    """Many flat quads (e.g. the 1280 P2 pads) as one PolyData.

    ``polys`` is (npoly, 4, 2) in the plane's (u, v).
    """
    npoly = len(polys)
    uv = [i for i in range(3) if i != AXES[normal_axis]]
    pts = np.zeros((npoly * 4, 3))
    flat = polys.reshape(-1, 2)
    pts[:, uv[0]], pts[:, uv[1]] = flat[:, 0], flat[:, 1]
    pts[:, AXES[normal_axis]] = offset
    idx = np.arange(npoly * 4).reshape(npoly, 4)
    faces = np.hstack([np.full((npoly, 1), 4), idx]).ravel()
    mesh = pv.PolyData(pts, faces=faces)
    if scalars is not None:
        mesh.cell_data['s'] = np.asarray(scalars)
    return mesh


def strip_lines(n_strips, span_u, span_v, offset, along='u', normal_axis='z',
                width=None):
    """Readout strips as thin quads -- the visual signature of a strip detector.

    ``span_u``/``span_v`` are (lo, hi) in the plane's own coordinates; strips run
    along ``along`` and are pitched across the other coordinate.
    """
    lo_u, hi_u = span_u
    lo_v, hi_v = span_v
    across = (lo_v, hi_v) if along == 'u' else (lo_u, hi_u)
    pitch = (across[1] - across[0]) / n_strips
    w = pitch * 0.5 if width is None else width
    centres = across[0] + pitch * (np.arange(n_strips) + 0.5)

    polys = np.zeros((n_strips, 4, 2))
    for i, c in enumerate(centres):
        if along == 'u':
            polys[i] = [[lo_u, c - w / 2], [hi_u, c - w / 2],
                        [hi_u, c + w / 2], [lo_u, c + w / 2]]
        else:
            polys[i] = [[c - w / 2, lo_v], [c + w / 2, lo_v],
                        [c + w / 2, hi_v], [c - w / 2, hi_v]]
    return quads_mesh(polys, offset, normal_axis=normal_axis)


def ground_shadow(outline3d, light_dir, plane_value, plane_axis='y',
                  lift=1.5, penumbra=((1.00, 1.00), (1.045, 0.62),
                                      (1.095, 0.34), (1.15, 0.16))):
    """Project a closed outline onto the ground plane along the key light.

    VTK's shadow-map pass is unusable at this scene's scale (a 2-metre scene in
    a 4096-px map shows the map frustum as straight-edged brightness steps
    across the table).  Since every object here has a known outline, the shadow
    it casts on the table is just that outline projected along the key
    direction -- exact, artefact-free, and under full control.

    Returns a list of (polygon, weight): nested slightly-grown copies drawn at
    graded opacity give a penumbra that a single hard polygon would not.
    """
    u = np.asarray(light_dir, float)
    u = -u / np.linalg.norm(u)                     # direction of travel
    ax = AXES[plane_axis]
    if abs(u[ax]) < 1e-6:
        return []

    pts = np.asarray(outline3d, float)
    t = (plane_value - pts[:, ax]) / u[ax]
    if np.any(t < 0):                              # light is below the plane
        return []
    proj = pts + t[:, None] * u[None, :]
    proj[:, ax] = plane_value + lift

    c = proj.mean(axis=0)
    out = []
    n = len(proj)
    face = np.hstack([[n], np.arange(n)])
    for g, w in penumbra:
        out.append((pv.PolyData(c + (proj - c) * g, faces=face).triangulate(),
                    w))
    return out


def rect_outline(center, u_size, v_size, normal='z'):
    """The four corners of a rectangular face, as a closed 3-D outline."""
    n = AXES[normal]
    uv = [i for i in range(3) if i != n]
    c = np.asarray(center, float)
    pts = []
    for su, sv in ((-1, -1), (+1, -1), (+1, +1), (-1, +1)):
        p = c.copy()
        p[uv[0]] += su * u_size / 2
        p[uv[1]] += sv * v_size / 2
        pts.append(p)
    return np.array(pts)


def tube(p0, p1, radius, n_sides=24):
    """A straight tube between two 3-D points."""
    return pv.Line(np.asarray(p0, float), np.asarray(p1, float)).tube(
        radius=radius, n_sides=n_sides, capping=True)


def arrow_head(tip, direction, length, radius, resolution=24):
    """A cone with its APEX at ``tip``, pointing along ``direction``.

    Tracks are drawn as plain tubes, which say nothing about which way the
    particle went; a head on the exit end fixes that without needing a caption
    or a particular camera.
    """
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    tip = np.asarray(tip, float)
    # pv.Cone puts its apex at centre + direction * height / 2
    return pv.Cone(center=tuple(tip - d * length / 2), direction=tuple(d),
                   height=length, radius=radius, resolution=resolution,
                   capping=True)


def tracks_with_heads(tracks, radius, head_len=None, head_radius=None,
                      n_sides=24):
    """Tubes plus an arrow head on the exit end of each track.

    ``tracks`` is a list of (entry, exit) in TRAVEL order, so the head always
    goes on the second point.
    """
    head_len = head_len if head_len is not None else radius * 12.0
    head_radius = head_radius if head_radius is not None else radius * 3.4
    out = []
    for a, b in tracks:
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d /= n
        shaft_end = b - d * head_len * 0.92
        out.append(tube(a, shaft_end, radius, n_sides=n_sides))
        out.append(arrow_head(b, d, head_len, head_radius))
    return out


def cylinder(center, direction, radius, height, resolution=48):
    return pv.Cylinder(center=center, direction=direction, radius=radius,
                       height=height, resolution=resolution, capping=True)


# --------------------------------------------------------------------------- #
# Composite: a rectangular chamber (MX17, M3, uRWELL)
# --------------------------------------------------------------------------- #
def rect_chamber(center, pcb_size, active_size, frame_size, pcb_thick,
                 normal='z', drift_gap=None, drift_dir=+1,
                 n_strips=None, active_span=None):
    """Return a dict of named meshes for one rectangular MPGD chamber.

    center       -- centre of the PCB, scene mm
    pcb_size     -- (u, v) of the readout board
    active_size  -- (u, v) of the metallised / efficient area
    frame_size   -- (u, v) outer size of the mechanical frame
    drift_gap    -- if given, a translucent gas volume of this depth is placed
                    on the ``drift_dir`` side, with the cathode plane capping it
    """
    n = AXES[normal]
    c = np.asarray(center, float)
    out = {}

    out['pcb'] = slab(c, pcb_size[0], pcb_size[1], pcb_thick, normal=normal)

    face = c.copy()
    face[n] += drift_dir * (pcb_thick / 2 + 0.4)
    out['active'] = slab(face, active_size[0], active_size[1], 0.8, normal=normal)

    if n_strips:
        span = active_span or ((-active_size[0] / 2, active_size[0] / 2),
                               (-active_size[1] / 2, active_size[1] / 2))
        uv = [i for i in range(3) if i != n]
        su = (span[0][0] + c[uv[0]], span[0][1] + c[uv[0]])
        sv = (span[1][0] + c[uv[1]], span[1][1] + c[uv[1]])
        out['strips'] = strip_lines(n_strips, su, sv,
                                    face[n] + drift_dir * 0.5,
                                    along='u', normal_axis=normal)

    if drift_gap:
        gas_c = c.copy()
        gas_c[n] += drift_dir * (pcb_thick / 2 + drift_gap / 2)
        out['gas'] = slab(gas_c, active_size[0], active_size[1], drift_gap,
                          normal=normal)
        cath = c.copy()
        cath[n] += drift_dir * (pcb_thick / 2 + drift_gap)
        out['cathode'] = slab(cath, pcb_size[0], pcb_size[1], 1.6, normal=normal)
        # frame runs from the PCB up to the cathode
        fr_c = c.copy()
        fr_c[n] += drift_dir * (pcb_thick / 2 + drift_gap / 2)
        out['frame'] = frame_ring(fr_c, frame_size[0], frame_size[1],
                                  pcb_size[0] * 0.995, pcb_size[1] * 0.995,
                                  drift_gap + pcb_thick, normal=normal)
    else:
        out['frame'] = frame_ring(c, frame_size[0], frame_size[1],
                                  pcb_size[0] * 0.995, pcb_size[1] * 0.995,
                                  pcb_thick * 2.2, normal=normal)
    return out


# --------------------------------------------------------------------------- #
# Composite: the P2 BASKET fan chamber
# --------------------------------------------------------------------------- #
def fan_chamber(pcb_band, frame_bands, pads_lab, z, pcb_thick,
                frame_depth=34.0, normal_axis='z', pad_side=+1):
    """P2 BASKET: an annulus-sector PCB with its 1280 pads and an aluminium rim.

    ``pcb_band``    -- (inner_arc, outer_arc), matched (n, 2) polylines in the
                       lab (x, height) plane, i.e. the board outline.
    ``frame_bands`` -- list of such pairs making up the rim (outer rim, inner
                       rim, the two radial side bars).  Building them from the
                       fan's own (radius, phi) description in the caller keeps
                       the rim a true fan rim rather than a normal-offset guess.
    ``pads_lab``    -- (npad, 4, 2) pad rectangles in the same plane.
    """
    out = {}
    out['pcb'] = band_prism(pcb_band[0], pcb_band[1],
                            z - pcb_thick / 2, pcb_thick,
                            normal_axis=normal_axis)
    out['pads'] = quads_mesh(pads_lab,
                             z + pad_side * (pcb_thick / 2 + 0.4),
                             normal_axis=normal_axis)
    bars = [band_prism(a, b, z - frame_depth / 2, frame_depth,
                       normal_axis=normal_axis) for a, b in frame_bands]
    out['frame'] = bars[0].merge(bars[1:]) if len(bars) > 1 else bars[0]
    return out


def loft_rect_to_circle(centre_a, width, thickness, centre_b, radius,
                        axis='y', n_around=72, n_steps=26, ease=True):
    """A fishtail light guide: rectangular cross-section morphing into a circle.

    ``centre_a`` is the middle of the rectangular end (the scintillator edge,
    ``width`` x ``thickness``), ``centre_b`` the middle of the circular end (the
    photocathode, ``radius``).  Both cross-sections are sampled along the SAME
    ray angles from their own centre, so corresponding points stay opposite one
    another and the surface cannot twist.

    A real guide has to collect the whole edge of the paddle and deliver it to a
    round photocathode, so a parallel-sided stub or a flat taper is the wrong
    shape -- this is the shape that does the job.
    """
    a = np.asarray(centre_a, float)
    b = np.asarray(centre_b, float)
    ax = AXES[axis]
    uv = [i for i in range(3) if i != ax]

    phi = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    # rectangle sampled along the same rays: distance to whichever wall is hit
    hw, ht = width / 2, thickness / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        r_rect = np.minimum(np.abs(hw / np.where(c == 0, 1e-12, c)),
                            np.abs(ht / np.where(s == 0, 1e-12, s)))
    rect = np.stack([r_rect * c, r_rect * s], axis=1)
    circ = np.stack([radius * c, radius * s], axis=1)

    t = np.linspace(0.0, 1.0, n_steps)
    w = t * t * (3 - 2 * t) if ease else t          # smoothstep

    rings = []
    for k in range(n_steps):
        prof = (1 - w[k]) * rect + w[k] * circ
        pts = np.zeros((n_around, 3))
        pts[:, ax] = a[ax] + (b[ax] - a[ax]) * t[k]
        pts[:, uv[0]] = a[uv[0]] + prof[:, 0]
        pts[:, uv[1]] = a[uv[1]] + prof[:, 1]
        rings.append(pts)

    pts = np.concatenate(rings)
    faces = []
    for k in range(n_steps - 1):
        o0, o1 = k * n_around, (k + 1) * n_around
        for i in range(n_around):
            j = (i + 1) % n_around
            faces.append([4, o0 + i, o0 + j, o1 + j, o1 + i])
    mesh = pv.PolyData(pts, faces=np.hstack(faces))
    return mesh.compute_normals(auto_orient_normals=True, inplace=False)
