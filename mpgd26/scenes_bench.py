#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenes_bench.py -- the Saclay cosmic test bench, in 3-D.

The stack, bottom to top (all z from the run config's ``bench_geometry`` and
``detectors``, except the scintillators -- see geometry.ASSUMPTIONS):

    z = -110 mm   trigger scintillator, 60 x 60 cm      (drawn, not configured)
    z =   24 mm   M3 reference Micromegas, 50 x 50 cm
    z =  144 mm   M3 reference Micromegas
    z =  232 mm   P1, the lower test slot
    z =  702 mm   P2, the upper test slot
    z = 1185 mm   M3 reference Micromegas
    z = 1302 mm   M3 reference Micromegas
    z = 1420 mm   trigger scintillator, 60 x 60 cm

A chamber in either test slot may be an MX17 (40 x 40 cm, 30 mm drift gap) or a
P2 BASKET fan, which is what ``mx17_det3_p2_det1_overnight_6-27-26`` actually
ran: P2_1 in the lower slot, mx17_3 in the upper one.

Frame: +Z vertically up (the bench's own frame, as used by every run config and
by the whole analysis chain), X/Y transverse.
"""
from __future__ import annotations

import os

import numpy as np

import geometry as G
import meshes as M
import style as S
import scenes_sps as SPS

RNG = np.random.default_rng(20260807)

GUIDE_LEN = 230.0           # paddle edge -> photocathode
PMT_RADIUS = 34.0           # photocathode radius; the guide funnels the
                            # whole 60 cm paddle edge down to this circle

FLOOR_Z = -400.0            # drawn floor, below the bottom paddle
FLOOR_HALF = 3200.0


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def add_m3(p, z, label=None):
    """One M3 reference Micromegas: 50 x 50 cm active in a light frame."""
    parts = M.rect_chamber(
        center=(0.0, 0.0, z),
        pcb_size=(G.M3_ACTIVE_MM + 24, G.M3_ACTIVE_MM + 24),
        active_size=(G.M3_ACTIVE_MM, G.M3_ACTIVE_MM),
        frame_size=(G.M3_FRAME_MM, G.M3_FRAME_MM),
        pcb_thick=G.M3_THICK_MM, normal='z', drift_gap=None,
        n_strips=56)
    p.add_mesh(parts['frame'], **S.mat('alu', S.COL['alu']))
    p.add_mesh(parts['pcb'], **S.mat('pcb', S.COL['m3_pcb']))
    p.add_mesh(parts['active'], **S.mat('copper', S.COL['copper']))
    p.add_mesh(parts['strips'], **S.mat('mesh', S.COL['copper_hot']))
    parts['outline'] = M.rect_outline((0.0, 0.0, z), G.M3_FRAME_MM,
                                      G.M3_FRAME_MM, normal='z')
    return parts


# Which parts of a chamber the alignment's in-plane angle should actually turn.
#
# theta_deg in an alignment.json is the rotation that carries detector-LOCAL
# STRIP coordinates into the M3 frame -- on these detectors it is ~90 deg, which
# is the strip-map convention, not a chamber that was physically stood on its
# side.  A square aluminium box turned 89 deg about its own centre is
# mechanically the same box, so turning the whole body would add no information
# and would actively mislead: it swings the frame's specular reflection from
# bright to dark and makes two identical chambers look like different objects.
# The strip direction is the thing that genuinely differs, so that is what turns.
ROTATE_WITH_STRIPS = ('strips', 'active', 'pads', 'pads_live')


def place(parts, x=0.0, y=0.0, theta_deg=0.0, z=0.0,
          rotate_keys=ROTATE_WITH_STRIPS):
    """Move a chamber to its measured position in the bench frame.

    ``(x, y)`` translates everything; ``theta_deg`` turns ``rotate_keys``, or
    EVERYTHING if ``rotate_keys`` is None.  Both are applied to the outline the
    cast shadow is projected from as well, so a moved chamber moves its shadow.

    Rotating a subset is only right for a chamber that is rotationally
    symmetric at 90 deg -- a square MX17, where the strip direction is the only
    thing the angle changes.  A P2 fan is not: its pads are part of the board,
    so it has to turn as one rigid object or the pads slide off the outline.
    """
    if not (x or y or theta_deg):
        return parts
    pivot = (0.0, 0.0, z)
    for k, mesh in parts.items():
        if k == 'outline':
            continue
        if theta_deg and (rotate_keys is None or k in rotate_keys):
            mesh = mesh.rotate_z(theta_deg, point=pivot, inplace=False,
                                 transform_all_input_vectors=True)
        parts[k] = mesh.translate((x, y, 0.0), inplace=False,
                                  transform_all_input_vectors=True)

    o = np.asarray(parts['outline'], float).copy()
    o[:, 0] += x
    o[:, 1] += y
    parts['outline'] = o
    return parts


def add_mx17(p, z, drift_dir=+1, x=0.0, y=0.0, theta_deg=0.0):
    """An MX17 chamber in a test slot, drift volume facing the incoming muons."""
    parts = M.rect_chamber(
        center=(0.0, 0.0, z),
        pcb_size=(G.MX17_PCB_MM, G.MX17_PCB_MM),
        active_size=(G.MX17_ACTIVE_MM, G.MX17_ACTIVE_MM),
        frame_size=(G.MX17_PCB_MM + 2 * G.MX17_FRAME_MM,
                    G.MX17_PCB_MM + 2 * G.MX17_FRAME_MM),
        pcb_thick=8.0, normal='z',
        drift_gap=G.MX17_DRIFT_GAP_MM, drift_dir=drift_dir,
        n_strips=64)
    w = G.MX17_PCB_MM + 2 * G.MX17_FRAME_MM
    parts['outline'] = M.rect_outline((0.0, 0.0, z), w, w, normal='z')
    parts = place(parts, x, y, theta_deg, z)

    p.add_mesh(parts['frame'], **S.mat('alu', S.COL['alu']))
    p.add_mesh(parts['pcb'], **S.mat('pcb', S.COL['pcb']))
    p.add_mesh(parts['active'], **S.mat('copper', S.COL['copper']))
    p.add_mesh(parts['strips'], **S.mat('mesh', S.COL['copper_hot']))
    p.add_mesh(parts['gas'], **S.mat('gas', S.COL['gas']))
    # the drift volume is the point of a micro-TPC chamber: outline it so the
    # 30 mm gap reads as a volume rather than a pane of glass
    p.add_mesh(parts['gas'].extract_feature_edges(), color=S.COL['gas'],
               line_width=2.0, lighting=False, opacity=0.75)
    p.add_mesh(parts['cathode'], **S.mat('plastic', '#e2e9f0', opacity=0.16))
    return parts


def add_p2_flat(p, z, pads_lab, sectors, x=0.0, y=0.0, theta_deg=0.0):
    """A P2 BASKET in a test slot, lying flat (fan in the horizontal plane).

    On the bench the boards lie horizontally in the rail levels, so the fan's
    own (x, height) plane maps onto the bench's (x, y) and the chamber normal
    is +Z -- the same ``fan_chamber`` code as the SPS scene.  The one thing that
    does *not* carry over is the origin: in the SPS lab frame the fan's height
    coordinate is measured from the table (130 -> 683 mm), so used unshifted it
    would sit half a metre off the bench axis.  Here it is centred on the slot,
    which is what a board resting in a rail level does.
    """
    fr_lo, fr_hi, _, _ = SPS.p2_frame_extent()
    y_mid = G.P2_APEX_HEIGHT - (fr_lo + fr_hi) / 2

    # pads up, and ALL of them read out: the sector-3-6 subset is the SPS
    # telescope's cabling on that run, not a property of the board
    m = SPS.p2_meshes(z, pads_lab, sectors, pad_side=+1,
                      live_sectors=None)
    for key in ('frame', 'pcb', 'pads', 'pads_live'):
        m[key] = m[key].translate((0.0, -y_mid, 0.0), inplace=False)

    o = SPS.p2_outline3d(z)
    o[:, 1] -= y_mid
    m['outline'] = o
    # rigid: the fan's pads belong to its board
    m = place(m, x, y, theta_deg, z, rotate_keys=None)

    p.add_mesh(m['frame'], **S.mat('alu_matte', S.COL['alu']))
    p.add_mesh(m['pcb'], **S.mat('pcb', S.COL['pcb']))
    if m['pads'].n_cells:                      # empty when every pad is live
        p.add_mesh(m['pads'], **S.mat('copper', S.COL['copper_dead'],
                                      metallic=0.55, roughness=0.62))
    p.add_mesh(m['pads_live'], **S.mat('copper', S.COL['copper']))
    return m


def add_scintillator(p, z, pmt_dir=-1):
    """A 60 x 60 cm trigger paddle with a light guide and PMT.

    ``pmt_dir`` is the sign along BENCH Y that the light guide and PMT stick
    out on.  Both paddles use -1, so the two photomultipliers are on the same
    side of the bench -- which is how they are actually plumbed, and it keeps
    them out of the way of the open -y face of the frame.
    """
    out = {}
    slab = M.slab((0.0, 0.0, z), G.SCINT_MM, G.SCINT_MM, G.SCINT_THICK_MM,
                  normal='z')
    p.add_mesh(slab, **S.mat('scint', S.COL['scint']))
    # Light guide: a fishtail.  It has to take the WHOLE 60 cm paddle edge --
    # the full rectangular cross-section, corner to corner -- and deliver it to
    # a round photocathode, so it morphs from rectangle to circle along its
    # length.  A flat taper would not cover the crystal.
    half = G.SCINT_MM / 2
    y0 = pmt_dir * half
    y1 = pmt_dir * (half + GUIDE_LEN)
    p.add_mesh(M.loft_rect_to_circle((0.0, y0, z), G.SCINT_MM,
                                     G.SCINT_THICK_MM,
                                     (0.0, y1, z), PMT_RADIUS, axis='y'),
               **S.mat('plastic', S.COL['guide'], specular=0.35,
                       specular_power=45))

    # PMT: glass envelope, then the base / divider can behind it
    p.add_mesh(M.cylinder((0.0, y1 + pmt_dir * 60, z), (0, 1, 0),
                          radius=PMT_RADIUS, height=120),
               **S.mat('plastic', '#c9d3dd', opacity=0.55, specular=0.9,
                       specular_power=70))
    p.add_mesh(M.cylinder((0.0, y1 + pmt_dir * 190, z), (0, 1, 0),
                          radius=PMT_RADIUS * 0.88, height=150),
               **S.mat('plastic', S.COL['pmt']))
    out['outline'] = M.rect_outline((0.0, 0.0, z), G.SCINT_MM, G.SCINT_MM,
                                    normal='z')
    return out


# --------------------------------------------------------------------------- #
# Bench structure
# --------------------------------------------------------------------------- #
def add_structure(p, theme, floor=True):
    """The rack: two uprights on +y, with the rails cantilevered out over -y.

    Leaving the -y face open is what makes the stack visible: a post on each
    corner puts an upright across the near face of every hero view, and the
    detectors slide in from that side anyway.
    """
    a = G.BENCH_POST_XY
    s = G.BENCH_POST_SECTION
    z0, z1 = G.BENCH_POST_Z
    for sx in (-a, a):
        p.add_mesh(M.slab((sx, a, (z0 + z1) / 2), s, s, z1 - z0, normal='z'),
                   **S.mat('alu_matte', S.COL['alu_dark']))
    # top and bottom cross-rails: along x between the two posts, and along y
    # reaching out over the open -y side
    for zc in (z0 + s / 2, z1 - s / 2):
        p.add_mesh(M.slab((0.0, a, zc), 2 * a + s, s, s, normal='z'),
                   **S.mat('alu_matte', S.COL['alu_dark']))
        for sx in (-a, a):
            p.add_mesh(M.slab((sx, 0.0, zc), s, 2 * a + s, s, normal='z'),
                       **S.mat('alu_matte', S.COL['alu_dark']))
    if floor:
        p.add_mesh(M.slab((0.0, 0.0, FLOOR_Z - 25), 2 * FLOOR_HALF,
                          2 * FLOOR_HALF, 50, normal='z'),
                   **S.mat('alu_matte', S.THEMES[theme]['floor'],
                           specular=0.05, specular_power=90, ambient=0.30))


def add_level_rails(p, z_lo=None, z_hi=1150.0):
    """Short brackets on the four uprights at every rail level.

    ``bench_geometry`` gives the levels as ``bottom_level_z`` + k x
    ``level_z_spacing`` (82 mm, then every 97 mm); drawing them is what makes
    the bench read as the modular rack it is, and it is also what fixes the
    test slots P1 = 227 mm and P2 = 697 mm as *levels* rather than free heights.
    """
    z_lo = G.BENCH_BOTTOM_LEVEL_Z if z_lo is None else z_lo
    a, s = G.BENCH_POST_XY, G.BENCH_POST_SECTION
    z = z_lo
    while z <= z_hi:
        for sx in (-a, a):
            for sy in (a,):
                p.add_mesh(M.slab((sx - np.sign(sx) * s * 0.62, sy, z),
                                  s * 0.5, s * 0.92, 7, normal='z'),
                           **S.mat('alu_matte', S.COL['alu_dark'],
                                   ambient=0.30))
        z += G.BENCH_LEVEL_SPACING


def add_shelf(p, z, half_width, drop=15.0, section=20.0, opacity=1.0):
    """The two rails a plane of half-width ``half_width`` rests on.

    The uprights stand just outside the widest element, so a plane is carried
    by a pair of rails spanning post to post -- a real load path, and the thing
    that stops every plane in the stack from looking suspended in mid-air.
    """
    a = G.BENCH_POST_XY
    zc = z - drop
    inset = min(half_width * 0.82, a - section)
    kw = S.mat('alu_matte', S.COL['alu_dark'], ambient=0.30)
    if opacity < 1.0:
        kw['opacity'] = opacity
    for sy in (-inset, inset):
        p.add_mesh(M.slab((0.0, sy, zc), 2 * a, section, section, normal='z'),
                   **kw)


# --------------------------------------------------------------------------- #
# Cosmic muons
# --------------------------------------------------------------------------- #
def cosmic_tracks(n=9, accept_only=True, max_try=20000):
    """Muons that fire the top+bottom scintillator coincidence.

    Zenith angles are drawn from the standard cos^2(theta) sea-level
    distribution and azimuth uniformly; a track is kept only if it crosses both
    60 x 60 cm paddles, which is exactly the bench's trigger condition and is
    what makes the drawn tracks steep-limited rather than isotropic.
    """
    z_top, z_bot = G.BENCH_SCINT_Z['top'], G.BENCH_SCINT_Z['bottom']
    half = G.SCINT_MM / 2
    out = []
    for _ in range(max_try):
        if len(out) >= n:
            break
        # sample cos^2(theta) by rejection, theta in [0, pi/2)
        while True:
            th = np.arccos(RNG.uniform(0, 1))
            if RNG.uniform() < np.cos(th) ** 2:
                break
        ph = RNG.uniform(0, 2 * np.pi)
        # entry point on the top paddle
        x0, y0 = RNG.uniform(-half, half, 2)
        tx = np.tan(th) * np.cos(ph)
        ty = np.tan(th) * np.sin(ph)
        dz = z_bot - z_top
        x1, y1 = x0 + tx * dz, y0 + ty * dz
        if accept_only and (abs(x1) > half or abs(y1) > half):
            continue
        pad = 110.0
        out.append((np.array([x0 - tx * pad, y0 - ty * pad, z_top + pad]),
                    np.array([x1 + tx * pad, y1 + ty * pad, z_bot - pad])))
    return out


def real_tracks(ray_dir, n=7, file_nums=(0,), chi2_cut=None, min_nclus=None,
                seed=20260807, require_paddles=True, overshoot=110.0):
    """Real reconstructed cosmic muons, straight out of the M3 telescope.

    ``ray_dir`` is a run's ``m3_tracking_root*`` directory.  The rays are
    already in the bench frame -- the files carry Z_Up = 1302 and Z_Down = 24,
    which are exactly the top and bottom M3 plane heights in ``geometry.py`` --
    so a ray is a straight line through the scene with no transform at all.

    Quality cuts default to the recipe recorded in
    ``mx_june_cosmic_qa/qa_config.py`` (chi2 < 1.0 on both planes AND
    NClus = 4 on both), which is the one the whole June analysis uses.  Drawing
    anything looser would put visibly bad fits in a figure.

    Returns the same (start, end) list as ``cosmic_tracks``, so the two are
    interchangeable at the call site.
    """
    import sys as _sys

    qa = os.path.join(G.REPO, 'mx_june_cosmic_qa')
    if qa not in _sys.path:
        _sys.path.insert(0, qa)
    from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS, setup_paths
    setup_paths()
    from M3RefTracking import M3RefTracking

    chi2_cut = M3_CHI2_CUT if chi2_cut is None else chi2_cut
    min_nclus = M3_MIN_NCLUS if min_nclus is None else min_nclus

    rays = M3RefTracking(ray_dir if ray_dir.endswith(os.sep) else ray_dir + os.sep,
                         file_nums=list(file_nums), single_track=True,
                         chi2_cut=chi2_cut, min_nclus=min_nclus)
    d = rays.ray_data
    xu, yu, zu = (np.asarray(ak_flat(d[k])) for k in ('X_Up', 'Y_Up', 'Z_Up'))
    xd, yd, zd = (np.asarray(ak_flat(d[k])) for k in ('X_Down', 'Y_Down',
                                                      'Z_Down'))

    good = np.isfinite(xu) & np.isfinite(yu) & np.isfinite(xd) & np.isfinite(yd)
    if require_paddles:
        # the bench triggers on the two 60 x 60 cm paddles, so a drawn track
        # should be one that could actually have fired it
        half = G.SCINT_MM / 2
        for z_p in G.BENCH_SCINT_Z.values():
            t = (z_p - zd) / np.where(zu - zd == 0, np.nan, zu - zd)
            good &= (np.abs(xd + t * (xu - xd)) < half) & \
                    (np.abs(yd + t * (yu - yd)) < half)
    idx = np.flatnonzero(good)
    if idx.size == 0:
        raise RuntimeError(f'no rays survived the cuts in {ray_dir}')

    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(n, idx.size), replace=False)

    # The rays are fitted between the two M3 planes (z = 24 and 1302), but the
    # muon that made them came through BOTH scintillators, so extrapolate the
    # straight line out past each paddle rather than stopping at the tracker.
    z_hi = max(G.BENCH_SCINT_Z.values()) + overshoot
    z_lo = min(G.BENCH_SCINT_Z.values()) - overshoot

    out = []
    for i in pick:
        lo = np.array([xd[i], yd[i], zd[i]])          # bottom M3 plane
        hi = np.array([xu[i], yu[i], zu[i]])          # top M3 plane
        u = (hi - lo) / np.linalg.norm(hi - lo)       # points upwards
        top = lo + u * (z_hi - lo[2]) / u[2]
        bot = lo + u * (z_lo - lo[2]) / u[2]
        # returned in TRAVEL order -- a cosmic goes downwards -- so the arrow
        # head always belongs on the second point, same as cosmic_tracks()
        out.append((top, bot))
    return out


def ak_flat(arr):
    """Awkward or numpy -> flat numpy, one entry per (single-track) event."""
    a = np.asarray(arr.tolist(), dtype=object) if hasattr(arr, 'tolist') \
        else np.asarray(arr)
    try:
        return np.asarray(arr, dtype=float).ravel()
    except Exception:
        return np.array([float(v[0]) if np.ndim(v) else float(v) for v in a])


def add_tracks(p, tracks, radius=3.4, color=None):
    """Muons, each with an arrow head on the exit (downward) end."""
    color = color or S.COL['track_mu']
    for mesh in M.tracks_with_heads(tracks, radius):
        p.add_mesh(mesh, **S.mat('glow', color))
    return tracks
