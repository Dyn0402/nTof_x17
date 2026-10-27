"""diagram.py -- the 60 ns depth-bin mechanism as an inline-SVG schematic for the FD slide note.

Two panels (a shallow and a steep track) of the drift gap cut into depth bins.  Each bin's charge is drawn as the
model sees it (a uniform bar along the bin's segment of the track) and as it really is (a few lumps: ionisation
clusters x gain), above the readout strips.  When a bin's segment is shorter than a strip the two look the same
to the strips; once it spans a strip or more they do not.
"""
import math

import numpy as np


def panel(sd, ox, oy, ang_deg, v_dt_mm, seed, label, w=760, h=520):
    """one panel at (ox, oy) in svg px.  Gap 28 mm drawn as 4.6 depth bins of v*dt each."""
    rs = np.random.default_rng(seed)
    G = 28.0
    sc = 24.0                       # px per mm
    pitch = 0.78
    tan = math.tan(math.radians(ang_deg))
    x_mesh = 2.0                   # mm, track crosses the mesh here
    out = []
    top = oy + 50
    gx0 = ox + 90
    gw = w - 100
    ghp = 400.0                     # gap height in px (depth compressed)
    vz = ghp / G                    # px per mm of depth
    # cathode, mesh
    out.append(sd.line(gx0, top, gx0 + gw, top, sd.INK, 3))
    out.append(sd.line(gx0, top + ghp, gx0 + gw, top + ghp, sd.INK, 3))
    out.append(sd.T(gx0 + gw, top - 8, 'cathode', 19, sd.MUT, 'end'))
    out.append(sd.T(gx0 + gw, top + ghp - 8, 'mesh', 19, sd.MUT, 'end'))
    # depth-bin boundaries
    nb = int(G / v_dt_mm) + 1
    for k in range(1, nb):
        y = top + ghp - k * v_dt_mm * vz
        if y > top:
            out.append(sd.line(gx0, y, gx0 + gw, y, sd.RULE, 1.5, '6 5'))
    yb0, yb1 = top + ghp, top + ghp - v_dt_mm * vz
    out.append(sd.line(gx0 - 14, yb0, gx0 - 14, yb1, sd.INK, 2))
    out.append(sd.T(gx0 - 20, (yb0 + yb1) / 2 + 6, '1 bin', 19, sd.INK, 'end'))
    X = lambda xmm: gx0 + 20 + xmm * sc  # noqa: E731
    # track
    out.append(sd.line(X(x_mesh), top + ghp, X(x_mesh + G * tan), top, sd.INK, 2.5, '3 4'))
    # bins: model bar vs real lumps, drawn just below each bin's mid-depth line
    for k in range(nb):
        z0, z1 = k * v_dt_mm, min((k + 1) * v_dt_mm, G)
        if z0 >= G:
            break
        xa, xb = x_mesh + z0 * tan, x_mesh + z1 * tan
        ym = top + ghp - 0.5 * (z0 + z1) * vz
        # model: uniform bar along the segment
        out.append(f'<rect x="{min(X(xa), X(xb)) :.1f}" y="{ym - 9:.1f}" width="{max(abs(X(xb) - X(xa)), 3):.1f}" '
                   f'height="18" fill="{sd.BLUE}" fill-opacity="0.30" stroke="{sd.BLUE}" stroke-width="1.5" rx="3"/>')
        # real: a few lumps along the segment
        ncl = rs.poisson(2.8 * (z1 - z0) / math.cos(math.radians(ang_deg)))
        for _ in range(max(ncl, 1)):
            u = rs.uniform(0, 1)
            size = min(rs.pareto(1.0) + 1, 12)
            xs = xa + u * (xb - xa)
            out.append(f'<circle cx="{X(xs):.1f}" cy="{ym:.1f}" r="{2.0 + 1.2 * math.sqrt(size):.1f}" '
                       f'fill="{sd.RED}" fill-opacity="0.85"/>')
    # strips
    sy = top + ghp + 26
    ns = int((gw - 40) / (pitch * sc))
    for i in range(ns):
        xs = gx0 + 20 + i * pitch * sc
        out.append(f'<rect x="{xs + 1:.1f}" y="{sy:.1f}" width="{pitch * sc - 2:.1f}" height="18" fill="{sd.GREY}" '
                   f'fill-opacity="0.55"/>')
    out.append(sd.T(gx0 + gw / 2, sy + 46, 'readout strips, 0.78 mm pitch', 19, sd.MUT))
    seg = v_dt_mm * tan
    out.append(sd.T(gx0, oy + 22, label, 26, sd.INK, 'start', 600))
    out.append(sd.T(gx0, sy + 80, f'one bin spans {seg:.2f} mm across the strips = {seg / pitch:.1f} strip', 23,
                    sd.RED if seg > pitch else sd.GREEN, 'start', 600))
    return ''.join(out)


def mechanism_svg(sd, v_dt_mm=2.15):
    w, h = 1664, 600
    inner = (panel(sd, 0, 0, 10, v_dt_mm, 4, 'shallow track, 10°') +
             panel(sd, 850, 0, 35, v_dt_mm, 5, 'steep track, 35°'))
    return sd.svg(w, h, inner, '60 ns depth bins against the strips')
