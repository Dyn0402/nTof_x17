#!/usr/bin/env python3
"""Cache the geometry that make_readout_layout.py draws from the MX17 readout-board production gerbers.

Source: MX17_Geant/design/gerbers/readout_pcb/ (DFS3498A_*.gbr; set GERBER_DIR to override), read with
MX17_Geant/scripts/gerber/gerber_outline.py.  Output: data/readout_gerber_extract.npz (gerber frame, mm: origin at the
centre of the pad matrix; pad centres at +-199.29).  Gerber file names are the fab's: L2-pads = board L4 (pads),
L3-TrackY = L5 (Y strips), L4-TrackX = L6 (X strips).

Extracted:
  fan_top    segments (x0,y0,x1,y1) of every Y strip whose on-layer copper reaches the +y connectors (the even strips,
             256 of 512), long strip bus excluded.  fan_right: the same for X strips and the +x connectors.
  via_y, via_x  landing pads 0.6 x 1.0 mm 1.1 mm beyond the strip ends: the odd strips leave the layer here (their
             fan-out is on L7, whose artwork is not in the gerber set).
  conn       the 1200 connector contact pads (0.55 mm), four blocks of 300.
  patch_*    a 4.7 x 4.7 mm window at the matrix centre: pads, via dots, Y/X buses and stubs.
"""
import collections, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
G = Path(os.environ.get('GERBER_DIR', Path.home() / 'CLionProjects/MX17_Geant/design/gerbers/readout_pcb'))
sys.path.insert(0, str(Path(os.environ.get('GERBER_PY', Path.home() / 'CLionProjects/MX17_Geant/scripts/gerber'))))
from gerber_outline import parse          # noqa: E402
from scipy.spatial import cKDTree         # noqa: E402

TOL, LONG = 0.15, 300.0


def fan(fn, side):
    g = parse(str(G / fn)); S = g.segments; n = len(S)
    pts = np.array([(s.x0, s.y0) for s in S] + [(s.x1, s.y1) for s in S])
    P = list(range(n))

    def f(a):
        while P[a] != a:
            P[a] = P[P[a]]; a = P[a]
        return a
    for i, j in cKDTree(pts).query_pairs(TOL):
        a, b = f(i % n), f(j % n)
        if a != b:
            P[a] = b
    comp = collections.defaultdict(list)
    for i in range(n):
        comp[f(i)].append(i)
    out = []
    for idx in comp.values():
        ss = [S[i] for i in idx]
        if sum(max(abs(s.x1 - s.x0), abs(s.y1 - s.y0)) > LONG for s in ss) != 1:
            continue
        ext = max(max(s.y0, s.y1) for s in ss) if side == 'top' else max(max(s.x0, s.x1) for s in ss)
        if ext < 215:
            continue
        out += [(s.x0, s.y0, s.x1, s.y1) for s in ss if max(abs(s.x1 - s.x0), abs(s.y1 - s.y0)) <= LONG]
    return np.array(out), g


def main():
    fan_top, gy = fan('DFS3498A_L3-TrackY.gbr', 'top')
    fan_right, gx = fan('DFS3498A_L4-TrackX.gbr', 'right')
    gp = parse(str(G / 'DFS3498A_L2-pads.gbr'))
    via_y = np.array([(f.x, f.y) for f in gp.flashes if f.aperture == 15])
    via_x = np.array([(f.x, f.y) for f in gp.flashes if f.aperture == 16])
    conn = np.array([(f.x, f.y) for f in gp.flashes if f.aperture == 12])
    w = 4.7
    inw = lambda x, y: -1 <= x <= w - 1 and -1 <= y <= w - 1
    pad = np.array([(f.x, f.y) for f in gp.flashes if f.aperture == 10 and inw(f.x, f.y)])
    def seg_in(g, vertical):
        r = []
        for s in g.segments:
            if max(s.x0, s.x1) < -1 or min(s.x0, s.x1) > w - 1 or max(s.y0, s.y1) < -1 or min(s.y0, s.y1) > w - 1:
                continue
            r.append((s.x0, s.y0, s.x1, s.y1))
        return np.array(r)
    sy, sx = seg_in(gy, True), seg_in(gx, False)
    dot_y = np.array([(f.x, f.y) for f in gy.flashes if f.aperture == 10 and inw(f.x, f.y)])
    dot_x = np.array([(f.x, f.y) for f in gx.flashes if f.aperture == 10 and inw(f.x, f.y)])
    np.savez_compressed(HERE / 'data' / 'readout_gerber_extract.npz', fan_top=fan_top, fan_right=fan_right,
                        via_y=via_y, via_x=via_x, conn=conn, patch_pad=pad, patch_seg_y=sy, patch_seg_x=sx,
                        patch_dot_y=dot_y, patch_dot_x=dot_x)
    print('fan_top', fan_top.shape, 'fan_right', fan_right.shape, 'conn', conn.shape, 'pad', pad.shape)


if __name__ == '__main__':
    main()
