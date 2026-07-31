#!/usr/bin/env python3
"""
14_board_map.py — det4's efficiency map drawn on the readout board, with the beam spot.

Puts the fine-kernel (3 mm) within-5 mm efficiency map onto a scale drawing of
the MX17 readout PCB taken from the Gerbers in ~/x17/mx17_gerbers, so the live
band can be found on the real hardware, and marks where to aim an 80 mm beam.

Board geometry from `Gerber pcb readout/DFS3498A_det.gbr` (board outline, the
four mezzanine footprints, the frame mounting holes) and
`DFS3498A_activearea.gbr` (the 399.36 mm metallised square). Gerber coordinates
have the active area centred on (0,0); detector-local strip coordinates run
0-398.58 mm, so local = gerber + 199.68.

Connector blocks come from the strip map itself (`mx17_m1_map.csv`): connector k
covers channels 64(k-1)..64k-1, i.e. strip positions 49.92(k-1) .. +49.14 mm.
The Gerber confirms two 4-connector mezzanines per plane, centred at local
~99 mm and ~299 mm along their edge.

ORIENTATION: drawn with the X-plane connectors along the bottom and the Y-plane
connectors on the right, as requested. On the physical board both connector
banks are on the two edges that carry the mezzanine footprints (gerber +X and
+Y); which bank belongs to which plane could not be confirmed from the routing
layers (both L3-TrackY and L4-TrackX fan out to both edges), so treat the
edge assignment as the drawing convention and the connector *numbering* — which
is what you need to find a strip — as measured.

    ../../.venv/bin/python mx_june_cosmic_qa/det4_sps_assessment/14_board_map.py
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths                # noqa: E402
setup_paths()
import matplotlib                                            # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import Rectangle, Circle             # noqa: E402
import pandas as pd                                          # noqa: E402
import cosmic_micro_tpc_analysis as cm                       # noqa: E402
from common.Mx17StripMap import Mx17StripMap                 # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE              # noqa: E402

sys.path.insert(0, HERE)
from importlib import import_module                          # noqa: E402
ref_to_det = import_module('01_uniformity').ref_to_det

GERBER = '/home/dylan/x17/mx17_gerbers/Gerber pcb readout'
ACTIVE_HALF = 199.68              # DFS3498A_activearea.gbr: 399.36 mm square
OFFSET = ACTIVE_HALF              # local = gerber + OFFSET
BEAM_D = 80.0
KERNEL = 3.0
BAND = (177.0, 215.0)     # the live band, detector-local X [mm]


def parse_gerber(fn):
    """Minimal Gerber reader: returns {aperture: [(x, y), ...]} flashes and draw path."""
    txt = open(os.path.join(GERBER, fn), errors='ignore').read()
    sc = 10 ** int(re.search(r'%FSLAX\d(\d)', txt).group(1))
    ap = {int(m.group(1)): (m.group(2), [float(v) for v in m.group(3).split('X')])
          for m in re.finditer(r'%ADD(\d+)([A-Z]),([\d.X]+)\*%', txt)}
    flashes, draws = {}, []
    cur = None
    x = y = 0.0
    for line in txt.splitlines():
        line = line.strip()
        m = re.fullmatch(r'D(\d+)\*', line)
        if m and int(m.group(1)) >= 10:
            cur = int(m.group(1))
            continue
        m = re.fullmatch(r'(?:X(-?\d+))?(?:Y(-?\d+))?D0([123])\*', line)
        if not m:
            continue
        if m.group(1):
            x = int(m.group(1)) / sc
        if m.group(2):
            y = int(m.group(2)) / sc
        op = int(m.group(3))
        if op == 3:
            flashes.setdefault(cur, []).append((x, y))
        else:
            draws.append((x, y, op))
    return ap, flashes, draws


def sliding(x, y, val, gx, gy, kernel, minn):
    r2 = kernel ** 2
    out = np.full((len(gx), len(gy)), np.nan)
    for i, xg in enumerate(gx):
        dx2 = (x - xg) ** 2
        near = dx2 <= r2
        if not near.any():
            continue
        xs, ys, vs = dx2[near], y[near], val[near]
        for j, yg in enumerate(gy):
            m = (xs + (ys - yg) ** 2) <= r2
            if m.sum() >= minn:
                out[i, j] = vs[m].mean()
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--key', default='g_det4')
    ap_.add_argument('--kernel', type=float, default=KERNEL)
    ap_.add_argument('--minn', type=int, default=1)
    ap_.add_argument('--grid', type=float, default=1.33)
    ap_.add_argument('--out', default=HERE)
    ap_.add_argument('--views', default='none,ccw',
                     help="comma-separated subset of none,ccw,cw")
    args = ap_.parse_args()
    cfg = get_config(args.key)

    # ---------------- efficiency map in detector-local coordinates ----------
    params = cm.load_alignment(os.path.join(cfg.OUT_BASE, 'alignment_tpc_veto50',
                                            'alignment.json'))
    d = pd.read_csv(os.path.join(cfg.OUT_BASE, 'efficiency', 'ray_hit_miss_list.csv'))
    d = d[np.isfinite(d.x) & np.isfinite(d.y)]
    lx, ly = ref_to_det(d.x.to_numpy(), d.y.to_numpy(), params)
    within = d['within'].astype(str).str.lower().isin(('true', '1')).to_numpy(float)
    gx = np.arange(0, 398.58 + args.grid, args.grid)
    gy = np.arange(0, 398.58 + args.grid, args.grid)
    eff = sliding(lx, ly, within, gx, gy, args.kernel, args.minn)

    # ---------------- best 80 mm beam spot ----------------------------------
    ay0, ay1 = TRUE_ACTIVE['y']
    R = BEAM_D / 2
    best = (-1, None)
    for i, cx in enumerate(gx):
        if cx - R < 0 or cx + R > 398.58:
            continue
        for j, cy in enumerate(gy):
            if cy - R < ay0 or cy + R > ay1:
                continue
            m = (lx - cx) ** 2 + (ly - cy) ** 2 <= R ** 2
            if m.sum() < 200:
                continue
            v = within[m].mean()
            if v > best[0]:
                best = (v, (float(cx), float(cy)), int(m.sum()))
    spot_eff, (bx, by), n_in = best
    # efficiency vs spot centre Y at the best X, so the choice can be judged
    scan_y = []
    for cy in np.arange(ay0 + R, ay1 - R + 1, 10.0):
        m = (lx - bx) ** 2 + (ly - cy) ** 2 <= R ** 2
        if m.sum() >= 200:
            scan_y.append((float(cy), float(within[m].mean()), int(m.sum())))
    # strip / channel numbers of the live band, for finding it on the hardware
    def strip_of(mm):
        i = int(round(mm / 0.78))
        return i, i // 64 + 1, i % 64
    band_ch = {f'{v:.0f} mm': dict(zip(('strip', 'connector', 'channel_in_connector'),
                                       strip_of(v)))
               for v in (177.0, 215.0, bx - R, bx + R)}

    # ---------------- board geometry from the Gerbers -----------------------
    _, fl, draws = parse_gerber('DFS3498A_det.gbr')
    outline = np.array([(x, y) for x, y, op in draws]) + OFFSET
    mezz = np.array(fl[10]) + OFFSET          # 2.1 mm mezzanine mounting holes
    frame = np.array(fl[15]) + OFFSET         # 4.2 mm frame mounting holes

    sm = Mx17StripMap(cfg.MAP_CSV_PATH)
    conn = {}
    for axis in ('x', 'y'):
        conn[axis] = []
        for k in range(1, 9):
            p = [sm.lookup(axis, k, l) for l in range(64)]
            v = [q[0] if axis == 'x' else q[1] for q in p if q]
            conn[axis].append((k, float(min(v)), float(max(v))))

    rep = dict(run_key=args.key, kernel_mm=args.kernel,
               local_to_gerber_offset_mm=-OFFSET,
               beam=dict(diameter_mm=BEAM_D,
                         centre_local_mm=[bx, by],
                         centre_gerber_mm=[bx - OFFSET, by - OFFSET],
                         mean_efficiency_in_spot=float(spot_eff),
                         n_rays_in_spot=int(n_in),
                         x_connectors_covered=[k for k, a, b in conn['x']
                                               if b >= bx - R and a <= bx + R],
                         y_connectors_covered=[k for k, a, b in conn['y']
                                               if b >= by - R and a <= by + R],
                         scan_over_centre_y=scan_y,
                         landmark_strips=band_ch),
               board=dict(outline_local_mm=[[float(v) for v in p] for p in outline],
                          n_mezzanine_holes=len(mezz), n_frame_holes=len(frame)),
               connectors=conn)
    with open(os.path.join(args.out, f'board_map_{args.key}.json'), 'w') as f:
        json.dump(rep, f, indent=1)

    # ------------------------------- draw -----------------------------------
    # Two views of the same drawing:
    #   rot=False  X bank at the bottom, Y bank on the right, bands vertical
    #              (the bench view; local X horizontal)
    #   'ccw'/'cw' the same rotated 90 deg counter-clockwise / clockwise. Either
    #              puts the gain bands HORIZONTAL, which is the beam mounting:
    #              a left-right yaw of the board then sweeps the track ALONG a
    #              band, so it never leaves the live stripe, and the inclination
    #              is seen by the Y plane, which gets the micro-TPC lever arm.
    #              Tilting up-down instead gives the lever arm to the X plane and
    #              crosses the bands (measured fine up to |tan| ~ 0.6, see
    #              13_beam_window.py). CCW puts the X bank on the RIGHT and the
    #              Y bank on TOP; CW puts them left and bottom.
    def render(mode):
        rot = mode != 'none'
        ccw = mode == 'ccw'
        def T(x, y):
            return (y, x) if rot else (x, y)

        def rect(x, y, w, h, **kw):
            (a, b) = T(x, y)
            return Rectangle((a, b), h if rot else w, w if rot else h, **kw)

        fig, ax = plt.subplots(figsize=(13.5, 12))
        x0, x1 = outline[:, 0].min(), outline[:, 0].max()
        y0, y1 = outline[:, 1].min(), outline[:, 1].max()
        y0, y1 = 398.58 - y1, 398.58 - y0        # mirror the board margin in Y
        ax.add_patch(rect(x0, y0, x1 - x0, y1 - y0, fc='#f3f0e8', ec='#3a3a3a',
                          lw=1.8, zorder=0))
        ax.add_patch(rect(-0.39, -0.39, 399.36, 399.36, fc='none', ec='#3a3a3a',
                          lw=1.0, ls='--', zorder=1))

        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('#e8e5dc')
        m = np.ma.masked_invalid(eff)
        if rot:
            im = ax.imshow(m, origin='lower', extent=[gy[0], gy[-1], gx[0], gx[-1]],
                           vmin=0, vmax=1, cmap=cmap, interpolation='nearest',
                           zorder=2)
        else:
            im = ax.imshow(m.T, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]],
                           vmin=0, vmax=1, cmap=cmap, interpolation='nearest',
                           zorder=2)
        ax.add_patch(rect(0, TRUE_ACTIVE['y'][0], 398.58,
                          TRUE_ACTIVE['y'][1] - TRUE_ACTIVE['y'][0],
                          fc='none', ec='w', lw=1.0, ls=':', zorder=3))

        fx_, fy_ = T(frame[:, 0], 398.58 - frame[:, 1])
        ax.plot(fx_, fy_, 'o', ms=4.5, mfc='none', mec='#3a3a3a', lw=.9, zorder=4)

        def draw_mezz(pts, horiz):
            for lo, hi in ((0, 199.68), (199.68, 400)):
                key = pts[:, 0] if horiz else pts[:, 1]
                g = pts[(key >= lo) & (key < hi)]
                if not len(g):
                    continue
                a0, a1 = g[:, 0].min(), g[:, 0].max()
                b0, b1 = g[:, 1].min(), g[:, 1].max()
                pad = 34
                if horiz:
                    ax.add_patch(rect(a0 - pad, b0 - 5, a1 - a0 + 2 * pad,
                                      b1 - b0 + 10, fc='#ddd6c4', ec='#3a3a3a',
                                      lw=1.0, zorder=4))
                else:
                    ax.add_patch(rect(a0 - 5, b0 - pad, a1 - a0 + 10,
                                      b1 - b0 + 2 * pad, fc='#ddd6c4',
                                      ec='#3a3a3a', lw=1.0, zorder=4))
                gxx, gyy = T(g[:, 0], g[:, 1])
                ax.plot(gxx, gyy, 's', ms=2.6, color='#3a3a3a', zorder=5)
        top = mezz[mezz[:, 1] > 398.58].copy()
        top[:, 1] = 398.58 - top[:, 1]
        draw_mezz(top, horiz=True)
        draw_mezz(mezz[mezz[:, 0] > 398.58], horiz=False)

        hb = wb = 15.0
        for k, a, b in conn['x']:
            ax.add_patch(rect(a, -hb - 3, b - a, hb, fc='#0072b2', ec='k',
                              lw=.8, alpha=.9, zorder=6))
            ltx, lty = T((a + b) / 2, -hb / 2 - 3)
            ax.text(ltx, lty, f'X{k}', ha='center', va='center', color='w',
                    fontsize=9.5, fontweight='bold',
                    rotation=90 if rot else 0, zorder=7)
        for k, a, b in conn['y']:
            ax.add_patch(rect(398.58 + 3, a, wb, b - a, fc='#d55e00', ec='k',
                              lw=.8, alpha=.9, zorder=6))
            ltx, lty = T(398.58 + 3 + wb / 2, (a + b) / 2)
            ax.text(ltx, lty, f'Y{k}', ha='center', va='center', color='w',
                    fontsize=9.5, fontweight='bold',
                    rotation=0 if rot else 90, zorder=7)
        lblx = T(199, -hb - 50)
        ax.text(*lblx, f'X-plane connectors (FEU {cfg.MX17_FEUS[0]}) — these strips '
                f'measure local X, the coordinate the bands live in',
                ha='center', va='center' if rot else 'top', fontsize=9.5,
                color='#0072b2', rotation=90 if rot else 0)
        lbly = (199, 450) if ccw else ((455, 409) if rot else T(472, 199))
        ax.text(*lbly, f'Y-plane connectors (FEU {cfg.MX17_FEUS[1]})' if not rot
                else f'Y-plane connectors\n(FEU {cfg.MX17_FEUS[1]})',
                ha='center' if not rot or ccw else 'left', va='center',
                rotation=0 if rot else 270, fontsize=9.5, color='#d55e00')

        s0, c0, ch0 = strip_of(177.0)
        s1, c1, ch1 = strip_of(215.0)
        for v in (177.0, 215.0):
            if rot:
                ax.plot([0, 398.58], [v, v], color='w', ls='--', lw=1.1, zorder=6)
            else:
                ax.plot([v, v], [0, 398.58], color='w', ls='--', lw=1.1, zorder=6)
        bt = T(196, 388) if not rot else ((352, 258) if ccw else (60, 258))
        ax.text(*bt, 'live band  X 177–215 mm\n'
                f'strips {s0}–{s1}  =  X{c0}.ch{ch0} → X{c1}.ch{ch1}',
                color='w', ha='center', va='top' if not rot else 'center',
                fontsize=9, zorder=7,
                bbox=dict(fc='#00000055', ec='none', pad=2))

        # faint marker for the free optimum found by the scan
        ax.add_patch(Circle(T(bx, by), R, fc='none', ec='r', lw=1.2, ls=':',
                            alpha=.55, zorder=7))
        ax.text(*T(bx, by), f'free optimum\n({bx:.0f}, {by:.0f})  eff {spot_eff:.2f}',
                color='r', alpha=.7, ha='center', va='center', fontsize=8,
                zorder=7, bbox=dict(fc='#ffffff99', ec='none', pad=1.2))
        # the proposed spot: X4+X5 and Y4+Y5 only
        for k, a, b in conn['x']:
            if k in covered('x', spot_x):
                ax.add_patch(rect(a, -hb - 3, b - a, hb, fc='none', ec='r',
                                  lw=2.2, zorder=7))
        for k, a, b in conn['y']:
            if k in covered('y', spot_y):
                ax.add_patch(rect(398.58 + 3, a, wb, b - a, fc='none', ec='r',
                                  lw=2.2, zorder=7))
        ax.add_patch(Circle(T(spot_x, spot_y), R, fc='none', ec='r', lw=2.8,
                            zorder=8))
        ax.plot(*[[v] for v in T(spot_x, spot_y)], '+', color='r', ms=16, mew=2.2,
                zorder=8)
        ax.annotate(f'beam Ø{BEAM_D:.0f} mm  →  X4+X5, Y4+Y5 only\n'
                    f'detector-local ({spot_x:.0f}, {spot_y:.0f}) mm\n'
                    f'gerber ({spot_x - OFFSET:+.0f}, {spot_y - OFFSET:+.0f}) mm\n'
                    f'centred on the Y4/Y5 interface and the live band\n'
                    f'mean efficiency {e2:.2f}  (n={m2.sum():,})',
                    T(spot_x + R * .71, spot_y - R * .71),
                    T(x1 + 4, 336) if not rot
                    else ((-124, 430) if ccw else (y1 + 30, 95)),
                    color='r', fontsize=10.5, ha='left', va='top',
                    arrowprops=dict(arrowstyle='->', color='r', lw=1.6), zorder=9)

        if ccw:
            ax.set_xlim(y1 + 26, y0 - 210)             # inverted -> 90 deg CCW
            ax.set_ylim(x0 - 12, x1 + 12)
            ax.set_xlabel('detector-local Y [mm], increasing LEFT'
                          '      (gerber Y = local Y − 199.68)')
            ax.set_ylabel('detector-local X [mm]      (gerber X = local X − 199.68)')
            sub = ('rotated 90° CCW for the beam: bands are HORIZONTAL, so a '
                   'left–right yaw of the board runs the track along a band '
                   '(lever arm on the Y plane).  X bank right, Y bank top.')
        elif rot:
            ax.set_xlim(y0 - 26, y1 + 210)
            ax.set_ylim(x1 + 12, x0 - 12)              # inverted -> 90 deg CW
            ax.set_xlabel('detector-local Y [mm]      (gerber Y = local Y − 199.68)')
            ax.set_ylabel('detector-local X [mm], increasing DOWN'
                          '      (gerber X = local X − 199.68)')
            sub = ('rotated 90° CW for the beam: bands are HORIZONTAL, so a '
                   'left–right yaw of the board runs the track along a band '
                   '(lever arm on the Y plane).  X bank left, Y bank bottom.')
        else:
            ax.set_xlim(x0 - 12, x1 + 122)
            ax.set_ylim(y0 - 26, y1 + 10)
            ax.set_xlabel('detector-local X [mm]      (gerber X = local X − 199.68)')
            ax.set_ylabel('detector-local Y [mm]      (gerber Y = local Y − 199.68)')
            sub = ('bench view: X bank at the bottom, bands vertical '
                   '(mirrored in Y vs the Gerber)')
        ax.set_aspect('equal')
        cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, shrink=.72)
        cb.set_label('efficiency within 5 mm of the M3 reference')
        ax.set_title(f'{cfg.DET_NAME} — efficiency ({args.kernel:.0f} mm sliding '
                     f'kernel) on the MX17 readout board, with the proposed beam '
                     f'spot\n{cfg.RUN}/{cfg.SUB_RUN}   |   board from the DFS3498A '
                     f'Gerbers\n{sub}', fontsize=10.5)
        suffix = {'none': '', 'cw': '_rot90cw', 'ccw': '_rot90ccw'}[mode]
        fig.savefig(os.path.join(args.out, f'board_map_{args.key}{suffix}.png'),
                    dpi=145, bbox_inches='tight')
        plt.close(fig)

    # ---- the spot actually proposed: centred on the Y4/Y5 interface in Y and
    # on the middle of the live band in X, which puts it on X4+X5 and Y4+Y5 only
    spot_y = 0.5 * (conn['y'][3][2] + conn['y'][4][1])   # Y4 end .. Y5 start
    spot_x = 0.5 * (BAND[0] + BAND[1])                  # middle of the live band
    m2 = (lx - spot_x) ** 2 + (ly - spot_y) ** 2 <= R ** 2
    e2 = float(within[m2].mean())

    def covered(axis, c):
        return [k for k, a, b in conn[axis] if b >= c - R and a <= c + R]
    rep['beam']['proposed_connector_aligned'] = dict(
        centre_local_mm=[float(spot_x), float(spot_y)],
        centre_gerber_mm=[float(spot_x - OFFSET), float(spot_y - OFFSET)],
        mean_efficiency_in_spot=e2, n_rays_in_spot=int(m2.sum()),
        x_connectors_covered=covered('x', spot_x),
        y_connectors_covered=covered('y', spot_y),
        y4_y5_interface_mm=float(spot_y),
        margin_to_connector_edges_mm=dict(
            x_low=float((spot_x - R) - conn['x'][3][1]),
            x_high=float(conn['x'][4][2] - (spot_x + R)),
            y_low=float((spot_y - R) - conn['y'][3][1]),
            y_high=float(conn['y'][4][2] - (spot_y + R))),
        note='free optimum is elsewhere and better; see beam.centre_local_mm')
    with open(os.path.join(args.out, f'board_map_{args.key}.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    for mode in args.views.split(','):
        render(mode.strip())

    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ('board', 'connectors')}, indent=1))


if __name__ == '__main__':
    main()
