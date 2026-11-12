#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_track_pad_frames.py -- do the drawn track and the drawn pad agree?

The renderer places pads with ``geometry.p2_pad_to_lab``, which carries
``P2_MIRROR = -1`` (the beam's-eye convention the board scripts use).  The
published uRWELL -> P2 alignment instead maps a uRWELL track point into the
*pad* frame (pad_cx, pad_cy).  Composing the two is only meaningful if the
handedness matches, and that is not something to assume: if it is wrong the
figure still renders, it just quietly puts the hit pad on the wrong side of
the fan.

So: for every real uRWELL track, project to each station, push it through the
published rigid transform to get a pad-frame point, map that to the lab, and
compare with the lab position of the track itself at that z.  The right
convention is the one where those agree to about a pad.

Run:  ../.venv/bin/python tools/check_track_pad_frames.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import geometry as G          # noqa: E402

ALIGN = os.path.join(os.path.dirname(HERE), 'sps_beam_test_26', 'analysis',
                     'urw_mapping', 'mapping_alignment.json')
TRACKS = os.path.join(HERE, 'data', 'urwell_tracks.csv')
Z_FRONT, Z_BACK = 0.0, 1370.0


def rigid(align, name):
    st = align['urwell_to_p2_frame']['stations'][name]
    r = st['rigid_mean']
    th = np.deg2rad(r['theta_deg'])
    A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return A, np.array([r['dx'], r['dy']]), float(st['z_mm'])


def main():
    align = json.load(open(ALIGN))
    rows = list(csv.DictReader(open(TRACKS)))
    print(f'{len(rows)} real uRWELL tracks, '
          f'{len(align["urwell_to_p2_frame"]["stations"])} stations\n')

    c = G.URW_ACTIVE_MM / 2
    _m, polys, apex = G.load_p2_pads()

    for mirror in (-1, +1):
        print(f'--- P2_MIRROR = {mirror:+d} '
              f'{"(as the renderer draws pads)" if mirror == G.P2_MIRROR else ""}')
        for name in ('P2_IN', 'P2_MID', 'P2_OUT'):
            A, t, z = rigid(align, name)
            dx, dy = [], []
            for r in rows:
                f = np.array([float(r['front_x_mm']), float(r['front_y_mm'])])
                b = np.array([float(r['back_x_mm']), float(r['back_y_mm'])])
                u = f + (b - f) * (z - Z_FRONT) / (Z_BACK - Z_FRONT)
                # the track, in the lab, using the renderer's own convention
                trk = np.array([u[0] - c, G.SPS_BEAM_HEIGHT + (u[1] - c)])
                # the same point pushed into the pad frame, then to the lab
                pad = A @ u + t
                lab = G.p2_pad_to_lab(pad, apex=apex, mirror=mirror)
                dx.append(lab[0] - trk[0])
                dy.append(lab[1] - trk[1])
            dx, dy = np.array(dx), np.array(dy)
            print(f'   {name}:  dx {np.median(dx):+8.2f} mm '
                  f'(spread {dx.std():5.2f})   '
                  f'dy {np.median(dy):+8.2f} mm (spread {dy.std():5.2f})')
        print()

    # ---------------------------------------------------------------------- #
    # Neither mirror reconciles the two conventions, because the fitted
    # uRWELL -> pad rotation composed with the pad -> lab rotation closes to
    # ~180 deg (documented), and the renderer's extra x mirror then makes the
    # whole composition a REFLECTION: x agrees and y is negated.
    #
    # The fix is not to pick a mirror but to stop mixing conventions: draw the
    # track through the SAME pad-frame -> lab composition as the pads.  Use one
    # station's transform for the whole line (so it stays exactly straight) and
    # each station's own transform for its hit pad.  The disagreement that
    # introduces is the station-to-station spread of the fit, which the
    # alignment note gives as 0.7 mm in x and 1.6 mm in y.
    print('--- track drawn through the pad-frame composition (P2_MID '
          'transform), pads through their own')
    A0, t0, _z0 = rigid(align, 'P2_MID')
    for name in ('P2_IN', 'P2_MID', 'P2_OUT'):
        A, t, z = rigid(align, name)
        dx, dy = [], []
        for r in rows:
            f = np.array([float(r['front_x_mm']), float(r['front_y_mm'])])
            b = np.array([float(r['back_x_mm']), float(r['back_y_mm'])])
            u = f + (b - f) * (z - Z_FRONT) / (Z_BACK - Z_FRONT)
            trk = G.p2_pad_to_lab(A0 @ u + t0, apex=apex)
            pad = G.p2_pad_to_lab(A @ u + t, apex=apex)
            dx.append(pad[0] - trk[0])
            dy.append(pad[1] - trk[1])
        dx, dy = np.array(dx), np.array(dy)
        print(f'   {name}:  dx {np.median(dx):+6.2f} mm '
              f'(spread {dx.std():4.2f})   '
              f'dy {np.median(dy):+6.2f} mm (spread {dy.std():4.2f})')
    print()

    # where does the beam actually land on the fan, in the lab?
    A, t, z = rigid(align, 'P2_MID')
    u = np.array([[float(r['front_x_mm']), float(r['front_y_mm'])] for r in rows])
    v = np.array([[float(r['back_x_mm']), float(r['back_y_mm'])] for r in rows])
    mid = u + (v - u) * (z - Z_FRONT) / (Z_BACK - Z_FRONT)
    pad = (A @ mid.T).T + t
    for mirror in (-1, +1):
        lab = G.p2_pad_to_lab(pad, apex=apex, mirror=mirror)
        print(f'beam spot on P2_MID, lab, mirror {mirror:+d}: '
              f'x {lab[:, 0].mean():+7.1f} +- {lab[:, 0].std():4.1f}, '
              f'height {lab[:, 1].mean():7.1f} +- {lab[:, 1].std():4.1f} mm')
    print(f'   (the beam axis is drawn at height '
          f'{G.SPS_BEAM_HEIGHT:.0f} mm above the table; the documented spot '
          f'height is 250 mm)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
