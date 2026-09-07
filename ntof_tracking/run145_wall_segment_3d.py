#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run145_wall_segment_3d.py — the run_79 colored-tracks 3D figure, remade on
run_145 for ALL FOUR arms.

Per arm: the waveform-first tracks that carry a same-arm in-time SiPM wall
hit, colored by the wall segment that fired, drawn through the 3D model
(chamber + wall bars + He-3 capsule). Each fan should land on its segment at
the wall and sweep back through the capsule — the wall/target spread ratio is
the number under the picture, with the label-shuffled null as the scale.

Differences from run_79: full nTOF matching via the slim file (join on
eventId, no clock fit), FULL COVERAGE (the slope_reliable gate is gone —
2026-08-13, the head-on band is real and it is the image core), per-plane
angle constants w0/kw applied, and angles rescaled by the per-arm in-situ
k from the pointing-coincidence estimator.

Renders one still per arm ('all segments' emphasis) plus a 2x2 composite.
Reuses the run79_wall_segment_gif scene machinery.

Usage:
    python -m ntof_tracking.run145_wall_segment_3d [--arms A,B,C,D]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import uproot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ntof_tracking.reco import geometry as geo                      # noqa: E402
from ntof_tracking import run79_wall_segment_gif as W               # noqa: E402
from ntof_tracking.run145_target_imaging import (                   # noqa: E402
    apply_w0_kw, load_tracks)

SLIM = os.environ.get(
    'RUN145_SLIM',
    '/media/dylan/data/x17/slim/out_224670/'
    'ntof_hits_run_145_stat090_0000_224670.root')
OUT = ('/media/dylan/data/x17/beam_july/analysis/wft/run_145/'
       'stat090_0000/imaging_fullcov/wall_3d')
DT_LO, DT_HI = -100.0, 60.0
WAL_CODE = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# In-situ angle scale per arm: median per-track k on the pointing-coincident
# subset. B's drift field is not nominal (degrador absent) and its k is NOT a
# velocity statement — used here only so B's fan is drawn at the scale its own
# data prefers.
#
# READ from the imaging summary, never hard-coded: this used to be the literal
# {'A': 1.239, 'B': 1.918, 'C': 1.488, 'D': 1.136}, which is a figure that
# silently keeps drawing the previous reconstruction's angle scale after the
# reconstruction changes.
def _k_insitu(summary=None):
    p = summary or os.path.join(os.path.dirname(OUT), 'imaging_summary.json')
    with open(p) as f:
        S = json.load(f)
    k = {r['arm']: r['k_phys'] for r in S['results'] if 'k_phys' in r}
    if not k:
        raise SystemExit(f'FATAL: no k_phys in {p} — run '
                         'run145_target_imaging first')
    return k


K_INSITU = _k_insitu()
# az/elev per arm derive from the arm normal; +45 deg matches the run_79 view
ARM_AZ = {a: float(np.degrees(np.arctan2(geo.W_HAT[a][0], geo.W_HAT[a][2])))
          for a in geo.ARMS}


def merged_table(arm: str):
    """events_prelim + earliest in-time same-arm WAL hit from the slim file."""
    df, meta = load_tracks('run_145', 'stat090_0000', arm)
    df, _ = apply_w0_kw(df, arm, meta['bundle']['v_drift'], meta)
    df = df.copy()
    df['x_tan_theta'] *= K_INSITU[arm]
    df['y_tan_theta'] *= K_INSITU[arm]
    f = uproot.open(SLIM)
    h = f['hits'].arrays(['eventId', 'det', 'detn', 'dt_ns', 'is_control'],
                         library='np')
    ev = f['events'].arrays(['eventId', 'bunch'], library='np')
    m = ((h['is_control'] == 0) & (h['det'] == WAL_CODE[arm])
         & (h['dt_ns'] >= DT_LO) & (h['dt_ns'] <= DT_HI))
    o = np.argsort(np.abs(h['dt_ns'][m]))
    first = pd.DataFrame(dict(eventId=h['eventId'][m][o],
                              detn=h['detn'][m][o])) \
        .drop_duplicates('eventId', keep='first').set_index('eventId')
    df['wal_detn'] = df.event_id.map(first['detn']).astype(float)
    evd = pd.DataFrame(ev).set_index('eventId')
    df['BunchNumber'] = df.event_id.map(evd['bunch']).fillna(-1).astype(int)

    keep = (df['x_ok'] & df['y_ok'] & df['x_quality_ok'] & df['y_quality_ok']
            & np.isfinite(df['wal_detn'])
            & (df['x_tan_theta'].abs() < W.TAN_SANE)
            & (df['y_tan_theta'].abs() < W.TAN_SANE)
            & df['x_n_strips'].between(*W.N_STRIPS)
            & df['y_n_strips'].between(*W.N_STRIPS))
    d = df[keep].copy()
    d['seg'] = ((d['wal_detn'] - 1) // 2).astype(int)
    d = d[d['seg'].between(0, W.N_WALL_SEG - 1)]
    return d


def segment_stats(d, arm, tr):
    """run79's segment_stats, but the wall/target coordinate is the projection
    onto the ARM's u axis — run79 used global X, which is u only on arm A
    (B/D's u is along Z, C's is −X: spreads came out 0 or mirrored)."""
    u_hat = np.asarray(geo.U_HAT[arm], float)
    out = {}
    for s in range(W.N_WALL_SEG):
        ds = d[d['seg'] == s]
        grp = W.N_WALL_SEG - 1 - s                       # descending order
        lo, hi = W.wall_segment_u(grp)
        p_wall, p_tgt = W.track_points(ds, tr, -W.WALL_DEPTH, W.W_IN[arm])
        u = p_wall @ u_hat
        t = p_tgt @ u_hat
        out[s] = dict(n=int(len(ds)), lo=lo, hi=hi,
                      med=float(np.median(u)) if len(u) else np.nan,
                      inside=float(np.mean((u >= lo) & (u <= hi)))
                      if len(u) else np.nan,
                      med_target=float(np.median(t)) if len(u) else np.nan)
    return out


def convergence_null(d, arm, tr, seed=7):
    sh = d.copy()
    sh['wal_detn'] = np.random.default_rng(seed).permutation(
        sh['wal_detn'].to_numpy())
    sh['seg'] = ((sh['wal_detn'] - 1) // 2).astype(int)
    return W.convergence(segment_stats(sh, arm, tr))


def render_arm(arm: str, d, tr, th, size=(1150, 1000)):
    """One still: the arm's scene, all four segments emphasised."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=size, border=False)
    pl.set_background(th['surface'])
    pl.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
    act = W.build_scene(pl, d, arm, tr, 'descending', th)
    W.apply_weights(act, np.full(W.N_WALL_SEG, 0.42), th)
    focal = 145.0 * np.asarray(geo.W_HAT[arm], float)
    az, el = np.radians(ARM_AZ[arm] + 45.0), np.radians(32.0)
    pos = focal + 1520.0 * np.array(
        [np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    pl.camera_position = [tuple(pos), tuple(focal), (0, 1, 0)]
    pl.render()
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arms', default='A,B,C,D')
    ap.add_argument('--theme', default='light', choices=sorted(W.THEME))
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    os.makedirs(a.out, exist_ok=True)
    th = W.THEME[a.theme]
    cfg = json.load(open('/media/dylan/data/x17/beam_july/runs/run_145/'
                         'run_config.json'))
    trs = geo.detector_transforms(cfg)

    panels, summary = {}, {}
    for arm in a.arms.split(','):
        d = merged_table(arm)
        tr = trs[f'mx17_{arm}']
        seg = segment_stats(d, arm, tr)
        cv = W.convergence(seg)
        nl = convergence_null(d, arm, tr)
        summary[arm] = dict(n=int(len(d)), k=K_INSITU[arm], seg=seg,
                            convergence=cv, null=nl)
        print(f'[{arm}] {len(d):,} tracks; wall spread '
              f'{cv["spread_wall_mm"]:.0f} mm -> target '
              f'{cv["spread_target_mm"]:.0f} mm '
              f'(null {nl["spread_wall_mm"]:.0f}/'
              f'{nl["spread_target_mm"]:.0f})')
        img = render_arm(arm, d, tr, th)
        p = os.path.join(a.out, f'wall3d_{arm}.png')
        Image.fromarray(img).save(p)
        panels[arm] = (img, cv, nl, len(d))
        print('wrote', p)

    with open(os.path.join(a.out, 'wall3d_summary.json'), 'w') as f:
        json.dump(dict(status='PRELIMINARY', run='run_145',
                       subrun='stat090_0000', slim=SLIM, k_insitu=K_INSITU,
                       coverage='full (no slope_reliable gate, 2026-08-13)',
                       arms=summary), f, indent=1, default=str)

    # --- 2x2 composite with header + legend
    NOTE = {'A': 'the reference arm',
            'B': 'field not nominal — positions carry it',
            'C': 'bundle-limited, qualitative',
            'D': 'known +u compression anomaly visible'}
    arms = list(panels)
    w0, h0 = panels[arms[0]][0].shape[1], panels[arms[0]][0].shape[0]
    sc = 0.72
    w1, h1 = int(w0 * sc), int(h0 * sc)
    head, band, foot = 118, 66, 60
    canvas = Image.new('RGB', (2 * w1, head + 2 * (h1 + band) + foot),
                       th['surface'])
    dr = ImageDraw.Draw(canvas)
    f_t = W._font(30, True)
    f_s = W._font(18)
    f_l = W._font(19, True)
    f_c = W._font(16)
    dr.text((26, 16), 'run_145 — beam tracks by the SiPM wall segment that '
                      'triggered them', font=f_t, fill=th['ink'])
    dr.text((26, 58), 'waveform-first tracks, full coverage (head-on band '
                      'included), in-situ angle scale per arm;', font=f_s,
            fill=th['muted'])
    dr.text((26, 82), 'each fan lands on its segment at the wall and sweeps '
                      'back through the He-3 capsule', font=f_s,
            fill=th['muted'])
    for i, arm in enumerate(arms):
        img, cv, nl, n = panels[arm]
        x, y = (i % 2) * w1, head + (i // 2) * (h1 + band)
        im = Image.fromarray(img).resize((w1, h1), Image.LANCZOS)
        canvas.paste(im, (x, y))
        dr.text((x + 22, y + h1 + 6),
                f'mx17_{arm} — {n:,} tracks — wall {cv["spread_wall_mm"]:.0f}'
                f' mm → target {cv["spread_target_mm"]:.0f} mm '
                f'(null {nl["spread_wall_mm"]:.0f} mm)',
                font=f_l, fill=th['ink'])
        dr.text((x + 22, y + h1 + 34), NOTE.get(arm, ''), font=f_c,
                fill=th['muted'])
    y = head + 2 * (h1 + band) + 8
    x = 26
    for s in range(W.N_WALL_SEG):
        dr.rounded_rectangle([x, y + 2, x + 26, y + 18], radius=3,
                             fill=W.SEG_COLOR[s])
        dr.text((x + 34, y), f'segment {s}', font=f_c, fill=th['ink'])
        x += 160
    dr.text((x + 24, y), 'PRELIMINARY — sub-run 0000, slim nTOF matching',
            font=f_c, fill=th['muted'])
    p = os.path.join(a.out, 'wall3d_run145_all_arms.png')
    canvas.save(p)
    print('wrote', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
