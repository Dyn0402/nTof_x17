#!/usr/bin/env python3
"""
run145_note_figs.py — the eight figures the run_145 note embeds.

`make_run145_note.py` reads them from
`<analysis>/run_145/<subrun>/imaging/note_figs_fullcov/`, but the script that
first drew them (2026-08-13) was never committed, so the note's figures could
not be regenerated when the reconstruction changed — a page whose numbers and
whose pictures come from different reconstructions. This is that script,
re-derived from the note's own captions and reusing `run145_target_imaging`'s
selection verbatim so the figures and the summary JSON always describe the
same tracks.

    ../../.venv/bin/python -m ntof_tracking.run145_note_figs \
        --run run_145 --subrun stat090_0000 --arm A \
        --slim .../ntof_hits_run_145_stat090_0000_224670.root
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402

from ntof_tracking import run145_target_imaging as I              # noqa: E402

U_CAL_MAX = 130.0        # |u| beyond this: edge acceptance + window truncation
U_CAL_MIN = 40.0
TAN_FLOOR = 0.10         # the k estimator divides by tan
CAPSULE_R = 10.0
FIGSIZE = (7.2, 4.6)
DPI = 150


# ------------------------------------------------------------------ selection
def prepare(run, subrun, arm, slim, trs):
    """Everything the figures share, on exactly the imaging selection."""
    df, meta = I.load_tracks(run, subrun, arm)
    v_bundle = meta['bundle']['v_drift']
    df, _ = I.apply_w0_kw(df, arm, v_bundle, meta)
    tr = trs[f'mx17_{arm}']
    d_perp, foot_x = I.plane_geometry(tr)
    L = d_perp          # the PERPENDICULAR distance; |centre| drops the pinwheel

    ok = (df['x_ok'] & df['y_ok']).to_numpy()
    sane = ((np.abs(df['x_tan_theta']) < I.TAN_SANE)
            & (np.abs(df['y_tan_theta']) < I.TAN_SANE)).to_numpy()
    base = ok & sane

    wal = I.slim_wall_events(slim, arm)
    inwal = df['event_id'].isin(wal).to_numpy()
    sel = base & inwal
    # No sign flipping here any more (2026-08-20). The in-plane sign is a
    # measured property of the reconstruction and lives in I.local_x /
    # I.track_lines; the old per-figure `tan *= sign(slope)` was the mirror of
    # it, about a plane centre that sits 16 mm off the beam axis.
    coin, cinfo = I.pointing_coincidence(slim, arm, df, sel, foot_x=foot_x)
    df = df.copy()

    # lever arm measured from the foot of the perpendicular, not the centre
    u = I.local_x(df['x_p0'].to_numpy())
    u_unflipped, tan_unflipped = u.copy(), df['x_tan_theta'].to_numpy().copy()
    lever = u - foot_x
    tan = df['x_tan_theta'].to_numpy()
    inc = (sel & (np.abs(tan) > TAN_FLOOR)
           & (np.abs(lever) > U_CAL_MIN) & (np.abs(lever) < U_CAL_MAX))
    k_i = (lever[inc] / L) / tan[inc]
    k_i = k_i[(k_i > 0) & (k_i < 5)]          # wrong-sign = not from target
    k_coin = (lever[inc & coin] / L) / tan[inc & coin]
    k_coin = k_coin[(k_coin > 0) & (k_coin < 5)]
    return dict(df=df, meta=meta, tr=tr, L=L, v_bundle=v_bundle,
                d_perp=d_perp, foot_x=foot_x, lever=lever, u=u, tan=tan,
                u_unflipped=u_unflipped, tan_unflipped=tan_unflipped,
                base=base, sel=sel, coin=coin, cinfo=cinfo, inc=inc,
                k_track=float(np.median(k_i)) if len(k_i) else float('nan'),
                k_phys=float(np.median(k_coin)) if len(k_coin) else float('nan'),
                n_inc=int(inc.sum()), k_i=k_i, arm=arm)


def backproject(S, mask, k):
    """Closest approach to the beam axis, with tan scaled by k."""
    df = S['df'].copy()
    df['x_tan_theta'] = df['x_tan_theta'] * k
    df['y_tan_theta'] = df['y_tan_theta'] * k
    P0, D = I.track_lines(df, S['tr'], mask)
    r, y, xz = I.axis_approach(P0, D)
    return r, y, xz


IMG_HALF = 45.0          # the note's fig2 window


def _panel(ax, xz, y, title, r=None):
    """Top-down view: the plane transverse to the beam axis (global X, Z).

    NOT (transverse, along-beam) -- the capsule is a 10 mm bore seen end-on,
    and plotting one transverse coordinate against the beam axis smears an
    80 mm-long source across the frame and hides the focus entirely."""
    b = np.linspace(-IMG_HALF, IMG_HALF, 91)
    ax.hist2d(xz[:, 0], xz[:, 1], bins=[b, b], cmap='viridis')
    ax.add_patch(plt.Circle((0, 0), CAPSULE_R, fill=False, color='orangered',
                            ls='--', lw=1.4))
    ax.set_xlim(-IMG_HALF, IMG_HALF); ax.set_ylim(-IMG_HALF, IMG_HALF)
    ax.set_xlabel('global X [mm]'); ax.set_ylabel('global Z [mm]')
    ax.set_aspect('equal')
    if r is not None:
        f = float(np.mean(r < CAPSULE_R))
        title += f'   {100 * f:.1f} % inside r < 10 mm'
    ax.set_title(title, fontsize=9)


# -------------------------------------------------------------------- figures
def fig1(S, out, mask=None, ax=None, title=None):
    """tan theta (x) vs u, the point-source band."""
    m = S['sel'] if mask is None else mask
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hexbin(S['lever'][m], S['tan'][m], gridsize=80, cmap='viridis',
              mincnt=1, extent=(-200, 200, -1, 1))
    uu = np.linspace(-200, 200, 10)
    ax.plot(uu, uu / S['L'], 'w--', lw=1.2,
            label=f'tan = u/L, L = {S["L"]:.0f} mm')
    for lo, hi in ((-200, -U_CAL_MAX), (U_CAL_MAX, 200)):
        ax.axvspan(lo, hi, color='k', alpha=0.18, lw=0)
    ax.set_xlim(-200, 200); ax.set_ylim(-1, 1)
    ax.set_xlabel('u on the strip plane [mm], from the beam-axis perpendicular')
    ax.set_ylabel(r'tan $\theta$ (x)')
    ax.set_title(title or f'arm {S["arm"]}: {int(m.sum()):,} two-plane '
                          'wall-matched tracks', fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    if own:
        fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig2(S, out):
    """Back-projection at the bundle scale and at the in-situ scale."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (k, lab) in zip(axes, ((1.0, 'bundle prior'),
                                   (S['k_track'], 'in-situ'))):
        r, y, xz = backproject(S, S['sel'], k)
        _panel(ax, xz, y,
               f'{lab}  v = {S["v_bundle"] / k:.1f} um/ns  (k = {k:.2f})', r)
    fig.suptitle(f'Arm {S["arm"]} back-projection: closest approach to the '
                 'beam axis (top-down)', fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig3(S, out):
    """Per-track angle scale for inclined tracks."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(S['k_i'], bins=np.linspace(0, 4, 81), color='#3b6ea5')
    ax.axvline(S['k_track'], color='k', lw=1.5,
               label=f'median k = {S["k_track"]:.3f}  ->  '
                     f'v = {S["v_bundle"] / S["k_track"]:.1f} um/ns')
    ax.axvline(S['k_phys'], color='crimson', lw=1.2, ls='--',
               label=f'coincident subset k = {S["k_phys"]:.3f}')
    ax.set_xlabel('per-track k = (u/L) / tan'); ax.set_ylabel('tracks')
    ax.set_title(f'arm {S["arm"]}: {S["n_inc"]:,} inclined tracks '
                 f'(|tan| > {TAN_FLOOR}, {U_CAL_MIN:.0f} < |u| < '
                 f'{U_CAL_MAX:.0f} mm)', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig4(S, out):
    """Profile along the beam axis of the in-situ image."""
    r, y, _ = backproject(S, S['sel'], S['k_phys'])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(y[r < 30], bins=np.linspace(-200, 200, 81), color='#3b6ea5')
    ax.axvspan(-40, 40, color='orange', alpha=0.25, lw=0,
               label='He-3 capsule extent (~80 mm)')
    ax.set_xlabel('position along the beam axis [mm]')
    ax.set_ylabel('tracks with axis miss < 30 mm')
    ax.set_title(f'arm {S["arm"]}: axial profile at the in-situ scale',
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig5(S, out):
    """The image before and after the pointing-coincidence requirement."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (m, lab) in zip(axes, ((S['sel'], 'wall-matched'),
                                   (S['sel'] & S['coin'],
                                    'pointing-coincident'))):
        r, y, xz = backproject(S, m, S['k_phys'])
        _panel(ax, xz, y, f'{lab}  n = {int(m.sum()):,}', r)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig6(S, out):
    """u extrapolated to the wall, split by which SiPM pair actually fired."""
    import uproot
    from collections import defaultdict
    arm = S['arm']
    t = uproot.open(S['slim'])['hits']
    a = t.arrays(['eventId', 'det', 'detn', 'dt_ns', 'is_control'],
                 library='np')
    it = ((a['is_control'] == 0) & (a['dt_ns'] >= I.DT_WINDOW[0])
          & (a['dt_ns'] <= I.DT_WINDOW[1]) & (a['det'] == I.WAL_CODE[arm]))
    fired = defaultdict(set)
    for eid, dn in zip(a['eventId'][it], a['detn'][it]):
        fired[int(eid)].add((int(dn) - 1) // 2)

    # same convention as pointing_coincidence (corrected frame 2026-08-20):
    # outward extrapolation ADDS tan*d, and the pinwheel comes off in the
    # corrected local x
    u_wall = (S['u_unflipped'] + I.STRIPS_TO_WALL * S['tan_unflipped']
              - S['foot_x'])
    eids = S['df']['event_id'].to_numpy()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rows, colors = [], ['#d95f02', '#7570b3', '#1b9e77', '#e7298a']
    for pair in range(I.N_WALL_SEG):
        m = np.array([S['sel'][i] and pair in fired.get(int(eids[i]), ())
                      for i in range(len(eids))])
        if m.sum() < 20:
            continue
        lo, hi = I._wall_seg_u(pair)          # detn pair order is ascending
        vals = u_wall[m]
        inside = float(np.mean((vals >= lo) & (vals < hi)))
        rows.append(dict(pair=pair, n=int(m.sum()),
                         median=float(np.median(vals)), inside=inside))
        ax.hist(vals, bins=np.linspace(-250, 250, 101), histtype='step',
                lw=1.4, color=colors[pair],
                label=f'pair {pair}: n={int(m.sum()):,}, '
                      f'med={np.median(vals):+.0f} mm, {100 * inside:.0f} % in')
        ax.axvspan(lo, hi, color=colors[pair], alpha=0.10, lw=0)
    ax.set_xlabel('track u extrapolated to the wall plane [mm]')
    ax.set_ylabel('tracks')
    ax.set_title(f'arm {arm}: external position truth from the SiPM wall',
                 fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)
    return rows


def fig7(S, out):
    """The pointing correlation with and without the same-arm wall hit."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig1(S, None, mask=S['base'], ax=axes[0],
         title=f'all reconstructed: {int(S["base"].sum()):,}')
    fig1(S, None, mask=S['sel'], ax=axes[1],
         title=f'same-arm SiPM wall hit: {int(S["sel"].sum()):,}')
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig8(S, out):
    """The back-projection under those same two selections."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (m, lab) in zip(axes, ((S['base'], 'all reconstructed'),
                                   (S['sel'], 'same-arm wall hit'))):
        r, y, xz = backproject(S, m, S['k_phys'])
        _panel(ax, xz, y, f'{lab}  n = {int(m.sum()):,}', r)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--arm', default='A')
    ap.add_argument('--slim', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(I.ANALYSIS_BASE, a.run, a.subrun, 'imaging',
                                'note_figs_fullcov')
    os.makedirs(out, exist_ok=True)
    trs, _ = I.transforms(a.run)
    S = prepare(a.run, a.subrun, a.arm, a.slim, trs)
    S['slim'] = a.slim

    fig1(S, os.path.join(out, 'fig1_tan_vs_u.png'))
    fig2(S, os.path.join(out, 'fig2_image.png'))
    fig3(S, os.path.join(out, 'fig3_kdist.png'))
    fig4(S, os.path.join(out, 'fig4_yprofile.png'))
    fig5(S, os.path.join(out, 'fig5_coincidence.png'))
    rows = fig6(S, os.path.join(out, 'fig6_wall_pointing.png'))
    fig7(S, os.path.join(out, 'fig7_cmp_tan.png'))
    fig8(S, os.path.join(out, 'fig8_cmp_image.png'))

    with open(os.path.join(out, 'note_figs_coin.json'), 'w') as f:
        json.dump(dict(arm=a.arm, run=a.run, subrun=a.subrun,
                       pointing_coincidence=S['cinfo'], wall_rows=rows,
                       k_track=S['k_track'], k_phys=S['k_phys'],
                       drawn_by='ntof_tracking/run145_note_figs.py'),
                  f, indent=1)
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
