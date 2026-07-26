#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary plot: reconstructed TRACK RATE vs time since the gamma flash, in fine
2 ms bins, for several resist HV settings, superimposed on the Geant4 in-gate
IPC production spectrum.

The question it answers: **when the IPC signal actually arrives, are we
tracking?** The IPC curve (blue fill, left axis) is where the physics is; the
step histograms (right axis) are what each resist setting delivers. Overlaying
them makes the gain trade-off concrete — raising resist raises gain but
lengthens post-flash saturation, so the high-resist curves are suppressed
exactly where the IPC thermal peak sits.

NOTE ON RESIST 560 V: it was NOT scanned in run_67. The ladder is 520-550 V in
5 V steps. The default set is 550/540/530/520, which keeps the operator's
requested 10 V spacing and extends to the low-gain end instead of the
(unmeasured) high end.

Normalisation: tracks per pulse per ms, i.e. (events with a 3D x/y pair in the
bin) / (spills in the cell) / (bin width). Directly comparable in shape to the
IPC axis, which is IPC pairs / pulse / ms.

The three plastic thresholds are POOLED by default (~90-110 tracks per 2 ms bin
vs ~23-43 for one threshold). Since the normalisation divides by the pooled
spill count, the result is the average over the three threshold settings; use
--mip to restrict to one at the cost of ~2x noisier bins.

Run: .venv/bin/python ntof_july_analysis/run67_scan/tracks_vs_ipc.py
     [--drift 600] [--resists 550 540 530 520] [--bw 2] [--mip 90]
Outputs -> <OUT_BASE>/summary/
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import scan_lib as L  # noqa: E402
import flash_timing as FT  # noqa: E402  (IPC loading + drawing conventions)

OUT = os.path.join(L.OUT_BASE, 'summary')
DEFAULT_RESISTS = [550, 540, 530, 520]

# high resist = hot colour, low resist = cool; distinct from the IPC blue fill
RCOL = {550: '#d7191c', 545: '#e8613c', 540: '#fdae61', 535: '#c8a415',
        530: '#4d9221', 525: '#3182bd', 520: '#762a83'}


def track_rate(ev, det, resist, drift, bw, mip=None):
    """(bin_left, tracks/pulse/ms, err) for one (drift, resist) cell."""
    g = ev[(ev.drift == drift) & (ev.resist == resist)]
    if mip is not None:
        g = g[g.mip == mip]
    if g.empty:
        return None
    n_spill = g.drop_duplicates(['subrun', 'burst']).shape[0]
    hit = g[g[f'n_pair_{det}'] > 0]
    # Bins start at the gate opening, not at 0: a bin straddling t=1 ms is part
    # dead time, so dividing its counts by the full bin width misstates the rate
    # in the single most important bin of the plot (the IPC peak is at 5.3 ms).
    edges = np.arange(L.READOUT_START_MS, FT.TMAX + bw, bw)
    k, _ = np.histogram(hit.dt_ms.to_numpy(), bins=edges)
    # Poisson counting error on the bin, propagated through the same scaling.
    scale = 1.0 / max(n_spill, 1) / bw
    return edges[:-1], k * scale, np.sqrt(k) * scale, n_spill, len(g)


def fig_one_det(ev, ipc, det, drift, resists, bw, mip, fname):
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    FT._draw_ipc(ax, ipc)
    FT._shade_gate(ax)
    ax2 = ax.twinx()
    top = 0.0
    for r in resists:
        out = track_rate(ev, det, r, drift, bw, mip)
        if out is None:
            print(f'    resist {r} V not present at drift {drift} — skipped')
            continue
        x, y, e, ns, nev = out
        ax2.step(x, y, where='post', color=RCOL.get(r, 'k'), lw=1.9,
                 label=f'resist {r} V   ({nev} triggers, {ns} spills)',
                 zorder=5)
        ax2.errorbar(x + bw / 2, y, yerr=e, fmt='none', ecolor=RCOL.get(r, 'k'),
                     alpha=0.45, lw=0.9, zorder=5)
        top = max(top, float((y + e).max()))
    ax2.set_ylim(0, top * 1.25)
    ax2.set_ylabel(f'reconstructed tracks / pulse / ms   (Det {det})',
                   fontsize=11)
    # one legend for both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h2 + h1, l2 + l1, fontsize=8.5, loc='upper right', framealpha=0.93)
    thr = 'all three plastic thresholds pooled' if mip is None \
        else f'{L.MIP_LABEL[mip]} only'
    ax.set_title(
        f'run_67 — Det {det} track rate vs the in-gate IPC spectrum\n'
        f'drift {drift} V, {bw:g} ms bins, {thr}',
        fontsize=13)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print('  wrote', p)
    return p


def fig_all_dets(ev, ipc, drift, resists, bw, mip, fname):
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    for ax, det in zip(axes.ravel(), 'ABCD'):
        FT._draw_ipc(ax, ipc)
        ax.set_title(f'Det {det}' + ('  (clean M1 — reference)' if det == 'A'
                                     else ''), fontsize=11)
        ax2 = ax.twinx()
        top = 0.0
        for r in resists:
            out = track_rate(ev, det, r, drift, bw, mip)
            if out is None:
                continue
            x, y, e, ns, nev = out
            ax2.step(x, y, where='post', color=RCOL.get(r, 'k'), lw=1.6,
                     label=f'{r} V', zorder=5)
            top = max(top, float(y.max()))
        ax2.set_ylim(0, max(top, 1e-9) * 1.25)
        ax2.set_ylabel('tracks / pulse / ms', fontsize=9)
        if det == 'A':
            ax2.legend(fontsize=8, title='resist', title_fontsize=8,
                       loc='upper right')
    thr = 'all thresholds pooled' if mip is None else L.MIP_LABEL[mip]
    fig.suptitle(f'run_67 — track rate vs the in-gate IPC spectrum, all four '
                 f'chambers (drift {drift} V, {bw:g} ms bins, {thr})',
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print('  wrote', p)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', type=int, default=600)
    ap.add_argument('--resists', type=int, nargs='+', default=DEFAULT_RESISTS)
    ap.add_argument('--bw', type=float, default=2.0, help='bin width [ms]')
    ap.add_argument('--mip', type=int, default=None, choices=[141, 113, 90])
    ap.add_argument('--det', default='A')
    args = ap.parse_args(argv)

    ev, _, _ = L.load_all()
    ev = ev[ev.flash_ok & ~ev.is_leader].copy()
    have = sorted(ev[ev.drift == args.drift].resist.unique())
    missing = [r for r in args.resists if r not in have]
    if missing:
        print(f'  NOT SCANNED at drift {args.drift} V: {missing} '
              f'(available: {have}) — those curves are omitted')
    ipc = FT.load_ipc()
    print(f'  IPC in-gate: {ipc["ipc_pulse"]:.3g} pairs/pulse')

    tag = 'allthr' if args.mip is None else f'm{args.mip}'
    fig_one_det(ev, ipc, args.det, args.drift, args.resists, args.bw, args.mip,
                f'tracks_vs_ipc_det{args.det}_dr{args.drift}_{args.bw:g}ms_{tag}.png')
    fig_all_dets(ev, ipc, args.drift, args.resists, args.bw, args.mip,
                 f'tracks_vs_ipc_alldets_dr{args.drift}_{args.bw:g}ms_{tag}.png')
    print('done ->', OUT)


if __name__ == '__main__':
    main()
