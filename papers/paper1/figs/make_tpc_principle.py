#!/usr/bin/env python3
"""make_tpc_principle.py -- Paper I, the TPC principle figure (fig.tpc_principle), 2026-11-03 rework.

    .venv/bin/python papers/paper1/figs/make_tpc_principle.py [--cache <dir with principle_7884.pkl>] [--out ...]

Dylan 11-03 on the 11-02 version (microtpc_principle): too narrow and tall, far too much text; "total rework".
Now one full-width row, three panels sharing one vertical axis (depth above the mesh = drift time):

(a) the chamber in section (x-z, to scale in z, the amplification stage enlarged): a muon crosses the drift gap, its
    ionisation drifts straight down to the mesh, the avalanche lands on the resistive strips and couples to the
    readout strips below; the time of arrival measures the depth, t = t0 + z / v.
(b) the X view of a real chamber A muon (Run 1 event 7884, pass 1): ADC per strip and 60 ns sample, with the
    reference track (dashed) and the fitted track;
(c) the forward model at that fit, on the same colour scale.

The event and its model come from make_principle.py's replay cache (principle_<event>.pkl: the event replayed through
wft.reco exactly as the pass ran it; the replay's w equals the pass table's).  Nothing here uses hit times.
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', '..')]
import evd  # noqa: E402
import paper_style as ps  # noqa: E402

CACHE = '/media/ucla/work/claude-jobs/17c92c46-displays/principle'


def sketch(ax, g, v_mm, t0, tlim):
    """The section, in mm (x) against z (mm above the mesh, mapped onto the time axis of the data panels)."""
    import matplotlib.patches as mp
    z = lambda zz: t0 + zz / v_mm            # depth -> the shared (time) coordinate
    X0, X1 = 0.0, 30.0
    ink, mut = ps.INK, ps.MUTED
    # drift volume and cathode
    ax.add_patch(mp.Rectangle((X0, z(0)), X1 - X0, z(g) - z(0), fc='#f3f6fb', ec='none', zorder=0))
    ax.plot([X0, X1], [z(g)] * 2, color=ink, lw=1.6, zorder=3)
    ax.text(X1 + 0.6, z(g), 'cathode', va='center', ha='left', fontsize=7.5, color=ink)
    # mesh (dotted line of wires)
    ax.plot([X0, X1], [z(0)] * 2, color=ink, lw=0.9, ls=(0, (1, 1.2)), zorder=3)
    ax.text(X1 + 0.6, z(0), 'mesh', va='center', ha='left', fontsize=7.5, color=ink)
    # amplification gap, resistive strips and readout below the mesh (enlarged, schematic)
    # below the mesh, enlarged: amplification gap (avalanche), resistive strips, insulator, readout strips
    for i in range(14):
        xc = X0 + 1.2 + i * 2.1
        ax.add_patch(mp.Rectangle((xc - 0.8, z(-3.0)), 1.6, z(-2.2) - z(-3.0), fc='#7f8a99', ec='none', zorder=2))
        ax.add_patch(mp.Rectangle((xc - 0.8, z(-5.6)), 1.6, z(-4.6) - z(-5.6), fc='#c98b2b', ec='none', zorder=2))
    ax.text(X1 + 0.6, z(-2.6), 'resistive strips', va='center', ha='left', fontsize=7.5, color=ink)
    ax.text(X1 + 0.6, z(-5.1), 'readout strips', va='center', ha='left', fontsize=7.5, color=ink)
    ax.text(X0 - 0.6, z(-1.2), 'avalanche', va='center', ha='right', fontsize=6.5, color=mut)
    # the muon: enters at the cathode, leaves at the mesh, tilted like event 7884 in X (tan 0.17)
    tan = 0.17
    xm = 12.0
    xs = lambda zz: xm + tan * zz
    ax.annotate('', xy=(xs(-1.8), z(-1.8)), xytext=(xs(g + 5), z(g + 5)),
                arrowprops=dict(arrowstyle='-|>', color=ps.ACCENT, lw=1.3, mutation_scale=9), zorder=5)
    ax.text(xs(g + 4.2) + 0.9, z(g + 4.2), 'muon', color=ps.ACCENT, fontsize=7.5, va='center', ha='left')
    # ionisation clusters and their drift down to the mesh
    rng = np.random.default_rng(3)
    for zz in np.linspace(2.5, g - 2.5, 7):
        x = xs(zz) + rng.normal(0, 0.15)
        ax.plot(x, z(zz), 'o', ms=2.4, color=ps.DET_COLOR['det3'], zorder=6)
        ax.annotate('', xy=(x, z(0.6)), xytext=(x, z(zz - 0.6)),
                    arrowprops=dict(arrowstyle='-|>', color=ps.DET_COLOR['det3'], lw=0.6, mutation_scale=5,
                                    alpha=0.7), zorder=4)
    # drift field and gap dimension
    ax.annotate('', xy=(X0 - 1.6, z(0)), xytext=(X0 - 1.6, z(g)),
                arrowprops=dict(arrowstyle='<->', color=mut, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(X0 - 2.3, z(g / 2), f'$g$ = {g:.1f} mm', rotation=90, va='center', ha='right', fontsize=7.5, color=mut)
    ax.text(xs(g * 0.62) + 3.0, z(g * 0.62), r'$t = t_0 + z/v$', fontsize=8, color=ink, ha='left', va='center',
            bbox=dict(fc='white', ec='none', pad=1.0))
    ax.set_xlim(X0 - 5.5, X1 + 13.5)
    ax.set_ylim(*tlim)
    ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=CACHE)
    ap.add_argument('--event', type=int, default=7884)
    ap.add_argument('--view', default='x')
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'tpc_principle'))
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ps.use()
    r = pickle.load(open(os.path.join(a.cache, f'principle_{a.event}.pkl'), 'rb'))
    v_mm = float(r['v_drift']) * 1e-3
    d = r['views'][a.view]
    g = evd.GAP
    t0 = float(d['t0'])
    tlim = (t0 - 7.0 / v_mm, t0 + (g + 7.0) / v_mm)
    fig = plt.figure(figsize=(ps.FULL[0], 2.75))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.45, 2.0], left=0.0, right=0.885, bottom=0.16, top=0.9,
                             wspace=0.2)
    gs = outer[0, 1].subgridspec(1, 2, wspace=0.06)
    axs_ = fig.add_subplot(outer[0, 0])
    sketch(axs_, g, v_mm, t0, tlim)
    m3 = r.get('m3') or {}
    cen = float(d['p0'] + d['w'] * 0.5 * g / v_mm)
    half = 9.0
    W, M, P = np.asarray(d['W'], float), np.asarray(d['M'], float), np.asarray(d['pos'], float)
    vmax = max(float(np.percentile(W, 99.5)), 60.0)
    axd = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1], sharey=axd)
    for ax, A, title in ((axd, W, '(b) data'), (axm, M, '(c) model at the fit')):
        pc = evd.cells(ax, A, P, cen, half, vmax)
        evd.gap_lines(ax, t0, v_mm, label=False)
        ax.set_xlim(cen - half, cen + half)
        ax.set_ylim(*tlim)
        ax.set_xlabel('$x$ [mm]')
        ax.set_title(title, loc='left', fontsize=8, pad=3)
        ax.tick_params(direction='in', top=True, right=True, length=2.5)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color('#555555')
        evd.fit(ax, dict(view='x', p0=d['p0'], w=d['w'], t0=t0, role='first'), v_mm)
    if m3:
        evd.reference(axd, float(m3[f'mesh_{a.view}']), float(m3[f'tan_{a.view}']), t0, v_mm)
    axd.set_ylabel('drift time [ns]')
    axm.tick_params(labelleft=False)
    sec = axm.secondary_yaxis('right', functions=(lambda t: (t - t0) * v_mm, lambda zz: zz / v_mm + t0))
    sec.set_ylabel('$z$ above the mesh [mm]')
    sec.tick_params(direction='in', length=2.5)
    axs_.set_title('(a)', loc='left', fontsize=8, pad=3)
    cax = fig.add_axes([0.965, 0.16, 0.012, 0.74])
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label('amplitude [ADC]', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    h = {}
    for ax in (axd, axm):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            h.setdefault(ll, hh)
    axd.legend(list(h.values()), list(h.keys()), loc='upper left', fontsize=6.5, frameon=True, framealpha=0.9,
               edgecolor='none', handlelength=2.2, borderpad=0.3)
    ps.save(fig, a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
