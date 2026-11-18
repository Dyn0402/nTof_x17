"""Panel 4 figure: one cosmic muon as waveforms, strip by strip, with the forward model at the fit on top.

Event 7884 of chamber A (Run 1), the muon of Paper I's data-against-model figure; the samples and the model
come from papers/paper1_v2/display_lab/events.json (built by display_lab/extract_events.py).

    ../.venv/bin/python make_waterfall.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'papers', 'paper1_v2', 'display_lab', 'events.json')
TS = 60.0  # ns per sample


def panel(ax, m, gap, v, title, cmap):
    P, W, M = np.array(m['P']), np.array(m['W']), np.array(m['M'])
    t = np.arange(W.shape[1]) * TS
    keep = t <= 1500                                   # mplot3d does not clip to the axis limits
    t, W, M = t[keep], W[:, keep], M[:, keep]
    top = max(W.max(), M.max())
    ax.computed_zorder = False                         # paint in the order given: far strips first
    for z, i in enumerate(np.argsort(P)[::-1]):
        col = cmap(0.25 + 0.7 * W[i].max() / top)
        verts = [(t[0], 0)] + list(zip(t, np.clip(W[i], 0, None))) + [(t[-1], 0)]
        poly = Poly3DCollection([[(a, P[i], b) for a, b in verts]], facecolor=col, edgecolor='none', alpha=0.85,
                                zorder=2 * z + 1)
        ax.add_collection3d(poly)
        mm = np.where(M[i] > 0.015 * top, M[i], np.nan)   # the model only where it has signal
        ax.plot(t, np.full_like(t, P[i]), mm, color='#111827', lw=1.8, zorder=2 * z + 2)
    # the fitted track on the floor: position against arrival time across the gap
    tt = np.array([m['t0'], m['t0'] + gap / v])
    ax.plot(tt, m['p0'] + m['w'] * (tt - m['t0']), [0, 0], color='#d9631e', lw=4, solid_capstyle='round', zorder=999)
    ax.set_xlim(t[0], t[-1]), ax.set_ylim(P.min() - 0.4, P.max() + 0.4), ax.set_zlim(0, top * 1.02)
    ax.set_xlabel('time [ns]', labelpad=12), ax.set_ylabel('strip position [mm]', labelpad=14)
    ax.text2D(0.93, 0.80, 'amplitude [ADC]', transform=ax.transAxes, ha='right', fontsize=15)
    ax.set_title(title, fontsize=21, fontweight='bold', y=0.98)
    ax.view_init(elev=28, azim=-62)
    ax.set_box_aspect((1.4, 1.25, 0.95), zoom=0.98)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor('#f5f7fa'), a.pane.set_edgecolor('#d5dbe3')
    ax.grid(False)


def main():
    d = json.load(open(SRC))
    ev = d['events'][0]
    assert ev['id'] == 7884
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(14, 7.0))
    for k, (view, cmap) in enumerate((('x', plt.cm.Blues), ('y', plt.cm.Greens))):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        panel(ax, d['model7884'][view], ev['gap'], ev['v'], view.upper() + ' view', cmap)
    fig.subplots_adjust(left=0.0, right=0.95, bottom=0.04, top=0.98, wspace=0.06)
    out = os.path.join(HERE, 'assets', 'waterfall_7884.png')
    fig.savefig(out, dpi=260)
    print('wrote', out)


if __name__ == '__main__':
    main()
