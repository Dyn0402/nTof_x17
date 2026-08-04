#!/usr/bin/env python3
"""
08_connector_test.py — could det4's stripes just be dead X-plane channels?

The stripes live in detector-local X, which the X plane (FEU 6) measures, so the
obvious suspicion is readout: dead strips, a bad connector, a flaky Panasonic
cable. This script puts that hypothesis through four tests it has to survive.

  T1  Do the "dead" strips fire at all?  A dead channel yields *nothing*; a
      low-gain region of chamber yields hits that are simply small.
  T2  Do the band edges respect the 64-channel connector boundaries (49.8 mm)?
      A connector fault cannot produce a band that straddles one.
  T3  Does the *other* plane lose charge at the same X?  A deaf X plane leaves
      the Y plane's charge untouched.
  T4  Does the pattern survive on the Y plane's own coordinate?  If the Y plane
      is healthy everywhere in its own coordinate but not as a function of X,
      the loss is upstream of both readouts.

Also draws the 2-D efficiency map at a binning fine enough to resolve the
stripes (4 mm in X, 25 mm in Y).

    ../../.venv/bin/python sps_beam_test_26/det4_sps_assessment/08_connector_test.py g_det4
"""
import argparse
import json
import os
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
from matplotlib.colors import LogNorm                        # noqa: E402
import uproot                                                # noqa: E402
import cosmic_micro_tpc_analysis as cm                       # noqa: E402
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS          # noqa: E402
from common.Mx17StripMap import Mx17StripMap                 # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE              # noqa: E402

sys.path.insert(0, HERE)
from importlib import import_module                          # noqa: E402
_uni = import_module('01_uniformity')

PITCH = 0.78
CONN_MM = 64 * PITCH        # 49.92 mm per 64-channel connector


def per_strip(cfg):
    """Per-strip occupancy and amplitude on both planes, discharge events removed."""
    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel', 'amplitude',
                                          'significance'], library='pd')
    n_ev = int(raw['eventId'].nunique())
    det = cm.apply_significance_floor(raw[raw['feu'].isin(cfg.MX17_FEUS)],
                                      rel=SIG_REL_FLOOR)
    mult = det.groupby('eventId').size()
    det = det[~det['eventId'].isin(set(mult[mult > SPARK_VETO_HITS].index))]
    # the significance floor is a *relative* cut inside each event; for "did this
    # channel ever produce a hit" we want the unfiltered list as well
    rawdet = raw[raw['feu'].isin(cfg.MX17_FEUS)]
    out = {}
    for axis, feu in (('x', cfg.MX17_FEUS[0]), ('y', cfg.MX17_FEUS[1])):
        occ = np.zeros(512)
        occ_raw = np.zeros(512)
        amp = np.full(512, np.nan)
        sub = det[det.feu == feu]
        for ch, s in sub.groupby('channel'):
            c = int(ch)
            if 0 <= c < 512:
                occ[c] = len(s) / n_ev
                amp[c] = float(np.median(s['amplitude']))
        for ch, s in rawdet[rawdet.feu == feu].groupby('channel'):
            c = int(ch)
            if 0 <= c < 512:
                occ_raw[c] = len(s) / n_ev
        out[axis] = dict(feu=feu, occ=occ, occ_raw=occ_raw, amp=amp)
    return out, n_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key', nargs='?', default='g_det4')
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()
    cfg = get_config(args.key)

    sm = Mx17StripMap(cfg.MAP_CSV_PATH)
    posx = np.array([sm.lookup('x', *Mx17StripMap.feu_channel_to_connector(c))[0]
                     for c in range(512)])
    posy = np.array([sm.lookup('y', *Mx17StripMap.feu_channel_to_connector(c))[1]
                     for c in range(512)])

    sp, n_ev = per_strip(cfg)
    rec, _ = _uni.categorise(args.key)
    bands = json.load(open(os.path.join(args.out,
                                        f'stripes_{args.key}.json')))['bands_mm']

    ax0, ax1 = TRUE_ACTIVE['x']
    ay0, ay1 = TRUE_ACTIVE['y']
    inside = ((rec['x'] >= ax0) & (rec['x'] <= ax1)
              & (rec['y'] >= ay0) & (rec['y'] <= ay1))
    x, y, c = rec['x'][inside], rec['y'][inside], rec['cat'][inside]
    xe = np.arange(ax0, ax1 + 4, 4.0)
    ye = np.arange(ay0, ay1 + 25, 25.0)
    tot, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    near, _, _ = np.histogram2d(x[c == 4], y[c == 4], bins=[xe, ye])
    with np.errstate(invalid='ignore'):
        effmap = np.where(tot >= 8, near / tot, np.nan)

    inb = np.zeros(512, bool)
    for a, b in bands:
        inb |= (posx >= a) & (posx <= b)
    act = (posx >= ax0) & (posx <= ax1)

    # ---- T1: do dead-band strips fire? -------------------------------------
    def stats(sel):
        o = sp['x']['occ'][sel]
        return dict(n_strips=int(sel.sum()),
                    n_zero_occ=int((sp['x']['occ_raw'][sel] == 0).sum()),
                    median_occ=float(np.median(o)),
                    median_amp=float(np.nanmedian(sp['x']['amp'][sel])))
    t1 = dict(live=stats(act & inb), dead=stats(act & ~inb))
    t1['dead_strips_total_Xplane'] = int((sp['x']['occ_raw'] == 0).sum())
    t1['dead_strips_total_Yplane'] = int((sp['y']['occ_raw'] == 0).sum())
    t1['n_events'] = n_ev

    # ---- T2: do band edges respect connector boundaries? -------------------
    edges = np.array([e for b in bands for e in b])
    edges = edges[(edges > ax0 + 2) & (edges < ax1 - 2)]     # ignore chamber ends
    d = np.abs(((edges + CONN_MM / 2) % CONN_MM) - CONN_MM / 2)
    t2 = dict(n_edges=int(len(edges)),
              mean_dist_to_connector_boundary_mm=float(d.mean()),
              expected_if_random_mm=float(CONN_MM / 4),
              min_dist_mm=float(d.min()),
              n_edges_within_3mm=int((d < 3).sum()),
              n_bands_straddling_a_boundary=int(sum(
                  1 for a, b in bands if (a // CONN_MM) != (b // CONN_MM))))

    # ---- T3/T4: per-connector level, and the Y plane's own coordinate ------
    conn = dict()
    for k in range(8):
        s = slice(k * 64, (k + 1) * 64)
        conn[k + 1] = dict(
            x_mm=[float(posx[s].min()), float(posx[s].max())],
            x_occ=float(sp['x']['occ'][s].sum()),
            x_amp=float(np.nanmedian(sp['x']['amp'][s])),
            y_occ=float(sp['y']['occ'][s].sum()),
            y_amp=float(np.nanmedian(sp['y']['amp'][s])),
            frac_strips_in_live_band=float(inb[s].mean()))
    xo = np.array([conn[k]['x_occ'] for k in conn])
    yo = np.array([conn[k]['y_occ'] for k in conn])
    frac = np.array([conn[k]['frac_strips_in_live_band'] for k in conn])
    t3 = dict(per_connector=conn,
              corr_Xconn_occ_vs_live_fraction=float(np.corrcoef(xo, frac)[0, 1]),
              Yplane_connector_occ_spread=float(yo.max() / max(yo.min(), 1e-9)),
              Xplane_connector_occ_spread=float(xo.max() / max(xo.min(), 1e-9)))

    rep = dict(run_key=args.key, T1_do_dead_strips_fire=t1,
               T2_connector_boundaries=t2, T3_per_connector=t3)
    with open(os.path.join(args.out, f'connector_test_{args.key}.json'), 'w') as f:
        json.dump(rep, f, indent=1)

    # ---------------------------- figure ------------------------------------
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 1.1, 1.1, 1.1], hspace=.45)

    a = fig.add_subplot(gs[0])
    im = a.imshow(effmap.T, origin='lower', aspect='auto', vmin=0, vmax=1,
                  extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap='viridis',
                  interpolation='nearest')
    for k in range(1, 8):
        a.axvline(k * CONN_MM, color='w', ls='--', lw=1.2, alpha=.9)
    a.set_ylabel('detector-local Y [mm]')
    a.set_title(f'{args.key} — efficiency within 5 mm, 4 mm x 25 mm cells '
                f'(dashed = FEU {sp["x"]["feu"]} connector boundaries)')
    fig.colorbar(im, ax=a, fraction=.025, pad=.01, label='efficiency')

    a = fig.add_subplot(gs[1])
    o = np.argsort(posx)
    a.semilogy(posx[o], np.clip(sp['x']['occ_raw'][o], 1e-6, None), lw=.8,
               color='#0072b2', label=f'X plane (FEU {sp["x"]["feu"]}) occupancy')
    for aa, bb in bands:
        a.axvspan(aa, bb, color='#009e73', alpha=.15)
    for k in range(1, 8):
        a.axvline(k * CONN_MM, color='k', ls='--', lw=1.0)
    a.set_ylabel('hits / event / strip')
    a.legend(fontsize=8, loc='lower right')
    a.set_title('T1/T2: every strip fires; green bands straddle the dashed '
                'connector boundaries')
    a.grid(alpha=.3, which='both')

    a = fig.add_subplot(gs[2])
    a.plot(posx[o], sp['x']['amp'][o], lw=.8, color='#0072b2', label='X plane')
    for aa, bb in bands:
        a.axvspan(aa, bb, color='#009e73', alpha=.15)
    for k in range(1, 8):
        a.axvline(k * CONN_MM, color='k', ls='--', lw=1.0)
    a.set_ylabel('median hit amplitude [ADC]')
    a.legend(fontsize=8)
    a.set_title('the strips between the bands are not dead — they are quiet')
    a.grid(alpha=.3)

    a = fig.add_subplot(gs[3])
    oy = np.argsort(posy)
    a.semilogy(posy[oy], np.clip(sp['y']['occ_raw'][oy], 1e-6, None), lw=.8,
               color='#d55e00', label=f'Y plane (FEU {sp["y"]["feu"]}) occupancy '
                                      'vs its OWN coordinate')
    a.set_xlabel('detector-local coordinate [mm]')
    a.set_ylabel('hits / event / strip')
    a.legend(fontsize=8, loc='lower right')
    a.set_title('T4: the Y plane is smooth in Y — the stripes exist only in X, '
                'for both planes')
    a.grid(alpha=.3, which='both')
    fig.savefig(os.path.join(args.out, f'connector_test_{args.key}.png'),
                dpi=115, bbox_inches='tight')

    # ------------------- standalone efficiency map --------------------------
    xe2 = np.arange(ax0, ax1 + 6, 6.0)
    ye2 = np.arange(ay0, ay1 + 45, 45.0)
    t2d, _, _ = np.histogram2d(x, y, bins=[xe2, ye2])
    n2d, _, _ = np.histogram2d(x[c == 4], y[c == 4], bins=[xe2, ye2])
    with np.errstate(invalid='ignore'):
        m2d = np.where(t2d >= 10, n2d / t2d, np.nan)
    fig2, ax2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1], hspace=.12))
    im = ax2[0].imshow(m2d.T, origin='lower', aspect='auto', vmin=0, vmax=1,
                       extent=[xe2[0], xe2[-1], ye2[0], ye2[-1]], cmap='viridis',
                       interpolation='nearest')
    for k in range(1, 8):
        ax2[0].axvline(k * CONN_MM, color='w', ls='--', lw=1.3)
    ax2[0].set_ylabel('detector-local Y [mm]')
    ax2[0].set_title(f'{args.key} ({cfg.DET_NAME}) — efficiency within 5 mm of the '
                     f'M3 reference, 6 x 45 mm cells\n'
                     f'dashed = FEU {sp["x"]["feu"]} connector boundaries '
                     f'(64 channels = 49.9 mm)')
    fig2.colorbar(im, ax=ax2[0], fraction=.02, pad=.01, label='efficiency')
    xc = 0.5 * (xe2[:-1] + xe2[1:])
    with np.errstate(invalid='ignore'):
        projx = np.where(t2d.sum(1) >= 30, n2d.sum(1) / t2d.sum(1), np.nan)
    ax2[1].plot(xc, projx, 'k-', lw=1.3)
    for k in range(1, 8):
        ax2[1].axvline(k * CONN_MM, color='k', ls='--', lw=1.0)
    for aa, bb in bands:
        ax2[1].axvspan(aa, bb, color='#009e73', alpha=.15)
    ax2[1].set_ylim(0, 1.02)
    ax2[1].set_ylabel('efficiency')
    ax2[1].set_xlabel('detector-local X [mm]')
    ax2[1].grid(alpha=.3)
    ax2[1].set_title('projection on X (green = live bands from the charge profile)',
                     fontsize=10)
    fig2.savefig(os.path.join(args.out, f'efficiency_map_{args.key}.png'),
                 dpi=115, bbox_inches='tight')

    print(json.dumps({k: v for k, v in rep.items() if k != 'T3_per_connector'},
                     indent=1))
    print('\nper connector (X plane):')
    print(f'{"conn":>5}{"X range [mm]":>18}{"X occ":>9}{"X amp":>8}'
          f'{"Y occ":>9}{"Y amp":>8}{"live frac":>11}')
    for k, v in conn.items():
        print(f'{k:5d}{v["x_mm"][0]:8.0f}-{v["x_mm"][1]:<8.0f}{v["x_occ"]:9.3f}'
              f'{v["x_amp"]:8.0f}{v["y_occ"]:9.3f}{v["y_amp"]:8.0f}'
              f'{v["frac_strips_in_live_band"]:11.2f}')
    print(f'\ncorr(X-connector occupancy, live-band fraction) = '
          f'{t3["corr_Xconn_occ_vs_live_fraction"]:.3f}')


if __name__ == '__main__':
    main()
