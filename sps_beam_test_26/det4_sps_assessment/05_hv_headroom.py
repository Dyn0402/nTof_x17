#!/usr/bin/env python3
"""
05_hv_headroom.py — does more amplification voltage fill in det4's dead stripes?

The 6-23 overnight run stepped det4's resist voltage 465 -> 525 V (drift 600 V,
20 min per point). Its efficiency was never measured because that run's M3
reference is degraded (~4 % clean tracks, see JUNE_RESULTS_SUMMARY.md §3) — but
none of the questions here need a reference:

  * the X plane's strips measure detector-local X, so the per-strip occupancy of
    FEU 6 IS the stripe pattern, read straight off the raw hits;
  * gain shows up as median hit amplitude and as cluster size;
  * the discharge rate shows up as the fraction of events over the spark veto.

So the scan answers the operational question directly: is det4 simply being run
below its plateau (dead regions fill in with voltage), or is the pattern fixed
and only the live stripes get louder?

    ../../.venv/bin/python sps_beam_test_26/det4_sps_assessment/05_hv_headroom.py
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import setup_paths                          # noqa: E402
setup_paths()
import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
import uproot                                              # noqa: E402
import cosmic_micro_tpc_analysis as cm                     # noqa: E402
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS        # noqa: E402
from common.Mx17StripMap import Mx17StripMap               # noqa: E402

RUN = '/home/dylan/x17/cosmic_bench/det3_det4/mx17_det3_det4_overnight_6-23-26'
DET_FEUS = {'mx17_4': (6, 8), 'mx17_3': (3, 4)}


def strip_positions(map_csv, axis):
    sm = Mx17StripMap(map_csv)
    pos = np.full(512, np.nan)
    for ch in range(512):
        k, lc = Mx17StripMap.feu_channel_to_connector(ch)
        p = sm.lookup(axis, k, lc)
        if p is not None:
            pos[ch] = p[0] if axis == 'x' else p[1]
    return pos


def point(path, feus, posx):
    fs = sorted(glob.glob(os.path.join(path, 'combined_hits_root', '*.root')))
    if not fs:
        return None
    raw = uproot.concatenate([f'{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel',
                                          'amplitude', 'significance'], library='pd')
    n_ev = int(raw['eventId'].nunique())
    det = raw[raw['feu'].isin(feus)].copy()
    if not len(det):
        return None
    det = cm.apply_significance_floor(det, rel=SIG_REL_FLOOR)
    mult = det.groupby('eventId').size()
    spark_ev = set(mult[mult > SPARK_VETO_HITS].index)
    good = det[~det['eventId'].isin(spark_ev)]
    fx, fy = feus
    gx, gy = good[good.feu == fx], good[good.feu == fy]

    # per-strip occupancy on the X plane = the stripe pattern, per event
    occ = np.zeros(512)
    amp = np.full(512, np.nan)
    for ch, sub in gx.groupby('channel'):
        c = int(ch)
        if 0 <= c < 512:
            occ[c] = len(sub) / max(n_ev, 1)
            amp[c] = float(np.median(sub['amplitude']))

    nper_x = gx.groupby('eventId').size()
    nper_y = gy.groupby('eventId').size()
    ev = set(good['eventId'].unique())
    n3 = sum(1 for e in ev if nper_x.get(e, 0) >= 3 and nper_y.get(e, 0) >= 3)
    return dict(
        n_events=n_ev,
        frac_fired=len(ev) / max(n_ev, 1),
        spark_frac=len(spark_ev) / max(n_ev, 1),
        frac_ge3_both=n3 / max(n_ev, 1),
        mean_strips_x=float(nper_x.mean()) if len(nper_x) else 0.0,
        mean_strips_y=float(nper_y.mean()) if len(nper_y) else 0.0,
        median_amp_x=float(gx['amplitude'].median()) if len(gx) else np.nan,
        median_amp_y=float(gy['amplitude'].median()) if len(gy) else np.nan,
        occ=occ, amp=amp, pos=posx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default='mx17_4')
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    feus = DET_FEUS[args.det]
    posx = strip_positions(os.path.join(REPO, 'mx17_m1_map.csv'), 'x')

    pts = []
    for d in sorted(glob.glob(os.path.join(RUN, 'resist_*V_drift_*V'))):
        hv = int(re.search(r'resist_(\d+)V', os.path.basename(d)).group(1))
        r = point(d, feus, posx)
        if r is None:
            print(f'  {hv} V: no data')
            continue
        r['hv'] = hv
        pts.append(r)
        print(f'  {hv:3d} V  events {r["n_events"]:6d}  fired {r["frac_fired"]:.3f}  '
              f'spark {r["spark_frac"]:.3f}  ge3both {r["frac_ge3_both"]:.3f}  '
              f'strips {r["mean_strips_x"]:.2f}/{r["mean_strips_y"]:.2f}  '
              f'amp {r["median_amp_x"]:.0f}/{r["median_amp_y"]:.0f}')
    pts.sort(key=lambda p: p['hv'])

    # Connectors 7-8 of FEU 6 recorded nothing in this run (they are the
    # highest-occupancy connectors in the 6-24 run, so this is a cabling/enable
    # state of the 6-23 run, not chamber). Score only connectors that are read
    # out, or the live fraction carries a constant dead offset.
    tot = np.array([sum(p['occ'][k * 64:(k + 1) * 64].sum() for p in pts)
                    for k in range(8)])
    live_conn = tot > 0.01 * tot.max()      # a few stray hits is not "read out"
    read = np.repeat(live_conn, 64)
    print(f'connectors read out in this run: '
          f'{[k + 1 for k in range(8) if live_conn[k]]}  '
          f'(summed occupancy {np.round(tot, 3).tolist()})')

    # "live fraction": strips whose occupancy is above 25 % of the point's own
    # 90th-percentile occupancy — a scale-free measure of how much of the
    # chamber is amplifying, insensitive to the overall gain level.
    for p in pts:
        o = p['occ'][read & np.isfinite(p['pos'])]
        ref = np.percentile(o, 90)
        p['live_frac'] = float(np.mean(o > 0.25 * ref)) if ref > 0 else np.nan
        # absolute version: occupancy above a fixed hits/event floor
        p['live_frac_abs'] = float(np.mean(o > 0.005))
        p['n_strips_scored'] = int(read.sum())

    hv = np.array([p['hv'] for p in pts])
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    a = axs[0, 0]
    a.plot(hv, [p['median_amp_x'] for p in pts], 'o-', color='#0072b2', label='X plane')
    a.plot(hv, [p['median_amp_y'] for p in pts], 'o-', color='#d55e00', label='Y plane')
    a.set_ylabel('median hit amplitude [ADC]')
    a.set_yscale('log')
    a.set_title(f'{args.det} — gain vs resist HV (drift 600 V)')
    a.legend(fontsize=8)
    a.grid(alpha=.3, which='both')

    a = axs[0, 1]
    a.plot(hv, [p['frac_ge3_both'] for p in pts], 'o-', color='k',
           label='events with >=3 strips on both planes')
    a.plot(hv, [p['frac_fired'] for p in pts], 'o-', color='#009e73',
           label='events with any hit')
    a.plot(hv, [p['spark_frac'] for p in pts], 'o-', color='#cc79a7',
           label='discharge fraction')
    a.set_ylabel('fraction of triggers')
    a.set_ylim(0, 1.02)
    a.legend(fontsize=8)
    a.grid(alpha=.3)
    a.set_title('reconstructability and discharges')

    a = axs[1, 0]
    a.plot(hv, [p['live_frac'] for p in pts], 'o-', color='k',
           label='> 25 % of this point\'s own p90 occupancy')
    a.plot(hv, [p['live_frac_abs'] for p in pts], 'o-', color='#e69f00',
           label='> 0.005 hits/event (absolute)')
    a.set_xlabel('resist HV [V]')
    a.set_ylabel('fraction of X strips that are live')
    a.set_ylim(0, 1.02)
    a.legend(fontsize=8)
    a.grid(alpha=.3)
    a.set_title('does the dead area fill in?')

    a = axs[1, 1]
    cmap = plt.get_cmap('viridis')
    for i, p in enumerate(pts):
        o = np.argsort(np.where(read, p['pos'], np.nan))
        a.plot(p['pos'][o], np.where(read, p['occ'], np.nan)[o], lw=.9, color=cmap(i / max(len(pts) - 1, 1)),
               label=f'{p["hv"]} V' if p['hv'] % 15 == 0 else None)
    a.set_xlabel('detector-local X [mm]')
    a.set_ylabel('X-plane hits per event per strip')
    a.set_yscale('log')
    a.legend(fontsize=7, ncol=2)
    a.grid(alpha=.3, which='both')
    a.set_title('stripe pattern at every HV point')
    fig.suptitle(f'{args.det} resist HV scan, 6-23 run — reference-free')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f'hv_headroom_{args.det}.png'), dpi=115)

    np.savez(os.path.join(args.out, f'hv_headroom_{args.det}.npz'),
             hv=hv, pos=posx,
             occ=np.array([p['occ'] for p in pts]),
             amp=np.array([p['amp'] for p in pts]))
    with open(os.path.join(args.out, f'hv_headroom_{args.det}.json'), 'w') as f:
        json.dump([{k: v for k, v in p.items()
                    if k not in ('occ', 'amp', 'pos')} for p in pts], f, indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
