#!/usr/bin/env python3
"""
02_bands.py — what is the stripe structure in det4's efficiency map?

01_uniformity.py shows det4's within-5 mm efficiency varying 0 -> 93 % across
25 mm cells with an excess dispersion 3x the binomial expectation, organised in
vertical bands. Two candidate causes:

  (a) an electronics/connector effect — bands would line up with the 64-channel
      connector boundaries of the plane that measures that coordinate;
  (b) a chamber effect (bulk/mesh/pillar defects) — bands would sit anywhere.

This script separates them by putting three things on the same axis:
  * within-5 mm efficiency vs detector-local coordinate, finely binned;
  * per-plane reconstruction success (which plane fails where);
  * the reference-free per-strip occupancy and median hit amplitude, which
    measures gain directly without any tracking.

    ../../.venv/bin/python mx_june_cosmic_qa/det4_sps_assessment/02_bands.py g_det4 sat_det3
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

from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
import uproot                                              # noqa: E402
import cosmic_micro_tpc_analysis as cm                     # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions  # noqa: E402
from wft import compat                                     # noqa: E402
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS        # noqa: E402
from common.Mx17StripMap import Mx17StripMap               # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE            # noqa: E402

sys.path.insert(0, HERE)
from importlib import import_module                        # noqa: E402
_uni = import_module('01_uniformity')
ref_to_det = _uni.ref_to_det


def strip_profiles(cfg):
    """Reference-free per-strip occupancy + median amplitude, per plane.

    Returns {plane_axis: dict(pos, n, amp_med, feu, channel)} where plane_axis
    is 'x'/'y' as used by the strip map (axis 'x' strips measure local Y).
    """
    sm = Mx17StripMap(cfg.MAP_CSV_PATH)
    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel',
                                          'amplitude', 'significance'], library='pd')
    det = raw[raw['feu'].isin(cfg.MX17_FEUS)].copy()
    # drop discharge events: they dominate strip counts and are not gain
    mult = det.groupby('eventId').size()
    keep = set(mult[mult <= SPARK_VETO_HITS].index)
    det = det[det['eventId'].isin(keep)]

    out = {}
    for axis, feu in (('x', cfg.MX17_FEUS[0]), ('y', cfg.MX17_FEUS[1])):
        sub = det[det['feu'] == feu]
        g = sub.groupby('channel')['amplitude']
        chans = np.array(sorted(g.groups.keys()))
        n = np.array([len(g.get_group(c)) for c in chans], float)
        amp = np.array([np.median(g.get_group(c)) for c in chans], float)
        pos, conn = [], []
        for c in chans:
            k, lc = Mx17StripMap.feu_channel_to_connector(int(c))
            p = sm.lookup(axis, k, lc)
            # in mx17_m1_map.csv the 'x' strips carry the varying x_position and
            # the 'y' strips the varying y_position, i.e. plane X measures local X
            pos.append(np.nan if p is None else (p[0] if axis == 'x' else p[1]))
            conn.append(k)
        out[axis] = dict(feu=feu, channel=chans, pos=np.array(pos, float),
                         n=n, amp_med=amp, connector=np.array(conn))
    return out


def ray_profiles(key, R=5.0):
    """Per-ray local position + per-plane reconstruction success."""
    cfg = get_config(key)
    params = cm.load_alignment(os.path.join(cfg.OUT_BASE, 'wft', 'alignment',
                                            'alignment.json'))
    df = compat.load_table(os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet'),
                           max_dropped=None)
    results = compat.as_event_results(df)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, _ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm)
            for r in results if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}
    okx = dict(zip(df['event_id'].to_numpy(), df['x_ok'].to_numpy()))
    oky = dict(zip(df['event_id'].to_numpy(), df['y_ok'].to_numpy()))

    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel',
                                          'significance'], library='pd')
    det_raw = raw[raw['feu'].isin(cfg.MX17_FEUS)]
    det_lo, det_hi = int(det_raw['eventId'].min()), int(det_raw['eventId'].max())
    mult = (cm.apply_significance_floor(det_raw, rel=SIG_REL_FLOOR)
            .groupby('eventId').size())
    mult_by_ev = mult.to_dict()

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    py = np.array(yr)
    lx, ly = ref_to_det(px, py, params)
    rows = []
    for e, x, y, xl, yl in zip((int(v) for v in evn), px, py, lx, ly):
        if e < det_lo or e > det_hi or not (np.isfinite(xl) and np.isfinite(yl)):
            continue
        if mult_by_ev.get(e, 0) > SPARK_VETO_HITS:
            continue                              # sparks excluded here
        near = e in reco and np.hypot(x - reco[e][0], y - reco[e][1]) <= R
        rows.append((xl, yl, bool(okx.get(e, False)), bool(oky.get(e, False)), near))
    a = np.array(rows, dtype=[('lx', 'f8'), ('ly', 'f8'), ('x_ok', '?'),
                              ('y_ok', '?'), ('near', '?')])
    return a, cfg


def profile(v, flag, edges):
    idx = np.digitize(v, edges) - 1
    n = np.zeros(len(edges) - 1)
    k = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        m = idx == i
        n[i] = m.sum()
        k[i] = flag[m].sum()
    with np.errstate(invalid='ignore'):
        return np.where(n >= 15, k / n, np.nan), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+')
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for key in args.keys:
        a, cfg = ray_profiles(key)
        sp = strip_profiles(cfg)
        ax0, ax1 = TRUE_ACTIVE['x']
        ay0, ay1 = TRUE_ACTIVE['y']
        inb = ((a['lx'] >= ax0) & (a['lx'] <= ax1)
               & (a['ly'] >= ay0) & (a['ly'] <= ay1))
        a = a[inb]

        ex = np.arange(ax0, ax1 + 8, 8.0)
        ey = np.arange(ay0, ay1 + 8, 8.0)
        pnx, nx = profile(a['lx'], a['near'], ex)
        pny, ny = profile(a['ly'], a['near'], ey)
        # which plane fails where
        pxx, _ = profile(a['lx'], a['x_ok'], ex)
        pyx, _ = profile(a['lx'], a['y_ok'], ex)
        pxy, _ = profile(a['ly'], a['x_ok'], ey)
        pyy, _ = profile(a['ly'], a['y_ok'], ey)

        fig, axs = plt.subplots(2, 2, figsize=(15, 9), sharex='col')
        cx = 0.5 * (ex[:-1] + ex[1:])
        cy = 0.5 * (ey[:-1] + ey[1:])
        for col, (c, pn, ppx, ppy, lab) in enumerate(
                [(cx, pnx, pxx, pyx, 'local X'), (cy, pny, pxy, pyy, 'local Y')]):
            axs[0, col].plot(c, pn, 'k-', lw=2, label='within 5 mm')
            axs[0, col].plot(c, ppx, '-', color='#0072b2', label=f'X-plane ok (FEU {cfg.MX17_FEUS[0]})')
            axs[0, col].plot(c, ppy, '-', color='#d55e00', label=f'Y-plane ok (FEU {cfg.MX17_FEUS[1]})')
            axs[0, col].set_ylim(0, 1.02)
            axs[0, col].set_ylabel('fraction of non-spark rays')
            axs[0, col].legend(fontsize=8)
            axs[0, col].set_title(f'{key} — vs {lab}')
            axs[0, col].grid(alpha=.3)

        # strip-level, reference free.  plane X measures local X.
        for col, axis in enumerate(('x', 'y')):
            s = sp[axis]
            o = np.argsort(s['pos'])
            ax = axs[1, col]
            ax.plot(s['pos'][o], s['n'][o] / np.nanmedian(s['n']), '-', lw=.8,
                    color='#009e73', label='occupancy / median')
            ax.plot(s['pos'][o], s['amp_med'][o] / np.nanmedian(s['amp_med']),
                    '-', lw=.8, color='#cc79a7', label='median amplitude / median')
            ax.set_ylim(0, 2.2)
            ax.set_xlabel(f'detector-local {"X" if axis == "x" else "Y"} [mm]')
            ax.set_ylabel('normalised')
            ax.legend(fontsize=8)
            ax.grid(alpha=.3)
            ax.set_title(f'per-strip, FEU {s["feu"]} (reference-free)')
            for b in np.unique(s['connector']):
                m = s['connector'] == b
                ax.axvline(np.nanmin(s['pos'][m]), color='gray', ls=':', lw=.8)
        fig.suptitle(f'{key} ({cfg.DET_NAME}) — where the efficiency goes, '
                     f'{len(a):,} non-spark active-area rays')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f'bands_{key}.png'), dpi=110)

        np.savez(os.path.join(args.out, f'bands_{key}.npz'),
                 ex=ex, ey=ey, pnx=pnx, pny=pny, nx=nx, ny=ny,
                 pxx=pxx, pyx=pyx, pxy=pxy, pyy=pyy,
                 **{f'strip_{ax}_{k}': v for ax, d in sp.items()
                    for k, v in d.items() if k != 'feu'})
        # quantify: connector-to-connector amplitude spread per plane
        rep = {}
        for axis in ('x', 'y'):
            s = sp[axis]
            per = {}
            for b in np.unique(s['connector']):
                m = (s['connector'] == b) & np.isfinite(s['pos'])
                if m.sum() > 10:
                    per[int(b)] = dict(amp=float(np.median(s['amp_med'][m])),
                                       occ=float(np.median(s['n'][m])))
            amps = np.array([v['amp'] for v in per.values()])
            occs = np.array([v['occ'] for v in per.values()])
            rep[f'feu{s["feu"]}_{axis}'] = dict(
                per_connector=per,
                amp_spread_pct=float(100 * (amps.max() - amps.min()) / np.median(amps)),
                occ_spread_pct=float(100 * (occs.max() - occs.min()) / np.median(occs)))
        rep['eff_profile_x_min_max'] = [float(np.nanmin(pnx)), float(np.nanmax(pnx))]
        rep['eff_profile_y_min_max'] = [float(np.nanmin(pny)), float(np.nanmax(pny))]
        with open(os.path.join(args.out, f'bands_{key}.json'), 'w') as f:
            json.dump(rep, f, indent=1)
        print(key, json.dumps({k: v for k, v in rep.items()
                               if not k.startswith('feu')}, indent=1))
        for k, v in rep.items():
            if k.startswith('feu'):
                print(f'  {k}: amp spread {v["amp_spread_pct"]:.0f} %, '
                      f'occ spread {v["occ_spread_pct"]:.0f} %')


if __name__ == '__main__':
    main()
