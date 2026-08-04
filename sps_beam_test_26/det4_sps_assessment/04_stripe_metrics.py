#!/usr/bin/env python3
"""
04_stripe_metrics.py — measure det4's gain stripes and score the chamber inside them.

03 established that det4's collected charge varies by orders of magnitude with
detector-local X, identically on both readout planes — i.e. the amplification
region, not the readout, is patterned. This script:

  1. measures the stripe pattern (period, width, area fraction) from the
     reference-free per-ray charge profile;
  2. splits the chamber into "live" and "dead" stripes and scores each the way
     the fleet is scored — efficiency within 5 mm, core sigma, angular
     resolution — so the question "is there a fiducial region where det4 is a
     working detector?" gets a number.

    ../../.venv/bin/python sps_beam_test_26/det4_sps_assessment/04_stripe_metrics.py g_det4
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
import pandas as pd                                        # noqa: E402
import cosmic_micro_tpc_analysis as cm                     # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions  # noqa: E402
from wft import compat                                     # noqa: E402
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS        # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE            # noqa: E402

sys.path.insert(0, HERE)
from importlib import import_module                        # noqa: E402
ref_to_det = import_module('01_uniformity').ref_to_det


def robust_sigma(a):
    """MAD-based sigma, as used by mx_june_wft/03_angles.py."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


def rstd(v, ns=3, it=5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    for _ in range(it):
        m, s = np.median(v), np.std(v)
        k = np.abs(v - m) <= ns * s
        if k.all() or k.sum() < 10:
            break
        v = v[k]
    return float(np.std(v)) if len(v) else np.nan


def build(key, R=5.0):
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
    # per-plane angles and their reference truth, for the angular resolution
    # split. The reference tangents must be rotated into the detector's raw
    # strip frame first — same step as mx_june_wft/03_angles.py.
    ang = {}
    for r in results:
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
            continue
        ang[int(r.event_id)] = cm._rotate_ref_tangents(r, params)
    tanx = dict(zip(df['event_id'].to_numpy(), df['x_tan_theta'].to_numpy()))
    tany = dict(zip(df['event_id'].to_numpy(), df['y_tan_theta'].to_numpy()))
    relx = dict(zip(df['event_id'].to_numpy(), df['x_slope_reliable'].to_numpy()))
    rely = dict(zip(df['event_id'].to_numpy(), df['y_slope_reliable'].to_numpy()))

    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel', 'amplitude',
                                          'significance'], library='pd')
    det = raw[raw['feu'].isin(cfg.MX17_FEUS)].copy()
    lo, hi = int(det['eventId'].min()), int(det['eventId'].max())
    det = cm.apply_significance_floor(det, rel=SIG_REL_FLOOR)
    mult = det.groupby('eventId').size()
    spark_ev = set(mult[mult > SPARK_VETO_HITS].index)
    fired = set(int(e) for e in det['eventId'].unique())
    agg = (det.groupby(['eventId', 'feu'])
              .agg(n=('channel', 'size'), q=('amplitude', 'sum')).reset_index())
    wide = agg.pivot(index='eventId', columns='feu', values=['n', 'q']).fillna(0.0)
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    fx, fy = cfg.MX17_FEUS

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    py = np.array(yr)
    lx, ly = ref_to_det(px, py, params)
    ray = pd.DataFrame({'eventId': [int(v) for v in evn], 'rx': px, 'ry': py,
                        'lx': lx, 'ly': ly})
    ray = ray[(ray.eventId >= lo) & (ray.eventId <= hi)]
    ax0, ax1 = TRUE_ACTIVE['x']
    ay0, ay1 = TRUE_ACTIVE['y']
    ray = ray[(ray.lx >= ax0) & (ray.lx <= ax1) & (ray.ly >= ay0) & (ray.ly <= ay1)]
    ray = ray.join(wide, on='eventId').fillna(0.0)
    ray['spark'] = ray.eventId.isin(spark_ev)
    ray['fired'] = ray.eventId.isin(fired)
    ray['nx'] = ray.get(f'n_{fx}', 0.0)
    ray['ny'] = ray.get(f'n_{fy}', 0.0)
    ray['qx'] = ray.get(f'q_{fx}', 0.0)
    ray['qy'] = ray.get(f'q_{fy}', 0.0)
    r = []
    for e, x, y in zip(ray.eventId, ray.rx, ray.ry):
        if e in reco:
            r.append(float(np.hypot(x - reco[e][0], y - reco[e][1])))
        else:
            r.append(np.nan)
    ray['resid'] = r
    ray['near'] = ray.resid <= R
    ray['dthx'] = [np.degrees(np.arctan(tanx.get(e, np.nan)))
                   - np.degrees(np.arctan(ang.get(e, (np.nan, np.nan))[0]))
                   if relx.get(e, False) else np.nan for e in ray.eventId]
    ray['dthy'] = [np.degrees(np.arctan(tany.get(e, np.nan)))
                   - np.degrees(np.arctan(ang.get(e, (np.nan, np.nan))[1]))
                   if rely.get(e, False) else np.nan for e in ray.eventId]
    return ray, cfg


def score(sub, label):
    n = len(sub)
    if n == 0:
        return dict(label=label, n=0)
    ns = sub[~sub.spark]
    res = ns.resid.to_numpy()
    return dict(
        label=label, n_rays=int(n),
        spark=float(sub.spark.mean()),
        no_hit=float((~sub.fired).mean()),
        within_5mm=float(np.nan_to_num(sub.near.to_numpy(), nan=0.0).mean()),
        reco_at_all=float(np.isfinite(sub.resid.to_numpy()).mean()),
        mean_strips_x=float(ns.nx.mean()), mean_strips_y=float(ns.ny.mean()),
        median_q_x=float(ns.qx.median()), median_q_y=float(ns.qy.median()),
        core_sigma_mm=rstd(res[np.isfinite(res) & (res < 15)]),
        median_r_mm=float(np.nanmedian(res)) if np.isfinite(res).any() else np.nan,
        sigma_theta_x=robust_sigma(ns.dthx.to_numpy()),
        sigma_theta_y=robust_sigma(ns.dthy.to_numpy()),
        bias_theta_x=float(np.nanmedian(ns.dthx.to_numpy()))
        if np.isfinite(ns.dthx.to_numpy()).any() else np.nan,
        bias_theta_y=float(np.nanmedian(ns.dthy.to_numpy()))
        if np.isfinite(ns.dthy.to_numpy()).any() else np.nan,
        n_angle=int(np.isfinite(ns.dthx.to_numpy()).sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--bin', type=float, default=2.0)
    ap.add_argument('--split', action='store_true',
                    help='find bands on even eventIds, score the odd ones')
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    key = args.key

    ray, cfg = build(key)
    # The bands are found from the data, so finding and scoring them on the same
    # rays would let a fluctuation pick its own bin. Define the bands on even
    # eventIds and score the odd ones (--split), which costs nothing but makes
    # the live-stripe efficiency an out-of-sample number.
    if args.split:
        train = ray[ray.eventId % 2 == 0]
        test = ray[ray.eventId % 2 == 1]
    else:
        train = test = ray
    ns = train[~train.spark]
    ax0, ax1 = TRUE_ACTIVE['x']
    e = np.arange(ax0, ax1 + args.bin, args.bin)
    c = 0.5 * (e[:-1] + e[1:])
    idx = np.digitize(ns.lx.to_numpy(), e) - 1
    qtot = (ns.qx + ns.qy).to_numpy()
    med = np.full(len(c), np.nan)
    eff = np.full(len(c), np.nan)
    cnt = np.zeros(len(c))
    for i in range(len(c)):
        m = idx == i
        cnt[i] = m.sum()
        if m.sum() >= 8:
            med[i] = np.median(qtot[m])
            eff[i] = np.nan_to_num(ns.near.to_numpy()[m], nan=0.0).mean()

    # stripe pattern: threshold at the geometric midpoint of the profile in log q
    lq = np.log10(np.clip(med, 1, None))
    thr = 0.5 * (np.nanpercentile(lq, 10) + np.nanpercentile(lq, 90))
    live = lq > thr
    # contiguous live bands
    bands, i = [], 0
    while i < len(live):
        if live[i] and np.isfinite(lq[i]):
            j = i
            while j + 1 < len(live) and live[j + 1]:
                j += 1
            bands.append((float(e[i]), float(e[j + 1])))
            i = j + 1
        else:
            i += 1
    widths = [b - a for a, b in bands]
    centres = [0.5 * (a + b) for a, b in bands]
    spacing = list(np.diff(centres)) if len(centres) > 1 else []

    inband = np.zeros(len(test), bool)
    for a, b in bands:
        inband |= (test.lx.to_numpy() >= a) & (test.lx.to_numpy() <= b)
    rep = dict(
        run_key=key, detector=cfg.DET_NAME, bin_mm=args.bin,
        split_sample=bool(args.split),
        n_bands=len(bands), bands_mm=bands,
        band_width_mm=dict(median=float(np.median(widths)) if widths else np.nan,
                           min=float(np.min(widths)) if widths else np.nan,
                           max=float(np.max(widths)) if widths else np.nan),
        band_spacing_mm=dict(median=float(np.median(spacing)) if spacing else np.nan,
                             mean=float(np.mean(spacing)) if spacing else np.nan,
                             values=[float(s) for s in spacing]),
        live_area_fraction=float(sum(widths) / (ax1 - ax0)),
        charge_contrast=float(np.nanmax(med) / max(np.nanmin(med), 1e-9)),
        live=score(test[inband], 'live stripes'),
        dead=score(test[~inband], 'between stripes'),
        whole=score(test, 'whole active area'),
    )
    with open(os.path.join(args.out, f'stripes_{key}.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    np.savez(os.path.join(args.out, f'stripes_{key}.npz'), e=e, c=c, med=med,
             eff=eff, cnt=cnt, live=live, bands=np.array(bands))

    fig, axs = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axs[0].semilogy(c, med, 'k-', lw=1)
    axs[0].axhline(10 ** thr, color='r', ls='--', label='live/dead threshold')
    for a, b in bands:
        axs[0].axvspan(a, b, color='#009e73', alpha=.18)
    axs[0].set_ylabel('median summed amplitude, both planes [ADC]')
    axs[0].legend(fontsize=8)
    axs[0].set_title(f'{key} ({cfg.DET_NAME}) — amplification stripes, '
                     f'{args.bin:.0f} mm bins')
    axs[0].grid(alpha=.3, which='both')
    axs[1].plot(c, eff, 'k-', lw=1)
    for a, b in bands:
        axs[1].axvspan(a, b, color='#009e73', alpha=.18)
    axs[1].set_ylabel('efficiency within 5 mm')
    axs[1].set_xlabel('detector-local X [mm]')
    axs[1].set_ylim(0, 1.02)
    axs[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f'stripes_{key}.png'), dpi=115)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
