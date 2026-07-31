#!/usr/bin/env python3
"""
03_charge_vs_position.py — is the banding in the chamber or in one plane's readout?

02_bands.py shows det4's efficiency swinging 0 -> 98 % with detector-local X.
That coordinate is measured by the X plane (FEU 6), so a dead/weak group of
X-plane channels would reproduce it. The two hypotheses separate cleanly on the
*other* plane:

  X-plane readout defect  -> at a bad X, the Y plane still collects its normal
                             charge (the muon still ionises, the mesh still
                             amplifies, only one readout is deaf).
  chamber gain defect     -> at a bad X, BOTH planes lose charge, because there
                             is no avalanche there to share.

So: per M3 ray, count hits and sum amplitude *separately per plane*, and profile
both against detector-local X. This uses only M3 ray positions and raw hits — no
detector reconstruction — so it cannot be an artefact of the cluster finder.

    ../../.venv/bin/python mx_june_cosmic_qa/det4_sps_assessment/03_charge_vs_position.py g_det4 sat_det3
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
from M3RefTracking import M3RefTracking, get_xy_positions  # noqa: E402
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS        # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE            # noqa: E402

sys.path.insert(0, HERE)
from importlib import import_module                        # noqa: E402
ref_to_det = import_module('01_uniformity').ref_to_det


def collect(key):
    cfg = get_config(key)
    params = cm.load_alignment(os.path.join(cfg.OUT_BASE, 'wft', 'alignment',
                                            'alignment.json'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    lx, ly = ref_to_det(px, np.array(yr), params)
    ray = pd.DataFrame({'eventId': [int(v) for v in evn], 'lx': lx, 'ly': ly})

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

    fx, fy = cfg.MX17_FEUS
    agg = (det.groupby(['eventId', 'feu'])
              .agg(n=('channel', 'size'), q=('amplitude', 'sum'),
                   amax=('amplitude', 'max')).reset_index())
    wide = agg.pivot(index='eventId', columns='feu',
                     values=['n', 'q', 'amax']).fillna(0.0)
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    ray = ray[(ray['eventId'] >= lo) & (ray['eventId'] <= hi)]
    ray = ray[~ray['eventId'].isin(spark_ev)]
    ray = ray.join(wide, on='eventId')
    ray = ray.fillna(0.0)
    ray = ray.rename(columns={f'n_{fx}': 'nx', f'n_{fy}': 'ny',
                              f'q_{fx}': 'qx', f'q_{fy}': 'qy',
                              f'amax_{fx}': 'ax', f'amax_{fy}': 'ay'})
    for c in ('nx', 'ny', 'qx', 'qy', 'ax', 'ay'):
        if c not in ray:
            ray[c] = 0.0
    ax0, ax1 = TRUE_ACTIVE['x']
    ay0, ay1 = TRUE_ACTIVE['y']
    ray = ray[(ray.lx >= ax0) & (ray.lx <= ax1) & (ray.ly >= ay0) & (ray.ly <= ay1)]
    return ray, cfg


def prof(v, w, edges, fn=np.mean, minn=15):
    idx = np.digitize(v, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() >= minn:
            out[i] = fn(w[m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+')
    ap.add_argument('--bin', type=float, default=8.0)
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for key in args.keys:
        ray, cfg = collect(key)
        fx, fy = cfg.MX17_FEUS
        ax0, ax1 = TRUE_ACTIVE['x']
        ay0, ay1 = TRUE_ACTIVE['y']
        ex = np.arange(ax0, ax1 + args.bin, args.bin)
        ey = np.arange(ay0, ay1 + args.bin, args.bin)
        cxs, cys = 0.5 * (ex[:-1] + ex[1:]), 0.5 * (ey[:-1] + ey[1:])

        fig, axs = plt.subplots(2, 2, figsize=(15, 9))
        for col, (v, e, c, lab) in enumerate(
                [(ray.lx.to_numpy(), ex, cxs, 'local X'),
                 (ray.ly.to_numpy(), ey, cys, 'local Y')]):
            a = axs[0, col]
            a.plot(c, prof(v, ray.nx.to_numpy(), e), color='#0072b2',
                   label=f'X plane (FEU {fx}) strips/ray')
            a.plot(c, prof(v, ray.ny.to_numpy(), e), color='#d55e00',
                   label=f'Y plane (FEU {fy}) strips/ray')
            a.axhline(3, color='gray', ls=':', label='3 strips (reco threshold)')
            a.set_ylabel('mean fired strips per ray')
            a.set_title(f'{key} — mean cluster size vs {lab}')
            a.legend(fontsize=8)
            a.grid(alpha=.3)
            b = axs[1, col]
            b.plot(c, prof(v, ray.qx.to_numpy(), e, np.median), color='#0072b2',
                   label=f'X plane (FEU {fx}) sum amp')
            b.plot(c, prof(v, ray.qy.to_numpy(), e, np.median), color='#d55e00',
                   label=f'Y plane (FEU {fy}) sum amp')
            b.set_xlabel(f'detector-local {lab[-1]} [mm]')
            b.set_ylabel('median summed amplitude per ray [ADC]')
            b.legend(fontsize=8)
            b.grid(alpha=.3)
        fig.suptitle(f'{key} ({cfg.DET_NAME}) — per-plane charge vs position, '
                     f'{len(ray):,} non-spark active-area rays (reference-free)')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f'charge_{key}.png'), dpi=110)

        nxp = prof(ray.lx.to_numpy(), ray.nx.to_numpy(), ex)
        nyp = prof(ray.lx.to_numpy(), ray.ny.to_numpy(), ex)
        qxp = prof(ray.lx.to_numpy(), ray.qx.to_numpy(), ex, np.median)
        qyp = prof(ray.lx.to_numpy(), ray.qy.to_numpy(), ex, np.median)
        ok = np.isfinite(nxp) & np.isfinite(nyp)
        rep = dict(
            run_key=key, n_rays=int(len(ray)),
            mean_strips=dict(x=float(ray.nx.mean()), y=float(ray.ny.mean())),
            median_sum_amp=dict(x=float(ray.qx.median()), y=float(ray.qy.median())),
            frac_rays_ge3_strips=dict(x=float((ray.nx >= 3).mean()),
                                      y=float((ray.ny >= 3).mean()),
                                      both=float(((ray.nx >= 3) & (ray.ny >= 3)).mean())),
            vs_localX=dict(
                x_plane_strips=[float(np.nanmin(nxp)), float(np.nanmax(nxp))],
                y_plane_strips=[float(np.nanmin(nyp)), float(np.nanmax(nyp))],
                x_plane_q=[float(np.nanmin(qxp)), float(np.nanmax(qxp))],
                y_plane_q=[float(np.nanmin(qyp)), float(np.nanmax(qyp))],
                corr_xplane_yplane=float(np.corrcoef(nxp[ok], nyp[ok])[0, 1]),
                rel_rms_x_plane=float(np.nanstd(nxp) / np.nanmean(nxp)),
                rel_rms_y_plane=float(np.nanstd(nyp) / np.nanmean(nyp))),
        )
        np.savez(os.path.join(args.out, f'charge_{key}.npz'), ex=ex, ey=ey,
                 nxp=nxp, nyp=nyp, qxp=qxp, qyp=qyp,
                 lx=ray.lx.to_numpy(), ly=ray.ly.to_numpy(),
                 nx=ray.nx.to_numpy(), ny=ray.ny.to_numpy(),
                 qx=ray.qx.to_numpy(), qy=ray.qy.to_numpy())
        with open(os.path.join(args.out, f'charge_{key}.json'), 'w') as f:
            json.dump(rep, f, indent=1)
        print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
