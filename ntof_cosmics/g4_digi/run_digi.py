#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_digi.py -- drive digitise.py: synthetic muons (the gate) or Geant4 steps
through the production reconstruction under a bundle.

    PYTHONPATH=. .venv/bin/python ntof_cosmics/g4_digi/run_digi.py muons --n 400
    PYTHONPATH=. .venv/bin/python ntof_cosmics/g4_digi/run_digi.py g4 --steps <steps.parquet>

Outputs: OUT/<label>.parquet, one row per digitised event (truth + reco row).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ntof_cosmics.g4_digi import digitise as DG  # noqa: E402

# all overridable for condor (condor/run_digi_job.sh sets them)
OUT = Path(os.environ.get('G4DIGI_OUT', '/media/dylan/data/x17/ntof_cosmics/g4_digi'))
BUNDLES = Path(os.environ.get('G4DIGI_BUNDLES', Path.home() / 'scratch' / 'ntof_insitu' / 'bundles'))
IS2_TRACKS = Path(os.environ.get('G4DIGI_IS2_TRACKS', Path.home() / 'scratch/ntof_insitu/beamseed/is2/run_145/stat090_0000/tracks/tracks.parquet'))
REDUCED = Path(os.environ.get('G4DIGI_REDUCED', '/media/dylan/data/x17/ntof_cosmics/g4_angle/neutrons_nose'))
D_PERP = 234.6
FOOT = {'A': 16.35, 'C': 17.3}
# beam conditions (is2 on run_145 stat090_0000, coincident gated tracks):
# median x q_sum and y/x q_sum per arm
Q_X_MEDIAN = {'A': 1553.0, 'C': 1585.0}
YX = {'A': 0.881, 'C': 0.688}


def t0_sampler(arm: str):
    t = pd.read_parquet(IS2_TRACKS, columns=['arm', 'gated', 'coinc_this_arm', 'x_t0'])
    x = t[(t.arm == arm) & t.gated & t.coinc_this_arm.astype(bool)].x_t0.dropna().to_numpy()
    return x


def t0_pin_residuals(arm: str, t0_abs_x: dict, tracks: Path) -> np.ndarray:
    """For a bundle that pins t0 (n_TOF pass 1: t0 = t0_abs[ftst], sigma 15 ns): the data's x t0 minus the pin, on
    gated coincident tracks of ``tracks`` (a pass-1 stage-3 table).  The signal t0 is then the overlay trigger's own
    pin plus a residual drawn from this, so the simulated time keeps its per-trigger clock-phase correlation."""
    t = pd.read_parquet(tracks, columns=['arm', 'gated', 'coinc_this_arm', 'x_t0', 'x_ftst'])
    g = t[(t.arm == arm) & t.gated.astype(bool) & t.coinc_this_arm.astype(bool)]
    r = g.x_t0 - g.x_ftst.astype(int).astype(str).map({str(k): float(v) for k, v in t0_abs_x.items()})
    return r.dropna().to_numpy()


def _line(w, x, q):
    """edep-weighted line x(w): (slope, x at the mesh) -- reduce_gap_wall._fit."""
    W = q.sum()
    mw, mx = (q * w).sum() / W, (q * x).sum() / W
    sww = (q * (w - mw) ** 2).sum()
    b = (q * (w - mw) * (x - mx)).sum() / sww if sww > 0 else np.nan
    return float(b), float(mx + b * (DG.W_MESH - mw))


def run(jobs_list, bundle, state, min_strips, n_jobs, sim_bundle=None):
    rows = []
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=DG.worker_init,
                             initargs=(str(bundle), state, min_strips, sim_bundle and str(sim_bundle))) as pool:
        for i, r in enumerate(pool.map(DG.run_one, jobs_list, chunksize=4)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f'  {i + 1}/{len(jobs_list)}', flush=True)
    return pd.DataFrame(rows)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['muons', 'g4'])
    ap.add_argument('--arm', default='A')
    ap.add_argument('--bundle', default='is2_A')
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--sub', default='stat090_0000')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--steps', nargs='*', default=None, help='g4 step parquet(s) (extract_steps.py)')
    ap.add_argument('--g4-arm', type=int, default=2, help='sim arm id (2 = A, 3 = C)')
    ap.add_argument('--adc-per-e', type=float, default=None)
    ap.add_argument('--sim-bundle', default=None,
                    help='bundle the signal is generated with (default: --bundle); e.g. is2 physics, prod reco')
    ap.add_argument('--v-true', type=float, default=None, help='default: the sim bundle v')
    ap.add_argument('--min-strips', type=int, default=3)
    ap.add_argument('--q-x-median', type=float, default=None,
                    help='data median x q_sum the charge scale is set to (default: is2 run_145, Q_X_MEDIAN)')
    ap.add_argument('--yx', type=float, default=None,
                    help='data median y/x q_sum ratio that sets the x/y electron split (default: is2 run_145, YX)')
    ap.add_argument('--t0-tracks', default=None,
                    help='pass-1 stage-3 table (e.g. stage3_vt/tracks_run_145_stat090_0000.parquet): with a pinned '
                         'bundle, draw t0 as the overlay trigger\'s pin + a residual from this table')
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--label', default=None)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tan-u', nargs=2, type=float, default=None, metavar=('LO', 'HI'),
                    help='muons: |tan_u| uniform in [LO, HI] (random sign), u at the mesh uniform over '
                         '+-150 mm instead of pointing at the capsule (steep tracks would miss the plane)')
    ap.add_argument('--tan-max', type=float, default=None,
                    help='override wft.reco.TAN_MAX (raw-tan plausibility cut) in the forked workers')
    ap.add_argument('--w-scan-half', type=float, default=None,
                    help='override wft.reco.W_SCAN_HALF (start-scan slope range, mm/ns; step kept)')
    ap.add_argument('--select', default='wall', choices=['wall', 'fullgap', 'none'],
                    help='g4: wall = full gap + own track reaches the wall (default)')
    return ap


def setup(a):
    """Overlay, bundles and the per-event job list for parsed args ``a``:
    (jobs, bundle path, sim bundle path or None, overlay state).  Shared with
    mirror_chi2.py, which re-digitises the same events."""
    from wft.calib import CalibrationBundle
    bpath = Path(a.bundle) if Path(a.bundle).is_dir() else BUNDLES / a.bundle
    cal = CalibrationBundle.load(str(bpath))
    spath = None
    if a.sim_bundle:
        spath = Path(a.sim_bundle) if Path(a.sim_bundle).is_dir() else BUNDLES / a.sim_bundle
    v_true = a.v_true or (CalibrationBundle.load(str(spath)).v_drift if spath else cal.v_drift)
    from ntof_tracking import wft_beam as WB
    tag = a.tag or WB.subrun_tags(WB.beam_config(a.arm, a.run, a.sub))[0]
    print(f'overlay: {a.run}/{a.sub} tag {tag}; bundle {bpath.name} (v {cal.v_drift}, kw {cal.kw}); '
          f'sim {spath.name if spath else bpath.name}, v_true {v_true}')
    ov = DG.Overlay(a.arm, a.run, a.sub, tag)
    print(f'  {len(ov.events)} quiet overlay triggers, {ov.n_sample} samples')
    state = ov.state()
    pin = getattr(cal, 't0_abs', None)
    if pin and a.t0_tracks:
        # pinned bundle: t0 = pin[ftst_x of the overlay trigger] + data residual
        res = t0_pin_residuals(a.arm, pin['x'], Path(a.t0_tracks))
        print(f'  t0 from the pin + {len(res)} pass-1 residuals (median {np.median(res):+.1f} ns)')
        _pin_x = {int(k): float(v) for k, v in pin['x'].items()}

        def draw_t0(o, rng):
            return _pin_x[int(o['ftst_x'])] + float(rng.choice(res))
    else:
        t0s = t0_sampler(a.arm)

        def draw_t0(o, rng):
            return float(rng.choice(t0s))
    if a.q_x_median:
        Q_X_MEDIAN[a.arm] = a.q_x_median
    XY_SPLIT = 1.0 / (1.0 + (a.yx or YX[a.arm]))
    rng = np.random.default_rng(a.seed)

    jobs = []
    if a.mode == 'muons':
        # ~224 electrons per vertical MIP: 2.7 clusters/mm x 29.9 mm x 2.77
        adc = a.adc_per_e or Q_X_MEDIAN[a.arm] / (224 * XY_SPLIT)
        for i in range(a.n):
            if a.tan_u:
                tu = rng.uniform(*a.tan_u) * rng.choice((-1, 1))
                tv = rng.uniform(-0.3, 0.3)
                u0 = rng.uniform(-150, 150)
            else:
                tu = rng.uniform(-0.55, 0.55)
                tv = rng.uniform(-0.3, 0.3)
                u0 = FOOT[a.arm] + D_PERP * tu + rng.normal(0, 5)
            v0 = rng.uniform(-60, 60)
            st = DG.straight_steps(rng, u0, v0, tu, tv)
            o = ov.events[i % len(ov.events)]
            jobs.append(dict(eid=int(o['eid']), steps=st, ov=o, t0=draw_t0(o, rng),
                             adc_per_e=adc, xy_split=XY_SPLIT, v_true=v_true, seed=int(rng.integers(1 << 31)),
                             truth=dict(arm=a.arm, i=i, tan_u=tu, tan_v=tv, u_mesh=u0, v_mesh=v0, kind='mu')))
    else:
        S = pd.concat([pd.read_parquet(f) for f in a.steps], ignore_index=True)
        S = S[S.armID == a.g4_arm]
        T = pd.read_parquet(OUT / 'g4_truth.parquet') if (OUT / 'g4_truth.parquet').exists() else None
        keys = list(S.groupby(['file', 'eventID']).groups)
        # the reduced tables (g4_angle/reduce_gap_wall) say which (event, arm)
        # had a full-gap track whose own track reaches the wall: the sim
        # analogue of the data's pointing coincidence
        red = []
        for f in sorted(glob.glob(str(REDUCED / 'r*.parquet'))):
            r = pd.read_parquet(f, columns=['eventID', 'arm', 'w_lo', 'w_hi', 'wall_same_track', 'edep_gap'])
            r['file'] = 'neutron_bg_job' + Path(f).stem[1:]
            red.append(r[r.arm == a.g4_arm])
        red = pd.concat(red)
        full = red.w_hi - red.w_lo > 20
        wall = full & red.wall_same_track.astype(object).fillna(False).astype(bool)
        adc = a.adc_per_e
        if adc is None:
            # ADC per electron puts this library's median wall-reaching full-gap
            # track on the data's median x charge.  Measured on the library's own
            # reduction (was a fixed 247 e-, the July run's value: the EAR2 gun
            # gives ~255 with the old gas, the 2026-10-15 gas fix ~10 % fewer)
            n_med = float(red.edep_gap[wall].median()) / DG.W_ION
            adc = Q_X_MEDIAN[a.arm] / (n_med * XY_SPLIT)
            print(f'median wall-reaching full-gap track: {n_med:.1f} e- -> adc/e {adc:.2f}')
        if a.select != 'none':
            ok = wall if a.select == 'wall' else full
            good = set(zip(red.file[ok], red.eventID[ok]))
            keys = [k for k in keys if k in good]
            print(f'selection {a.select}: {len(keys)} (event, arm) groups')
        rng.shuffle(keys)
        keys = keys[:a.n]
        G = S.set_index(['file', 'eventID']).sort_index()
        for i, k in enumerate(keys):
            st = G.loc[k].reset_index(drop=True)
            o = ov.events[i % len(ov.events)]
            q = st.edep.to_numpy(float)
            tu, um = _line(st.w.to_numpy(float), st.u.to_numpy(float), q)
            tv, vm = _line(st.w.to_numpy(float), st.v.to_numpy(float), q)
            dom = st.groupby('trackID').edep.sum()
            ke = st[st.trackID == dom.idxmax()].sort_values('time').ke.iat[0]
            truth = dict(arm=a.arm, i=i, file=k[0], eventID=int(k[1]), kind='g4', tan_u=tu, tan_v=tv,
                         u_mesh=um, v_mesh=vm, w_lo=float(st.w.min()), w_hi=float(st.w.max()),
                         edep_gap=float(q.sum()), dom_ke_gap=float(ke), dom_share=float(dom.max() / q.sum()))
            jobs.append(dict(eid=int(o['eid']), steps=st[['u', 'v', 'w', 'edep', 'time']], ov=o,
                             t0=draw_t0(o, rng), adc_per_e=adc, xy_split=XY_SPLIT, v_true=v_true,
                             seed=int(rng.integers(1 << 31)), truth=truth))
    print(f'{len(jobs)} events, adc/e {adc:.2f}, v_true {v_true}')
    return jobs, bpath, spath, state


def main() -> int:
    a = parser().parse_args()
    jobs, bpath, spath, state = setup(a)
    if a.tan_max is not None:
        # the pool forks, so the workers inherit the patched module global
        from wft import reco as wreco
        wreco.TAN_MAX = a.tan_max
    if a.w_scan_half is not None:
        from wft import reco as wreco
        wreco.W_SCAN_HALF = a.w_scan_half
    print(f'TAN_MAX {__import__("wft.reco", fromlist=["x"]).TAN_MAX} (raw)')
    R = run(jobs, bpath, state, a.min_strips, a.jobs, spath)
    OUT.mkdir(parents=True, exist_ok=True)
    lab = a.label or f'{a.mode}_{a.arm}_{bpath.name}'
    R.to_parquet(OUT / f'{lab}.parquet', index=False)
    print(f'-> {OUT / lab}.parquet ({len(R)} rows, {int(R.x_ok.sum()) if "x_ok" in R else 0} x fits)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
