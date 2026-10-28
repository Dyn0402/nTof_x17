#!/usr/bin/env python3
"""gc_ntof.py -- GC, n_TOF: the gas drift of the four chambers vs time, two routes.

1. Production angle scale per sub-run (no new reconstruction): the capsule-band k of every production run's k_arm JSON
   (<out>/kcal/k_arm_run_<n>.json, per_subrun.band), arms A and C (B never certifies, D not in the in-situ chain).
   k is the factor the stage-3 table applies to the raw tangent (tan = k tan_raw, tan_raw = (w - w0)/(kw v_pin)), so
   k is proportional to 1/v_true and dln v = -dln k.  Stage 3 applies ONE k per run and arm (k_eff = norm x
   k_band(run)/k_band(ref)), so the sub-run-to-sub-run spread of k_band inside a run is the tan-scale error the
   table carries.  Sub-run times from dream_daq.log.
2. Model-free column duration (ntof_colsum.py, FA's estimator on the beam waveforms) on a representative set of
   sub-runs: the falling edge of the per-block stacked strip sum relative to the trigger (the trigger fixes t0 at
   n_TOF; the column end = latency + gap / v), see `column_end`.
Writes results/ntof_kband.json, results/ntof_colsum.json.
"""
import datetime as dt
import glob
import json
import os
import re

import numpy as np

from gc_common import RES, stem_time, tmean

KCAL = '/media/dylan/data/x17/sept26_prelim/kcal'
NTOF_EOS = '/eos/user/d/dneff/x17rerun/followups/gc/ntof'     # the products stay on EOS (~35 MB each, 79 sub-runs)
NTOF_TMP = '/tmp/gc_ntof_stream'                               # one at a time: copied, reduced, deleted
SAMPLE_NS = 60.0


def subrun_times():
    t = {}
    for ln in open(os.path.join(RES, 'ntof_subrun_times.txt')):
        m = re.match(r'(run_\d+) (\S+ \S+),\d+ INFO: Subrun (started|finished): (\S+)', ln)
        if not m:
            continue
        run, ts, what, sub = m.groups()
        t.setdefault((run, sub), {})[what] = dt.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()
    return t


def kband():
    times = subrun_times()
    out = []
    for f in sorted(glob.glob(os.path.join(KCAL, 'k_arm_run_*.json'))):
        if not re.fullmatch(r'k_arm_run_\d+\.json', os.path.basename(f)):
            continue
        d = json.load(open(f))
        run = d['run']
        for arm in 'AC':
            A = d['arms'].get(arm) or {}
            ps = A.get('per_subrun') or {}
            subs = []
            for s, v in ps.items():
                b = (v or {}).get('band')
                tt = times.get((run, s), {})
                if b is None or not np.isfinite(b) or 'started' not in tt:
                    continue
                subs.append(dict(subrun=s, k=float(b), t=0.5 * (tt['started'] + tt.get('finished', tt['started']))))
            if not subs:
                continue
            k = np.array([x['k'] for x in subs])
            kr = (A.get('per_estimator') or {}).get('band')
            # v relative to the run's applied (one-per-run) scale: v_sub / v_run = k_run / k_sub
            for x in subs:
                x['v_rel'] = float(np.median(k) / x['k'])
            vr = np.array([x['v_rel'] for x in subs])
            out.append(dict(run=run, arm=arm, k_run=kr, verdict=A.get('verdict'), n_sub=len(subs), subs=subs,
                            within_range_pct=float((vr.max() - vr.min()) * 100), within_rms_pct=float(vr.std() * 100),
                            t_start=min(x['t'] for x in subs), t_end=max(x['t'] for x in subs)))
    # sub-run-to-sub-run noise floor of the band estimator: half the rms of consecutive differences (arm A, the dry
    # chamber: its v hardly moves, so this is mostly estimator noise)
    for arm in 'AC':
        dif = []
        for r in out:
            if r['arm'] == arm and r['n_sub'] >= 3:
                v = np.array([x['v_rel'] for x in sorted(r['subs'], key=lambda z: z['t'])])
                dif += list(np.diff(v))
        print(f'{arm}: consecutive-subrun rms/sqrt2 = {np.std(dif) / np.sqrt(2) * 100:.2f} %  ({len(dif)} pairs)')
    allk = {a: np.array([x['k'] for r in out if r['arm'] == a for x in r['subs']]) for a in 'AC'}
    summ = {a: dict(campaign_range_pct=float((allk[a].max() / allk[a].min() - 1) * 100),
                    campaign_rms_pct=float(allk[a].std() / allk[a].mean() * 100)) for a in 'AC'}
    json.dump(dict(runs=out, summary=summ), open(os.path.join(RES, 'ntof_kband.json'), 'w'), indent=1)
    for r in out:
        print(f"{r['run']:8s} {r['arm']} n={r['n_sub']:2d} k_run={r['k_run']:.3f} within range {r['within_range_pct']:5.2f} % "
              f"rms {r['within_rms_pct']:4.2f} %")
    print(summ)


def column_end(S, frac=0.5):
    """falling-edge crossing [ns after window start] of a normalised stacked strip sum: the last sample above `frac` of
    the plateau (median of the samples above 0.5 max), linearly interpolated.  NaN if the edge is not in the window."""
    m = np.nanmax(S)
    if not m > 0:
        return np.nan
    pl = np.median(S[S > 0.5 * m])
    lvl = frac * pl
    above = np.flatnonzero(S >= lvl)
    if not len(above):
        return np.nan
    j = above[-1]
    if j >= len(S) - 1:
        return np.nan
    return float(SAMPLE_NS * (j + (S[j] - lvl) / (S[j] - S[j + 1])))


def _stream():
    """yield the local path of each EOS product in turn (scp via the shared lxplus connection), deleted after use"""
    import subprocess
    names = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'lxplus', f'ls {NTOF_EOS}'], capture_output=True, text=True,
                           check=True).stdout.split()
    os.makedirs(NTOF_TMP, exist_ok=True)
    for n in sorted(x for x in names if x.endswith('.npz')):
        loc = os.path.join(NTOF_TMP, n)
        subprocess.run(['scp', '-q', f'lxplus:{NTOF_EOS}/{n}', loc], check=True)
        try:
            yield loc
        finally:
            os.remove(loc)


def colsum_series(block_n=2000):
    out = []
    for f in _stream():
        run, sub = os.path.basename(f)[:-4].split('__')
        z = np.load(f)
        for arm in 'ABCD':
            if f'{arm}_eid' not in z.files:
                continue
            tags = [str(s) for s in z[f'{arm}_acq_tags']]
            t0 = np.array([stem_time(t).timestamp() for t in tags])
            acq = z[f'{arm}_acq']
            ts = np.where(z[f'{arm}_x_ts'] >= 0, z[f'{arm}_x_ts'], z[f'{arm}_y_ts']).astype(float)
            t = t0[acq] + ts * 1e-8
            for p in 'xy':
                ok = (z[f'{arm}_{p}_has'] & (z[f'{arm}_{p}_nsig'] >= 5) & (z[f'{arm}_{p}_ncl'] == 1)
                      & ~z[f'{arm}_{p}_sat'])
                S = z[f'{arm}_{p}_S'][ok].astype(float)
                tt = t[ok]
                if len(S) < block_n:
                    continue
                Sn = S / np.maximum(S.max(axis=1, keepdims=True), 1.0)
                o = np.argsort(tt); Sn, tt = Sn[o], tt[o]
                T = z[f'{arm}_{p}_T'][ok][o]; t0f = z[f'{arm}_{p}_t0'][ok][o]
                nb = len(tt) // block_n
                for b in range(nb):
                    sl = slice(b * block_n, (b + 1) * block_n)
                    st = np.median(Sn[sl], axis=0)
                    out.append(dict(run=run, subrun=sub, arm=arm, view=p, t=float(np.median(tt[sl])),
                                    t_end=column_end(st), t_end30=column_end(st, 0.3), T_evt=tmean(T[sl]),
                                    t0_evt=tmean(t0f[sl]), n=block_n, stack=st.round(4).tolist()))
        print(f'{os.path.basename(f)}: {sum(1 for x in out if x["run"] == run and x["subrun"] == sub)} blocks')
    json.dump(out, open(os.path.join(RES, 'ntof_colsum.json'), 'w'))


if __name__ == '__main__':
    kband()
    if os.environ.get('GC_COLSUM', '1') == '1':
        colsum_series()


T_START_NS = 30.0      # the column's mesh end in the n_TOF window (stack rise ~ the first sample): T = t_end - 30 ns
BIN_S = 1800.0


def colsum_runs():
    """per run and arm (A, C: the arms whose column end is inside the 1.2 us window): 30-min bins of the column end,
    both views averaged, v_rel = (t_end_run - T_START) / (t_end_bin - T_START); intrinsic rms (noise from consecutive
    differences removed) and worst bin.  Also the campaign series (run means) for the comparison with the band k."""
    X = json.load(open(os.path.join(RES, 'ntof_colsum.json')))
    out, camp = [], []
    for arm in 'AC':
        runs = sorted({x['run'] for x in X if x['arm'] == arm}, key=lambda r: int(r[4:]))
        for run in runs:
            v = [x for x in X if x['arm'] == arm and x['run'] == run and np.isfinite(x['t_end'])]
            if not v:
                continue
            t = np.array([x['t'] for x in v]); te = np.array([x['t_end'] for x in v])
            view = np.array([x['view'] for x in v])
            # per view offset removed (X and Y read different absolute ends), then the common series
            for p in 'xy':
                k = view == p
                if k.any():
                    te[k] = te[k] - np.median(te[k]) + np.median(te)
            T = te - T_START_NS
            Tm = float(np.median(T))
            camp.append(dict(run=run, arm=arm, t=float(np.median(t)), T=Tm, n_blocks=len(v)))
            edges = np.arange(t.min(), t.max() + BIN_S, BIN_S)
            tb, vb, eb = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = (t >= lo) & (t < hi)
                if k.sum() >= 2:
                    tb.append(float(t[k].mean())); vb.append(Tm / float(np.mean(T[k])))
                    eb.append(float(np.std(T[k]) / np.sqrt(k.sum()) / np.mean(T[k])))
            vb = np.array(vb); eb = np.array(eb)
            if len(vb) < 3:
                continue
            noise = float(np.std(np.diff(vb)) / np.sqrt(2))
            sig = float(np.sqrt(max(vb.var() - np.mean(eb ** 2), 0.0)))
            h = float(np.max(np.abs(vb / vb.mean() - 1)))
            out.append(dict(run=run, arm=arm, n_bins=len(vb), sigma=sig, h=h, noise=noise, t=tb, v_rel=vb.tolist(),
                            err=eb.tolist(), dur_h=float((t.max() - t.min()) / 3600),
                            note=f'{len(vb)} x 30 min, {(t.max() - t.min()) / 3600:.1f} h'))
            print(f'{run:8s} {arm}: {len(vb):2d} bins  rms {sig * 100:.2f} %  worst {h * 100:.2f} %  '
                  f'(bin noise {np.sqrt(np.mean(eb ** 2)) * 100:.2f} %)')
    json.dump(out, open(os.path.join(RES, 'ntof_colsum_runs.json'), 'w'), indent=1)
    json.dump(camp, open(os.path.join(RES, 'ntof_colsum_campaign.json'), 'w'), indent=1)
    # campaign cross-check: column-end v (1/T) against the band k (dln v = -dln k), run by run
    kb = {(r['run'], r['arm']): r['k_run'] for r in json.load(open(os.path.join(RES, 'ntof_kband.json')))['runs']}
    for arm in 'AC':
        c = [x for x in camp if x['arm'] == arm and (x['run'], arm) in kb]
        a = np.log(np.array([1 / x['T'] for x in c])); b = np.log(1 / np.array([kb[(x['run'], arm)] for x in c]))
        a -= a.mean(); b -= b.mean()
        sl = float(np.polyfit(b, a, 1)[0]); r = float(np.corrcoef(a, b)[0, 1])
        print(f'campaign {arm}: {len(c)} runs, dln(1/T) vs dln(1/k) slope {sl:.2f}, r = {r:.2f}, '
              f'1/T range {(np.exp(a.max() - a.min()) - 1) * 100:.1f} %, 1/k range {(np.exp(b.max() - b.min()) - 1) * 100:.1f} %')


if __name__ == '__main__' and os.environ.get('GC_RUNS', '1') == '1':
    colsum_runs()
