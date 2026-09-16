#!/usr/bin/env python3
"""
two_track_synth -- synthetic two-track planes for the joint fit.

Step 1 of HANDOFF_JOINT_TWO_TRACK_FIT.md §7.1: generate planes from
``wft/model.py`` itself, under the production bundle and with the run's own
noise, and ask whether ``wft.reco.fit_plane_two`` gets the two tracks back.
No file I/O, no seeder, no selector -- only the fit's logic is under test here,
which is why it is the first gate and not the last.

What it establishes, and what it cannot:

* **can**: does the optimiser find both tracks, from what separation, with what
  bias on p0 and tan; does it split a *one*-track plane (the false-split rate
  against a perfectly modelled single track); how the model-selection statistic
  separates the two populations; what one attempt costs.
* **cannot**: anything about real charge, real noise structure, real clusters
  with holes, or the seeder. Those are the overlay bench (``intra_bench.py``)
  and the single-track A/B.

    python -m sept26_prelim_analysis.two_track_synth run --n 600 --jobs 12
    python -m sept26_prelim_analysis.two_track_synth summarise
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths            # noqa: E402

RUN, SUBRUN = 'run_145', 'stat090_0000'
ARMS = ('A', 'C')
PAD = 3                     # wft.io.extract_window's production pad
NSAMP = 20                  # run_145 DAQ window
SIG_SEED = 5.0              # strip significance that would have been a hit
MATCH_MM = 3.0              # same truth-matching tolerance as intra_bench
SCHEMA = 'sept26_prelim/two_track_synth/1'

#: separations scanned [mm]; dense where the seed gap merges the tracks
SEP_GRID = np.array([0.0, 1.5, 3.0, 4.5, 6.0, 9.0, 12.0, 16.0, 20.0, 24.0,
                     32.0, 48.0])
#: t0 offsets of the second track [ns]: prompt pair, and an accidental
DT_GRID = np.array([0.0, 200.0])
#: charge of the second track relative to the first
QRATIO_GRID = np.array([1.0, 0.5, 0.25])

#: drawn per event, from the run_145 clean-donor distributions (probe, 2026-09-16)
T0_MED = {'A': -40.0, 'C': 6.0}
T0_SIG = {'A': 90.0, 'C': 120.0}
NOISE_MED = 13.3
NOISE_LO, NOISE_HI = 4.5, 22.2
Q_MED = 1450.0              # NNLS charge of a clean single track
TAN_SIG = 0.20
UEND_LO, UEND_HI = 600.0, 1080.0

_CAL = None
_ARM = None


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def _bundle(arm: str) -> str:
    return str(paths.require(
        paths.spell('out', 'reco_fullpass', RUN, SUBRUN, f'mx17_{arm}',
                    'calib_bundle_prelim'), f'arm {arm} bundle'))


def _init(arm: str) -> None:
    global _CAL, _ARM
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _CAL = CalibrationBundle.load(_bundle(arm))
    _ARM = arm
    wm.use_calibration(_CAL)
    wm.set_nsamp(NSAMP)


def _profile(uend_ns: float, rng) -> np.ndarray:
    """A track's charge against drift depth: flat to the end of its column,
    with the bin-to-bin scatter a real ionisation profile has."""
    from wft import model as wm
    k = max(1, min(wm.K, int(round(uend_ns / wm.DT))))
    q = np.zeros(wm.K)
    q[:k] = rng.uniform(0.6, 1.4, k)
    return q


def make_plane(plane, tracks, noise_lvl, rng):
    """One window of one plane holding ``tracks`` = [(p0, w, t0, q_tot)].

    The window is cut the way production cuts it: the strips a 5-sigma hit
    would have appeared on, plus the seeder's pad -- so a merged pair gets one
    window exactly as ``seed_candidates`` would give it.

    Each track is also recorded as *detectable or not on its own*: a steep,
    faint track spreads its charge over 30 mm and never reaches 5 sigma on any
    strip, so it would not have been seeded and is not the joint fit's to find.
    Counting those as failures would flatter nothing and confuse everything."""
    from wft import model as wm
    allpos = np.arange(512) * wm.PITCH
    W = np.zeros((512, wm.NSAMP))
    truth, solo = [], []
    for p0, w, t0, qt in tracks:
        q = _profile(rng.uniform(UEND_LO, UEND_HI), rng)
        q *= qt / q.sum()
        Wi = (wm.build_matrix(plane, allpos, p0, w, t0, wm.HYPER) @ q
              ).reshape(512, wm.NSAMP)
        W += Wi
        solo.append(Wi)
        truth.append(dict(p0=p0, w=w, t0=t0, q=float(q.sum())))
    live = np.flatnonzero(W.max(axis=1) / noise_lvl > SIG_SEED)
    if len(live) < 3:
        return None, truth
    lo, hi = max(0, live.min() - PAD), min(511, live.max() + PAD)
    ch = np.arange(lo, hi + 1)
    for t_, Wi in zip(truth, solo):
        n_own = int((Wi.max(axis=1) / noise_lvl > SIG_SEED).sum())
        t_['n_own_strips'] = n_own
        t_['detectable'] = bool(n_own >= 3
                                and allpos[lo] - PAD * wm.PITCH <= t_['p0']
                                <= allpos[hi] + PAD * wm.PITCH)
    noise = np.full(len(ch), noise_lvl)
    Wn = W[ch] + rng.normal(0.0, noise_lvl, (len(ch), wm.NSAMP))
    return dict(W=Wn, pos=allpos[ch], noise=noise, ch=ch), truth


def _one(job):
    """One synthetic event: build it, fit one track, try two."""
    from wft import model as wm
    from wft import reco as wr
    (idx, plane, sep, dt, qratio, n_true, seed) = job
    rng = np.random.default_rng(seed)
    v = _CAL.v_drift
    t0 = float(rng.normal(T0_MED[_ARM], T0_SIG[_ARM]))
    noise_lvl = float(np.clip(rng.normal(NOISE_MED, 4.0), NOISE_LO, NOISE_HI))
    p0a = float(rng.uniform(80.0, 300.0))
    # |tan| < TAN_MAX, since a candidate outside it is not plausible anyway
    wa = float(np.clip(rng.normal(0.0, TAN_SIG), -0.5, 0.5) * v * 1e-3)
    wb = float(np.clip(rng.normal(0.0, TAN_SIG), -0.5, 0.5) * v * 1e-3)
    qa = float(Q_MED * np.exp(rng.normal(0.0, 0.35)))
    tracks = [(p0a, wa, t0, qa)]
    if n_true == 2:
        tracks.append((p0a + sep, wb, t0 + dt, qa * qratio))
    P, truth = make_plane(plane, tracks, noise_lvl, rng)
    row = dict(idx=idx, arm=_ARM, plane=plane, sep=sep, dt=dt, qratio=qratio,
               n_true=n_true, t0=t0, noise=noise_lvl, seeded=P is not None,
               tan_a=wa * 1e3 / v, tan_b=wb * 1e3 / v)
    if P is None:
        return row
    row['n_strips'] = int(P['W'].shape[0])
    t = time.perf_counter()
    try:
        f = wr.fit_plane(P, plane, _CAL)
    except Exception:
        f = None
    row['t_one'] = time.perf_counter() - t
    if f is None:
        return row
    row.update(one_p0=f.p0, one_w=f.w, one_tan=f.tan_theta,
               one_chi2dof=f.chi2 / max(f.dof, 1))
    probe = wr.two_track_probe(P, plane, f, _CAL.hyper)
    if probe is None:
        return row
    trig = wr.two_track_triggers(probe)
    row.update({f'trig_{k}': v2 for k, v2 in trig.items()})
    t = time.perf_counter()
    try:
        r = wr.fit_plane_two(P, plane, _CAL, f, probe=probe, f_thresh=-np.inf)
    except Exception:
        r = None
    row['t_two'] = time.perf_counter() - t
    if r is None:
        return row
    ca, cb = r['children']
    row.update(fstat=r['fstat'], f_total=r['f_total'], dchi2=r['dchi2'],
               chi2dof_two=r['chi2_two'] / max(r['dof'], 1), sep_fit=r['sep'],
               dist=r['dist'], tie_t0=r['tie_t0'], nfev=r['nfev'],
               overlap=r['overlap'],
               guards_ok=r['guards_ok'],
               distinguishable=r['distinguishable'],
               column_shared=r['column_shared'],
               both_plausible=r['both_plausible'],
               ch_p0_a=ca.p0, ch_p0_b=cb.p0, ch_tan_a=ca.tan_theta,
               ch_tan_b=cb.tan_theta, ch_t0_a=ca.t0, ch_t0_b=cb.t0)
    # truth matching: each true track to its nearest child, both distinct
    tp = [t_['p0'] for t_ in truth]
    cp, ct = [ca.p0, cb.p0], [ca.tan_theta, cb.tan_theta]
    order = (0, 1) if (abs(cp[0] - tp[0]) + abs(cp[1] - tp[-1])) <= \
        (abs(cp[1] - tp[0]) + abs(cp[0] - tp[-1])) else (1, 0)
    for lab, j in (('a', 0), ('b', 1)):
        if j >= n_true:
            continue
        row[f'd_p0_{lab}'] = cp[order[j]] - tp[j]
        row[f'd_tan_{lab}'] = ct[order[j]] - truth[j]['w'] * 1e3 / v
        row[f'found_{lab}'] = bool(abs(row[f'd_p0_{lab}']) < MATCH_MM)
        row[f'detectable_{lab}'] = bool(truth[j].get('detectable', False))
        row[f'n_own_{lab}'] = int(truth[j].get('n_own_strips', 0))
    if n_true == 2:
        row['both_detectable'] = bool(row['detectable_a'] and row['detectable_b'])
        row['both_found'] = bool(row['found_a'] and row['found_b'])
    return row


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def out_dir() -> Path:
    return paths.out('two_track_synth')


def run(arms, n_per_cell: int, jobs: int, seed: int) -> None:
    rows = []
    t_start = time.time()
    for arm in arms:
        jobs_list, k = [], 0
        rng = np.random.default_rng(seed)
        for plane in ('x', 'y'):
            for sep in SEP_GRID:
                for dt in DT_GRID:
                    for qr in QRATIO_GRID:
                        for _ in range(n_per_cell):
                            jobs_list.append((k, plane, float(sep), float(dt),
                                              float(qr), 2, int(rng.integers(1 << 62))))
                            k += 1
            # the one-track control: the false-split rate lives here
            for _ in range(n_per_cell * len(SEP_GRID) * len(DT_GRID)):
                jobs_list.append((k, plane, 0.0, 0.0, 0.0, 1,
                                  int(rng.integers(1 << 62))))
                k += 1
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init,
                                 initargs=(arm,)) as pool:
            for i, r in enumerate(pool.map(_one, jobs_list, chunksize=8)):
                rows.append(r)
                if (i + 1) % 2000 == 0:
                    print(f'[synth] {arm}: {i + 1:,}/{len(jobs_list):,}', flush=True)
        print(f'[synth] arm {arm}: {len(jobs_list):,} events', flush=True)
    D = pd.DataFrame(rows)
    od = out_dir()
    D.to_parquet(od / 'synth.parquet', index=False)
    (od / 'run.meta.json').write_text(json.dumps(dict(
        schema=SCHEMA, run=RUN, subrun=SUBRUN, arms=list(arms), seed=seed,
        n_per_cell=n_per_cell, sep_grid=SEP_GRID.tolist(), dt_grid=DT_GRID.tolist(),
        qratio_grid=QRATIO_GRID.tolist(), nsamp=NSAMP, n_events=int(len(D)),
        minutes=round((time.time() - t_start) / 60, 1),
        built=time.strftime('%Y-%m-%dT%H:%M:%S')), indent=1))
    print(f'[synth] wrote {od}/synth.parquet ({len(D):,} rows, '
          f'{(time.time() - t_start) / 60:.1f} min)')


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def rsig(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan
    q = np.percentile(v, [16, 84])
    return float(0.5 * (q[1] - q[0]))


def threshold_scan(D: pd.DataFrame, bands) -> pd.DataFrame:
    """Two-track efficiency against the false-split rate, over the threshold.

    Read it as the operating curve: pick the threshold from this, not from one
    number (HANDOFF_JOINT_TWO_TRACK_FIT §6)."""
    rows = []
    ok = D.fstat.notna() & D.guards_ok.fillna(False).astype(bool)
    # a pair only counts against the fit when BOTH legs would have been seeded
    # on their own -- a faint, steep second track never reached 5 sigma anywhere
    det = D.get('both_detectable', pd.Series(True, index=D.index)).fillna(False).astype(bool)
    for arm, g in D.groupby('arm'):
        gd = det.reindex(g.index, fill_value=False)
        gok = ok.reindex(g.index, fill_value=False)
        sing = g[(g.n_true == 1) & gok]
        pair = g[(g.n_true == 2) & gok & gd & g.both_found.fillna(False).astype(bool)]
        n_sing = int((g.n_true == 1).sum())
        for thr in [0, 5, 10, 20, 30, 50, 80, 120, 200, 300, 500, 1000]:
            row = dict(arm=arm, threshold=thr, n_single=n_sing,
                       false_split=float((sing.fstat >= thr).sum()) / max(n_sing, 1))
            for lo, hi in bands:
                sel = g[(g.n_true == 2) & gd & (g.sep >= lo) & (g.sep < hi)]
                got = sel.index.isin(pair[(pair.sep >= lo) & (pair.sep < hi)
                                          & (pair.fstat >= thr)].index)
                row[f'eff_{lo:g}_{hi:g}'] = float(got.sum()) / max(len(sel), 1)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise() -> None:
    od = out_dir()
    D = pd.read_parquet(od / 'synth.parquet')
    bands = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 400)]
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 60)

    per_sep = (D[(D.n_true == 2) & D.get('both_detectable', True).fillna(False).astype(bool)]
               .groupby(['arm', 'plane', 'dt', 'sep'])
               .agg(n=('idx', 'size'), seeded=('seeded', 'mean'),
                    both_found=('both_found', 'mean'),
                    found_a=('found_a', 'mean'), found_b=('found_b', 'mean'),
                    guards=('guards_ok', 'mean'),
                    fstat_med=('fstat', 'median'),
                    chi2dof_one=('one_chi2dof', 'median'),
                    chi2dof_two=('chi2dof_two', 'median'),
                    trig_resid=('trig_residual', 'mean'),
                    trig_width=('trig_width', 'mean'),
                    guards_overlap=('column_shared', 'mean'),
                    guards_dist=('distinguishable', 'mean'),
                    rsig_dp0_a=('d_p0_a', rsig), rsig_dp0_b=('d_p0_b', rsig),
                    rsig_dtan_a=('d_tan_a', rsig), rsig_dtan_b=('d_tan_b', rsig),
                    t_two=('t_two', 'median'), nfev=('nfev', 'median'))
               .reset_index())
    per_sep.to_csv(od / 'by_separation.csv', index=False)

    single = (D[D.n_true == 1].groupby(['arm', 'plane'])
              .agg(n=('idx', 'size'), chi2dof=('one_chi2dof', 'median'),
                   trig_resid=('trig_residual', 'mean'),
                   trig_width=('trig_width', 'mean'),
                   guards=('guards_ok', 'mean'),
                   fstat_med=('fstat', 'median'),
                   fstat_p95=('fstat', lambda v: np.nanpercentile(v, 95)),
                   fstat_p99=('fstat', lambda v: np.nanpercentile(v, 99)),
                   fstat_max=('fstat', 'max'),
                   rsig_dp0=('d_p0_a', rsig)).reset_index())
    single.to_csv(od / 'single_control.csv', index=False)

    scan = threshold_scan(D, bands)
    scan.to_csv(od / 'threshold_scan.csv', index=False)

    print('\n== two tracks, by separation (coincident, all charge ratios) ==')
    print(per_sep[per_sep.dt == 0].drop(columns=['dt']).round(3).to_string(index=False))
    print('\n== one-track control (the false-split population) ==')
    print(single.round(3).to_string(index=False))
    print('\n== threshold scan: efficiency vs false splits ==')
    print(scan.round(4).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--arms', nargs='+', default=list(ARMS))
    r.add_argument('--n', type=int, default=40, help='events per (plane, sep, dt, qratio) cell')
    r.add_argument('--jobs', type=int, default=12)
    r.add_argument('--seed', type=int, default=20260916)
    sub.add_parser('summarise')
    a = ap.parse_args()
    if a.cmd == 'run':
        run(a.arms, a.n, a.jobs, a.seed)
    else:
        summarise()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
