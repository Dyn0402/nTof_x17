"""pilot_compare.py -- is2_v1 pilot stage 3 against production, per run
(HANDOFF_TRACKING §16 next step 1, §18 next step 2).

The pilot reconstructed the first sub-run of each of the 36 runs with the
is2 in-situ bundles, min 3 strips, stage 2 at TAN_MAX 1.0 raw (one-sided
search) and a TRUE-tan acceptance of 0.6 at stage 3.  Production is
stage3_fullpass: TAN_MAX 0.6 raw at stage 2, its own per-run k where certified.

Per (run, arm):
  gated_reco / gated      what the reco accepts, and after the 0.6-true cut
  confirmed               SiPM-wall matches minus the off-time control, on
                          gated tracks inside the wall (det_a_scint.match_run)
  confirmed_acc           the same restricted to |true tan| < 0.6 in BOTH
                          chains -- the like-for-like yield comparison
  late_frac               max(x_t0, y_t0) > 300 ns among gated
  mirror_zone             gated pilot tracks with 0.3 <= |true tan| < 0.6,
                          where the one-sided mirror basin lives (§17)

Usage:  pilot_compare.py match   (wall matching, cached per run/arm/chain)
        pilot_compare.py summary
        pilot_compare.py <step> --version is2_v2   (the full two-sided re-pass)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sept26_prelim_analysis import det_a_scint as DS

ROOT = Path('/media/dylan/data/x17/sept26_prelim')
PILOT = ROOT / 'stage3_is2_v1_pilot'
PROD = ROOT / 'stage3_fullpass'   # stage3_campaign is the early low-statistics campaign (~1/15 of the events)
SUBSET = json.loads((ROOT / 'pilot_subset_is2_v1.json').read_text())
OUT = Path(__file__).resolve().parent / 'results' / 'pilot_is2_v1'
CACHE = ROOT / 'pilot_is2_v1_compare'
ARMS = ('A', 'C')
ACC_TRUE = 0.6
LATE_NS = 300
CHAINS = {'prod': PROD, 'pilot': PILOT}


def _configure(version):
    """Point the module at a full re-pass instead of the is2_v1 pilot.  The
    re-pass keeps the 'pilot' chain key; its sub-runs are every one with a
    track table in BOTH stage3_<version> and production, so an incomplete
    stage 3 compares like with like."""
    global PILOT, SUBSET, OUT, CACHE, CHAINS
    PILOT = ROOT / f'stage3_{version}'
    have = lambda d: {p.stem[len('tracks_'):] for p in d.glob('tracks_run_*_stat090_*.parquet')}
    both = have(PILOT) & have(PROD)
    SUBSET = {}
    for rs in sorted(both):
        run, sub = rs.split('_stat090_')
        SUBSET.setdefault(run, []).append(f'stat090_{sub}')
    OUT = Path(__file__).resolve().parent / 'results' / f'fullpass_{version}'
    CACHE = ROOT / f'fullpass_{version}_compare'
    CHAINS = {'prod': PROD, 'pilot': PILOT}
    print(f'{version}: {len(both)} sub-runs in {len(SUBSET)} runs common to both chains')


def _subs(run):
    return SUBSET[run]


def match():
    for run in sorted(SUBSET, key=lambda r: int(r.split('_')[1])):
        for chain, src in CHAINS.items():
            for arm in ARMS:
                f = CACHE / chain / f'scint_{run}_{arm}.parquet'
                if f.exists():
                    continue
                f.parent.mkdir(parents=True, exist_ok=True)
                try:
                    M, _ = DS.match_run(run, _subs(run), src, None, DS.layer_geometry(run, arm))
                except Exception as e:  # noqa: BLE001 -- a run without k is reported, not fatal
                    print(f'  !! {run} {chain} {arm}: {str(e).splitlines()[0][:120]}')
                    continue
                M.to_parquet(f, index=False)
                print(f'  {run} {chain} {arm}: {len(M):,} matched rows', flush=True)


COLS = ['arm', 'gated', 'x_t0', 'y_t0', 'tanx', 'tany', 'angle_calibrated']


def summary():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted(SUBSET, key=lambda r: int(r.split('_')[1])):
        for chain, src in CHAINS.items():
            T = pd.concat([pd.read_parquet(src / f'tracks_{run}_{s}.parquet',
                                           columns=COLS + (['gated_reco', 'in_acceptance'] if chain == 'pilot' else []))
                           for s in _subs(run)], ignore_index=True)
            for arm in ARMS:
                g = T[T.arm == arm]
                G = g[g.gated]
                tt = np.maximum(G.tanx.abs(), G.tany.abs())
                r = dict(run=run, arm=arm, chain=chain, gated=len(G),
                         gated_reco=int(g.gated_reco.sum()) if chain == 'pilot' else len(G),
                         gated_acc=int((tt < ACC_TRUE).sum()),
                         mirror_zone=int(((tt >= 0.3) & (tt < ACC_TRUE)).sum()),
                         late_frac=float((np.maximum(G.x_t0, G.y_t0) > LATE_NS).mean()) if len(G) else np.nan,
                         calibrated=bool(g.angle_calibrated.astype('boolean').fillna(False).any()))
                f = CACHE / chain / f'scint_{run}_{arm}.parquet'
                if f.exists():
                    S = pd.read_parquet(f)
                    w = S[S.on_wall.astype(bool)]   # an empty object column would select columns, not rows
                    wa = w[np.maximum(w.tanx.abs(), w.tany.abs()) < ACC_TRUE]
                    r.update(on_wall=len(w),
                             confirmed=int(w.match_wall.sum() - w.match_wall_ctrl.sum()),
                             confirmed_acc=int(wa.match_wall.sum() - wa.match_wall_ctrl.sum()),
                             wall_rate=float(w.match_wall.mean()) if len(w) else np.nan,
                             wall_ctrl=float(w.match_wall_ctrl.mean()) if len(w) else np.nan)
                rows.append(r)
    R = pd.DataFrame(rows)
    P = R[R.chain == 'prod'].set_index(['run', 'arm'])
    idx = R.set_index(['run', 'arm']).index
    for c in ('gated', 'gated_acc', 'confirmed', 'confirmed_acc'):
        if c in R:
            R[f'{c}_vs_prod'] = R[c].values / idx.map(P[c]).values.astype(float)
    R.to_csv(OUT / 'pilot_compare.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 500):
        print(R[R.chain == 'pilot'].round(3).to_string(index=False))
        tot = R.groupby(['arm', 'chain'])[['gated', 'gated_reco', 'gated_acc', 'mirror_zone',
                                           'confirmed', 'confirmed_acc']].sum()
        print('\n', tot)
    return R


# Runs whose wall rate dips in BOTH chains (a scintillator condition, not the
# reco): kept out of the per-angle confirmation so they do not dilute a bin.
WALL_DIP = {'run_110', 'run_114', 'run_132', 'run_162', 'run_126'}
TAN_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]


def bins():
    """Net wall confirmation (match - off-time control) per |true tan| bin,
    per view, over the run/arms both chains can extrapolate.  A mirrored fit
    sends the extrapolation to the wrong tile, so mirrors show up here as a
    loss at 0.3-0.6.  Caveat: the chains' k differ (~0.95 vs 1.23 on A), so a
    true-tan bin holds different raw-angle tracks in each."""
    rows = []
    for arm in ARMS:
        have = {c: {f.stem.split('scint_')[1].rsplit('_', 1)[0]
                    for f in (CACHE / c).glob(f'scint_*_{arm}.parquet')} for c in CHAINS}
        runs = sorted((have['prod'] & have['pilot']) - WALL_DIP)
        for chain in CHAINS:
            S = pd.concat([pd.read_parquet(CACHE / chain / f'scint_{r}_{arm}.parquet') for r in runs])
            S = S[S.on_wall.astype(bool)]
            for v in ('tanx', 'tany'):
                b = pd.cut(S[v].abs(), TAN_BINS)
                for iv, g in S.groupby(b, observed=True):
                    rows.append(dict(arm=arm, view=v, chain=chain, lo=iv.left, hi=iv.right, n=len(g),
                                     n_runs=len(runs), net=float(g.match_wall.mean() - g.match_wall_ctrl.mean())))
    B = pd.DataFrame(rows)
    B.to_csv(OUT / 'confirm_by_tan.csv', index=False)
    print(B.pivot_table(index=['arm', 'view', 'lo'], columns='chain', values=['n', 'net']).round(3))
    return B


INSITU = Path.home() / 'scratch' / 'ntof_insitu'


def cosmic():
    """Track by track on the A-C cosmic truth: is2w (one-sided) against is2ts
    (two-sided), 0.2 <= |true tan| < 0.5.  For the tracks the search changes
    (|d tan_raw| > 0.01): chi2 and t0 change, and raw/true before and after."""
    T = pd.read_parquet(INSITU / 'truth.parquet')
    T = T[~T.train]
    rows = []
    for arm in ARMS:
        W = pd.read_parquet(INSITU / f'reco_is2w_s3_{arm}.parquet')
        S = pd.read_parquet(INSITU / f'reco_is2ts_s3_{arm}.parquet')
        M = T[T.arm == arm].merge(W, on=['subrun', 'event_id']).merge(
            S, on=['subrun', 'event_id'], suffixes=('_w', '_s'))
        for ax in 'xy':
            t, w, s = M[f'tan_{ax}'], M[f'{ax}_tan_theta_w'], M[f'{ax}_tan_theta_s']
            m = (t.abs() >= 0.2) & (t.abs() < 0.5) & np.isfinite(w) & np.isfinite(s)
            ch = m & ((s - w).abs() > 0.01)
            dc = (M[f'{ax}_chi2_s'] - M[f'{ax}_chi2_w'])[ch]
            dt = (M[f'{ax}_t0_s'] - M[f'{ax}_t0_w'])[ch]
            ew, es = (w - t).abs()[ch], (s - t).abs()[ch]
            rows.append(dict(arm=arm, view=ax, n=int(m.sum()), changed=int(ch.sum()),
                             sign_flips=int((np.sign(s) != np.sign(w))[ch].sum()),
                             dchi2_med=float(dc.median()), frac_chi2_lower=float((dc < 0).mean()),
                             dt0_med_ns=float(dt.median()),
                             ratio_truth_w=float((w / t)[ch].median()), ratio_truth_s=float((s / t)[ch].median()),
                             abs_err_w=float(ew.median()), abs_err_s=float(es.median()),
                             frac_closer=float((es < ew).mean())))
    C = pd.DataFrame(rows)
    C.to_csv(OUT / 'cosmic_two_sided_changed.csv', index=False)
    print(C.round(3).to_string(index=False))
    return C


RR = Path(__file__).resolve().parent / 'results' / 'repass_readiness'


def valley():
    """Synthetic steep muons (mirror_chi2.py): right-sign two-sided fits
    against the refit from the true side.  Where t0 disagrees by > 30 ns, does
    it cost chi2 or angle?  (A flat p0-t0 valley costs neither.)"""
    rows = []
    for arm in ARMS:
        d = pd.read_parquet(RR / f'mirror_ts_{arm}.parquet')
        d = d[(np.sign(d.ts_raw) == np.sign(d.tan_u)) & (d.tan_u.abs() > 0.3)]
        ddt = d.ts_t0 - d.t0_right
        big = d[ddt.abs() > 30]
        dc = big.ts_chi2 - big.chi2_right
        rows.append(dict(arm=arm, n=len(d), n_t0_off_30ns=len(big), frac=len(big) / len(d),
                         dchi2_med=float(dc.median()), dchi2_p90=float(dc.quantile(0.9)),
                         dof_med=float(big.dof.median()), frac_earlier=float((ddt[ddt.abs() > 30] < 0).mean()),
                         raw_ratio_ts_over_right=float((big.ts_raw / big.raw_right).median()),
                         dp0_med_mm=float((big.ts_p0 - big.p0_right).abs().median())))
    V = pd.DataFrame(rows)
    V.to_csv(OUT / 'synthetic_t0_valley.csv', index=False)
    print(V.round(3).to_string(index=False))
    return V


def same():
    """The same physical track in both chains (single-track events, track
    point within 3 mm, both on the wall): wall confirmation with each chain's
    own direction, binned by the PILOT's |true tan|.  This removes the k
    difference from the per-angle comparison (pilot/prod true tan on matched
    tracks: A 0.85, C 0.90 x / 1.01 y) and asks which scale points at the
    tile that fired."""
    rows = []
    for arm in ARMS:
        parts = []
        for f in sorted((CACHE / 'prod').glob(f'scint_*_{arm}.parquet')):
            run = f.stem.split('scint_')[1].rsplit('_', 1)[0]
            if run in WALL_DIP:
                continue
            p = pd.read_parquet(f)
            n = pd.read_parquet(CACHE / 'pilot' / f.name)
            m = p[p.n_trk == 1].merge(n[n.n_trk == 1], on=['run', 'subrun', 'event_id'], suffixes=('_p', '_n'))
            m = m[((m.u_mm_p - m.u_mm_n).abs() < 3) & ((m.v_mm_p - m.v_mm_n).abs() < 3)
                  & m.on_wall_p.astype(bool) & m.on_wall_n.astype(bool)]
            parts.append(m)
        M = pd.concat(parts)
        t = np.maximum(M.tanx_n.abs(), M.tany_n.abs())
        for iv, d in M.groupby(pd.cut(t, TAN_BINS[:7]), observed=True):
            rows.append(dict(arm=arm, lo=iv.left, hi=iv.right, n=len(d),
                             scale_x=float((d.tanx_n / d.tanx_p).median()),
                             scale_y=float((d.tany_n / d.tany_p).median()),
                             net_prod=float(d.match_wall_p.mean() - d.match_wall_ctrl_p.mean()),
                             net_pilot=float(d.match_wall_n.mean() - d.match_wall_ctrl_n.mean()),
                             only_prod=float((d.match_wall_p & ~d.match_wall_n).mean()),
                             only_pilot=float((d.match_wall_n & ~d.match_wall_p).mean())))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'same_track_confirm.csv', index=False)
    print(S.round(3).to_string(index=False))
    return S


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('match', 'summary', 'bins', 'cosmic', 'valley', 'same'))
    ap.add_argument('--version', default=None,
                    help='compare a full re-pass (stage3_<version>, e.g. is2_v2) instead of the is2_v1 pilot')
    a = ap.parse_args()
    if a.version:
        _configure(a.version)
    {'match': match, 'summary': summary, 'bins': bins, 'cosmic': cosmic, 'valley': valley, 'same': same}[a.step]()
