#!/usr/bin/env python3
"""compare.py -- compare the n_TOF tracks of a phase-4 test configuration with a reference, acquisition by acquisition.

Both inputs are analysis-view directories (tracks_run_<run>_<subrun>.parquet, the output of stage3_chain.sh, or
stage3_r2_final itself).  Only the acquisitions (run, subrun, tag, arm) present in the CONFIGURATION are compared, and
the reference is cut to the same acquisitions, so a test that covers part of a sub-run (a sizing cluster) compares
like with like.

Per chamber, and per time-of-flight bin (TOF_BINS_MS, t_since_flash_ns) where it makes sense, it reports:

  counts       gated tracks in each set and gated tracks per trigger (triggers per tag and ToF bin from the stage-1
               context and the trigger-time product, the same that stage 3 attaches)
  matching     events (same run, subrun, tag, arm, event_id) with gated tracks in both sets, only in the configuration,
               only in the reference.  Inside an event the tracks are paired by nearest (x_local, y_local); the
               unpaired extras of an event count as "only" tracks.
  differences  paired tracks: config - reference of x_local, y_local, tanx, tany, x_t0, y_t0: median, half-width of
               the central 68 % and of the central 95 %, the fraction differing by more than --tol, and the maximum
  flags        fraction t0_railed (all gated, and split children x_split_child | y_split_child); low_charge, d_faint,
               censored_window among tracks in acceptance before those cuts (gated_uncensored)
  pairs        events with >= 2 gated tracks in one chamber, per event with >= 1 (the same-chamber pair fraction)
  pointing     late gated tracks (>= --late-ms): fraction with dca_source_mm < 15 mm, and the pointing half-width:
               the 68 % quantile of dca_source_mm among tracks within 45 mm (track_quality.R1/R2)
  wall         wall_<arm> fraction of the paired tracks, and of the tracks that only one set has (says whether added
               tracks are real particles: a real track fires the wall behind it at the paired-track rate)

    python compare.py --cfg DIR --ref $WORK_DATA/ntof_pass1/stage3_r2_final --out DIR [--name all] [--tol 1e-4]

Writes <out>/<name>_vs_ref_summary.txt (the readable tables), <name>_vs_ref_bins.csv (all numbers per chamber and
bin), and <name>_vs_ref_diffrows.csv (paired tracks whose differences exceed --tol, for a reproduction check).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get('NTOF_X17_REPO', '/home/dylan/PycharmProjects/nTof_x17'))
sys.path[:0] = [str(REPO)]

TOF_BINS_MS = [(1, 3), (3, 5), (5, 8), (8, 12), (12, 16), (16, 25), (25, 75)]
KEY = ['run', 'subrun', 'tag', 'arm', 'event_id']
DIFF_COLS = ['x_local', 'y_local', 'tanx', 'tany', 'x_t0', 'y_t0']
R1, R2 = 15.0, 45.0
COLS = sorted(set(KEY + DIFF_COLS + [
    'track_id', 'gated', 'gated_uncensored', 't_since_flash_ns', 't0_railed', 'x_split_child', 'y_split_child',
    'low_charge', 'd_faint', 'censored_window', 'dca_source_mm', 'wall_A', 'wall_B', 'wall_C', 'wall_D']))


# ----------------------------------------------------------------------------------------------------- input
def read_view(d: Path, keep: set | None = None) -> pd.DataFrame:
    """All tracks of a view directory (only the columns used), cut to the acquisitions in `keep` if given."""
    import pyarrow.parquet as pq
    parts = []
    for f in sorted(Path(d).glob('tracks_run_*.parquet')):
        run, sub = f.stem.split('_')[2], '_'.join(f.stem.split('_')[3:])
        if keep is not None and not any(k[0] == run and k[1] == sub for k in keep):
            continue
        names = pq.read_schema(f).names
        t = pq.read_table(f, columns=[c for c in COLS if c in names]).to_pandas()
        t['run'], t['subrun'] = run, sub
        for c in COLS:
            if c not in t:
                t[c] = np.nan
        parts.append(t)
    if not parts:
        raise SystemExit(f'no tables under {d}')
    t = pd.concat(parts, ignore_index=True)
    t['run'] = t['run'].astype(str).str.replace('run_', '', regex=False)
    if keep is not None:
        k = pd.DataFrame(sorted(keep), columns=['run', 'subrun', 'tag', 'arm'])
        t = t.merge(k, on=['run', 'subrun', 'tag', 'arm'], how='inner')
    for c in ('gated', 'gated_uncensored', 't0_railed', 'x_split_child', 'y_split_child', 'low_charge', 'd_faint',
              'censored_window', 'wall_A', 'wall_B', 'wall_C', 'wall_D'):
        t[c] = t[c].astype('boolean').fillna(False).astype(bool)
    t['ms'] = t['t_since_flash_ns'].astype(float) / 1e6
    t['wall'] = [w for w in _own_wall(t)]
    return t


def _own_wall(t):
    w = np.zeros(len(t), bool)
    for a in 'ABCD':
        m = (t['arm'] == a).to_numpy()
        w[m] = t.loc[m, f'wall_{a}'].to_numpy(bool)
    return w


def acquisitions(t: pd.DataFrame) -> set:
    return set(map(tuple, t[['run', 'subrun', 'tag', 'arm']].drop_duplicates().astype(str).to_numpy()))


def triggers(acq: set) -> pd.DataFrame:
    """Triggers per (run, subrun, tag, ToF bin) from the stage-1 context (tag, eventId) and the trigger-time product
    (t_pred_ns, is_flash), the inputs stage 3 attaches t_since_flash_ns from.  Empty if they are not found."""
    wd = Path(os.environ.get('WORK_DATA', '/media/ucla/work')) / 'ntof_pass1' / 'context_r2'
    os.environ.setdefault('X17_TRIGTIME_DIR', str(wd / 'trigtime'))
    from sept26_prelim_analysis import trigger_time as TT
    rows = []
    for run, sub in sorted({(a[0], a[1]) for a in acq}):
        s1 = wd / 'stage1' / f'candidates_run_{run}_{sub}.parquet'
        if not s1.exists():
            continue
        e = pd.read_parquet(s1, columns=['tag', 'eventId'])
        tt = TT.read(f'run_{run}', sub).set_index('eventId')
        tp = tt['t_pred_ns'].reindex(e['eventId'].to_numpy()).to_numpy()
        fl = tt['is_flash'].reindex(e['eventId'].to_numpy()).to_numpy()
        ms = TT.flash_time_and_energy(tp, np.nan_to_num(fl) == 1)[0] / 1e6
        tags = {a[2] for a in acq if a[0] == run and a[1] == sub}
        for lo, hi in TOF_BINS_MS + [(0, 1e9)]:
            m = (ms >= lo) & (ms < hi)
            c = e.loc[m].groupby('tag').size()
            for tag in tags:
                rows.append(dict(run=run, subrun=sub, tag=tag, lo=lo, hi=hi, n_trig=int(c.get(tag, 0))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- matching
def pair(cfg: pd.DataFrame, ref: pd.DataFrame):
    """(paired, only_cfg, only_ref): gated tracks matched by event, then by nearest (x_local, y_local) inside it."""
    a = cfg[cfg.gated].copy()
    b = ref[ref.gated].copy()
    a['_i'], b['_i'] = np.arange(len(a)), np.arange(len(b))
    ga = a.groupby(KEY)['_i'].apply(list)
    gb = b.groupby(KEY)['_i'].apply(list)
    ia, ib = [], []
    for k in ga.index.intersection(gb.index):
        la, lb = ga[k], gb[k]
        if len(la) == 1 and len(lb) == 1:
            ia.append(la[0]); ib.append(lb[0]); continue
        pa = a[['x_local', 'y_local']].to_numpy(float)[la]
        pb = b[['x_local', 'y_local']].to_numpy(float)[lb]
        d = np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1])
        d = np.where(np.isfinite(d), d, 1e9)
        used_a, used_b = set(), set()
        for flat in np.argsort(d, axis=None):
            i, j = np.unravel_index(flat, d.shape)
            if i in used_a or j in used_b:
                continue
            used_a.add(i); used_b.add(j); ia.append(la[i]); ib.append(lb[j])
            if len(used_a) == min(len(la), len(lb)):
                break
    A = a.iloc[ia].reset_index(drop=True)
    B = b.iloc[ib].reset_index(drop=True)
    paired = A.join(B[DIFF_COLS + ['track_id', 'wall', 'ms']].add_suffix('_ref'))
    only_a = a.loc[~a['_i'].isin(ia)]
    only_b = b.loc[~b['_i'].isin(ib)]
    return paired, only_a, only_b


# ------------------------------------------------------------------------------------------------------ stats
def _w(x, q):
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    lo, hi = np.quantile(x, [0.5 - q / 2, 0.5 + q / 2])
    return 0.5 * (hi - lo)


def diff_stats(p: pd.DataFrame, tol: float) -> dict:
    out = {}
    for c in DIFF_COLS:
        d = (p[c].astype(float) - p[f'{c}_ref'].astype(float)).to_numpy()
        f = np.isfinite(d)
        out[f'd_{c}_med'] = float(np.median(d[f])) if f.any() else np.nan
        out[f'd_{c}_w68'] = _w(d, 0.68)
        out[f'd_{c}_w95'] = _w(d, 0.95)
        out[f'd_{c}_fbig'] = float(np.mean(np.abs(d[f]) > tol)) if f.any() else np.nan
        out[f'd_{c}_max'] = float(np.max(np.abs(d[f]))) if f.any() else np.nan
    return out


def set_stats(t: pd.DataFrame, late_ms: float) -> dict:
    g = t[t.gated]
    pre = t[t.gated_uncensored]
    split = g[g.x_split_child | g.y_split_child]
    ev = g.groupby(KEY).size()
    late = g[g.ms >= late_ms]
    dca = late['dca_source_mm'].to_numpy(float)
    core = dca[dca < R2]
    return dict(
        n_gated=len(g), n_events=len(ev), pair_frac=float((ev >= 2).mean()) if len(ev) else np.nan,
        n_pair_events=int((ev >= 2).sum()),
        t0_railed=float(g.t0_railed.mean()) if len(g) else np.nan, n_split=len(split),
        t0_railed_split=float(split.t0_railed.mean()) if len(split) else np.nan,
        low_charge=float(pre.low_charge.mean()) if len(pre) else np.nan,
        d_faint=float(pre.d_faint.mean()) if len(pre) else np.nan,
        censored=float(pre.censored_window.mean()) if len(pre) else np.nan,
        wall=float(g.wall.mean()) if len(g) else np.nan,
        n_late=len(late), point15=float(np.mean(dca < R1)) if len(dca) else np.nan,
        point_hw=float(np.quantile(core, 0.68)) if len(core) > 20 else np.nan)


def compare(cfg: pd.DataFrame, ref: pd.DataFrame, trig: pd.DataFrame, late_ms: float, tol: float):
    rows, bad = [], []
    for arm in sorted(set(cfg.arm) | set(ref.arm)):
        for lo, hi in [(0, 1e9)] + TOF_BINS_MS:
            sel = lambda t: t[(t.arm == arm) & ((t.ms >= lo) & (t.ms < hi) if hi < 1e9 else True)]
            c, r = sel(cfg), sel(ref)
            p, oa, ob = pair(c, r)
            row = dict(arm=arm, bin='all' if hi >= 1e9 else f'{lo}-{hi}')
            for name, t in (('cfg', c), ('ref', r)):
                for k, v in set_stats(t, late_ms).items():
                    row[f'{name}_{k}'] = v
            if len(trig):
                acq = {(a[0], a[1], a[2]) for a in acquisitions(c) | acquisitions(r) if a[3] == arm}
                m = (trig.lo == lo) & (trig.hi == hi)
                nt = trig[m][trig[m][['run', 'subrun', 'tag']].apply(tuple, axis=1).isin(acq)].n_trig.sum()
                row['n_trig'] = int(nt)
                row['cfg_per_trig'] = row['cfg_n_gated'] / nt if nt else np.nan
                row['ref_per_trig'] = row['ref_n_gated'] / nt if nt else np.nan
            ec, er = set(map(tuple, c[c.gated][KEY].drop_duplicates().to_numpy())), \
                set(map(tuple, r[r.gated][KEY].drop_duplicates().to_numpy()))
            row.update(ev_both=len(ec & er), ev_only_cfg=len(ec - er), ev_only_ref=len(er - ec),
                       n_paired=len(p), n_only_cfg=len(oa), n_only_ref=len(ob),
                       matched_frac_ref=len(p) / max(row['ref_n_gated'], 1),
                       wall_paired=float(p.wall.mean()) if len(p) else np.nan,
                       wall_only_cfg=float(oa.wall.mean()) if len(oa) else np.nan,
                       wall_only_ref=float(ob.wall.mean()) if len(ob) else np.nan,
                       only_cfg_headon=float((oa.tanx.abs() < 0.08).mean()) if len(oa) else np.nan,
                       only_ref_headon=float((ob.tanx.abs() < 0.08).mean()) if len(ob) else np.nan)
            row.update(diff_stats(p, tol))
            rows.append(row)
            if hi >= 1e9 and len(p):
                d = np.zeros(len(p), bool)
                for col in DIFF_COLS:
                    x = (p[col].astype(float) - p[f'{col}_ref'].astype(float)).abs().to_numpy()
                    d |= np.nan_to_num(x, nan=0.0) > tol
                    d |= np.isfinite(p[col].astype(float)).to_numpy() != np.isfinite(p[f'{col}_ref'].astype(float)).to_numpy()
                bad.append(p[d])
    return pd.DataFrame(rows), (pd.concat(bad, ignore_index=True) if bad else pd.DataFrame())


# ------------------------------------------------------------------------------------------------------ output
def summary(R: pd.DataFrame, name: str, late_ms: float, tol: float) -> str:
    L = [f'{name} vs ref.  Gated tracks; paired = same event, nearest position.  Differences config - ref '
         f'(median / 68 % half-width / 95 % half-width; f>tol = fraction with |d| > {tol:g}).', '']
    f = lambda v, p=3: '   -' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{p}f}'
    for arm, g in R.groupby('arm'):
        L.append(f'=== chamber {arm}')
        L.append(f"{'ToF ms':>7} {'n_trig':>7} {'cfg':>6} {'ref':>6} {'cfg/trg':>8} {'ref/trg':>8} {'paired':>6} "
                 f"{'onlyC':>6} {'onlyR':>6} {'ev C':>5} {'ev R':>5} {'wallP':>6} {'wallC':>6} {'wallR':>6} "
                 f"{'hoC':>5} {'pairC':>6} {'pairR':>6}")
        for _, r in g.iterrows():
            L.append(f"{r['bin']:>7} {r.get('n_trig', 0):>7} {r.cfg_n_gated:>6} {r.ref_n_gated:>6} "
                     f"{f(r.get('cfg_per_trig'), 4):>8} {f(r.get('ref_per_trig'), 4):>8} {r.n_paired:>6} "
                     f"{r.n_only_cfg:>6} {r.n_only_ref:>6} {r.ev_only_cfg:>5} {r.ev_only_ref:>5} "
                     f"{f(r.wall_paired):>6} {f(r.wall_only_cfg):>6} {f(r.wall_only_ref):>6} "
                     f"{f(r.only_cfg_headon, 2):>5} {f(r.cfg_pair_frac, 4):>6} {f(r.ref_pair_frac, 4):>6}")
        a = g[g['bin'] == 'all'].iloc[0]
        L.append('  differences (all ToF):')
        for c in DIFF_COLS:
            L.append(f"    {c:>8}: med {a[f'd_{c}_med']:+.4g}  w68 {a[f'd_{c}_w68']:.4g}  w95 {a[f'd_{c}_w95']:.4g}  "
                     f"f>tol {a[f'd_{c}_fbig']:.4f}  max {a[f'd_{c}_max']:.4g}")
        L.append(f"  flags cfg/ref: t0_railed {f(a.cfg_t0_railed, 4)}/{f(a.ref_t0_railed, 4)};  split children "
                 f"t0_railed {f(a.cfg_t0_railed_split)}/{f(a.ref_t0_railed_split)} (n {a.cfg_n_split}/{a.ref_n_split});"
                 f"  low_charge {f(a.cfg_low_charge, 4)}/{f(a.ref_low_charge, 4)};  d_faint {f(a.cfg_d_faint, 4)}/"
                 f"{f(a.ref_d_faint, 4)};  censored_window {f(a.cfg_censored, 4)}/{f(a.ref_censored, 4)}")
        L.append(f"  pointing (>= {late_ms:g} ms) cfg/ref: n {a.cfg_n_late}/{a.ref_n_late};  dca<15 "
                 f"{f(a.cfg_point15)}/{f(a.ref_point15)};  half-width (q68 within 45 mm) {f(a.cfg_point_hw, 2)}/"
                 f"{f(a.ref_point_hw, 2)} mm")
        L.append(f"  same-chamber pair events cfg/ref: {a.cfg_n_pair_events}/{a.ref_n_pair_events} "
                 f"({f(a.cfg_pair_frac, 4)}/{f(a.ref_pair_frac, 4)} of events)")
        L.append('')
    L.append('Columns: onlyC/onlyR = tracks only in the config / reference; ev C/R = events with gated tracks only in '
             'one set; wallP/C/R = wall fraction of paired / only-config / only-ref tracks; hoC = fraction of the '
             'only-config tracks at |tanx| < 0.08; pairC/R = events with >= 2 gated tracks in the chamber.')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True, type=Path, help='analysis view of the configuration')
    ap.add_argument('--ref', required=True, type=Path, help='analysis view of the reference (e.g. stage3_r2_final)')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--name', default=None, help='configuration name for the output files (default: cfg dir name)')
    ap.add_argument('--late-ms', type=float, default=25.0, help='pointing sample: ToF >= this [ms]')
    ap.add_argument('--tol', type=float, default=1e-4, help='a paired difference above this counts as different')
    ap.add_argument('--arms', default='ACD')
    a = ap.parse_args()
    name = a.name or a.cfg.parent.name
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = read_view(a.cfg)
    cfg = cfg[cfg.arm.isin(list(a.arms))]
    acq = acquisitions(cfg)
    ref = read_view(a.ref, keep=acq)
    print(f'{len(acq)} acquisitions; cfg {len(cfg)} tracks ({int(cfg.gated.sum())} gated), ref {len(ref)} '
          f'({int(ref.gated.sum())} gated)', flush=True)
    for k in sorted(acq):
        print('  ', ' '.join(k))
    trig = triggers(acq)
    R, bad = compare(cfg, ref, trig, a.late_ms, a.tol)
    R.to_csv(a.out / f'{name}_vs_ref_bins.csv', index=False)
    bad.to_csv(a.out / f'{name}_vs_ref_diffrows.csv', index=False)
    txt = summary(R, name, a.late_ms, a.tol)
    (a.out / f'{name}_vs_ref_summary.txt').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
