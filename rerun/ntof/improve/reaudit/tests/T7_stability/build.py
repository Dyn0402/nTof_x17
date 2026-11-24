#!/usr/bin/env python3
"""T7_stability build: one pass over stage3_r2_final, per (sub-run, arm) statistics + small per-sub-run caches.

Outputs (all under this directory):
  cache/stats/<run>_<sub>.parquet   one row per arm (the stability table rows)
  cache/trk/<run>_<sub>.parquet     late gated tracks (A, B, C, D) with the few columns needed later (crossing, maps)
  cache/occ/<run>_<sub>.npz         x_local / y_local occupancy histograms per arm (late gated, and all tracks)

Run from the repo root:  nice -n 10 .venv/bin/python <this> [--workers 3]
"""
import argparse, csv, json, os, sys, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = Path('/home/dylan/PycharmProjects/nTof_x17')
sys.path[:0] = [str(REPO / 'rerun/ntof'), str(REPO / 'rerun/ntof/stage3'), str(REPO)]
WD = Path(os.environ.get('WORK_DATA', '/media/ucla/work'))
VIEW = WD / 'ntof_pass1/stage3_r2_final'
TRIG = WD / 'ntof_pass1/context_r2/trigtime'
OUT = Path(__file__).resolve().parent / 'cache'
LATE_NS = 25e6
EDGES = np.linspace(-200, 200, 101)          # 4 mm occupancy bins
COLS = ['run', 'subrun', 'tag', 'event_id', 'arm', 'track_id', 'gated', 't_since_flash_ns', 'x_local', 'y_local',
        'tanx', 'tany', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z', 'x_q_sum', 'y_q_sum', 'chi2dof_x', 'chi2dof_y',
        'x_n_strips', 'y_n_strips', 'x_t0', 'y_t0', 'x_ftst', 'y_ftst', 'dtc_ns', 't0_railed', 'low_charge',
        'censored_window', 'd_faint', 'x_split_child', 'y_split_child', 'wall_A', 'wall_B', 'wall_C', 'wall_D',
        'x_q_uend', 'y_q_uend', 'drift_t_end_ns', 'drift_railed', 'v_t', 'dca_source_mm', 'event_class']
QCOLS = ['x_q_sum', 'y_q_sum', 'chi2dof_x', 'chi2dof_y', 'x_n_strips', 'y_n_strips', 'x_dt0', 'y_dt0', 'dtc_ns',
         'x_q_uend', 'y_q_uend', 'drift_t_end_ns', 'v_t', 'gap_end', 'gap_xu']
VIEWAX = {'A': ('x', 'z'), 'C': ('x', 'z'), 'B': ('z', 'x'), 'D': ('z', 'x')}


def b(s):
    return s.astype('boolean').fillna(False).astype(bool).to_numpy()


def one(job):
    f, rows, canon, rclass = job
    key = f.stem.replace('tracks_run_', '')
    try:
        import quality_join as Q
        from sept26_prelim_analysis import capsule_tof as CT
        have = set(pq.read_schema(f).names)
        t = pq.read_table(f, columns=[c for c in COLS if c in have]).to_pandas()
        for c in COLS:
            if c not in t:
                t[c] = np.nan
        run, sub = str(t.run.iloc[0]), str(t.subrun.iloc[0])
        tags = sorted(t.tag.astype(str).unique())
        pins = Q._pins(rows, canon)
        for p in 'xy':
            ft = t[f'{p}_ftst'].fillna(-1).astype(int).to_numpy()
            pin = np.array([pins.get((tg, a), {}).get(p, {}).get(int(v), np.nan)
                            for tg, a, v in zip(t.tag, t.arm, ft)])
            t[f'{p}_dt0'] = t[f'{p}_t0'].to_numpy(float) - pin
        t['gap_end'] = t.v_t * t.drift_t_end_ns / 1000.0            # mm (v_t in um/ns)
        t['gap_xu'] = t.v_t * (t.x_q_uend - t.x_t0) / 1000.0
        g = b(t.gated)
        tm = t.t_since_flash_ns.to_numpy(float)
        timed = np.isfinite(tm)
        prod = rclass == 'production'
        late = (tm >= LATE_NS) if prod else np.ones(len(t), bool)
        # triggers
        ntrig = ntrig_late = np.nan
        tf = TRIG / f'trigtime_run_{run.replace("run_", "")}_{sub}.parquet'
        if prod and tf.exists():
            tt = pd.read_parquet(tf, columns=['eventId', 'is_flash', 'matched', 't_pred_ns'])
            nf = tt.is_flash.to_numpy() == 0
            ntrig = int(nf.sum())
            ntrig_late = int((nf & (tt.t_pred_ns.to_numpy() >= LATE_NS)).sum())
        meta = json.loads(f.with_suffix('.meta.json').read_text())
        out, trk, occ = [], [], {}
        for arm in 'ABCD':
            m = (t.arm == arm).to_numpy()
            a = t[m]
            ga, la, ta = g[m], late[m], timed[m]
            L = a[la]
            LG = a[la & ga]
            r = dict(run=run, subrun=sub, arm=arm, rclass=rclass, tag0=tags[0], n_tags=len(tags),
                     n_events_meta=meta.get('n_events'), n_trig=ntrig, n_trig_late=ntrig_late,
                     n_tracks=int(m.sum()), n_gated=int(ga.sum()), n_late=int(la.sum()), n_late_gated=len(LG),
                     frac_untimed=float(1 - ta.mean()) if prod and len(a) else np.nan,
                     frac_gated_late=len(LG) / max(len(L), 1))
            for c in ('t0_railed', 'low_charge', 'censored_window', 'd_faint', 'x_split_child', 'drift_railed'):
                r[f'fL_{c}'] = float(b(L[c]).mean()) if len(L) else np.nan          # among all late tracks
                r[f'fLG_{c}'] = float(b(LG[c]).mean()) if len(LG) else np.nan       # among late gated
            r['fLG_wall'] = float(b(LG[f'wall_{arm}']).mean()) if len(LG) else np.nan
            r['fLG_wall_known'] = float(LG[f'wall_{arm}'].notna().mean()) if len(LG) else np.nan
            d = LG.dca_source_mm.to_numpy(float)
            r['fLG_dca15'] = float(np.mean(d[np.isfinite(d)] < 15)) if np.isfinite(d).sum() > 20 else np.nan
            for c in QCOLS:
                v = LG[c].to_numpy(float)
                v = v[np.isfinite(v)]
                if len(v) >= 20:
                    q = np.percentile(v, [16, 50, 84])
                    r[f'{c}_p16'], r[f'{c}_med'], r[f'{c}_p84'] = q
                    r[f'{c}_mean'] = float(v.mean())
            # untrimmed means for the quantised times and gap: non-railed tracks only
            nr = LG[~b(LG.drift_railed)]
            for c in ('drift_t_end_ns', 'gap_end', 'x_q_uend', 'gap_xu'):
                v = nr[c].to_numpy(float)
                v = v[np.isfinite(v)]
                r[f'{c}_nrmean'] = float(v.mean()) if len(v) >= 20 else np.nan
                r[f'{c}_nrerr'] = float(v.std() / np.sqrt(len(v))) if len(v) >= 20 else np.nan
            for c in ('x_local', 'y_local', 'tanx', 'tany'):
                v = LG[c].to_numpy(float)
                v = v[np.isfinite(v)]
                r[f'{c}_mean'] = float(v.mean()) if len(v) >= 20 else np.nan
                r[f'{c}_rms'] = float(v.std()) if len(v) >= 20 else np.nan
            # same-chamber two-track fraction (late gated, t0 <= 300 both views; same_chamber_compare cuts)
            sc = LG[(LG.x_t0 <= 300) & (LG.y_t0 <= 300)]
            if len(sc):
                ne = sc.groupby('event_id').size()
                r['n_ev_1p'] = int(len(ne)); r['frac_2trk'] = float((ne >= 2).mean())
            # late triggers with >= 1 late gated / >= 1 late track in this arm
            r['n_ev_late_gated'] = int(LG.event_id.nunique())
            r['n_ev_late_any'] = int(L.event_id.nunique())
            # source crossing (pointing), late gated, t0 <= 300
            if arm != 'B' and len(sc) > 100:
                c_, n_ = VIEWAX[arm]
                P = {k: sc[f'p0_{k}'].to_numpy(float) for k in 'xyz'}
                D = {k: sc[f'd_{k}'].to_numpy(float) for k in 'xyz'}
                with np.errstate(all='ignore'):
                    tt_ = -P[n_] / D[n_]
                    pos, disp, hgt = P[c_], tt_ * D[c_], P['y'] + tt_ * D['y']
                ok = np.isfinite(pos) & np.isfinite(disp) & np.isfinite(hgt)
                sel = ok & (np.abs(sc.tanx.to_numpy(float) if arm != 'D' else sc.tanx.to_numpy(float)) >= 0.0)
                try:
                    x0, e0, n0 = CT.crossing(pos[sel], disp[sel])
                except Exception:
                    x0, e0, n0 = np.nan, np.nan, 0
                r['xcross'], r['xcross_err'], r['xcross_n'] = x0, e0, n0
                # y band crossing (height): disp_y vs global y position
                with np.errstate(all='ignore'):
                    dy = tt_ * D['y']
                oky = ok & np.isfinite(dy)
                try:
                    y0, ey, ny = CT.crossing(P['y'][oky], dy[oky])
                except Exception:
                    y0, ey, ny = np.nan, np.nan, 0
                r['ycross'], r['ycross_err'] = y0, ey
                s = sc[ok].copy()
                s['pos'], s['disp'], s['hgt'] = pos[ok], disp[ok], hgt[ok]
            else:
                s = sc.copy()
                s['pos'] = s['disp'] = s['hgt'] = np.nan
            s['ms'] = s.t_since_flash_ns / 1e6
            s['wall'] = b(s[f'wall_{arm}'])
            trk.append(s[['run', 'subrun', 'tag', 'event_id', 'arm', 'x_local', 'y_local', 'tanx', 'tany', 'pos', 'disp',
                          'hgt', 'ms', 'wall', 'dca_source_mm', 'x_dt0', 'y_dt0', 'x_q_sum', 'y_q_sum']]
                       .astype({c: 'float32' for c in ['x_local', 'y_local', 'tanx', 'tany', 'pos', 'disp', 'hgt',
                                                       'ms', 'dca_source_mm', 'x_dt0', 'y_dt0', 'x_q_sum',
                                                       'y_q_sum']}))
            for nm, sub_ in (('lg', LG), ('all', a), ('g', a[ga])):
                for p in ('x_local', 'y_local'):
                    occ[f'{arm}_{nm}_{p}'] = np.histogram(sub_[p].to_numpy(float), EDGES)[0].astype('int32')
            out.append(r)
        S = pd.DataFrame(out)
        (OUT / 'stats').mkdir(parents=True, exist_ok=True)
        (OUT / 'trk').mkdir(parents=True, exist_ok=True)
        (OUT / 'occ').mkdir(parents=True, exist_ok=True)
        S.to_parquet(OUT / 'stats' / f'{key}.parquet', index=False)
        pd.concat(trk, ignore_index=True).to_parquet(OUT / 'trk' / f'{key}.parquet', index=False)
        np.savez_compressed(OUT / 'occ' / f'{key}.npz', **occ)
        return key, None
    except Exception:
        return key, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    import ntof_ledger as L
    canon = L.load_canon(str(REPO / 'rerun/ntof/ledger_canon.json'))
    reg = [r for r in csv.DictReader(open(REPO / 'rerun/ntof/ntof_registry.csv')) if r['excluded'] != 'True']
    groups = {}
    for r in reg:
        groups.setdefault((r['run'], r['subrun']), []).append(r)
    jobs = []
    for (run, sub), rows in groups.items():
        f = VIEW / f'tracks_run_{run}_{sub}.parquet'
        key = f'{run}_{sub}'
        if a.only and a.only not in key:
            continue
        if f.exists() and not (OUT / 'stats' / f'{key}.parquet').exists():
            jobs.append((f, rows, canon, rows[0]['rclass']))
    missing = sorted(set(p.stem.replace('tracks_run_', '') for p in VIEW.glob('tracks_run_*.parquet'))
                     - {f'{r}_{s}' for r, s in groups})
    print(len(jobs), 'jobs;', 'view tables without registry rows:', missing, flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (k, err) in enumerate(ex.map(one, jobs, chunksize=1)):
            if err:
                print('FAIL', k, err, flush=True)
            elif i % 25 == 0:
                print(i, k, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
