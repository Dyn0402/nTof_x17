#!/usr/bin/env python3
"""window_charge.py -- does each pass-1 track keep its charge once the ringing is removed from the waveforms?

    python -m rerun.ntof.ringing.window_charge measure [--subruns run_145/stat090_0000,...] [--jobs 6]
    python -m rerun.ntof.ringing.window_charge eval

For every gated A/C track of the sub-runs (the pass-1 analysis view), and for each view, we take the channels around
the fitted track: |position - p0| <= |tan| x GAP + PAD strips.  On the waveform of that event we compute, once with the
pass-1 processing (pedestal, plain 64-channel block median) and once with the ringing removed (deflect.py, every
event):

    n_sig    channels in the window whose peak is above 5 sigma
    q_peak   the sum of those channels' peaks [ADC]

A track made of ionisation keeps its peaks (deflect.py preserves signal peaks to 1.000, median).  A track fitted on a
ringing lobe loses them.  We then cut on the corrected charge, min over views of q_peak(corrected) divided by the
run's late reference value, and evaluate it exactly like the charge cuts (explore6.py): every rule is set to keep the
same share of reference tracks (>= 30 ms, wall-confirmed, within 15 mm of the source), then per ToF bin the kept
fraction, the pointing-excess efficiency and the purity.
Out: $WORK_DATA/ntof_pass1/ana/ringing/wq_<run>_<sub>.parquet, wq_eval.csv, wq_angle.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
WD = os.environ.get('WORK_DATA', '/media/ucla/work')
OUT = Path(WD) / 'ntof_pass1' / 'ana' / 'ringing'
VIEW = Path(WD) / 'ntof_pass1' / 'stage3_vt'
WF_ROOTS = [f'{WD}/ntof_pass1/runs/', f'{WD}/claude-jobs/bcad1a92/wf/']
FEUS = {'A': (3, 4), 'C': (7, 8)}
SUBRUNS = ('run_145/stat090_0000', 'run_145/stat090_0001', 'run_145/stat090_0002', 'run_104/stat090_0003',
           'run_139/stat090_0003')
GAP = 28.0            # mm, drift gap (A 28.08, C similar): the column length is |tan| x GAP
PAD = 3               # strips on each side, as the pass-1 windows (pad_strips)
COLS = ['run', 'subrun', 'tag', 'event_id', 'arm', 'track_id', 'gated', 'x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta',
        'x_n_strips', 'y_n_strips', 'x_q_sum', 'y_q_sum', 't_since_flash_ns', 'p0_x', 'p0_z', 'd_x', 'd_z', 'tanx',
        'tany', 'wall_A', 'wall_C']
EDG = [1, 3, 5, 8, 12, 16, 20, 30, 75]
TB = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def pos_map(run, sub, feu):
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    base = next(b for b in WF_ROOTS if os.path.isdir(f'{b}{run}/{sub}/decoded_root'))
    arm = next(a for a, f in FEUS.items() if feu in f)
    fx, fy = FEUS[arm]
    cfg = wb.BeamConfig(KEY='x', RUN=run, SUB_RUN=sub, DET_NAME=f'mx17_{arm}', MX17_FEU_X=fx, MX17_FEU_Y=fy,
                        BASE_PATH=next(b for b in WF_ROOTS if os.path.isfile(f'{b}{run}/run_config.json')),
                        OUT_BASE='/tmp')
    return wio.strip_position_map(cfg)[feu], base


def measure_file(args):
    import pandas as pd
    from wft import io as wio
    from rerun.ntof.ringing import deflect as DF
    run, sub, arm, plane, feu, path, trk = args
    pos, _ = pos_map(run, sub, feu)
    pitch = float(np.nanmedian(np.abs(np.diff(pos))))
    rdr = wio.FeuReader(path)
    nz = np.maximum(rdr.noise.astype(np.float64), 1.0)
    ids = set(trk.event_id.astype(int))
    want = np.where(np.isin(rdr.event_ids, np.fromiter(ids, dtype=np.int64)))[0]
    by_ev = {e: g for e, g in trk.groupby('event_id')}
    rows = []
    for lo in range(0, len(want), wio.CHUNK):
        block = want[lo:lo + wio.CHUNK]
        arr = rdr.tree.arrays(['eventId', 'amplitude'], entry_start=int(block[0]), entry_stop=int(block[-1]) + 1,
                              library='np')
        base = int(block[0])
        for i in block:
            a = arr['amplitude'][i - base]
            if len(a) != rdr.n_sample * 512:
                continue
            eid = int(arr['eventId'][i - base])
            raw = a.reshape(-1, 512).astype(np.float32) - rdr.ped
            P = DF.plain_cm(raw).T                       # (512, ns)
            C = DF.deflect(raw, rdr.noise).T
            n_neg = int(((P / nz[:, None]).min(axis=1) < -5).sum())
            pk_p, pk_c = P.max(axis=1), C.max(axis=1)
            for r in by_ev[eid].itertuples():
                p0, t = getattr(r, f'{plane}_p0'), abs(getattr(r, f'{plane}_tan_theta'))
                if not np.isfinite(p0) or not np.isfinite(t):
                    continue
                win = np.abs(pos - p0) <= min(t, 1.0) * GAP + PAD * pitch
                sp = win & (pk_p > 5 * nz)
                sc = win & (pk_c > 5 * nz)
                rows.append((r.tag, eid, arm, r.track_id, plane, int(win.sum()), int(sp.sum()), int(sc.sum()),
                             float(pk_p[sp].sum()), float(pk_c[sc].sum()), n_neg))
    return pd.DataFrame(rows, columns=['tag', 'event_id', 'arm', 'track_id', 'plane', 'n_win', 'n_sig_plain',
                                       'n_sig_corr', 'q_plain', 'q_corr', 'n_neg5'])


def measure(subruns, jobs):
    import pandas as pd
    from wft import io as wio
    for rs in subruns:
        run, sub = rs.split('/')
        t = pd.read_parquet(VIEW / f'tracks_{run}_{sub}.parquet', columns=COLS)
        t = t[t.arm.isin(list(FEUS)) & t.gated.fillna(False).astype(bool)]
        work = []
        for arm, (fx, fy) in FEUS.items():
            for plane, feu in (('x', fx), ('y', fy)):
                _, base = pos_map(run, sub, feu)
                for f in wio.subrun_files(base, run, sub, feu):
                    tag = wio.file_tag(f)
                    trk = t[(t.arm == arm) & (t.tag == tag)]
                    if len(trk):
                        work.append((run, sub, arm, plane, feu, f, trk))
        with ProcessPoolExecutor(jobs) as ex:
            parts = list(ex.map(measure_file, work))
        d = pd.concat(parts, ignore_index=True)
        w = d.pivot_table(index=['tag', 'event_id', 'arm', 'track_id'], columns='plane',
                          values=['n_win', 'n_sig_plain', 'n_sig_corr', 'q_plain', 'q_corr', 'n_neg5'], aggfunc='first')
        w.columns = [f'{p}_{v}' for v, p in w.columns]
        m = t.merge(w.reset_index(), on=['tag', 'event_id', 'arm', 'track_id'], how='inner')
        m.to_parquet(OUT / f'wq_{run}_{sub}.parquet', index=False)
        print(f'{rs}: {len(m)} tracks measured of {len(t)}', flush=True)


def evaluate():
    import pandas as pd
    from sept26_prelim_analysis.capsule_tof import crossing
    d = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob('wq_run_*.parquet'))], ignore_index=True)
    d = d[d.t_since_flash_ns.notna()].copy()
    d['ms'] = d.t_since_flash_ns / 1e6
    d['pos'], d['disp'] = d.p0_x, -d.p0_z / d.d_z * d.d_x
    d['qmin'] = np.minimum(d.x_q_sum, d.y_q_sum)
    d['wall'] = np.where(d.arm == 'A', d.wall_A.astype('boolean').fillna(False), d.wall_C.astype('boolean').fillna(False))
    d['qc'] = np.minimum(d.x_q_corr, d.y_q_corr)
    d['qp'] = np.minimum(d.x_q_plain, d.y_q_plain)
    d['keepfrac'] = np.minimum(d.x_q_corr / d.x_q_plain.replace(0, np.nan), d.y_q_corr / d.y_q_plain.replace(0, np.nan))
    d['ncs'] = np.minimum(d.x_n_sig_corr, d.y_n_sig_corr)
    rng = np.random.default_rng(0)
    rows, ang = [], []
    for arm in FEUS:
        g = d[d.arm == arm].copy()
        late = g[(g.ms >= 20) & (g.qmin >= 600)]
        u0, _, _ = crossing(late.pos.to_numpy(), late.disp.to_numpy())
        g['miss'] = g.pos + g.disp - u0
        g['ref'] = (g.ms >= 30) & g.wall & (g.miss.abs() < 15)
        ref = g.ref.to_numpy()

        def exc(s):
            if len(s) < 30:
                return np.nan
            m = np.abs(s.miss.to_numpy())
            sh = np.abs(s.pos.to_numpy() + rng.permutation(s.disp.to_numpy()) - u0)
            return float((m < 15).mean() - (sh < 15).mean())
        exl = exc(g[g.ms >= 30])
        target = float((g.qmin[ref] >= 600).mean())
        mipc = g[g.ref].groupby('run').qc.median()
        scores = {'q600 (pass-1 fit charge)': g.qmin, 'peak charge, plain CM': g.qp,
                  'peak charge, ringing removed': g.qc / g.run.map(mipc),
                  'kept fraction of the peaks': g.keepfrac, 'channels > 5 sigma after removal': g.ncs}
        for name, s in scores.items():
            s = s.to_numpy(float)
            thr = 600.0 if name.startswith('q600') else float(np.nanquantile(s[ref], 1 - target))
            c = np.nan_to_num(s, nan=-1e9) >= thr
            for lo, hi in zip(EDG[:-1], EDG[1:]):
                m = ((g.ms >= lo) & (g.ms < hi)).to_numpy()
                s0, s1 = g[m], g[m & c]
                e0, e1 = exc(s0), exc(s1)
                rows.append(dict(arm=arm, rule=name, thr=thr, ref_kept=float(c[ref].mean()), lo=lo, hi=hi, n=len(s0),
                                 kept=len(s1) / max(len(s0), 1),
                                 eff=len(s1) * e1 / (len(s0) * e0) if e0 and e0 > 0.01 else np.nan, purity=e1 / exl))
            L = g[(g.ms >= 30) & g.wall]
            i = np.digitize(L.tanx.abs(), TB) - 1
            cl = c[((g.ms >= 30) & g.wall).to_numpy()]
            ang.append(dict(arm=arm, rule=name, **{f'{TB[k]}-{TB[k + 1]}': float(cl[i == k].mean())
                                                   for k in range(len(TB) - 1)}))
    E = pd.DataFrame(rows)
    E.to_csv(OUT / 'wq_eval.csv', index=False)
    pd.DataFrame(ang).to_csv(OUT / 'wq_angle.csv', index=False)
    pd.set_option('display.width', 250)
    for arm in FEUS:
        e = E[E.arm == arm]
        print(f'--- {arm} (reference kept: ' + ', '.join(f'{r} {v:.4f}' for r, v in
                                                       e.groupby('rule').ref_kept.first().items()) + ')')
        for v in ('kept', 'eff', 'purity'):
            print(v)
            print(e.pivot_table(index='rule', columns='lo', values=v).round(2).to_string())
        print(e.pivot_table(index='rule', columns='lo', values='n').iloc[:1].to_string())
    print(pd.DataFrame(ang).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('measure', 'eval'))
    ap.add_argument('--subruns', default=','.join(SUBRUNS))
    ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == 'measure':
        measure(a.subruns.split(','), a.jobs)
    else:
        evaluate()


if __name__ == '__main__':
    main()
