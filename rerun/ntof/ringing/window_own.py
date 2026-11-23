#!/usr/bin/env python3
"""window_own.py -- the per-track ringing test of window_charge.py, on the fit's own window, for A, C and D.

    python -m rerun.ntof.ringing.window_own measure [--subruns ...] [--jobs 4] [--sv 2.0]
    python -m rerun.ntof.ringing.window_own eval

window_charge.py looked at |position - p0| <= |tan| x 28 mm + 3 strips: symmetric about p0, so twice as wide as the
column on the side the track never reaches.  Here every gated track of the analysis view (A, C and D) is measured on
two windows per view:

    own     the fit window: n_strips strips (the reco's own count, seed cluster + 3 live strips each side) in position
            order, centred on the column centre p0 + tan x GAP / 2 (the column runs from p0 at the mesh to
            p0 + tan x GAP at the cathode; tan has the reco sign in the view)
    sym     window_charge.py's window, for comparison

and three waveform versions: plain (pass 1), v1 (deflect.deflect) and v3 (deflect.deflect3: v1 + a local second
component where a block carries one).  Per (window, version): n_sig = channels whose peak is above 5 sigma and q = the
sum of their peaks.  D's quality_join cut d_faint (amp_norm < 0.25 or x window >= 80 % censored) is recomputed here.
Out: $WORK_DATA/ntof_pass1/ana/ringing/wo_<run>_<sub>.parquet, wo_eval.csv, wo_angle.csv
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
from rerun.ntof.ringing.window_charge import OUT, VIEW, SUBRUNS, EDG, TB, pos_map   # noqa: E402

FEUS = {'A': (3, 4), 'C': (7, 8), 'D': (1, 2)}
GAP = 28.0
PAD = 3
VERSIONS = ('plain', 'v1', 'v3')
COLS = ['run', 'subrun', 'tag', 'event_id', 'arm', 'track_id', 'gated', 'x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta',
        'x_n_strips', 'y_n_strips', 'x_q_sum', 'y_q_sum', 'x_cens_frac', 'y_cens_frac', 't_since_flash_ns', 'p0_x',
        'p0_z', 'd_x', 'd_z', 'tanx', 'tany', 'wall_A', 'wall_C', 'wall_D']
D_AMP_MIN, D_CENS_X = 0.25, 0.8


def _pos_wrap(run, sub, feu):
    # window_charge.pos_map knows A/C only; D's FEUs are 1/2
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from rerun.ntof.ringing.window_charge import WF_ROOTS
    arm = next(a for a, f in FEUS.items() if feu in f)
    fx, fy = FEUS[arm]
    base = next(b for b in WF_ROOTS if os.path.isdir(f'{b}{run}/{sub}/decoded_root')
                and any(n.endswith(f'_{feu:02d}.root') for n in os.listdir(f'{b}{run}/{sub}/decoded_root')))
    cfg = wb.BeamConfig(KEY='x', RUN=run, SUB_RUN=sub, DET_NAME=f'mx17_{arm}', MX17_FEU_X=fx, MX17_FEU_Y=fy,
                        BASE_PATH=next(b for b in WF_ROOTS if os.path.isfile(f'{b}{run}/run_config.json')),
                        OUT_BASE='/tmp')
    return wio.strip_position_map(cfg)[feu], base


def windows(pos, p0, tan, n_strips, pitch):
    order = np.argsort(pos)
    order = order[np.isfinite(pos[order])]
    ps = pos[order]
    t = float(np.clip(tan, -1.0, 1.0))
    cen = p0 + 0.5 * t * GAP
    k = int(np.argmin(np.abs(ps - cen)))
    n = int(n_strips) if np.isfinite(n_strips) and n_strips > 0 else 7
    lo = max(0, k - n // 2)
    hi = min(len(order), lo + n)
    lo = max(0, hi - n)
    own = np.zeros(len(pos), bool)
    own[order[lo:hi]] = True
    sym = np.abs(pos - p0) <= min(abs(t), 1.0) * GAP + PAD * pitch
    return own, sym


def measure_file(args):
    import pandas as pd
    from wft import io as wio
    from rerun.ntof.ringing import deflect as DF
    run, sub, arm, plane, feu, path, trk, sv = args
    pos, _ = _pos_wrap(run, sub, feu)
    pitch = float(np.nanmedian(np.abs(np.diff(np.sort(pos[np.isfinite(pos)])))))
    rdr = wio.FeuReader(path)
    nz = np.maximum(rdr.noise.astype(np.float64), 1.0)
    want = np.where(np.isin(rdr.event_ids, trk.event_id.astype(np.int64).unique()))[0]
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
            if eid not in by_ev:
                continue
            raw = a.reshape(-1, 512).astype(np.float32) - rdr.ped
            W1 = DF.deflect(raw, rdr.noise)
            V = {'plain': DF.plain_cm(raw), 'v1': W1, 'v3': DF.deflect3(raw, rdr.noise, sv_nsig=sv, w1=W1)}
            pk = {k: v.max(axis=0) for k, v in V.items()}
            n_neg = int(((V['plain'] / nz[None, :]).min(axis=0) < -5).sum())
            for r in by_ev[eid].itertuples():
                p0, t, n = getattr(r, f'{plane}_p0'), getattr(r, f'{plane}_tan_theta'), getattr(r, f'{plane}_n_strips')
                if not np.isfinite(p0) or not np.isfinite(t):
                    continue
                own, sym = windows(pos, p0, t, n, pitch)
                rec = dict(tag=r.tag, event_id=eid, arm=arm, track_id=r.track_id, plane=plane, n_neg5=n_neg,
                           n_own=int(own.sum()), n_sym=int(sym.sum()))
                for wn, w in (('own', own), ('sym', sym)):
                    for vn in VERSIONS:
                        s = w & (pk[vn] > 5 * nz)
                        rec[f'n_{wn}_{vn}'] = int(s.sum())
                        rec[f'q_{wn}_{vn}'] = float(pk[vn][s].sum())
                rows.append(rec)
    return pd.DataFrame(rows)


def measure(subruns, jobs, sv):
    import pandas as pd
    from wft import io as wio
    for rs in subruns:
        run, sub = rs.split('/')
        out = OUT / f'wo_{run}_{sub}.parquet'
        if out.exists():
            print(f'{rs}: exists', flush=True)
            continue
        t = pd.read_parquet(VIEW / f'tracks_{run}_{sub}.parquet', columns=COLS)
        t = t[t.arm.isin(list(FEUS)) & t.gated.fillna(False).astype(bool)]
        work = []
        for arm, (fx, fy) in FEUS.items():
            for plane, feu in (('x', fx), ('y', fy)):
                _, base = _pos_wrap(run, sub, feu)
                for f in wio.subrun_files(base, run, sub, feu):
                    tag = wio.file_tag(f)
                    trk = t[(t.arm == arm) & (t.tag == tag)]
                    if len(trk):
                        work.append((run, sub, arm, plane, feu, f, trk, sv))
        with ProcessPoolExecutor(jobs) as ex:
            parts = list(ex.map(measure_file, work))
        d = pd.concat([p for p in parts if len(p)], ignore_index=True)
        vals = [c for c in d.columns if c not in ('tag', 'event_id', 'arm', 'track_id', 'plane')]
        w = d.pivot_table(index=['tag', 'event_id', 'arm', 'track_id'], columns='plane', values=vals, aggfunc='first')
        w.columns = [f'{p}_{v}' for v, p in w.columns]
        m = t.merge(w.reset_index(), on=['tag', 'event_id', 'arm', 'track_id'], how='inner')
        m.to_parquet(out, index=False)
        print(f'{rs}: {len(m)} tracks measured of {len(t)}', flush=True)


def evaluate():
    import pandas as pd
    from sept26_prelim_analysis.capsule_tof import crossing
    d = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob('wo_run_*.parquet'))], ignore_index=True)
    d = d[d.t_since_flash_ns.notna()].copy()
    d['ms'] = d.t_since_flash_ns / 1e6
    # pointing in global coordinates as capsule_tof.VIEW: A, C measure x (normal z); D measures z (normal x)
    isd = d.arm == 'D'
    d['pos'] = np.where(isd, d.p0_z, d.p0_x)
    d['disp'] = np.where(isd, -d.p0_x / d.d_x * d.d_z, -d.p0_z / d.d_z * d.d_x)
    d['qmin'] = np.minimum(d.x_q_sum, d.y_q_sum)
    with np.errstate(divide='ignore', invalid='ignore'):
        d['amp'] = np.minimum(d.x_q_sum / d.x_n_strips, d.y_q_sum / d.y_n_strips)
    rng = np.random.default_rng(0)
    rows, ang = [], []
    for arm in FEUS:
        g = d[d.arm == arm].copy()
        g['wall'] = g[f'wall_{arm}'].astype('boolean').fillna(False).astype(bool)
        refm = g[(g.ms >= 30) & g.wall & (np.maximum(g.x_cens_frac.fillna(0), g.y_cens_frac.fillna(0)) < 0.5)]
        ref_amp = refm.groupby('run').amp.median()
        g['amp_norm'] = g.amp / g.run.map(ref_amp)
        g['d_faint'] = (g.amp_norm < D_AMP_MIN) | (g.x_cens_frac.fillna(0) >= D_CENS_X) if arm == 'D' else False
        base = ~g.d_faint.astype(bool)
        late = g[(g.ms >= 20) & base & (g.qmin >= 600)]
        u0, _, _ = crossing(late.pos.to_numpy(), late.disp.to_numpy())
        g['miss'] = g.pos + g.disp - u0
        g['ref'] = (g.ms >= 30) & g.wall & (g.miss.abs() < 15) & base
        ref = g.ref.to_numpy()

        def exc(s):
            if len(s) < 30:
                return np.nan
            m = np.abs(s.miss.to_numpy())
            sh = np.abs(s.pos.to_numpy() + rng.permutation(s.disp.to_numpy()) - u0)
            return float((m < 15).mean() - (sh < 15).mean())
        exl = exc(g[(g.ms >= 30) & base])
        nmin = lambda w, v: np.minimum(g[f'x_n_{w}_{v}'], g[f'y_n_{w}_{v}']).to_numpy()
        rules = {'gated (view)': np.ones(len(g), bool),
                 'd_faint removed' if arm == 'D' else 'gated (same)': base.to_numpy(),
                 'q >= 600': (g.qmin >= 600).to_numpy() & base.to_numpy()}
        for w in ('sym', 'own'):
            for v in ('v1', 'v3'):
                rules[f'{w}/{v}: >= 1 ch'] = (nmin(w, v) >= 1) & base.to_numpy()
                rules[f'{w}/{v}: >= 1 ch & q600'] = (nmin(w, v) >= 1) & (g.qmin >= 600).to_numpy() & base.to_numpy()
            rules[f'{w}/plain: >= 1 ch'] = (nmin(w, 'plain') >= 1) & base.to_numpy()
        rules['own/v3: >= 2 ch'] = (nmin('own', 'v3') >= 2) & base.to_numpy()
        for name, c in rules.items():
            row = dict(arm=arm, rule=name, ref_kept=float(c[ref].mean()) if ref.any() else np.nan)
            for lo, hi in zip(EDG[:-1], EDG[1:]):
                m = ((g.ms >= lo) & (g.ms < hi)).to_numpy()
                s0, s1 = g[m], g[m & c]
                e0, e1 = exc(s0), exc(s1)
                rows.append(dict(row, lo=lo, hi=hi, n=len(s0), kept=len(s1) / max(len(s0), 1),
                                 eff=len(s1) * e1 / (len(s0) * e0) if e0 and e0 > 0.01 else np.nan,
                                 purity=e1 / exl if exl else np.nan))
            L = ((g.ms >= 30) & g.wall & (g.miss.abs() < 15)).to_numpy()
            i = np.digitize(g.tanx.abs().to_numpy()[L], TB) - 1
            cl = c[L]
            ang.append(dict(arm=arm, rule=name, **{f'{TB[k]}-{TB[k + 1]}': float(cl[i == k].mean()) if (i == k).any()
                                                   else np.nan for k in range(len(TB) - 1)}))
    E = pd.DataFrame(rows)
    E.to_csv(OUT / 'wo_eval.csv', index=False)
    A = pd.DataFrame(ang)
    A.to_csv(OUT / 'wo_angle.csv', index=False)
    pd.set_option('display.width', 250)
    for arm in FEUS:
        e = E[E.arm == arm]
        print(f'===== {arm}')
        print(e.groupby('rule', sort=False).ref_kept.first().round(4).to_string())
        for v in ('kept', 'purity', 'eff'):
            print(v)
            print(e.pivot_table(index='rule', columns='lo', values=v, sort=False).round(2).to_string())
        print(e.pivot_table(index='rule', columns='lo', values='n', sort=False).iloc[:1].to_string())
    print(A.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('measure', 'eval'))
    ap.add_argument('--subruns', default=','.join(SUBRUNS))
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--sv', type=float, default=2.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == 'measure':
        measure(a.subruns.split(','), a.jobs, a.sv)
    else:
        evaluate()


if __name__ == '__main__':
    main()
