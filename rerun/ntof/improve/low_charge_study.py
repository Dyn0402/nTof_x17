#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
low_charge_study.py -- can a fake-track cut follow the noise of each event, instead of a flat charge threshold?
(tracking audit 2026-11-22, finding 14; AUDIT_FIX_ROADMAP.md item 2.8.)

Before ~12 ms most gated A/C tracks are fits to the baseline ringing after the flash.  A flat charge cut
(min(x_q_sum, y_q_sum) >= 600 ADC) removes most of them.  Dylan asked whether a threshold set by the measured noise of
that very event does better.  This study measures, per event and per plane, quantities that a real track does not
create, from the decoded waveforms exactly as the reconstruction sees them (wft.io.FeuReader: pedestal from the
file's first events, plain 64-channel block common mode removed, per-channel noise sigma):

    cm_amp     the largest coherent (common-mode) block median removed from the event, in units of the median
               channel noise.  The flash ringing is coherent per chip.
    n_neg5     channels with a sample below -5 sigma after the common mode.  Ionisation gives positive pulses only,
               so negative excursions come from ringing.
    n_pos5     channels with a sample above +5 sigma (track strips plus positive ringing).
    rms_off    robust rms (1.4826 MAD) of the common-mode-subtracted samples, in ADC, over the channels that are NOT
               near a +5 sigma channel (each such channel and +-3 neighbours are left out): the event's own noise
               level away from the signal.
    zrms_off   the same in units of each channel's pedestal noise (1 = a quiet event).
    neg_amp    median depth [ADC] of the negative excursions of the n_neg5 channels (0 when there are none): the
               amplitude of the ringing in this event and plane.  A positive ringing lobe fitted as a track has about
               this amplitude per strip.
    neg_p90    the 90th percentile of the same depths.

Per track it then defines charge significances that use the event's own noise,

    sig_q     = min over views of q_sum / (rms_off * sqrt(n_strips))          (window charge over its noise charge)
    sig_amp   = min over views of (q_sum / n_strips) / rms_off               (mean strip amplitude over the noise)

and compares them with the charge-only rules of the 2026-11-22 study (flat 600 ADC; charge / the run's MIP peak and
path length; mean strip amplitude / the run's reference value) and with event-level vetoes on n_neg5 and zrms_off.
Every rule is set to keep the same fraction of reference tracks (>= 30 ms, wall-confirmed, within 15 mm of the source)
as the flat 600 ADC; then, per ToF bin, we give the kept fraction, the pointing-excess efficiency (real tracks kept,
estimated from the excess within 15 mm over a shuffled null) and the purity (excess of the kept tracks over the late
excess).

    python -m rerun.ntof.improve.low_charge_study noise [--subruns run_145/stat090_0000,...] [--jobs 6]
    python -m rerun.ntof.improve.low_charge_study eval

Waveforms: $X17_WF_ROOTS (colon-separated bases holding <run>/<sub>/decoded_root/, fa80c06 decode as in pass 1);
default the pass-1 staged copy $WORK_DATA/ntof_pass1/runs and $WORK_DATA/claude-jobs/bcad1a92/wf.
Out: $WORK_DATA/ntof_pass1/ana/low_charge/ (noise_<run>_<sub>_<feu>.parquet, CSV tables).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

WD = os.environ.get('WORK_DATA', '/media/ucla/work')
FEUS = {'A': (3, 4), 'C': (7, 8)}            # (x, y) FEUs of A and C in run_145 (ntof_registry.csv)
SUBRUNS = ('run_145/stat090_0000', 'run_145/stat090_0001', 'run_145/stat090_0002')
EDG = [1, 3, 5, 8, 12, 16, 20, 30, 75]
TB = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
NEIGH = 3
TARGET = {'A': 0.9972, 'C': 0.9869}


def out_dir() -> Path:
    d = Path(WD) / 'ntof_pass1' / 'ana' / 'low_charge'
    d.mkdir(parents=True, exist_ok=True)
    return d


def wf_roots() -> list[str]:
    env = os.environ.get('X17_WF_ROOTS')
    return env.split(':') if env else [f'{WD}/ntof_pass1/runs', f'{WD}/claude-jobs/bcad1a92/wf']


def files_for(run: str, sub: str, feu: int) -> list[str]:
    for r in wf_roots():
        f = sorted(glob.glob(f'{r}/{run}/{sub}/decoded_root/*_{feu:02d}.root'))
        if f:
            return f
    return []


# ------------------------------------------------------------------------------------------------- per-event noise
def noise_file(path: str):
    """Per-event noise measures of one decoded FEU file (one acquisition tag)."""
    import pandas as pd
    from wft import io as wio
    rdr = wio.FeuReader(path)
    nz = np.maximum(rdr.noise.astype(np.float64), 1.0)
    nz_med = float(np.median(nz))
    nblk = 512 // wio.CNS_BLOCK
    tag = wio.file_tag(path)
    rows = []
    idx = np.arange(rdr.n_entries)
    for lo in range(0, len(idx), wio.CHUNK):
        block = idx[lo:lo + wio.CHUNK]
        arr = rdr.tree.arrays(['eventId', 'amplitude'], entry_start=int(block[0]), entry_stop=int(block[-1]) + 1,
                              library='np')
        for j in range(len(block)):
            a = arr['amplitude'][j]
            if len(a) != rdr.n_sample * 512:
                continue
            raw = a.reshape(-1, 512).astype(np.float32) - rdr.ped              # (ns, 512), before the common mode
            ns = raw.shape[0]
            cms = np.median(raw.reshape(ns, nblk, wio.CNS_BLOCK), axis=2)       # (ns, nblk)
            W = (raw - np.repeat(cms, wio.CNS_BLOCK, axis=1)).T                 # (512, ns), what the fit sees
            z = W / nz[:, None]
            zmax, zmin = z.max(axis=1), z.min(axis=1)
            pos = zmax > 5.0
            near = pos.copy()
            for k in range(1, NEIGH + 1):
                near[k:] |= pos[:-k]
                near[:-k] |= pos[k:]
            off = ~near
            if off.sum() >= 32:
                w = W[off].ravel()
                rms_off = 1.4826 * float(np.median(np.abs(w - np.median(w))))
                zz = z[off].ravel()
                zrms_off = 1.4826 * float(np.median(np.abs(zz - np.median(zz))))
            else:
                rms_off = zrms_off = np.nan
            neg = zmin < -5.0
            depth = -W.min(axis=1)[neg]
            rows.append((int(arr['eventId'][j]), tag, float(np.abs(cms).max() / nz_med), int(neg.sum()),
                         int(pos.sum()), rms_off, zrms_off, float(np.median(depth)) if len(depth) else 0.0,
                         float(np.percentile(depth, 90)) if len(depth) else 0.0))
    return pd.DataFrame(rows, columns=['event_id', 'tag', 'cm_amp', 'n_neg5', 'n_pos5', 'rms_off', 'zrms_off',
                                       'neg_amp', 'neg_p90'])


def _noise_job(args):
    run, sub, arm, plane, feu, f = args
    d = noise_file(f)
    d['arm'], d['plane'] = arm, plane
    return run, sub, feu, d


def noise(subruns, jobs):
    import pandas as pd
    work = []
    for rs in subruns:
        run, sub = rs.split('/')
        for arm, (fx, fy) in FEUS.items():
            for plane, feu in (('x', fx), ('y', fy)):
                fs = files_for(run, sub, feu)
                if not fs:
                    print(f'  no waveforms for {run}/{sub} FEU {feu}', flush=True)
                for f in fs:
                    work.append((run, sub, arm, plane, feu, f))
    print(f'{len(work)} files', flush=True)
    parts = {}
    with ProcessPoolExecutor(jobs) as ex:
        for i, (run, sub, feu, d) in enumerate(ex.map(_noise_job, work)):
            parts.setdefault((run, sub, feu), []).append(d)
            print(f'[{i + 1}/{len(work)}] {run}/{sub} FEU {feu}: {len(d)} events', flush=True)
    for (run, sub, feu), ds in parts.items():
        pd.concat(ds, ignore_index=True).to_parquet(out_dir() / f'noise_{run}_{sub}_{feu:02d}.parquet', index=False)


# ------------------------------------------------------------------------------------------------------ evaluation
def tracks(subruns):
    """Gated, timed A/C tracks of the sub-runs, with the pointing quantities of tracking_audit.load."""
    import pandas as pd
    import pyarrow.dataset as ds
    from sept26_prelim_analysis import pass1_tracks as P1
    from sept26_prelim_analysis.capsule_tof import crossing
    from rerun.ntof.improve import tracking_audit as TA
    runs = sorted({s.split('/')[0] for s in subruns})
    subs = sorted({s.split('/')[1] for s in subruns})
    d = P1.read(TA.COLS, arms=('A', 'C'), runs=runs, filter=ds.field('subrun').isin(subs))
    d = d[d.gated.fillna(False).astype(bool) & d.t_since_flash_ns.notna()].copy()
    d['ms'] = d.t_since_flash_ns / 1e6
    t = -d.p0_z / d.d_z
    d['pos'] = d.p0_x
    d['disp'] = t * d.d_x
    d['wall'] = False
    for a in ('A', 'C'):
        m = d.arm == a
        d.loc[m, 'wall'] = d.loc[m, f'wall_{a}'].astype('boolean').fillna(False).astype(bool)
    d['qmin'] = np.minimum(d.x_q_sum, d.y_q_sum)
    d['miss'] = np.nan
    for a in ('A', 'C'):
        m = (d.arm == a).to_numpy()
        g = d[m]
        late = g[(g.ms >= 20) & (g.qmin >= 600)]
        u0, _, _ = crossing(late.pos.to_numpy(), late.disp.to_numpy())
        d.loc[m, 'miss'] = (g.pos + g.disp - u0).to_numpy()
    d['ref'] = (d.ms >= 30) & d.wall & (d.miss.abs() < 15)
    d['sec'] = np.sqrt(1 + d.tanx ** 2 + d.tany ** 2)
    return d


def attach_noise(d):
    import pandas as pd
    parts = []
    for f in sorted(out_dir().glob('noise_*.parquet')):
        parts.append(pd.read_parquet(f))
    n = pd.concat(parts, ignore_index=True)
    w = n.pivot_table(index=['tag', 'event_id', 'arm'], columns='plane',
                      values=['cm_amp', 'n_neg5', 'n_pos5', 'rms_off', 'zrms_off', 'neg_amp', 'neg_p90'],
                      aggfunc='first')
    w.columns = [f'{p}_{v}' for v, p in w.columns]
    w = w.reset_index()
    m = d.merge(w, on=['tag', 'event_id', 'arm'], how='left', validate='many_to_one')
    print(f'noise joined for {m.x_rms_off.notna().mean():.4f} of the tracks', flush=True)
    return m[m.x_rms_off.notna() & m.y_rms_off.notna()].copy()


def rules(g):
    """Per-track scores (higher = more track-like) and the event-level vetoes."""
    S = {}
    ref = g.ref.to_numpy()
    S['q600 (flat ADC)'] = g.qmin.to_numpy(float)
    mip = g[g.ref].groupby('run')[['x_q_sum', 'y_q_sum']].median()
    S['charge / run MIP'] = (np.minimum(g.x_q_sum / g.run.map(mip.x_q_sum), g.y_q_sum / g.run.map(mip.y_q_sum))
                             / g.sec).to_numpy(float)
    amp = {}
    for p in 'xy':
        r = g[f'{p}_q_sum'] / g[f'{p}_n_strips']
        amp[p] = r / g.run.map(r[g.ref].groupby(g.run[g.ref]).median())
    S['amplitude / run ref'] = np.minimum(amp['x'], amp['y']).to_numpy(float)
    rx, ry = g.x_rms_off.clip(lower=0.5), g.y_rms_off.clip(lower=0.5)
    S['sig_q (event noise)'] = np.minimum(g.x_q_sum / (rx * np.sqrt(g.x_n_strips)),
                                          g.y_q_sum / (ry * np.sqrt(g.y_n_strips))).to_numpy(float)
    S['sig_amp (event noise)'] = np.minimum(g.x_q_sum / g.x_n_strips / rx, g.y_q_sum / g.y_n_strips / ry).to_numpy(float)
    S['-n_neg5 (event veto)'] = -np.maximum(g.x_n_neg5, g.y_n_neg5).to_numpy(float)
    S['-zrms_off (event veto)'] = -np.maximum(g.x_zrms_off, g.y_zrms_off).to_numpy(float)
    # the track's mean strip amplitude against the ringing amplitude of its own event and plane (floor: 3 x the
    # off-signal rms, so a quiet event is judged against its noise)
    ring = {p: np.maximum(g[f'{p}_neg_amp'], 3 * g[f'{p}_rms_off'].clip(lower=0.5)) for p in 'xy'}
    S['amp / event ringing'] = np.minimum(g.x_q_sum / g.x_n_strips / ring['x'],
                                          g.y_q_sum / g.y_n_strips / ring['y']).to_numpy(float)
    ringq = {p: np.maximum(g[f'{p}_neg_amp'], 3 * g[f'{p}_rms_off'].clip(lower=0.5)) * np.sqrt(g[f'{p}_n_strips'])
             for p in 'xy'}
    S['charge / event ringing'] = np.minimum(g.x_q_sum / ringq['x'], g.y_q_sum / ringq['y']).to_numpy(float)
    return S, ref


def _exc(s, rng):
    m = np.abs(s.miss.to_numpy())
    if len(m) < 50:
        return np.nan
    src = (s.pos + s.disp - s.miss).to_numpy()
    sh = np.abs(s.pos.to_numpy() + rng.permutation(s.disp.to_numpy()) - src)
    return float((m < 15).mean() - (sh < 15).mean())


def evaluate():
    import pandas as pd
    d = attach_noise(tracks(SUBRUNS))
    rng = np.random.default_rng(0)
    rows, ang, feat = [], [], []
    for a in ('A', 'C'):
        g = d[d.arm == a].reset_index(drop=True)
        S, ref = rules(g)
        # matched to the CAMPAIGN reference efficiency of the flat 600 ADC (2026-11-22 study, 8.3 M tracks): the few
        # thousand reference tracks of a few sub-runs cannot resolve 99.7 %
        e600 = TARGET[a]
        exl = _exc(g[g.ms >= 30], rng)
        # what the noise measures look like (medians), reference against the earliest bin
        for name, sel in (('reference', g.ref), ('1-5 ms', g.ms < 5), ('5-12 ms', (g.ms >= 5) & (g.ms < 12)),
                          ('12-20 ms', (g.ms >= 12) & (g.ms < 20)), ('>= 30 ms', g.ms >= 30)):
            s = g[sel]
            feat.append(dict(arm=a, sample=name, n=len(s), **{c: float(s[c].median()) for c in
                        ('x_rms_off', 'y_rms_off', 'x_zrms_off', 'y_zrms_off', 'x_n_neg5', 'y_n_neg5', 'x_neg_amp', 'y_neg_amp',
                         'x_cm_amp', 'y_cm_amp')}))
        late_wall = ((g.ms >= 30) & g.wall).to_numpy()
        keeps = {}
        for name, sc in S.items():
            thr = 600.0 if name.startswith('q600') else float(np.quantile(sc[ref], 1 - e600))
            keeps[name] = (thr, sc >= thr)
        # AND of the flat charge and the ringing ratio, the ratio threshold scanned so the AND keeps the target
        q6 = S['q600 (flat ADC)'] >= 600
        rr = S['amp / event ringing']
        for t in np.quantile(rr[ref], np.linspace(0, 0.05, 501)):
            if (q6 & (rr >= t))[ref].mean() < e600:
                break
            tbest = t
        keeps['q600 AND amp/ringing'] = (float(tbest), q6 & (rr >= tbest))
        for name, (thr, keep) in keeps.items():
            r = dict(arm=a, rule=name, threshold=thr, keeps_reference=float(keep[ref].mean()),
                     keeps_ge20=float(keep[(g.ms >= 20).to_numpy()].mean()), late_excess=exl)
            for lo, hi in zip(EDG[:-1], EDG[1:]):
                m = ((g.ms >= lo) & (g.ms < hi)).to_numpy()
                s, k = g[m], g[m & keep]
                ea, ek = _exc(s, rng), _exc(k, rng)
                r[f'kept_{lo}_{hi}'] = len(k) / max(len(s), 1)
                r[f'eff_{lo}_{hi}'] = len(k) * ek / (len(s) * ea) if ea and np.isfinite(ea) and ea > 0.005 else np.nan
                r[f'purity_{lo}_{hi}'] = ek / exl
                r[f'n_{lo}_{hi}'] = len(s)
            rows.append(r)
            L = g[late_wall]
            for v in ('tanx', 'tany'):
                i = np.digitize(L[v].abs(), TB) - 1
                for kb in range(len(TB) - 1):
                    ang.append(dict(arm=a, rule=name, view=v, lo=TB[kb], hi=TB[kb + 1], n=int((i == kb).sum()),
                                    eff=float(keep[late_wall][i == kb].mean()) if (i == kb).any() else np.nan))
    R, A, F = pd.DataFrame(rows), pd.DataFrame(ang), pd.DataFrame(feat)
    R.to_csv(out_dir() / 'rules_tof.csv', index=False)
    A.to_csv(out_dir() / 'rules_angle.csv', index=False)
    F.to_csv(out_dir() / 'noise_features.csv', index=False)
    pd.set_option('display.width', 250)
    print(F.round(2).to_string(index=False))
    for a in ('A', 'C'):
        print(f'\n{a}: kept / pointing-excess efficiency / purity per ToF bin (rules matched to the 600 ADC reference '
              f'efficiency)')
        for _, r in R[R.arm == a].iterrows():
            cells = ' '.join(f"{lo}-{hi} {r[f'kept_{lo}_{hi}']:.2f}/{r[f'eff_{lo}_{hi}']:.2f}/{r[f'purity_{lo}_{hi}']:.2f}"
                             for lo, hi in zip(EDG[:-1], EDG[1:]))
            print(f"  {r.rule:24s} ref {r.keeps_reference:.4f} | {cells}")
        print('  efficiency of late wall tracks vs |tan x|:')
        for name in R[R.arm == a].rule.unique():
            s = A[(A.arm == a) & (A.rule == name) & (A.view == 'tanx')]
            print(f'    {name:24s} ' + ' '.join(f'{e:.3f}' for e in s.eff))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['noise', 'eval'])
    ap.add_argument('--subruns', default=','.join(SUBRUNS))
    ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args()
    if a.step == 'noise':
        noise(a.subruns.split(','), a.jobs)
    else:
        evaluate()


if __name__ == '__main__':
    main()
