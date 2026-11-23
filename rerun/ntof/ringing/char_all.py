#!/usr/bin/env python3
"""char_all.py -- the ringing measured the same way on every chamber's FEUs (A 3/4, C 7/8, D 1/2; open items 2026-11-23).

    python -m rerun.ntof.ringing.char_all extract --run run_145 --sub stat090_0000 --tag 000 [--n 4000]
    python -m rerun.ntof.ringing.char_all table   --run run_145 --sub stat090_0000 --tag 000

`extract` caches the first N complete events of each FEU (pedestal-subtracted, before any common mode) with their time
since the flash.  `table` prints and writes, per FEU: the dominant common-mode frequency, the coherence (block-block,
x-y FEU, across chambers), the event-to-event phase correlation, the rank-one fraction of the residual, the stability of
the per-channel coupling, the ToF decay of the channels below -5 sigma before / after deflect.py, and signal preservation
on late events.  Out: $WORK_DATA/ntof_pass1/ana/ringing/char_<run>_<sub>_<tag>.csv
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
WD = os.environ.get('WORK_DATA', '/media/ucla/work')
OUT = Path(WD) / 'ntof_pass1' / 'ana' / 'ringing'
CACHE = OUT / 'cache'
WF_ROOTS = [f'{WD}/ntof_pass1/runs/', f'{WD}/claude-jobs/bcad1a92/wf/']
FEU_OF = {1: 'D x', 2: 'D y', 3: 'A x', 4: 'A y', 7: 'C x', 8: 'C y'}
TRIG = [Path(WD) / 'ntof_pass1/context_r2/trigtime', Path('/media/dylan/data/x17/sept26_prelim/trigtime')]
BINS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 80]


def wf_file(run, sub, tag, feu):
    for b in WF_ROOTS:
        d = Path(f'{b}{run}/{sub}/decoded_root')
        if d.is_dir():
            f = sorted(d.glob(f'*_{tag}_{feu:02d}.root'))
            if f:
                return f[0]
    raise FileNotFoundError(f'{run}/{sub} tag {tag} FEU {feu}')


def trigtime(run, sub):
    for t in TRIG:
        f = t / f'trigtime_{run}_{sub}.parquet'
        if f.exists():
            return pd.read_parquet(f).drop_duplicates('eventId').set_index('eventId')
    raise FileNotFoundError(f'trigtime {run} {sub}')


def cache_path(run, sub, tag, feu):
    return CACHE / f'{run}_{sub}_{tag}_feu{feu}.npz'


def extract(run, sub, tag, n, feus=tuple(FEU_OF)):
    from wft import io as wio
    tt = trigtime(run, sub)
    for feu in feus:
        out = cache_path(run, sub, tag, feu)
        if out.exists():
            continue
        r = wio.FeuReader(str(wf_file(run, sub, tag, feu)))
        a = r.tree.arrays(['eventId', 'amplitude'], entry_stop=int(n * 1.05) + 10, library='np')
        ok = np.array([len(x) == r.n_sample * 512 for x in a['amplitude']])
        ev = a['eventId'][ok][:n]
        A = np.stack([x.reshape(-1, 512) for x in a['amplitude'][ok][:n]]).astype(np.float32) - r.ped
        ms = tt.reindex(ev).t_pred_ns.to_numpy() / 1e6
        np.savez_compressed(out, A=A.astype(np.float16), ms=ms, ev=ev, noise=r.noise)
        print(f'FEU {feu}: {len(ev)} events cached', flush=True)


def load(run, sub, tag, feu):
    z = np.load(cache_path(run, sub, tag, feu))
    return z['A'].astype(np.float32), z['ms'], z['ev'], np.maximum(z['noise'].astype(np.float64), 1.0)


def table(run, sub, tag):
    from rerun.ntof.ringing.deflect import deflect
    rows, cms = [], {}
    for feu, name in FEU_OF.items():
        if not cache_path(run, sub, tag, feu).exists():
            continue
        A, ms, ev, nz = load(run, sub, tag, feu)
        N, ns, _ = A.shape
        B = A.reshape(N, ns, 8, 64)
        c = np.median(B, axis=3)                                  # (N, ns, 8)
        cms[feu] = (ev, c)
        R = (B - c[..., None]).reshape(N, ns, 512)
        nneg = ((R / nz).min(axis=1) < -5).sum(axis=1)
        G = np.stack([deflect(A[i], nz) for i in range(N)])
        nneg2 = ((G / nz).min(axis=1) < -5).sum(axis=1)
        npos, npos2 = ((R / nz).max(axis=1) > 5).sum(axis=1), ((G / nz).max(axis=1) > 5).sum(axis=1)
        early = (ms > 3) & (ms < 8)
        # dominant frequency of the block-averaged CM, 3-8 ms
        cm = c[early].mean(axis=2)
        cm = cm - cm.mean(axis=1, keepdims=True)
        sp = np.abs(np.fft.rfft(cm * np.hanning(ns), n=400, axis=1))
        fr = np.fft.rfftfreq(400, d=60e-9)
        fdom = fr[3:][np.argmax(sp[:, 3:], axis=1)]
        spm = sp.mean(0)
        # coherence: block-block correlation of the CM over samples, early events
        ce = c[early] - c[early].mean(axis=1, keepdims=True)
        cbb = []
        for e in ce[:500]:
            m = np.corrcoef(e.T)
            cbb.append(np.nanmean(m[np.triu_indices(8, 1)]))
        # phase: consecutive-event CM correlation, early
        ix = np.flatnonzero(early)
        pc = [np.corrcoef(cm[k], cm[k + 1])[0, 1] for k in range(min(len(cm) - 1, 500))]
        # rank-one fraction and coupling stability (early ringing events)
        er = np.flatnonzero(early & (nneg > 20))[:300]
        r1, bet = [], []
        for e in er:
            fe = []
            for b in range(8):
                M = R[e][:, b * 64:(b + 1) * 64]
                s = np.linalg.svd(M - M.mean(0), compute_uv=False)
                fe.append(s[0] ** 2 / np.sum(s ** 2))
                cc = c[e, :, b]
                bet.append(np.r_[e, b, B[e, :, b, :].T @ cc / max(cc @ cc, 1e-6)])
            r1.append(np.median(fe))
        bet = np.array(bet) if len(bet) else np.zeros((0, 66))
        stab = []
        if len(bet):
            for b in range(8):
                bb = bet[bet[:, 1] == b, 2:]
                stab += [np.corrcoef(bb[k], bb[k + 1])[0, 1] for k in range(len(bb) - 1)]
        # signal preservation: late channels with a clean positive pulse
        lt = ms > 30
        z0 = R[lt] / nz
        sig = (z0.max(1) > 8) & (z0.min(1) > -3)
        q0 = np.clip(R[lt], 0, None).sum(1)[sig]
        q1 = np.clip(G[lt], 0, None).sum(1)[sig]
        row = dict(feu=feu, view=name, n_events=N, noise_med=float(np.median(nz)),
                   f_dom_MHz=float(np.median(fdom) / 1e6), f_peak_mean_spectrum_MHz=float(fr[3:][np.argmax(spm[3:])] / 1e6),
                   cm_rms_3_8ms=float(np.sqrt((cm ** 2).mean())), corr_block_block=float(np.nanmedian(cbb)),
                   corr_consecutive_events=float(np.nanmedian(pc)), rank1_frac=float(np.median(r1)) if r1 else np.nan,
                   beta_stability=float(np.nanmedian(stab)) if stab else np.nan,
                   sig_keep_med=float(np.median(q1 / q0)) if sig.any() else np.nan,
                   sig_keep_p10=float(np.percentile(q1 / q0, 10)) if sig.any() else np.nan, n_sig_ch=int(sig.sum()))
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (ms >= lo) & (ms < hi)
            if m.sum() < 5:
                continue
            row[f'neg5 {lo}-{hi}'] = f'{np.median(nneg[m]):.0f}({np.percentile(nneg[m], 90):.0f})->{np.median(nneg2[m]):.0f}({np.percentile(nneg2[m], 90):.0f})'
            row[f'pos5 {lo}-{hi}'] = f'{np.median(npos[m]):.0f}->{np.median(npos2[m]):.0f}'
            row[f'gate {lo}-{hi}'] = float((nneg[m] >= 3).mean())
        rows.append(row)
        print(f'{name} done', flush=True)
    # cross-FEU coherence (same events)
    cross = []
    for f1 in cms:
        for f2 in cms:
            if f2 <= f1:
                continue
            e1, c1 = cms[f1]
            e2, c2 = cms[f2]
            com, i1, i2 = np.intersect1d(e1, e2, return_indices=True)
            v = []
            for a, b in zip(i1[:2000], i2[:2000]):
                x, y = c1[a].mean(1), c2[b].mean(1)
                if x.std() > 1 and y.std() > 1:
                    v.append(np.corrcoef(x, y)[0, 1])
            cross.append(dict(a=FEU_OF[f1], b=FEU_OF[f2], n=len(v), corr_med=float(np.median(v)) if v else np.nan))
    T = pd.DataFrame(rows)
    X = pd.DataFrame(cross)
    T.to_csv(OUT / f'char_{run}_{sub}_{tag}.csv', index=False)
    X.to_csv(OUT / f'char_cross_{run}_{sub}_{tag}.csv', index=False)
    pd.set_option('display.width', 250)
    print(T.T.to_string())
    print(X.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('extract', 'table'))
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--sub', default='stat090_0000')
    ap.add_argument('--tag', default='000')
    ap.add_argument('--n', type=int, default=4000)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.step == 'extract':
        extract(a.run, a.sub, a.tag, a.n)
    else:
        table(a.run, a.sub, a.tag)


if __name__ == '__main__':
    main()
