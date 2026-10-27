#!/usr/bin/env python3
"""build_raw71_cache.py --out DIR [--per-plateau 600]

det4 head-on RAW events of run_71 (flat mount, Ar/CF4/iso, 64 x 60 ns, drift 700 / 450 / 275 V) from the det4-only
sample dump `wf_run71_raw_det4only.npz` (RAW_RUN71_REANALYSIS_2026-08-04) into the bench calibration-cache format,
one pickle per plateau, for qfit.py.  Per view: the brightest strip (ch 510 / 372 excluded: they oscillate), peak
400-3000 ADC, +-16 strips by position; per-strip baseline = mean of the present samples 0-6, central strip pre-level
|.| < 15 ADC (pile-up / oscillation veto, the reanalysis' clean cut).  RAW drops ~24 % of sample groups (FEU
bandwidth): a missing sample is stored as an exact 0.0 and the fit must run with WFT_ZS_CENSOR=1 (censored); real
samples that land on 0.0 after the baseline are nudged to 1e-3.  tan_x = tan_y = 0 (head-on).
"""
import argparse, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM        # noqa: E402

NPZ = '/media/dylan/data/x17/sps_run53_det4_check/staging/run_71/wf_run71_raw_det4only.npz'
BAD = (510, 372)
NS = 64


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--per-plateau', type=int, default=600)
    ap.add_argument('--half-win', type=int, default=16)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = np.load(NPZ)
    ev, ch, sm, am = D['ev'], D['ch'].astype(int), D['samp'].astype(int), D['amp'].astype(np.float32)
    lab = D['ev_plateau']
    o = np.argsort(ev, kind='stable'); ev, ch, sm, am = ev[o], ch[o], sm[o], am[o]
    b = np.r_[0, np.flatnonzero(np.diff(ev)) + 1, len(ev)]
    out = {}
    for i in range(len(b) - 1):
        s, e = b[i], b[i + 1]; eid = int(ev[s]); L = str(lab[eid]) if eid < len(lab) else ''
        if not L or len(out.get(L, {})) >= a.per_plateau:
            continue
        c, t, x = ch[s:e], sm[s:e], am[s:e]
        rec = dict(eid=eid, tan_x=0.0, tan_y=0.0); ok = True
        for view in ('x', 'y'):
            k = (VIEW[c] == view) & (t >= 0) & (t < NS)
            if not k.any():
                ok = False; break
            chs = np.unique(c[k]); W = np.zeros((len(chs), NS), np.float32); have = np.zeros_like(W, bool)
            ci = np.searchsorted(chs, c[k]); W[ci, t[k]] = x[k]; have[ci, t[k]] = True
            npre = np.maximum(have[:, :7].sum(1), 1)
            base = np.where(have[:, :7], W[:, :7], 0).sum(1) / npre
            W = np.where(have, W - base[:, None], 0.0)
            W[have & (W == 0.0)] = 1e-3
            pk = W.max(1); pk[np.isin(chs, BAD)] = -1
            j = int(np.argmax(pk))
            if not (400 <= pk[j] <= 3000) or abs(base[j]) > 15 or have[j, :7].sum() == 0:
                ok = False; break
            sel = np.abs(POSITION_MM[chs] - POSITION_MM[chs[j]]) <= a.half_win * PITCH_MM
            sel &= ~np.isin(chs, BAD)
            cc = chs[sel]; WW = W[sel]; oo = np.argsort(POSITION_MM[cc])
            q = np.clip(WW, 0, None).sum(1)
            rec[view] = dict(ch=cc[oo].astype(np.int16), pos=POSITION_MM[cc[oo]].astype(np.float32), W=WW[oo],
                             noise=np.full(len(cc), 10.0, np.float32))
            rec[f'ref_mesh_{view}'] = float((q * POSITION_MM[cc]).sum() / max(q.sum(), 1e-9)); rec[f'ftst_{view}'] = 0
        if ok:
            out.setdefault(L, {})[eid] = rec
    for L, evs in out.items():
        p = os.path.join(a.out, f'run71_raw_{L}.pkl'); pickle.dump(evs, open(p, 'wb'), protocol=4); print(L, len(evs), p)


if __name__ == '__main__':
    main()
