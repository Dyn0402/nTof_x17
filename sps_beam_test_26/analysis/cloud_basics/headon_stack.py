#!/usr/bin/env python3
"""headon_stack.py -- late-charge observable on head-on beam tracks, done carefully.

Input: an `extract_det4_only.py --keep 12` cache of run_71 RAW (both views
head-on: flat mount, beam perpendicular).  Per event and view the strips are
indexed by their offset o = -12..12 from det4's own leading strip (hits are
used for candidate finding only).

What this does differently from every earlier stack (§10, §11, §14):
  * time = DREAM sample index.  The window is opened by the beam trigger, which
    is asynchronous to the 60 ns clock: no data-driven alignment of any kind.
  * missing samples (the FEU's dropped RAW packets: ~20-25 %, rising slightly
    through the window) stay NaN.  Each (offset, sample) cell is averaged over
    the events that HAVE it, and the strip means are summed afterwards.  The
    loss is in ~5-sample packet groups, independent of the signal, so this is
    unbiased; zero-filling them (as spike_test.py did) is not.
  * per-strip baseline = that strip's mean over the pre-trigger samples.
  * sums over |o| <= h for h = 0, 1, 2, 4, 8, 12: the width at which a view's
    late/early ratio stops changing is the width that contains its charge.
  * bootstrap over events for every quoted ratio.

    headon_stack.py <cache.npz> [--tag masked] [--nboot 200] [--split none|charge|rate|pos]
Output: results/headon_<tag>.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.dirname(HERE)
sys.path[:0] = [ANA, os.path.join(os.path.dirname(ANA), 'det4_sps_assessment')]
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM         # noqa: E402

PLATEAUS = (('raw700', 243), ('raw450', 150), ('raw275', 92))
NSMP, SNS = 64, 60.0
KEEP = 12
WIDTHS = (0, 1, 2, 4, 8, 12)
PRE = slice(0, 9)              # pre-trigger samples (signal onset is at sample ~11)
SIDX = np.round(POSITION_MM / PITCH_MM).astype(int)


def load(path):
    Z = np.load(path, allow_pickle=True)
    ev, ch, s, a = Z['ev'], Z['ch'].astype(int), Z['samp'].astype(int), Z['amp']
    lead = {'x': Z['ev_pX'], 'y': Z['ev_pY']}
    return ev, ch, s, a, lead, Z['ev_plateau'], Z['ev_t_wall']


def cube(ev, ch, s, a, lead, events, view):
    """(n_ev, 2*KEEP+1, NSMP) with NaN for absent samples."""
    k = (VIEW[ch] == view) & np.isin(ev, events)
    e_i = np.searchsorted(events, ev[k])                    # events is sorted
    s0 = np.round(lead[view][ev[k]] / PITCH_MM).astype(int)
    o = SIDX[ch[k]] - s0 + KEEP
    ok = (o >= 0) & (o <= 2 * KEEP) & (s[k] >= 0) & (s[k] < NSMP)
    C = np.full((len(events), 2 * KEEP + 1, NSMP), np.nan, np.float32)
    C[e_i[ok], o[ok], s[k][ok]] = a[k][ok]
    return C


def stack_sums(C):
    """strip means over present events, per-strip baseline removed, then width sums."""
    with np.errstate(invalid='ignore'):
        M = np.nanmean(C, axis=0)                       # (offsets, samples)
    M = M - np.nanmean(M[:, PRE], axis=1, keepdims=True)
    return {h: np.nansum(M[KEEP - h:KEEP + h + 1], axis=0) for h in WIDTHS}


def metrics(S, t):
    """plateau level ratios relative to the 1080-1260 ns mean (just after the peak)."""
    ref = S[(t >= 1080) & (t <= 1260)].mean()
    out = {f'r{int(lo)}': float(S[(t >= lo) & (t < lo + 300)].mean() / ref)
           for lo in (1500, 1800, 2100, 2400, 3000, 3480)}
    out['peak'] = float(S.max())
    out['ref'] = float(ref)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache')
    ap.add_argument('--tag', default='masked')
    ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--max-events', type=int, default=20000)
    a = ap.parse_args()
    ev, ch, s, amp, lead, plat, twall = load(a.cache)
    t = np.arange(NSMP) * SNS
    rng = np.random.default_rng(1)
    res = {'t': t.tolist(), 'widths': list(WIDTHS)}
    labs = [lab for lab, _ in PLATEAUS]
    pcode = np.full(len(plat), -1, np.int8)                 # per EVENT, not per sample
    for i, lab in enumerate(labs):
        pcode[plat == lab] = i
    for i, (lab, E) in enumerate(PLATEAUS):
        events = np.unique(ev[pcode[ev] == i])[:a.max_events]
        r = {'E_Vcm': E, 'n': int(len(events))}
        for view in ('x', 'y'):
            C = cube(ev, ch, s, amp, lead, events, view)
            S = stack_sums(C)
            pres = np.isfinite(C[:, KEEP, :]).mean(0)
            r[view] = {'present_frac_centre': pres.tolist(),
                       'sum': {str(h): S[h].tolist() for h in WIDTHS},
                       'metrics': {str(h): metrics(S[h], t) for h in WIDTHS}}
            # bootstrap the ratios of every width
            boots = {h: [] for h in WIDTHS}
            curves = {h: [] for h in WIDTHS}
            for _ in range(a.nboot):
                Sb = stack_sums(C[rng.integers(0, len(C), len(C))])
                for h in WIDTHS:
                    boots[h].append(metrics(Sb[h], t))
                    ref = Sb[h][(t >= 1080) & (t <= 1260)].mean()
                    curves[h].append(Sb[h] / ref)
            # per-sample bootstrap spread of the curve normalised to its 1.08-1.26 us level
            r[view]['band'] = {str(h): np.std(curves[h], axis=0).tolist() for h in WIDTHS}
            r[view]['metrics_err'] = {str(h): {k: float(np.std([b[k] for b in boots[h]]))
                                               for k in boots[h][0]} for h in WIDTHS}
            m = r[view]['metrics']; e = r[view]['metrics_err']
            print(f'{lab} {view} n={len(events)}  ' + '  '.join(
                f'±{h}: r1800 {m[str(h)]["r1800"]:.3f}±{e[str(h)]["r1800"]:.3f} '
                f'r2400 {m[str(h)]["r2400"]:.3f}±{e[str(h)]["r2400"]:.3f}' for h in (2, 4, 8, 12)))
        res[lab] = r
    json.dump(res, open(os.path.join(HERE, 'results', f'headon_{a.tag}.json'), 'w'))


if __name__ == '__main__':
    main()
