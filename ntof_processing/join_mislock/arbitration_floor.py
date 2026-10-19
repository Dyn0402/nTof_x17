#!/usr/bin/env python3
"""Is there a segment length below which lock arbitration cannot work?

Asked by the recovery campaign 2026-08-12 after run_82's 5-minute scan
sub-runs (~50 clusters) came back AmbiguousLock with six locks whose
intensity correlations differ by ~0.001.

Method: take healthy sub-runs whose lock is confidently known, truncate to
the first N trigger clusters, and at each N measure both discriminants
against the supercycle-shifted impostor locks (true offset ± k x 39.6 s and
± k x 43.2 s, k = 1..3, refined):

  - count margin: matched clusters at the true lock minus the best impostor
  - r_sig: Fisher-z separation of the intensity correlation, exactly as
    select_lock computes it (best vs second among near-tied locks)

The floor is where BOTH stay below their acceptance thresholds
(MARGIN_CLEAR = 10, R_SIG = 3).

Usage: arbitration_floor.py [run subrun [run subrun ...]]
Env:   X17_BEAM_JULY (combined hits + beam CSVs)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ntof_july_analysis import pulse_match as pm            # noqa: E402

SC = [39.6, 43.2]                       # supercycle basic-period multiples
KS = [-3, -2, -1, 1, 2, 3]
NS = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 10 ** 9]

DEFAULT = [('run_79', 'stat090_0001'), ('run_96', 'stat090_0001'),
           ('run_86', 'stat090_0001')]


def lock_stats(c_t, sizes, anchor, pt, pe, off):
    off = pm._refine(c_t, anchor, pt, off)
    pick, _, m = pm._at_offset(c_t, anchor, pt, off)
    n = int(m.sum())
    r = float('nan')
    if n > 10:
        with np.errstate(invalid='ignore'):
            r = float(np.corrcoef(sizes[m], pe[pick[m]])[0, 1])
    return off, n, r


def fisher_sig(ra, na, rb, nb):
    za = np.arctanh(np.clip(ra, -0.999999, 0.999999))
    zb = np.arctanh(np.clip(rb, -0.999999, 0.999999))
    se = np.sqrt(1 / max(na - 3, 1) + 1 / max(nb - 3, 1))
    return float((za - zb) / se)


def study(run, subrun):
    eid, t_rel, anchor = pm._event_times(run, subrun)
    if eid is None:
        print(f'{run}/{subrun}: no data')
        return None
    starts = np.concatenate([[0], np.where(np.diff(t_rel) > pm.GAP_S)[0] + 1])
    c_t = t_rel[starts]
    sizes = np.diff(np.r_[starts, len(t_rel)]).astype(float)
    span = float(t_rel.max())
    pt, pe = pm._load_pulses([anchor + s for s in
                              np.arange(0, span + 600 + 43200, 43200)])
    true_off, locks, diag = pm.select_lock(c_t, sizes, anchor, pt, pe)
    print(f'\n== {run}/{subrun}: {len(c_t)} clusters over '
          f'{span / 60:.1f} min; true lock {true_off:+.2f} s '
          f'(margin {diag["margin"]}, by {diag["chosen_by"]})')

    imposters = sorted({round(true_off + k * sc, 3)
                        for sc in SC for k in KS
                        if abs(true_off + k * sc) <= pm.SEARCH_S})
    rows = []
    print(f'{"N":>6} {"min":>6} {"n_true":>6} {"margin":>6} {"r_true":>7} '
          f'{"r_best_wrong":>12} {"r_sig":>6}  verdict')
    for N in NS:
        cN = c_t[:N]
        sN = sizes[:N]
        if len(cN) < 15:
            continue
        _, n_t, r_t = lock_stats(cN, sN, anchor, pt, pe, true_off)
        wrong = [lock_stats(cN, sN, anchor, pt, pe, o) for o in imposters]
        # impostors that survive dedupe against the true lock
        wrong = [(o, n, r) for o, n, r in wrong
                 if abs(o - true_off) >= pm.LOCK_GROUP_S and n > 10]
        if not wrong:
            continue
        n_w = max(n for _, n, _ in wrong)
        margin = n_t - n_w
        # near-tied contenders, as select_lock defines them
        cont = [(o, n, r) for o, n, r in wrong
                if n_t - n < pm.MARGIN_CLEAR and np.isfinite(r)]
        r_w = max((r for _, _, r in cont), default=float('nan'))
        sig = (fisher_sig(r_t, n_t, r_w, max(n for _, n, r in cont
                                             if r == r_w))
               if cont and np.isfinite(r_t) else float('nan'))
        if margin >= pm.MARGIN_CLEAR:
            verdict = 'count'
        elif np.isfinite(sig) and sig >= pm.R_SIG:
            verdict = 'intensity'
        else:
            verdict = 'AMBIGUOUS'
        nn = len(cN)
        rows.append(dict(N=nn, span_min=float(cN[-1] - cN[0]) / 60,
                         n_true=n_t, margin=margin, r_true=r_t,
                         r_wrong=r_w, r_sig=sig, verdict=verdict))
        print(f'{nn:>6} {rows[-1]["span_min"]:>6.1f} {n_t:>6} {margin:>6} '
              f'{r_t:>7.3f} {r_w:>12.3f} '
              f'{sig if np.isfinite(sig) else float("nan"):>6.2f}  {verdict}')
    return dict(run=run, subrun=subrun, true_off=true_off, rows=rows)


def main():
    args = sys.argv[1:]
    pairs = (list(zip(args[0::2], args[1::2])) if args else DEFAULT)
    out = [r for r in (study(*p) for p in pairs) if r]
    with open('arbitration_floor.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('\nDONE -> arbitration_floor.json')


if __name__ == '__main__':
    main()
