#!/usr/bin/env python3
"""Closure: does a slimmed file still support the FULL match?

The slim can only use the global map (K, T0) -- the per-bunch clock fit needs
the matched sample and so cannot run before the slim exists. So the question is
whether a slim cut at +-W on the ARM-AGNOSTIC global prediction still contains
everything the downstream chain needs:

  1. the per-bunch (da_b, dk_b) fit, and
  2. the final +-25 ns accept, cross-validated,

reproducing the published 95.84 % / 0.049 % of DREAM_NTOF_CALIBRATION.md.

Run on the wall AND plastic candidate lists of the complete reference pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ntof_dream_merge' / 'match_study' / 'scripts'))

import study_common as sc            # noqa: E402
import window_scan as ws             # noqa: E402
import fit_perbunch as fp            # noqa: E402

WINDOW = 25.0
SHIFT = sc.SHIFT_NS                  # +100 us accidental control
SLIMS = (100.0, 150.0, 200.0, 250.0, 300.0, 500.0, None)


def per_event_nearest(ev, cd, arm_off_none, W, shift=0.0):
    """Nearest residual per event under the global map, after a +-W slim.

    The slim keeps candidates within +-W of the ARM-AGNOSTIC prediction; the
    residual is then reported on the arm-corrected map, which is what the fit
    and the accept use.
    """
    # arm-agnostic prediction decides membership of the slim
    ei, r_raw, ci = ws.residuals(ev['bunch'], ev['t'], cd['bunch'],
                                 cd['t_raw'], shift=shift, search=2000.0)
    keep = np.ones(r_raw.size, bool) if W is None else (np.abs(r_raw) <= W)
    # arm-corrected residual, for everything that survived
    return ei[keep], (cd['t'][ci[keep]] - cd['t_raw'][ci[keep]]) + r_raw[keep]


def perbunch_corr(t, bn, r_ev, r_res, n_ev, core=200.0):
    """Cross-validated per-bunch correction, as fit_perbunch.main() builds it."""
    best = np.full(n_ev, np.nan)
    if r_ev.size:
        o = np.argsort(np.abs(r_res))[::-1]
        best[r_ev[o]] = r_res[o]
    good = np.isfinite(best) & (np.abs(best) < core)
    corr = np.full(n_ev, np.nan)
    for b in np.unique(bn):
        idx = np.flatnonzero(bn == b)
        half = np.arange(idx.size) % 2
        for h in (0, 1):
            f = idx[(half == h) & good[idx]]
            e = idx[half == 1 - h]
            if f.size < fp.MIN_EVENTS:
                continue
            a, k, _ = fp._fit_bunch(t[f], best[f])
            if np.isfinite(a):
                corr[e] = a + k * t[e]
    return corr


def main():
    arm_off, tb = ws.apply_timebase('fitarm')
    print(f'global map K={tb["K"]:.6e}  T0={tb["T0"]:+.2f} ns  '
          f'arm offsets {tb["arm_offsets_ns"]}')
    print(f'slim keeps |t_cand - predict(t_dream, arm=None)| <= W\n')

    data = []
    for sub in sc.SUBRUNS:
        ev, cd = ws.load(sub, 'wp', '', arm_off)   # cd['t'] is arm-corrected
        raw = dict(np.load(sc.DATA / f'cand_{sub}_wp.npz'))
        # rebuild the un-shifted candidate time, aligned to cd's sort order
        cd['t_raw'] = cd['t'] + np.asarray(arm_off)[cd['arm']]
        data.append((sub, ev, cd))

    print(f'{"slim W":>8} {"eff@25ns":>10} {"accid":>9} {"purity":>10} '
          f'{"cands kept":>12} {"per trigger":>12}')
    print('-' * 66)
    for W in SLIMS:
        eff_n = eff_d = acc_n = 0
        nkept = 0
        for sub, ev, cd in data:
            n_ev = ev['t'].size
            ei, r = per_event_nearest(ev, cd, None, W)
            nkept += np.unique(np.stack([ei, np.round(r, 3)]), axis=1).shape[1]
            corr = perbunch_corr(ev['t'].astype(float), ev['bunch'],
                                 ei, r, n_ev)
            ok = np.isfinite(corr)
            rc = r - corr[ei]
            m = np.zeros(n_ev, bool)
            m[np.unique(ei[np.abs(rc) <= WINDOW])] = True
            eff_n += int((m & ok).sum())
            eff_d += int(ok.sum())

            eic, rcc = per_event_nearest(ev, cd, None, W, shift=SHIFT)
            rcc = rcc - corr[eic]
            mc = np.zeros(n_ev, bool)
            mc[np.unique(eic[np.abs(rcc) <= WINDOW])] = True
            acc_n += int((mc & ok).sum())
        eff, acc = eff_n / eff_d, acc_n / eff_d
        pur = 1 - acc / eff if eff else float('nan')
        lbl = 'no slim' if W is None else f'{W:g} ns'
        print(f'{lbl:>8} {eff:>9.4%} {acc:>9.4%} {pur:>9.4%} '
              f'{nkept:>12,} {nkept/eff_d:>12.2f}')


if __name__ == '__main__':
    main()
