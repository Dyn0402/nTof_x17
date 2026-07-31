#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
window_scan.py -- how the DREAM<->n_TOF match trades misses against fakes as the
accept window opens.

For every DREAM trigger this takes the reconstructed n_TOF sector SINGLES of the
same bunch (built by build_candidates.py) and records the residual

    r = t_singles - (t_DREAM + K t_DREAM + T0)

for every candidate within +-2 us, not just the nearest one -- so any accept
window can afterwards be evaluated exactly, without touching the ROOT files
again. The same thing is done with the DREAM time shifted by +100 us, which
destroys every real coincidence while leaving the local singles rate untouched:
that is the CONTROL, and it measures the chance of matching by accident.

From the two residual sets:

    efficiency(W)  fraction of DREAM triggers with a candidate inside W
    false(W)       the same on the control -- P(an event with no true partner is
                   matched anyway)
    purity(W)      1 - false(W) (1 - eff_true(W)) / eff(W), with
                   eff_true = (eff - false)/(1 - false); i.e. of the events the
                   window matches, the fraction whose match is the real one

Both are computed globally, per time-since-flash bin (the n_TOF singles rate
falls by ~100x between 1 ms and 80 ms, so the best window is not the same at
both ends), per arm, and for both legs of the trigger: wall-only, and the full
wall .AND. plastic SINGLES.

Windows scanned:
  * symmetric,  |r| <= w
  * the main band alone, and the main band plus the delayed wall satellite,
    each with its own half-width
  * the greedy S/B window: bins of the residual histogram ranked by
    (signal - control)/control and added one at a time. This is the best any
    window can do at a given efficiency, so it says how much the simple shapes
    leave on the table.

USAGE
    python window_scan.py [--subs stat090_0000,stat090_0001] [--legs wp,w]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from study_common import (DATA, K, T0, BANDS, SHIFT_NS, SUBRUNS, predicted_time)

SEARCH_NS = 2_000.0          # kept per event; wide enough to see the flat floor
KEY_SCALE = 1e9
T_BINS = ((1, 3), (3, 10), (10, 20), (20, 40), (40, 80))
ARMS = ('A', 'B', 'C', 'D')

# Half-widths scanned, log-spaced from 1 ns to 2 us.
WINDOWS = np.unique(np.round(np.geomspace(1.0, 2000.0, 90), 1))


def _pack(bunch, t):
    return np.asarray(bunch, np.float64) * KEY_SCALE + np.asarray(t, np.float64)


def residuals(ev_bunch, ev_t, c_bunch, c_t, shift=0.0, search=SEARCH_NS,
              extra=None):
    """(event index, residual, candidate index) for every candidate within
    +-search of an event's predicted position, same bunch.

    c_bunch/c_t must be sorted by (bunch, t) -- build_candidates writes them so.
    `extra` is a per-event correction added to the prediction (the per-bunch
    clock fit of fit_perbunch.py).
    """
    pred = predicted_time(np.asarray(ev_t, float), shift)
    if extra is not None:
        pred = pred + np.asarray(extra, float)
    key_c = _pack(c_bunch, c_t)
    lo = np.searchsorted(key_c, _pack(ev_bunch, pred - search), side='left')
    hi = np.searchsorted(key_c, _pack(ev_bunch, pred + search), side='right')
    n = hi - lo
    total = int(n.sum())
    if total == 0:
        return (np.array([], np.int64), np.array([]), np.array([], np.int64))
    ev_idx = np.repeat(np.arange(len(pred)), n)
    ci = np.repeat(lo, n) + (np.arange(total) - np.repeat(np.cumsum(n) - n, n))
    return ev_idx, c_t[ci] - np.repeat(pred, n), ci


def _matched(n_ev, ev_idx, res, window):
    """Boolean per event: any residual inside `window` (a list of (lo, hi))."""
    m = np.zeros(res.size, bool)
    for lo, hi in window:
        m |= (res >= lo) & (res <= hi)
    out = np.zeros(n_ev, bool)
    if m.any():
        out[np.unique(ev_idx[m])] = True
    return out


def _nearest_abs(n_ev, ev_idx, res):
    """Per event, the smallest |residual| (inf where the event has none)."""
    out = np.full(n_ev, np.inf)
    np.minimum.at(out, ev_idx, np.abs(res))
    return out


def multiplicity(n_ev, ev_idx, res, arm, window):
    """How ambiguous a match is: candidates per matched event, and whether they
    come from more than one arm.

    Being "matched" is not the same as knowing WHICH n_TOF coincidence fired the
    trigger. Two candidates in the window means the wall time, the amplitude and
    the arm are all a coin toss, and the arm is what the Micromegas cross-check
    and any wall-pointing analysis keys on.
    """
    m = np.zeros(res.size, bool)
    for lo, hi in window:
        m |= (res >= lo) & (res <= hi)
    idx, ar = ev_idx[m], arm[m]
    n = np.bincount(idx, minlength=n_ev)
    pair = np.unique(idx.astype(np.int64) * 4 + ar.astype(np.int64))
    n_arm = np.bincount((pair // 4).astype(np.int64), minlength=n_ev)
    hit = n > 0
    return dict(mean_n=float(n[hit].mean()) if hit.any() else np.nan,
                frac_multi=float((n[hit] > 1).mean()) if hit.any() else np.nan,
                frac_multi_arm=float((n_arm[hit] > 1).mean()) if hit.any() else np.nan)


def purity(eff, false):
    """Fraction of MATCHED events whose match is the true partner.

    eff = eff_true + (1 - eff_true) false  (an event with no true partner in the
    window is still matched with probability `false`), so eff_true =
    (eff - false)/(1 - false) and the contamination of the matched sample is
    (eff - eff_true)/eff.
    """
    eff = np.asarray(eff, float)
    false = np.asarray(false, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        eff_true = (eff - false) / (1 - false)
        p = np.where(eff > 0, eff_true / eff, np.nan)
    return np.clip(p, 0.0, 1.0), np.clip(eff_true, 0.0, 1.0)


def scan_symmetric(n_ev, sig, ctl, mask=None, windows=WINDOWS):
    """eff/false/purity vs symmetric half-width, exactly (nearest |r| suffices)."""
    ns = _nearest_abs(n_ev, *sig[:2])
    nc = _nearest_abs(n_ev, *ctl[:2])
    if mask is not None:
        ns, nc, n = ns[mask], nc[mask], int(mask.sum())
    else:
        n = n_ev
    if n == 0:
        z = np.full(len(windows), np.nan)
        return dict(w=windows, n=0, eff=z, false=z, purity=z, eff_true=z)
    eff = np.array([(ns <= w).mean() for w in windows])
    false = np.array([(nc <= w).mean() for w in windows])
    p, et = purity(eff, false)
    return dict(w=windows, n=n, eff=eff, false=false, purity=p, eff_true=et)


def scan_bands(n_ev, sig, ctl, sat_centre, mask=None, windows=WINDOWS):
    """Same, for a main band plus a satellite band of the same half-width."""
    n = n_ev if mask is None else int(mask.sum())
    eff, false = [], []
    for w in windows:
        win = [(-w, w), (sat_centre - w, sat_centre + w)]
        ms = _matched(n_ev, sig[0], sig[1], win)
        mc = _matched(n_ev, ctl[0], ctl[1], win)
        if mask is not None:
            ms, mc = ms[mask], mc[mask]
        eff.append(ms.mean() if n else np.nan)
        false.append(mc.mean() if n else np.nan)
    eff, false = np.array(eff), np.array(false)
    p, et = purity(eff, false)
    return dict(w=windows, n=n, eff=eff, false=false, purity=p, eff_true=et,
                sat_centre=sat_centre)


def greedy_window(n_ev, sig, ctl, mask=None, bin_ns=5.0, rng=SEARCH_NS):
    """Rank residual bins by excess-over-control and add them one at a time.

    The resulting (efficiency, purity) trace is the frontier no window of any
    shape can beat with these bins, so it bounds what window tuning can buy.
    """
    edges = np.arange(-rng, rng + bin_ns, bin_ns)
    keep_s = np.ones(sig[0].size, bool)
    keep_c = np.ones(ctl[0].size, bool)
    if mask is not None:
        keep_s = mask[sig[0]]
        keep_c = mask[ctl[0]]
    hs = np.histogram(sig[1][keep_s], bins=edges)[0].astype(float)
    hc = np.histogram(ctl[1][keep_c], bins=edges)[0].astype(float)
    order = np.argsort(-(hs - hc) / np.maximum(hc, 1.0))
    n = n_ev if mask is None else int(mask.sum())
    eff, false, nb = [], [], []
    chosen = np.zeros(len(edges) - 1, bool)
    # add bins in blocks so the trace is cheap but still fine-grained
    for stop in np.unique(np.round(np.geomspace(1, len(order), 60)).astype(int)):
        chosen[order[:stop]] = True
        win = _bins_to_intervals(edges, chosen)
        ms = _matched(n_ev, sig[0], sig[1], win)
        mc = _matched(n_ev, ctl[0], ctl[1], win)
        if mask is not None:
            ms, mc = ms[mask], mc[mask]
        eff.append(ms.mean() if n else np.nan)
        false.append(mc.mean() if n else np.nan)
        nb.append(int(stop))
    eff, false = np.array(eff), np.array(false)
    p, et = purity(eff, false)
    best = _bins_to_intervals(edges, np.isin(np.arange(len(chosen)),
                                             order[:nb[int(np.nanargmax(
                                                 np.array(eff) * p))]]))
    return dict(n_bins=np.array(nb), n=n, eff=eff, false=false, purity=p,
                eff_true=et, bin_ns=bin_ns, best_window=best)


def _bins_to_intervals(edges, chosen):
    """Merge selected histogram bins into a list of (lo, hi) intervals."""
    out, i = [], 0
    c = np.asarray(chosen)
    while i < c.size:
        if not c[i]:
            i += 1
            continue
        j = i
        while j + 1 < c.size and c[j + 1]:
            j += 1
        out.append((float(edges[i]), float(edges[j + 1])))
        i = j + 1
    return out


def load(sub, leg, tag='', arm_off=None):
    ev = np.load(DATA / f'events_{sub}{tag}.npz')
    cd = dict(np.load(DATA / f'cand_{sub}_{leg}{tag}.npz'))
    if arm_off is not None:
        # Put every arm on one time base by moving the CANDIDATES, so a single
        # (K, T0) still describes the whole sample. Re-sort: the shift is a few
        # ns and the arrays must stay ordered by (bunch, t) for the search.
        cd['t'] = cd['t'] - np.asarray(arm_off, float)[cd['arm']]
        o = np.lexsort((cd['t'], cd['bunch']))
        cd = {k: (v[o] if v.shape[:1] == o.shape else v) for k, v in cd.items()}
    return ev, cd


def apply_timebase(mode: str):
    """Set (K, T0) and the per-arm candidate offsets for `mode`.

    legacy  the constants the merge was built with, fitted on the OFFICIAL
            processing (K = 1.089e-4, T0 = -197.5 ns)
    fit     re-fitted on this candidate processing by fit_timebase.py
    fitarm  the same, plus the per-arm offset -- the four arms' flash times
            differ by ~25 ns, which is invisible at +-150 ns and dominant below
            +-50 ns
    """
    import study_common
    if mode == 'legacy':
        return None, dict(K=study_common.K, T0=study_common.T0)
    if mode == 'perbunch':
        arm, info = apply_timebase('fitarm')
        info['perbunch'] = True
        return arm, info
    p = DATA / 'timebase.json'
    if not p.exists():
        raise SystemExit('run fit_timebase.py first (data/timebase.json missing)')
    tb = json.loads(p.read_text())
    study_common.K = tb['fitted']['K']
    study_common.T0 = tb['fitted']['T0']
    info = dict(K=study_common.K, T0=study_common.T0)
    if mode != 'fitarm':
        return None, info
    off = [float(np.mean([v['a'] for v in tb['per_arm'][a].values()]))
           for a in ARMS]
    info['arm_offsets_ns'] = {a: o for a, o in zip(ARMS, off)}
    return np.array(off), info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--subs', default=','.join(SUBRUNS))
    ap.add_argument('--legs', default='wp,w')
    ap.add_argument('--tag', default='')
    ap.add_argument('--timebase', default='fitarm',
                    choices=('legacy', 'fit', 'fitarm', 'perbunch'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out_path = args.out or str(DATA / f'window_scan_{args.timebase}.npz')

    arm_off, tb_info = apply_timebase(args.timebase)
    print(f'time map [{args.timebase}]: K = {tb_info["K"]:.6e}, '
          f'T0 = {tb_info["T0"]:.2f} ns'
          + (f", arm offsets {tb_info['arm_offsets_ns']}"
             if 'arm_offsets_ns' in tb_info else ''))

    subs = args.subs.split(',')
    legs = args.legs.split(',')
    store, summary = {}, {}

    for leg in legs:
        # C2 is a second control at -100 us. The +100 us one could in principle
        # be biased at the end of the acquisition window, where a forward shift
        # runs out of data; if the two agree, it is not.
        E, S, C, C2 = [], [], [], []
        off = 0
        for sub in subs:
            ev, cd = load(sub, leg, args.tag, arm_off)
            eb, et = ev['bunch'], ev['t']
            extra = None
            if args.timebase == 'perbunch':
                # always the 'wp' fit, for both legs: it is the same wall time
                # either way, and the wall-AND-plastic leg is the clean sample to
                # fit a clock on (0.5 % accidental against 9 %).
                z = np.load(DATA / f'perbunch_corr_{sub}_wp.npz')
                extra = z['corr_cv']
                cov = float(np.isfinite(extra).mean())
                # events whose bunch could not be fitted keep the global map --
                # they stay in the denominator, so the efficiency below is the
                # honest one for the whole sub-run
                extra = np.nan_to_num(extra)
                print(f'  per-bunch correction covers {cov:.2%} of events')
            sig = residuals(eb, et, cd['bunch'], cd['t'], extra=extra)
            ctl = residuals(eb, et, cd['bunch'], cd['t'], shift=SHIFT_NS,
                            extra=extra)
            ct2 = residuals(eb, et, cd['bunch'], cd['t'], shift=-SHIFT_NS,
                            extra=extra)
            S.append((sig[0] + off, sig[1], cd['arm'][sig[2]]))
            C.append((ctl[0] + off, ctl[1], cd['arm'][ctl[2]]))
            C2.append((ct2[0] + off, ct2[1], cd['arm'][ct2[2]]))
            E.append(et)
            off += et.size
            print(f'{sub} [{leg}]: {et.size:,} events, {cd["t"].size:,} candidates, '
                  f'{sig[1].size:,} residuals in +-{SEARCH_NS:.0f} ns '
                  f'({ctl[1].size:,} control)')
        ets = np.concatenate(E)
        n_ev = ets.size
        sig = tuple(np.concatenate([s[i] for s in S]) for i in range(3))
        ctl = tuple(np.concatenate([c[i] for c in C]) for i in range(3))
        ct2 = tuple(np.concatenate([c[i] for c in C2]) for i in range(3))

        # residual histograms, for the figures
        edges = np.arange(-SEARCH_NS, SEARCH_NS + 2.0, 2.0)
        store[f'{leg}/edges'] = edges
        store[f'{leg}/hist_sig'] = np.histogram(sig[1], bins=edges)[0]
        store[f'{leg}/hist_ctl'] = np.histogram(ctl[1], bins=edges)[0]
        for lo, hi in T_BINS:
            m = (ets >= lo * 1e6) & (ets < hi * 1e6)
            store[f'{leg}/hist_sig_{lo}_{hi}'] = np.histogram(
                sig[1][m[sig[0]]], bins=edges)[0]
            store[f'{leg}/hist_ctl_{lo}_{hi}'] = np.histogram(
                ctl[1][m[ctl[0]]], bins=edges)[0]

        # residual vs time-since-flash: the 108.9 ppm clock correction is 8.7 us
        # at 80 ms, so a flat band here is the proof that K and T0 are right
        tedges = np.geomspace(1e6, 8e7, 60)
        redges = np.arange(-600.0, 600.0 + 4.0, 4.0)
        store[f'{leg}/h2_tedges'] = tedges
        store[f'{leg}/h2_redges'] = redges
        store[f'{leg}/h2'] = np.histogram2d(ets[sig[0]], sig[1],
                                            bins=(tedges, redges))[0]

        # where the two bands actually sit, measured not assumed
        h = store[f'{leg}/hist_sig'] - store[f'{leg}/hist_ctl']
        c = 0.5 * (edges[1:] + edges[:-1])
        main_pk = float(c[np.abs(c) < 200][h[np.abs(c) < 200].argmax()])
        s = (c > 150) & (c < 700)
        sat_pk = float(c[s][h[s].argmax()])
        print(f'  [{leg}] main band peak {main_pk:+.1f} ns, '
              f'satellite peak {sat_pk:+.1f} ns')

        def record(name, d):
            for k, v in d.items():
                if k == 'best_window':
                    store[f'{leg}/{name}/{k}'] = np.array(v, float).reshape(-1, 2)
                else:
                    store[f'{leg}/{name}/{k}'] = np.asarray(v)

        record('sym', scan_symmetric(n_ev, sig, ctl))
        record('band', scan_bands(n_ev, sig, ctl, sat_pk))
        record('greedy', greedy_window(n_ev, sig, ctl))
        for lo, hi in T_BINS:
            m = (ets >= lo * 1e6) & (ets < hi * 1e6)
            record(f'sym_t{lo}_{hi}', scan_symmetric(n_ev, sig, ctl, mask=m))
            record(f'band_t{lo}_{hi}', scan_bands(n_ev, sig, ctl, sat_pk, mask=m))
        for ai, a in enumerate(ARMS):
            # an event is "arm a" if its candidates in the search window are from
            # arm a; the scan below simply restricts the candidates, not the events
            sa = tuple(x[sig[2] == ai] for x in sig)
            ca = tuple(x[ctl[2] == ai] for x in ctl)
            record(f'sym_arm{a}', scan_symmetric(n_ev, sa, ca))

        # the operating points worth quoting
        pts, masks = {}, {}
        for name, win in (('current_bands', list(BANDS)),
                          ('main_only_150', [(-150.0, 150.0)]),
                          ('sym_100', [(-100.0, 100.0)]),
                          ('sym_75', [(-75.0, 75.0)]),
                          ('tight_50', [(-50.0, 50.0)]),
                          ('tight_25', [(-25.0, 25.0)]),
                          ('tight_15', [(-15.0, 15.0)]),
                          ('tight_10', [(-10.0, 10.0)]),
                          ('asym_100_60', [(-100.0, 60.0)]),
                          ('asym_75_45', [(-75.0, 45.0)])):
            ms = _matched(n_ev, sig[0], sig[1], win)
            mc = _matched(n_ev, ctl[0], ctl[1], win)
            p, et_ = purity(ms.mean(), mc.mean())
            row = dict(window=win, eff=float(ms.mean()), false=float(mc.mean()),
                       purity=float(p), eff_true=float(et_))
            row.update(multiplicity(n_ev, sig[0], sig[1], sig[2], win))
            row['false_minus'] = float(
                _matched(n_ev, ct2[0], ct2[1], win).mean())
            row['per_t'] = {}
            for lo, hi in T_BINS:
                m = (ets >= lo * 1e6) & (ets < hi * 1e6)
                if m.sum() == 0:
                    continue
                pp, ee = purity(ms[m].mean(), mc[m].mean())
                row['per_t'][f'{lo}-{hi}'] = dict(
                    n=int(m.sum()), eff=float(ms[m].mean()),
                    false=float(mc[m].mean()), purity=float(pp),
                    eff_true=float(ee))
            pts[name] = row
            masks[name] = ms
        summary[leg] = dict(n_events=int(n_ev), main_peak=main_pk,
                            sat_peak=sat_pk, points=pts,
                            n_candidates=int(sig[1].size))
        store[f'{leg}/t_event'] = ets
        for name, m in masks.items():
            store[f'{leg}/matched_{name}'] = m

    # the question the merge actually asks: given a wall coincidence, is the
    # plastic partner there? Both legs are matched on the same events in the
    # same window, so this is a straight conditional.
    if 'wp' in legs and 'w' in legs:
        for name in summary['wp']['points']:
            mw = store[f'w/matched_{name}']
            mp = store[f'wp/matched_{name}']
            summary['wp']['points'][name]['plastic_given_wall'] = (
                float((mw & mp).sum() / max(mw.sum(), 1)))
            summary['wp']['points'][name]['wall_only_no_plastic'] = (
                float((mw & ~mp).mean()))

    np.savez_compressed(out_path, **store)
    with open(DATA / f'window_scan_summary_{args.timebase}.json', 'w') as f:
        json.dump(dict(subs=subs, timebase=args.timebase, **tb_info,
                       search_ns=SEARCH_NS, legs=summary), f, indent=1,
                  default=float)

    print('\noperating points (both sub-runs, all arms):')
    for leg in legs:
        print(f'\n  leg {leg}  ({summary[leg]["n_events"]:,} DREAM events)')
        print('    window                       eff     false    purity  eff_true'
              '   >1cand  >1arm')
        for name, r in summary[leg]['points'].items():
            w = ' '.join(f'[{a:+.0f},{b:+.0f}]' for a, b in r['window'])
            print(f'    {name:<16} {w:<22} {r["eff"]:6.1%} {r["false"]:8.2%} '
                  f'{r["purity"]:8.2%} {r["eff_true"]:8.1%} '
                  f'{r["frac_multi"]:8.2%} {r["frac_multi_arm"]:6.2%}')
        if 'plastic_given_wall' in summary[leg]['points']['current_bands']:
            r = summary[leg]['points']['current_bands']
            print(f'    plastic partner present for '
                  f'{r["plastic_given_wall"]:.2%} of wall-matched triggers '
                  f'(current bands)')
    print(f'\n-> {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
