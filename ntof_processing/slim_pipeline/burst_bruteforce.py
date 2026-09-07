#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
burst_bruteforce.py -- pair ONE DREAM burst with ONE n_TOF bunch by exhaustion.

The production chain locks a whole sub-run at once (pulse_match count scan ->
coincidence_arbiter on 32 sampled pulses -> clock fit on the segment). Every
step of that needs statistics a lone burst does not have, so the last few
hundred unmatched pulses of the campaign are all in the same place: a handful
of real bursts at the tail of a sub-run (run_82/stat090_h2i5_0007,
run_110/stat090_0003), or a single burst inside a matched sub-run whose
assigned bunch shows no coincidence (the LOW_COINC outliers). Nothing in the
chain can look at one burst and ask "which bunch, at what offset, holds its
partner hits?" -- this does, by trying all of them.

THE MEASUREMENT. For a burst with triggers t_i (DREAM ns since its flash) and
a bunch with wall-AND-plastic singles candidates c_k (n_TOF ns since ITS
flash), every difference c_k - t_i (1 + K_SEED) is histogrammed at 2 us over
+-100 ms. At the RIGHT pairing the ~90 triggers of a beam pulse pile 60-96 %
of themselves into one bin (the whole burst shares one T0, and K_SEED is good
to ~2 us over 80 ms). A WRONG pairing is NOT flat -- both series are dense in
the same places (flash region, thermal peak), and at the reference pair a
wrong bunch still puts ~0.4 of the triggers into its tallest bin -- so the
coarse peak only chooses who gets refined (the top REFINE_TOP bunches by
peak), and its significance is quoted against the LOCAL floor. The decision
is then made at 20 ns: a per-burst (da, dk) line is fitted on the core
residuals exactly as `clockfit.fit_perbunch` would, and the fraction of
triggers with a candidate inside +-C.ACCEPT_NS of the corrected prediction is
the ranking statistic -- 78-99 % at the right bunch, 2-6 % at the runner-up
on every burst scanned on 2026-08-16. It is the SAME per-pulse coincidence
fraction the ledger and clock_qa use, so a burst this tool calls matched is
matched by the production definition and not by a looser one.

WHAT IT FOUND (2026-08-16). Two kinds of recoverable burst, both handed to
production through burst_fixes.json (see slim.apply_fixes / apply_burst_fix):
  * sub-run tails of 6-17 real bursts (run_82/stat090_h2i5_0007 x 224581,
    run_110/stat090_0003 x 224625): every burst pairs with the run's last
    bunches at one consistent implied offset -> a whole-segment lock, with the
    clock bootstrap's peak floor lowered for the small sample;
  * single LOW_COINC bursts inside matched sub-runs whose winning lag differs
    from their neighbours' by ~1.0 ms (run_124/0006, run_124/0010, run_130/
    0000: the flash trigger was dropped and the first single ~1 ms later was
    tagged as the flash) or by 4.4 ms (run_102/0002 burst 0, recorded
    mid-gate) -> a per-burst flash re-reference; and one burst the join put
    three pulses off (run_118/0005 burst 1212, bunch 680 -> 677) -> a bunch
    override. Bursts of 11-14 triggers with no partner anywhere in +-120 s x
    +-100 ms stay unmatched, honestly.

CONTROLS. With --controls N, N bursts the sub-run already matched are scanned
the same way, blind, and their winners are printed beside the bunch the
production join gave them. A tool that cannot re-find the known pairings has
no business proposing new ones.

    python3 burst_bruteforce.py run_82 stat090_h2i5_0007 224581 \\
        --ntof-source /scratch/ntof --window-s 400 --out bf.json
    python3 burst_bruteforce.py run_124 stat090_0006 224654 --bursts 521 \\
        --controls 4 --window-s 120

Output: a JSON with, per burst, the ranked bunches (z, coarse peak, fraction),
the refined match of the winner, and -- for a sub-run with no lock -- the
pulse_match offset each burst implies, so a consistent set can be handed to
`Segment(accept_offset_s=...)`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / 'ntof_july_analysis'))

from ntof_processing.slim_pipeline import clockfit as cf      # noqa: E402
from ntof_processing.slim_pipeline import config as C          # noqa: E402
from ntof_processing.slim_pipeline.slim import (               # noqa: E402
    OFFSET_BUNCHES, Segment, _bind_ntof, pass1_candidates)

COARSE_BIN_NS = 2000.0        # K_SEED is good to ~2 us over the 80 ms burst
COARSE_HALF_NS = 100e6        # +-100 ms: wider than the burst, so a flash
                              # mis-tag of up to a whole gate is still seen
FINE_BIN_NS = 20.0
FINE_HALF_NS = 20_000.0       # around the coarse peak
CORE_NS = 250.0               # residuals inside this feed the (da, dk) line
MIN_LINE = 15                 # fewer core points than this: da = median, dk = 0
CTL_SHIFT_NS = 100_000.0      # accidental control, as clockfit.efficiency
REFINE_TOP = 12               # coarse candidates that get the fine treatment


def dream_bursts(run: str, subrun: str):
    """{burst_id: dict(t=ns since flash (no flash), flash_ns, n)} + anchor."""
    import pulse_match as pm
    from ntof_dream_merge.bunch_join import dream_events
    ev = dream_events(run, subrun)
    anchor = pm._anchor_epoch(run, subrun)
    t0 = int(ev['trigger_ns'].min())
    out = {}
    for b, g in ev.groupby('burst_id'):
        fl = g.loc[g['is_flash'], 'trigger_ns']
        flash_ns = int(fl.iloc[0]) if len(fl) else int(g['trigger_ns'].min())
        t = g.loc[~g['is_flash'], 't_since_flash_ns'].to_numpy().astype(np.float64)
        out[int(b)] = dict(t=t, flash_ns=flash_ns, n=int(t.size),
                           rel_s=(flash_ns - t0) / 1e9,
                           eid=g.loc[~g['is_flash'], 'eventId'].to_numpy())
    return out, anchor, ev


def subrun_lock(run: str, subrun: str):
    """pulse_match's own lock for the sub-run, or None if it has none."""
    import pulse_match as pm
    try:
        d = pm.match_subrun(run, subrun)
    except Exception as e:                                   # noqa: BLE001
        print(f'  pulse_match: no lock ({type(e).__name__}: {str(e)[:100]})')
        return None
    if not d or d.get('offset_s') is None:
        return None
    return float(d['offset_s'])


def coarse_scan(te, tc, K=cf.K_SEED, bin_ns=COARSE_BIN_NS, half=COARSE_HALF_NS,
                local_ns=200_000.0, gap_ns=6_000.0):
    """(peak lag ns, peak counts, LOCAL floor, z) of the c - t(1+K) histogram.

    The floor is taken from the peak's own neighbourhood (+-local_ns, minus
    +-gap_ns around it), NOT the whole +-100 ms: both time series are strongly
    structured in time-since-flash (the flash region, the thermal peak), so a
    wrong pairing is not flat -- it piles up wherever both distributions are
    dense, ~0.4 of the triggers per 2 us bin at the reference pair. A global
    median calls that a 50-sigma peak. Against its neighbours it is nothing.
    """
    if te.size == 0 or tc.size == 0:
        return None
    d = (tc[None, :] - te[:, None] * (1.0 + K)).ravel()
    d = d[np.abs(d) < half]
    if d.size == 0:
        return None
    nb = int(2 * half / bin_ns)
    idx = ((d + half) / bin_ns).astype(np.int64)
    idx = np.clip(idx, 0, nb - 1)
    h = np.bincount(idx, minlength=nb)
    i = int(h.argmax())
    w, g = int(local_ns / bin_ns), int(gap_ns / bin_ns)
    lo, hi = max(0, i - w), min(nb, i + w + 1)
    nbr = np.r_[h[lo:max(lo, i - g)], h[min(hi, i + g + 1):hi]]
    floor = float(np.median(nbr)) if nbr.size else 0.0
    z = (h[i] - floor) / np.sqrt(max(floor, 1.0))
    lag = -half + (i + 0.5) * bin_ns
    return dict(lag_ns=float(lag), peak=int(h[i]), floor=floor,
                z=float(z), n_pairs=int(d.size), nb=nb)


def refine(te, tc, lag_ns, K=cf.K_SEED, accept_ns=C.ACCEPT_NS):
    """Fine T0, per-burst (da, dk), and the +-accept coincidence fraction."""
    d = tc[None, :] - te[:, None] * (1.0 + K)
    m = np.abs(d - lag_ns) < FINE_HALF_NS
    if not m.any():
        return None
    dd = d[m]
    edges = np.arange(lag_ns - FINE_HALF_NS, lag_ns + FINE_HALF_NS + FINE_BIN_NS,
                      FINE_BIN_NS)
    h, _ = np.histogram(dd, bins=edges)
    i = int(h.argmax())
    T0 = 0.5 * (edges[i] + edges[i + 1])
    # per-burst line on the core residuals, nearest candidate per trigger
    r = d - T0
    ai = np.argmin(np.abs(r), axis=1)
    best = r[np.arange(te.size), ai]
    core = np.abs(best) < CORE_NS
    if core.sum() >= MIN_LINE:
        dk, da = np.polyfit(te[core], best[core], 1)
        keep = core & (np.abs(best - (da + dk * te)) < 100.0)
        if keep.sum() >= MIN_LINE:
            dk, da = np.polyfit(te[keep], best[keep], 1)
    elif core.sum() >= 3:
        da, dk = float(np.median(best[core])), 0.0
    else:
        da, dk = 0.0, 0.0
    corr = da + dk * te
    rc = r - corr[:, None]
    hit = (np.abs(rc) <= accept_ns).any(axis=1)
    ctl = (np.abs(rc - CTL_SHIFT_NS) <= accept_ns).any(axis=1)
    resid = rc[np.arange(te.size), np.argmin(np.abs(rc), axis=1)]
    return dict(T0_ns=float(T0), fine_peak=int(h[i]), da_ns=float(da),
                dk=float(dk), n_core=int(core.sum()),
                frac=float(hit.mean()), n_hit=int(hit.sum()),
                ctl_frac=float(ctl.mean()),
                resid_med_ns=float(np.median(resid[hit])) if hit.any() else None,
                resid_mad_ns=float(1.4826 * np.median(np.abs(
                    resid[hit] - np.median(resid[hit])))) if hit.sum() > 2 else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dream_run')
    ap.add_argument('dream_subrun')
    ap.add_argument('ntof_run', type=int)
    ap.add_argument('--ntof-source', default=None)
    ap.add_argument('--bursts', default=None,
                    help='comma list of burst ids; default every burst with '
                         '>= --min-trig triggers')
    ap.add_argument('--min-trig', type=int, default=10)
    ap.add_argument('--controls', type=int, default=0,
                    help='also scan this many bursts the sub-run already '
                         'matched (needs a lock), spread over the sub-run')
    ap.add_argument('--window-s', type=float, default=400.0,
                    help='bunches within +-this of each burst\'s estimated '
                         'epoch (estimate = anchor + rel + lock, or anchor + '
                         'rel if the sub-run has no lock)')
    ap.add_argument('--all-bunches', action='store_true')
    ap.add_argument('--offset-guess', type=float, default=None,
                    help='use this pulse_match offset for the epoch estimate '
                         'instead of the sub-run lock / 0')
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    t_start = time.time()
    run, sub, nt = a.dream_run, a.dream_subrun, a.ntof_run
    print(f'== brute force {run}/{sub} x n_TOF {nt}')
    bursts, anchor, ev = dream_bursts(run, sub)
    lock = a.offset_guess if a.offset_guess is not None else subrun_lock(run, sub)
    print(f'  {len(bursts)} bursts, anchor {anchor:.0f}, lock '
          f'{"none" if lock is None else format(lock, "+.3f")} s')

    if a.bursts:
        want = [int(x) for x in a.bursts.split(',')]
    else:
        want = [b for b, d in bursts.items() if d['n'] >= a.min_trig]
    # bind the n_TOF files FIRST: the controls' join reads PKUP through ntof_io
    seg = Segment(run, sub, nt,
                  ntof_source=Path(a.ntof_source) if a.ntof_source else None)
    io, files = _bind_ntof(seg)
    controls = []
    if a.controls and lock is not None:
        # bursts the production join matched: use the join itself so the
        # control has a KNOWN answer to compare with
        from ntof_dream_merge.bunch_join import dream_event_to_bunch
        try:
            j = dream_event_to_bunch(run, sub, nt, accept_offset_s=lock,
                                     events=ev)
            bm = j.attrs['burst_map']
            known = {b: k for b, k in zip(bm['burst_id'], bm['bunch']) if k > 0}
        except Exception as e:                               # noqa: BLE001
            print(f'  join for controls failed: {type(e).__name__}: {e}')
            known = {}
        pool = [b for b in sorted(known) if b not in want
                and bursts[b]['n'] >= 40]
        if pool:
            pick = np.unique(np.linspace(0, len(pool) - 1,
                                         min(a.controls, len(pool))).astype(int))
            controls = [pool[i] for i in pick]
    else:
        known = {}
    todo = list(want) + controls
    print(f'  scanning {len(want)} target burst(s) + {len(controls)} control(s)')

    pk = io.pkup_bunches(nt)
    ps, bn, e10 = pk['psTime_s'], pk['BunchNumber'], pk['intensity_e10']
    beam = ~(e10 < C.EMPTY_PULSE_E10)

    off_est = 0.0 if lock is None else lock
    epoch_est = {b: anchor + bursts[b]['rel_s'] + off_est for b in todo}
    if a.all_bunches:
        sel = beam.copy()
    else:
        sel = np.zeros(bn.size, bool)
        for b in todo:
            sel |= np.abs(ps + 0.829 - epoch_est[b]) <= a.window_s
        n_empty = int((sel & ~beam).sum())
        sel &= beam
        print(f'  window +-{a.window_s:g} s: {int(sel.sum())} beam bunches '
              f'({n_empty} empty pulses skipped)')
    take = np.asarray(bn[sel], np.int64)
    if take.size == 0:
        print('  no bunches in range -- nothing to scan')
        return 1
    print(f'  bunches {take.min()}-{take.max()} ({take.size})')

    # top/bottom offsets on a proper sample, as arbiter_measure does
    from ntof_dream_merge import dream_trigger as dt
    from ntof_dream_merge import fast_singles as fs
    fs.REPAIR_TFLASH = False
    off_sample = np.asarray(bn[beam][:OFFSET_BUNCHES], np.int64)
    t0 = time.time()
    offs = {arm: fs.measure_tb_offsets(nt, off_sample, arm) for arm in dt.ARMS}
    print(f'  tb offsets on {off_sample.size} bunches [{time.time()-t0:.0f} s]')
    cd, _o, _thr = pass1_candidates(seg, take, offsets=offs,
                                    log=lambda *x: None)
    print(f'  {cd["t"].size:,} candidates over {take.size} bunches '
          f'[{time.time()-t_start:.0f} s]')
    cb, ct = cd['bunch'].astype(np.int64), cd['t'].astype(np.float64)
    order = np.argsort(cb, kind='stable')
    cb, ct = cb[order], ct[order]
    starts = np.searchsorted(cb, take, side='left')
    ends = np.searchsorted(cb, take, side='right')
    ps_of = dict(zip(bn.tolist(), ps.tolist()))
    e10_of = dict(zip(bn.tolist(), e10.tolist()))

    results = []
    for b in todo:
        te = bursts[b]['t']
        rows = []
        t1 = time.time()
        for j, bunch in enumerate(take):
            tc = ct[starts[j]:ends[j]]
            if tc.size == 0:
                continue
            r = coarse_scan(te, tc)
            if r is None:
                continue
            r.update(bunch=int(bunch), n_cand=int(tc.size),
                     frac_coarse=r['peak'] / max(te.size, 1))
            rows.append(r)
        rows.sort(key=lambda r: -r['peak'])
        top = rows[:max(a.top, REFINE_TOP)]
        for r in top:
            j = int(np.searchsorted(take, r['bunch']))
            r['fine'] = refine(te, ct[starts[j]:ends[j]], r['lag_ns'])
        # the decisive statistic is the fine, line-corrected +-25 ns fraction;
        # the coarse peak only chose who gets refined
        top.sort(key=lambda r: (-(r['fine']['frac'] if r['fine'] else -1.0),
                                -r['z']))
        top = top[:a.top]
        best = top[0] if top else None
        second = top[1] if len(top) > 1 else None
        fine = best['fine'] if best else None
        f1 = fine['frac'] if fine else 0.0
        f2 = second['fine']['frac'] if second and second['fine'] else 0.0
        if best is None:
            verdict = 'NO_CANDIDATES'
        elif f1 >= C.PULSE_MIN_FRAC and f2 < 0.5 * f1:
            verdict = 'MATCH'
        elif f1 >= C.PULSE_MIN_FRAC:
            verdict = 'AMBIGUOUS'
        elif f1 >= 0.3 and f2 < 0.5 * f1:
            verdict = 'WEAK'
        else:
            verdict = 'NO_MATCH'
        rec = dict(burst_id=b, n_trig=int(te.size), rel_s=bursts[b]['rel_s'],
                   role='control' if b in controls else 'target',
                   known_bunch=known.get(b), verdict=verdict,
                   candidates=[dict(bunch=r['bunch'], z=round(r['z'], 1),
                                    peak=r['peak'], floor=round(r['floor'], 2),
                                    frac_coarse=round(r['frac_coarse'], 3),
                                    lag_us=round(r['lag_ns'] / 1e3, 1),
                                    n_cand=r['n_cand'],
                                    e10=round(e10_of.get(r['bunch'], float('nan')), 0),
                                    frac=(round(r['fine']['frac'], 3)
                                          if r['fine'] else None),
                                    ctl_frac=(round(r['fine']['ctl_frac'], 4)
                                              if r['fine'] else None))
                               for r in top],
                   fine=fine, seconds=round(time.time() - t1, 1))
        if best is not None:
            imp = ps_of[best['bunch']] + 0.829 - (anchor + bursts[b]['rel_s'])
            rec['implied_offset_s'] = float(imp)
            rec['join_resid_ms'] = float(
                (epoch_est[b] - 0.829 - ps_of[best['bunch']]) * 1e3)
        results.append(rec)
        tag = f'[{rec["role"]}]'
        if best is None:
            print(f'  burst {b:5d} n={te.size:4d} {tag:9s} no candidates')
            continue
        kb = f' known {known[b]}' if b in known else ''
        fr = f' frac25 {fine["frac"]:.0%} (ctl {fine["ctl_frac"]:.1%}) ' \
             f'da {fine["da_ns"]:+.0f} ns dk {fine["dk"]*1e6:+.1f} ppm ' \
             f'core {fine["n_core"]}' if fine else ''
        s2 = (f' 2nd bunch {second["bunch"]} frac {f2:.0%} z {second["z"]:.1f}'
              if second else '')
        print(f'  burst {b:5d} n={te.size:4d} {tag:9s} -> bunch {best["bunch"]}'
              f'{kb} z {best["z"]:.1f} peak {best["peak"]}/{te.size} '
              f'lag {best["lag_ns"]/1e3:+.1f} us{s2}{fr} '
              f'implied offset {rec["implied_offset_s"]:+.3f} s  {verdict}')

    # THE FLASH SHIFT, from the controls. A target whose winning lag differs
    # from the controls' is a burst whose DREAM time base is referenced to the
    # wrong trigger; the production fix (slim.apply_burst_fix) adds a
    # constant to its t_since_flash_ns. That constant is the difference of the
    # fitted intercepts (T0 + da, both measured here with the same K_SEED, so
    # the seed's own error cancels): accurate to ~10-20 ns, well inside the
    # +-400 ns the per-bunch fit then searches.
    ctrl_int = [r['fine']['T0_ns'] + r['fine']['da_ns'] for r in results
                if r['role'] == 'control' and r['fine']
                and r['fine']['frac'] >= C.PULSE_MIN_FRAC]
    if ctrl_int:
        ref = float(np.median(ctrl_int))
        spread = float(1.4826 * np.median(np.abs(np.array(ctrl_int) - ref)))
        print(f'  control intercept {ref:+.1f} ns (MAD {spread:.1f} ns, '
              f'{len(ctrl_int)} controls)')
        for r in results:
            if r['role'] == 'target' and r['fine']:
                sh = r['fine']['T0_ns'] + r['fine']['da_ns'] - ref
                r['flash_shift_ns'] = float(sh)
                r['control_intercept_ns'] = ref
                r['control_intercept_mad_ns'] = spread
                print(f'  burst {r["burst_id"]:5d}: flash shift '
                      f'{sh:+.1f} ns relative to the controls')

    if a.out:
        out = dict(dream_run=run, dream_subrun=sub, ntof_run=nt,
                   anchor_epoch=anchor, lock_s=lock, window_s=a.window_s,
                   bunches_scanned=[int(take.min()), int(take.max()), int(take.size)],
                   n_candidates=int(ct.size), results=results,
                   seconds=round(time.time() - t_start))
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f'  -> {a.out}')
    print(f'  done in {time.time()-t_start:.0f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
