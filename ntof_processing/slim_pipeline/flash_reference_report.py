#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flash_reference_report.py -- aggregate the flash sweep and answer the question.

`flash_reference_sweep.py` measures, for every DREAM burst of the campaign, the
two independent signatures of a mis-tagged gamma flash (the ~1 ms gate gap that
should precede the first physics trigger, and the ~4,000-hit flash event
itself). This module pools those measurements, flags the bursts, and joins them
against what the matching chain DID with each one:

    flagged burst  x  per-bunch fit record (fitted / da_ns / n_core)
                   x  pulse-ledger terminal state (MATCHED / LOW_COINC / ...)

which is the actual question. A flagged burst that came out UNMATCHED is a
burst the chain already refused -- no harm done, and three of those are the
ones the 2026-08-16 brute force recovered. A flagged burst that came out
MATCHED, with its own per-bunch correction, would be a product carrying a
silently wrong DREAM time base: the class this sweep exists to find.

    python3 flash_reference_report.py --sweep DIR [--ledger DIR] [--out DIR]

Writes <out>/flash_reference.json (every flagged burst, with its joins),
flash_reference.csv (the same as a table), and figures + report.html via
make_flash_report.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from ntof_processing.slim_pipeline.flash_reference_sweep import (  # noqa: E402
    GAP_FRAC, MIN_BURST_TRIG, NHITS_FRAC)

LEDGER = Path('/media/dylan/data/x17/slim_recovery_2026-08-13/pulse_ledger')

# ONLY A PULSE THAT DELIVERED PROTONS HAS A GAMMA FLASH, so only its bursts can
# be asked whether they are timed from one. The states below are the ones whose
# burst sat under real beam; EMPTY_PULSE, NO_BEAM_PULSE and NOT_COINC_TRIGGERED
# are not judged.
#
# This is not a refinement, it is the difference between a working test and a
# broken one. The sweep references each burst against its SUB-RUN's median, and
# in a sub-run that is mostly empty pulses that median is the median of bursts
# with NO flash: measured over the campaign, a no-beam burst's first event
# carries 1-3 hits against ~4,080 for a beam burst. Referenced against that, the
# HEALTHY bursts of such a sub-run come out 30x high and get flagged -- 348 of
# them on the first pass (run_112/0000, run_108/0000, run_126/0000 ...), every
# one a false positive with a perfectly normal 1 ms gap and 4,100-hit flash.
BEAM_STATES = frozenset({
    'MATCHED', 'LOW_COINC', 'UNKNOWN_COINC', 'TOO_FEW_TRIGGERS',
    'NTOF_NO_BUNCH', 'UNJOINED', 'SEGMENT_FAILED', 'NOT_ATTEMPTED'})
# Fewest judgeable bursts a sub-run needs before its own median is used as the
# reference; below it the campaign median of sub-run medians stands in. Both
# signatures are stable enough campaign-wide for that to be safe -- gap1 runs
# 991-1,170 us over 281,437 beam bursts -- but a per-sub-run reference is still
# preferred where it exists, because the gate width is a DAQ setting and the
# flash hit count depends on which chambers were live.
MIN_REF_BURSTS = 20


def load_sweep(d: Path):
    out = {}
    for p in sorted(d.glob('run_*.json')):
        try:
            out[p.stem] = json.loads(p.read_text())
        except ValueError as e:
            print(f'!! {p.name}: {e}')
    return out


def load_ledger(d: Path):
    """{(run, subrun): {burst_id: (state, ntof_run, bunch, frac)}}"""
    out = {}
    for p in sorted(d.glob('run_*.json')):
        if p.name == 'campaign_ledger.json':
            continue
        try:
            led = json.loads(p.read_text())
        except ValueError:
            continue
        b = led['bursts']
        out[(led['run'], led['subrun'])] = {
            int(i): (s, r, bn, f) for i, s, r, bn, f in
            zip(b['burst_id'], b['state'], b['ntof_run'], b['bunch'],
                b['frac'])}
    return out


def burst_fixed(prod: dict, burst_id: int):
    """Was a burst_fixes.json override APPLIED to this burst in the product?

    `slim.apply_burst_fix` records what it did in burst_map.json under `fix`,
    so a burst the sweep flags on the raw DREAM files may already be corrected
    downstream -- the override lives in the join, not in the DREAM data, so
    the mis-tagged flash stays visible here forever. Without this the four
    bursts fixed on 2026-08-16 read as an unsolved silent class.
    """
    for rec in prod.values():
        fx = ((rec.get('burst_map') or {}).get('fix') or {}).get(str(burst_id))
        if fx:
            return fx
    return None


def bunch_record(prod: dict, burst_id: int):
    """(ntof_run, per-bunch fit record) for a burst, via the burst map."""
    for nt, rec in prod.items():
        bm = rec.get('burst_map')
        if not bm:
            continue
        try:
            i = bm['burst_id'].index(burst_id)
        except ValueError:
            continue
        bunch = bm['bunch'][i]
        if bunch is None or bunch < 0:
            return nt, bunch, None
        try:
            j = rec['bunch'].index(bunch)
        except ValueError:
            return nt, bunch, None
        return nt, bunch, dict(
            n_triggers=rec['n_triggers'][j], has_beam=rec['has_beam'][j],
            fitted=rec['fitted'][j], da_ns=rec['da_ns'][j], dk=rec['dk'][j],
            n_core=rec['n_core'][j])
    return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep', required=True, type=Path)
    ap.add_argument('--ledger', type=Path, default=LEDGER)
    ap.add_argument('--out', type=Path, default=HERE / 'flash_reference')
    ap.add_argument('--since-run', type=int, default=79)
    a = ap.parse_args()

    sweep = load_sweep(a.sweep)
    if not sweep:
        print(f'no sweep files under {a.sweep}')
        return 1
    ledger = load_ledger(a.ledger) if a.ledger.is_dir() else {}
    camp_path = a.ledger / 'campaign_ledger.json'
    camp_states = (json.loads(camp_path.read_text()).get('states', {})
                   if camp_path.exists() else {})
    print(f'{len(sweep)} DREAM run(s) swept, {len(ledger)} ledger sub-run(s)')

    # ---- pass 1: per sub-run reference over the BEAM bursts alone
    subs = []
    n_bursts = n_err = n_noflash = 0
    for run, d in sorted(sweep.items()):
        try:
            rn = int(run.split('_')[1])
        except (IndexError, ValueError):
            continue
        for sub, r in sorted(d['subruns'].items()):
            if 'error' in r:
                n_err += 1
                continue
            b = r['bursts']
            n = np.asarray(b['n_trig'])
            n_bursts += int(n.size)
            st = ledger.get((run, sub), {})
            state = np.array([st.get(int(i), (None,) * 4)[0]
                              for i in b['burst_id']], dtype=object)
            beam = (n >= MIN_BURST_TRIG) & np.array(
                [s in BEAM_STATES for s in state])
            g1 = np.asarray(b['gap1_ns'], float)
            fh = np.asarray(b['flash_nhits'], float)
            n_noflash += int(((n >= MIN_BURST_TRIG) & ~beam).sum())
            subs.append(dict(run=run, rn=rn, subrun=sub, bursts=b, n=n,
                             g1=g1, fh=fh, state=state, beam=beam,
                             gap1_med_ns=(float(np.median(g1[beam]))
                                          if beam.sum() >= MIN_REF_BURSTS
                                          else None),
                             flash_nhits_med=(float(np.median(fh[beam]))
                                              if beam.sum() >= MIN_REF_BURSTS
                                              else None),
                             n_beam=int(beam.sum()),
                             products=(d.get('products') or {}).get(sub, {})))
    have = [s for s in subs if s['gap1_med_ns']]
    camp_g = float(np.median([s['gap1_med_ns'] for s in have])) if have else None
    camp_h = float(np.median([s['flash_nhits_med'] for s in have])) if have \
        else None
    print(f'campaign reference from {len(have)} sub-run(s) with '
          f'>= {MIN_REF_BURSTS} beam bursts: gap1 {camp_g/1e3:.1f} us, '
          f'flash hits {camp_h:.0f}')

    # ---- pass 2: flag, and join against what the chain did
    flagged, pooled_g, pooled_h, per_sub = [], [], [], []
    n_judged = 0
    for s in subs:
        gm = s['gap1_med_ns'] or camp_g
        hm = s['flash_nhits_med'] or camp_h
        per_sub.append(dict(run=s['run'], subrun=s['subrun'],
                            n_bursts=int(s['n'].size), n_beam=s['n_beam'],
                            gap1_med_ns=gm, flash_nhits_med=hm,
                            own_reference=bool(s['gap1_med_ns'])))
        beam = s['beam']
        if not beam.any() or not gm or not hm:
            continue
        n_judged += int(beam.sum())
        g1, fh, b = s['g1'], s['fh'], s['bursts']
        pooled_g.append(g1[beam] / gm)
        pooled_h.append(fh[beam] / hm)
        hit = beam & ((g1 < GAP_FRAC * gm) | (fh < NHITS_FRAC * hm))
        for i in np.flatnonzero(hit):
            led = ledger.get((s['run'], s['subrun']), {}).get(
                int(b['burst_id'][i]), (None,) * 4)
            nt, bunch, fit = bunch_record(s['products'], int(b['burst_id'][i]))
            flagged.append(dict(
                run=s['run'], subrun=s['subrun'],
                burst_id=int(b['burst_id'][i]), since_run=s['rn'] >= a.since_run,
                n_trig=int(s['n'][i]), t_rel_s=round(b['t_rel_s'][i], 1),
                gap1_ns=float(g1[i]), gap1_frac=round(float(g1[i] / gm), 4),
                flash_nhits=int(fh[i]),
                nhits_frac=round(float(fh[i] / hm), 4),
                both=bool(g1[i] < GAP_FRAC * gm and fh[i] < NHITS_FRAC * hm),
                own_reference=bool(s['gap1_med_ns']),
                state=led[0], ledger_ntof=led[1], ledger_bunch=led[2],
                ledger_frac=led[3], fixed=burst_fixed(s['products'],
                                                      int(b['burst_id'][i])),
                product_ntof=nt, product_bunch=bunch, fit=fit))
    g = np.concatenate(pooled_g) if pooled_g else np.zeros(0)
    h = np.concatenate(pooled_h) if pooled_h else np.zeros(0)

    print(f'{n_bursts:,} clusters, {n_judged:,} judged (>= {MIN_BURST_TRIG} '
          f'triggers AND a beam pulse under them); {n_noflash:,} big clusters '
          f'skipped as no-beam -- with no protons there is no gamma flash to '
          f'be timed from; {n_err} sub-run(s) errored')
    if g.size:
        print(f'gap1 / sub-run median: pct '
              f'{np.percentile(g, [0.01, 1, 50, 99, 99.99]).round(3)}')
        print(f'flash hits / sub-run median: pct '
              f'{np.percentile(h, [0.01, 1, 50, 99, 99.99]).round(3)}')
    print(f'\n{len(flagged)} burst(s) flagged '
          f'(gap1 < {GAP_FRAC} x median OR flash hits < {NHITS_FRAC} x median)')

    # THE ANSWER: what did the chain do with the flagged bursts?
    by_state: dict = {}
    for f in flagged:
        by_state.setdefault(f['state'] or 'no ledger entry', []).append(f)
    matched = by_state.get('MATCHED', [])
    corrected = [f for f in matched if f['fixed']]
    silent = [f for f in matched if not f['fixed']]
    matched_fitted = [f for f in silent if (f['fit'] or {}).get('fitted')]
    print('  by ledger state: ' + ', '.join(
        f'{k} {len(v)}' for k, v in sorted(by_state.items(),
                                           key=lambda kv: -len(kv[1]))))
    print(f'  {len(corrected)} flagged burst(s) already carry a '
          f'burst_fixes.json override in the product (the mis-tag is in the '
          f'DREAM data; the correction is in the join, so it stays visible '
          f'here):')
    for f in corrected:
        fx = f['fixed']
        print(f'     {f["run"]}/{f["subrun"]} burst {f["burst_id"]} -- '
              f'flash shift {fx.get("flash_shift_ns", 0)/1e6:+.3f} ms, bunch '
              f'{fx.get("was_bunch")} -> {fx.get("bunch")}, now '
              f'{f["ledger_frac"]:.0%} coincident')
    if silent:
        print(f'  !! {len(silent)} flagged burst(s) count as MATCHED with NO '
              f'correction applied -- {len(matched_fitted)} of them with '
              f'their own per-bunch fit. These are the silent class.')
        for f in silent[:40]:
            fit = f['fit'] or {}
            print(f'     {f["run"]}/{f["subrun"]} burst {f["burst_id"]} '
                  f'n={f["n_trig"]} gap1 {f["gap1_ns"]/1e3:.1f} us '
                  f'({f["gap1_frac"]:.3f}x) flash hits {f["flash_nhits"]} '
                  f'({f["nhits_frac"]:.3f}x) -> bunch {f["ledger_bunch"]} '
                  f'coinc {f["ledger_frac"]} fitted {fit.get("fitted")} '
                  f'da {fit.get("da_ns")}')
    else:
        print('  no flagged burst is MATCHED without a correction: every '
              'mis-tagged flash the sweep finds was either refused by the '
              'chain or already fixed, so no product carries an uncorrected '
              'one.')

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'flash_reference.json').write_text(json.dumps(dict(
        thresholds=dict(gap_frac=GAP_FRAC, nhits_frac=NHITS_FRAC,
                        min_burst_trig=MIN_BURST_TRIG),
        n_runs=len(sweep), n_bursts=n_bursts, n_judged=n_judged,
        n_noflash_skipped=n_noflash, n_subrun_errors=n_err,
        since_run=a.since_run, beam_states=sorted(BEAM_STATES),
        # the ledger's own totals, so the report can quote a rate per class
        # without re-deriving (or hard-coding) a denominator
        ledger_states=camp_states,
        min_ref_bursts=MIN_REF_BURSTS,
        campaign_ref=dict(gap1_ns=camp_g, flash_nhits=camp_h),
        pooled=dict(
            gap1_frac_pct={str(p): float(np.percentile(g, p)) for p in
                           (0.01, 0.1, 1, 5, 50, 95, 99, 99.9, 99.99)}
            if g.size else {},
            nhits_frac_pct={str(p): float(np.percentile(h, p)) for p in
                            (0.01, 0.1, 1, 5, 50, 95, 99, 99.9, 99.99)}
            if h.size else {},
            gap1_frac_hist=np.histogram(np.clip(g, 0, 2), bins=200,
                                        range=(0, 2))[0].tolist()
            if g.size else [],
            nhits_frac_hist=np.histogram(np.clip(h, 0, 2), bins=200,
                                         range=(0, 2))[0].tolist()
            if h.size else []),
        by_state={k: len(v) for k, v in by_state.items()},
        n_matched_flagged=len(matched), n_corrected=len(corrected),
        n_silent=len(silent), n_silent_fitted=len(matched_fitted),
        flagged=flagged, per_subrun=per_sub), indent=1))
    with open(a.out / 'flash_reference.csv', 'w', newline='') as fh_:
        w = csv.writer(fh_)
        w.writerow(['run', 'subrun', 'burst_id', 'n_trig', 't_rel_s',
                    'gap1_us', 'gap1_frac', 'flash_nhits', 'nhits_frac',
                    'both_signatures', 'ledger_state', 'ntof_run', 'bunch',
                    'coincidence', 'fitted', 'da_ns', 'n_core',
                    'correction_applied_ns'])
        for f in flagged:
            fit = f['fit'] or {}
            w.writerow([f['run'], f['subrun'], f['burst_id'], f['n_trig'],
                        f['t_rel_s'], round(f['gap1_ns'] / 1e3, 1),
                        f['gap1_frac'], f['flash_nhits'], f['nhits_frac'],
                        int(f['both']), f['state'], f['ledger_ntof'],
                        f['ledger_bunch'], f['ledger_frac'], fit.get('fitted'),
                        fit.get('da_ns'), fit.get('n_core'),
                        (f['fixed'] or {}).get('flash_shift_ns')])
    print(f'\n-> {a.out}/flash_reference.json + .csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
