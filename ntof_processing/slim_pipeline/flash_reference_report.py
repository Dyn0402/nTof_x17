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
    print(f'{len(sweep)} DREAM run(s) swept, {len(ledger)} ledger sub-run(s)')

    flagged, pooled_g, pooled_h, per_sub = [], [], [], []
    n_bursts = n_judged = n_err = 0
    for run, d in sorted(sweep.items()):
        try:
            rn = int(run.split('_')[1])
        except (IndexError, ValueError):
            continue
        for sub, r in sorted(d['subruns'].items()):
            if 'error' in r:
                n_err += 1
                continue
            ref, b = r['ref'], r['bursts']
            n = np.asarray(b['n_trig'])
            g1 = np.asarray(b['gap1_ns'], float)
            fh = np.asarray(b['flash_nhits'], float)
            n_bursts += int(n.size)
            big = n >= MIN_BURST_TRIG
            gm, hm = ref['gap1_med_ns'], ref['flash_nhits_med']
            per_sub.append(dict(run=run, subrun=sub, n_bursts=int(n.size),
                                n_big=int(big.sum()), gap1_med_ns=gm,
                                flash_nhits_med=hm,
                                phys_nhits_med=ref['phys_nhits_med']))
            if not big.any() or not gm or not hm:
                continue
            n_judged += int(big.sum())
            pooled_g.append(g1[big] / gm)
            pooled_h.append(fh[big] / hm)
            hit = big & ((g1 < GAP_FRAC * gm) | (fh < NHITS_FRAC * hm))
            for i in np.flatnonzero(hit):
                led = ledger.get((run, sub), {}).get(int(b['burst_id'][i]),
                                                     (None,) * 4)
                nt, bunch, fit = bunch_record(
                    (d.get('products') or {}).get(sub, {}),
                    int(b['burst_id'][i]))
                flagged.append(dict(
                    run=run, subrun=sub, burst_id=int(b['burst_id'][i]),
                    since_run=rn >= a.since_run,
                    n_trig=int(n[i]), t_rel_s=round(b['t_rel_s'][i], 1),
                    gap1_ns=float(g1[i]), gap1_frac=round(float(g1[i] / gm), 4),
                    flash_nhits=int(fh[i]),
                    nhits_frac=round(float(fh[i] / hm), 4),
                    both=bool(g1[i] < GAP_FRAC * gm and fh[i] < NHITS_FRAC * hm),
                    state=led[0], ledger_ntof=led[1], ledger_bunch=led[2],
                    ledger_frac=led[3],
                    product_ntof=nt, product_bunch=bunch, fit=fit))
    g = np.concatenate(pooled_g) if pooled_g else np.zeros(0)
    h = np.concatenate(pooled_h) if pooled_h else np.zeros(0)

    print(f'{n_bursts:,} bursts, {n_judged:,} with >= {MIN_BURST_TRIG} '
          f'triggers (the judgeable ones); {n_err} sub-run(s) errored')
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
    matched_fitted, matched_unfitted = [], []
    for f in by_state.get('MATCHED', []):
        (matched_fitted if (f['fit'] or {}).get('fitted')
         else matched_unfitted).append(f)
    print('  by ledger state: ' + ', '.join(
        f'{k} {len(v)}' for k, v in sorted(by_state.items(),
                                           key=lambda kv: -len(kv[1]))))
    if by_state.get('MATCHED'):
        print(f'  !! {len(by_state["MATCHED"])} flagged burst(s) count as '
              f'MATCHED -- {len(matched_fitted)} of them with their own '
              f'per-bunch correction. These are the silent class.')
        for f in by_state['MATCHED'][:40]:
            fit = f['fit'] or {}
            print(f'     {f["run"]}/{f["subrun"]} burst {f["burst_id"]} '
                  f'n={f["n_trig"]} gap1 {f["gap1_ns"]/1e3:.1f} us '
                  f'({f["gap1_frac"]:.3f}x) flash hits {f["flash_nhits"]} '
                  f'({f["nhits_frac"]:.3f}x) -> bunch {f["ledger_bunch"]} '
                  f'coinc {f["ledger_frac"]} fitted {fit.get("fitted")} '
                  f'da {fit.get("da_ns")}')
    else:
        print('  no flagged burst is MATCHED: every mis-tagged flash the sweep '
              'finds was refused by the chain, so no product carries one.')

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'flash_reference.json').write_text(json.dumps(dict(
        thresholds=dict(gap_frac=GAP_FRAC, nhits_frac=NHITS_FRAC,
                        min_burst_trig=MIN_BURST_TRIG),
        n_runs=len(sweep), n_bursts=n_bursts, n_judged=n_judged,
        n_subrun_errors=n_err, since_run=a.since_run,
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
        n_matched_flagged=len(by_state.get('MATCHED', [])),
        n_matched_flagged_fitted=len(matched_fitted),
        flagged=flagged, per_subrun=per_sub), indent=1))
    with open(a.out / 'flash_reference.csv', 'w', newline='') as fh_:
        w = csv.writer(fh_)
        w.writerow(['run', 'subrun', 'burst_id', 'n_trig', 't_rel_s',
                    'gap1_us', 'gap1_frac', 'flash_nhits', 'nhits_frac',
                    'both_signatures', 'ledger_state', 'ntof_run', 'bunch',
                    'coincidence', 'fitted', 'da_ns', 'n_core'])
        for f in flagged:
            fit = f['fit'] or {}
            w.writerow([f['run'], f['subrun'], f['burst_id'], f['n_trig'],
                        f['t_rel_s'], round(f['gap1_ns'] / 1e3, 1),
                        f['gap1_frac'], f['flash_nhits'], f['nhits_frac'],
                        int(f['both']), f['state'], f['ledger_ntof'],
                        f['ledger_bunch'], f['ledger_frac'], fit.get('fitted'),
                        fit.get('da_ns'), fit.get('n_core')])
    print(f'\n-> {a.out}/flash_reference.json + .csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
