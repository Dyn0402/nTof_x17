#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unfitted_bunches.py -- why some bunches never get their own clock correction.

    python unfitted_bunches.py <slim_root> -o perbunch.csv          # stage A
    python unfitted_bunches.py --pkup perbunch.csv -o joined.csv    # stage B

`clockfit.fit_perbunch` fits (da_b, dk_b) per bunch and needs
`PB_MIN_EVENTS` (20) DREAM events whose nearest n_TOF candidate is already
inside +-200 ns of the global map. Bunches below that get NO per-bunch term:
their events keep the global (K, T0, arm) map only, are still slimmed (the
correction enters as NaN -> 0), but are dropped from the quoted efficiency
denominator (`efficiency()` counts only `isfinite(corr)`).

Fleet-wide that is 1.7 % of bunches, and the campaign's four
'bunches fitted' WARNs are segments where it reaches 8-22 %. This script
answers what those bunches are:

  stage A  from the slim products alone -- per bunch, how many DREAM triggers,
           how many were flash, how many matched at +-25 ns on the GLOBAL map,
           and whether the bunch was fitted.
  stage B  join the n_TOF PKUP beam record -- pulse intensity and tflash --
           so 'parasitic' can be tested rather than assumed.

Stage A reads only the written slim; stage B needs the n_TOF source (EOS), so
it runs on lxplus.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import uproot

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

ACCEPT_NS = 25.0
FIELDS = ('dream_run', 'dream_subrun', 'ntof_run', 'bunch', 'n_triggers',
          'n_phys', 'n_flash', 'fitted', 'n_core', 'n_matched',
          'resid_med_ns', 'resid_iqr_ns', 't_first_ms', 't_last_ms')


def per_bunch(d: Path) -> list[dict]:
    """One row per bunch of one segment, from the slim file only."""
    root = sorted(d.glob('ntof_hits_*.root'))
    if not root:
        return []
    import json
    prov = json.loads((d / 'provenance.json').read_text())
    with uproot.open(root[0]) as f:
        b = f['bunches'].arrays(library='np')
        ev = f['events'].arrays(library='np')
    b = {k: (b[k] if not isinstance(b, np.ndarray) else b[k]) for k in
         ('bunch', 'n_triggers', 'fitted', 'n_core')}
    eb = ev['bunch'].astype(np.int64)
    phys = ev['is_flash'] == 0
    matched = (ev['matched'] == 1) & phys
    resid = ev['residual_ns']
    t_ms = ev['t_dream_ns'] / 1e6

    order = np.argsort(b['bunch'])
    bun = b['bunch'][order]
    idx = np.searchsorted(bun, eb)
    idx = np.clip(idx, 0, bun.size - 1)
    ok = bun[idx] == eb

    n_phys = np.bincount(idx[ok & phys], minlength=bun.size)
    n_flash = np.bincount(idx[ok & ~phys], minlength=bun.size)
    n_match = np.bincount(idx[ok & matched], minlength=bun.size)
    tmin = np.full(bun.size, np.nan)
    tmax = np.full(bun.size, np.nan)
    rmed = np.full(bun.size, np.nan)
    riqr = np.full(bun.size, np.nan)
    # per-bunch residual summary, matched events only
    o2 = np.argsort(idx[ok & matched], kind='stable')
    ii = idx[ok & matched][o2]
    rr = resid[ok & matched][o2]
    if ii.size:
        edges = np.searchsorted(ii, np.arange(bun.size + 1))
        for k in range(bun.size):
            s = rr[edges[k]:edges[k + 1]]
            if s.size >= 5:
                q = np.percentile(s, [25, 50, 75])
                rmed[k], riqr[k] = q[1], q[2] - q[0]
    o3 = np.argsort(idx[ok & phys], kind='stable')
    i3 = idx[ok & phys][o3]
    t3 = t_ms[ok & phys][o3]
    if i3.size:
        e3 = np.searchsorted(i3, np.arange(bun.size + 1))
        for k in range(bun.size):
            s = t3[e3[k]:e3[k + 1]]
            if s.size:
                tmin[k], tmax[k] = float(s.min()), float(s.max())

    rows = []
    for k in range(bun.size):
        rows.append(dict(
            dream_run=prov['dream_run'], dream_subrun=prov['dream_subrun'],
            ntof_run=prov['ntof_run'], bunch=int(bun[k]),
            n_triggers=int(b['n_triggers'][order][k]),
            n_phys=int(n_phys[k]), n_flash=int(n_flash[k]),
            fitted=int(b['fitted'][order][k]), n_core=int(b['n_core'][order][k]),
            n_matched=int(n_match[k]),
            resid_med_ns=round(float(rmed[k]), 3) if np.isfinite(rmed[k]) else '',
            resid_iqr_ns=round(float(riqr[k]), 3) if np.isfinite(riqr[k]) else '',
            t_first_ms=round(float(tmin[k]), 3) if np.isfinite(tmin[k]) else '',
            t_last_ms=round(float(tmax[k]), 3) if np.isfinite(tmax[k]) else ''))
    return rows


def stage_a(root: Path, out: Path):
    dirs = sorted(p.parent for p in root.rglob('ntof_hits_*.root'))
    print(f'{len(dirs)} segment(s) under {root}')
    n = 0
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, d in enumerate(dirs):
            try:
                rows = per_bunch(d)
            except Exception as e:                              # noqa: BLE001
                print(f'  !! {d}: {type(e).__name__}: {e}')
                continue
            w.writerows(rows)
            n += len(rows)
            print(f'  [{i+1}/{len(dirs)}] {d.parents[1].name}/{d.parent.name}: '
                  f'{len(rows)} bunches')
    print(f'-> {out}  ({n:,} bunch rows)')


def stage_b(csv_in: Path, out: Path, source: Path | None = None):
    """Join the PKUP beam record (intensity, tflash) onto stage A's rows."""
    import ntof_dream_merge.ntof_io as io
    from ntof_processing.slim_pipeline import config as C

    rows = list(csv.DictReader(csv_in.open()))
    runs = sorted({int(r['ntof_run']) for r in rows})
    print(f'{len(rows):,} rows, {len(runs)} n_TOF run(s)')
    book = {}
    for r in runs:
        files = C.ntof_files(r, source)
        io.ntof_paths = lambda _r, _f=files: _f
        io.ntof_path = lambda _r, _f=files: _f[0]
        src = Path(source) if source else C.NTOF_DONE
        # Same rule as `slim._bind_ntof`: the per-run caches are keyed by run
        # number only, so an official and a reprocessed run224572 sharing a
        # directory would silently mix. And on lxplus the default cache lives
        # under $X17_BEAM_JULY on EOS -- redirect it first.
        if C.CACHE_BASE:
            io.CACHE_DIR = Path(C.CACHE_BASE)
            io.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        io.CACHE_DIR = io.variant_cache(src, files)
        try:
            p = io.pkup_bunches(r)
        except Exception as e:                                  # noqa: BLE001
            print(f'  !! run{r}: {type(e).__name__}: {e}')
            continue
        book[r] = {int(b): (float(i_), float(t_), bool(x_), float(s_))
                   for b, i_, t_, x_, s_
                   in zip(p['BunchNumber'], p['intensity_e10'], p['tflash_ns'],
                          p['pstime_recovered'], p['psTime_s'])}
        ii = p['intensity_e10']
        print(f'  run{r}: {len(book[r]):,} bunches, intensity median '
              f'{np.median(ii):.1f}e10, range {ii.min():.1f}-{ii.max():.1f}')
    fields = list(rows[0].keys()) + ['intensity_e10', 'tflash_ns',
                                     'pstime_recovered', 'ps_time_s']
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rec = book.get(int(r['ntof_run']), {}).get(int(r['bunch']))
            r['intensity_e10'] = round(rec[0], 4) if rec else ''
            r['tflash_ns'] = round(rec[1], 2) if rec else ''
            r['pstime_recovered'] = int(rec[2]) if rec else ''
            r['ps_time_s'] = round(rec[3], 3) if rec else ''
            w.writerow(r)
    print(f'-> {out}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path, nargs='?',
                    help='directory tree of slim products (stage A)')
    ap.add_argument('--pkup', type=Path,
                    help='stage A csv to join the beam record onto (stage B)')
    ap.add_argument('--source', type=Path, help='n_TOF source dir (stage B)')
    ap.add_argument('-o', '--out', type=Path, required=True)
    a = ap.parse_args()
    if a.pkup:
        stage_b(a.pkup, a.out, a.source)
    elif a.root:
        stage_a(a.root, a.out)
    else:
        ap.error('give a slim root (stage A) or --pkup (stage B)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
