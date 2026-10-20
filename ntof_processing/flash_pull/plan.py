#!/usr/bin/env python3
"""Choose the runs and the raw files of the flash pull.  Runs on lxplus (it lists
CTA and the EOS disk staging area).

    python -m ntof_processing.flash_pull.plan run_inventory.csv --out plan.csv

`run_inventory.csv` (inputs/) has one row per run under the X17_measurement
archive: bunches with beam from the official processed file, and the raw file
counts.  A run is taken if it has >= MIN_BEAM_BUNCHES bunches above 1e12 ppp, or
if it has no readable processed file but more than MIN_UNPROCESSED_GB of raw (the
31 production runs n_TOF never processed: big raw means beam).

Per run, FILES_PER_RUN files (FILES_PER_RUN_SPECIAL for config.SPECIAL_RUNS) at
evenly spaced positions in the run, never the first or last file (the run's
start-up and its possibly short tail).  Each file is read from the EOS disk copy
if that file is there, else from tape -- per FILE, so a partial disk copy is
used for what it has.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'ntof_processing.flash_pull'

from . import config as F                         # noqa: E402
from ..waveform_pull import config as WC          # noqa: E402

MIN_BEAM_BUNCHES = 100
MIN_UNPROCESSED_GB = 50
RAW_RE = re.compile(r'run(\d+)_(\d+)_s1\.raw')


def cta_indices(run: int) -> list[int]:
    r = subprocess.run(['xrdfs', WC.CTA_XRD, 'ls', f'{WC.CTA_BASE}/{run}/stream1'],
                       capture_output=True, text=True)
    return sorted(int(m.group(2)) for m in map(RAW_RE.search, r.stdout.split()) if m)


def disk_files(run: int) -> dict:
    d = WC.NTOF_RAW / str(run) / 'stream1'
    try:
        names = os.listdir(d)
    except OSError:
        return {}
    return {int(m.group(2)): str(d / n) for n in names if (m := RAW_RE.search(n))}


def choose(idx: list[int], k: int) -> list[int]:
    inner = idx[1:-1] if len(idx) > 2 else idx
    if len(inner) <= k:
        return inner
    return sorted({inner[int((i + 0.5) * len(inner) / k)] for i in range(k)})


def take(row: dict) -> bool:
    if row['proc'] == 'ok':
        return int(row['n_beam_bunch']) >= MIN_BEAM_BUNCHES
    return int(row['cta_bytes']) > MIN_UNPROCESSED_GB * 1e9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('inventory', type=Path)
    ap.add_argument('--out', type=Path, default=Path('plan.csv'))
    a = ap.parse_args(argv)
    rows = [r for r in csv.DictReader(open(a.inventory)) if take(r)]
    out = []
    for r in rows:
        run = int(r['run'])
        k = F.FILES_PER_RUN_SPECIAL if run in F.SPECIAL_RUNS else F.FILES_PER_RUN
        idx = cta_indices(run)
        if not idx:
            print(f'{run}: nothing on tape, skipped', file=sys.stderr)
            continue
        disk = disk_files(run)
        for i in choose(idx, k):
            src = disk.get(i) or WC.cta_url(run, i)
            out.append(dict(run=run, idx=i, source='disk' if i in disk else 'tape', path=src,
                            n_files_run=len(idx), proc=r['proc'], median_ppp=r['median_ppp']))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    nd = sum(o['source'] == 'disk' for o in out)
    print(f'{len(rows)} runs, {len(out)} files: {nd} on disk, {len(out) - nd} on tape -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
