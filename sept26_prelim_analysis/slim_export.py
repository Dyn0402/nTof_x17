#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slim_export.py -- the n_TOF slim for one sub-run, as a narrow parquet.

The campaign deliverable is "tracks and n_TOF hits for the full dataset, small
enough to pull home".  The slim ROOT files are 50-90 MB per sub-run (~25 GB
across the 293), and converting them at CERN and pulling the parquet moves a
fraction of that.

WHAT IS KEPT, AND WHY NOTHING IS CUT ON TIME.

  * **The full +-1000 ns `dt_ns`,** not the accept window.  This is the whole
    reason `HANDOFF_ACCIDENTAL_TIMING.md` could be written at all, and the
    decision on 2026-09-08 was explicitly that the window stays re-tunable
    offline -- so exporting an already-windowed slim would destroy the thing
    the decision was protecting.
  * **`is_control` hits,** all of them.  They are the n_TOF processing's own
    random-coincidence control, dead flat across the full range, and they are
    the accidental normalisation the two-component fit needs.  Nothing else
    supplies it.
  * **Every family** -- wall, plastic AND liquid.  LIQ is unused by the pair
    analysis today but is the neutron-energy handle, and re-running 293 condor
    jobs to recover a column would cost more than carrying it.

So this is a format conversion plus a column projection, not a selection.  The
only rows dropped are those `read_slim` itself does not carry.

    python -m sept26_prelim_analysis.slim_export --run run_145 --subrun stat090_0000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths                      # noqa: E402
from sept26_prelim_analysis.scintillators import read_slim    # noqa: E402

SCHEMA = 'sept26_prelim/slim_export/1'

#: Downcast targets. The slim is mostly small integers and one float time;
#: float64 everywhere would roughly double the product for no precision that
#: exists in the data (dt_ns is quantised well above float32 resolution).
DTYPES = {
    'eventId': 'int64', 'det': 'int8', 'detn': 'int16',
    'tof': 'float32', 'dt_ns': 'float32', 'amp': 'float32',
    'area_0': 'float32', 'satuflag': 'int8', 'pileup1': 'int8',
    'is_control': 'int8',
}


def export(run: str, subrun: str, out_dir: Path) -> dict:
    d = read_slim(run, [subrun])
    for c, t in DTYPES.items():
        if c in d.columns:
            # errors='ignore' would hide a real dtype surprise; let it raise.
            d[c] = d[c].astype(t)
    for c in ('family', 'arm', 'subrun'):
        if c in d.columns:
            d[c] = d[c].astype('category')
    d['run'] = run
    d['run'] = d['run'].astype('category')

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f'ntof_hits_{run}_{subrun}.parquet'
    # snappy, not zstd: the LCG_105 pyarrow (11.0.0) the condor workers use is
    # built WITHOUT the zstd codec and raises ArrowNotImplementedError. Snappy
    # is available everywhere and costs ~20 % size against zstd here.
    d.to_parquet(p, index=False, compression='snappy')

    meta = dict(schema=SCHEMA, run=run, subrun=subrun,
                n_hits=int(len(d)),
                n_events=int(d.eventId.nunique()),
                n_control=int((d.is_control == 1).sum()),
                dt_ns_min=float(d.dt_ns.min()), dt_ns_max=float(d.dt_ns.max()),
                families={k: int(v) for k, v in
                          d.family.value_counts().items()},
                bytes=int(p.stat().st_size))
    (out_dir / f'ntof_hits_{run}_{subrun}.meta.json').write_text(
        json.dumps(meta, indent=1))
    return meta


def read_export(run: str, subruns, slim_dir: Path | None = None) -> pd.DataFrame:
    """The exported parquet for one run, as `read_slim` would have returned it.

    Campaign-wide the ROOT slims stay at CERN and only these parquets come home,
    so every consumer that used to call `scintillators.read_slim` reads this
    instead. Columns, dtypes and derived fields (`family`, `arm`, `in_time`)
    are the same by construction -- `export` writes exactly what `read_slim`
    produced -- so nothing downstream has to branch on the source.

    Raises rather than falling back to the ROOT reader: a silent fallback would
    read whichever sub-runs happen to be staged locally and quietly analyse a
    different sample than the one asked for.
    """
    d = Path(slim_dir) if slim_dir else paths.out('slim')
    out = []
    missing = []
    for sub in subruns:
        p = d / f'ntof_hits_{run}_{sub}.parquet'
        if not p.exists():
            missing.append(sub)
            continue
        out.append(pd.read_parquet(p))
    if missing:
        raise FileNotFoundError(
            f'no exported slim for {run} sub-run(s) {", ".join(missing)} under '
            f'{d}\n  run `sept26_prelim_analysis.slim_export` for them, or '
            f'fetch the campaign slim pass.')
    d = pd.concat(out, ignore_index=True)
    for c in ('family', 'arm', 'subrun', 'run'):
        if c in d.columns:
            d[c] = d[c].astype(str)         # de-categorise: groupbys downstream
    return d                                # assume plain object dtype


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', required=True)
    ap.add_argument('--subrun', required=True)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    out = a.out or paths.out('slim')
    m = export(a.run, a.subrun, out)
    print(f'{a.run}/{a.subrun}: {m["n_hits"]:,} hits, '
          f'{m["n_events"]:,} events, {m["n_control"]:,} control, '
          f'{m["bytes"] / 1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
