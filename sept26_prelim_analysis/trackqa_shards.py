#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trackqa_shards.py -- the track table as web-browsable parquet, one shard per
sub-run.

WHY THIS EXISTS.  `tracking_qa` reduces 29 M tracks to quantiles, and a
quantile is where a lead stops: it says arm D's chi2/dof p50 moved in run_116
and it cannot say which tracks moved it.  Answering that has meant pulling the
11.4 GB campaign table onto a machine that has it.  This makes the same table
answerable from a browser, on a laptop, with nothing installed.

HOW IT IS BROWSABLE WITHOUT A SERVER.  CERN's web hosting is Apache over EOS --
static files, no CGI, no database -- but it honours HTTP range requests
(measured 2026-09-10: `206 Partial Content`, correct `content-range`).  Parquet
is laid out so that a reader who can seek can pull the footer, then just the
column chunks it wants.  So a page can read TWO columns of a 60 k-track sub-run
for ~0.4 MB instead of downloading 14 MB, and the cost of a histogram is the
columns it plots rather than the size of the file.  That is the whole design;
everything below follows from it.

WHAT IS KEPT.  Every column `tracking_qa` profiles -- all of VARS, all of
FLAGS, every PATHOLOGY input -- plus the geometry needed to draw a track and
the n_TOF columns already merged into the table.  The point of keeping exactly
the QA set is that the page and the frozen tables then compute the same numbers
from the same columns, so the two can be checked against each other instead of
being two independent claims about the same run.

**Ungated tracks are kept.**  They are half the table and shipping only the
survivors would make the one question the gate raises -- what does it throw
away, and did that change? -- the one question the page cannot answer.  Pass
``--gated-only`` for a build half the size that cannot answer it.

SNAPPY, NOT ZSTD.  The browser-side reader (hyparquet) ships snappy and gzip;
gzip needs a second package and zstd a wasm blob.  Snappy costs ~11 % over zstd
here and nothing in dependencies, which is the right trade for a site that has
no external dependencies at all.

ROW GROUPS ARE THE OTHER TUNING KNOB.  Small groups make "show me this event"
cheap and a full-column scan chatty; large groups do the reverse.  Measured on
one sub-run (59 k tracks, 50 columns):

    row group   file      13-col scan        200-row slab
        4 096   7.58 MB   0.40 MB / 30 req   0.52 MB / 1 req
       16 384   7.73 MB   3.28 MB / 54 req   ~1.0 MB / 1 req
       50 000   7.84 MB   0.42 MB /  4 req   6.64 MB / 4 req

16 384 is the compromise and the default.

    python -m sept26_prelim_analysis.trackqa_shards --run run_145
    python -m sept26_prelim_analysis.trackqa_shards          # all 292 sub-runs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths                      # noqa: E402
from sept26_prelim_analysis.campaign_tracks import _condition  # noqa: E402

SCHEMA = 'sept26_prelim/trackqa_shards/1'

#: Identity and the clock.  `run` and `subrun` are NOT here: they are constant
#: within a shard and the shard is named for them, so carrying them would be
#: ~10 bytes a track to repeat the filename 100 000 times.
ID_COLS = ['tag', 'event_id', 'arm', 'track_id', 'event_class',
           'select_reason', 'bunch']

#: The eleven gates, exactly `tracking_qa.FLAGS`.  A page that has these can
#: recompute every `frac_*` in the frozen tables.
FLAG_COLS = ['gated', 'x_quality_ok', 'y_quality_ok', 'x_plausible',
             'y_plausible', 'x_slope_reliable', 'y_slope_reliable',
             'tan_sane', 'drift_railed', 'x_isochronous', 'y_isochronous']

#: Every column behind `tracking_qa.VARS` and `tracking_qa.PATHOLOGY`.
VAR_COLS = ['chi2dof_x', 'chi2dof_y', 'x_n_strips', 'y_n_strips',
            'x_tan_err', 'y_tan_err', 'x_p0_err', 'y_p0_err', 'x_t0_err',
            'tanx', 'tany', 'x_t0', 'y_t0', 'drift_len_mm',
            'q_total', 'q_per_len', 'x_q_u50', 'y_q_u50',
            'x_n_dropped', 'y_n_dropped', 'n_cand_x', 'n_cand_y']

#: Enough to draw the track: a point, a direction, where it lands.
GEOM_COLS = ['x_local', 'y_local', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z',
             'drift_t_end_ns', 'path_len_mm', 'target_x_mm', 'target_y_mm',
             'target_z_mm', 'dca_axis_mm', 'angle_to_beam_deg', 'in_bore']

#: What the angle depends on.  `k_arm` is the open blocker of 2026-09-10, so a
#: page that plots an angle without showing the scale it used is a trap.
CALIB_COLS = ['k_arm', 'v_drift_um_ns', 'angle_calibrated', 'depth_grid_edge_ns']

#: The n_TOF side, already merged into the track table upstream.  The two time
#: columns were all-null until 2026-09-11 and are filled from the slim's own
#: ``events`` tree -- see :mod:`sept26_prelim_analysis.trigger_time`.  They are
#: null for a track whose trigger stage 1 never joined to an n_TOF bunch, which
#: is the same population as a null ``bunch`` and nothing to do with this
#: product; select on ``bunch`` when a plot needs the joined sample.
NTOF_COLS = ['t_since_flash_ns', 'e_neutron_keV', 'is_flash', 'n_coinc_arms',
             'arms_lit']

COLUMNS = ID_COLS + FLAG_COLS + VAR_COLS + GEOM_COLS + CALIB_COLS + NTOF_COLS

#: Columns that some sub-runs genuinely do not have, and whose absence is not a
#: reason to refuse the shard.  `select_reason` is missing from 4 of the 292
#: stage-3 tables (all of run_154 and one of run_79) because those were built
#: before `build_tracks` started recording it.  Everything else in COLUMNS is
#: REQUIRED: it is either a column `tracking_qa` profiles -- so a shard without
#: it cannot be checked against the frozen tables, which is the one property
#: this product exists to have -- or geometry the page draws.
#:
#: An optional column that is absent is written as all-null and NAMED in the
#: shard's index entry, so the page can say "this sub-run does not carry it"
#: rather than showing an empty distribution that looks like a measurement.
OPTIONAL_COLS = ('select_reason',)

#: Columns kept as strings.  Everything else is downcast; see `_shrink`.
STR_COLS = ('tag', 'arm', 'event_class', 'select_reason', 'arms_lit')

SRC_RE = re.compile(r'^tracks_(run_\d+)_(.+)\.parquet$')


#: Key under which a shard records, in its own parquet metadata, which optional
#: columns were filled in rather than read.  The alternative -- inferring it
#: from an all-null column -- does not work: an all-null column carries no
#: statistics to read, and a sub-run where every track genuinely has no
#: `select_reason` is indistinguishable from one where the column was absent.
#: The file has to say so itself.
ABSENT_KEY = b'sept26_absent'


def read_absent(path: Path) -> list:
    """Which optional columns a built shard filled in, from its own metadata."""
    import pyarrow.parquet as pq
    md = pq.ParquetFile(path).schema_arrow.metadata or {}
    raw = md.get(ABSENT_KEY)
    return json.loads(raw.decode()) if raw else []


def _shrink(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, int64 -> the narrowest int that holds it.

    The table is written by pandas at native width, and float64 doubles the
    product for precision that is not in the data: a drift time is quantised at
    the sampling clock and a charge is a 12-bit ADC sum.  `event_id` is capped
    at int32 rather than downcast per shard, so the column has the same type in
    every file and a reader can assume it.
    """
    for c in df.columns:
        if c in STR_COLS:
            # NOT `.astype(str)`: that turns a genuine null into the four-letter
            # string 'None', which then reads as a value in the browser and in
            # any groupby downstream. A missing select_reason must stay missing.
            df[c] = df[c].astype('object').where(df[c].notna(), None)
        elif df[c].dtype == np.float64:
            df[c] = df[c].astype('float32')
        elif df[c].dtype == np.int64:
            df[c] = (df[c].astype('int32') if c == 'event_id'
                     else pd.to_numeric(df[c], downcast='integer'))
    return df


def build_shard(src: Path, dest: Path, gated_only: bool = False,
                row_group: int = 16384) -> dict:
    """One `tracks_<run>_<subrun>.parquet` -> one web shard. Returns its index
    entry."""
    m = SRC_RE.match(src.name)
    if not m:
        raise ValueError(f'not a stage3 track table: {src.name}')
    run, subrun = m.group(1), m.group(2)

    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(src).schema.names)
    absent = [c for c in COLUMNS if c not in have]
    fatal = [c for c in absent if c not in OPTIONAL_COLS]
    if fatal:
        raise KeyError(f'{src.name}: missing {fatal} -- not a build_tracks '
                       f'output, or the schema moved.')

    df = pd.read_parquet(src, columns=[c for c in COLUMNS if c in have])
    for c in absent:
        # Present as a column, empty as a value, and typed like everywhere else:
        # the shard keeps ONE schema across all 292 files, so the page never has
        # to branch on which columns a sub-run happens to have. A bare
        # `df[c] = None` would give arrow's `null` type instead of `string`,
        # which is a different schema and a reader that may refuse it.
        df[c] = pd.Series([None] * len(df), dtype='object', index=df.index)
    df = df[COLUMNS]
    n_all = len(df)
    if gated_only:
        df = df[df['gated'].to_numpy()]
    df = _shrink(df.copy())
    # Sorted so an event's tracks land in one row group: "show me event N" is
    # then one range request, not one per arm.
    df = df.sort_values(['event_id', 'arm', 'track_id']).reset_index(drop=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    table = pa.Table.from_pandas(df, preserve_index=False)
    # Force `string`, not `null`, for anything filled in above.
    if absent:
        schema = table.schema
        for c in absent:
            i = schema.get_field_index(c)
            if i >= 0 and pa.types.is_null(schema.field(i).type):
                schema = schema.set(i, pa.field(c, pa.string()))
        table = table.cast(schema)
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        ABSENT_KEY: json.dumps(absent).encode(),
    })
    pq.write_table(table, dest, compression='snappy',
                   row_group_size=row_group)

    n_gated = int(df['gated'].sum())
    return {
        'run': run, 'subrun': subrun, 'file': dest.name,
        'absent': absent,
        'bytes': dest.stat().st_size,
        'n_tracks': len(df), 'n_all': n_all, 'n_gated': n_gated,
        'n_events': int(df['event_id'].nunique()),
        'tags': sorted(df['tag'].unique().tolist()),
        'arms': sorted(df['arm'].unique().tolist()),
        'condition': _condition(run),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', type=Path, default=None,
                    help='default <out>/stage3_fullpass')
    ap.add_argument('--dest', type=Path, default=None,
                    help='default <out>/trackqa_web/tracks')
    ap.add_argument('--run', default=None, help='one run; default all present')
    ap.add_argument('--gated-only', action='store_true',
                    help='half the size, and blind to what the gate removed')
    ap.add_argument('--row-group', type=int, default=16384)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    src = a.src or paths.out('stage3_fullpass')
    dest = a.dest or (paths.out('trackqa_web') / 'tracks')
    paths.require(src, 'stage3 track tables')

    files = sorted(p for p in Path(src).glob('tracks_run_*.parquet')
                   if SRC_RE.match(p.name)
                   and (a.run is None or SRC_RE.match(p.name).group(1) == a.run))
    if not files:
        print(f'no stage3 shards under {src}'
              + (f' for {a.run}' if a.run else ''), file=sys.stderr)
        return 1

    index, n_bytes, n_tracks = [], 0, 0
    for i, p in enumerate(files, 1):
        m = SRC_RE.match(p.name)
        out = Path(dest) / f'{m.group(1)}__{m.group(2)}.parquet'
        if out.exists() and not a.force:
            # Re-indexed, not rebuilt: the index must list every shard that is
            # there, or a resumed run would ship a truncated index.
            df = pd.read_parquet(out, columns=['event_id', 'arm', 'tag', 'gated'])
            entry = {'run': m.group(1), 'subrun': m.group(2), 'file': out.name,
                     'bytes': out.stat().st_size, 'n_tracks': len(df),
                     'n_all': None, 'n_gated': int(df['gated'].sum()),
                     'n_events': int(df['event_id'].nunique()),
                     'tags': sorted(df['tag'].unique().tolist()),
                     'arms': sorted(df['arm'].unique().tolist()),
                     'absent': read_absent(out),
                     'condition': _condition(m.group(1))}
        else:
            entry = build_shard(p, out, a.gated_only, a.row_group)
        index.append(entry)
        n_bytes += entry['bytes']
        n_tracks += entry['n_tracks']
        print(f'  [{i:3d}/{len(files)}] {entry["run"]}/{entry["subrun"]:<18s}'
              f' {entry["n_tracks"]:>7d} tracks  {entry["bytes"]/1e6:6.2f} MB',
              flush=True)

    meta = {
        'schema': SCHEMA,
        'generated': pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%S'),
        'src': str(src),
        'gated_only': bool(a.gated_only),
        'row_group': a.row_group,
        'compression': 'snappy',
        'columns': COLUMNS,
        'optional_columns': list(OPTIONAL_COLS),
        'n_shards': len(index), 'n_tracks': n_tracks, 'bytes': n_bytes,
        'runs': sorted({e['run'] for e in index},
                       key=lambda r: int(r.split('_')[1])),
        'shards': index,
    }
    idx = Path(dest).parent / 'shard_index.json'
    idx.write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(f'\n{len(index)} shards, {n_tracks:,} tracks, {n_bytes/1e9:.2f} GB')
    print(f'index -> {idx}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
