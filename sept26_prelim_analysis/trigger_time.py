#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trigger_time.py -- the per-trigger time since the gamma flash, and E_n with it.

**The time base was never missing.**  ``neutron_energy.py`` and STATUS both
record ``t_since_flash_ns`` as blocked on a three-line patch to
``slim.py`` plus a campaign-wide slim regeneration on EOS.  That is wrong, and
the reason it went unnoticed for a month is that the quantity is in the slim
under a different name, in a tree nothing downstream reads past ``bunch``:

    slim.py::main       ev_t = ev['t_since_flash_ns']       <- the DREAM side
    slim.py::events     t_dream_ns = ev_t                   <- written, renamed
                        t_pred_ns  = predict(ev_t, K, T0)   <- and calibrated

``bunch_join.dream_events`` builds the DREAM-side value as
``trigger_timestamp_ns - <the burst's first trigger>``; the first trigger of a
burst *is* the flash trigger, so this is already a flash-referenced time, at
the 10 ns granularity of the DREAM timestamp.  ``clockfit`` then maps it onto
the n_TOF time base with the segment's own fitted clock,

    t_nTOF = t_DREAM (1 + K + dk_b) + T0 + a_arm + da_b

and ``t_pred_ns`` is that map applied, per-bunch correction included.  It is
the calibrated quantity and is what this module carries into stage 3.  The
patch ``neutron_energy.slim_patch()`` describes would have added a *third*
copy of the same number, on the hit side.

VERIFIED, not assumed (run_145/stat090_0000, 2026-09-11):

  * flash triggers have ``t_dream_ns`` exactly 0.0, all 639 of them;
  * physics triggers span 0.993-75.04 ms, the N93B gate opening ~1 ms after
    the flash and the acquisition window closing at 75 ms;
  * the fitted map recovers K = 1.118e-4 and T0 = -279 ns from the two columns,
    the values clockfit documents;
  * ``tof - (t_pred_ns + dt_ns)`` -- the slim's own hit times minus this
    prediction -- lands on 11.60-11.65 us per detector tree with a 5-14 ns
    spread.  That is ``tflash``, the per-tree cable delay ``ntof_io`` describes,
    recovered here as a residual.  It closes the loop: the hits and the
    triggers are on the same flash reference.

**And it corrects a number that is on the board.**  ``neutron_energy.py`` took
the hits' ``tof`` floor of 1.00408e6 ns to BE the flash and inferred a reach of
"2.4 MeV down to 0.36 meV".  The real offset is ``tflash`` ~ 11.6 us, not
1.004 ms, so the earliest neutron the window admits is **2.0 eV**, not 2.4 MeV
-- which is what ``../CLAUDE.md`` says independently ("below ~2 eV, which is
every neutron after the 1 ms flash veto") and what the E0/M1 argument in
``ipc_born.py`` rests on.  The window spans 2.0 eV to 0.44 meV with the bulk at
the thermal peak; there is no resonance region in this dataset and no cut can
make one.

WHAT A FLASH TRIGGER GETS.  ``t_since_flash_ns`` = 0 exactly -- true, and the
honest value -- and ``e_neutron_keV`` = NaN, because a gamma flash is not a
neutron and :func:`neutron_energy.energy_eV` refuses beta >= 1 rather than
clipping.  About 1 % of triggers.

    python -m sept26_prelim_analysis.trigger_time extract --all      # on lxplus
    python -m sept26_prelim_analysis.trigger_time backfill           # at home
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import neutron_energy as ne          # noqa: E402
from sept26_prelim_analysis import paths                         # noqa: E402

SCHEMA = 'sept26_prelim/trigtime/1'

#: Columns lifted out of the slim's ``events`` tree.  ``bunch`` and ``matched``
#: are already in stage 1, but a product that cannot be checked against its
#: source on its own is a product nobody trusts, so they ride along.
EVENT_COLS = ('eventId', 'bunch', 'is_flash', 'matched',
              't_dream_ns', 't_pred_ns')

#: The two stage-3 columns this fills.
T_COL, E_COL = 't_since_flash_ns', 'e_neutron_keV'


# --------------------------------------------------------------------------- #
# extraction -- runs where the slims are (lxplus / EOS)
# --------------------------------------------------------------------------- #
def extract(slim_root: str | Path) -> pd.DataFrame:
    """The ``events`` tree of one slim, narrowed and downcast.

    float32 on the times is 6e-8 relative, ~4 ns at the 75 ms end -- three
    orders below the 3.2 ns per-bunch flash jitter the calibration quotes, so
    it costs nothing and halves the product.
    """
    import uproot

    d = uproot.open(str(slim_root))['events'].arrays(list(EVENT_COLS),
                                                     library='pd')
    for c, t in (('eventId', 'int64'), ('bunch', 'int32'),
                 ('is_flash', 'int8'), ('matched', 'int8'),
                 ('t_dream_ns', 'float32'), ('t_pred_ns', 'float32')):
        d[c] = d[c].astype(t)
    return d


def slim_path(run: str, subrun: str, runs_root: str | Path | None = None) -> str:
    root = str(runs_root) if runs_root else str(paths.root('runs'))
    f = sorted(glob.glob(f'{root}/{run}/{subrun}/ntof_hits/*.root'))
    if not f:
        raise FileNotFoundError(f'no slim ROOT under {root}/{run}/{subrun}/ntof_hits')
    return f[0]


def product(run: str, subrun: str, out_dir: Path | None = None) -> Path:
    d = Path(out_dir) if out_dir else paths.out('trigtime')
    return d / f'trigtime_{run}_{subrun}.parquet'


def write_subrun(run: str, subrun: str, runs_root=None,
                 out_dir: Path | None = None, force: bool = False) -> dict:
    dst = product(run, subrun, out_dir)
    if dst.exists() and not force:
        return dict(run=run, subrun=subrun, skipped=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    d = extract(slim_path(run, subrun, runs_root))
    d.to_parquet(dst, index=False, compression='snappy')
    phys = d.loc[d.is_flash == 0, 't_pred_ns']
    return dict(run=run, subrun=subrun, skipped=False, n=len(d),
                n_flash=int((d.is_flash == 1).sum()),
                t_min_ms=float(phys.min()) / 1e6 if len(phys) else np.nan,
                t_max_ms=float(phys.max()) / 1e6 if len(phys) else np.nan)


# --------------------------------------------------------------------------- #
# use -- runs where the tracks are
# --------------------------------------------------------------------------- #
def read(run: str, subrun: str, out_dir: Path | None = None) -> pd.DataFrame:
    p = product(run, subrun, out_dir)
    if not p.exists():
        raise FileNotFoundError(
            f'no trigger-time product at {p} -- run '
            f'"python -m sept26_prelim_analysis.trigger_time extract '
            f'--run {run} --subrun {subrun}" where the slims are')
    return pd.read_parquet(p)


def flash_time_and_energy(t_pred_ns, is_flash):
    """``(t_since_flash_ns, e_neutron_keV)`` from the two slim columns.

    The flash trigger's own time since the flash is 0 by construction, so the
    fitted map's value for it (T0, a few hundred ns of the other sign) is
    replaced -- and its energy is NaN, which the conversion already gives for
    any time that implies beta >= 1.
    """
    t = np.asarray(t_pred_ns, dtype=np.float64).copy()
    t[np.asarray(is_flash).astype(bool)] = 0.0
    return t, ne.energy_keV(t)


def attach(tracks: pd.DataFrame, run: str, subrun: str,
           out_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Fill ``t_since_flash_ns`` and ``e_neutron_keV`` on a stage-3 frame.

    Joins on ``event_id``, which is unique within a sub-run (and only within
    one -- see ``build_tracks``), so this is always called per sub-run.
    """
    tt = read(run, subrun, out_dir).set_index('eventId')
    eid = tracks['event_id'].to_numpy()
    tp = tt['t_pred_ns'].reindex(eid).to_numpy()
    fl = tt['is_flash'].reindex(eid).to_numpy()
    miss = ~np.isfinite(tp)
    t, e = flash_time_and_energy(tp, np.nan_to_num(fl) == 1)
    t[miss] = np.nan
    e[miss] = np.nan
    tracks = tracks.copy()
    tracks[T_COL], tracks[E_COL] = t, e
    ok = np.isfinite(t) & ~miss
    prov = dict(
        source='slim events tree, t_pred_ns (n_TOF time base, fitted clock)',
        product=str(product(run, subrun, out_dir)),
        flight_path_m=ne.FLIGHT_PATH_M,
        n_tracks=int(len(tracks)), n_with_time=int(ok.sum()),
        n_no_trigger_in_slim=int(miss.sum()),
        n_flash_trigger=int((np.nan_to_num(fl) == 1).sum()),
        t_ms=dict(min=float(np.nanmin(t) / 1e6) if ok.any() else None,
                  max=float(np.nanmax(t) / 1e6) if ok.any() else None),
        e_eV=dict(min=float(np.nanmin(e) * 1e3) if ok.any() else None,
                  max=float(np.nanmax(e) * 1e3) if ok.any() else None))
    return tracks, prov


# --------------------------------------------------------------------------- #
# backfill -- the stage-3 tables that were built before this existed
# --------------------------------------------------------------------------- #
def subruns_from(src: Path) -> list[tuple[str, str]]:
    out = []
    for p in sorted(Path(src).glob('tracks_run_*.parquet')):
        if p.name == 'tracks_campaign.parquet':
            continue
        stem = p.stem[len('tracks_'):]
        run, sub = stem.rsplit('_', 2)[0], '_'.join(stem.rsplit('_', 2)[1:])
        out.append((run, sub))
    return out


def backfill_subrun(src: Path, run: str, subrun: str,
                    out_dir: Path | None = None) -> dict:
    """Rewrite one stage-3 parquet with the two columns filled.

    In place: every consumer -- the shards, the QA, the campaign table -- names
    these files, and a parallel tree of "same but with time" would be the kind
    of split this analysis has been bitten by before.  The sidecar records what
    changed and when.
    """
    p = Path(src) / f'tracks_{run}_{subrun}.parquet'
    df = pd.read_parquet(p)
    df, prov = attach(df, run, subrun, out_dir)
    tmp = p.with_suffix('.parquet.new')
    df.to_parquet(tmp, index=False, compression='snappy')
    os.replace(tmp, p)          # never leave a half-written table behind
    mp = Path(src) / f'tracks_{run}_{subrun}.meta.json'
    if mp.exists():
        meta = json.loads(mp.read_text())
        meta.pop('not_populated', None)
        meta['trigger_time'] = dict(
            backfilled=datetime.now(timezone.utc).isoformat(timespec='seconds'),
            **prov)
        mp.write_text(json.dumps(meta, indent=1, default=str))
    return prov


def backfill_campaign(src: Path, out_dir: Path | None = None,
                      row_group: int = 500_000) -> dict:
    """Rewrite ``tracks_campaign.parquet`` streaming, row group by row group.

    11.4 GB and 29 M rows: re-concatenating the parts to fill two columns would
    need most of the machine's memory for a change that is a per-row lookup.
    Instead each row group is read, patched against the per-(run, sub-run)
    trigger-time tables held as a dict of Series, and written straight out.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = Path(src) / 'tracks_campaign.parquet'
    tmp = p.with_suffix('.parquet.new')
    f = pq.ParquetFile(p)
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    n_rows = n_ok = 0
    writer = None
    try:
        for rg in range(f.metadata.num_row_groups):
            df = f.read_row_group(rg).to_pandas()
            t = np.full(len(df), np.nan)
            e = np.full(len(df), np.nan)
            for (run, sub), idx in df.groupby(['run', 'subrun'],
                                              observed=True).groups.items():
                key = (str(run), str(sub))
                if key not in tables:
                    tables[key] = read(run, sub, out_dir).set_index('eventId')
                tt = tables[key]
                pos = df.index.get_indexer(idx)
                eid = df.loc[idx, 'event_id'].to_numpy()
                tp = tt['t_pred_ns'].reindex(eid).to_numpy()
                fl = np.nan_to_num(tt['is_flash'].reindex(eid).to_numpy()) == 1
                tt_, ee_ = flash_time_and_energy(tp, fl)
                bad = ~np.isfinite(tp)
                tt_[bad] = np.nan
                ee_[bad] = np.nan
                t[pos], e[pos] = tt_, ee_
            df[T_COL], df[E_COL] = t, e
            n_rows += len(df)
            n_ok += int(np.isfinite(t).sum())
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, tbl.schema, compression='snappy')
            writer.write_table(tbl)
            if rg % 10 == 0:
                print(f'    row group {rg + 1}/{f.metadata.num_row_groups}  '
                      f'{n_rows:,} rows', flush=True)
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp, p)
    return dict(n_rows=n_rows, n_with_time=n_ok,
                n_subruns=len(tables), bytes=int(p.stat().st_size))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='mode', required=True)

    ex = sub.add_parser('extract', help='slim events tree -> trigtime parquet')
    ex.add_argument('--run')
    ex.add_argument('--subrun')
    ex.add_argument('--all', action='store_true',
                    help='every <run>/<subrun> with a slim under --runs-root')
    ex.add_argument('--list', type=Path, default=None,
                    help='file of "<run> <subrun>" lines instead of a scan')
    ex.add_argument('--runs-root', default=None)
    ex.add_argument('--out', type=Path, default=None)
    ex.add_argument('--force', action='store_true')

    bf = sub.add_parser('backfill', help='fill the two columns in stage 3')
    bf.add_argument('--src', type=Path,
                    default=None, help='stage-3 directory '
                                       '(default <out>/stage3_fullpass)')
    bf.add_argument('--out', type=Path, default=None,
                    help='trigtime directory (default <out>/trigtime)')
    bf.add_argument('--campaign', action='store_true',
                    help='also rewrite tracks_campaign.parquet (streaming)')
    bf.add_argument('--campaign-only', action='store_true')

    a = ap.parse_args()

    if a.mode == 'extract':
        if a.list:
            pairs = [tuple(l.split()) for l in a.list.read_text().split('\n') if l.strip()]
        elif a.all:
            root = Path(a.runs_root or str(paths.root('runs')))
            pairs = sorted((p.parts[-3], p.parts[-2])
                           for p in root.glob('run_*/*/ntof_hits'))
        else:
            pairs = [(a.run, a.subrun)]
        rows = []
        for i, (run, s) in enumerate(pairs, 1):
            try:
                r = write_subrun(run, s, a.runs_root, a.out, a.force)
            except FileNotFoundError as exc:
                print(f'[{i}/{len(pairs)}] {run}/{s}  MISSING -- {exc}')
                continue
            rows.append(r)
            if not r.get('skipped'):
                print(f'[{i}/{len(pairs)}] {run}/{s}  {r["n"]:,} triggers  '
                      f'{r["t_min_ms"]:.3f}-{r["t_max_ms"]:.3f} ms', flush=True)
        print(f'\n{sum(1 for r in rows if not r.get("skipped"))} written, '
              f'{sum(1 for r in rows if r.get("skipped"))} already present')
        return 0

    src = a.src or paths.out('stage3_fullpass')
    if not a.campaign_only:
        pairs = subruns_from(src)
        print(f'backfilling {len(pairs)} stage-3 sub-run table(s) under {src}')
        tot = ok = 0
        for i, (run, s) in enumerate(pairs, 1):
            try:
                pr = backfill_subrun(src, run, s, a.out)
            except FileNotFoundError as exc:
                print(f'  [{i}/{len(pairs)}] {run}/{s}  SKIPPED -- {exc}')
                continue
            tot += pr['n_tracks']; ok += pr['n_with_time']
            if i % 20 == 0 or i == len(pairs):
                print(f'  [{i}/{len(pairs)}] {run}/{s}  {tot:,} tracks, '
                      f'{ok:,} timed', flush=True)
        print(f'\n{ok:,} of {tot:,} tracks carry a time since the flash')
    if a.campaign or a.campaign_only:
        print(f'\nrewriting {src}/tracks_campaign.parquet (streaming)')
        r = backfill_campaign(src, a.out)
        print(f'  {r["n_with_time"]:,} of {r["n_rows"]:,} rows timed, '
              f'{r["n_subruns"]} sub-runs, {r["bytes"] / 1e9:.1f} GB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
