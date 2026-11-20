#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pass1_tracks.py -- where downstream analysis finds the n_TOF pass-1 products, and one reader.

The n_TOF pass 1 (`rerun/ntof/NTOF_PASS_PLAN_2026-11-02.md`, final 2026-11-20) replaces
``stage3_fullpass``, ``stage3_is2_v2``, ``stage3_is2_v2_cf`` and ``reco_fullpass`` /
``fullpass_is2_v2``.  Three local directories, all under ``$WORK_DATA/ntof_pass1/`` (desktop ucla;
``~/work_data/ntof_pass1/`` on the laptop; ``$X17_PASS1`` overrides the base):

``stage3/``     the pass's stage 3 as on EOS ``Analysis_pass1/stage3/`` (MD5SUMS): one table per
                sub-run (543), all four arms, frozen-v columns + ``<col>_vt`` twins.
``stage3_vt/``  **the analysis view** (``rerun/ntof/stage3/vt_view.py``): the same tables with the
                v(t) angles under the plain names (``tanx`` = stage-3 ``tanx_vt``, ``gated`` =
                ``gated_vt``, ...) and the frozen ones as ``<col>_frozen``.  This is what a script
                of the is2_v2 era reads: same file names (``tracks_run_<r>_<sub>.parquet``), same
                columns.  Decision (Dylan 2026-11-17): v(t) table correction, acceptance true
                |tan| < 0.5 (``gated`` already includes it).  There is no k: ``k_arm`` is 1,
                and ``tan_raw_x/y`` hold the v(t) angle too (frozen as ``tan_raw_*_frozen``).
                B has no angle (positions only: its angles are null, gated = gated_reco).
``reco_tree/``  the canonical stage-2 outputs merged per (sub-run, arm) in the old layout
                ``run_<r>/<sub>/mx17_<arm>/events_prelim.parquet`` (``rerun/ntof/stage3/reco_tree.py``).

Both derived directories are regenerable from EOS; neither is archived.

    from sept26_prelim_analysis import pass1_tracks as p1
    d = p1.read(['run', 'arm', 'event_id', 'gated', 'tanx', 'tany'], arms=('A', 'C'))

    python -m sept26_prelim_analysis.pass1_tracks          # where everything resolves
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

EOS_PASS1 = '/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1'
EOS_STAGE3 = EOS_PASS1 + '/stage3'
#: the analysis acceptance on the v(t) angle (vt_join.TAN_MAX_VT); `gated` in the view applies it
TAN_MAX_VT = 0.5
CHAIN = 'pass1'
#: runs before the 27 July access: chamber A x connector 8 dead (CLAUDE.md)
PRE_ACCESS_RUNS = ('run_79', 'run_81')


def base() -> Path:
    env = os.environ.get('X17_PASS1')
    cands = [Path(env).expanduser()] if env else []
    wd = os.environ.get('WORK_DATA')
    if wd:
        cands.append(Path(wd) / 'ntof_pass1')
    cands += [Path('/media/ucla/work/ntof_pass1'), Path.home() / 'work_data' / 'ntof_pass1']
    for p in cands:
        if (p / 'stage3').is_dir():
            return p
    raise FileNotFoundError('n_TOF pass-1 products not found; tried ' + ', '.join(map(str, cands)) +
                            f'; set $X17_PASS1, or fetch stage3/ from {EOS_STAGE3}')


def _need(name: str, how: str) -> Path:
    p = base() / name
    if not p.is_dir():
        raise FileNotFoundError(f'{p} missing: build it with {how}')
    return p


def stage3_dir(view: str = 'vt') -> Path:
    """``view='vt'`` the analysis view (default), ``'raw'`` the stage 3 as on EOS."""
    if view == 'raw':
        return _need('stage3', f'a copy of {EOS_STAGE3}')
    return _need('stage3_vt', 'rerun/ntof/stage3/vt_view.py')


def reco_dir() -> Path:
    return _need('reco_tree', 'rerun/ntof/stage3/reco_tree.py')


def files(runs: Iterable | None = None, view: str = 'vt') -> list[Path]:
    """Sub-run tables, sorted; ``runs`` as ints or 'run_145' strings."""
    fs = sorted(stage3_dir(view).glob('tracks_run_*_*.parquet'))
    if runs is not None:
        want = {r if str(r).startswith('run_') else f'run_{int(r)}' for r in runs}
        fs = [f for f in fs if '_'.join(f.stem.split('_')[1:3]) in want]
    return fs


_SCHEMA = {}


def campaign_schema(view: str = 'vt'):
    """One schema over every sub-run table: beam-off sub-runs lack the flash columns, which read as null."""
    if view not in _SCHEMA:
        import pyarrow as pa
        import pyarrow.parquet as pq
        _SCHEMA[view] = pa.unify_schemas([pq.read_schema(f).remove_metadata() for f in files(None, view)],
                                         promote_options='permissive')
    return _SCHEMA[view]


def read(columns: Sequence[str], *, view: str = 'vt', arms: Iterable[str] | None = None,
         runs: Iterable | None = None, filter=None) -> pd.DataFrame:
    """Read ``columns`` over the campaign (or ``runs``), optionally for ``arms`` only.

    ``filter`` is an extra ``pyarrow.dataset`` expression.
    """
    import pyarrow.dataset as ds
    fs = files(runs, view)
    if not fs:
        raise FileNotFoundError(f'no pass-1 stage-3 tables for runs={runs}')
    dset = ds.dataset([str(f) for f in fs], format='parquet', schema=campaign_schema(view))
    flt = filter
    if arms is not None:
        e = ds.field('arm').isin(list(arms))
        flt = e if flt is None else (flt & e)
    return dset.to_table(columns=list(columns), filter=flt).to_pandas()


def _main():
    b = base()
    print(f'pass-1 base {b}')
    for n in ('stage3', 'stage3_vt', 'reco_tree'):
        p = b / n
        k = len(list(p.glob('tracks_run_*.parquet'))) if n.startswith('stage3') else \
            len(list(p.glob('run_*'))) if p.is_dir() else 0
        print(f'  {n:10s} {"present" if p.is_dir() else "MISSING"}  {k}')


if __name__ == '__main__':
    _main()
