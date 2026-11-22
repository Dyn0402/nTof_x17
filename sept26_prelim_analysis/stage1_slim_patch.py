#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage1_slim_patch.py -- re-attach the n_TOF columns of stage 1 from EVERY slim of a sub-run.

Tracking audit 2026-11-22, finding 12: 77 of the 293 production sub-runs span two (one spans three) n_TOF runs and so
have one slim per n_TOF run, and ``candidate_filter`` read the first only.  Its triggers in the other file(s) got no
n_TOF columns (bunch, is_flash, matched, coinc/wall/plastic/liq per arm) and could never be ``IMPLIED``.

The fix is table-level and exact, so stage 1 (one ~1.5 h condor job per sub-run over the combined hits) is not re-run:

  * the strip-side columns (per-arm strips, track/separated/point/blob counts, ``n_arms_*``, ``n_sep_best``,
    ``n_busy_arms``) do not depend on the slim and are kept;
  * the n_TOF columns are dropped and re-joined on ``eventId`` from ``candidate_filter.arm_flags_subrun`` over all the
    slims (``ntof_run`` added: bunch restarts per n_TOF run);
  * the class: in ``candidate_filter.classify`` the slim enters through one branch only -- an event with exactly one
    lit arm and fewer than two separated clusters is ``IMPLIED`` when ``n_coinc_arms >= 2``, else ``SINGLE`` -- so
    only SINGLE/IMPLIED rows can change, and they are re-decided by that rule.  Before patching, the rule is checked
    to reproduce the existing classes from the existing columns, and the SINGLE/IMPLIED set is checked against the
    stored strip-side numbers; either failing stops the sub-run;
  * ``census_*``, ``census_per_tag_*`` and ``arm_rates_*`` are rewritten from the patched table, with
    ``candidate_filter``'s own functions.  ``hot_mask_*`` and ``bench_*`` do not change and are not copied.

    python -m sept26_prelim_analysis.stage1_slim_patch --list aff.txt \\
        --in-dir <old stage1> --out-dir <new dir> [--runs-root <dir with run_X/<sub>/ntof_hits>] [--force]

``--list``: lines ``run_X stat090_NNNN``.  ``--out-dir`` is required and must differ from ``--in-dir``: the originals
are never overwritten.  Writes, per sub-run, ``candidates_*.parquet``, the three CSVs and ``slim_patch_*.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import candidate_filter as CF   # noqa: E402
from sept26_prelim_analysis import paths                    # noqa: E402

SI = ('SINGLE', 'IMPLIED')


def single_implied(n_coinc_arms) -> np.ndarray:
    """``classify``'s last branch: IMPLIED when >= 2 arms are in n_TOF coincidence, else SINGLE.

    A trigger with no n_TOF record counts 0 coincident arms, as ``run_subrun`` does (``coinc.get(ev, 0)``).
    """
    n = pd.to_numeric(pd.Series(n_coinc_arms), errors='coerce').fillna(0).to_numpy()
    return np.where(n >= 2, 'IMPLIED', 'SINGLE')


def check_rule(df: pd.DataFrame, strict: bool, where: str) -> None:
    """Refuse to patch unless the rule reproduces the table as it is.

    (1) the SINGLE/IMPLIED rows are exactly the one-lit-arm, < 2 separated, not-busy rows of ``classify``;
    (2) on them, the rule applied to the stored ``n_coinc_arms`` gives the stored class.
    """
    lit = df['n_arms_strict' if strict else 'n_arms_loose'].to_numpy()
    shape = ((df['n_busy_arms'].to_numpy() < CF.BUSY_MIN_ARMS) & (lit == 1)
             & (df['n_sep_best'].to_numpy() < 2))
    si = df['cls'].isin(SI).to_numpy()
    if (shape != si).any():
        raise RuntimeError(f'{where}: {int((shape != si).sum())} rows where SINGLE/IMPLIED does not match '
                           f'the stored strip-side numbers (strict={strict}) -- not patching')
    old = df.loc[si, 'n_coinc_arms'] if 'n_coinc_arms' in df else pd.Series(np.zeros(si.sum()))
    bad = int((single_implied(old) != df.loc[si, 'cls'].to_numpy()).sum())
    if bad:
        raise RuntimeError(f'{where}: the SINGLE/IMPLIED rule does not reproduce {bad} stored classes -- not patching')


def patch(df: pd.DataFrame, flags: pd.DataFrame | None, strict: bool = True, where: str = '') -> pd.DataFrame:
    """The candidates table with its n_TOF columns and SINGLE/IMPLIED classes re-derived from ``flags``.

    ``flags`` is ``candidate_filter.arm_flags_subrun`` over every slim (None: no slim, columns dropped).
    The other columns, the row order and the column order are kept; ``ntof_run`` goes before ``bunch``.
    """
    check_rule(df, strict, where)
    orig = list(df.columns)
    base = df.drop(columns=[c for c in CF.NTOF_COLS if c in df.columns])
    if flags is not None:
        if flags['eventId'].duplicated().any():
            raise ValueError(f'{where}: duplicate eventId in the slim flags')
        out = base.merge(flags[['eventId'] + CF.NTOF_COLS], on='eventId', how='left')
        if len(out) != len(df) or not (out['eventId'].to_numpy() == df['eventId'].to_numpy()).all():
            raise RuntimeError(f'{where}: the join changed the rows')
    else:
        out = base
    si = out['cls'].isin(SI).to_numpy()
    if si.any():
        nc = out.loc[si, 'n_coinc_arms'] if 'n_coinc_arms' in out else np.zeros(si.sum())
        out.loc[si, 'cls'] = single_implied(nc)
    # column order: the original one, ntof_run in front of bunch (or the n_TOF block before run/subrun/tag)
    have = [c for c in CF.NTOF_COLS if c in out.columns]
    rest = [c for c in orig if c not in CF.NTOF_COLS]
    if 'bunch' in orig:          # where the n_TOF block was
        i = sum(c not in CF.NTOF_COLS for c in orig[:orig.index('bunch')])
    else:
        i = rest.index('run') if 'run' in rest else len(rest)
    return out[rest[:i] + have + rest[i:]]


def strict_of(in_dir: Path, stem: str, default: bool) -> bool:
    """The ``strict`` the stage-1 run used, from its bench sidecar, else ``default``."""
    p = in_dir / f'bench_{stem}.json'
    if p.exists():
        v = json.loads(p.read_text()).get('strict')
        if v is not None:
            return bool(v)
    return default


def patch_subrun(run: str, sub: str, in_dir: Path, out_dir: Path, runs_root=None,
                 force: bool = False, default_strict: bool = True) -> dict:
    stem = f'{run}_{sub}'
    src = in_dir / f'candidates_{stem}.parquet'
    dst = out_dir / f'candidates_{stem}.parquet'
    if dst.exists() and not force:
        return dict(run=run, subrun=sub, skipped=True)
    df = pd.read_parquet(paths.require(src, f'stage-1 candidates for {run}/{sub}'))
    slims = paths.slim_files(run, sub, runs_root)
    if not slims:
        raise FileNotFoundError(f'no slim for {run}/{sub} under {runs_root or paths.spell("runs")}')
    flags = CF.arm_flags_subrun(slims, where=f'{run}/{sub}')
    strict = strict_of(in_dir, stem, default_strict)
    new = patch(df, flags, strict=strict, where=f'{run}/{sub}')

    def n_flag(d):
        return int(d['bunch'].notna().sum()) if 'bunch' in d else 0

    before = df['cls'].value_counts().to_dict()
    after = new['cls'].value_counts().to_dict()
    info = dict(
        run=run, subrun=sub, built=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        built_by='sept26_prelim_analysis/stage1_slim_patch.py (tracking audit 2026-11-22, finding 12)',
        source=str(src), slims=[p.name for p in slims],
        ntof_runs=[paths.slim_ntof_run(p) for p in slims], strict=strict,
        n_triggers=int(len(df)), n_with_ntof_before=n_flag(df), n_with_ntof_after=n_flag(new),
        cls_before={k: int(v) for k, v in before.items()}, cls_after={k: int(v) for k, v in after.items()},
        n_cls_changed=int((df['cls'].to_numpy() != new['cls'].to_numpy()).sum()))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.parquet.part')
    new.to_parquet(tmp, index=False)
    tmp.replace(dst)
    CF.census(new).to_csv(out_dir / f'census_{stem}.csv', index=False)
    if 'tag' in new and new['tag'].astype(str).str.len().gt(0).any():
        CF.census_per_tag(new).to_csv(out_dir / f'census_per_tag_{stem}.csv', index=False)
    CF.arm_rates(new).to_csv(out_dir / f'arm_rates_{stem}.csv', index=False)
    (out_dir / f'slim_patch_{stem}.json').write_text(json.dumps(info, indent=1))
    return info


def read_list(p: Path) -> list[tuple[str, str]]:
    out = []
    for line in Path(p).read_text().splitlines():
        line = line.split('#')[0].strip()
        if not line:
            continue
        f = line.replace('/', ' ').split()
        if len(f) != 2:
            raise ValueError(f'{p}: expected "run_X stat090_NNNN", got {line!r}')
        out.append((f[0], f[1]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--list', type=Path, required=True, help='lines "run_X stat090_NNNN"')
    ap.add_argument('--in-dir', type=Path, required=True, help='the existing stage-1 directory (read only)')
    ap.add_argument('--out-dir', type=Path, required=True, help='where the patched tables go (not --in-dir)')
    ap.add_argument('--runs-root', type=Path, default=None,
                    help='directory holding run_X/<sub>/ntof_hits (default: the runs root, $X17_RUNS)')
    ap.add_argument('--loose', action='store_true',
                    help='assume stage 1 ran --loose where its bench sidecar does not say')
    ap.add_argument('--force', action='store_true', help='redo sub-runs already in --out-dir')
    a = ap.parse_args(argv)
    if a.out_dir.resolve() == a.in_dir.resolve():
        ap.error('--out-dir must differ from --in-dir: the original stage-1 tables are never overwritten')
    pairs = read_list(a.list)
    bad = 0
    for i, (run, sub) in enumerate(pairs, 1):
        try:
            r = patch_subrun(run, sub, a.in_dir, a.out_dir, a.runs_root, a.force, default_strict=not a.loose)
        except Exception as exc:                                    # noqa: BLE001
            bad += 1
            print(f'[{i}/{len(pairs)}] {run}/{sub}  FAILED -- {type(exc).__name__}: {exc}', flush=True)
            continue
        if r.get('skipped'):
            print(f'[{i}/{len(pairs)}] {run}/{sub}  exists (use --force)', flush=True)
            continue
        print(f'[{i}/{len(pairs)}] {run}/{sub}  n_TOF runs {r["ntof_runs"]}: with n_TOF record '
              f'{r["n_with_ntof_before"]:,} -> {r["n_with_ntof_after"]:,} of {r["n_triggers"]:,}; '
              f'IMPLIED {r["cls_before"].get("IMPLIED", 0)} -> {r["cls_after"].get("IMPLIED", 0)}', flush=True)
    print(f'{len(pairs) - bad} / {len(pairs)} sub-runs patched into {a.out_dir}')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
