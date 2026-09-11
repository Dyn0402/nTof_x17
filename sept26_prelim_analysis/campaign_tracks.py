#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
campaign_tracks.py -- run stage 3 over every sub-run that has a reco, and
concatenate into the one table that leaves this machine.

The deliverable is "tracks and n_TOF hits for the full dataset" to analyse
locally. This is the tracks half; `slim_export.py` is the other.

**The FULL stage-3 schema ships -- fit diagnostics included, float64 kept.**
An earlier version of this module projected to ~83 columns and downcast to
float32 to hit a size budget; the budget was invented rather than required
(Dylan, 2026-09-09), and the per-plane diagnostics (`*_dchi2`, `*_ftst`,
`*_rank`, `*_n_seed`, the errors) are exactly what a resolution or
gate-efficiency study needs. At ~10 GB campaign-wide the table is not
inconvenient enough to justify throwing any of it away.

ANGLES ARE NOT A DEPENDENCY OF THIS STEP, WHICH IS WHY IT CAN RUN FIRST.
`build_tracks` always writes `tan_raw_x`/`tan_raw_y` and records the `k_arm`
it used plus an `angle_calibrated` flag. Where no calibration exists the raw
tangents are still there and `k` is a single scalar per (arm, run), so the
calibrated angle can be applied to the shipped table later without
re-reconstructing anything:

    tanx = tan_raw_x / k_arm

So a missing `k_arm_<run>.json` costs the `tanx`/`angle_to_beam_deg` columns
for that run, and nothing else. It does not block the pass, and it is not
silently papered over -- `angle_calibrated` says which rows have it.

`--jobs` runs sub-runs in parallel; each is independent.

    python -m sept26_prelim_analysis.campaign_tracks --jobs 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')


def expected_tags(run: str, sub: str) -> dict:
    """Per arm, the file tags the stage-2 allowlist selected events in.

    This is the completeness reference. Stage-2 jobs land asynchronously, so a
    sub-run can easily have 5 of its 13 tags present; `build_tracks.load_reco`
    globs whatever is there and would build a PARTIAL table with no indication
    that it is partial. Comparing against the allowlist is the only honest
    check -- the number of tags is not derivable from the products themselves.
    """
    import json
    p = paths.out('stage2') / f'allowlist_{run}_{sub}.json'
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    ev = doc.get('events', doc)
    return {arm: {t for t, e in tags.items() if e}
            for arm, tags in ev.items() if isinstance(tags, dict)}


def incomplete_arms(run: str, sub: str, reco: Path) -> dict:
    """{arm: (n_present, n_expected)} for arms missing stage-2 output."""
    want = expected_tags(run, sub)
    if not want:
        return {}
    bad = {}
    for arm, tags in want.items():
        d = Path(reco) / f'mx17_{arm}'
        if not d.is_dir():
            bad[arm] = (0, len(tags))
            continue
        have = {p.name[len('events_'):-len('.candidates.parquet')]
                for p in d.glob('events_*.candidates.parquet')}
        # A sub-run reconstructed directly by wft_beam (run_145's local pass)
        # carries one merged `events_prelim.candidates.parquet` instead of the
        # per-tag files, and is complete. Treat that as complete rather than
        # reporting 0/N against tags it was never going to have.
        if 'prelim' in have:
            continue
        if not tags <= have:
            bad[arm] = (len(have & tags), len(tags))
    return bad

#: The 27 July access is a condition boundary, found 2026-09-09 from the
#: campaign stage-1 census (OVERNIGHT_2026-09-08.md). Median clean strips per
#: trigger steps DOWN across it -- arm A from ~10 to 0, C from ~2 to 0 -- and
#: run_79/run_81 consequently carry about TWICE every other run's INTER and
#: INTRA fraction. INTER is the signal topology, so pooling across this
#: boundary puts a detector artefact into the pair rate.
#:
#: Stamped as a column so a downstream analysis has to CHOOSE to pool rather
#: than doing it by not knowing. run_79 and run_81 are the only in-sample runs
#: before the access; the 23 July noise boundary needs no column because the
#: whole in-sample set is already on its noisy side.
LAST_PRE_ACCESS_RUN = 81


def _condition(run: str) -> str:
    try:
        n = int(str(run).split('_')[1])
    except (IndexError, ValueError):
        return 'unknown'
    return 'pre_access_27jul' if n <= LAST_PRE_ACCESS_RUN else 'post_access_27jul'

def _k_for(run: str, k_from: str | None):
    """The angle scale to use for `run`, and where it came from.

    `k_from` pins every run to one run's certified k. That is the 2026-09-09
    decision: the calibration pass showed the run-to-run spread on arm A is
    3.7 % over two weeks, while the SAME run measured two ways differs by
    6.5 % -- so k does not meaningfully drift and the estimator is
    sample-dependent. Applying one well-measured k campaign-wide is therefore
    better than per-run values that each carry an uncalibrated offset, at the
    price of a ~4-7 % systematic on the angle scale.

    Returns (k dict, source string). The source is stamped on every row so a
    borrowed k can never be read back as a per-run measurement.
    """
    src_run = k_from or run
    kf = paths.out('kcal') / f'k_arm_{src_run}.json'
    if not kf.exists():
        return {}, 'none'
    cal = json.loads(kf.read_text())
    k = {a: float(v) for a, v in (cal.get('apply') or {}).items()}
    if not k:
        return {}, 'none'
    return k, ('self' if src_run == run else src_run)


def _one(run: str, sub: str, reco: str, out_dir: str, k_from=None):
    """One sub-run, in its own process. Returns (run, sub, n, err)."""
    from sept26_prelim_analysis import build_tracks as BT
    try:
        s1 = paths.out('stage1') / f'candidates_{run}_{sub}.parquet'
        al = paths.out('stage2') / f'allowlist_{run}_{sub}.parquet'
        k, ksrc = _k_for(run, k_from)
        tracks, meta = BT.build(run, sub, Path(reco),
                                stage1=s1 if s1.exists() else None,
                                allow=al if al.exists() else None,
                                out_dir=Path(out_dir), k_arm=k)
        return run, sub, int(meta['n_tracks']), int(meta['n_gated']), ksrc, None
    except Exception:                                          # noqa: BLE001
        return run, sub, 0, 0, 'none', traceback.format_exc(limit=3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--fullpass', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--k-from', default=None,
                    help='apply this run\'s certified k to EVERY run (e.g. '
                         'run_145). Stamped in k_source so a borrowed scale is '
                         'never mistaken for a per-run measurement.')
    ap.add_argument('--rebuild', action='store_true',
                    help='rebuild sub-runs whose track table is already newer '
                         'than its inputs')
    ap.add_argument('--allow-partial', action='store_true',
                    help='build sub-runs whose stage-2 output is incomplete. '
                         'OFF by default: a partial tag set produces a track '
                         'table that looks complete and is not.')
    ap.add_argument('--combine-only', action='store_true',
                    help='skip building; just concatenate what is already there')
    a = ap.parse_args()
    base = a.fullpass or paths.out('fullpass')
    out = a.out or paths.out('stage3_campaign')

    work, partial, fresh = [], [], []
    for r in sorted(p for p in base.iterdir() if p.is_dir()
                    and p.name.startswith('run_')):
        for s in sorted(p for p in r.iterdir() if p.is_dir()):
            if not any((s / f'mx17_{arm}').is_dir() for arm in ARMS):
                continue
            bad = incomplete_arms(r.name, s.name, s)
            if bad and not a.allow_partial:
                partial.append((r.name, s.name, bad))
                continue
            # Skip a sub-run whose track table is already newer than every
            # reco input it was built from. The chain is meant to be re-run as
            # the campaign lands, and without this each pass rebuilds all of it.
            # Compared against the INPUTS, not just existence, so a sub-run
            # that gained tags since the last build is correctly rebuilt.
            dest = out / f'tracks_{r.name}_{s.name}.parquet'
            if dest.exists() and not a.rebuild:
                newest = max((p.stat().st_mtime
                              for p in s.rglob('events_*.parquet')), default=0)
                if dest.stat().st_mtime >= newest:
                    fresh.append((r.name, s.name))
                    continue
            work.append((r.name, s.name, str(s)))
    if fresh:
        print(f'{len(fresh)} sub-run(s) already up to date, skipped')
    if partial:
        print(f'{len(partial)} sub-run(s) SKIPPED as incomplete -- stage 2 has '
              f'not finished them. Re-run when it has (this is idempotent); '
              f'--allow-partial builds them anyway.')
        for run, sub, bad in partial[:8]:
            detail = ', '.join(f'{k} {v[0]}/{v[1]} tags' for k, v in sorted(bad.items()))
            print(f'    {run}/{sub}: {detail}')
        if len(partial) > 8:
            print(f'    ... and {len(partial) - 8} more')

    if not a.combine_only:
        print(f'building {len(work)} sub-run(s) on {a.jobs} process(es)')
        ok = fail = nocal = 0
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(_one, run, sub, reco, str(out), a.k_from)
                    for run, sub, reco in work]
            for f in as_completed(futs):
                run, sub, n, ng, ksrc, err = f.result()
                if err:
                    fail += 1
                    print(f'  !! {run}/{sub}\n{err}')
                else:
                    ok += 1
                    nocal += (1 if ksrc == 'none' else 0)
                    note = {'none': '   [no k_arm -- raw angles only]',
                            'self': ''}.get(ksrc, f'   [k from {ksrc}]')
                    print(f'  {run}/{sub}: {n:,} segments, {ng:,} gated{note}')
        print(f'built {ok}, failed {fail}, {nocal} without an angle calibration')

    # ---- combine
    parts = sorted(out.glob('tracks_run_*.parquet'))
    parts = [p for p in parts if p.name != 'tracks_campaign.parquet']
    if not parts:
        print('nothing to combine')
        return 1
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    if 'run' in df.columns:
        df['condition'] = df['run'].map(_condition).astype('category')
        df['k_source'] = (a.k_from if a.k_from else 'self')
        df['k_source'] = df['k_source'].astype('category')
    dest = out / 'tracks_campaign.parquet'
    df.to_parquet(dest, index=False, compression='snappy')

    meta = dict(n_subruns=len(parts), n_tracks=int(len(df)),
                n_gated=int(df.gated.sum()) if 'gated' in df else None,
                n_calibrated=int(df.angle_calibrated.sum())
                if 'angle_calibrated' in df else None,
                runs=sorted(df.run.unique().tolist()) if 'run' in df else [],
                bytes=int(dest.stat().st_size),
                n_columns=int(len(df.columns)),
                condition_counts=(df.condition.value_counts().to_dict()
                                  if 'condition' in df else {}))
    (out / 'tracks_campaign.meta.json').write_text(json.dumps(meta, indent=1))
    print(f'\ncombined {len(parts)} sub-run(s) -> {dest}')
    print(f'  {meta["n_tracks"]:,} segments, {meta["n_gated"]:,} gated, '
          f'{meta["n_calibrated"]:,} with a calibrated angle')
    print(f'  {meta["bytes"] / 1e6:.0f} MB, {len(df.columns)} columns '
          f'(full stage-3 schema -- fit diagnostics kept)')
    if meta['condition_counts']:
        print(f'  condition: {meta["condition_counts"]}  '
              f'(27 Jul access boundary -- do not pool INTER/INTRA across it)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
