#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0 -- freeze the sample.  ``PLAN.md`` §3, stage 0.

Decides which (run, sub-run) enter the analysis, records **what each cut costs**,
and writes one row per sub-run so every later denominator traces back here.

    python -m sept26_prelim_analysis.freeze_sample            # -> sample.csv + figure
    python -m sept26_prelim_analysis.freeze_sample --explain  # cut ledger only

Three inputs, all frozen, none re-derived here:

* ``x17-runs.json``  -- run-level survey of EOS: mode, target, gas, FEUs, status,
  sub-run count, events, size.  Frozen by the site's ``freeze_x17_runs.py``.
* ``x17-match.json`` -- one record per (run, sub-run) from the pulse ledger: the
  n_TOF run it joined to, delivered/matched/lost pulses, and the 21-check verdict.
* ``data/slim_inventory_*.tsv`` -- every n_TOF slim product actually on EOS.
  A sub-run without one cannot enter the sample, however good its match record.

**The sub-run naming differs between the three** and that is the one fiddly part
of this script.  The run survey counts sub-runs but does not name them; the match
ledger strips the configuration prefix (``stat090_0000`` -> ``0000``); the slim
files carry the full name.  :func:`_short` is the single place that reconciles
them, and :func:`_check_join` fails loudly if the reconciliation ever stops
covering the sample rather than quietly dropping rows.

The cuts, in the order ``PLAN.md`` §3 lists them, each costing what it costs:
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from . import paths, figstyle
except ImportError:                                    # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paths, figstyle                             # type: ignore

HERE = Path(__file__).resolve().parent
SITE_DATA = Path.home() / 'PycharmProjects' / 'dylan-cern-site' / 'data'

#: First run of the production-trigger period (wall AND plastic coincidence).
#: Also the first run on the noisy side of the 23-July RdClk_Div change, so the
#: same boundary discharges PLAN.md §2.1 -- see ntof_pedestal_qa/README.md.
FIRST_PRODUCTION_RUN = 79

#: Runs inside the production window that are not production: scans that happen
#: to be taken with beam.  Named, not pattern-matched, so adding one is a
#: deliberate act with a reason attached.
NOT_PRODUCTION = {
    82: 'watermark x inter-packet-delay 2x2 DAQ scan',
    161: 'detector-A resist x drift HV scan on beam',
}

#: Arm-A x view, channels 448-511 of FEU 3, were disconnected through run_79.
#: run_79 stays in the sample and carries the flag; see PLAN.md §2.2.
A_X_MASK_THROUGH_RUN = 79


def _short(subrun: str) -> str:
    """Sub-run name as the match ledger spells it.

    The ledger drops the configuration prefix, so ``stat090_0000`` is ``0000``
    there.  Names that are already bare, or that use another prefix entirely
    (``cosbounce_...``), pass through unchanged.
    """
    return re.sub(r'^stat090_', '', subrun)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_runs(site: Path = SITE_DATA) -> dict:
    """``{run number: run record}`` from the frozen EOS survey."""
    d = json.loads(paths.require(site / 'x17-runs.json', 'run registry').read_text())
    return {r['n']: r for r in d['runs']}


def load_match(site: Path = SITE_DATA) -> dict:
    """``{(run number, short sub-run): match record}`` from the pulse ledger."""
    d = json.loads(paths.require(site / 'x17-match.json', 'match ledger').read_text())
    out = {}
    for s in d['segs']:
        m = re.fullmatch(r'run_(\d+)', s['d'])
        if m:
            out[(int(m.group(1)), s['s'])] = s
    return out


def load_slim(inventory: Path | None = None) -> dict:
    """``{(run number, full sub-run name): (n_TOF run, bytes)}`` from EOS.

    The inventory is a checked-in listing rather than a live walk: it takes two
    minutes over EOS, and stage 0 has to be reproducible on a laptop with no
    ticket.  Regenerate it with the command in ``data/README.md`` when the tree
    has moved on.
    """
    if inventory is None:
        found = sorted((HERE / 'data').glob('slim_inventory_*.tsv'))
        if not found:
            raise FileNotFoundError(
                f'no slim inventory in {HERE / "data"} -- see data/README.md '
                f'for the one-line command that regenerates it')
        inventory = found[-1]
    out = {}
    for line in Path(inventory).read_text().splitlines():
        if not line.strip():
            continue
        name, size = line.split('\t')
        m = re.fullmatch(r'ntof_hits_run_(\d+)_(.+)_(\d{6})\.root', name)
        if not m:
            raise ValueError(f'unparseable slim filename in {inventory}: {name}')
        out[(int(m.group(1)), m.group(2))] = (int(m.group(3)), int(size))
    return out


def _check_join(rows: pd.DataFrame) -> None:
    """Fail if the name reconciliation stopped covering the sample.

    A silent join failure here would look exactly like a run genuinely having no
    slim product, and would quietly shrink the denominator of every efficiency
    downstream.  So it is an error, not a warning.
    """
    kept = rows[rows.in_sample]
    orphans = kept[~kept.has_slim & ~kept.has_match]
    if len(orphans) > len(kept) * 0.05:
        raise RuntimeError(
            f'{len(orphans)} of {len(kept)} selected sub-runs matched neither the '
            f'pulse ledger nor the slim inventory -- the sub-run naming '
            f'reconciliation in _short() has probably stopped working.\n'
            f'  examples: {orphans.head(5)[["run", "subrun"]].to_dict("records")}')


# --------------------------------------------------------------------------- #
# The cuts
# --------------------------------------------------------------------------- #
def build(site: Path = SITE_DATA, inventory: Path | None = None) -> pd.DataFrame:
    """One row per (run, sub-run) over the whole campaign, with every cut's verdict.

    Rows that fail a cut are **kept**, flagged with the first cut they failed, so
    the table is the cut ledger as well as the sample.  ``in_sample`` is the
    single boolean that says what the analysis runs on.
    """
    runs, match, slim = load_runs(site), load_match(site), load_slim(inventory)

    # Sub-run names come from the slim inventory where it has them, and from the
    # match ledger otherwise, because the run survey only counts sub-runs.
    names: dict[int, set] = {}
    for (run, sub) in slim:
        names.setdefault(run, set()).add(sub)
    for (run, short) in match:
        have = {_short(s) for s in names.get(run, set())}
        if short not in have:
            names.setdefault(run, set()).add(short)

    rows = []
    for run in sorted(runs):
        r = runs[run]
        for sub in sorted(names.get(run, set())) or [None]:
            short = _short(sub) if sub else None
            m = match.get((run, short)) if short else None
            sl = slim.get((run, sub)) if sub else None

            # The cuts, in PLAN.md order.  First failure wins, so `cut` reads as
            # "what excluded this sub-run" rather than "everything wrong with it".
            cut = None
            if r['mode'] != 'beam' or not r['phys']:
                cut = f"not beam physics ({r['mode']})"
            elif run < FIRST_PRODUCTION_RUN:
                cut = f'before the production trigger (run_{FIRST_PRODUCTION_RUN})'
            elif run in NOT_PRODUCTION:
                cut = NOT_PRODUCTION[run]
            elif r['tgt'] != '3He':
                cut = f"target {r['tgt']}"
            elif r['gas'] != 'Ar/Iso 90/10':
                cut = f"gas {r['gas']}"
            elif r['st'] != 'complete':
                cut = f"run status {r['st']}"
            elif r['feus'] != [8]:
                cut = f"FEUs {r['feus']}"
            elif m is None or m.get('st') != 'ok':
                cut = f"not joined to n_TOF ({m.get('st') if m else 'no ledger record'})"
            elif sl is None:
                cut = 'no slim product on EOS'

            rows.append(dict(
                run=run, subrun=sub, short=short,
                mode=r['mode'], phys=r['phys'], target=r['tgt'], gas=r['gas'],
                run_status=r['st'], feus=str(r['feus']),
                run_events=r['ev'], run_nsub=r['nsub'], run_gb=r['gb'],
                t_start=r['t'], why=r['why'],
                ntof_run=(sl[0] if sl else (m.get('n') if m else None)),
                minutes=(m.get('min') if m else None),
                pulses_delivered=(m.get('pd') if m else None),
                pulses_matched=(m.get('pm') if m else None),
                pulses_lost=(m.get('pl') if m else None),
                match_status=(m.get('st') if m else None),
                match_verdict=(m.get('v') if m else None),
                has_match=m is not None,
                has_slim=sl is not None,
                slim_bytes=(sl[1] if sl else None),
                cut=cut,
                in_sample=cut is None,
                # Conditions that travel with the row rather than being applied here.
                flag_a_x_mask=(run <= A_X_MASK_THROUGH_RUN),
                noise_config=('quiet' if run <= 67 else
                              'unplaced' if run == 68 else 'noisy'),
            ))

    df = pd.DataFrame(rows)
    _check_join(df)
    return df


def unenumerated(df: pd.DataFrame, site: Path = SITE_DATA) -> pd.DataFrame:
    """Sub-runs that exist on EOS but that neither input names.

    Sub-run names come from the slim inventory and the pulse ledger, so a
    sub-run with **neither** is invisible to :func:`build` -- it does not appear
    even as a cut row.  The run survey counts sub-runs independently (``nsub``),
    and the difference is exactly those.

    They matter because they are part of the honest denominator: "293 sub-runs
    entered" means nothing without "and 3 more exist that we could not place".
    run_145's ``stat090_0003`` is one of them, and it is the development run.
    """
    runs = load_runs(site)
    out = []
    for run in sorted(df[df.run >= FIRST_PRODUCTION_RUN].run.unique()):
        r = runs[run]
        if r['mode'] != 'beam' or not r['phys']:
            continue
        listed = len(df[df.run == run])
        if listed < (r['nsub'] or 0):
            out.append(dict(run=run, nsub_on_eos=r['nsub'], nsub_named=listed,
                            missing=r['nsub'] - listed, run_events=r['ev'],
                            why=str(r['why'])[:60]))
    return pd.DataFrame(out)


def cut_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """What each cut cost, in sub-runs and in beam hours.

    Costs are **sequential**, not independent: a sub-run is charged to the first
    cut it failed, so the column sums to the campaign and no sub-run is
    double-counted.  Beam time is per sub-run from the match ledger where it
    exists; sub-runs with no ledger record contribute no hours, which is why the
    early-cut rows understate hours rather than inventing them.
    """
    order, seen = [], set()
    for c in df.cut:
        key = c if c is not None else '(kept)'
        if key not in seen:
            seen.add(key)
            order.append(key)
    out = []
    for key in order:
        sel = df[df.cut.isna()] if key == '(kept)' else df[df.cut == key]
        out.append(dict(
            cut=key, subruns=len(sel),
            runs=sel.run.nunique(),
            hours=round((sel.minutes.fillna(0).sum()) / 60.0, 1),
            pulses=int(sel.pulses_matched.fillna(0).sum()),
        ))
    led = pd.DataFrame(out).sort_values('subruns', ascending=False)
    return led.reset_index(drop=True)


def summarise(df: pd.DataFrame) -> dict:
    """The headline numbers stage 0 is responsible for."""
    s = df[df.in_sample]
    runs = s.run.nunique()
    # run_events is per RUN, so sum it once per run, not once per sub-run.
    ev = sum(df[df.run == r].run_events.iloc[0] or 0 for r in s.run.unique())
    return dict(
        runs=runs, subruns=len(s),
        run_events_total=int(ev),
        beam_hours=round(s.minutes.fillna(0).sum() / 60.0, 1),
        pulses_matched=int(s.pulses_matched.fillna(0).sum()),
        pulses_delivered=int(s.pulses_delivered.fillna(0).sum()),
        slim_gb=round(s.slim_bytes.fillna(0).sum() / 1e9, 1),
        eos_tb=round(sum(df[df.run == r].run_gb.iloc[0] or 0
                         for r in s.run.unique()) / 1000.0, 2),
        runs_with_a_x_mask=sorted(s[s.flag_a_x_mask].run.unique().tolist()),
    )


# --------------------------------------------------------------------------- #
# The figure -- the first slide of the talk
# --------------------------------------------------------------------------- #
def timeline(df: pd.DataFrame, out_dir: Path):
    """Beam time across the campaign, with what each cut removed shaded out."""
    import matplotlib.pyplot as plt
    import numpy as np

    figstyle.use()
    fig, ax = figstyle.figure(figsize=figstyle.BANNER)

    d = df[df.minutes.notna()].copy()
    d['day'] = pd.to_datetime(d.t_start, unit='s').dt.floor('D')
    grp = (d.groupby(['day', d.in_sample.map({True: 'kept', False: 'cut'})])
             .minutes.sum().unstack(fill_value=0) / 60.0)
    for col in ('cut', 'kept'):
        if col not in grp:
            grp[col] = 0.0
    grp = grp.sort_index()

    x = np.arange(len(grp))
    ax.bar(x, grp['kept'], color=figstyle.DET_COLOR['A'], width=0.78,
           label=f'in the sample ({grp["kept"].sum():.0f} h)')
    ax.bar(x, grp['cut'], bottom=grp['kept'], color=figstyle.LINE, width=0.78,
           label=f'cut ({grp["cut"].sum():.0f} h)')
    # A "05 Aug" label is ~5 % of the frame wide, so more than ~9 of them
    # collide. Stride to that budget rather than to a fixed every-third-day.
    stride = max(1, -(-len(grp) // 9))
    ax.set_xticks(x[::stride])
    ax.set_xticklabels([d.strftime('%-d %b') for d in grp.index[::stride]])
    ax.set_ylabel('beam hours / day')
    ax.legend(loc='upper left', ncol=2)
    figstyle.title(
        ax, f'{len(df[df.in_sample])} sub-runs enter the analysis, '
            f'{grp["kept"].sum():.0f} of {grp.sum().sum():.0f} beam hours',
        'the cuts are almost all "not a production run" -- the production period '
        'itself survives nearly intact')
    figstyle.note(fig, 'sept26_prelim_analysis/freeze_sample.py - '
                       'x17-runs.json + x17-match.json + the EOS slim inventory')
    grp_out = grp.reset_index().rename(columns={'day': 'date'})
    return figstyle.save(fig, out_dir / 'sample_timeline', data=grp_out)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--site-data', type=Path, default=SITE_DATA)
    ap.add_argument('--inventory', type=Path, default=None)
    ap.add_argument('--explain', action='store_true',
                    help='print the cut ledger and stop; write nothing')
    args = ap.parse_args()

    df = build(args.site_data, args.inventory)
    led = cut_ledger(df)
    summ = summarise(df)
    un = unenumerated(df, args.site_data)

    print('cut ledger -- each sub-run charged to the FIRST cut it failed\n')
    print(led.to_string(index=False))
    print('\nsample\n')
    for k, v in summ.items():
        print(f'  {k:<22} {v}')

    if len(un):
        print('\nsub-runs on EOS that neither input names -- part of the '
              'denominator, not of the sample\n')
        print(un.to_string(index=False))
        print(f'\n  {int(un.missing.sum())} sub-run(s) across {len(un)} run(s). '
              f'They have no pulse-ledger record AND no slim product, so nothing '
              f'downstream can use them; they are counted here so the denominator '
              f'stays honest.')

    if args.explain:
        return 0

    out_dir = paths.out('stage0')
    df.to_csv(out_dir / 'sample.csv', index=False)
    led.to_csv(out_dir / 'cut_ledger.csv', index=False)
    (out_dir / 'sample_summary.json').write_text(json.dumps(summ, indent=1))
    print(f'\n  -> {out_dir / "sample.csv"}  ({len(df)} rows)')
    print(f'  -> {out_dir / "cut_ledger.csv"}')
    print(f'  -> {out_dir / "sample_summary.json"}')
    timeline(df, paths.figures('stage0'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
