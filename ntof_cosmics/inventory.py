#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inventory.py -- what beam-off cosmic data the n_TOF campaign actually holds,
one row per sub-run, classified so a cosmic analysis can pick its sample
without re-deriving any of this.

WHY. Cosmics, and more generally any single particle that crosses the set-up
in a straight line, are a background in the X17 opening-angle region (the
back-to-back >170 deg population in `sept26_prelim_analysis/tight_coincidence`
is 40 % of the tight opposing sample).  The beam-off runs are where that
population can be studied with nothing else on top.  Stage 0
(`freeze_sample.py`) cut all 47 of them as "not beam physics" and nothing has
read them since; PLAN sec S4 names run_83/run_146 as the cosmic-rate source and
that was never done.

WHAT EACH COLUMN ANSWERS, and from where (nothing typed in by hand):

  era          `pre_production` (run_54-74: different HV, latency 35, 32
               samples, several are HV ladders) or `production_point`
               (run_80 onward: run_79's HV and readout, only the trigger
               differs)                                    -- run_config.json
  noise        the 23-July RdClk_Div boundary (CLAUDE.md): `quiet` <= run_67,
               `ambiguous` run_68, `noisy` >= run_69
  a_x_conn8    chamber A x-view connector 8 dead 22 July -> 27 July access;
               live again from run_83 (CLAUDE.md)
  resist_*/drift_*  configured HV per sub-run          -- run_config.json
  ntof_beam    n_TOF-destined PS pulses inside the sub-run, from the
               per-minute `beam_class` log (minute granularity: a pulse in the
               minute that straddles an edge is counted)  -- slow_control
  ntof_beam_interior  the same, counting only minutes at least EDGE_S from
               either end of the sub-run.  beam_state = none / edge / interior.
               `edge` sub-runs are usable after an event-level veto around the
               pulse times; `interior` ones need that veto to be trusted.
  beam_minutes minutes of the sub-run the log classes `on`
  ntof_runs    n_TOF runs recording during the sub-run, with the overlap in
               minutes                                   -- x17-match.json
  ntof_quiet_min  of those minutes, how many were in an n_TOF run with no
               protons (bytes per bunch < QUIET_BYTES_PER_BUNCH): n_TOF taking
               PS-timed triggers with nothing on target, i.e. a random-trigger
               view of the scintillators during our cosmic running
                                     -- ntof_processing completed ledger

The DREAM products (decoded_root, hits_root, combined) come from the frozen
EOS survey (`x17-runs.json`).  A spot listing of EOS on 2026-10-02 agreed: all
47 runs decoded except run_148, which is empty.

    .venv/bin/python ntof_cosmics/inventory.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SITE_DATA = Path.home() / 'PycharmProjects' / 'dylan-cern-site' / 'data'
RUNS_DIR = Path('/media/dylan/data/x17/beam_july/runs')
BEAM_LOG = Path('/media/dylan/data/x17/beam_july/slow_control/beam_intensity')
NTOF_LEDGER = (REPO / 'ntof_processing' / 'campaign_qa' / 'results'
               / 'completed_ledger_2026-08-12b.csv')
NTOF_LEDGER_GB = (REPO / 'ntof_processing' / 'campaign_qa' / 'results'
                  / 'ledger_2026-08-11.csv')
OUT = HERE / 'results'

#: A beam n_TOF run costs ~10 MB per bunch, a no-proton one ~50 kB.  Anything
#: under this is "recording, no protons".  The gap is two decades wide.
QUIET_BYTES_PER_BUNCH = 0.5e6

#: The production operating point (run_79): resist A/B/C/D, drift all four.
PROD_RESIST = (540, 540, 525, 520)
PROD_DRIFT = (700, 700, 700, 700)
#: How far from a sub-run's start or end a pulse must be to count as "inside"
#: the cosmic running rather than the beam-loss / beam-return transition.
EDGE_S = 120


def noise_condition(run: int) -> str:
    if run <= 67:
        return 'quiet'
    if run == 68:
        return 'ambiguous'
    return 'noisy'


def a_x_conn8_dead(run: int, t_start: float) -> bool:
    """Chamber A x connector 8 was disconnected 22 July -> 27 July access."""
    t22 = dt.datetime(2026, 7, 22).timestamp()
    return run < 83 and t_start >= t22


def load_beam_class() -> pd.DataFrame:
    files = sorted(BEAM_LOG.glob('beam_class_*.csv'))
    if not files:
        raise FileNotFoundError(
            f'no beam_class logs in {BEAM_LOG} -- scp them from '
            f'/eos/experiment/ntof/data/x17/july_beam/slow_control/beam_intensity')
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_ntof_quiet() -> dict[int, str]:
    """{n_TOF run: 'quiet' | 'beam' | 'unknown'} from bytes per bunch."""
    gb = {}
    for r in csv.DictReader(NTOF_LEDGER_GB.open()):
        try:
            gb[int(r['run'])] = float(r['off_GB'] or 'nan') * 1e9
        except ValueError:
            pass
    out = {}
    for r in csv.DictReader(NTOF_LEDGER.open()):
        run = int(r['run'])
        try:
            bunches = int(r['run_last'])
        except ValueError:
            out[run] = 'unknown'
            continue
        size = float(r['merged_bytes'])
        if size <= 0:
            size = gb.get(run, float('nan'))
        if not bunches or size != size:
            out[run] = 'unknown'
        else:
            out[run] = 'quiet' if size / bunches < QUIET_BYTES_PER_BUNCH else 'beam'
    return out


def load_match() -> dict[tuple[int, str], list[tuple[int, float]]]:
    d = json.loads((SITE_DATA / 'x17-match.json').read_text())
    out = defaultdict(list)
    for s in d['segs']:
        m = re.fullmatch(r'run_(\d+)', s['d'])
        if m:
            if s.get('n') is not None:
                out[(int(m.group(1)), s['s'])].append(
                    (int(s['n']), float(s.get('min') or 0)))
    return out


def subrun_hv(cfg: dict) -> dict[str, tuple]:
    out = {}
    for s in cfg.get('sub_runs', []):
        h = s.get('hvs', {})
        res = tuple(h.get('5', {}).get(str(c)) for c in (1, 2, 3, 4))
        dri = tuple(h.get('9', {}).get(str(c)) for c in (0, 1, 2, 3))
        out[s['sub_run_name']] = (res, dri)
    return out


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    survey = json.loads((SITE_DATA / 'x17-runs.json').read_text())
    cosmic = [r for r in survey['runs'] if r['mode'] == 'cosmics']
    bc = load_beam_class()
    quiet = load_ntof_quiet()
    match = load_match()

    rows = []
    for r in cosmic:
        n = r['n']
        cfg = json.loads((RUNS_DIR / f'run_{n}' / 'run_config.json').read_text())
        dq = cfg['dream_daq_info']
        hv = subrun_hv(cfg)
        era = 'production_point' if n >= 80 else 'pre_production'
        for sr in r['sr']:
            name, t0, secs, ev, raw, dec, hits, comb = sr[:8]
            t0 = t0 or r['t']
            t1 = t0 + (secs or 0)
            b = bc[(bc.unix_ts >= t0 - 59) & (bc.unix_ts < t1)]
            # a minute bin entirely inside the sub-run, at least EDGE_S from
            # either end: beam there is beam during the cosmic running, not the
            # ramp the auto-substitution leaves at a run boundary
            inner = (b.unix_ts >= t0 + EDGE_S) & (b.unix_ts + 60 <= t1 - EDGE_S)
            res, dri = hv.get(name, ((None,) * 4, (None,) * 4))
            ov = match.get((n, name), [])
            rows.append(dict(
                run=n, subrun=name, era=era,
                start=dt.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M'),
                t_start=t0, seconds=secs, events=ev,
                rate_hz=(ev / secs) if ev and secs else None,
                decoded=dec, hits=hits, combined=comb,
                latency=dq.get('latency'),
                n_samples=dq.get('n_samples_per_waveform'),
                zero_suppress=dq.get('zero_suppress'),
                resist_A=res[0], resist_B=res[1], resist_C=res[2], resist_D=res[3],
                drift_A=dri[0], drift_B=dri[1], drift_C=dri[2], drift_D=dri[3],
                at_prod_hv=(res == PROD_RESIST and dri == PROD_DRIFT),
                noise=noise_condition(n),
                a_x_conn8_dead=a_x_conn8_dead(n, t0),
                ntof_beam=int(b.ntof_beam.sum()),
                ntof_beam_interior=int(b.ntof_beam[inner].sum()),
                beam_minutes=int((b.beam_class == 'on').sum()),
                ntof_runs=' '.join(f'{nt}:{m:g}' for nt, m in ov),
                ntof_min=round(sum(m for _, m in ov), 1),
                ntof_quiet_min=round(sum(m for nt, m in ov
                                         if quiet.get(nt) == 'quiet'), 1),
                ntof_beam_min=round(sum(m for nt, m in ov
                                        if quiet.get(nt) == 'beam'), 1),
            ))
    sub = pd.DataFrame(rows)
    sub['clean_beam_off'] = sub.ntof_beam == 0
    sub['beam_state'] = 'none'
    sub.loc[sub.ntof_beam > 0, 'beam_state'] = 'edge'
    sub.loc[sub.ntof_beam_interior > 0, 'beam_state'] = 'interior'

    def runrow(g: pd.DataFrame) -> pd.Series:
        first = g.iloc[0]
        return pd.Series(dict(
            era=first.era, start=first.start,
            subruns=len(g), hours=round(g.seconds.sum() / 3600, 2),
            events=int(g.events.fillna(0).sum()),
            rate_hz=round(g.events.fillna(0).sum() / max(g.seconds.sum(), 1), 1),
            decoded_subruns=int((g.decoded > 0).sum()),
            hv=('production' if g.at_prod_hv.all() else
                'mixed' if g.at_prod_hv.any() else 'other'),
            hv_points=g[['resist_A', 'resist_B', 'resist_C', 'resist_D',
                         'drift_A']].drop_duplicates().shape[0],
            latency=first.latency, n_samples=first.n_samples,
            noise=first.noise, a_x_conn8_dead=bool(g.a_x_conn8_dead.any()),
            ntof_beam=int(g.ntof_beam.sum()),
            ntof_beam_interior=int(g.ntof_beam_interior.sum()),
            edge_subruns=int((g.beam_state == 'edge').sum()),
            interior_subruns=int((g.beam_state == 'interior').sum()),
            usable_hours=round(g.loc[g.beam_state != 'interior',
                                     'seconds'].sum() / 3600, 2),
            usable_events=int(g.loc[g.beam_state != 'interior',
                                    'events'].fillna(0).sum()),
            clean_subruns=int(g.clean_beam_off.sum()),
            clean_hours=round(g.loc[g.clean_beam_off, 'seconds'].sum() / 3600, 2),
            clean_events=int(g.loc[g.clean_beam_off, 'events'].fillna(0).sum()),
            ntof_quiet_h=round(g.ntof_quiet_min.sum() / 60, 2),
            ntof_beam_h=round(g.ntof_beam_min.sum() / 60, 2),
        ))

    run = sub.groupby('run', sort=True).apply(runrow, include_groups=False)
    run['category'] = run.apply(categorise, axis=1)
    return sub, run.reset_index()


def categorise(r: pd.Series) -> str:
    if r.events == 0:
        return 'empty'
    if r.era == 'pre_production':
        return 'pre-production (HV scan/ladder)' if r.hv_points > 1 \
            else 'pre-production (fixed HV)'
    if r.ntof_beam == 0:
        return 'production point, clean'
    if r.ntof_beam_interior == 0:
        return 'production point, beam at the edges only'
    return 'production point, beam inside the running'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sub, run = build()
    sub.to_csv(a.out / 'cosmic_subruns.csv', index=False)
    run.to_csv(a.out / 'cosmic_runs.csv', index=False)

    pd.set_option('display.width', 250)
    print(run[['run', 'start', 'category', 'subruns', 'hours', 'events',
               'rate_hz', 'noise', 'a_x_conn8_dead', 'ntof_beam',
               'ntof_beam_interior', 'clean_hours', 'usable_hours',
               'ntof_quiet_h']].to_string(index=False))
    print('\nby category:')
    print(run.groupby('category')[['subruns', 'hours', 'events', 'clean_hours',
                                   'usable_hours', 'usable_events',
                                   'ntof_quiet_h']].sum().to_string())
    print(f'\nwrote {a.out}/cosmic_subruns.csv ({len(sub)} rows), '
          f'cosmic_runs.csv ({len(run)} rows)')


if __name__ == '__main__':
    main()
