#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hot_seed_strata.py -- label every trigger by how much of its cluster is hot.

The one thing you need before you can read any hot-wildcard comparison on
chamber D. D's 42 flagged x channels carry **55.6 %** of the plane's hits, so
the frozen (unmasked) sample is not one population that a mask thins evenly --
it is two populations that the mask separates, and every aggregate number over
it is a mixture whose composition changes when you turn the mask on. Compare
the aggregates and the mask looks like a catastrophe; compare within a stratum
and it is measurable.

The stratum is the hot content of the **baseline** cluster -- the one the
frozen, unmasked seeder would have picked -- because that is defined for every
event in both tables and does not depend on the setting under test:

    all-hot     the largest baseline cluster has ZERO clean strips
    mostly-hot  it has some, but under half
    clean       at least half its strips are clean

What that split shows on D/run_145, in the FROZEN (unmasked) products
(measured 2026-09-08, tag 000, x plane, 6 690 triggers):

    stratum        n   chi2/dof   quality_ok   p0 median   p0 IQR
    all-hot     1948       1.27       99.6 %       65 mm   142 mm
    mostly-hot  1853      20.2        93.4 %      137 mm   308 mm
    clean       2339      22.5        94.6 %      257 mm   228 mm
    no cluster   550        --         0.0 %          --       --

**The noise fits better than the physics** -- 1.27 against 22, and it is the
only stratum at 99.6 % quality_ok. A coherent-noise column on a bad connector
is a smooth, wide, dilute deposit and the forward model explains it almost
perfectly, while a real track has to be fitted against real diffusion and real
sharing. And it is not spread over the plane: its p0 piles up at 65 mm with an
IQR of 142, which is the u in [50, 70) mm band of HANDOFF_D_NOISY_CHANNELS.md
Sec. 2.2, while the clean stratum sits at 257 mm across the whole plane.

So the all-hot stratum -- 29 % of D's triggers, on channels running 35x their
neighbours' occupancy -- was holding the frozen table's median chi2/dof down
at 10.7 and its convergence rate up at 72.6 %. Removing it *must* make both
look worse. That is the whole explanation of the "regression" in
HANDOFF_HOT_WILDCARD_TUNING.md Sec. 3, and it is why that handoff's success
criterion "the number of gated tracks in D goes UP" cannot be met by any
correct mask.

    python -m sept26_prelim_analysis.hot_seed_strata --arm all --run run_145

Writes ``<out>/hot_strata/hot_strata_<run>_<arm>.parquet``.

**This is also the production hot-channel cut.** :func:`dropped_events` is the
one definition of which triggers the analysis throws away for being noise, and
``k_arm.coincident_tracks`` applies it by default -- so the strata have to be
built before ``k_arm`` runs, which is the order in ``rerun_chain.sh``.
``k_robustness`` deliberately asks for the UNCUT sample so its ``baseline``
stays a true baseline and its ``no_hotstrip`` row stays a measurement of what
the cut does.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import sys

import numpy as np
import pandas as pd

REPO = __file__.rsplit('/sept26_prelim_analysis/', 1)[0]
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ntof_tracking import wft_beam as wb        # noqa: E402
from wft import io as wio                        # noqa: E402
from wft import seed as wseed                    # noqa: E402

from . import paths                              # noqa: E402

# --------------------------------------------------------------------------- #
# The production cut -- one definition, used by k_arm and measured by
# k_robustness's `no_hotstrip` variant.  Do not re-spell either of these
# anywhere else.
# --------------------------------------------------------------------------- #
#: Strata a trigger is thrown away for. Only the extreme one: a cluster with
#: ZERO clean strips is not a track, it is the noise column itself.
#: ``mostly-hot`` is deliberately NOT dropped -- it still contains real tracks
#: (fitted p0 spread across the whole plane, IQR 308 mm), and dropping it costs
#: another 22 % of D for a rounding change in the angle scale.
DROP_STRATA = ('all-hot',)
#: Which plane's stratum condemns a trigger. D's fault is an x-plane connector
#: fault, and x alone keeps the cut an exact no-op on A, B and C. Measured
#: alternatives on D: y alone removes 3.8 % and buys nothing (spread 0.102,
#: reproducibility unchanged at 0.059); x and y together remove 7.2 % for
#: spread 0.086 against 0.087 -- twice the sample for a rounding difference,
#: and it stops being a no-op on A.
DROP_PLANES = ('x',)


def strata_path(run: str, arm: str):
    return paths.out('hot_strata') / f'hot_strata_{run}_{arm}.parquet'


@lru_cache(maxsize=None)
def strata_available(run: str, arm: str) -> bool:
    """Has the strata table been built for this (run, arm)?

    Callers must distinguish this from ``dropped_events`` coming back empty.
    Both give "0 triggers dropped", and they mean opposite things:

        available, 0 dropped   -- the chamber is clean (A, B and C are)
        NOT available          -- the cut did not run at all

    The classification is per run condition (CLAUDE.md), so a run whose
    ``noisy_channels``/``hot_seed_strata`` steps have not been run gets no cut.
    That is legal, and it is exactly the kind of thing that reads later as
    "chamber D was clean in that run". ``rerun_chain.sh`` builds the strata
    ahead of ``k_arm``; a chain that does not must either add them or accept an
    uncut sample knowingly.
    """
    return strata_path(run, arm).exists()


@lru_cache(maxsize=None)
def warn_if_unavailable(run: str, arm: str) -> bool:
    """Say once, per (run, arm), that the cut is not being applied."""
    if strata_available(run, arm):
        return True
    print(f'  [hot] {run}/{arm}: no strata table -- the hot-channel cut is NOT '
          f'applied. Build it with "python -m sept26_prelim_analysis.'
          f'hot_seed_strata --arm all --run {run}" (needs noisy_channels first).')
    return False


@lru_cache(maxsize=None)
def dropped_events(run: str, arm: str) -> dict:
    """``{subrun: set(event_id)}`` -- the triggers the analysis throws away.

    Keyed by sub-run because ``event_id`` restarts at each one: a flat set
    would mask the wrong triggers in every sub-run but the first.

    Returns ``{}`` when the table has not been built for this arm. That is a
    legal state (a run whose strata have not been made yet) but it is not a
    silent one -- every caller reports how many triggers it actually dropped,
    so "0" is visible rather than inferred.

    Cached: ``k_arm`` asks once per (arm, sub-run) and the table is one file
    per arm covering all of them. Callers only read it -- do not mutate the
    returned dict, it is shared.
    """
    p = strata_path(run, arm)
    if not p.exists():
        return {}
    cols = [f'{pl}_stratum' for pl in DROP_PLANES]
    d = pd.read_parquet(p, columns=['event_id', 'subrun'] + cols)
    sel = np.zeros(len(d), bool)
    for c in cols:
        sel |= d[c].isin(DROP_STRATA).to_numpy()
    return {s: set(g.event_id.tolist()) for s, g in d[sel].groupby('subrun')}


def hot_channels(run: str, arm: str) -> dict:
    """``{'x': array, 'y': array}`` of hot channels, from the classifier's CSV."""
    p = paths.require(paths.out('noisy_channels')
                      / f'noisy_channels_{run}.csv', 'noisy_channels table')
    d = pd.read_csv(p)
    d = d[(d.arm == arm) & (d.cls == 'hot')]
    return {pl: d[d.plane == pl].channel.to_numpy(dtype=int) for pl in ('x', 'y')}


def baseline_top_cluster(pos: np.ndarray, channels: np.ndarray,
                         hot_set: np.ndarray, min_strips: int) -> tuple:
    """(total, clean) of the cluster the UNMASKED seeder would pick.

    Deliberately reimplemented in six lines rather than called through
    ``wft.seed``: this must keep reporting the pre-mask baseline whatever the
    seeder's policy becomes, or the stratum stops being a fixed reference.
    """
    good = np.isfinite(pos)
    pos, channels = pos[good], channels[good]
    if len(pos) < min_strips:
        return -1, -1
    o = np.argsort(pos)
    pos, channels = pos[o], channels[o]
    lab = wseed._cluster_labels(pos, wseed.GAP_THRESHOLD_MM)
    hot_mask = (np.isin(channels, hot_set) if len(hot_set)
                else np.zeros(len(channels), bool))
    best = (-1, -1)
    for c in range(int(lab.max()) + 1):
        m = lab == c
        tot = int(m.sum())
        if tot >= min_strips and tot > best[0]:
            best = (tot, int((m & ~hot_mask).sum()))
    return best


def stratum(total: int, clean: int) -> str:
    if total < 0:
        return 'no cluster'
    if clean == 0:
        return 'all-hot'
    return 'mostly-hot' if clean < 0.5 * total else 'clean'


def build(run: str, subrun: str, arm: str, tags=None) -> pd.DataFrame:
    """One sub-run's strata. ``event_id`` is unique only WITHIN a sub-run --
    the id restarts, so every consumer must join on (subrun, event_id) and the
    column is written here rather than left to be remembered."""
    hot = hot_channels(run, arm)
    cfg = wb.beam_config(arm, run, subrun)
    all_tags = wb.subrun_tags(cfg)
    tags = all_tags[:tags] if tags else all_tags
    pos_maps = wio.strip_position_map(cfg)
    fx, fy = cfg.MX17_FEU_X, cfg.MX17_FEU_Y
    rows = []
    for tag in tags:
        path = wb.hits_file_for_tag(cfg, tag)
        if not path:
            continue
        df = wb.read_hits_tag(path, (fx, fy))
        df = wseed.apply_significance_floor(df, wseed.SIG_REL_FLOOR)
        for eid, g in df.groupby('eventId', sort=False):
            rec = {'event_id': int(eid), 'subrun': subrun, 'tag': tag}
            for plane, feu in (('x', fx), ('y', fy)):
                gp = g[g['feu'] == feu]
                tot = cln = -1
                if 0 < len(gp) <= wb.BUSY_PLANE_HITS:
                    ch = gp['channel'].to_numpy(dtype=int)
                    tot, cln = baseline_top_cluster(
                        pos_maps[feu][ch], ch, hot.get(plane, np.array([], int)),
                        wb.MIN_STRIPS_BEAM)
                rec[f'{plane}_base_total'] = tot
                rec[f'{plane}_base_clean'] = cln
                rec[f'{plane}_stratum'] = stratum(tot, cln)
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--arm', default='D', help='or "all"')
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns', nargs='+', default=None,
                    help='default: every sub-run of the run present on disk')
    ap.add_argument('--tags', type=int, default=None,
                    help='only the first N file tags (default: all)')
    a = ap.parse_args()

    subruns = a.subruns or sorted(
        p.name for p in (paths.root('runs') / a.run).iterdir() if p.is_dir())
    arms = ('A', 'B', 'C', 'D') if a.arm == 'all' else (a.arm,)
    for arm in arms:
        d = pd.concat([build(a.run, s, arm, a.tags) for s in subruns],
                      ignore_index=True)
        out = paths.out('hot_strata') / f'hot_strata_{a.run}_{arm}.parquet'
        d.to_parquet(out, index=False)
        print(f'{arm}: {len(d):,} triggers over {len(subruns)} sub-run(s) -> {out}')
        for plane in ('x', 'y'):
            print(f'  {plane}: ' + '  '.join(
                f'{k} {v:,}' for k, v in
                d[f'{plane}_stratum'].value_counts().items()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
