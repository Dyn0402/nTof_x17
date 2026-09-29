#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pair_timing.py -- why the two arms of an inter-chamber pair are not prompt with
each other, and why the difference is not flat either.

THE QUESTION (Dylan, 2026-09-16).  The n_TOF trees are on one flash-calibrated
time base and the scintillators resolve a few ns, so a pair born at the capsule
should show |t(arm1) - t(arm2)| of order 10 ns.  An uncorrelated second
particle should give a difference flat across the readout window.  The
published two-arm Delta-t (`accidental_timing.py`, `tight_coincidence.py`) is
neither: a peak at zero on a ~+-150 ns triangle with a bump near +-50 ns.

WHAT THIS MODULE MEASURES, each one separating a cause:

  1. `arm_alignment`   -- every trigger, every pair of wall groups (both bar
     ends, averaged) in two different arms.  The detector's own answer to "are
     the arms timed in": a peak position and width per arm pair, no Micromegas.
  2. `trigger_offsets` -- where a wall hit's dt_ns sits as a function of the
     TRIGGER arm vs the HIT arm.  dt_ns is referenced to an arm-agnostic DREAM
     prediction, so the DREAM trigger-path delay moves every hit of an event
     together; it cancels in t1 - t2 but not in a per-arm |t| cut.
  3. `leg_classes`     -- for the pairs the published figure plots, what the
     per-arm "loose tag" actually picked on each leg: a prompt hit, a plastic
     AFTER-PULSE (ntof_processing/pss_ringing: ~4 per large plastic pulse,
     +20 ns .. 1 us, 81 ns echo), or an accidental single.  The tag is a random
     wall-OR-plastic hit inside (-100, +60) ns, which is also why the published
     Delta-t can never be wider than ~+-160 ns.
  4. `wall_full_range` -- the same pairs, and every inter pair, timed with the
     wall alone (no after-pulses) over the full +-1000 ns slim, against the
     accidental wall density measured on all triggers.
  5. MM `t0` of both tracks -- an independent, coarse (~55 ns per plane) clock
     that exists for every pair, scintillator hit or not.

After-pulses are re-flagged here from the exported parquet, which dropped the
slim's `shadow_amp`: a plastic hit is flagged when an earlier hit on the same
channel within AP_HOLD_NS is more than AP_RATIO times larger (on `amp`, not
`amp_0`).  Checked on run_145 against the stored flag inside the tag window:
58 562 agree, 45 missed, 3 604 extra (~4 % over-flagging of clean hits).

Pre-27-July-access runs (run_79, run_81) are excluded, as in
`tight_coincidence.py`, so the pair sample is the one the published campaign
figure plots.

    python -m sept26_prelim_analysis.pair_timing            # campaign
    python -m sept26_prelim_analysis.pair_timing --runs run_145
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.scintillators import DT_WINDOW  # noqa: E402

SCHEMA = 'sept26_prelim/pair_timing/1'
ARMS = ('A', 'B', 'C', 'D')
SLIM_NS = 1000.0
#: Two bar ends belong to one crossing if they agree this well (the bar is
#: 500 mm; scintillators.DT_PHYS_MAX_NS).
END_MATCH_NS = 12.0
#: Where an event's reference (trigger) wall group is looked for: the DREAM
#: trigger-path offsets span -17 .. +8 ns, plus the peak's own width.
REF_WINDOW = (-45.0, 35.0)
#: "Prompt" = within this of the event's reference wall time.
PROMPT_NS = 25.0
AP_HOLD_NS = 1000.0
AP_RATIO = 20.0
PRE_ACCESS_RUNS = ('run_79', 'run_81')
HIST_EDGES = np.arange(-1000.0, 1000.0 + 1e-9, 2.0)


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def wall_groups(h: pd.DataFrame) -> pd.DataFrame:
    """One row per wall crossing seen at BOTH bar ends: mean time, geometric amp.

    Averaging the ends removes the propagation along the bar (+-2.5 ns over
    500 mm), which is the one position-dependent term in a wall time.
    """
    w = h[(h.family == 'WAL') & (h.is_control == 0)]
    w = w.assign(grp=(w.detn - 1) // 2, end=np.where(w.detn % 2 == 1, 1, 2))
    k = ['subrun', 'eventId', 'arm', 'grp']
    a = w[w.end == 1][k + ['dt_ns', 'amp']]
    b = w[w.end == 2][k + ['dt_ns', 'amp']]
    m = a.merge(b, on=k, suffixes=('1', '2'))
    m = m[(m.dt_ns1 - m.dt_ns2).abs() < END_MATCH_NS]
    m = m.assign(t=0.5 * (m.dt_ns1 + m.dt_ns2), a=np.sqrt(m.amp1 * m.amp2))
    # one end can match two of the other end's hits; keep the larger crossing
    m = m.sort_values('a', ascending=False).drop_duplicates(
        k + ['dt_ns1']).drop_duplicates(k + ['dt_ns2'])
    return m[k + ['t', 'a']].reset_index(drop=True)


def afterpulse_flag(p: pd.DataFrame) -> np.ndarray:
    """True for a plastic hit sitting in the after-pulse train of a bigger one.

    Vectorised look-back (the slim's own `shadow_prev` idiom): step k hits back
    along each (event, det, detn) stream while still inside AP_HOLD_NS.
    """
    o = np.lexsort((p.dt_ns.to_numpy(), p.detn.to_numpy(), p.det.to_numpy(),
                    p.eventId.to_numpy(), p.subrun.astype(str).to_numpy()))
    key = pd.factorize(p.subrun.astype(str).to_numpy()[o])[0].astype(np.int64)
    key = ((key * 10_000_000 + p.eventId.to_numpy()[o]) * 1000
           + p.det.to_numpy()[o] * 10 + p.detn.to_numpy()[o])
    t = p.dt_ns.to_numpy()[o].astype(np.float64)
    amp = p.amp.to_numpy()[o].astype(np.float64)
    pmax = np.zeros(t.size)
    active = np.arange(t.size)
    k = 1
    while active.size:
        j = active - k
        ok = j >= 0
        active, j = active[ok], j[ok]
        ok = (key[j] == key[active]) & (t[active] - t[j] <= AP_HOLD_NS)
        active, j = active[ok], j[ok]
        np.maximum.at(pmax, active, amp[j])
        k += 1
    flag = np.zeros(t.size, bool)
    flag[o] = pmax > AP_RATIO * amp
    return flag


def event_reference(g: pd.DataFrame) -> pd.DataFrame:
    """The event's trigger crossing: the largest wall group near dt_ns = 0."""
    r = g[(g.t > REF_WINDOW[0]) & (g.t < REF_WINDOW[1])]
    r = r.sort_values('a', ascending=False).drop_duplicates(
        ['subrun', 'eventId'])
    return r[['subrun', 'eventId', 'arm', 't']].rename(
        columns={'arm': 'ref_arm', 't': 't_ref'})


def loose_tag(h: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """`accidental_timing.arm_tag_time(require_both=False)`, reproduced so each
    picked hit keeps its identity (family, after-pulse flag, amplitude)."""
    s = h[h.family.isin(('WAL', 'PSS')) & (h.is_control == 0)
          & (h.dt_ns >= DT_WINDOW[0]) & (h.dt_ns <= DT_WINDOW[1])]
    s = s.sample(frac=1.0, random_state=seed)
    return s.drop_duplicates(['subrun', 'eventId', 'arm'], keep='first')


# --------------------------------------------------------------------------- #
# per run
# --------------------------------------------------------------------------- #
def _pairs(run: str, subruns, dca_max: float) -> pd.DataFrame:
    """Real inter-chamber pairs (source_imaging's sample) with both tracks' t0."""
    from sept26_prelim_analysis import source_imaging as SI
    from sept26_prelim_analysis import acceptance as AC
    t = SI._track_table(run, subruns, dca_max, extra_cols=('x_t0', 'y_t0'))
    pr = SI._pairs_real(t)
    if pr.empty:
        return pd.DataFrame()
    v = SI._vertex_frame(t, pr, False)
    a, b = t.loc[pr.i.to_numpy()], t.loc[pr.j.to_numpy()]
    t0a = 0.5 * (a.x_t0.to_numpy() + a.y_t0.to_numpy())
    t0b = 0.5 * (b.x_t0.to_numpy() + b.y_t0.to_numpy())
    swap = a.arm.to_numpy() > b.arm.to_numpy()
    v['t0_1'] = np.where(swap, t0b, t0a)
    v['t0_2'] = np.where(swap, t0a, t0b)
    v = v[v.topology == 'inter'].copy()
    if v.empty:
        return pd.DataFrame()
    ek = v.key1.str.split(':', n=1, expand=True)
    v['subrun'], v['eventId'] = ek[0], ek[1].astype(np.int64)
    v['topo'] = [AC.topology(x, y) for x, y in zip(v.arm1, v.arm2)]
    v['run'] = run
    return v.drop(columns=['key', 'key2', 'mixed']).reset_index(drop=True)


def _nearest_group(P, G, arm_col, ref_col='t_ref'):
    """Per pair, the given arm's largest wall group anywhere in +-1000 ns."""
    g = G.rename(columns={'arm': arm_col})
    m = P[['pid', 'subrun', 'eventId', arm_col]].merge(
        g, on=['subrun', 'eventId', arm_col])
    m = m.sort_values('a', ascending=False).drop_duplicates('pid')
    return m.set_index('pid')


def analyse_run(run: str, subruns, dca_max: float = 30.0) -> dict:
    from sept26_prelim_analysis.slim_export import read_export
    P = _pairs(run, subruns, dca_max)
    out = dict(run=run, n_triggers=0, align=np.zeros((6, HIST_EDGES.size - 1)),
               offs=[], base=np.zeros((4, 4, HIST_EDGES.size - 1)),
               base_n=np.zeros(4), pairs=pd.DataFrame(), legs=pd.DataFrame(),
               pss=np.zeros((2, HIST_EDGES.size - 1)))
    for sub in subruns:
        h = read_export(run, [sub])
        meta = paths.out('slim') / f'ntof_hits_{run}_{sub}.meta.json'
        out['n_triggers'] += json.load(open(meta))['n_events']
        G = wall_groups(h)
        R = event_reference(G)

        # (1) arm alignment: all wall-group combinations in different arms
        x = G.merge(G, on=['subrun', 'eventId'], suffixes=('1', '2'))
        x = x[x.arm1 < x.arm2]
        for i, (p1, p2) in enumerate([(a, b) for ai, a in enumerate(ARMS)
                                      for b in ARMS[ai + 1:]]):
            s = x[(x.arm1 == p1) & (x.arm2 == p2)]
            out['align'][i] += np.histogram(s.t1 - s.t2, HIST_EDGES)[0]

        # (2) trigger offsets: hit-arm x ref-arm medians, prompt groups only
        o = G.merge(R[['subrun', 'eventId', 'ref_arm']], on=['subrun', 'eventId'])
        o = o[(o.t > REF_WINDOW[0]) & (o.t < REF_WINDOW[1])]
        out['offs'].append(o.groupby(['ref_arm', 'arm']).t.agg(['sum', 'size']))

        # (4b) baseline wall density in each arm, relative to the reference
        b = G.merge(R, on=['subrun', 'eventId'])
        for ri, ra in enumerate(ARMS):
            out['base_n'][ri] += int((R.ref_arm == ra).sum())
            for xi, xa in enumerate(ARMS):
                s = b[(b.ref_arm == ra) & (b.arm == xa)]
                out['base'][ri, xi] += np.histogram(s.t - s.t_ref, HIST_EDGES)[0]

        # plastic spectrum, clean vs after-pulse, on single-reference events
        ps = h[(h.family == 'PSS') & (h.is_control == 0)]
        ps = ps.merge(R, on=['subrun', 'eventId'])
        ps = ps[ps.arm == ps.ref_arm]
        if len(ps):
            f = afterpulse_flag(ps)
            dtr = (ps.dt_ns - ps.t_ref).to_numpy()
            out['pss'][0] += np.histogram(dtr[~f], HIST_EDGES)[0]
            out['pss'][1] += np.histogram(dtr[f], HIST_EDGES)[0]

        Ps = P[P.subrun == sub].copy() if len(P) else P
        if Ps.empty:
            continue
        Ps['pid'] = Ps.index.to_numpy()
        ev = Ps[['subrun', 'eventId']].drop_duplicates()
        hp = h.merge(ev, on=['subrun', 'eventId'])
        Ps = Ps.merge(R, on=['subrun', 'eventId'], how='left')
        Ps.index = Ps.pid.to_numpy()
        Gp = G.merge(ev, on=['subrun', 'eventId'])

        # (4) wall, full range: largest group per arm, and prompt flags
        for k in ('1', '2'):
            n = _nearest_group(Ps, Gp, 'arm' + k)
            Ps['tw' + k] = n.t.reindex(Ps.pid).to_numpy()
            Ps['aw' + k] = n.a.reindex(Ps.pid).to_numpy()
            pr = Gp.rename(columns={'arm': 'arm' + k}).merge(
                Ps[['pid', 'subrun', 'eventId', 'arm' + k, 't_ref']],
                on=['subrun', 'eventId', 'arm' + k])
            pr = pr[(pr.t - pr.t_ref).abs() < PROMPT_NS]
            Ps['wprompt' + k] = Ps.pid.isin(pr.pid).to_numpy()
            Ps['nwall' + k] = (Gp.rename(columns={'arm': 'arm' + k})
                               .merge(Ps[['pid', 'subrun', 'eventId', 'arm' + k]],
                                      on=['subrun', 'eventId', 'arm' + k])
                               .groupby('pid').size().reindex(Ps.pid)
                               .fillna(0).astype(int).to_numpy())
        # wall groups of the NON-reference arm, relative to t_ref (the excess)
        oth = np.where(Ps.ref_arm == Ps.arm1, Ps.arm2,
                       np.where(Ps.ref_arm == Ps.arm2, Ps.arm1, None))
        Ps['other_arm'] = oth
        q = Gp.merge(Ps[['pid', 'subrun', 'eventId', 'other_arm', 't_ref']]
                     .dropna(subset=['other_arm']),
                     left_on=['subrun', 'eventId', 'arm'],
                     right_on=['subrun', 'eventId', 'other_arm'])
        q = q.assign(dt_other=q.t - q.t_ref)[['pid', 'dt_other', 'a']]
        q['run'] = run
        q['subrun'] = sub
        out.setdefault('other', []).append(q)

        # (3) loose tag legs
        tag = loose_tag(hp)
        ap = np.zeros(len(tag), bool)
        pmask = (tag.family == 'PSS').to_numpy()
        if pmask.any():
            hpp = hp[(hp.family == 'PSS') & (hp.is_control == 0)].reset_index(drop=True)
            f = afterpulse_flag(hpp)
            fl = hpp.assign(ap=f)[['subrun', 'eventId', 'det', 'detn', 'dt_ns', 'ap']]
            tg = tag.reset_index(drop=True).merge(
                fl, on=['subrun', 'eventId', 'det', 'detn', 'dt_ns'], how='left')
            ap = tg.ap.astype('boolean').fillna(False).to_numpy(bool)
            tag = tg
        else:
            tag = tag.reset_index(drop=True)
        tag['ap'] = ap
        legs = []
        for k in ('1', '2'):
            L = Ps[['pid', 'subrun', 'eventId', 'arm' + k, 't_ref', 'ref_arm']].merge(
                tag.rename(columns={'arm': 'arm' + k})[
                    ['subrun', 'eventId', 'arm' + k, 'dt_ns', 'family', 'amp', 'ap']],
                on=['subrun', 'eventId', 'arm' + k])
            L = L.rename(columns={'arm' + k: 'arm'}).assign(leg=k)
            legs.append(L)
        L = pd.concat(legs, ignore_index=True)
        L['run'] = run
        out['legs'] = pd.concat([out['legs'], L], ignore_index=True)
        out['pairs'] = pd.concat([out['pairs'], Ps.reset_index(drop=True)],
                                 ignore_index=True)
    out['other'] = (pd.concat(out['other'], ignore_index=True)
                    if out.get('other') else pd.DataFrame())
    out['offs'] = (pd.concat(out['offs']).groupby(level=[0, 1]).sum()
                   if out['offs'] else pd.DataFrame())
    return out


def classify_legs(L: pd.DataFrame) -> pd.DataFrame:
    """One class per tagged leg.  'prompt' is relative to the event's own
    reference wall crossing, so the trigger-path delay does not enter."""
    rel = L.dt_ns - L.t_ref
    prompt = rel.abs() < PROMPT_NS
    L = L.assign(rel=rel)
    L['cls'] = np.select(
        [L.t_ref.isna(),
         (L.family == 'PSS') & L.ap,
         prompt],
        ['no reference', 'plastic after-pulse', 'prompt'],
        default='accidental single')
    return L


def _runs(src) -> dict:
    rs = {}
    for p in sorted(src.glob('tracks_run_*_stat090_*.parquet')):
        m = re.match(r'tracks_(run_\d+)_(stat090_\d+)\.parquet$', p.name)
        if m and m.group(1) not in PRE_ACCESS_RUNS:
            rs.setdefault(m.group(1), []).append(m.group(2))
    return rs


def _job(args):
    run, subs, dca = args
    try:
        return analyse_run(run, subs, dca)
    except FileNotFoundError as exc:
        return dict(run=run, skipped=str(exc).splitlines()[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--runs', default='', help='comma list; default = campaign')
    ap.add_argument('--dca', type=float, default=30.0)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    rs = _runs(paths.out('stage3_fullpass'))
    if a.runs:
        rs = {r: rs[r] for r in a.runs.split(',')}
    stem = 'campaign' if not a.runs else a.runs.replace(',', '_')
    od = paths.out('pair_timing')

    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_job, [(k, v, a.dca) for k, v in rs.items()]):
            if 'skipped' in r:
                print(f"  {r['run']}: skipped -- {r['skipped']}")
                continue
            print(f"  {r['run']}: {r['n_triggers']:,} triggers, "
                  f"{len(r['pairs'])} inter pairs", flush=True)
            res.append(r)

    align = sum(r['align'] for r in res)
    base = sum(r['base'] for r in res)
    base_n = sum(r['base_n'] for r in res)
    pss = sum(r['pss'] for r in res)
    offs = pd.concat([r['offs'] for r in res if len(r['offs'])]).groupby(
        level=[0, 1]).sum()
    pairs = pd.concat([r['pairs'] for r in res], ignore_index=True)
    pairs['pid'] = pairs.run + ':' + pairs.pid.astype(str)
    legs = classify_legs(pd.concat([r['legs'] for r in res], ignore_index=True))
    legs['pid'] = legs.run + ':' + legs.pid.astype(str)
    other = pd.concat([r['other'] for r in res if len(r['other'])],
                      ignore_index=True)
    other['pid'] = other.run + ':' + other.pid.astype(str)

    names = [f'{a_}{b_}' for i, a_ in enumerate(ARMS) for b_ in ARMS[i + 1:]]
    c = 0.5 * (HIST_EDGES[:-1] + HIST_EDGES[1:])
    pd.DataFrame(align.T, columns=names).assign(dt_ns=c).to_parquet(
        od / f'align_hist_{stem}.parquet', index=False)
    np.savez_compressed(od / f'baseline_{stem}.npz', base=base, base_n=base_n,
                        pss=pss, edges=HIST_EDGES)
    (offs['sum'] / offs['size']).unstack().to_csv(
        od / f'trigger_offsets_{stem}.csv')
    pairs.to_parquet(od / f'pairs_{stem}.parquet', index=False)
    legs.to_parquet(od / f'legs_{stem}.parquet', index=False)
    other.to_parquet(od / f'other_arm_wall_{stem}.parquet', index=False)
    json.dump(dict(schema=SCHEMA, runs={k: v for k, v in rs.items()},
                   n_triggers=int(sum(r['n_triggers'] for r in res)),
                   n_pairs=int(len(pairs)), dt_window=list(DT_WINDOW),
                   prompt_ns=PROMPT_NS, ref_window=list(REF_WINDOW),
                   ap_hold_ns=AP_HOLD_NS, ap_ratio=AP_RATIO,
                   excluded=list(PRE_ACCESS_RUNS)),
              open(od / f'pair_timing_{stem}.meta.json', 'w'), indent=1)
    print(f'{len(pairs)} inter pairs; legs:\n'
          f'{legs.cls.value_counts().to_string()}\n-> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
