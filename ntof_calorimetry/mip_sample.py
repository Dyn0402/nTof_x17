#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mip_sample.py -- the minimum-ionising sample that C1.1 (plastic scale) and C4
(liquid salvage) of PLAN.md both run on: particles that certainly crossed an
arm's whole scintillator stack, with that stack read out channel by channel.

TWO SAMPLES, ONE TABLE LAYOUT.  One row per (particle, probe arm), with the
scintillator columns named exactly as `ntof_scint_stack.extract` writes them
(``w{n}_amp_on``, ``p{1,2}_amp_on``, ``l_amp_on`` ... and the same-width
``_off`` window before the trigger), so one analysis reads both.

``beam``    In-beam THROUGH-GOERS: exactly one gated track in each of two
            OPPOSITE arms (A-C or B-D), the two lines within ``SEP_MAX`` mm of
            each other and the joined line > ``JDCA_MIN`` mm from the beam axis
            (`inbeam_through_goers` selection).  The direction is the line
            through the two chambers' track points: k-independent, and good to
            ~1 mrad over the ~1 m baseline -- no angle calibration enters.  The
            scintillator columns are the scint-stack per-track tables'.  A probe
            arm's own plastic is UNBIASED by the trigger when the partner arm
            satisfied the (emulated) hardware trigger, ``partner_hw``.
``cosmic``  run_149 beam-off cosmics on the n_TOF clock (`clock_match.py`),
            single tracks per arm, slopes ``tanx``/``tany`` of the cosmic full
            pass (``k`` of run_147).  Pure cosmic muons, no flash, no beam
            backgrounds, but only the ~16 % of the time n_TOF records.  The
            n_TOF WAL/PSS/LIQ trees are read around the matched singles time;
            each tree's window is centred on its own measured peak (raw tof is
            not on a common zero across trees, see `clock_match`).

    python -m ntof_calorimetry.mip_sample beam
    python -m ntof_calorimetry.mip_sample cosmic
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_cosmics import clock_match as CM  # noqa: E402
from ntof_cosmics import cosmic_wall_scale as CWS  # noqa: E402
from ntof_cosmics.cosmic_tracks import _axis_dca  # noqa: E402
from ntof_scint_stack.extract import MV_PER_ADC, hit_columns  # noqa: E402
from ntof_tracking.reco.geometry import U_HAT, V_HAT, W_HAT  # noqa: E402
from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.source_imaging import _dca_two_lines  # noqa: E402

OUT = paths.spell('calo')
CAMPAIGN = paths.spell('out', 'stage3_fullpass', 'tracks_campaign.parquet')
STACK = paths.spell('scint', 'tracks')
COSMIC_TRACKS = REPO / 'ntof_cosmics' / 'results' / 'tracking' / 'k_run_147'
PAIRS = (('A', 'C'), ('B', 'D'))
#: generous here; analyses cut tighter (sep < 20 mm is the default there)
SEP_MAX = 60.0
JDCA_MIN = 60.0
CAMP_COLS = ['run', 'subrun', 'event_id', 'arm', 'gated', 'p0_x', 'p0_y', 'p0_z',
             'd_x', 'd_y', 'd_z']
STACK_KEEP = ['run', 'subrun', 'event_id', 'arm', 'u_mm', 'v_mm', 'tan_raw_x', 'tan_raw_y',
              'n_trk', 't_since_flash_ns', 'x_t0', 'y_t0', 'q_per_len',
              'hw_A', 'hw_B', 'hw_C', 'hw_D'] + hit_columns('on') + hit_columns('off')


def slopes(J: np.ndarray, arm: str) -> tuple:
    """(du/dw, dv/dw) of direction(s) J in the arm's local frame, w outward."""
    w = J @ W_HAT[arm]
    return (J @ U_HAT[arm]) / w, (J @ V_HAT) / w


# --------------------------------------------------------------------------- #
# beam: through-goers
# --------------------------------------------------------------------------- #
def _pairs(t: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    k = ['run', 'subrun', 'event_id']
    t = t[t.arm.isin([a, b])]
    n = t.groupby(k + ['arm']).size().unstack(fill_value=0)
    ok = n[(n.get(a, 0) == 1) & (n.get(b, 0) == 1)].index
    t = t.set_index(k)
    t = t[t.index.isin(ok)].reset_index()
    A = t[t.arm == a].sort_values(k).reset_index(drop=True)
    B = t[t.arm == b].sort_values(k).reset_index(drop=True)
    P1, D1 = A[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), A[['d_x', 'd_y', 'd_z']].to_numpy(float)
    P2, D2 = B[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), B[['d_x', 'd_y', 'd_z']].to_numpy(float)
    _v, sep = _dca_two_lines(P1, D1, P2, D2)
    J = P2 - P1
    base = np.linalg.norm(J, axis=1)
    J /= base[:, None]
    out = A[k].copy()
    out['pair'] = a + b
    out['sep'] = sep
    out['baseline_mm'] = base
    out['jdca'] = _axis_dca(P1, J)[0]
    out['vert_deg'] = np.degrees(np.arccos(np.abs(J[:, 1])))
    for arm in (a, b):
        out[f'tu_{arm}'], out[f'tv_{arm}'] = slopes(J, arm)
    return out


def build_beam() -> pd.DataFrame:
    T = pq.read_table(CAMPAIGN, columns=CAMP_COLS,
                      filters=[('gated', '==', True)]).to_pandas()
    for c in ('run', 'subrun', 'arm'):
        T[c] = T[c].astype(str)
    P = pd.concat([_pairs(T, a, b) for a, b in PAIRS], ignore_index=True)
    P = P[(P.sep < SEP_MAX) & (P.jdca > JDCA_MIN)]
    print(f'{len(P):,} through-goer pairs (sep < {SEP_MAX:g}, jdca > {JDCA_MIN:g}):',
          P.pair.value_counts().to_dict(), flush=True)
    rows = []
    for f in sorted(STACK.glob('stack_run_*.parquet')):
        run = f.stem.replace('stack_', '')
        Pr = P[P.run == run]
        if not len(Pr):
            continue
        S = pd.read_parquet(f, columns=STACK_KEEP)
        for c in ('run', 'subrun', 'arm'):
            S[c] = S[c].astype(str)
        S = S[S.n_trk == 1]
        for pair in ('AC', 'BD'):
            Q = Pr[Pr.pair == pair]
            for probe, partner in ((pair[0], pair[1]), (pair[1], pair[0])):
                x = Q.merge(S[S.arm == probe], on=['run', 'subrun', 'event_id'], how='inner')
                x['partner'] = partner
                x['partner_hw'] = x[f'hw_{partner}'].to_numpy()
                x['self_hw'] = x[f'hw_{probe}'].to_numpy()
                x['tu'] = x[f'tu_{probe}']
                x['tv'] = x[f'tv_{probe}']
                rows.append(x.drop(columns=[c for c in x.columns
                                            if c[:3] in ('tu_', 'tv_')]))
        print(f'  {run}: {len(Pr)} pairs', flush=True)
    D = pd.concat(rows, ignore_index=True)
    D['sample'] = 'beam'
    D['ms'] = D.t_since_flash_ns / 1e6
    # sign check: the joined line against the probe's own raw slope
    for arm in 'ABCD':
        m = (D.arm == arm) & (D.tan_raw_x.abs() < 0.6)
        if m.sum() > 50:
            print(f'  {arm}: corr(tu, tan_raw_x) = {np.corrcoef(D.tu[m], D.tan_raw_x[m])[0, 1]:+.2f}, '
                  f'corr(tv, tan_raw_y) = {np.corrcoef(D.tv[m], D.tan_raw_y[m])[0, 1]:+.2f}, n = {m.sum()}')
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT / 'mip_beam.parquet', index=False)
    print(f'wrote {OUT / "mip_beam.parquet"} ({len(D):,} probe rows)')
    return D


# --------------------------------------------------------------------------- #
# cosmic: run_149 on the n_TOF clock
# --------------------------------------------------------------------------- #
_TREES: dict = {}
#: per tree, the on window half-width about its peak, and the off window's
#: displacement (the same width, before the trigger, as the beam stack's)
HALF_NS = 50.0
OFF_SHIFT_NS = -480.0


def _tree(ntof: int, name: str) -> dict:
    key = (ntof, name)
    if key not in _TREES:
        want = ['BunchNumber', 'detn', 'amp', 'tof', 'satuflag', 'area_0', 'pileup1']
        parts = [uproot.open(f)[name].arrays(want, library='np') for f in CM._parts(ntof)]
        d = {k: np.concatenate([p[k] for p in parts]) for k in want}
        o = np.lexsort((d['tof'], d['BunchNumber']))
        _TREES[key] = {k: v[o] for k, v in d.items()}
    return _TREES[key]


def _hits(trig: pd.DataFrame, name: str) -> pd.DataFrame:
    """Every hit of tree ``name`` within +-1 us of each matched trigger."""
    out = []
    for ntof, g in trig.groupby('ntof'):
        W = _tree(int(ntof), name)
        key = W['BunchNumber'].astype(np.float64) * 1e9 + W['tof']
        k0 = g.bunch.to_numpy(np.float64) * 1e9 + g.tof.to_numpy()
        lo, hi = np.searchsorted(key, k0 - 1000), np.searchsorted(key, k0 + 1000)
        n = hi - lo
        it = np.repeat(np.arange(len(g)), n)
        ih = np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        out.append(pd.DataFrame(dict(
            event_id=g.event_id.to_numpy()[it], trig_arm=g.ntof_arm.to_numpy()[it],
            detn=W['detn'][ih].astype(int), amp=W['amp'][ih] * MV_PER_ADC,
            area=W['area_0'][ih] * MV_PER_ADC, sat=W['satuflag'][ih],
            pu=W['pileup1'][ih], dt=W['tof'][ih] - g.tof.to_numpy()[it])))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _peak(dt: np.ndarray) -> float:
    n, e = np.histogram(dt, bins=1000, range=(-1000, 1000))
    k = n.argmax()
    c = 0.5 * (e[k] + e[k + 1])
    return float(np.median(dt[np.abs(dt - c) < 20]))


def _wide(h: pd.DataFrame, pre: dict, win: str, centre: pd.Series) -> pd.DataFrame:
    """Largest hit per (event, channel) inside the window -> stack-style columns."""
    sh = 0.0 if win == 'on' else OFF_SHIFT_NS
    x = h[(h.dt - centre - sh).abs() <= HALF_NS].copy()
    x['ch'] = x.detn.map(pre)
    x = x.dropna(subset=['ch']).sort_values('amp', ascending=False)
    x = x.drop_duplicates(['event_id', 'ch'])
    x['dtc'] = x.dt - centre.loc[x.index]
    parts = []
    for short, col in (('amp', 'amp'), ('dt', 'dtc'), ('sat', 'sat'), ('area', 'area'), ('pu', 'pu')):
        p = x.pivot_table(index='event_id', columns='ch', values=col, aggfunc='first')
        p.columns = [f'{c}_{short}_{win}' for c in p.columns]
        parts.append(p)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


def build_cosmic() -> pd.DataFrame:
    subs = sorted({f.name.split('run_149_')[1].rsplit('_', 1)[0]
                   for f in CM.OUT.glob('pairs_run_149_*.csv')})
    rows = []
    for sub in subs:
        trig = CWS.matched_triggers(sub)
        if not len(trig):
            continue
        t = pd.read_parquet(COSMIC_TRACKS / f'tracks_run_149_{sub}.parquet')
        t = t[t.gated.astype('boolean').fillna(False)].copy()
        t['n_trk'] = t.groupby(['event_id', 'arm']).event_id.transform('size')
        t = t.merge(trig[['event_id', 'ntof_arm', 'ntof', 'bunch', 'tof', 'res']],
                    on='event_id', how='inner')
        for arm in 'ABCD':
            ta = t[t.arm == arm]
            if not len(ta):
                continue
            ev = trig[trig.event_id.isin(ta.event_id)]
            H = {fam: _hits(ev, f'{fam}{arm}') for fam in ('WAL', 'PSS', 'LIQ')}
            rows.append(dict(sub=sub, arm=arm, t=ta, H=H))
        print(f'{sub}: {len(trig)} matched, {t.event_id.nunique()} with a gated track', flush=True)
    # per (arm, tree, triggering arm) peak: from the self-triggered events,
    # and for other triggering arms the wall's peak + (tree - wall) on self
    out = []
    for arm in 'ABCD':
        R = [r for r in rows if r['arm'] == arm]
        if not R:
            continue
        H = {fam: pd.concat([r['H'][fam].assign(sub=r['sub']) for r in R], ignore_index=True)
             for fam in ('WAL', 'PSS', 'LIQ')}
        pk = {fam: {} for fam in H}
        for fam, h in H.items():
            for ta, g in h.groupby('trig_arm'):
                pk[fam][ta] = _peak(g.dt.to_numpy())
        rel = {fam: pk[fam].get(arm, np.nan) - pk['WAL'].get(arm, np.nan) for fam in H}
        print(f'{arm}: peaks (ns) on own triggers WAL {pk["WAL"].get(arm, np.nan):.1f}, '
              f'PSS - WAL {rel["PSS"]:+.1f}, LIQ - WAL {rel["LIQ"]:+.1f}; '
              f'WAL by trigger arm {({k: round(v, 1) for k, v in pk["WAL"].items()})}')
        pre = {'WAL': {n: f'w{n}' for n in range(1, 9)}, 'PSS': {1: 'p1', 2: 'p2'},
               'LIQ': {1: 'l'}}
        for r in R:
            t = r['t'].copy()
            W = []
            for fam, h in r['H'].items():
                if not len(h):
                    continue
                c = h.trig_arm.map(lambda a, fam=fam: pk['WAL'].get(a, np.nan) + rel[fam])
                for win in ('on', 'off'):
                    W.append(_wide(h, pre[fam], win, c))
            W = pd.concat(W, axis=1) if W else pd.DataFrame(index=pd.Index([], name='event_id'))
            t = t.merge(W.reset_index(), on='event_id', how='left')
            for c in hit_columns('on') + hit_columns('off'):
                if c not in t.columns:
                    t[c] = np.nan
            t['subrun'] = r['sub']
            out.append(t)
    D = pd.concat(out, ignore_index=True)
    from sept26_prelim_analysis.det_a_scint import layer_geometry
    u0 = {a: layer_geometry('run_149', a)['u_mm'] for a in 'ABCD'}
    D['u_mm'] = D.x_local + D.arm.map(u0)
    D['v_mm'] = D.y_local
    D['tu'], D['tv'] = D.tanx, D.tany
    D['self_hw'] = D.ntof_arm == D.arm
    D['partner_hw'] = D.ntof_arm != D.arm      # another arm triggered: unbiased
    D['sample'] = 'cosmic'
    D['ms'] = np.nan
    keep = ['run', 'subrun', 'event_id', 'arm', 'n_trk', 'u_mm', 'v_mm', 'tu', 'tv',
            'tan_raw_x', 'tan_raw_y', 'k_arm', 'x_t0', 'y_t0', 'q_per_len', 'ntof_arm',
            'self_hw', 'partner_hw', 'sample', 'ms'] + hit_columns('on') + hit_columns('off')
    D = D[keep]
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT / 'mip_cosmic.parquet', index=False)
    print(f'wrote {OUT / "mip_cosmic.parquet"} ({len(D):,} tracks):',
          D[D.n_trk == 1].arm.value_counts().to_dict())
    return D


# --------------------------------------------------------------------------- #
# crossings: where the particle went through each layer
# --------------------------------------------------------------------------- #
def crossings(D: pd.DataFrame) -> pd.DataFrame:
    """Per row: the crossing of each layer, the plastic bar it is in and how
    far from that bar's edges, whether it is on the liquid, and what fired.

    The direction is ``tu, tv`` (the joined line for through-goers, the
    calibrated cosmic slope otherwise).  Layer levers, plastic positions and
    the wall/plastic alignment offsets in u are the scint-stack's
    (`geometry.csv`, `ana/meta.json`: ``off_w``, ``off_p``); the liquid has no
    measured offset.  Columns are named as `ntof_scint_stack.ana.predict`'s.
    """
    import json
    from ntof_scint_stack import ana as SA
    cal = json.loads((paths.spell('scint', 'ana', 'meta.json')).read_text())['cal']
    out = []
    for arm, t in D.groupby('arm'):
        g, c = SA.geometry(arm), cal[arm]
        P = pd.DataFrame(index=t.index)
        u, v, tu, tv = (t[k].to_numpy(float) for k in ('u_mm', 'v_mm', 'tu', 'tv'))
        P['u_w'] = u + g['L_wall'] * tu - c['off_w']
        P['v_w'] = v + g['L_wall'] * tv
        P['u_p'] = u + g['L_plas'] * tu - c['off_p']
        P['v_p'] = v + g['L_plas'] * tv
        P['u_l'] = u + g['L_ls'] * tu - g['u_ls']       # liquid-centred
        P['v_l'] = v + g['L_ls'] * tv - g['v_ls']
        P['cos'] = 1 / np.sqrt(1 + tu ** 2 + tv ** 2)
        uw = P.u_w.to_numpy()
        grp = np.digitize(uw, SA.WALL_EDGES) - 1
        grp[(uw < SA.WALL_EDGES[0]) | (uw >= SA.WALL_EDGES[-1]) | ~np.isfinite(uw)] = -1
        grp[np.abs(P.v_w.to_numpy()) > SA.WALL_HALF_V] = -1
        P['grp'] = grp
        P['d_edge_w'] = np.min(np.abs(uw[:, None] - SA.WALL_EDGES[None, :]), 1)
        up, vp = P.u_p.to_numpy(), P.v_p.to_numpy()
        on_p = (up >= g['plas_lo']) & (up <= g['plas_hi']) & (np.abs(vp) <= SA.PLAS_HALF_V)
        P['bar'] = np.where(on_p, np.where(up < g['plas_gap'], 1, 2), -1)
        P['d_gap_p'] = np.abs(up - g['plas_gap'])
        P['d_edge_p'] = np.minimum.reduce([up - g['plas_lo'], g['plas_hi'] - up,
                                           SA.PLAS_HALF_V - np.abs(vp)])
        ul, vl = P.u_l.to_numpy(), P.v_l.to_numpy()
        P['on_l'] = (np.abs(ul) <= SA.LS_HALF_U) & (np.abs(vl) <= SA.LS_HALF_V)
        P['d_edge_l'] = np.minimum(SA.LS_HALF_U - np.abs(ul), SA.LS_HALF_V - np.abs(vl))
        for w in ('on', 'off'):
            e1 = np.zeros(len(t), bool)
            e2 = np.zeros(len(t), bool)
            for gg in range(4):
                m = grp == gg
                e1 |= m & t[f'w{2 * gg + 1}_amp_{w}'].notna().to_numpy()
                e2 |= m & t[f'w{2 * gg + 2}_amp_{w}'].notna().to_numpy()
            P[f'wboth_{w}'], P[f'wany_{w}'] = e1 & e2, e1 | e2
            f1 = t[f'p1_amp_{w}'].notna().to_numpy()
            f2 = t[f'p2_amp_{w}'].notna().to_numpy()
            P[f'pm_{w}'] = np.where(P.bar == 1, f1, np.where(P.bar == 2, f2, False))
            P[f'po_{w}'] = np.where(P.bar == 1, f2, np.where(P.bar == 2, f1, False))
            P[f'lf_{w}'] = t[f'l_amp_{w}'].notna().to_numpy()
            P[f'pa_{w}'] = np.where(P.bar == 1, t[f'p1_amp_{w}'], np.where(P.bar == 2, t[f'p2_amp_{w}'], np.nan))
            P[f'po_a_{w}'] = np.where(P.bar == 1, t[f'p2_amp_{w}'], np.where(P.bar == 2, t[f'p1_amp_{w}'], np.nan))
        P['psat'] = np.nan_to_num(np.where(P.bar == 1, t.p1_sat_on, t.p2_sat_on)) > 0
        P['la'], P['larea'] = t.l_amp_on.to_numpy(), t.l_area_on.to_numpy()
        P['lsat'] = np.nan_to_num(t.l_sat_on.to_numpy()) > 0
        P['l_dt'] = t.l_dt_on.to_numpy()
        P['pdt'] = np.where(P.bar == 1, t.p1_dt_on, t.p2_dt_on)
        out.append(P)
    return D.join(pd.concat(out))


#: cosmic arms with no k (B): slope = K_FILL x raw, the A/C cosmic scale
#: (A-C line 1.11, A's own wall 1.15); flagged ``k_fill``
K_FILL = 1.1


def load(sample: str) -> pd.DataFrame:
    D = pd.read_parquet(OUT / f'mip_{sample}.parquet').reset_index(drop=True)
    D['k_fill'] = D.tu.isna() & D.tan_raw_x.notna()
    D.loc[D.k_fill, 'tu'] = K_FILL * D.tan_raw_x[D.k_fill]
    D.loc[D.k_fill, 'tv'] = K_FILL * D.tan_raw_y[D.k_fill]
    return crossings(D)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('sample', choices=['beam', 'cosmic'])
    a = ap.parse_args()
    (build_beam if a.sample == 'beam' else build_cosmic)()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
