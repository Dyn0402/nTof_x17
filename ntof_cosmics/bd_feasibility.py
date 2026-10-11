#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bd_feasibility.py -- can chambers B and D get what A and C got: an angle truth
from beam-off cosmics, and (for B) a characterisation as the hit-mode chamber
it is?  Uses only products already on disk (the run_149 full-pass tables under
results/tracking/, and run_149 combined hits under ~/scratch/ntof_insitu/beam
for occupancy, which is QA -- hits never set a position here).

Geometry (run_149 config): the four chambers form a square around the
vertical beam, centres at 234.6 mm -- A at +z, C at -z (normal z), B at -x,
D at +x (normal x).  Two consequences:

  * opposing pairs (A-C, B-D) give a straight line through both chambers'
    track points with a 469 mm lever: the in-situ truth A and C were
    calibrated on.  For D the only opposing partner is B, which has no
    drift field and so no angle -- but the line needs only B's POSITION;
  * perpendicular pairs (A-D, A-B, C-B, C-D): a straight line crossing two
    chambers at 90 deg obeys tan_1 * tan_2 = 1 in the horizontal view, so
    both chambers see |tan| ~ 1 -- outside the 0.6 the reconstruction gates
    on.  They test large angles only (the TAN_MAX question), not D's core.

Steps (outputs to results/bd_feasibility/):

    occupancy   hit-level: per arm, events with >= 3 hits in both planes,
                hits per plane, cluster extent, time span (combined hits, QA)
    lines       opposing pairs from the full-pass tracks: n, the joined-line
                slope j against the measured raw slope, per arm; A-C is the
                control that must reproduce the known A/C numbers
    crossing    the efficiency of the far chamber for straight cosmics that the
                near chamber's track says crossed it (A->C control, D->B):
                reco level on all 87 sub-runs, hit level (>= 3 hits in both
                planes, accidental baseline subtracted) where hits are local.
                D's line uses D's own cosmic wall scale (cosmic_wall_scale.py
                --arm D), A's the A-C line value
    report      report.html (+ the D wall summary from cosmic_wall_scale)
    all
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

OUT = HERE / 'results' / 'bd_feasibility'
TRACKS = HERE / 'results' / 'tracking' / 'k_run_150'
HITS = Path.home() / 'scratch' / 'ntof_insitu' / 'beam' / 'run_149'
FEU_ARM = {1: ('D', 'x'), 2: ('D', 'y'), 3: ('A', 'x'), 4: ('A', 'y'),
           5: ('B', 'x'), 6: ('B', 'y'), 7: ('C', 'x'), 8: ('C', 'y')}
#: opposing pair -> (normal axis, in-plane axes), as cosmic_tracks.slope_check
PAIRS = {('A', 'C'): ('z', ('x', 'y')), ('B', 'D'): ('x', ('z', 'y'))}
COLS = ['subrun', 'event_id', 'arm', 'gated', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z', 'k_arm',
        'tan_raw_x', 'tan_raw_y', 'x_n_strips', 'y_n_strips', 'x_q_sum', 'y_q_sum', 'x_t0',
        'x_quality_ok', 'y_quality_ok', 'x_plausible', 'y_plausible']
J_BINS = [0.1, 0.2, 0.3, 0.45, 0.6]
CENTRE = {'A': np.array([-16.4, 0, 234.6]), 'B': np.array([-234.1, 0, -15.8]),
          'C': np.array([17.3, 0, -234.6]), 'D': np.array([234.1, 0, 15.5])}   # run_149 config
HALF_MM = 180.0                         # inside the active area, with a margin
WALL = Path('/media/dylan/data/x17/ntof_cosmics/cosmic_wall_scale')
#: (near, far, normal axis, horizontal in-plane axis, near arm's true/raw scale source)
CROSS = (('A', 'C', 'z', 'x', 'A-C line (§7)'), ('D', 'B', 'x', 'z', 'D cosmic wall'))
HIT_FEU = {'A': (3, 4), 'B': (5, 6), 'C': (7, 8), 'D': (1, 2)}


# --------------------------------------------------------------------------- #
def occupancy(max_files: int = 3) -> pd.DataFrame:
    import uproot
    rows = []
    for sub in sorted(p.name for p in HITS.iterdir() if p.is_dir()):
        for f in sorted(glob.glob(str(HITS / sub / 'combined_hits_root' / '*.root')))[:max_files]:
            h = uproot.open(f)['hits'].arrays(['eventId', 'channel', 'sample', 'amplitude', 'feu'], library='pd')
            h['arm'] = h.feu.map(lambda x: FEU_ARM[x][0])
            h['pl'] = h.feu.map(lambda x: FEU_ARM[x][1])
            g = h.groupby(['eventId', 'arm', 'pl']).agg(n=('channel', 'size'), lo=('channel', 'min'),
                                                       hi=('channel', 'max'), s0=('sample', 'min'),
                                                       s1=('sample', 'max'), amp=('amplitude', 'median')).reset_index()
            lit = g[g.n >= 3]
            both = lit.groupby(['eventId', 'arm']).pl.nunique()
            n_ev = int(h.eventId.nunique())
            for (arm, pl), q in lit.groupby(['arm', 'pl']):
                rows.append(dict(subrun=sub, file=Path(f).name, arm=arm, plane=pl, n_events=n_ev,
                                 lit_both=int((both.xs(arm, level='arm') == 2).sum()),
                                 hits_med=float(q.n.median()), extent_med=float((q.hi - q.lo).median()),
                                 span_med=float((q.s1 - q.s0).median()), amp_med=float(q.amp.median())))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'occupancy_files.csv', index=False)
    S = R.groupby(['arm', 'plane']).agg(files=('file', 'nunique'), n_events=('n_events', 'sum'),
                                        lit_both=('lit_both', 'sum'), hits_med=('hits_med', 'median'),
                                        extent_med=('extent_med', 'median'), span_med=('span_med', 'median'),
                                        amp_med=('amp_med', 'median')).reset_index()
    S['lit_both_per_event'] = S.lit_both / S.n_events
    S.to_csv(OUT / 'occupancy.csv', index=False)
    print(S.round(3).to_string(index=False))
    return S


# --------------------------------------------------------------------------- #
def _tracks() -> pd.DataFrame:
    fs = sorted(glob.glob(str(TRACKS / 'tracks_run_149_*.parquet')))
    return pd.concat((pd.read_parquet(f, columns=COLS) for f in fs), ignore_index=True)


def lines() -> pd.DataFrame:
    T = _tracks()
    n_by_arm = T.groupby('arm').agg(rows=('gated', 'size'), gated=('gated', 'sum')).reset_index()
    n_by_arm.to_csv(OUT / 'track_rows.csv', index=False)
    print(n_by_arm.to_string(index=False), '\n')
    key = ['subrun', 'event_id']
    single = T.groupby(key + ['arm']).arm.transform('size') == 1
    out, samp = [], []
    for (a1, a2), (nrm, inpl) in PAIRS.items():
        # the "partner" end only contributes a position: for B-D that is B, any
        # reconstructed row (B has no angle, so its gate is meaningless);
        # the measured end must be gated.  A-C (control): both gated, as §7.
        for meas, part in ((a2, a1), (a1, a2)):
            M = T[single & (T.arm == meas) & T.gated]
            Pm = T[single & (T.arm == part) & (T.gated if part in 'AC' else True)]
            m = M.merge(Pm, on=key, suffixes=('', '_p'))
            J = {ax: (m[f'p0_{ax}'] - m[f'p0_{ax}_p']).to_numpy() for ax in 'xyz'}
            for ax in inpl:
                j = J[ax] / J[nrm]
                raw = m[f'tan_raw_{"x" if ax != "y" else "y"}'].to_numpy()
                s = m[f'd_{ax}'].to_numpy() / m[f'd_{nrm}'].to_numpy()
                k = m.k_arm.to_numpy()
                raw_g = s / k                      # raw tan in the global sign convention
                view = 'y' if ax == 'y' else 'x'
                ok = np.isfinite(j) & np.isfinite(raw_g)
                samp.append(pd.DataFrame(dict(pair=f'{a1}{a2}', meas=meas, view=view, j=j[ok], raw=raw_g[ok],
                                              subrun=m.subrun.to_numpy()[ok])))
                for lo, hi in zip(J_BINS[:-1], J_BINS[1:]):
                    b = ok & (np.abs(j) >= lo) & (np.abs(j) < hi)
                    r = j[b] / raw_g[b]
                    corr = np.corrcoef(j[ok & (np.abs(j) < 0.6)], raw_g[ok & (np.abs(j) < 0.6)])[0, 1] if ok.sum() > 10 else np.nan
                    out.append(dict(pair=f'{a1}{a2}', meas=meas, partner=part, view=view, lo=lo, hi=hi,
                                    n=int(b.sum()), true_over_raw=float(np.median(r)) if b.any() else np.nan,
                                    sign_ok=float((np.sign(j[b]) == np.sign(raw_g[b])).mean()) if b.any() else np.nan,
                                    corr_all=float(corr), n_pairs=int(len(m))))
    R = pd.DataFrame(out)
    R.to_csv(OUT / 'lines.csv', index=False)
    pd.concat(samp, ignore_index=True).to_parquet(OUT / 'line_samples.parquet', index=False)
    with pd.option_context('display.width', 200):
        print(R.round(3).to_string(index=False))
    return R


def _scale(near: str) -> float:
    if near == 'A':
        return 1.11
    return float(json.loads((WALL / f'arm_{near}' / 'summary.json').read_text())['s_best'])


def _pointing(T, near, far, nrm, uax):
    """near-chamber single gated tracks whose straight line (true/raw scale
    applied) crosses the far chamber's plane inside its active area."""
    key = ['subrun', 'tag', 'event_id']
    single = T.groupby(key + ['arm']).arm.transform('size') == 1
    X = T[single & (T.arm == near) & T.gated].copy()
    f = _scale(near) / X.k_arm.median()
    c = CENTRE[far]
    dn = c[{'x': 0, 'z': 2}[nrm]] - X[f'p0_{nrm}']
    su, sy = X[f'd_{uax}'] / X[f'd_{nrm}'] * f, X.d_y / X[f'd_{nrm}'] * f
    pu, py = X[f'p0_{uax}'] + su * dn, X.p0_y + sy * dn
    inn = ((pu - c[{'x': 0, 'z': 2}[uax]]).abs() < HALF_MM) & (py.abs() < HALF_MM) & (su.abs() < 0.6) & (sy.abs() < 0.6)
    return X[inn]


def _lit(sub: str, arm: str) -> tuple:
    """(events with >= 3 hits in both of arm's planes, all events with any hit), per tag."""
    import re
    import uproot
    lit, allev = [], []
    for f in sorted(glob.glob(str(HITS / sub / 'combined_hits_root' / '*.root'))):
        tag = re.findall(r'_(\d{3})_feu', f)[0]
        h = uproot.open(f)['hits'].arrays(['eventId', 'feu'], library='pd')
        allev.append(pd.DataFrame(dict(subrun=sub, tag=tag, event_id=h.eventId.unique())))
        h = h[h.feu.isin(HIT_FEU[arm])]
        g = h.groupby(['eventId', 'feu']).size().unstack(fill_value=0)
        a, b = HIT_FEU[arm]
        ok = g.index[(g.get(a, 0) >= 3) & (g.get(b, 0) >= 3)] if len(g) else []
        lit.append(pd.DataFrame(dict(subrun=sub, tag=tag, event_id=ok)))
    return pd.concat(lit), pd.concat(allev)


def crossing() -> pd.DataFrame:
    fs = sorted(glob.glob(str(TRACKS / 'tracks_run_149_*.parquet')))
    T = pd.concat((pd.read_parquet(f, columns=COLS + ['tag']) for f in fs), ignore_index=True)
    T['tag'] = T.tag.astype(str).str[-3:]
    key = ['subrun', 'tag', 'event_id']
    hit_subs = sorted(p.name for p in HITS.iterdir() if p.is_dir())
    rows = []
    for near, far, nrm, uax, src in CROSS:
        X = _pointing(T, near, far, nrm, uax)
        anyrow = X.merge(T[T.arm == far][key].drop_duplicates(), on=key, how='left', indicator=True)._merge == 'both'
        gat = X.merge(T[(T.arm == far) & T.gated][key].drop_duplicates(), on=key, how='left', indicator=True)._merge == 'both'
        Xh = X[X.subrun.isin(hit_subs)]
        L, E = zip(*[_lit(s, far) for s in hit_subs])
        L, E = pd.concat(L).assign(lit=True), pd.concat(E)
        m = Xh.merge(L, on=key, how='left')
        p_lit = float(m.lit.fillna(False).mean())
        base = float(len(L) / len(E))                   # far chamber lit on a random trigger
        rows.append(dict(near=near, far=far, scale=_scale(near), scale_source=src, n_pointing=int(len(X)),
                         far_any_row=float(anyrow.mean()), far_gated=float(gat.mean()),
                         n_pointing_hits=int(len(Xh)), far_lit=p_lit, lit_baseline=base,
                         far_hit_eff=(p_lit - base) / (1 - base)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'crossing.csv', index=False)
    print(R.round(3).to_string(index=False))
    return R


def report() -> Path:
    C = pd.read_csv(OUT / 'crossing.csv').set_index('far')
    Oc = pd.read_csv(OUT / 'occupancy.csv')
    Ln = pd.read_csv(OUT / 'lines.csv')
    Wd = json.loads((WALL / 'arm_D' / 'summary.json').read_text())
    Wa = json.loads((WALL / 'summary.json').read_text())
    Sd = pd.read_csv(WALL / 'arm_D' / 'scales.csv')
    eff = pd.read_csv(Path('/media/dylan/data/x17/sept26_prelim/efficiency_campaign/headline_per_run.csv'))
    eff = eff[eff.condition == 'post_access_27jul'].groupby(['arm', 'basis']).efficiency.median().reset_index()
    vD = 42.6 / Wd['s_best']
    bd = Ln[(Ln.pair == 'BD') & (Ln.meas == 'D')]
    ac = Ln[(Ln.pair == 'AC') & (Ln.meas == 'A')]

    def tab(df, fmt=None):
        return df.to_html(index=False, float_format=lambda v: f'{v:.3f}', border=0, classes='t')
    occ = Oc.pivot_table(index='arm', columns='plane', values=['lit_both_per_event', 'hits_med', 'extent_med', 'amp_med'])
    occ.columns = [f'{a} {b}' for a, b in occ.columns]
    sd_tab = Sd[['selection', 'sample', 'n', 's', 's_err']]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>B and D feasibility</title>
<style>body{{font-family:'IBM Plex Sans',Helvetica,Arial,sans-serif;max-width:1050px;margin:32px auto;padding:0 16px;color:#1c2230;background:#f5f3ee;line-height:1.45}}
h1{{font-size:30px}} h2{{font-size:22px;margin-top:34px}} .v{{border-left:6px solid #2f8a5b;padding:6px 16px;background:#fffdf9}}
.w{{border-left:6px solid #c99318;padding:6px 16px;background:#fffdf9}} table.t{{border-collapse:collapse;font-size:14px;margin:8px 0}}
table.t td,table.t th{{padding:4px 10px;border-bottom:1px solid #d8d3c8;text-align:right}} code{{font-size:13px}}</style></head><body>
<h1>Chambers B and D: can they get an in-situ angle calibration?</h1>
<p>run_149 beam-off cosmics (87 sub-runs; hit-level checks on the 14 sub-runs whose combined hits are local), {dt.date.today().isoformat()}. Generated by <code>ntof_cosmics/bd_feasibility.py report</code>.</p>
<div class="v"><p><b>D: yes, in x, from its own SiPM wall.</b> Cosmics at D's wall read true tan = <b>{Wd['s_best']:.3f}</b> × production raw (sub-run bootstrap {Wd['boot_lo']:.2f}–{Wd['boot_hi']:.2f}, {Wd['n_single_x']:,} single x-plane tracks). The same estimator gave A {Wa['s_best']:.2f}, which the A–C line confirmed (1.11). D's implied geometric drift speed is ≈ {vD:.1f} µm/ns, far below its bench bundle's 36.6, so D's error is about twice the v substitution: D needs its own in-situ v, as A and C did. D's y view has no truth from the wall.</p>
<p><b>B: not as D's partner on cosmics.</b> B lights (≥ 3 hits in both planes) on only <b>{100 * C.loc['B', 'far_hit_eff']:.0f} %</b> of the straight cosmics that D's track says crossed it, against <b>{100 * C.loc['C', 'far_hit_eff']:.0f} %</b> for C on A's (accidental baseline subtracted), and its reconstruction keeps a track on {100 * C.loc['B', 'far_any_row']:.1f} % (C {100 * C.loc['C', 'far_any_row']:.0f} %). On beam, scintillator-tagged, B's hit efficiency is comparable to the other chambers; on cosmics it is nearly blind. B–D lines therefore cannot give D a y-view truth at useful statistics without a dedicated B hit-mode reconstruction, and even then the crossing efficiency caps it.</p></div>

<h2>1 · D's angle scale from its own wall (cosmics)</h2>
<p>Edge likelihood <code>ntof_scint_stack.ana.fit_wall_u</code> on run_149 cosmics clock-matched to n_TOF (<code>cosmic_wall_scale.py build/ana --arm D</code>, lever {Wd['L']:.1f} mm). Profile minimum s = {Wd['s_best']:.3f}; Δ(−log L) ≈ 3 at 1.45 and 1.60. Valid on cosmics only: on beam the fit is degenerate with the boundary offsets.</p>
{tab(sd_tab)}
<p class="w">The |tan| bins are not usable: errors up to 800 and fitted edge widths σ from 0.04 to 12 mm (A's bins were irregular too, 1.5–9 mm). Use the overall value; treat the angle dependence as unmeasured. The row labelled "A-triggered" is the arm's own (D-) triggered sample. D's x plane has ~130 dead channels, one-sided (x_local +0…+57 and +150…+178 mm): the wall test sees fewer tracks on that side, which can bias an edge fit. Masked fits are a follow-up.</p>
<p>Pattern check against A: production's capsule k for D is 1.77, i.e. {1.77 / Wd['s_best']:.2f} × this truth; for A, 1.27 against 1.11 = 1.14. The capsule estimator overshoots both chambers by the same 14–16 %.</p>

<h2>2 · Opposing-pair lines: the A–C control and B–D</h2>
<p>Joined line through both chambers' track points, against the measured raw slope (no cleanliness cut). A–C reproduces the known numbers (A 1.03–1.13, C 1.33–1.61 × raw); B–D has no correlation worth the name ({bd.corr_all.iloc[0]:.2f}; A–C {ac.corr_all.iloc[0]:.2f}): most B–D "pairs" are not one particle.</p>
{tab(Ln[Ln.meas.isin(['A', 'D'])][['pair', 'meas', 'view', 'lo', 'hi', 'n', 'true_over_raw', 'sign_ok', 'corr_all']])}

<h2>3 · Does the far chamber see the crossing?</h2>
<p>Near-chamber single gated tracks, scaled to their true angle, extended to the far chamber's plane, inside its active area (±{HALF_MM:.0f} mm) at |tan| &lt; 0.6. Reco level on all sub-runs; hit level on the sub-runs with local hits, with the far chamber's lit rate on all triggers as the accidental baseline.</p>
{tab(C.reset_index()[['near', 'far', 'scale', 'n_pointing', 'far_any_row', 'far_gated', 'n_pointing_hits', 'far_lit', 'lit_baseline', 'far_hit_eff']])}

<h2>4 · Hit-level occupancy (QA only)</h2>
<p>Per arm and plane, over the local hit files: fraction of triggers with ≥ 3 hits in both planes, median hits, strip extent, sample span and amplitude of lit planes. B's cosmic clusters are 4–5 hits and ~5 strips wide (A/C 14–18 hits, 19–33 strips) at lower amplitude: with no field-shaping chain only part of the drift charge arrives. D's 90-strip extents are its hot channels.</p>
{occ.round(3).reset_index().to_html(index=False, border=0, classes='t')}
<p>Beam, scintillator-tagged (median over post-access runs, <code>efficiency_campaign/headline_per_run.csv</code>):</p>
{tab(eff)}

<h2>5 · What it would take</h2>
<ol>
<li><b>D in x (small):</b> build an in-situ D bundle with v ≈ {vD:.0f} (and the r06 det7 kernel it already has), reconstruct the clock-matched run_149 D tracks with seeder 3, and iterate the wall fit to s = 1. Needs D's waveforms (FEU 01/02) for those sub-runs from EOS. The per-run gas ratio comes from D's capsule band, as for A/C. Per-track truth (and kw per plane) is not available: one scale for x.</li>
<li><b>D in y:</b> no truth. Options: carry x's scale with a systematic (A's kw differ 3 % between views, C's 4.5 %), or the along-bar wall coordinate (corr −0.72 between bar halves; needs a slope fitted against tracks, i.e. the transfer B already rests on).</li>
<li><b>D cross-checks:</b> corner-cutting cosmics (A–D, C–D) at |tan| ≈ 1 test D (and A, C) above the 0.6 plausibility cut: the same sample answers the re-pass's TAN_MAX question.</li>
<li><b>B:</b> keep B as the hit-mode tagging chamber it is in the analysis (no angle). If B–D lines are still wanted: a hit-mode B position from the waveform charge centroid (no drift model), seeding at 3, and all 41 cosmic runs (57.6 h, ~3.4× run_149). At B's ~{100 * C.loc['B', 'far_hit_eff']:.0f} % crossing efficiency that is O({int(C.loc['B', 'n_pointing'] * C.loc['B', 'far_hit_eff'] * 3.4):,}) B–D lines campaign-wide, before any cleanliness cut.</li>
</ol>
<h2>What this does not rule out</h2>
<ul><li>A D wall scale biased by D's one-sided dead channels or hot strips (unmasked fit).</li>
<li>An angle-dependent D response: the per-bin wall fits are not stable enough to measure it.</li>
<li>B's cosmic blindness being a run_149 condition (HV, gas) rather than the chamber: B's HV for run_149 was not checked here.</li>
<li>That D's tracks used to predict B's crossings include junk: the A→C control uses the same method and gives C 60 %, so a D-side purity loss would have to be large to explain B's 11 %.</li></ul>
</body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.html').write_text(html)
    print('wrote', OUT / 'report.html')
    return OUT / 'report.html'


def main() -> int:
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('step', choices=['occupancy', 'lines', 'crossing', 'report', 'all'])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step in ('occupancy', 'all'):
        occupancy()
    if a.step in ('lines', 'all'):
        lines()
    if a.step in ('crossing', 'all'):
        crossing()
    if a.step in ('report', 'all'):
        report()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
