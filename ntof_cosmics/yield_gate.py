#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yield_gate.py -- pre-launch check 2 of the is2_v1 re-pass (HANDOFF_TRACKING
§14): does the in-situ reco hold its yield and purity on run periods other
than the smoke sub-run (run_145)?

The smoke test only compared condor with the local is2 reco.  Here is2 is
compared with PRODUCTION on a few tags of sub-runs from other periods, A and
C, in three chains:

    prod     the full pass itself (reco_fullpass, v 42.6, seeder min 5),
             re-built with today's stage-3 code and the k production applies
             (stage3_campaign: run_145's k_arm on every run)
    is2      in-situ bundles is2_A / is2_C, seeder min 3, TAN_MAX 0.6 raw
             (exactly the staged package), k = kcal_is2_v1
    is2w     the same with TAN_MAX widened to --wide raw (default 1.0), to
             see what the raw-tan plausibility cut costs on each period

Per (run, arm, chain): gated tracks, scintillator-confirmed tracks (wall
match minus the same-width off-time control, det_a_scint.match_run),
near-normal yield, late-t0 fraction, and the true-angle reach of the cut
(TAN_MAX x k for that run).

    python ntof_cosmics/yield_gate.py reco  [--runs run_86 run_110 run_156]
    python ntof_cosmics/yield_gate.py build
    python ntof_cosmics/yield_gate.py summary     # + cut scan + report.html

Work dir: ~/scratch/ntof_insitu/yieldgate (reco products, ~GB); tables go
to results/yield_gate/.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import seed_beam_test as SBT                                    # noqa: E402

WORK = Path.home() / 'scratch' / 'ntof_insitu' / 'yieldgate'
OUT = HERE / 'results' / 'yield_gate'
BUNDLES = {a: str(Path.home() / 'scratch' / 'ntof_insitu' / 'bundles' / f'is2_{a}') for a in 'AC'}
STAGE3_CAMPAIGN = Path('/media/dylan/data/x17/sept26_prelim/stage3_campaign')
KCAL = Path('/media/dylan/data/x17/sept26_prelim/kcal_is2_v1')
SUB = 'stat090_0000'
RUNS = ('run_86', 'run_110', 'run_156')
TAGS = ('000', '003')
ARMS = ('A', 'C')
TAN_MAX = 0.6
NEAR_NORMAL = 0.08


def _tags(run):
    return SBT.tags_of(run, SUB, 'A', TAGS)


def reco(runs, wide: float, jobs: int, chains):
    from wft import reco as wreco
    for run in runs:
        for chain in chains:
            tm = TAN_MAX if chain == 'is2' else wide
            # the pool forks, so the workers inherit the patched module global
            orig = wreco.TAN_MAX
            wreco.TAN_MAX = tm
            try:
                print(f'== {run} {chain} (TAN_MAX {tm} raw)', flush=True)
                SBT.reco(WORK, run, SUB, 3, ARMS, list(TAGS), jobs, BUNDLES, chain)
            finally:
                wreco.TAN_MAX = orig


def _prod_view(run) -> Path:
    """The production reco restricted to the same tags (symlinks)."""
    d = WORK / 'prod' / run / SUB
    for arm in ARMS:
        src = SBT.prod_dir(run, SUB, arm)
        dst = d / f'mx17_{arm}'
        dst.mkdir(parents=True, exist_ok=True)
        for tag in _tags(run):
            for f in src.glob(f'events_{tag}.*'):
                if not (dst / f.name).exists():
                    (dst / f.name).symlink_to(f)
    return d


def _k(run, chain) -> dict:
    if chain == 'prod':
        # what production DELIVERS: stage3_campaign applies run_145's k to every
        # run (stage3_fullpass's per-run k is missing for A/C on e.g. run_156)
        meta = json.loads((STAGE3_CAMPAIGN / f'tracks_{run}_{SUB}.meta.json').read_text())
        return {a: v for a, v in meta['k_arm']['applied'].items() if a in ARMS}
    return {a: v for a, v in json.loads((KCAL / f'k_arm_{run}.json').read_text())['apply'].items()}


def build(runs, chains):
    from sept26_prelim_analysis import build_tracks as BT
    from sept26_prelim_analysis import det_a_scint as DS
    meta = lambda run: json.loads((SBT.STAGE3 / f'tracks_{run}_{SUB}.meta.json').read_text())  # noqa: E731
    for run in runs:
        for chain in ('prod',) + tuple(chains):
            rdir = _prod_view(run) if chain == 'prod' else WORK / chain / run / SUB
            odir = WORK / chain / run / SUB / 'tracks'
            odir.mkdir(parents=True, exist_ok=True)
            k = _k(run, chain)
            print(f'== build {run} {chain} k={k}', flush=True)
            # B/D are absent from the restricted dirs: k only for A and C
            BT.build(run, SUB, rdir, stage1=Path(meta(run)['stage1']), allow=None,
                     out_dir=odir, k_arm=k)
            for arm in ARMS:
                f = odir / f'scint{arm}.parquet'
                M, _ = DS.match_run(run, [SUB], odir, None, DS.layer_geometry(run, arm))
                M.to_parquet(f, index=False)


def summary(runs, chains):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        for chain in ('prod',) + tuple(chains):
            odir = WORK / chain / run / SUB / 'tracks'
            T = pd.read_parquet(odir / f'tracks_{run}_{SUB}.parquet')
            k = _k(run, chain)
            for arm in ARMS:
                g = T[(T.arm == arm)]
                G = g[g.gated]
                base = g.x_quality_ok & g.y_quality_ok
                r = np.maximum(g.x_tan_theta.abs(), g.y_tan_theta.abs())
                tm = TAN_MAX if chain in ('prod', 'is2') else WIDE
                S = pd.read_parquet(odir / f'scint{arm}.parquet')
                w = S[S.on_wall]
                nn = (G.tan_raw_x.abs() < NEAR_NORMAL) & (G.tan_raw_y.abs() < NEAR_NORMAL)
                tt = np.maximum(G.tanx.abs(), G.tany.abs())
                rows.append(dict(
                    run=run, arm=arm, chain=chain, k=k[arm], tan_max_raw=tm,
                    true_reach=tm * k[arm], gated=len(G),
                    confirmed=int(w.match_wall.sum() - w.match_wall_ctrl.sum()),
                    wall_rate=float(w.match_wall.mean()), wall_ctrl=float(w.match_wall_ctrl.mean()),
                    near_normal=int(nn.sum()), late_frac=float((np.maximum(G.x_t0, G.y_t0) > 300).mean()),
                    true_tan_gt_0p58=int((tt > 0.58).sum()),
                    quality_ok=int(base.sum()),
                    fail_tan_frac=float((base & (r >= tm)).sum() / max(base.sum(), 1))))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'yield_gate.csv', index=False)
    P = R[R.chain == 'prod'].set_index(['run', 'arm'])
    R['gated_vs_prod'] = R.gated / R.set_index(['run', 'arm']).index.map(P.gated)
    R['confirmed_vs_prod'] = R.confirmed / R.set_index(['run', 'arm']).index.map(P.confirmed)
    R.to_csv(OUT / 'yield_gate.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(R.round(3).to_string(index=False))
    return R


WIDE = 1.0
CUT_SCAN = (0.6, 0.7, 0.8, 0.9, 1.0)


def scan(runs):
    """Post-hoc raw-tan cut on the is2w tracks: gated and scintillator-
    confirmed tracks kept below each cut.  Approximate: a real cut also
    re-ranks candidates (an implausible best candidate yields to the next),
    which this ignores -- compare the 0.6 row with the is2 chain to see how
    much that matters."""
    rows = []
    for run in runs:
        odir = WORK / 'is2w' / run / SUB / 'tracks'
        k = _k(run, 'is2w')
        for arm in ARMS:
            S = pd.read_parquet(odir / f'scint{arm}.parquet')
            raw = np.maximum(S.tanx.abs(), S.tany.abs()) / k[arm]
            for c in CUT_SCAN:
                T = S[raw < c]
                w = T[T.on_wall]
                rows.append(dict(run=run, arm=arm, cut_raw=c, cut_true=c * k[arm], gated=len(T),
                                 confirmed=int(w.match_wall.sum() - w.match_wall_ctrl.sum()),
                                 wall_rate=float(w.match_wall.mean())))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'cut_scan.csv', index=False)
    with pd.option_context('display.width', 200):
        print(R.pivot_table(index=['run', 'arm'], columns='cut_raw', values=['gated', 'confirmed']))
    return R


def report() -> Path:
    """results/yield_gate/report.html: checks 1 (steep synthetic muons) and 2
    (yield gate) before the is2_v1 re-pass, from the tables above and
    g4_digi/steep_check.py's steep_muons.csv."""
    import datetime as dt
    Y = pd.read_csv(OUT / 'yield_gate.csv')
    Cs = pd.read_csv(OUT / 'cut_scan.csv')
    St = pd.read_csv(HERE / 'results' / 'repass_readiness' / 'steep_muons.csv')

    def tab(df):
        return df.to_html(index=False, float_format=lambda v: f'{v:.3f}', border=0, classes='t', na_rep='')

    tot = Y.groupby(['arm', 'chain'])[['gated', 'confirmed', 'near_normal']].sum()
    rel = lambda arm, ch, c: tot.loc[(arm, ch), c] / tot.loc[(arm, 'prod'), c]  # noqa: E731
    sc = Cs.groupby(['arm', 'cut_raw'])[['gated', 'confirmed']].sum().reset_index()
    sc['purity'] = sc.confirmed / sc.gated
    sc['added_confirmed_frac'] = (sc.groupby('arm').confirmed.diff() / sc.groupby('arm').gated.diff())
    base = sc[sc.cut_raw == 0.6].set_index('arm')
    top = sc[sc.cut_raw == 1.0].set_index('arm')
    st = St[St.variant != 'tm1.2_ws042'].copy()
    st_tab = st.pivot_table(index=['arm', 'lo'], columns='variant',
                            values=['eff_gated', 'wrong_sign', 'ratio_right', 'sigma']).round(3)
    st_tab.columns = [f'{a} {b}' for a, b in st_tab.columns]
    ws = St[St.lo >= 0.3].pivot_table(index='arm', columns='variant', values='wrong_sign', aggfunc='mean')
    yt = Y[['run', 'arm', 'chain', 'k', 'true_reach', 'gated', 'confirmed', 'wall_rate', 'near_normal',
            'late_frac', 'fail_tan_frac', 'gated_vs_prod', 'confirmed_vs_prod']]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>is2_v1 pre-launch checks</title>
<style>body{{font-family:'IBM Plex Sans',Helvetica,Arial,sans-serif;max-width:1050px;margin:32px auto;padding:0 16px;color:#1c2230;background:#f5f3ee;line-height:1.45}}
h1{{font-size:30px}} h2{{font-size:22px;margin-top:34px}} .v{{border-left:6px solid #2f8a5b;padding:6px 16px;background:#fffdf9}}
.w{{border-left:6px solid #c99318;padding:6px 16px;background:#fffdf9}} table.t{{border-collapse:collapse;font-size:14px;margin:8px 0}}
table.t td,table.t th{{padding:4px 10px;border-bottom:1px solid #d8d3c8;text-align:right}} code{{font-size:13px}}</style></head><body>
<h1>is2_v1 re-pass: pre-launch checks 1 and 2</h1>
<p>{dt.date.today().isoformat()}. Generated by <code>ntof_cosmics/yield_gate.py summary</code> (check 2) and
<code>ntof_cosmics/g4_digi/steep_check.py</code> (check 1). Context: <code>HANDOFF_TRACKING_2026-10-06.md</code> §14–15.</p>
<div class="v">
<p><b>Check 2 · no per-period surprise.</b> On two tags each of run_86 (27 Jul), run_110 (31 Jul) and run_156 (8 Aug), the
staged is2 reco gives <b>A {100 * (rel('A', 'is2', 'confirmed') - 1):+.0f} %</b> and <b>C {100 * (rel('C', 'is2', 'confirmed') - 1):+.0f} %</b>
scintillator-confirmed tracks over production (per run A {', '.join(f'{100 * (v - 1):+.0f}' for v in Y[(Y.chain == 'is2') & (Y.arm == 'A')].confirmed_vs_prod)} %,
C {', '.join(f'{100 * (v - 1):+.0f}' for v in Y[(Y.chain == 'is2') & (Y.arm == 'C')].confirmed_vs_prod)} %; run_145 smoke A +44, C +37 %). C keeps
{100 * rel('C', 'is2', 'gated'):.0f} % of production's gated tracks, so its purity rises. Near-normal yield ×{rel('A', 'is2', 'near_normal'):.1f} (A),
×{rel('C', 'is2', 'near_normal'):.0f} (C). The staged cut's true-angle reach is only {Y[(Y.chain == 'is2') & (Y.arm == 'C')].true_reach.min():.2f}–{Y[(Y.chain == 'is2') & (Y.arm == 'C')].true_reach.max():.2f} on C
(per-run k 0.84–0.86) and {Y[(Y.chain == 'is2') & (Y.arm == 'A')].true_reach.min():.2f}–{Y[(Y.chain == 'is2') & (Y.arm == 'A')].true_reach.max():.2f} on A.</p>
<p><b>Widening TAN_MAX to 1.0 raw buys little confirmed yield.</b> Confirmed tracks rise A {100 * (top.loc['A', 'confirmed'] / base.loc['A', 'confirmed'] - 1):+.0f} %, C {100 * (top.loc['C', 'confirmed'] / base.loc['C', 'confirmed'] - 1):+.0f} %, while gated tracks rise
A {100 * (top.loc['A', 'gated'] / base.loc['A', 'gated'] - 1):+.0f} %, C {100 * (top.loc['C', 'gated'] / base.loc['C', 'gated'] - 1):+.0f} %. Only 9–20 % of the added tracks are wall-confirmed, against
{base.loc['A', 'purity']:.2f} (A) / {base.loc['C', 'purity']:.2f} (C) for those below 0.6.</p>
<p><b>Check 1 · the reco measures steep tracks when it gets the sign right, but mirror fits are common.</b> Synthetic MIPs at
|tan| 0–1.1 through the production code under is2: right-sign fits read raw/true 0.93–0.98 (A, to |tan| 0.9) and 0.96–1.0 (C, to 1.1).
But {100 * ws.loc['A', 'tm1.2']:.0f} % (A) and {100 * ws.loc['C', 'tm1.2']:.0f} % (C) of fits at |tan| ≥ 0.3 have the <i>wrong sign</i>, reading 1.2–4× too steep with half the charge.
These are the same strips and the only candidate, so not a candidate-choice error. Doubling the start scan (<code>W_SCAN_HALF</code> 0.021 → 0.042 mm/ns) makes them slightly more common:
they are not a basin the start scan misses. Whether χ² genuinely prefers them was not measured directly. With TAN_MAX 0.6, C's gated efficiency is ≈ 0 above true 0.7 and A's surviving fits there are biased low (0.89 → 0.67).</p>
</div>
<div class="w"><p><b>Recommendation (Dylan's decision).</b> Make the re-pass's stage-2 cut wide (1.0 raw) and apply the angular acceptance at stage 3 as a
true-angle cut. Stage 3 is cheap to redo and a stage-2 rejection is not. The candidate re-ranking a wide cut causes costs &lt; 1 % of confirmed tracks: a post-hoc 0.6 cut on the wide chain
gives {int(Cs[(Cs.cut_raw == 0.6) & (Cs.arm == 'A')].confirmed.sum()):,} / {int(Cs[(Cs.cut_raw == 0.6) & (Cs.arm == 'C')].confirmed.sum()):,} confirmed (A / C) against the real 0.6 chain's
{int(Y[(Y.chain == 'is2') & (Y.arm == 'A')].confirmed.sum()):,} / {int(Y[(Y.chain == 'is2') & (Y.arm == 'C')].confirmed.sum()):,}. Until the mirror fits are understood, set that stage-3
acceptance near 0.6 true. The mirrors are a separate stage-2 problem, and they bias opening angles inside the accepted range too. One candidate cause, untested: the unregularised NNLS depth profile (O10), which lets a fit explain part of the charge with a steep line.</p></div>

<h2>1 · Yield gate per period (check 2)</h2>
<p>prod = the full pass (v 42.6, seeder 5) rebuilt with today's stage 3 and the k production applies (run_145's on every run);
is2 = the staged package (is2 bundles, seeder 3, TAN_MAX 0.6 raw, k = kcal_is2_v1); is2w = the same with TAN_MAX 1.0 raw.
"confirmed" = SiPM-wall matches minus the same-width off-time control, over tracks whose extrapolation lands on the wall
(<code>det_a_scint.match_run</code>). Tags 000 and 003 of stat090_0000.</p>
{tab(yt)}
<h2>2 · Raw-tan cut scan on the wide chain (all three periods)</h2>
<p>Post-hoc |raw tan| &lt; cut on is2w gated tracks (both views). <i>added_confirmed_frac</i> = confirmed tracks gained per gated track gained from the previous row.</p>
{tab(sc)}
<h2>3 · Steep synthetic muons (check 1)</h2>
<p><code>run_digi.py muons --tan-u 0 1.1</code>: straight MIPs, |tan_u| uniform 0–1.1 with random sign, tan_v ±0.3, mesh position uniform ±150 mm, digitised into
quiet run_145 overlay triggers, 1500 per arm, same events for every variant. tm0.6 = as staged; tm1.2 = the cut out of the way.
<i>ratio_right</i> = median raw/true over right-sign fits; <i>sigma</i> = MAD of (raw − true) over all fits.</p>
{tab(st_tab.reset_index())}
<p>Wide start scan (<code>W_SCAN_HALF</code> 0.042), wrong-sign fraction at |tan| ≥ 0.3, averaged over bins:
A {ws.loc['A', 'tm1.2_ws042']:.3f} (vs {ws.loc['A', 'tm1.2']:.3f}), C {ws.loc['C', 'tm1.2_ws042']:.3f} (vs {ws.loc['C', 'tm1.2']:.3f}).</p>
<h2>What these checks do not rule out</h2>
<ul>
<li>That the 0.6–1.0 tracks are real but fail the wall test: steep tracks extrapolate with larger errors, and a mirrored angle sends the extrapolation to the wrong tile.
The wall test cannot separate junk from mis-measured real tracks.</li>
<li>That the mirror rate in data equals the synthetic one. The synthetic muons share the bundle's own signal model. The A rate (18–23 % at |tan| &gt; 0.35) was seen in data before; C's has not been checked against data.</li>
<li>Periods not sampled: run_79 (A connector 8 dead, must be masked) and the run_120s. Two tags per run is ~1/4 of a sub-run.</li>
<li>y-view and per-track effects of the change in candidate ranking beyond the confirmed-count comparison.</li>
<li>The pilot (check 3, one sub-run per run on condor) was not run: it should use whichever cut is chosen.</li>
</ul>
</body></html>"""
    (OUT / 'report.html').write_text(html)
    print('wrote', OUT / 'report.html')
    return OUT / 'report.html'


def main() -> int:
    global WIDE
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('step', choices=['reco', 'build', 'summary', 'report'])
    ap.add_argument('--runs', nargs='+', default=list(RUNS))
    ap.add_argument('--chains', nargs='+', default=['is2', 'is2w'])
    ap.add_argument('--wide', type=float, default=WIDE)
    ap.add_argument('--jobs', type=int, default=15)
    a = ap.parse_args()
    WIDE = a.wide
    if a.step == 'reco':
        reco(a.runs, a.wide, a.jobs, a.chains)
    elif a.step == 'report':
        report()
    elif a.step == 'build':
        build(a.runs, a.chains)
    else:
        summary(a.runs, a.chains)
        scan(a.runs)
        report()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
