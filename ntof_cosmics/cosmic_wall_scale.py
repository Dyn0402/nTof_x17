#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmic_wall_scale.py -- the SiPM-wall angle scale on run_149 COSMICS: the
decisive test of HANDOFF_TRACKING_2026-10-06.md §10c, item 3.

THE QUESTION.  On beam, A's wall edges say true = 0.89 x production raw tan
(`wall_edge_scale.py`); on cosmics the A-C line says 1.11.  Either the beam
particles really reconstruct differently (electrons vs muons), or the A-C
truth / the wall lever arms are off.  Put cosmics on the same wall:

  * raw scale  -- the same edge fit as beam (`ntof_scint_stack.ana.fit_wall_u`
                  and the u-binned `wall_edge_scale.measure`) on cosmic A
                  tracks.  Cosmics reading ~1.11 => the beam is different;
                  ~0.89 => the A-C truth or the levers are off.
  * truth scale -- the same fit with the A-C line's tan in place of the raw
                  tan.  Geometry consistent with the A-C truth => s = 1.

THE JOIN.  Beam-off DREAM triggers are put on the n_TOF clock by
`clock_match.py` (per sub-run x n_TOF run; |leave-one-out res| < 50 ns).  For
each matched trigger the n_TOF WALA tree is read in the bunch, around the
matched wall-singles time; every channel's largest hit in the window is kept,
exactly as `ntof_scint_stack.extract.wide_hits` does on the beam slim.
Triggers attributed to ANY arm are used (the A-C through-goers mostly trigger
on C): raw tof is not on a common zero across arms' trees with no flash, so
the window is centred per triggering arm on the measured peak of WALA dt.

Tracks: the cosmic full-pass track tables (`results/tracking/k_run_147/`, raw
tans are k-independent), gated, u_mm = x_local + the plane centre's u, the
frame `ntof_scint_stack.extract` uses.  Truth: the A-C line of
`insitu_calib.py truth` (`~/scratch/ntof_insitu/truth.parquet`).

    python ntof_cosmics/cosmic_wall_scale.py build      # per-track table
    python ntof_cosmics/cosmic_wall_scale.py ana        # the scales, profile, beam reference
    python ntof_cosmics/cosmic_wall_scale.py report     # report.html under OUT
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ntof_cosmics import clock_match as CM  # noqa: E402
from ntof_cosmics import wall_edge_scale as WES  # noqa: E402
from ntof_scint_stack import ana as SA  # noqa: E402
from ntof_scint_stack.extract import MV_PER_ADC  # noqa: E402
from sept26_prelim_analysis.det_a_scint import layer_geometry  # noqa: E402

RUN = 'run_149'
ARM = 'A'
TRACKS = HERE / 'results' / 'tracking' / 'k_run_147'
TRUTH = Path('~/scratch/ntof_insitu/truth.parquet').expanduser()
OUT = Path('/media/dylan/data/x17/ntof_cosmics/cosmic_wall_scale')
RES_NS = 50.0
#: hits kept within this of the per-arm peak of (WALA tof - matched singles
#: tof); on A's own triggers the peak is 10 ns wide (`build` prints it)
WIN_NS = 50.0


def matched_triggers(sub: str) -> pd.DataFrame:
    rows = []
    for f in sorted((CM.OUT).glob(f'pairs_{RUN}_{sub}_*.csv')):
        ntof = int(f.stem.rsplit('_', 1)[1])
        c = pd.read_csv(f)
        c = c[c.res.abs() <= RES_NS].assign(ntof=ntof)
        rows.append(c)
    if not rows:
        return pd.DataFrame()
    c = pd.concat(rows, ignore_index=True)
    c['ntof_arm'] = np.array(list('ABCD'))[c.arm.to_numpy()]
    # a trigger matched in two n_TOF runs (sub-run straddle) keeps the better
    c = c.loc[c.res.abs().groupby(c.eventId).idxmin()]
    return c.rename(columns={'eventId': 'event_id'})


_WALL: dict = {}


def wall_tree(ntof: int) -> dict:
    if ntof not in _WALL:
        want = ['BunchNumber', 'detn', 'amp', 'tof']
        parts = [uproot.open(f)[f'WAL{ARM}'].arrays(want, library='np') for f in CM._parts(ntof)]
        d = {k: np.concatenate([p[k] for p in parts]) for k in want}
        o = np.lexsort((d['tof'], d['BunchNumber']))
        _WALL[ntof] = {k: v[o] for k, v in d.items()}
    return _WALL[ntof]


def wall_hits(trig: pd.DataFrame) -> pd.DataFrame:
    """Every WALA hit within +-1 us of each trigger's matched singles time."""
    out = []
    for ntof, g in trig.groupby('ntof'):
        W = wall_tree(int(ntof))
        key = W['BunchNumber'].astype(np.float64) * 1e9 + W['tof']
        k0 = g.bunch.to_numpy(np.float64) * 1e9 + g.tof.to_numpy()
        lo, hi = np.searchsorted(key, k0 - 1000), np.searchsorted(key, k0 + 1000)
        n = hi - lo
        i_t = np.repeat(np.arange(len(g)), n)
        i_h = np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        out.append(pd.DataFrame(dict(
            event_id=g.event_id.to_numpy()[i_t], trig_arm=g.ntof_arm.to_numpy()[i_t],
            detn=W['detn'][i_h].astype(int),
            amp=W['amp'][i_h] * MV_PER_ADC, dt=W['tof'][i_h] - g.tof.to_numpy()[i_t])))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def peaks(h: pd.DataFrame) -> dict:
    """Per triggering arm, the peak of WALA dt (2 ns bins)."""
    out = {}
    for arm, g in h.groupby('trig_arm'):
        n, e = np.histogram(g.dt, bins=1000, range=(-1000, 1000))
        k = n.argmax()
        core = g.dt[(g.dt - 0.5 * (e[k] + e[k + 1])).abs() < 20]
        out[arm] = (float(np.median(core)), int(n[k]), int(len(g)))
    return out


def wide(h: pd.DataFrame, pk: dict) -> pd.DataFrame:
    c = h.trig_arm.map({a: v[0] for a, v in pk.items()})
    x = h[(h.dt - c).abs() <= WIN_NS]
    x = x.sort_values('amp', ascending=False).drop_duplicates(['event_id', 'detn'])
    p = x.pivot_table(index='event_id', columns='detn', values='amp', aggfunc='first')
    p.columns = [f'w{int(c)}_amp_on' for c in p.columns]
    for n in range(1, 9):
        if f'w{n}_amp_on' not in p.columns:
            p[f'w{n}_amp_on'] = np.nan
    return p[[f'w{n}_amp_on' for n in range(1, 9)]].reset_index()


def build() -> pd.DataFrame:
    geo = layer_geometry(RUN, ARM)
    truth = pd.read_parquet(TRUTH)
    truth = truth[truth.arm == ARM][['subrun', 'event_id', 'tan_x', 'tan_y', 'sep_mm']]
    T, H = [], []
    subs = sorted({f.name.split(f'{RUN}_')[1].rsplit('_', 1)[0]
                   for f in CM.OUT.glob(f'pairs_{RUN}_*.csv')})
    for sub in subs:
        trig = matched_triggers(sub)
        if not len(trig):
            continue
        t = pd.read_parquet(TRACKS / f'tracks_{RUN}_{sub}.parquet')
        t = t[t.arm == ARM].copy()
        for c in ('gated', 'x_quality_ok', 'x_plausible', 'y_quality_ok'):
            t[c] = t[c].astype('boolean').fillna(False).astype(bool)
        t['n_trk'] = t[t.gated].groupby('event_id').event_id.transform('size')
        t['n_trk_x'] = t[t.x_quality_ok & t.x_plausible].groupby('event_id').event_id.transform('size')
        h = wall_hits(trig)
        H.append(h.assign(subrun=sub))
        t = t.merge(trig[['event_id', 'ntof_arm', 'ntof', 'bunch', 'tof', 'res']],
                    on='event_id', how='inner')
        T.append(t)
        print(f'{sub}: {len(trig)} matched triggers ({(trig.ntof_arm == ARM).sum()} on arm A), '
              f'{t.event_id.nunique()} with an A track, {t.gated.sum()} gated', flush=True)
    H = pd.concat(H, ignore_index=True)
    pk = peaks(H)
    print('WALA dt peak per triggering arm (ns, peak count, hits):', pk)
    W = pd.concat([wide(g, pk).assign(subrun=s) for s, g in H.groupby('subrun')], ignore_index=True)
    T2 = []
    for t in T:
        sub = t.subrun.iloc[0]
        t = t.merge(W[W.subrun == sub].drop(columns='subrun'), on='event_id', how='left')
        t['u_mm'] = t.x_local + geo['u_mm']
        t['subrun'] = sub
        t = t.merge(truth, on=['subrun', 'event_id'], how='left', suffixes=('', '_true'))
        T2.append(t)
    T = pd.concat(T2, ignore_index=True)
    print(f'{T.tan_x.notna().sum()} A tracks with A-C truth')
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ['subrun', 'event_id', 'ntof_arm', 'gated', 'x_quality_ok', 'x_plausible',
            'y_quality_ok', 'n_trk', 'n_trk_x', 'u_mm', 'x_local', 'y_local', 'tan_raw_x',
            'tan_raw_y', 'x_t0', 'y_t0', 'q_per_len', 'chi2dof_x', 'x_n_strips',
            'x_slope_reliable', 'ntof', 'bunch', 'tof', 'res', 'tan_x', 'tan_y', 'sep_mm'] + \
        [f'w{n}_amp_on' for n in range(1, 9)]
    T[keep].to_parquet(OUT / 'tracks_A.parquet', index=False)
    H.to_parquet(OUT / 'wall_hits_A.parquet', index=False)
    for arm, g in H.groupby('trig_arm'):
        hist, e = np.histogram(g.dt - pk[arm][0], bins=40, range=(-200, 200))
        print(f'{arm}-triggered: WALA dt - peak, 10 ns bins -200..200:', ' '.join(map(str, hist)))
    print(f'wrote {OUT}/tracks_A.parquet ({len(T)} tracks)')
    return T


def scales(d: pd.DataFrame, L: float, tan_col: str, label: str) -> dict:
    x = d[d[tan_col].notna()].copy()
    x['tan_raw_x'] = x[tan_col]
    try:
        r = SA.fit_wall_u(x, L, 1.0)
    except Exception as err:          # noqa: BLE001
        r = dict(s=np.nan, s_err=np.nan, error=str(err))
    return dict(sample=label, tan=tan_col, **r)


def _cosmic_singles(d: pd.DataFrame, sel: pd.Series) -> pd.DataFrame:
    x = d[sel].copy()
    return x[x.groupby(['subrun', 'event_id']).event_id.transform('size') == 1]


def profile(x: pd.DataFrame, L: float, s_grid) -> pd.DataFrame:
    """-log L of the `fit_wall_u` model profiled over sigma, offsets, floors."""
    F = SA.wall_groups_fired(x)
    one = F.sum(1) == 1
    u, tn, gf = x.u_mm.to_numpy()[one], x.tan_raw_x.to_numpy()[one], np.argmax(F[one], 1)
    ub = SA.WALL_EDGES[1:4]
    X, T, Y, B = [], [], [], []
    for b in range(3):
        m = np.isin(gf, [b, b + 1]) & (np.abs(u + L * tn - ub[b]) < 90)
        X.append(u[m]), T.append(tn[m]), Y.append(gf[m] == b + 1), B.append(np.full(m.sum(), b))
    X, T, Y, B = map(np.concatenate, (X, T, Y, B))
    prev, rows = np.r_[np.log(8), 0, 0, 0, -3, 3], []
    for sv in s_grid:
        r = minimize(lambda p: SA._nll_edges(np.r_[sv, p], X, T, Y, B, ub, L), prev, method='L-BFGS-B')
        prev = r.x
        rows.append(dict(s=sv, nll=r.fun, sigma=float(np.exp(r.x[0]))))
    P = pd.DataFrame(rows)
    P['dnll'] = P.nll - P.nll.min()
    return P


def beam_reference() -> pd.DataFrame:
    """The beam scale with the pointing-safe estimator (u-binned medians,
    `wall_edge_scale.measure`), overall and by time since the flash.  The
    edge-likelihood fit is NOT used on beam: there tan ~ (u - u_c)/D, so s is
    degenerate with the per-boundary offsets (it returns 0.5 at > 20 ms)."""
    cols = ['run', 'arm', 'u_mm', 'tan_raw_x', 'n_trk', 't_since_flash_ns'] + \
        [f'w{i}_amp_on' for i in range(1, 9)]
    T = []
    for f in sorted(glob.glob(str(WES.STACK / 'stack_run_*.parquet'))):
        x = pd.read_parquet(f, columns=cols)
        T.append(x[(x.arm == ARM) & (x.n_trk == 1)])
    d = pd.concat(T, ignore_index=True)
    F = SA.wall_groups_fired(d)
    d = d[F.sum(1) == 1].copy()
    d['grp'] = np.argmax(SA.wall_groups_fired(d), 1)
    ms = d.t_since_flash_ns / 1e6
    rows = []
    for lab, m in [('all', np.ones(len(d), bool)), ('10-15 ms', (ms >= 10) & (ms < 15)),
                   ('15-20 ms', (ms >= 15) & (ms < 20)), ('20-30 ms', (ms >= 20) & (ms < 30)),
                   ('30-45 ms', (ms >= 30) & (ms < 45)), ('45-80 ms', (ms >= 45) & (ms < 80))]:
        R = [r for r in WES.measure(d[m], ARM, lab) if r['boundary'].startswith('outer')]
        if R:
            rows.append(dict(sample=lab, n=R[0]['n'], true_over_raw=R[0]['true_over_raw'],
                             D_eff=R[0]['D_eff']))
    return pd.DataFrame(rows)


def ana() -> int:
    """The cosmic wall scale s in u + L s tan, from the sharpness of the
    internal group boundaries (`fit_wall_u`).  The u-binned median test of
    `wall_edge_scale` needs a pointing source (u correlated with tan); cosmics
    have none, so for cosmics this fit is the estimator, and it is free of the
    offset degeneracy that makes it unusable on beam."""
    geo = layer_geometry(RUN, ARM)
    L = geo['w_wall'] - geo['w_strip']
    d = pd.read_parquet(OUT / 'tracks_A.parquet')
    sels = {'gated (beam selection)': d.gated,
            'x plane only (quality + |tan_x| < 0.6)': d.x_quality_ok & d.x_plausible}
    rows = []
    for lab, m in sels.items():
        x = _cosmic_singles(d, m)
        for samp, mm in (('all', np.ones(len(x), bool)), ('A-triggered', x.ntof_arm == ARM),
                         ('other-arm triggered', x.ntof_arm != ARM),
                         ('|tan| < 0.15', x.tan_raw_x.abs() < 0.15),
                         ('|tan| 0.15-0.3', x.tan_raw_x.abs().between(0.15, 0.3)),
                         ('|tan| 0.3-0.6', x.tan_raw_x.abs().between(0.3, 0.6))):
            rows.append(scales(x[mm], L, 'tan_raw_x', lab) | dict(selection=lab, sample=samp))
    R = pd.DataFrame(rows)
    x = _cosmic_singles(d, sels['x plane only (quality + |tan_x| < 0.6)'])
    P = profile(x, L, np.round(np.arange(0.70, 1.501, 0.025), 3))
    subs = x.subrun.unique()
    rng = np.random.default_rng(0)
    boot = [SA.fit_wall_u(pd.concat([x[x.subrun == s] for s in rng.choice(subs, len(subs))]),
                          L, 1.0)['s'] for _ in range(100)]
    Bm = beam_reference()
    summ = dict(L=L, n_tracks=int(len(d)), n_single_x=int(len(x)),
                n_truth=int(d.tan_x.notna().sum()),
                boot_median=float(np.median(boot)), boot_lo=float(np.percentile(boot, 16)),
                boot_hi=float(np.percentile(boot, 84)),
                s_best=float(P.s[P.dnll.idxmin()]),
                dnll_at_beam=float(np.interp(0.89, P.s, P.dnll)),
                ac_line=1.11, beam_all=float(Bm.true_over_raw.iloc[0]))
    print(f'lever L = {L:.1f} mm')
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(R[['selection', 'sample', 'n', 's', 's_err', 'sigma']].round(3).to_string(index=False))
        print(Bm.round(3).to_string(index=False))
    print(json.dumps(summ, indent=1))
    R.to_csv(OUT / 'scales.csv', index=False)
    P.to_csv(OUT / 'profile.csv', index=False)
    Bm.to_csv(OUT / 'beam_reference.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))
    return 0


def report() -> int:
    import matplotlib
    matplotlib.use('Agg')
    from sept26_prelim_analysis import figstyle as FSY
    FSY.use()
    R = pd.read_csv(OUT / 'scales.csv')
    P = pd.read_csv(OUT / 'profile.csv')
    Bm = pd.read_csv(OUT / 'beam_reference.csv')
    S = json.loads((OUT / 'summary.json').read_text())
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)

    fig, ax = FSY.figure(FSY.FIG)
    ax.plot(P.s, P.dnll, color=FSY.INK, lw=1.6)
    ax.axvspan(0.89, 0.93, color=FSY.DET_COLOR['B'], alpha=0.18, lw=0)
    ax.axvline(1.11, color=FSY.DET_COLOR['C'], lw=1.2, ls='--')
    ax.text(0.91, P.dnll.max() * 0.92, 'beam\n(wall, u-binned)', ha='center', va='top', fontsize=9,
            color=FSY.DET_COLOR['B'])
    ax.text(1.115, P.dnll.max() * 0.92, 'cosmic A-C line', ha='left', va='top', fontsize=9,
            color=FSY.DET_COLOR['C'])
    ax.set_xlabel('wall scale s  (true tan = s x production raw tan)')
    ax.set_ylabel('cosmic -log L  (relative)')
    FSY.title(ax, f"Cosmics at A's wall read s = {S['s_best']:.2f}, not the beam's 0.89",
              f"run_149, {S['n_single_x']} single tracks; edge fit profiled over width, offsets, floors")
    FSY.save(fig, OUT / 'figures' / 'cosmic_wall_profile.png', data=P)

    def tab(df, cols, fmt):
        h = ''.join(f'<th>{c}</th>' for c in cols)
        b = ''.join('<tr>' + ''.join(f'<td>{fmt.get(c, "{}").format(r[c])}</td>' for c in cols) + '</tr>'
                    for _, r in df.iterrows())
        return f'<table><tr>{h}</tr>{b}</table>'
    rr = R[['selection', 'sample', 'n', 's', 's_err', 'sigma']]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Cosmic wall scale</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 16px;color:#1b2430;background:#fff}}
table{{border-collapse:collapse;margin:1em 0}}td,th{{border-bottom:1px solid #e6e9ee;padding:3px 10px;text-align:right}}
th{{color:#6a7583;font-weight:600}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}
.verdict{{border-left:4px solid #8a3f8f;padding:.4em 1em;background:#f7f4f8}}</style></head><body>
<h1>Chamber A angle scale at the SiPM wall: cosmics against beam</h1>
<p class="verdict"><b>Verdict.</b> On run_149 cosmics, A's own SiPM wall measures true tan =
<b>{S['s_best']:.2f} x production raw</b> (sub-run bootstrap {S['boot_median']:.2f},
68 % {S['boot_lo']:.2f}-{S['boot_hi']:.2f}). That agrees with the cosmic A-C line
({S['ac_line']:.2f}). The beam value from the same wall is {S['beam_all']:.2f}, and the cosmic
likelihood disfavours it by &Delta;(-log L) = {S['dnll_at_beam']:.0f}. <b>So the wall
geometry and lever arms are consistent with the cosmic truth, and the 20-25 % beam/cosmic gap
is real.</b> Beam tracks in A reconstruct steeper than cosmic muons with the same bundle:
about 20 % on the late-time plateau (0.92), more at 10-20 ms after the flash.</p>
<h2>What was compared</h2>
<p>Beam-off DREAM triggers were put on the n_TOF clock (<code>clock_match.py</code>, 35 sub-runs x
n_TOF 224678-224687, 88-99 % matched, core residual 9-18 ns). For each matched trigger, A's wall
channels were read in a +-{WIN_NS:g} ns window around the matched singles time, as the beam
stack reads them. Tracks: the cosmic full-pass tables, u at the strip plane in the wall's
structure frame, lever L = {S['L']:.1f} mm. The fit is <code>ntof_scint_stack.ana.fit_wall_u</code>:
the scale s in u + L s tan, set by how sharp the three internal group boundaries are.</p>
<p>Two estimators, and why each sample uses its own: the u-binned median test needs a
pointing source (u correlated with tan), so it works on beam and not on cosmics. The edge
likelihood needs tan spread at fixed u, so it works on cosmics and not on beam: there, s is
degenerate with the per-boundary offsets and the fit returns 0.5 beyond 20 ms.</p>
<img src="figures/cosmic_wall_profile.png" alt="cosmic likelihood profile in s">
<h2>Cosmic fits</h2>{tab(rr, rr.columns, {'s': '{:.3f}', 's_err': '{:.3f}', 'sigma': '{:.1f}'})}
<p>The 17 tracks with A-C truth on matched triggers are too few for a truth-scale wall fit,
because n_TOF records only ~16 % of the time.</p>
<h2>Beam reference (u-binned, outer boundary pair)</h2>
{tab(Bm, Bm.columns, {'true_over_raw': '{:.3f}', 'D_eff': '{:.0f}'})}
<p>The beam scale does depend on time since the flash early on: 0.77 at 10-15 ms and 0.87
at 15-20 ms, then flat at 0.91-0.93 from 20 to 80 ms. Before 10 ms the edges are too wide to fit.
So part of the campaign-average 0.89 is a flash-related transient, but the plateau still sits
about 20 % below the cosmic 1.15.</p>
<h2>What this does not rule out</h2>
<ul><li>A beam-specific effect in the chamber: low-energy electrons (scattering, delta rays)
or <b>beam-on coherent noise</b>. The earlier noise-injection test used beam-off noise
(run_149) only, and beam common mode is 10-20x larger.</li>
<li>The cosmic scale rises with |tan| (1.0 -> 1.2), so a single number is a summary. The
beam's large-|tan| tracks sit at large |u|, so the two samples weight angle and position
differently.</li>
<li>The 10-20 ms transient: flash space charge or baseline recovery is plausible, not shown.</li>
<li>Steady-state space charge under beam (the plateau is flat from 20 to 80 ms).</li></ul>
<p><small>Generated by <code>ntof_cosmics/cosmic_wall_scale.py report</code>.</small></p>
</body></html>"""
    (OUT / 'report.html').write_text(html)
    print(f'wrote {OUT}/report.html')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['build', 'ana', 'report'])
    a = ap.parse_args()
    if a.step == 'build':
        build()
        return 0
    return ana() if a.step == 'ana' else report()


if __name__ == '__main__':
    raise SystemExit(main())
