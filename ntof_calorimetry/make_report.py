#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report.py -- figures and ``report.html`` for ntof_calorimetry, built from
the tables `scint_ecal` (C1) and `liquid_salvage` (C4) wrote.  Re-run after
either changes: numbers, figures and verdict text move together.

    python -m ntof_calorimetry.make_report      # -> OUT/report.html, OUT/figures/
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from ntof_calorimetry import landau as LD  # noqa: E402
from ntof_calorimetry.mip_sample import OUT  # noqa: E402
from ntof_scint_stack.extract import PLAS_THR  # noqa: E402
from sept26_prelim_analysis import figstyle as FS  # noqa: E402

C1, C4, M2, FIG = OUT / 'c1', OUT / 'c4', OUT / 'm2', OUT / 'figures'
E_MIP = 3.41
BARS = ['PSSA1', 'PSSA2', 'PSSC1', 'PSSC2', 'PSSD1', 'PSSD2']


def _load() -> dict:
    t = {k: pd.read_csv(C1 / f'{k}.csv') for k in
         ('mip_per_bar', 'calib_variants', 'path_check', 'flash_time', 'saturation')}
    t.update({f'liq_{k}': pd.read_csv(C4 / f'{k}.csv') for k in
              ('overall', 'maps', 'profiles', 'mip_scale', 'penetration')})
    t['calib'] = json.loads((C1 / 'calib_plastic_e.json').read_text())
    t['spec'] = pd.read_parquet(C1 / 'mip_spectra.parquet')
    t['liq'] = pd.read_parquet(C4 / 'liquid_tagged.parquet')
    if (M2 / 'summary.json').exists():
        t.update({f'm2_{k}': pd.read_csv(M2 / f'{k}.csv') for k in
                  ('resolution', 'two_mip', 'vs_path', 'depth', 'gain_map')})
        t['m2_summary'] = json.loads((M2 / 'summary.json').read_text())
        t['m2_sel'] = pd.read_parquet(M2 / 'dedx_selected.parquet')
    return t


def fig_m2(t) -> None:
    S, R = t['m2_sel'], t['m2_resolution']
    rng = np.random.default_rng(3)
    fig, ax = FS.figure(FS.FIG)
    edges = np.linspace(0, 5, 101)
    rows = []
    for arm, est, ls in (('A', 'q_whole', '-'), ('A', 'q_plat', '--'), ('C', 'q_plat', ':')):
        m = float(R[(R.arm == arm) & (R.estimator == est)].mpv.iloc[0])
        x = S[(S.arm == arm) & S[est].notna()]
        v = x[est].to_numpy() / m
        n, _ = np.histogram(v, edges)
        st = FS.det_style(arm)
        lab = f'{arm}, {"whole gap" if est == "q_whole" else "plateau"}'
        ax.stairs(n / n.sum(), edges, color=st['color'], lw=1.5, ls=ls, label=f'1 muon: {lab}')
        rows += [dict(series=f'1mip_{arm}_{est}', lo=a, f=b) for a, b in zip(edges[:-1], n / n.sum())]
        if est == 'q_whole':
            Q, p = x[est].to_numpy() * x.path_mm.to_numpy(), x.path_mm.to_numpy()
            i, j = rng.integers(0, len(v), (2, 20000))
            two = (Q[i] + Q[j] * p[i] / p[j]) / p[i] / m
            n2, _ = np.histogram(two, edges)
            ax.stairs(n2 / n2.sum(), edges, color=FS.INK, lw=1.3, label='2 muons (summed real tracks), A whole gap')
            rows += [dict(series='2mip_A', lo=a, f=b) for a, b in zip(edges[:-1], n2 / n2.sum())]
            cut = np.quantile(v, 0.9)
            ax.axvline(cut, color=FS.COPPER, lw=1.0, ls=':')
            ax.text(cut + 0.05, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.03,
                    '10 % 1-muon\nmis-tag', color=FS.COPPER, fontsize=8, va='top')
    ax.set_xlabel('road charge per mm of path / MPV')
    ax.set_ylabel('fraction')
    ax.legend(fontsize=8)
    FS.title(ax, 'One 30 mm gap cannot tell one MIP from two',
             'run_149 cosmics; the 2-muon sample is a best case (no reco or ZS effects)')
    FS.save(fig, FIG / 'm2_landau', data=pd.DataFrame(rows))

    Z = t['m2_depth']
    fig, ax = FS.figure(FS.HALF)
    for arm in 'AC':
        d = Z[(Z.arm == arm) & (Z.depth_mm >= 0) & (Z.depth_mm <= 40)]
        st = FS.det_style(arm)
        ax.plot(d.depth_mm, d.frac / d.frac.max(), color=st['color'], marker=st['marker'], ms=4,
                label=f'{st["label"]} (v = {"34" if arm == "A" else "28"} um/ns)')
    ax.set_xlabel('drift depth of the sample, mm')
    ax.set_ylabel('mean road charge per sample / peak')
    ax.legend(fontsize=8)
    FS.title(ax, 'No attachment visible on either chamber', 'all tracks, absolute charge, each depth from every window covering it')
    FS.save(fig, FIG / 'm2_depth', data=Z)


def fig_ladder(t) -> None:
    """Every calibration point over what the production line predicts there."""
    V = t['calib_variants'].set_index('ch')
    R = t['mip_per_bar']
    R = R[R['sample'] == 'cosmic'].set_index('ch')
    J = json.loads(Path(REPO / 'mx_july_beam_qa' / 'calib' / 'srccal_energy_calib.json')
                   .read_text())['channels']
    rows = []
    for ch in BARS:
        a, b = V.loc[ch, 'line_477_699_a'], V.loc[ch, 'line_477_699_b']
        for e, p in J[ch]['points'].items():
            e = float(e) / 1000
            rows.append(dict(ch=ch, e_mev=e, kind='source edge', mv=p['mv'],
                             ratio=p['mv'] / (a * e * 1000 + b), err=p['err'] / (a * e * 1000 + b)))
        r = R.loc[ch]
        rows.append(dict(ch=ch, e_mev=r.exp_mev, kind='cosmic MIP', mv=r.mpv,
                         ratio=r.mpv / (a * r.exp_mev * 1000 + b),
                         err=np.hypot(r.mpv_err / r.mpv, LD_SYS(r)) * r.mpv / (a * r.exp_mev * 1000 + b)))
    D = pd.DataFrame(rows)
    fig, ax = FS.figure(FS.FIG)
    off = dict(zip(BARS, np.linspace(-0.035, 0.035, len(BARS))))
    for ch in BARS:
        d = D[D.ch == ch]
        st = FS.det_style(ch[3])
        x = d.e_mev * (1 + off[ch])
        ax.errorbar(x, d.ratio, d.err, color=st['color'], marker=st['marker'], ms=6, lw=1.0,
                    mfc=st['color'] if ch[4] == '1' else FS.SURFACE, mec=st['color'], mew=1.3,
                    capsize=0, label=f'{ch[3]}{"L" if ch[4] == "1" else "R"} ({ch})')
    ax.axhline(1, color=FS.MUTED, lw=1.0, ls='--')
    ax.axvspan(3.9, 4.8, color=FS.BAND_SIGNAL, alpha=0.08, lw=0)
    ax.text(4.3, 1.13, 'X17 soft leg\nat >=140 deg', ha='center', va='top', fontsize=8.5,
            color=FS.BAND_SIGNAL)
    ax.set_xscale('log')
    ax.set_xticks([0.477, 0.699, 1.612, 3.41])
    ax.set_xticklabels(['Cs 0.48', 'Y 0.70', 'Y 1.61', 'MIP 3.41'])
    ax.set_xlabel('energy, MeV(ee)')
    ax.set_ylabel('measured mV / production line')
    ax.set_ylim(0.68, 1.16)
    ax.legend(ncol=3, fontsize=8, loc='lower left')
    FS.title(ax, 'The source line overshoots the MIP by up to 24 %',
             'production keVee line (through 477 and 699 keVee) extrapolated; open = R bar')
    FS.save(fig, FIG / 'c1_ladder', data=D)


def LD_SYS(r) -> float:
    v = np.array([r['mpv_closure_0.9'], r['mpv_closure_1'], r['mpv_prior_0.13'], r['mpv_prior_0.23']])
    return float(np.nanmax(np.abs(v / r.mpv - 1)))


def fig_spectra(t) -> None:
    S, R = t['spec'], t['mip_per_bar']
    fig, axs = plt.subplots(2, 3, figsize=FS.FULL, sharey=False)
    rows = []
    for ax, ch in zip(axs.T.flat, BARS):
        FS.strip(ax)
        arm, bar = ch[3], int(ch[4])
        x = S[(S['sample'] == 'cosmic') & (S.arm == arm) & (S.bar == bar)]
        r = R[(R.ch == ch) & (R['sample'] == 'cosmic')].iloc[0]
        edges = np.linspace(0, 2.6 * r.mpv, 36)
        n, _ = np.histogram(x.e_mv, edges)
        c = 0.5 * (edges[1:] + edges[:-1])
        st = FS.det_style(arm)
        ax.stairs(n, edges, color=FS.INK, lw=1.1)
        sc = LD.XI_OVER_MPV_PVT20 * r.mpv
        g = np.linspace(0, edges[-1], 400)
        f = LD._langaus_grid(g, r.mpv - LD.LANDAU_PEAK * sc, sc, r.sigma)
        hi = c > r.mpv
        w = edges[1] - edges[0]
        k = n[hi].sum() / max(np.interp(c[hi], g, f).sum() * w, 1e-12)
        ax.plot(g, k * f * w, color=st['color'], lw=1.8)
        thr = PLAS_THR[arm] * x.cos.median()
        ax.axvline(thr, color=FS.COPPER, lw=1.0, ls=':')
        ax.axvline(r.mpv, color=st['color'], lw=0.9, ls='--')
        ax.set_title(f'{ch}   MPV {r.mpv:.0f} mV = {r.exp_mev:.2f} MeV', fontsize=9.5)
        ax.set_xlabel('amplitude x cos, mV')
        rows += [dict(ch=ch, lo=a, hi=b, n=int(m)) for a, b, m in zip(edges[:-1], edges[1:], n)]
    axs[0, 0].set_ylabel('tracks / bin')
    axs[1, 0].set_ylabel('tracks / bin')
    axs[0, 0].text(0.98, 0.95, 'dotted: trigger\nthreshold x cos', transform=axs[0, 0].transAxes,
                   ha='right', va='top', fontsize=8, color=FS.COPPER)
    FS.fig_title(fig, 'Cosmic MIP in each plastic bar, beam-off run_149',
                 'Landau(x)Gauss, unbinned, each event truncated at its own trigger threshold')
    FS.save(fig, FIG / 'c1_mip_spectra', data=pd.DataFrame(rows))


def fig_path(t) -> None:
    P = t['path_check']
    fig, ax = FS.figure(FS.HALF)
    g = np.linspace(19.5, 29.5, 50)
    ax.plot(g, [LD.mpv(x) / LD.mpv(20) for x in g], color=FS.MUTED, lw=1.2, ls='--',
            label='Bichsel, 20 mm -> path')
    for arm, d in P.groupby('arm'):
        st = FS.det_style(arm)
        ax.errorbar(d.path_mm, d.mpv_rel, d.err, color=st['color'], marker=st['marker'], lw=0,
                    elinewidth=1, ms=6, label=st['label'])
    ax.set_xlabel('path through the bar, mm')
    ax.set_ylabel('MIP peak / bar mean (raw mV)')
    ax.legend(fontsize=8)
    FS.title(ax, 'Linear to 1.45 MIP', 'cosmic MPV in path terciles')
    FS.save(fig, FIG / 'c1_path', data=P)


def fig_liquid(t) -> None:
    P = t['liq_profiles']
    fig, axs = plt.subplots(1, 2, figsize=FS.WIDE)
    for ax, axis in zip(axs, ('u', 'v')):
        FS.strip(ax)
        for arm in 'ACD':
            d = P[(P.arm == arm) & (P.axis == axis) & (P.n >= 20)]
            st = FS.det_style(arm)
            ax.errorbar(d.mid, d.eff, d.err, color=st['color'], marker=st['marker'], lw=1.2,
                        ms=5, label=st['label'])
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=FS.MUTED, lw=0.8, ls=':')
        ax.set_xlabel(f'{axis} on the liquid face, mm' + ('  (A, D: PMT at +u)' if axis == 'u'
                                                          else '  (+v = up; C: PMT on top)'))
    axs[0].set_ylabel('liquid MIP efficiency')
    axs[1].legend(fontsize=8)
    FS.fig_title(fig, 'A and D pass 50 % only on the half nearest their PMT; C never',
                 'cosmic muons confirmed through the plastic, net of the pre-trigger window')
    FS.save(fig, FIG / 'c4_profiles', data=P)

    L = t['liq']
    S = t['liq_mip_scale'].set_index('arm')
    fig, ax = FS.figure(FS.HALF)
    edges = np.arange(0, 121, 4)
    rows = []
    for arm in 'ACD':
        x = L[(L['sample'] == 'cosmic') & (L.arm == arm) & L.lf_on]
        n, _ = np.histogram(x.la, edges)
        st = FS.det_style(arm)
        ax.stairs(n / max(n.sum(), 1), edges, color=st['color'], lw=1.4, label=st['label'])
        rows += [dict(arm=arm, lo=a, n=int(m)) for a, m in zip(edges[:-1], n)]
    ax.axvspan(16, 18, color=FS.COPPER, alpha=0.25, lw=0)
    ax.annotate('n_TOF ZS threshold', xy=(17, 0.21), xytext=(40, 0.22), fontsize=8,
                color=FS.COPPER, va='center', arrowprops=dict(arrowstyle='->', color=FS.COPPER, lw=0.8))
    exp_mv = 3.1 * float(S.mv_per_mev_source.dropna().mean())
    ax.annotate(f'source scale\npredicts ~{exp_mv:.0f} mV', xy=(118, 0.002), xytext=(70, 0.12),
                fontsize=8, color=FS.MUTED, arrowprops=dict(arrowstyle='->', color=FS.MUTED, lw=0.8))
    ax.set_xlabel('liquid amplitude, mV')
    ax.set_ylabel('fraction of fired')
    ax.legend(fontsize=8)
    FS.title(ax, 'A MIP leaves 20-28 mV, not ~145', 'the spectrum starts at the ZS threshold')
    FS.save(fig, FIG / 'c4_liquid_spectrum', data=pd.DataFrame(rows))

    M = t['liq_maps']
    M = M[(M['sample'] == 'cosmic') & (M.n >= 8)]
    fig, axs = plt.subplots(1, 4, figsize=(FS.WIDE[0], FS.WIDE[1] + 0.4),
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 0.05], wspace=0.3))
    cax, axs = axs[3], axs[:3]
    for ax, arm in zip(axs, 'ACD'):
        d = M[M.arm == arm]
        G = np.full((9, 9), np.nan)
        for _, r in d.iterrows():
            if 0 <= r.iu < 9 and 0 <= r.iv < 9:
                G[int(r.iv), int(r.iu)] = r.eff
        im = ax.imshow(G, origin='lower', extent=(-225, 225, -225, 225), vmin=0, vmax=1,
                       cmap='Blues')
        ax.set_title(f'liquid {arm}', fontsize=9.5)
        ax.set_xlabel('u, mm')
        ax.grid(False)
    axs[0].set_ylabel('v, mm')
    fig.colorbar(im, cax=cax, label='MIP efficiency')
    FS.fig_title(fig, 'Liquid MIP efficiency on the face, 50 mm cells',
                 'cosmics, cells with >= 8 tags; blank = no tags')
    FS.save(fig, FIG / 'c4_maps', data=M)


def _tab(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    h = ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns)
    body = ''
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                v = '' if not np.isfinite(v) else fmt.get(c, '{:.3g}').format(v)
            cells.append(f'<td>{html.escape(str(v))}</td>')
        body += '<tr>' + ''.join(cells) + '</tr>'
    return f'<div class="tw"><table><tr>{h}</tr>{body}</table></div>'


def report(t) -> str:
    R = t['mip_per_bar']
    Rc = R[R['sample'] == 'cosmic'].copy()
    cal = t['calib']['bars']
    lo, hi = Rc.ratio_line_477_699.min(), Rc.ratio_line_477_699.max()
    LS, PN, LO = t['liq_mip_scale'], t['liq_penetration'], t['liq_overall']
    loc = LO[LO['sample'] == 'cosmic'].set_index('arm')
    pn = PN.set_index('arm')
    P = t['path_check']
    thr = {ch: v['trigger_threshold_mevee'] for ch, v in cal.items()}

    c1 = Rc[['ch', 'n', 'mpv', 'mpv_err', 'sigma', 'exp_mev', 'mv_per_mev_mip', 'mv_per_mev_line',
             'ratio_line_477_699', 'ratio_origin_699', 'ratio_line_all3', 'edge1612_vs_line',
             'mpv_closure_0.9', 'mpv_closure_1', 'thr_mev_mip']].rename(columns={
        'mpv': 'MIP MPV mV', 'mpv_err': 'stat', 'sigma': 'sigma mV', 'exp_mev': 'Delta_p MeV',
        'mv_per_mev_mip': 'mV/MeV MIP', 'mv_per_mev_line': 'mV/MeV line',
        'ratio_line_477_699': 'MIP read by line / Delta_p', 'ratio_origin_699': '... 699 through 0',
        'ratio_line_all3': '... line through 3 edges', 'edge1612_vs_line': '1612 edge / line',
        'mpv_closure_0.9': 'MPV, thr raised to 0.9', 'mpv_closure_1': 'MPV, thr raised to 1.0',
        'thr_mev_mip': 'trigger thr MeVee'})
    beam = R[R['sample'] == 'beam'][['ch', 'n', 'mpv', 'mpv_err']].merge(
        Rc[['ch', 'mpv']], on='ch', suffixes=('_beam', '_cosmic'))
    beam['beam / cosmic'] = beam.mpv_beam / beam.mpv_cosmic
    sat = t['saturation'][['arm', 'bar', 'n_hits', 'n_satuflag', 'amp_max', 'repeated_top',
                           'max_mev_mip', 'repeated_top_mev']]
    lq = LO[LO['sample'].isin(['cosmic', 'beam']) & (LO.n > 0)][
        ['sample', 'arm', 'n', 'eff', 'err', 'p_off', 'amp_med_mv']]

    css = """body{font:15px/1.55 system-ui,-apple-system,sans-serif;max-width:980px;margin:2em auto;padding:0 16px;color:#1b2430;background:#fff}
h1{font-size:1.55em;margin-bottom:.2em}h2{margin-top:1.8em;border-bottom:1px solid #e6e9ee;padding-bottom:.2em}
.tw{overflow-x:auto}table{border-collapse:collapse;margin:.8em 0;font-size:13px}td,th{border-bottom:1px solid #e6e9ee;padding:3px 9px;text-align:right;white-space:nowrap}
th{color:#6a7583;font-weight:600}td:first-child,th:first-child{text-align:left}img{max-width:100%;margin:.6em 0}
.verdict{border-left:4px solid #8a3f8f;padding:.5em 1em;background:#f7f4f8}.cap{color:#6a7583;font-size:13.5px;margin-top:-.3em}
.k{display:inline-block;margin:0 1.6em .6em 0}.k b{display:block;font-size:1.5em}code{background:#f3f4f6;padding:0 .25em;border-radius:3px}"""
    f = lambda x, n=2: f'{x:.{n}f}'  # noqa: E731
    out = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calorimetry C1 + C4</title><style>{css}</style></head><body>
<h1>Plastic energy scale and liquid salvage (calorimetry plan C1, C4)</h1>
<p class="cap">ntof_calorimetry, PLAN.md steps 1-2. Generated by <code>ntof_calorimetry/make_report.py</code>.</p>

<div class="verdict"><p><b>Plastic (C1).</b> The production keVee scale reads a
minimum-ionising muon at <b>{f(lo)}-{f(hi)}</b> of its calculated deposit. Read this as
a <b>{f((1 - hi) * 100, 0)}-{f((1 - lo) * 100, 0)} % under-reading</b> at the 3-5 MeV a
stopped X17 lepton leaves. The cause is the calibration's lever arm, not the detector.
The line is fitted through two Compton edges 0.22 MeV apart and extrapolated ×5. The
plastic response itself is linear in path up to 1.45 MIP (~5 MeV). The source's own
1.6 MeV edge sits below the line by the same 12-22 % on the bars that have it. This does not
meet the plan's kill condition (an unexplained &gt; 20 %). The fix is to anchor on the MIP:
<code>c1/calib_plastic_e.json</code> gives mV/MeVee per bar to 4-7 %, from beam-off cosmics.</p>
<p><b>Liquids (C4).</b> The liquids are not dead, and they are not a lost cause of
mysterious origin. <b>They run at ~1/5-1/7 of the gain their source calibration claims.</b>
A muon crossing the 18 mm cell leaves {f(LS.mpv_mv.min(), 0)}-{f(LS.mpv_mv.max(), 0)} mV
(Landau MPV), against a 16-18 mV n_TOF zero-suppression threshold. That threshold is
{f(LS.zs_mev_mip.min(), 1)}-{f(LS.zs_mev_mip.max(), 1)} MeV on the MIP scale, so the
efficiency is threshold-limited: A {f(loc.loc['A', 'eff'] * 100, 0)} %, D
{f(loc.loc['D', 'eff'] * 100, 0)} %, C {f(loc.loc['C', 'eff'] * 100, 0)} % overall, rising
toward the PMT end on A and D. By the plan's criterion (MIP efficiency &gt; 50 % somewhere),
two regions survive: <b>A's PMT half</b> (u &ge; 0, 65-86 %) and <b>D's outer ~100 mm</b>
(u &ge; 50, 50-70 %). C is below 30 % everywhere; B is not measurable here. The signal was never recorded, so no reprocessing recovers it. The source "edges" were
never Compton edges: LIQA's Cs and Y-88 edges sit at the same amplitude.</p>
<p><b>A by-product that matters for the background.</b> In-beam "through-goers" at
&ge; 10 ms reach the liquid behind the plastic at only <b>{f(pn.loc['A', 'ratio'], 2)} &plusmn;
{f(pn.loc['A', 'ratio_err'], 2)}</b> (A) and {f(pn.loc['C', 'ratio'], 2)} (C) of the
position-matched cosmic expectation. Most of them are therefore <b>not penetrating
muons</b>. They are particles that stop in the stack, consistent with collinear pairs from
a common vertex off the beam axis. So in beam data the plastic cannot be calibrated on them,
and the through-going background is not purely cosmic.</p></div>

<div><span class="k"><b>{f(lo)}-{f(hi)}</b>MIP read by the production line / Bichsel</span>
<span class="k"><b>4-7 %</b>MIP-anchored scale, per bar</span>
<span class="k"><b>{f(min(thr.values()), 1)}-{f(max(thr.values()), 1)} MeVee</b>plastic trigger threshold</span>
<span class="k"><b>{f(LS.mv_per_mev_mip.min(), 0)}-{f(LS.mv_per_mev_mip.max(), 0)} mV/MeV</b>liquid, MIP (source: 46-48)</span></div>

<h2>What was compared</h2>
<p><b>Samples</b> (<code>mip_sample.py</code>):</p>
<ul>
<li><b>cosmic:</b> run_149 beam-off muons on the n_TOF clock (<code>clock_match.py</code>,
n_TOF 224678-87), one gated track per arm, with the n_TOF WAL/PSS/LIQ trees read around the
matched time.</li>
<li><b>beam:</b> A-C and B-D through-goers in the campaign full pass at &ge; 10 ms after the
flash, with the direction taken from the line joining the two chambers.</li>
</ul>
<p><b>Crossings:</b> the scint-stack levers and alignment offsets. <b>Selection:</b> the
predicted plastic bar fired, both ends of the predicted wall group fired, &gt; 20 mm from the
bar's outer edges, &gt; 15 mm from the L/R gap. <b>Expected deposit:</b> the
Landau-Vavilov-Bichsel most probable loss in 20 mm PVT, 3.36-3.42 MeV for
&beta;&gamma; 5-100 (<code>landau.mpv</code>). The mean is ~4.6 MeV, the wrong quantity for
a thin bar.</p>
<p><b>The fit, and why it needs care.</b> The hardware trigger needs a plastic bar above
PLAS_THR (112-151 mV), and on arm A that is <b>0.85-0.9 of the MIP peak</b>. A free fit
there is biased. Each event is therefore truncated at its own threshold × cos&theta;, and only
where the trigger needed this bar (no other arm fired, the other bar below threshold). The
Landau width is held at its physics value (&xi;/&Delta;<sub>p</sub> = 0.051), with a weak
prior on &sigma;/MPV (0.18 &plusmn; 0.05, taken from the bars whose threshold is far below
their peak). A free fit with no prior slides into a low-MPV, wide-&sigma; solution when the
threshold is near the peak. The closure columns show the result: the MPV after raising
every event's threshold to 0.9 and 1.0 of the peak.</p>

<h2>C1 — the plastic scale</h2>
<img src="figures/c1_ladder.png" alt="calibration ladder">
<p class="cap">Each calibration point over the production line's prediction at that energy.
The Cs-137 and Y-88 699 keVee edges sit on the line by construction. The Y-88 1.6 MeV edge
(labelled secondary because of cascade summing, which would push it <i>up</i>) and the MIP
both fall below it on A and D. C is within 3-6 %.</p>
{_tab(c1, {'n': '{:.0f}', 'MIP MPV mV': '{:.1f}', 'stat': '{:.1f}', 'sigma mV': '{:.1f}'})}
<img src="figures/c1_mip_spectra.png" alt="MIP spectra">
<p class="cap">Cosmic MIP spectra per bar with the truncated fit. On A the trigger threshold
sits just under the peak.</p>
<img src="figures/c1_path.png" alt="path check" style="max-width:520px">
<p class="cap">The MIP peak in raw mV (no path correction) against the path through the
bar, normalised per bar. It follows the Bichsel path dependence: the cos correction is right,
and the response is linear from 1.0 to 1.45 MIP, i.e. up to ~5 MeV, which is the whole
soft-leg range.</p>
<p><b>Saturation (C1.3).</b> <code>satuflag</code> is never set in this processing. The
largest amplitudes, and the first values that repeat (clipped pulses), are
{f(sat.repeated_top_mev.min(), 0)}-{f(sat.repeated_top_mev.max(), 0)} MeVee on the MIP scale
(2 V digitisers on a +950 mV baseline). Saturation does not matter below ~15 MeVee.</p>
{_tab(sat, {'n_hits': '{:.0f}', 'n_satuflag': '{:.0f}', 'amp_max': '{:.0f}', 'repeated_top': '{:.0f}', 'max_mev_mip': '{:.1f}', 'repeated_top_mev': '{:.1f}'})}
<p><b>Beam against cosmic, and time since flash (C1.4): not measured.</b> The beam
through-goers' plastic peak is 4-15 % below the cosmic one, and rises with time since the
flash ({f(t['flash_time'].mpv_rel.min())} at 10-15 ms to {f(t['flash_time'].mpv_rel.max())}
at 40-80 ms). That would be a PMT gain sag, except that C4 shows most of these particles do
not penetrate the stack. A change in population with time explains the same trend. Measuring
the plastic gain under beam needs a source of known energy in the beam data itself:
<sup>28</sup>Al &beta; endpoint (C1.2), or the 2.2 MeV H-capture Compton edge.</p>
{_tab(beam, {'n': '{:.0f}', 'mpv_beam': '{:.1f}', 'mpv_err': '{:.1f}', 'mpv_cosmic': '{:.1f}'})}

<h2>C4 — the liquids</h2>
<img src="figures/c4_liquid_spectrum.png" alt="liquid spectrum" style="max-width:520px">
<p class="cap">Liquid amplitude of cosmic muons confirmed through the plastic in front (a
deposit of 0.6-2.5 MIP in the predicted bar, both wall ends). The spectrum is a Landau cut
off by the ZS threshold. <code>amp_0</code>/<code>amp</code> = 1.03, so it is not a
pulse-fit artefact.</p>
{_tab(LS, {'n': '{:.0f}', 'mpv_mv': '{:.1f}', 'mpv_err': '{:.1f}', 'sigma_mv': '{:.1f}'})}
<img src="figures/c4_profiles.png" alt="liquid profiles">
<p class="cap">Efficiency along u and v. A and D, with the PMT at +u, rise toward it, and so
does their amplitude (A: 25 → 42 mV): light attenuation in front of a threshold. C, with its
PMT on top, <i>falls</i> toward the top (14 % → 2 %), which attenuation cannot do. That
fits the plan's hypothesis of an under-fill or gas bubble on the PMT side, for C only.</p>
<img src="figures/c4_maps.png" alt="liquid maps">
{_tab(lq, {'n': '{:.0f}', 'eff': '{:.3f}', 'err': '{:.3f}', 'p_off': '{:.3f}', 'amp_med_mv': '{:.1f}'})}
<p><b>Penetration test.</b> Each beam-tagged event's expected liquid probability is the
cosmic efficiency of its 50 mm cell. Observed / expected:</p>
{_tab(PN, {'n': '{:.0f}', 'expected': '{:.1f}', 'observed': '{:.1f}', 'ratio': '{:.3f}', 'ratio_err': '{:.3f}'})}
<p class="cap">The beam liquid hits are in time: they peak at -30 to -10 ns in the
(-100, +60) ns window, and their amplitudes match the cosmic ones (medians 26-30 mV in every
run). A timing loss and a beam-only gain change are both ruled out.</p>

{m2_section(t)}

<h2>What this does not rule out</h2>
<ul>
<li><b>The bar thickness.</b> 20 mm PVT, per the Geant4 geometry
(<code>SimConfig.hh</code>: "2.0 cm PVT ... corrected 2026-07-20, was 2.5"). The run_config
descriptions still say 20&times;30&times;2.5 cm, and so did <code>pss_mip_calib</code>. At
25 mm the expected MIP is 4.30 MeV, not 3.41. The MIP-anchored mV/MeVee would then drop by
20 %, and the source line would read 23-40 % low instead of 3-24 %. The direction of every
conclusion is unchanged, but the absolute scale depends on this one number. A caliper on a
bar settles it.</li>
<li><b>The MIP anchor rests on a calculation.</b> The Bichsel thin-layer MPV is taken as
good to ~3 %, and Birks to ~2 % (both inside the quoted 4-7 %). The C2 Geant4 response
replaces the calculation, and should come out within that.</li>
<li><b>Nonlinearity between 0.7 and 3.4 MeV.</b> The path check covers 3.4-5 MeV. Below
the MIP, only the source edges constrain the response, and they disagree with each other.
The 3-5 MeV soft-leg range is covered; 1-3 MeV (the <sup>28</sup>Al and capture-Compton
region) is not.</li>
<li><b>Gain under beam and after the flash.</b> Unmeasured (see C1.4). The cosmic scale is
end-of-campaign beam-off. The scint-stack late-track medians move ~1 % run to run, which
bounds a drift, not an offset.</li>
<li><b>Arm A's MPV</b> depends on the threshold model more than C's or D's (fit systematic 3-6 %).</li>
<li><b>Arm B</b> has no MIP measurement: too few cosmics cross its plastic with a usable slope
(no <code>k</code> for B).</li>
<li><b>The liquids' mechanical state.</b> C's top-loss is consistent with a bubble, but a
dead region of the PMT photocathode or an optical defect would look the same.</li>
<li><b>What the beam "through-goers" are.</b> This shows they do not penetrate. Whether
they are conversion pairs, Compton pairs or chance alignments is open.</li>
</ul>

<h2>Plan status</h2>
<ul>
<li>C1.1 MIP check: <b>done</b>, scale replaced by the MIP anchor
(<code>c1/calib_plastic_e.json</code>). C1.3 saturation: <b>done</b> (irrelevant below
15 MeVee). C1.4 flash-time: <b>blocked</b>, needs an in-beam energy reference. C1.2
<sup>28</sup>Al endpoint and C1.5 face maps: open.</li>
<li>C4 liquid salvage: <b>done</b>. Liquids are threshold-limited (gain against ZS);
usable as a hard-leg tag on A u &ge; 0 (65-86 %) and D u &ge; 50 (50-70 %). C3 should size
that subset.</li>
<li>M1 raw road charge: <b>done</b> (<code>mm_charge.py</code>). M2 cosmic dE/dx: <b>done,
kill condition met</b> &mdash; the 2-MIP line stops here (M3 not run: its statistical best
case is already ~40 % at 10 % mis-tag). M4: only the gain-monitor use survives.</li>
<li>Next: C2 (Geant4 single-electron response, condor on lxplus), then C3.</li>
</ul>
</body></html>"""
    return out


def m2_section(t) -> str:
    if 'm2_summary' not in t:
        return ''
    R, T, S = t['m2_resolution'], t['m2_two_mip'], t['m2_summary']
    rA = R[(R.arm == 'A') & (R.estimator == 'q_whole')].iloc[0]
    rAt = R[(R.arm == 'A') & (R.estimator == 'q_trunc_t')].iloc[0]
    rC = R[(R.arm == 'C') & (R.estimator == 'q_plat')].iloc[0]
    e10 = T[(T.mistag == 0.1)].eff_2mip
    rr = R[['arm', 'estimator', 'n', 'mpv', 'fwhm_over_mpv', 'sigma_eq', 'tail_gt2', 'q16', 'q84']]
    tt = T[['arm', 'estimator', 'mistag', 'eff_2mip']]
    gs = S['gain_map_spread']
    return f"""<h2>M1-M2 &mdash; MM dE/dx on cosmics</h2>
<div class="verdict"><p><b>Verdict: the plan's kill condition is met.</b> The raw road
charge of one 30 mm gap has FWHM/MPV &asymp; {rA.fwhm_over_mpv:.2f} (&sigma;<sub>eq</sub>
{rA.sigma_eq * 100:.0f} %) on A for the whole gap, and {rAt.sigma_eq * 100:.0f} % for the
truncated mean over depth samples. Truncation does not help: shaping and the resistive
kernel correlate neighbouring samples. Even in the best case (two real tracks' charges
added, no reconstruction or ZS losses), a cut keeping 10 % of single muons flags only
<b>{e10.min() * 100:.0f}-{e10.max() * 100:.0f} %</b> of two-muon tracks. That is not a 2-MIP
flag, so the 2-MIP line stops: M3 is not run. What the charge <i>can</i> do is monitor
gain per 40 mm cell: the spread is {gs['A'] * 100:.0f} % on A and {gs['C'] * 100:.0f} % on C.</p></div>
<p><b>Estimator</b> (<code>mm_charge.py</code>): the sum over the strips on each track's own
corridor (from the waveform reco: p0 + depth &times; tan, depth -3 to 33 mm, &plusmn;5 mm),
over all 20 samples, of the pedestal-subtracted ADC. The common mode is taken per sample
from each 64-channel block's channels <i>outside</i> the road; the stock CNS would subtract
part of a steep track's own charge. An off-road control road of the same width averages
{S['ctrl']['A']['ctrl_med'] * 100:+.2f} % of the signal (IQR {S['ctrl']['A']['ctrl_iqr'] * 100:.1f} %).
Q<sub>x</sub>/Q<sub>y</sub> = {S['xy']['A']['qx_over_qy']:.2f}. Sample: 14 run_149
sub-runs with waveforms on disk, {S['n_sel']['A']:,} A and {S['n_sel']['C']:,} C tracks.
<b>Arm C drifts at 28 &micro;m/ns</b>, so 30 mm takes ~1.1 &micro;s and its deep charge
falls off the 1.2 &micro;s window. Its whole-gap Q is truncated on ~94 % of tracks, and only
the plateau estimator (per-sample charge inside the drift / depth per sample) covers it:
&sigma;<sub>eq</sub> {rC.sigma_eq * 100:.0f} %.</p>
<img src="figures/m2_landau.png" alt="dE/dx distributions">
{_tab(rr, {'n': '{:.0f}', 'mpv': '{:.0f}', 'fwhm_over_mpv': '{:.2f}', 'sigma_eq': '{:.2f}', 'tail_gt2': '{:.2f}', 'q16': '{:.2f}', 'q84': '{:.2f}'})}
{_tab(tt, {'eff_2mip': '{:.2f}', 'mistag': '{:.2f}'})}
<img src="figures/m2_depth.png" alt="charge vs depth" style="max-width:520px">
<p class="cap">Mean road charge per 60 ns sample against the drift depth that sample
corresponds to. Each depth bin includes every track whose window covers it. On both chambers
the shape is the drift integrated by the shaper; C (28 &micro;m/ns) peaks earlier in depth
than A (34 &micro;m/ns), and it falls no faster afterwards. So attachment in C's wet gas
(B/C/D took ~0.8 % H<sub>2</sub>O in July) is not visible at this precision. An earlier
version of this plot normalised each track and used only C's in-window tracks. That
minority subset (early t0) produced a ~3&times; fall with depth, which was a selection
artefact.</p>
<p>The cos&theta; scaling holds: the plateau charge per mm of depth grows as sec&theta;
(A: 1.06 &rarr; 1.24 against sec 1.06 &rarr; 1.24) and flattens only in the steepest bin.
A against C on the same muon: Spearman {S['same_muon']['spearman']:.2f}
(n = {S['same_muon']['n']}), small. The two chambers measure nearly independently.</p>
"""


def main() -> int:
    FS.use()
    t = _load()
    fig_ladder(t)
    fig_spectra(t)
    fig_path(t)
    fig_liquid(t)
    if 'm2_summary' in t:
        fig_m2(t)
    (OUT / 'report.html').write_text(report(t))
    print(f'wrote {OUT / "report.html"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
