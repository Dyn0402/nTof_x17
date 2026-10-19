#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_same_chamber_report.py -- same-chamber (intra) pairs in A and C, on the
is2_v2 re-pass, read against production.  Figures + tables + report.html.

Inputs are `det_a_intra` output directories, one per (chain, arm), all built
with the same cuts (``--t0-max 300 --axis measured --wall``; the re-pass also
``--tan-max-true 0.6``):

    <src>/prod_A  prod_C     stage3_fullpass  (production reco, production k)
    <src>/cf_A    cf_C       stage3_is2_v2_cf (re-pass, capsule-focus k = 1.12 x is2)
    <src>/is2_A   is2_C      stage3_is2_v2    (re-pass, in-beam-muon k)

cf is the headline (Dylan, 2026-10-12: adopt the capsule-focus k, keep the
muon-vs-capsule tension open); is2 is the other end of that tension and is
shown as the systematic on every angle.

The page leads with what changed: production's intra pairs are mostly not two
particles (their legs confirm on the SiPM wall at a fraction of the single-track
rate), the re-pass removes most of them, and the pairs whose BOTH legs confirm
are a small, prompt, real two-particle sample.

    python -m sept26_prelim_analysis.make_same_chamber_report \\
        --src ~/x17_local/same_chamber --out <out>/same_chamber
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sept26_prelim_analysis import figstyle as fs
from sept26_prelim_analysis import paths

CHAINS = {'prod': 'production', 'cf': 're-pass, capsule-focus k',
          'is2': 're-pass, muon k'}
ARMS = ('A', 'C')
#: windows as in det_a_intra (PROMPT_NS, OFFTIME_NS): |dt0| <= 75 and 150-700 ns
PROMPT_W = 2 * 75.0
OFF_W = 2 * (700.0 - 150.0)
OPEN_BINS = np.arange(0.0, 90.1, 7.5)
DT_BINS = np.arange(-800.0, 801.0, 40.0)
SEP_BINS = np.arange(0.0, 401.0, 20.0)

PAIR_COLS = ['run', 'subrun', 'event_id', 'mixed', 'open_deg', 'du_mm', 'dv_mm',
             'sep_plane_mm', 'dt0_ns', 'v_r_mm', 'dca_pair_mm', 'vertex',
             'both_slope', 'prompt', 'offtime', 'pointing', 'onwall', 'wall',
             'wallctrl', 'n_legs_wall', 'tan_min_pair', 'wall2', 'wall2ctrl',
             'diffgrp']


def load(src: Path) -> dict:
    """{(chain, arm): dict(P=pairs, H=headline, meta)} for what exists."""
    out = {}
    for ch in CHAINS:
        for arm in ARMS:
            d = src / f'{ch}_{arm}'
            if not (d / 'pairs.parquet').exists():
                continue
            P = pd.read_parquet(d / 'pairs.parquet', columns=PAIR_COLS)
            H = pd.read_csv(d / 'headline_per_run.csv')
            meta = json.loads((d / 'det_a_intra.meta.json').read_text())
            VX = pd.read_csv(d / 'vertex_excess.csv')
            out[(ch, arm)] = dict(P=P, H=H, meta=meta, VX=VX)
    return out


def _rate(k, n):
    return (k / n, np.sqrt(max(k, 1)) / n) if n else (np.nan, np.nan)


def headline(D: dict) -> pd.DataFrame:
    """One row per (chain, arm): the numbers the page leads with."""
    rows = []
    for (ch, arm), x in D.items():
        P, H = x['P'], x['H']
        R, M = P[~P.mixed], P[P.mixed]
        n_trk = int(H.n_tracks.sum())
        trk_wall = int(H.n_trk_wall.sum()) - int(H.n_trk_wallctrl.sum())
        leg_wall = float(R.n_legs_wall.sum()) / (2 * len(R)) if len(R) else np.nan
        W = R[R.wall2]
        n_p, n_o = int(W.prompt.sum()), int(W.offtime.sum())
        prompt_exc = n_p - n_o * PROMPT_W / OFF_W
        mw = float(M.wall2.mean()) if len(M) else np.nan
        vr = float(W.vertex.mean()) if len(W) else np.nan
        mvx = M[M.wall2]
        vm = float(mvx.vertex.mean()) if len(mvx) else np.nan
        vx = x['VX'].set_index('selection')
        va = vx.loc['all'] if 'all' in vx.index else None
        rows.append(dict(
            vx_all_ratio=float(va.ratio) if va is not None else np.nan,
            vx_all_err=float(va.err) if va is not None else np.nan,
            vx_all_sigma=float(va.excess_sigma) if va is not None else np.nan,
            vx_all_k=int(va.k_real) if va is not None else 0,
            vx_all_exp=float(va.rate_mixed * va.n_real) if va is not None else np.nan,
            chain=ch, arm=arm, runs=int(H.run.nunique()), tracks=n_trk,
            tracks_in_multi=int(H.n_tracks_multi.sum()),
            trk_wall_net=trk_wall / n_trk if n_trk else np.nan,
            pairs=len(R), pairs_per_1k_trk=1e3 * len(R) / n_trk if n_trk else np.nan,
            leg_wall=leg_wall,
            pairs_wall1=int(R.wall.sum()), pairs_wall1ctrl=int(R.wallctrl.sum()),
            pairs_wall=len(W), pairs_wallctrl=int(R.wall2ctrl.sum()),
            wall_dca_med=float(W.dca_pair_mm.median()) if len(W) else np.nan,
            wall_mixed_dca_med=float(mvx.dca_pair_mm.median()) if len(mvx) else np.nan,
            wall_frac=len(W) / len(R) if len(R) else np.nan,
            wall_frac_mixed=mw,
            wall_corr=(len(W) / len(R)) / mw if len(R) and mw else np.nan,
            wall_prompt=n_p, wall_offtime=n_o, wall_prompt_excess=prompt_exc,
            wall_prompt_purity=prompt_exc / n_p if n_p else np.nan,
            all_prompt=int(R.prompt.sum()), all_offtime=int(R.offtime.sum()),
            all_prompt_frac=float(R.prompt.mean()) if len(R) else np.nan,
            wall_prompt_frac=n_p / len(W) if len(W) else np.nan,
            wall_open_med=float(W.open_deg.median()) if len(W) else np.nan,
            wallp_open_med=float(W[W.prompt].open_deg.median()) if n_p else np.nan,
            wall_mixed_open_med=float(mvx.open_deg.median()) if len(mvx) else np.nan,
            wall_vertex=int(W.vertex.sum()), wall_vertex_rate=vr,
            wall_vertex_rate_mixed=vm,
            wall_vertex_ratio=vr / vm if vm else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
CH_STYLE = {'prod': dict(color=fs.MUTED, ls='--'),
            'cf': dict(color=fs.ACCENT, ls='-'),
            'is2': dict(color=fs.COPPER, ls=':')}


def fig_legs(T: pd.DataFrame, fd: Path):
    """Wall confirmation: every track vs the legs of same-chamber pairs."""
    fig, ax = fs.figure(fs.FIG)
    chains = [c for c in ('prod', 'cf') if c in set(T.chain)]
    x = np.arange(len(ARMS))
    w = 0.2
    rows = []
    for i, ch in enumerate(chains):
        for j, arm in enumerate(ARMS):
            r = T[(T.chain == ch) & (T.arm == arm)]
            if r.empty:
                continue
            r = r.iloc[0]
            off = (i - 0.5) * 2 * w
            ax.bar(j + off - w / 2, r.trk_wall_net, w, color=CH_STYLE[ch]['color'],
                   alpha=0.35, edgecolor=CH_STYLE[ch]['color'])
            ax.bar(j + off + w / 2, r.leg_wall, w, color=CH_STYLE[ch]['color'])
            rows.append(dict(chain=ch, arm=arm, all_tracks=r.trk_wall_net,
                             pair_legs=r.leg_wall))
    ax.set_xticks(x, [f'chamber {a}' for a in ARMS])
    ax.set_ylabel('fraction confirmed on the SiPM wall')
    from matplotlib.patches import Patch
    h = []
    for ch in chains:
        c = CH_STYLE[ch]['color']
        h += [Patch(facecolor=c, alpha=0.35, edgecolor=c, label=f'{CHAINS[ch]}: all tracks'),
              Patch(facecolor=c, label=f'{CHAINS[ch]}: pair legs')]
    ax.legend(handles=h, frameon=False, fontsize=fs.BASE_PT * 0.85, loc='upper right')
    ax.set_ylim(0, max(0.75, ax.get_ylim()[1]))
    fs.title(ax, 'Pair legs confirm far less often than tracks at large',
             'all tracks: net of the off-time control; pair legs: gross (control ~1 %)')
    fs.preliminary(ax, 'upper left')
    return fs.save(fig, fd / 'leg_confirmation', data=pd.DataFrame(rows))


def fig_dt0(D: dict, fd: Path, chain='cf'):
    """In-chamber Δt0: all pairs vs both-legs-confirmed pairs."""
    fig, axs = fs.figure(fs.WIDE, ncols=2, sharey=False)
    mid = 0.5 * (DT_BINS[:-1] + DT_BINS[1:])
    data = {}
    for ax, arm in zip(axs, ARMS):
        if (chain, arm) not in D:
            continue
        R = D[(chain, arm)]['P']
        R = R[~R.mixed]
        for lab, sel, st in (('all pairs', R, dict(color=fs.MUTED, lw=1.2)),
                             ('two particles on the wall', R[R.wall2],
                              dict(color=fs.DET_COLOR[arm], lw=1.8))):
            h, _ = np.histogram(sel.dt0_ns.dropna(), DT_BINS)
            f = h / max(h.sum(), 1)
            ax.step(mid, f, where='mid', label=f'{lab} ({h.sum():,})', **st)
            data[f'{arm}_{lab}'] = h
        ax.axvspan(-75, 75, color=fs.GRID, zorder=0)
        ax.set_xlabel('Δt0 between the legs (ns)')
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT)
        ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.8)
        fs.preliminary(ax)
    axs[0].set_ylabel('fraction of pairs / 40 ns')
    fs.fig_title(fig, 'The legs share a prompt core, certified pairs or not',
                 f'{CHAINS[chain]}; shaded = the ±75 ns prompt window')
    return fs.save(fig, fd / f'dt0_{chain}',
                   data=pd.DataFrame(dict(dt0_ns=mid, **data)))


def fig_open(D: dict, fd: Path):
    """Opening angle of confirmed prompt pairs vs the event-mixed null; k bracket."""
    fig, axs = fs.figure(fs.WIDE, ncols=2, sharey=True)
    mid = 0.5 * (OPEN_BINS[:-1] + OPEN_BINS[1:])
    data = {}
    for ax, arm in zip(axs, ARMS):
        for ch in ('cf', 'is2'):
            if (ch, arm) not in D:
                continue
            P = D[(ch, arm)]['P']
            R = P[~P.mixed & P.wall2]
            h, _ = np.histogram(R.open_deg, OPEN_BINS)
            f = h / max(h.sum(), 1)
            e = np.sqrt(np.maximum(h, 1)) / max(h.sum(), 1)
            st = CH_STYLE[ch]
            if ch == 'cf':
                ax.errorbar(mid, f, yerr=[np.minimum(e, f), e], fmt='o', ms=4,
                            color=fs.DET_COLOR[arm],
                            label=f'two-particle pairs ({h.sum():,})')
                M = P[P.mixed & P.wall2]
                hm, _ = np.histogram(M.open_deg, OPEN_BINS)
                ax.step(mid, hm / max(hm.sum(), 1), where='mid', color=fs.MUTED,
                        lw=1.2, label='event-mixed, same selection')
                data[f'{arm}_mixed'] = hm
            else:
                ax.step(mid, f, where='mid', color=st['color'], ls=st['ls'], lw=1.4,
                        label=f'same, muon k ({h.sum():,})')
            data[f'{arm}_{ch}'] = h
        ax.set_xlabel('opening angle (deg)')
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT)
        ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.8)
        fs.preliminary(ax, 'lower right')
    axs[0].set_ylabel(f'fraction of pairs / {OPEN_BINS[1]:.1f}°')
    fs.fig_title(fig, 'Opening angle of certified two-particle pairs',
                 'capsule-focus k (points) against the muon k (dotted) and the event-mixed null')
    return fs.save(fig, fd / 'opening_angle', data=pd.DataFrame(dict(open_deg=mid, **data)))


def fig_vertex(D: dict, fd: Path):
    """How often the two lines meet on the axis, real over event-mixed."""
    fig, ax = fs.figure(fs.FIG)
    rows = []
    sels = ('all', 'wall2')
    labels = {'all': 'all pairs', 'wall2': 'two particles'}
    chains = [c for c in ('prod', 'cf', 'is2') if any(k[0] == c for k in D)]
    xt, xl = [], []
    i = 0
    for arm in ARMS:
        for sel in sels:
            for j, ch in enumerate(chains):
                if (ch, arm) not in D:
                    continue
                v = D[(ch, arm)]['VX'].set_index('selection')
                if sel not in v.index:
                    continue
                r = v.loc[sel]
                x = i + (j - (len(chains) - 1) / 2) * 0.22
                st = CH_STYLE[ch]
                YMAX = 5.0
                if r.ratio > YMAX:
                    ax.annotate(f'{r.ratio:.1f} ± {r.err:.1f}', (x, YMAX), xytext=(-9, -1),
                                textcoords='offset points', ha='right', va='top',
                                fontsize=fs.BASE_PT * 0.75, color=st['color'],
                                arrowprops=None)
                    ax.plot(x, YMAX * 0.985, marker='^', ms=6, color=st['color'],
                            mfc='white', mew=1.4, ls='')
                else:
                    ax.errorbar(x, r.ratio, yerr=r.err, fmt=fs.DET_MARKER[arm], ms=6,
                                color=st['color'], mec=st['color'], mew=1.4,
                                mfc=st['color'] if ch != 'prod' else 'white')
                rows.append(dict(arm=arm, selection=sel, chain=ch, k_real=int(r.k_real),
                                 n_real=int(r.n_real), k_mixed=int(r.k_mixed),
                                 n_mixed=int(r.n_mixed), ratio=r.ratio, err=r.err,
                                 sigma=r.excess_sigma))
            xt.append(i)
            xl.append(f'{arm}: {labels[sel]}')
            i += 1
    ax.axhline(1, color=fs.LINE, lw=1)
    ax.set_xticks(xt, xl)
    ax.set_ylabel('vertex rate, real / event-mixed')
    ax.set_ylim(0, 5.0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=CH_STYLE[c]['color'], marker='o', ls='',
                              mec=CH_STYLE[c]['color'], mew=1.4,
                              mfc=CH_STYLE[c]['color'] if c != 'prod' else 'white',
                              label=CHAINS[c]) for c in chains],
              frameon=False, fontsize=fs.BASE_PT * 0.85, loc='upper left')
    fs.title(ax, 'On the re-pass, same-chamber pairs meet on the axis 2x more than chance',
             'lines within 30 mm of each other and 20 mm of the measured axis; ratio to tracks from different triggers')
    fs.preliminary(ax, 'lower right')
    return fs.save(fig, fd / 'vertex_excess', data=pd.DataFrame(rows))


def fig_sep(D: dict, fd: Path, chain='cf'):
    """Separation in the two views: real/mixed, where the reco merges two tracks."""
    fig, axs = fs.figure(fs.WIDE, ncols=2, sharey=True)
    mid = 0.5 * (SEP_BINS[:-1] + SEP_BINS[1:])
    data = {}
    for ax, arm in zip(axs, ARMS):
        if (chain, arm) not in D:
            continue
        P = D[(chain, arm)]['P']
        for view, col, ls in (('u (x strips)', 'du_mm', '-'), ('v (y strips)', 'dv_mm', '--')):
            r, _ = np.histogram(P.loc[~P.mixed, col].abs(), SEP_BINS)
            m, _ = np.histogram(P.loc[P.mixed, col].abs(), SEP_BINS)
            with np.errstate(divide='ignore', invalid='ignore'):
                q = (r / r.sum()) / (m / m.sum())
            ax.plot(mid, q, ls=ls, color=fs.DET_COLOR[arm], marker=fs.DET_MARKER[arm],
                    ms=3.5, lw=1.4, label=view)
            data[f'{arm}_{col}_real'], data[f'{arm}_{col}_mixed'] = r, m
        ax.axhline(1, color=fs.LINE, lw=0.8)
        ax.set_xlabel('separation of the two impact points in one view (mm)')
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT)
        ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.8)
        fs.preliminary(ax, 'lower right')
    axs[0].set_ylabel('real / event-mixed (shape)')
    fs.fig_title(fig, 'Below ~20 mm in either view the reconstruction sees one track',
                 f'{CHAINS[chain]}, all pairs; the joint two-track fit (off in this pass) targets this')
    return fs.save(fig, fd / f'separation_{chain}', data=pd.DataFrame(dict(sep_mm=mid, **data)))


MODEL_STYLE = {'Al conversion': dict(color=fs.ACCENT, ls='-'),
               'Al internal': dict(color=fs.COPPER, ls='--'),
               'independent': dict(color=fs.MUTED, ls='-.'),
               'Al conversion, no scattering': dict(color=fs.TRACK, ls=':')}
MODEL_LABEL = {'Al conversion': '27Al γ conversion (Geant4, after scattering)',
               'Al internal': '27Al internal pairs (after the wall)',
               'independent': 'two leptons, unrelated directions',
               'Al conversion, no scattering': '27Al conversion, no scattering'}


def fig_physics_y(Y: pd.DataFrame, fd: Path):
    """Where along the beam the common-vertex pairs sit."""
    from sept26_prelim_analysis.same_chamber_physics import CAPSULE_V
    fig, axs = fs.figure(fs.WIDE, ncols=2, sharey=False)
    for ax, arm in zip(axs, ARMS):
        g = Y[(Y.arm == arm) & (Y.chain == 'cf')]
        ax.axvspan(*CAPSULE_V, color=fs.GRID, zorder=0, label='capsule (measured height)')
        ax.step(g.mid, g.expected, where='mid', color=fs.MUTED, lw=1.3,
                label='chance (event-mixed)')
        ax.errorbar(g.mid, g.real, yerr=np.sqrt(g.real), fmt=fs.DET_MARKER[arm], ms=4.5,
                    color=fs.DET_COLOR[arm], label='real pairs')
        ax.set_xlabel('vertex position along the beam, y (mm)')
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT)
        ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.78, loc='upper left')
        fs.preliminary(ax)
    axs[0].set_ylabel('vertexed pairs / 25 mm')
    fs.fig_title(fig, 'The excess common-vertex pairs come from the capsule',
                 'pairs whose two lines meet within 20 mm of the axis; re-pass, capsule-focus k')
    return fs.save(fig, fd / 'physics_vertex_y', data=Y)


def fig_physics_open(E: pd.DataFrame, F: pd.DataFrame, C: pd.DataFrame, fd: Path):
    """The excess opening angle against the folded source models."""
    fig, axs = fs.figure(fs.WIDE, ncols=2, sharey=True)
    for ax, arm in zip(axs, ARMS):
        e = E[(E.arm == arm) & (E.chain == 'cf')]
        tot = e.excess.sum()
        ax.errorbar(e.mid, e.excess / tot, yerr=e.err / tot, fmt=fs.DET_MARKER[arm], ms=5,
                    color=fs.DET_COLOR[arm], label=f'data excess ({tot:.0f} pairs)', zorder=5)
        for model, st in MODEL_STYLE.items():
            f = F[(F.arm == arm) & (F.model == model)]
            v = f.vertexed.to_numpy()
            c = C[(C.arm == arm) & (C.chain == 'cf') & (C.model == model)]
            ax.step(f.mid, v / v.sum(), where='mid', lw=1.5, **st,
                    label=MODEL_LABEL[model])
            if len(c):
                ax.text(0.98, 0.95 - 0.07 * list(MODEL_STYLE).index(model),
                        f'χ² {c.chi2.iloc[0]:.1f}/{c.ndf.iloc[0]}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=fs.BASE_PT * 0.8, color=st['color'])
        ax.axhline(0, color=fs.LINE, lw=0.8)
        ax.set_xlabel('opening angle (deg)')
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT)
        ax.legend(handles=[h for h in ax.get_legend_handles_labels()[0][-1:]],
                  frameon=False, fontsize=fs.BASE_PT * 0.8, loc='upper left')
        fs.preliminary(ax, 'lower right')
    hs, ls_ = axs[0].get_legend_handles_labels()
    fig.legend(hs[:-1], ls_[:-1], loc='lower center', ncol=2, frameon=False,
               fontsize=fs.BASE_PT * 0.82, bbox_to_anchor=(0.5, -0.1))
    axs[0].set_ylabel('fraction / 10°')
    fs.fig_title(fig, 'Scattered Al pairs and unrelated lepton pairs both fit; unscattered pairs do not',
                 'excess = vertexed real − chance, per angle bin; models folded through acceptance, '
                 'two-track resolution, angular resolution and the vertex test')
    return fs.save(fig, fd / 'physics_opening_angle', data=E)


def fig_tof(R: pd.DataFrame, F: pd.DataFrame, fd: Path):
    """Fraction of tracked events with a second same-chamber track, vs time since the flash."""
    fig, axs = fs.figure(fs.WIDE, ncols=2, sharey=True)
    for ax, arm in zip(axs, ARMS):
        r = R[R.arm == arm]
        x = 0.5 * (r.lo + r.hi)
        ax.errorbar(x, r.frac * 100, yerr=r.err * 100, fmt=fs.DET_MARKER[arm], ms=4,
                    color=fs.DET_COLOR[arm], label='data')
        f = F[F.arm == arm].iloc[0]
        late = r[r.lo >= f.late_ms]
        ax.plot(0.5 * (late.lo + late.hi), (f.a + f.b * late.rho) * 100, color=fs.INK, lw=1.3,
                label='fit a + b·rate (12–75 ms)')
        ax.axhline(f.a * 100, color=fs.ACCENT, lw=1.3, ls='--',
                   label=f'rate-independent part a = {100 * f.a:.3f} ± {100 * f.a_err:.3f} %')
        ax.axvspan(0, f.late_ms, color=fs.GRID, zorder=0)
        ax.set_yscale('log')
        ax.set_xlabel('time since the γ flash (ms)')
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT)
        ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.75, loc='upper right')
        fs.preliminary(ax, 'lower left')
    axs[0].set_ylabel('events with ≥ 2 tracks (%)')
    fs.fig_title(fig, 'After 12 ms the pair fraction flattens: same-capture pairs, not pile-up',
                 'shaded: flash and epithermal region, excluded; re-pass, capsule-focus k')
    return fs.save(fig, fd / 'cmp_tof_fraction', data=R)


def fig_g4_classes(T: pd.DataFrame, fd: Path):
    """Geant4: what reconstructable same-chamber pairs are."""
    fig, ax = fs.figure(fs.FIG)
    t = T[T.cls != '(singles in acceptance)']
    cls = (t.groupby('cls').acc.sum().sort_values().index.tolist())
    y = np.arange(len(cls))
    for k, arm in enumerate(ARMS):
        tt = t[t.arm == arm].set_index('cls').reindex(cls)
        share = tt.acc / tt.acc.sum()
        ax.barh(y + (k - 0.5) * 0.38, share.fillna(0), 0.36, color=fs.DET_COLOR[arm],
                label=f'chamber {arm}')
        for yi, v in zip(y, share.fillna(0)):
            if v > 0.02:
                ax.text(v + 0.01, yi + (k - 0.5) * 0.38, f'{100 * v:.0f} %', va='center',
                        fontsize=fs.BASE_PT * 0.75, color=fs.MUTED)
    ax.set_yticks(y, cls)
    ax.set_xlabel('share of reconstructable same-chamber pairs (triggered, after acceptance)')
    ax.set_xlim(0, 1.0)
    ax.legend(frameon=False, loc='lower right')
    fs.title(ax, 'Geant4: four in five same-chamber pairs are one photon\'s e⁺e⁻',
             '10⁹ thermal neutrons, nose-first capsule; ²⁷Al(n,γ) is 96 % of the triggers')
    fs.preliminary(ax, 'upper right')
    return fs.save(fig, fd / 'cmp_g4_classes', data=t)


def fig_expect(X: pd.DataFrame, F: pd.DataFrame, fd: Path):
    """Expected vs measured rate-independent pair fraction."""
    fig, axs = fs.figure((10.5, 3.8), ncols=2, sharey=False)
    for ax, arm in zip(axs, ARMS):
        x = X[X.arm == arm].reset_index(drop=True)
        lab = [f'{r.eps_source} (ε {r.eps:.2f}), {r.two_track}' for r in x.itertuples()]
        y = np.arange(len(x))
        ax.barh(y, x.predicted_frac * 100, 0.6, color=fs.GRID, edgecolor=fs.LINE)
        f = F[F.arm == arm].iloc[0]
        ax.axvspan((f.a - f.a_err) * 100, (f.a + f.a_err) * 100, color=fs.DET_COLOR[arm], alpha=0.25)
        ax.axvline(f.a * 100, color=fs.DET_COLOR[arm], lw=2, label='data (rate-independent part)')
        lab = [('digitised Geant4 through the reco' if r.eps_source.startswith('digitised') else
                f'ε {r.eps:.2f} ({r.eps_source.split(",")[0]}), '
                + ('bench recovery' if r.f_two < 1 else 'no two-track loss')) for r in x.itertuples()]
        ax.set_yticks(y, lab, fontsize=fs.BASE_PT * 0.72)
        ax.set_xlabel('pairs per tracked event (%)')
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT)
        ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.8, loc='lower right')
        fs.preliminary(ax, 'upper right')
    fs.fig_title(fig, 'Data has ~1/8 of the Geant4 prediction through the real reconstruction',
                 'grey: Geant4 (e⁺e⁻-dominated) under each efficiency assumption; band: data ± 1σ')
    return fs.save(fig, fd / 'cmp_expectation', data=X)


def fig_cmp_open(O: pd.DataFrame, C: pd.DataFrame, fd: Path):
    fig, axs = fs.figure(fs.WIDE, ncols=2, sharey=True)
    for ax, arm in zip(axs, ARMS):
        o = O[(O.arm == arm) & (O.chain == 'cf')]
        k = o.correlated.to_numpy(); tot = k.sum()
        ax.errorbar(o.mid, k / tot, yerr=o.err / tot, fmt=fs.DET_MARKER[arm], ms=5,
                    color=fs.DET_COLOR[arm], label=f'data, correlated ({tot:.0f} pairs)', zorder=5)
        for model, st, lab in (('g4_ee', dict(color=fs.ACCENT, ls='-'), 'Geant4 e⁺e⁻ (one γ)'),
                               ('toy_conv', dict(color=fs.COPPER, ls='--'), 'Geant4 Al conversion angle, toy acceptance'),
                               ('mixed', dict(color=fs.MUTED, ls='-.'), 'uncorrelated (event-mixed)')):
            m = o[model].to_numpy(float)
            c = C[(C.arm == arm) & (C.chain == 'cf') & (C.model == model)]
            chi = f' — χ² {c.chi2.iloc[0]:.1f}/{c.ndf.iloc[0]}' if len(c) else ''
            ax.step(o.mid, m / m.sum(), where='mid', lw=1.6, **st, label=lab + chi)
            if model == 'g4_ee':
                e = np.sqrt(o.g4_ee_var.to_numpy(float)) / m.sum()
                ax.fill_between(o.mid, m / m.sum() - e, m / m.sum() + e, step='mid',
                                color=fs.ACCENT, alpha=0.15, lw=0)
        ax.axhline(0, color=fs.LINE, lw=0.8)
        ax.set_xlabel('opening angle (deg)')
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT)
        ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.7, loc='upper right')
        fs.preliminary(ax, 'lower right')
    axs[0].set_ylabel('fraction / 10°')
    fs.fig_title(fig, 'In A the correlated pairs open like Geant4\'s e⁺e⁻, not like unrelated tracks',
                 'correlated = late real pairs − accidental share × late event-mixed shape; shapes only, Geant4 band = its statistics')
    return fs.save(fig, fd / 'cmp_opening', data=O)


def physics_section(out: Path, fd: Path) -> str:
    need = ['cmp_tof_fraction', 'cmp_tof_fit', 'cmp_g4_classes', 'cmp_expectation',
            'cmp_late_vertex', 'cmp_opening', 'cmp_opening_chi2', 'cmp_wall_double_ratio']
    if not all((out / f'{n}.csv').exists() for n in need):
        return ''
    R, F, T, X, V, O, C, W = (pd.read_csv(out / f'{n}.csv') for n in need)
    figs = [(fig_tof(R, F, fd), 'Pair fraction against time since the flash',
             'The fraction of tracked events with a second track in the same chamber. Before ~10 ms it is '
             'dominated by flash and epithermal junk. After 12 ms it is nearly flat while the event rate '
             'falls 3.5×: pairs from one capture arrive with their own trigger and are a constant fraction, '
             'two unrelated captures in one readout window scale with the rate. The fit separates the two.'),
            (fig_g4_classes(T, fd), 'What Geant4 says the pairs are',
             'Every two leptons of one simulated event crossing ≥ 20 mm of one drift gap, in triggered '
             'events, weighted by the measured acceptance (|tan| < 0.6, resolution, two-track efficiency). '
             'Two Compton electrons of two photons, the alternative that fitted as well as e⁺e⁻ yesterday, '
             'are 3–9 %. Internal pairs (not in Geant4) add ~18 % on top of the conversions.'),
            (fig_expect(X, F, fd), 'How many we should have seen',
             'Geant4 pairs per single track in acceptance, times the per-track efficiency ε (three '
             'estimates) and the two-track recovery of the overlay bench (47 % A / 39 % C), against the '
             'measured rate-independent fraction.'),
            (fig_cmp_open(O, C, fd), 'Opening angle of the correlated pairs',
             'Data: late (≥ 12 ms) real pairs minus the fitted accidental share (38 % A / 37 % C) with '
             'the late event-mixed shape. Geant4: one-photon e⁺e⁻ pairs through the same reconstruction '
             'response (band: Geant4 statistics, ~80 effective pairs per arm). χ² includes both errors.')]
    fA, fC = F[F.arm == 'A'].iloc[0], F[F.arm == 'C'].iloc[0]
    def xr(arm, src, two):
        r = X[(X.arm == arm) & (X.eps_source == src) & (X.two_track == two)]
        return r.iloc[0]
    bA = xr('A', 'digitised Geant4 through the reconstruction', 'measured (digitised)')
    bC = xr('C', 'digitised Geant4 through the reconstruction', 'measured (digitised)')
    rA, rC = X[X.arm == 'A'].data_over_g4, X[X.arm == 'C'].data_over_g4
    def chi(arm, model):
        r = C[(C.arm == arm) & (C.chain == 'cf') & (C.model == model)].iloc[0]
        return f'{r.chi2:.1f}/{r.ndf}'
    vA = V[(V.arm == 'A') & (V.chain == 'cf')].iloc[0]
    vC = V[(V.arm == 'C') & (V.chain == 'cf')].iloc[0]
    wA = W[(W.arm == 'A') & (W.chain == 'cf')].iloc[0]
    body = f"""<h2>Physics: did we see the aluminium e⁺e⁻ pairs?</h2>
<p class="verdict"><b>Same-capture pairs are there, with Geant4's e⁺e⁻ opening-angle shape in A, but at about one eighth of the rate Geant4 predicts.</b>
After 12 ms the fraction of tracked events that hold a second track in the same chamber has a part that does not depend on the event rate:
<b>{100 * fA.a:.3f} ± {100 * fA.a_err:.3f} % in A and {100 * fC.a:.3f} ± {100 * fC.a_err:.3f} % in C</b>, about {fA.correlated_pairs:,.0f} and {fC.correlated_pairs:,.0f} same-capture pairs.
The two chambers agree.
Geant4 says four in five reconstructable same-chamber pairs from a capture are the e⁺e⁻ of one ²⁷Al capture photon converting in the capsule wall.
Geant4 events digitised and put through the real reconstruction measure the pair response directly: P(2 tracks | 2 leptons) = 0.16 (A) / 0.14 (C) against P(≥ 1 track | 1 lepton) = 0.54 / 0.45.
With that response Geant4 predicts {100 * bA.predicted_frac:.2f} % (A) and {100 * bC.predicted_frac:.2f} % (C): <b>data/Geant4 = {bA.data_over_g4:.2f} and {bC.data_over_g4:.2f}</b>.
The cruder efficiency brackets give {rA.min():.2f}–{rA.max():.2f} (A) and {rC.min():.2f}–{rC.max():.2f} (C).
Even data's total late pair fraction, accidentals included (0.26 %), is ~4.6× below the digitised prediction, so the deficit does not hinge on the rate fit.
In A the correlated pairs' opening angle prefers Geant4's e⁺e⁻ (χ² {chi('A', 'g4_ee')}) over unrelated tracks (χ² {chi('A', 'mixed')}). The difference sits mostly in the 0–10° and 40–50° bins, so it is a preference, not a separation. In C neither shape fits well (χ² {chi('C', 'g4_ee')} and {chi('C', 'mixed')}).</p>
<p><b>Correction to yesterday's vertex result.</b> The ×2.14 (A) / ×1.67 (C) "common vertex on the axis" excess does not survive restricting to late tracks with a late-only event-mixed null. 77 % of A's pairs, and so most of its mixing pool, came from the first 12 ms. Late only: A ×{vA.ratio:.2f} ± {vA.ratio_err:.2f}, C ×{vC.ratio:.2f} ± {vC.ratio_err:.2f}. Geant4 explains why the vertex test is weak: the leptons scatter in the capsule wall, and a real e⁺e⁻ pair passes the test only 1.4–5 % of the time, barely above the chance rate of 2–3 %. The vertex figures above and the toy fits of the first physics pass rest on the early-dominated null; read them as superseded.</p>
<p><b>Energy, through the wall.</b> P(both legs confirmed | correlated pair) / P(confirmed | single)² is {wA.data_double_ratio:.2f} in A against Geant4's {wA.g4_double_ratio:.2f} for e⁺e⁻. But event-mixed pairs give a similar value, so this is not a discriminant on this sample.</p>
<p><b>What the factor ~8 could be — open.</b> Known effects make it smaller only modestly. The digitised sample used the one-sided search, not is2_v2's two-sided. The data's t0 ≤ 300 ns cut costs a pair ~0.8 more than a single. Internal pairs (not in Geant4) would make the prediction ~18 % <i>larger</i>. Candidates for the rest: Geant4's conversion yield, which scales with the capsule-wall material and thickness in the nose (unvalidated against the real capsule); a trigger or readout effect specific to two-lepton events; or a reconstruction loss the quiet-overlay digitisation does not reproduce in real busy events. Not resolved here.</p>
"""
    body += ''.join(f'<figure><img src="figures/{p.name}" alt="{html.escape(t)}">'
                    f'<figcaption><b>{html.escape(t)}.</b> {html.escape(c)}</figcaption></figure>'
                    for p, t, c in figs)
    return body


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
def _f(v, fmt):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '–'
    return fmt.format(v)


def table(T: pd.DataFrame, cols, heads, fmts) -> str:
    h = ''.join(f'<th>{html.escape(x)}</th>' for x in heads)
    body = ''
    for r in T.itertuples():
        body += '<tr>' + ''.join(
            f'<td>{_f(getattr(r, c), f) if f else html.escape(str(getattr(r, c)))}</td>'
            for c, f in zip(cols, fmts)) + '</tr>'
    return f'<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def build(src: Path, out: Path) -> Path:
    fs.use()
    D = load(src)
    if not D:
        raise SystemExit(f'no det_a_intra outputs under {src}')
    fd = out / 'figures'
    T = headline(D)
    T['chain_name'] = T.chain.map(CHAINS)
    order = {c: i for i, c in enumerate(CHAINS)}
    T = T.sort_values(['arm', 'chain'], key=lambda s: s.map(order) if s.name == 'chain' else s)
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / 'headline.csv', index=False)
    figs = []
    figs.append((fig_legs(T, fd), 'Wall confirmation, tracks vs pair legs',
                 'Light bars: the fraction of all gated tracks whose extrapolation '
                 'lands on a SiPM-wall group that fired in time, net of the off-time '
                 'control. Solid bars: the same for the legs of same-chamber pairs. '
                 'A leg that is a real particle should confirm about as often as any '
                 'track. Production\'s legs confirm at a fraction of that rate, so most '
                 'of production\'s same-chamber pairs are not two particles.'))
    figs.append((fig_dt0(D, fd, 'cf'), 'In-chamber coincidence',
                 'Δt0 is the difference of the two legs\' fitted mesh arrival times, '
                 'a coincidence test that needs no scintillator. All pairs (grey) '
                 'and two-particle pairs (both legs on different, fired wall groups) '
                 'share the same prompt core once the late class is cut. So the '
                 'in-chamber clock does not separate real pairs from the rest. '
                 'Its accidental pedestal is not flat, and the prompt fraction is '
                 'not identifiable from the shape (det_a_intra.dt0_models).'))
    figs.append((fig_vertex(D, fd), 'Common vertex on the axis',
                 'The fraction of pairs whose two reconstructed lines pass within 30 mm '
                 'of each other at a point within 20 mm of the measured beam axis, over '
                 'the same fraction for event-mixed pairs (tracks from different '
                 'triggers, which share no vertex). Production: open markers; it shows '
                 'no excess on its full pair sample, which junk dominates, and its '
                 'two-particle ratios rest on 13 and 11 vertexed pairs (arrows: off '
                 'scale, value printed). The re-pass '
                 'shows one in both chambers and both selections. None of the vertexed '
                 'real pairs are duplicates (no legs within 5 mm in either view, none '
                 'under 5 deg). The excess grows with opening angle (A: x1.6 at 10-25 '
                 'deg, x3.3 above 40 deg), as a real common vertex should.'))
    figs.append((fig_open(D, fd), 'Opening angle',
                 'Two-particle pairs with the capsule-focus k (points, Poisson '
                 'errors) and with the in-beam-muon k (dotted; angles 12 % shallower). '
                 'The grey step is the event-mixed null with the same wall '
                 'requirement: it carries the acceptance (two legs at least one 100 mm '
                 'wall group apart) and no pair physics. Shapes only.'))
    figs.append((fig_sep(D, fd, 'cf'), 'Two-track separation',
                 'Real over event-mixed pair density against the separation in each '
                 'view, shape-normalised, all pairs. The deficit below ~20 mm is the '
                 'reconstruction merging two tracks that share a strip band in one '
                 'view. This pass ran with the two-track fixes switched off.'))

    def row(ch, arm):
        r = T[(T.chain == ch) & (T.arm == arm)]
        return r.iloc[0] if len(r) else None

    cfA, cfC, pA, pC = row('cf', 'A'), row('cf', 'C'), row('prod', 'A'), row('prod', 'C')

    def g(r, c, fmt):
        return _f(getattr(r, c), fmt) if r is not None else '–'

    verdict = (
        f'<p class="verdict"><b>Same-chamber pairs on the re-pass, at the capsule-focus k.</b> '
        f'Most of what production called a same-chamber pair was not two particles. Production '
        f'has {g(pA,"pairs","{:,}")} (A) / {g(pC,"pairs","{:,}")} (C) pairs, and their legs confirm '
        f'on the SiPM wall {g(pA,"leg_wall","{:.0%}")} / {g(pC,"leg_wall","{:.0%}")} of the time, '
        f'against {g(pA,"trk_wall_net","{:.0%}")} / {g(pC,"trk_wall_net","{:.0%}")} for its tracks '
        f'in general. The re-pass keeps {g(cfA,"pairs","{:,}")} / {g(cfC,"pairs","{:,}")}. '
        f'<b>Pairs certified as two particles</b> (legs on two different wall groups, both fired) '
        f'number about the same in both chains: {g(cfA,"pairs_wall","{:,}")} / '
        f'{g(cfC,"pairs_wall","{:,}")} in the re-pass ({g(cfA,"runs","{:d}")} / {g(cfC,"runs","{:d}")} runs) '
        f'against {g(pA,"pairs_wall","{:,}")} / {g(pC,"pairs_wall","{:,}")} in production '
        f'({g(pA,"runs","{:d}")} / {g(pC,"runs","{:d}")} runs: production has no k for the rest). '
        f'So the re-pass removes junk and keeps the real pairs. Their off-time accidental count is {g(cfA,"pairs_wallctrl","{:,}")} / '
        f'{g(cfC,"pairs_wallctrl","{:,}")}. Their median opening angle at the capsule-focus k is '
        f'{g(cfA,"wall_open_med","{:.0f}")}° (A) / {g(cfC,"wall_open_med","{:.0f}")}° (C). '
        f'<b>Correction (2026-10-13):</b> the common-vertex excess quoted here first (×{g(cfA,"vx_all_ratio","{:.2f}")} A) '
        f'comes from an event-mixed null dominated by the first 12 ms after the flash and does not survive a late-only analysis. '
        f'See the physics section.</p>')

    t1 = table(T, ['chain_name', 'arm', 'runs', 'tracks', 'pairs', 'pairs_per_1k_trk',
                   'trk_wall_net', 'leg_wall', 'pairs_wall1', 'pairs_wall1ctrl',
                   'pairs_wall', 'pairs_wallctrl', 'wall_corr'],
               ['chain', 'arm', 'runs', 'tracks', 'pairs', 'pairs / 1k tracks',
                'tracks confirmed (net)', 'pair legs confirmed', 'both legs confirmed',
                'its off-time control', 'two particles', 'its off-time control',
                'two particles, real / mixed rate'],
               [None, None, '{:d}', '{:,}', '{:,}', '{:.1f}', '{:.1%}', '{:.1%}',
                '{:,}', '{:,}', '{:,}', '{:,}', '{:.2f}'])
    t2 = table(T, ['chain_name', 'arm', 'pairs_wall', 'wall_prompt_frac',
                   'all_prompt_frac', 'wall_open_med',
                   'wallp_open_med', 'wall_mixed_open_med', 'wall_dca_med',
                   'wall_mixed_dca_med', 'wall_vertex', 'wall_vertex_ratio'],
               ['chain', 'arm', 'two-particle pairs', 'prompt (±75 ns)',
                'prompt, all pairs', 'median open, all', 'median open, prompt', 'median open, mixed null',
                'median line DCA', 'mixed null DCA', 'vertexed',
                'vertex rate real / mixed'],
               [None, None, '{:,}', '{:.0%}', '{:.0%}', '{:.1f}°',
                '{:.1f}°', '{:.1f}°', '{:.0f} mm', '{:.0f} mm', '{:,}', '{:.2f}'])
    phys_html = physics_section(out, fd)
    fig_html = ''.join(
        f'<figure><img src="figures/{p.name}" alt="{html.escape(t)}">'
        f'<figcaption><b>{html.escape(t)}.</b> {html.escape(c)}</figcaption></figure>'
        for p, t, c in figs)
    m0 = next(iter(D.values()))['meta']
    prov = {f'{ch}_{arm}': dict(src=x['meta']['src'], runs=x['meta']['n_runs'],
                                tan_max_true=x['meta'].get('tan_max_true'),
                                t0_max=x['meta'].get('t0_max'), axis=x['meta'].get('axis'))
            for (ch, arm), x in D.items()}
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Same-chamber pairs</title>
<style>
:root{{--ink:#1b2430;--muted:#6a7583;--line:#d9dde3;--bg:#fff;--accent:#8a3f8f;--soft:#f6f4f8}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--ink:#e6e9ee;--muted:#9aa4b1;--line:#3a4250;--bg:#14181f;--accent:#c48ac8;--soft:#1d2129}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:16px}}
h1{{font-size:1.5em;margin-bottom:.2em}} h2{{font-size:1.15em;margin-top:2em;border-bottom:1px solid var(--line)}}
.sub{{color:var(--muted)}} .verdict{{background:var(--soft);border-left:4px solid var(--accent);padding:10px 14px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted);font-weight:600}}
figure{{margin:22px 0}} img{{max-width:100%;background:#fff}} figcaption{{color:var(--muted);font-size:13.5px}}
code{{font-size:12.5px}} li{{margin:.3em 0}}
</style></head><body>
<h1>Same-chamber pairs in A and C, on the is2_v2 re-pass</h1>
<p class="sub">Generated {datetime.now():%Y-%m-%d %H:%M} by <code>make_same_chamber_report.py</code> · post-access runs · preliminary</p>
{verdict}
<h2>What was compared</h2>
<ul>
<li><b>production</b>: <code>stage3_fullpass</code>, the August full pass, production bundles and per-run k.</li>
<li><b>re-pass, capsule-focus k</b> (headline): the is2_v2 re-pass (in-situ bundles, seeder minimum 3, two-sided slope search) with k = 1.12 × the in-beam-muon k, the scale at which the capsule image is sharpest (HANDOFF_TRACKING §21–22).</li>
<li><b>re-pass, muon k</b>: the same tracks at the in-beam A–C muon scale. The two k differ by 12 %, and nobody has explained why yet. Angles are shown at both.</li>
<li>Same cuts everywhere: gated, angle-calibrated, |true tan| &lt; 0.6 (re-pass), <b>t0 ≤ 300 ns</b> on both planes (the late NNLS-runaway class has wrong geometry and makes 0° ghost pairs), beam axis at each run's measured imaging crossing (x ≈ −9.5 mm).</li>
<li><b>Wall confirmation</b> per leg (<code>det_a_scint.match_run</code>): the extrapolated track lands on a SiPM-wall group that fired in time. The off-time control uses the same match against out-of-time hits. The event-mixed null pairs tracks from different triggers.</li>
<li><b>Two particles</b>: the wall is four 100 mm groups, so one real particle plus a junk leg beside it confirms "both legs" in the same group. The two-particle selection requires the legs to be predicted in <i>different</i> groups, each clear of a group boundary, with both groups fired in time.</li>
</ul>
<h2>Yield and purity</h2>
{t1}
<h2>The two-particle pairs</h2>
{t2}
<p class="sub">Prompt = |Δt0| ≤ 75 ns. Line DCA = closest approach of the two reconstructed lines. Vertexed = the lines meet within 30 mm of each other and 20 mm of the measured axis.</p>
<h2>Figures</h2>
{fig_html}
{phys_html}
<h2>What this does not rule out</h2>
<ul>
<li><b>The angle scale.</b> Every opening angle carries the unresolved 12 % muon/capsule k tension. The capsule k is adopted here and the muon k is shown beside it.</li>
<li><b>The two-particle selection is an acceptance cut.</b> It needs both legs on the wall, in different 100 mm groups, so it removes small-separation pairs and pushes the opening angle up. The event-mixed null carries the same cut, but no folded acceptance has been thrown for this selection, so no physics model is compared here.</li>
<li><b>Real pairs the wall cannot certify</b> (same group, or off the wall) are still in the "all pairs" sample, mixed with junk. Their number is not known.</li>
<li><b>Two tracks closer than ~20 mm in either view are not reconstructed.</b> The joint two-track fit and x/y pairing exist (now on main) but were off in this pass, and their thresholds must be recalibrated on the in-situ bundles before they ride a re-pass.</li>
<li><b>Two particles does not mean from the capsule.</b> Only a few two-particle pairs vertex on the axis. A shower, or a through-going particle with a delta ray, is also two particles. The vertex rate against the mixed null is the handle, and its statistics are thin.</li>
<li><b>Only A and C.</b> B and D have no in-situ bundle. The pre-access runs are excluded.</li>
</ul>
<h2>Provenance</h2>
<pre style="font-size:12px;overflow-x:auto">{html.escape(json.dumps(prov, indent=1))}</pre>
</body></html>"""
    p = out / 'report.html'
    p.write_text(page)
    print(f'wrote {p}')
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(Path.home() / 'x17_local' / 'same_chamber'))
    ap.add_argument('--out', default=str(paths.out('same_chamber')))
    a = ap.parse_args()
    build(Path(a.src), Path(a.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
