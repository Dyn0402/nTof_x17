#!/usr/bin/env python3
"""
make_two_track_limit_report.py -- report.html for two_track_limit.py.

Where is the real limit on resolving two tracks in one plane, and how far is
production from it? Built from whatever the ladder has written:

    <out>/two_track_limit/r1_asimov.csv           information limit
    <out>/two_track_limit/r2_oracle.parquet       perfect model, ideal fit
    <out>/two_track_limit/r4_synthprod*.parquet   production on the same planes
    <out>/two_track_limit/r3_real.parquet         real overlays, twins, production

A missing input drops its section, and the report says so.

    python -m sept26_prelim_analysis.make_two_track_limit_report
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import figstyle as fs          # noqa: E402
from sept26_prelim_analysis import paths                   # noqa: E402
from sept26_prelim_analysis.report_style import head       # noqa: E402

FSR = 0.01                  # false-split rate the thresholds are set at
PROD_F = 300.0              # TWO_TRACK_F
ARMS = ('A', 'C')
REAL_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
#: a real window counts as well modelled when its donor(s), each refitted alone
#: on it, reach chi2/dof below this. Above it the window holds charge the donor
#: does not explain (other clusters in the 61-strip window, saturation).
DONOR_CHI2DOF_MAX = 2.0


def src_dir() -> Path:
    return paths.out('two_track_limit')


def _pct(x, d=0):
    return '—' if x is None or not np.isfinite(x) else f'{100 * x:.{d}f}&nbsp;%'


def table(cols, rows) -> str:
    return ('<div class="tw"><table><thead><tr>'
            + ''.join(f'<th>{c}</th>' for c in cols)
            + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')


def tr(cells) -> str:
    return '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'


def figure_html(name, caption) -> str:
    return (f'<figure><img src="figures/{name}.png" alt="{caption}">'
            f'<figcaption>{caption} '
            f'<a class="src" href="figures/{name}.csv">numbers &#8599;</a>'
            f'</figcaption></figure>')


def _bool(s):
    return s.astype('boolean').fillna(False).astype(bool)


# --------------------------------------------------------------------------- #
# inputs -> tidy efficiency tables
# --------------------------------------------------------------------------- #
def load() -> dict:
    d = src_dir()
    L = {}
    if (d / 'r1_asimov.csv').exists():
        L['r1'] = pd.read_csv(d / 'r1_asimov.csv')
    if (d / 'r2_oracle.parquet').exists():
        L['r2'] = pd.read_parquet(d / 'r2_oracle.parquet')
    for tag, f in (('prod', 'r4_synthprod.parquet'), ('fixed', 'r4_synthprod_fixed.parquet')):
        if (d / f).exists():
            L[f'r4_{tag}'] = pd.read_parquet(d / f)
    if (d / 'r3_real.parquet').exists():
        R = pd.read_parquet(d / 'r3_real.parquet')
        if (d / 'r3_scan.parquet').exists():
            # production columns from the threshold scan: current at its own
            # threshold, fixed at the lowest threshold whose false-split rate
            # on real singles does not exceed current's (per chamber)
            Sc = pd.read_parquet(d / 'r3_scan.parquet')
            L['scan'] = Sc
            L['pick'] = matched_thresholds(Sc)
            cur = Sc[(Sc.variant == 'current_f0') & (Sc.F == PROD_F) & (Sc.n_true == 2)]
            fix = pd.concat([Sc[(Sc.variant == 'fixed_f0') & (Sc.F == F) & (Sc.arm == a)
                                & (Sc.n_true == 2)] for a, F in L['pick'].items()])
            R = (R.drop(columns=['prod_found'])
                 .merge(cur[['oid', 'found']].rename(columns={'found': 'prod_found'}),
                        on='oid', how='left')
                 .merge(fix[['oid', 'found']].rename(columns={'found': 'fixed_found'}),
                        on='oid', how='left'))
        L['r3'] = R
    return L


def fsr_table(Sc: pd.DataFrame) -> pd.Series:
    return Sc[Sc.n_true == 1].groupby(['variant', 'F', 'arm']).split.mean()


def matched_thresholds(Sc: pd.DataFrame) -> dict:
    """Per chamber, the lowest fixed-chain threshold whose false-split rate on
    real single donors does not exceed current production's at TWO_TRACK_F."""
    f = fsr_table(Sc)
    out = {}
    for a in ARMS:
        ref = f[('current_f0', PROD_F, a)]
        g = f.loc['fixed_f0'].xs(a, level='arm')
        out[a] = float(g[g <= ref].index.min())
    return out


def eff_oracle(R: pd.DataFrame) -> pd.DataFrame:
    """R2: Delta-chi2 above the FSR quantile of the singles, and both lines found."""
    out = []
    for (arm, tan), g in R.groupby(['arm', 'tan']):
        thr = float(np.quantile(g[g.n_true == 1].dchi2, 1 - FSR))
        for d, h in g[g.n_true == 2].groupby('d'):
            ok = (h.dchi2 > thr) & _bool(h.found)
            out.append(dict(arm=arm, tan=tan, d=d, stage='ideal', eff=ok.mean(),
                            n=len(h), thr=thr))
    return pd.DataFrame(out)


def eff_prod(P: pd.DataFrame, stage: str, trigger: bool = False) -> pd.DataFrame:
    P = P.copy()
    for c in ('prod_found', 'prod_accepted', 'prod_trig'):
        P[c] = _bool(P[c]) if c in P else False
    ok = P.prod_accepted & P.prod_found
    if trigger:
        ok &= P.prod_trig
    P['ok'] = ok
    e = (P[P.n_true == 2].groupby(['arm', 'tan', 'd'])
         .agg(eff=('ok', 'mean'), n=('ok', 'size')).reset_index().assign(stage=stage))
    fsr = (P[P.n_true == 1].groupby(['arm', 'tan']).prod_accepted.mean()
           .rename('fsr').reset_index())
    return e.merge(fsr, on=['arm', 'tan'], how='left')


def synth_ladder(L) -> pd.DataFrame:
    parts = []
    if 'r2' in L:
        parts.append(eff_oracle(L['r2']))
    if 'r4_prod' in L:
        parts.append(eff_prod(L['r4_prod'], 'production, with trigger', trigger=True))
        parts.append(eff_prod(L['r4_prod'], 'production, no trigger'))
    if 'r4_fixed' in L:
        parts.append(eff_prod(L['r4_fixed'], 'fixed: scale + grid, no trigger'))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def real_ladder(R: pd.DataFrame) -> pd.DataFrame:
    """R3/R4 on real overlays, per chamber, in bins of r.m.s. separation.
    Thresholds from each population's own singles, per (arm, plane)."""
    R = R.copy()
    R['bin'] = pd.cut(R.sep_rms, REAL_BINS, right=False)
    out = []
    for (arm, plane), g in R.groupby(['arm', 'plane']):
        nul = g[g.n_true == 1]
        thr = {k: float(np.quantile(nul[f'{k}_dchi2'], 1 - FSR)) for k in ('real', 'twin')}
        pr = g[g.n_true == 2].copy()
        pr['twin_ok'] = (pr.twin_dchi2 > thr['twin']) & _bool(pr.twin_found)
        pr['real_ok'] = (pr.real_dchi2 > thr['real']) & _bool(pr.real_found)
        pr['prod_ok'] = _bool(pr['prod_found']) if 'prod_found' in pr else False
        pr['fixed_ok'] = _bool(pr['fixed_found']) if 'fixed_found' in pr else np.nan
        pr = pr.assign(arm=arm, plane=plane, thr_real=thr['real'], thr_twin=thr['twin'])
        out.append(pr)
    P = pd.concat(out, ignore_index=True)
    T = (P.groupby(['arm', 'bin'], observed=True)
         .agg(twin=('twin_ok', 'mean'), real=('real_ok', 'mean'), prod=('prod_ok', 'mean'),
              fixed=('fixed_ok', 'mean'), n=('oid', 'size'),
              sep_mid=('sep_rms', 'median')).reset_index())
    T['lo'] = [b.left for b in T.bin]
    T['hi'] = [b.right for b in T.bin]
    return T.drop(columns='bin'), P


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
STAGE_STYLE = {
    'ideal': dict(color=fs.INK, ls='-', lw=1.8, marker='o'),
    'fixed: scale + grid, no trigger': dict(color=fs.ACCENT, ls='-', lw=1.6, marker='s'),
    'production, no trigger': dict(color=fs.COPPER, ls='--', lw=1.5, marker='^'),
    'production, with trigger': dict(color=fs.MUTED, ls=':', lw=1.5, marker='v'),
}


def fig_asimov(r1: pd.DataFrame, od: Path):
    fig, ax = fs.figure(fs.FIG)
    d = r1[r1.plane == 'x']
    for arm in ARMS:
        for tan, ls in ((0.0, '-'), (0.3, '--')):
            g = d[(d.arm == arm) & (d.tan == tan)].sort_values('d')
            g = g[g.d > 0]
            st = fs.det_style(arm)
            ax.plot(g.d, np.maximum(g.lam, 1e-2), ls=ls, color=st['color'],
                    marker=st['marker'], ms=4, lw=1.5, label=f'{arm}, tan θ = {tan:g}')
    ax.axhline(25, color=fs.MUTED, lw=0.9, ls=':')
    ax.set_xscale('log')
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 12])
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '4', '8', '12'])
    ax.text(12.2, 25, 'noise-only 99th pct ≈ 25', color=fs.MUTED, va='bottom', ha='right',
            fontsize=fs.BASE_PT * 0.8)
    ax.set_yscale('log')
    ax.set_ylim(0.1, 5e4)
    ax.set_xlabel('separation of two parallel tracks  d  [mm]')
    ax.set_ylabel('expected Δχ²  one vs two tracks  (λ)')
    ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.85, ncol=2, loc='lower right')
    fs.title(ax, 'The information is there from about one strip pitch',
             'noise-free forward model, best single-track fit; x plane, run_145 bundles')
    fs.preliminary(ax, 'upper left')
    fs.save(fig, od / 'r1_asimov', data=d)


def fig_synth(S: pd.DataFrame, od: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True, sharey=True)
    for i, arm in enumerate(ARMS):
        for j, tan in enumerate((0.0, 0.3)):
            ax = axes[i, j]
            fs.strip(ax)
            g = S[(S.arm == arm) & (S.tan == tan)]
            for stage, st in STAGE_STYLE.items():
                h = g[g.stage == stage].sort_values('d')
                if len(h):
                    ax.plot(h.d, 100 * h.eff, ms=3.5, label=stage, **st)
            ax.set_xscale('log')
            ax.set_ylim(-3, 103)
            ax.set_title(f'chamber {arm}, tan θ = {tan:g}', loc='left',
                         fontsize=fs.BASE_PT, color=fs.INK)
            if i == 1:
                ax.set_xlabel('separation d [mm]')
            if j == 0:
                ax.set_ylabel('pairs resolved [%]')
            ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 12])
            ax.set_xticklabels(['0.25', '0.5', '1', '2', '4', '8', '12'])
    axes[0, 1].legend(frameon=False, fontsize=fs.BASE_PT * 0.8, loc='center right')
    fs.preliminary(axes[1, 1], 'upper left')
    fig.suptitle('Same synthetic planes: the ideal fit against production',
                 x=0.01, ha='left', fontsize=fs.BASE_PT * 1.15, color=fs.INK)
    fig.tight_layout()
    fs.save(fig, od / 'r4_synth_ladder', data=S)


def fig_real(T: pd.DataFrame, od: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    styles = dict(twin=('perfect-model twin, ideal fit', fs.INK, '-', 'o'),
                  real=('real overlay, ideal fit', fs.ACCENT, '-', 's'),
                  fixed=('real overlay, fixed production', fs.DET_COLOR['A'], '-.', 'D'),
                  prod=('real overlay, production', fs.COPPER, '--', '^'))
    for ax, arm in zip(axes, ARMS):
        fs.strip(ax)
        g = T[T.arm == arm].sort_values('lo')
        x = 0.5 * (g.lo + g.hi)
        for k, (lab, c, ls, m) in styles.items():
            if g[k].isna().all():
                continue
            ax.plot(x, 100 * g[k], color=c, ls=ls, marker=m, ms=4, lw=1.5, label=lab)
        ax.set_xscale('log')
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16])
        ax.set_xticklabels(['0.25', '0.5', '1', '2', '4', '8', '16'])
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT, color=fs.INK)
        ax.set_xlabel('r.m.s. separation over the drift column [mm]')
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel('pairs resolved in this view [%]')
    axes[0].legend(frameon=False, fontsize=fs.BASE_PT * 0.8, loc='lower right')
    fs.preliminary(axes[1], 'lower right')
    fig.suptitle('Real overlays: model mismatch against algorithm',
                 x=0.01, ha='left', fontsize=fs.BASE_PT * 1.15, color=fs.INK)
    fig.tight_layout()
    fs.save(fig, od / 'r3_real_ladder', data=T)


def fig_roc(Sc: pd.DataFrame, pick: dict, od: Path):
    import matplotlib.pyplot as plt
    f = fsr_table(Sc)
    e = Sc[Sc.n_true == 2].groupby(['variant', 'F', 'arm']).found.mean()
    D = pd.DataFrame(dict(fsr=f, eff=e)).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, arm in zip(axes, ARMS):
        fs.strip(ax)
        for v, lab, c, m in (('current_f0', 'production', fs.COPPER, '^'),
                             ('fixed_f0', 'fixed: scale + grid, no trigger',
                              fs.DET_COLOR['A'], 'D')):
            g = D[(D.arm == arm) & (D.variant == v)].sort_values('F')
            ax.plot(100 * g.fsr, 100 * g.eff, color=c, marker=m, ms=4, lw=1.5, label=lab)
            mark = PROD_F if v == 'current_f0' else pick[arm]
            h = g[g.F == mark]
            ax.plot(100 * h.fsr, 100 * h.eff, marker='o', ms=10, mfc='none', color=c, lw=0)
            ax.annotate(f'F = {mark:g}', (100 * h.fsr.iloc[0], 100 * h.eff.iloc[0]),
                        textcoords='offset points', xytext=(8, -12), color=fs.MUTED,
                        fontsize=fs.BASE_PT * 0.8)
        ax.set_xscale('symlog', linthresh=0.5)
        ax.set_xlim(-0.05, 60)
        ax.set_xticks([0, 0.5, 1, 2, 5, 10, 20, 50])
        ax.set_xticklabels(['0', '0.5', '1', '2', '5', '10', '20', '50'])
        ax.minorticks_off()
        ax.set_ylim(0, 80)
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT, color=fs.INK)
        ax.set_xlabel('real single tracks split [%]')
    axes[0].set_ylabel('real pairs resolved, 0–24 mm [%]')
    axes[0].legend(frameon=False, fontsize=fs.BASE_PT * 0.8, loc='lower right')
    fs.preliminary(axes[1], 'lower right')
    fig.tight_layout()
    fs.save(fig, od / 'r3_roc', data=D)
    return D


def fig_null(R: pd.DataFrame, od: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    rows = []
    bins = np.logspace(-1, 4.5, 50)
    for ax, arm in zip(axes, ARMS):
        fs.strip(ax)
        g = R[(R.arm == arm) & (R.n_true == 1)]
        for k, c, lab in (('twin', fs.INK, 'perfect-model twin'),
                          ('real', fs.ACCENT, 'real single track')):
            v = np.clip(g[f'{k}_dchi2'].to_numpy(), 0.1, None)
            ax.hist(v, bins=bins, histtype='step', color=c, lw=1.5, label=lab)
            q = np.quantile(v, 1 - FSR)
            ax.axvline(q, color=c, lw=0.9, ls=':')
            rows.append(dict(arm=arm, population=k, n=len(v), median=np.median(v),
                             q99=q))
        ax.set_xscale('log')
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT, color=fs.INK)
        ax.set_xlabel('Δχ² gained by splitting a single track')
    axes[0].set_ylabel('windows')
    axes[0].legend(frameon=False, fontsize=fs.BASE_PT * 0.8)
    fs.preliminary(axes[1])
    fig.tight_layout()
    fs.save(fig, od / 'r3_null', data=pd.DataFrame(rows))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# validation: event-level bench and the split-ab contract
# --------------------------------------------------------------------------- #
BENCH_VARIANTS = [('before', 'pairing_rescue16_two_final_replace'),
                  ('fixed', 'fixed_{arm}_replace'),
                  ('fixed + profile pairing', 'fixed_{arm}_replace_profc')]
BENCH_BANDS = [(0, 12, '&lt; 12 mm'), (12, 24, '12–24 mm'), (24, 1e9, '&ge; 24 mm')]


def bench_table() -> pd.DataFrame:
    """Coincident overlays, both donors found and correctly x/y-paired."""
    from sept26_prelim_analysis import intra_bench as ib
    base = ib.out_dir()
    out = []
    for arm in ARMS:
        for lab, tmpl in BENCH_VARIANTS:
            d = base / tmpl.format(arm=arm)
            if not (d / 'overlays.parquet').exists():
                continue
            M = pd.read_parquet(d / 'overlays.parquet')
            C = pd.read_parquet(d / 'candidates.parquet')
            Dn = pd.read_parquet(d / 'donors.parquet' if (d / 'donors.parquet').exists()
                                 else base / 'donors.parquet')
            S = ib.score(M, C, Dn)
            o = S[(S['mode'] == 'overlay') & (S.cls == 'coincident') & (S.arm == arm)]
            ev = o.groupby('oid').agg(sx=('sep_x', 'first'), sy=('sep_y', 'first'),
                                      both=('track_found', 'all'))
            sep = np.minimum(ev.sx, ev.sy)
            row = dict(arm=arm, variant=lab)
            for lo, hi, b in BENCH_BANDS:
                row[b] = float(ev.both[(sep >= lo) & (sep < hi)].mean())
            out.append(row)
    return pd.DataFrame(out)


def validation_html() -> str:
    from sept26_prelim_analysis import intra_bench as ib
    B = bench_table()
    if not len(B):
        return ''
    rows = [tr([r['arm'], r['variant']] + [_pct(r[b], 1) for _lo, _hi, b in BENCH_BANDS])
            for _i, r in B.iterrows()]
    html = ('<section><h2>Validation · event level and the contract</h2>'
            '<p>The fixed chain at its matched thresholds (A&nbsp;F&nbsp;=&nbsp;1200, '
            'C&nbsp;F&nbsp;=&nbsp;2400, corroborated at 0.4&nbsp;F), on the full overlay bench '
            '(<code>--overlay replace</code>, all seven file tags, lxplus condor cluster '
            '4334051). A pair counts when both donors are found <i>and</i> correctly '
            'x/y-paired. <b>Profile pairing</b> adds the constrained depth-profile term '
            '(<code>xy_pairing_*_profc.json</code>, calibrated on stat090_0001).</p>'
            + table(['chamber', 'chain'] + [b for _lo, _hi, b in BENCH_BANDS], rows))
    f = ib.out_dir() / 'contract_fixed_vs_current.csv'
    if f.exists():
        R = pd.read_csv(f)
        R = R[R.tag == 'all']
        crow = [tr([r.arm, r.chain, f'{r.triggers:,}',
                    f'{r.clean_singles_split}/{r.clean_singles} ({_pct(r.frac_clean_split, 2)})',
                    r.events_fewer_tracks, r.clean_tracks_lost,
                    f'{r.not_recovered}/{r.prod_tracks}', r.events_split, r.events_more_tracks])
                for r in R.itertuples()]
        html += ('<h3>The contract on real triggers</h3>'
                 '<p><code>split-ab</code>: real triggers re-reconstructed with each chain and '
                 'matched to the frozen full pass, on the <b>same triggers</b> for both chains, '
                 'both with x/y re-pairing (which alone moves ~5&nbsp;% of unsplit events '
                 'against the frozen pass, which was reconstructed without it). The contract: '
                 'clean single muons split &le;&nbsp;0.66&nbsp;%, no event losing a track.</p>'
                 + table(['chamber', 'chain', 'triggers', 'clean singles split',
                          'events losing a track', 'clean tracks lost',
                          'production tracks not recovered', 'events split',
                          'events gaining a track'], crow))
    return html + '</section>'


# --------------------------------------------------------------------------- #
# the operating point: the split-ab F rescan against the bench
# --------------------------------------------------------------------------- #
CONTRACT_CLEAN_SPLIT = 0.0066           # clean single muons split, at most
LADDER_VARIANT = 'split_ab_ladder_{arm}_7tags'


def _cp_upper(k: int, n: int, cl: float = 0.90) -> float:
    """One-sided Clopper-Pearson upper limit on k/n."""
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(cl, k + 1, n - k))


def ladder_table(L: dict) -> pd.DataFrame:
    """Per (arm, F): the real-trigger contract (split-ab ladder, all seven tags)
    next to the bench (fixed_f0 scan, real overlays). Empty without a ladder."""
    from sept26_prelim_analysis import intra_bench as ib
    rows = []
    for arm in ARMS:
        f = ib.out_dir(LADDER_VARIANT.format(arm=arm)) / 'summary_ladder.csv'
        if not f.exists():
            continue
        rows.append(pd.read_csv(f))
    if not rows:
        return pd.DataFrame()
    D = pd.concat(rows, ignore_index=True)
    D['clean_split_ul90'] = [_cp_upper(int(r.clean_singles_split), int(r.clean_singles))
                             for r in D.itertuples()]
    D['passes'] = ((D.frac_clean_split <= CONTRACT_CLEAN_SPLIT)
                   & (D.events_fewer_tracks == 0))
    if 'scan' in L:
        Sc = L['scan'][L['scan'].variant == 'fixed_f0']
        b = Sc.groupby(['arm', 'F']).apply(lambda g: pd.Series(dict(
            bench_fsr=g[g.n_true == 1].split.mean(),
            bench_eff=g[g.n_true == 2].found.mean())), include_groups=False).reset_index()
        D = D.merge(b.astype({'F': float}), on=['arm', 'F'], how='left')
    return D


def ladder_pick(D: pd.DataFrame) -> dict:
    """Per chamber, the lowest F that meets the contract on real triggers."""
    out = {}
    for arm, g in D.groupby('arm'):
        ok = g[g.passes]
        if len(ok):
            out[arm] = float(ok.F.min())
    return out


def fig_ladder(D: pd.DataFrame, pick: dict, matched: dict, od: Path):
    import matplotlib.pyplot as plt
    arms = [a for a in ARMS if a in set(D.arm)]
    fig, axes = plt.subplots(2, len(arms), figsize=(9.6, 5.4), sharex=True,
                             squeeze=False)
    for j, arm in enumerate(arms):
        g = D[D.arm == arm].sort_values('F')
        c = fs.DET_COLOR[arm]
        top, bot = axes[0, j], axes[1, j]
        for ax in (top, bot):
            fs.strip(ax)
            ax.set_xscale('log')
            for F, ls, lab in ((matched.get(arm), ':', 'matched'), (pick.get(arm), '-', 'pick')):
                if F is not None:
                    ax.axvline(F, color=fs.MUTED, lw=0.9, ls=ls)
        top.fill_between(g.F, 100 * g.frac_clean_split, 100 * g.clean_split_ul90,
                         color=c, alpha=0.15, lw=0)
        top.plot(g.F, 100 * g.frac_clean_split, color=c, marker='o', ms=4, lw=1.5)
        top.axhline(100 * CONTRACT_CLEAN_SPLIT, color=fs.COPPER, lw=1.0, ls='--')
        top.annotate('contract', (g.F.max(), 100 * CONTRACT_CLEAN_SPLIT),
                     textcoords='offset points', xytext=(-4, 4), ha='right',
                     color=fs.MUTED, fontsize=fs.BASE_PT * 0.8)
        top.set_ylim(0, 300 * CONTRACT_CLEAN_SPLIT)   # 0-2 %: the band may run off the top
        top.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT, color=fs.INK)
        if 'bench_eff' in g:
            h = g.dropna(subset=['bench_eff'])
            bot.plot(h.F, 100 * h.bench_eff, color=c, marker='D', ms=4, lw=1.5)
        bot.set_ylim(0, 80)
        bot.set_xlabel('split threshold F')
        bot.set_xticks([300, 600, 1000, 2000, 4800])
        bot.set_xticklabels(['300', '600', '1000', '2000', '4800'])
        bot.minorticks_off()
    axes[0, 0].set_ylabel('clean singles split\non real triggers [%]')
    axes[1, 0].set_ylabel('real pairs resolved\non the bench [%]')
    fs.preliminary(axes[0, -1], 'upper right')
    fig.tight_layout()
    fs.save(fig, od / 'ladder_operating_point', data=D)


def ladder_html(L: dict, od: Path) -> str:
    D = ladder_table(L)
    if not len(D):
        return ('<section><h2>Operating point · the F rescan on real triggers</h2>'
                '<p>Not yet merged (<code>merge_two_track.py --pkg '
                '~/x17/two_track_ladder_condor</code>).</p></section>')
    pick = ladder_pick(D)
    matched = L.get('pick', {})
    fig_ladder(D, pick, matched, od)
    lo = D.groupby('arm').F.min()
    verdict = '; '.join(
        f'chamber {a}: F&nbsp;=&nbsp;{pick[a]:g}' + (
            f' (matched {matched[a]:g})' if a in matched else '') + (
            ', the bottom of the ladder, so the true minimum may be lower'
            if pick[a] == lo[a] else '')
        for a in ARMS if a in pick) or 'no F on the ladder meets the contract'
    rows = []
    for r in D.sort_values(['arm', 'F']).itertuples():
        mark = ' &larr; pick' if pick.get(r.arm) == r.F else (
            ' (matched)' if matched.get(r.arm) == r.F else '')
        rows.append(tr([r.arm, f'{r.F:g}{mark}',
                        f'{r.clean_singles_split}/{r.clean_singles} '
                        f'({_pct(r.frac_clean_split, 2)}, &le;&nbsp;{_pct(r.clean_split_ul90, 2)})',
                        r.events_fewer_tracks, r.clean_tracks_lost,
                        f'{r.not_recovered}/{r.prod_gated_tracks}', r.events_split,
                        r.events_more_tracks,
                        _pct(getattr(r, 'bench_fsr', np.nan), 1),
                        _pct(getattr(r, 'bench_eff', np.nan))]))
    return ('<section><h2>Operating point · the F rescan on real triggers</h2>'
            f'<p><b>Lowest F meeting the contract on real triggers: {verdict}.</b> '
            'The contract: clean single muons split &le;&nbsp;0.66&nbsp;% and no event '
            'losing a track. The upper limit in brackets is one-sided 90&nbsp;% '
            'Clopper&ndash;Pearson. The pick uses the point estimate, as the contract '
            'is written.</p>'
            '<p><code>split-ab</code> of the fixed chain, all seven tags, one condor pass '
            '(cluster 4348153). Every F is replayed exactly from the same attempts '
            '(<code>wft.reco.two_track_ladder</code>): attempts and their statistic do not '
            'depend on the threshold, only acceptance does. Bench columns: the fixed chain '
            'on real overlays at the same F (<code>r3_scan</code>, 600 single donors per '
            'view, all pairs 0&ndash;24&nbsp;mm). Those donors are split more readily than '
            'clean singles on real triggers, which is why the two false-split '
            'columns differ.</p>'
            + figure_html('ladder_operating_point',
                          'Top: clean single muons split on real triggers against the split '
                          'threshold F (band: 90&nbsp;% upper limit; dashed: the contract). '
                          'Bottom: real overlay pairs resolved on the bench at the same F. '
                          'Dotted: the threshold matched on bench donors; solid: the pick.')
            + table(['chamber', 'F', 'clean singles split', 'events losing a track',
                     'clean tracks lost', 'production tracks not recovered', 'events split',
                     'events gaining a track', 'bench singles split', 'bench pairs resolved'],
                    rows)
            + '</section>')


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
def synth_table(S: pd.DataFrame, arm: str, tan: float) -> str:
    g = S[(S.arm == arm) & (S.tan == tan)]
    stages = [s for s in STAGE_STYLE if s in set(g.stage)]
    rows = []
    for d in sorted(g.d.unique()):
        h = g[g.d == d].set_index('stage')
        rows.append(tr([f'{d:g}'] + [_pct(h.eff.get(s, np.nan)) for s in stages]))
    fsr = [g[g.stage == s].fsr.dropna() for s in stages]
    rows.append(tr(['<i>false splits</i>'] + [_pct(f.iloc[0], 1) if len(f) else '—'
                                             for f in fsr]))
    return table(['d [mm]'] + stages, rows)


def build() -> Path:
    fs.use()
    L = load()
    od = paths.out('two_track_limit', 'report')
    (od / 'figures').mkdir(parents=True, exist_ok=True)
    fd = od / 'figures'
    sec = []

    S = synth_ladder(L)
    T = Ta = None
    if 'r3' in L:
        R3 = L['r3']
        clean = R3[R3.donor_chi2dof < DONOR_CHI2DOF_MAX]
        T, _P = real_ladder(clean)
        Ta, _Pa = real_ladder(R3)
        kept = (R3.donor_chi2dof < DONOR_CHI2DOF_MAX).groupby([R3.arm, R3.n_true]).mean()

    # ---- verdict
    v = ['<section><h2>Verdict</h2>']
    if 'r2' in L:
        e2 = S[S.stage == 'ideal']
        lim = e2[e2.eff >= 0.95].groupby(['arm', 'tan']).d.min()
        v.append('<p class="verdict"><b>With a perfect model, two tracks in one plane are '
                 'resolved from about one strip pitch.</b> The ideal fit reaches 95&nbsp;% at '
                 + ', '.join(f'{d:g}&nbsp;mm ({a}, tan&nbsp;{t:g})' for (a, t), d in lim.items())
                 + f', at a {FSR:.0%} false-split rate, and never drops again. Everything '
                 'production loses above that on synthetic planes is algorithmic: the trigger, '
                 'the split statistic&rsquo;s scale, local minima in the search, and a fixed '
                 '1.2&nbsp;mm guard.</p>')
    if T is not None:
        g = T[(T.lo >= 3.0) & (T.hi <= 16.0)]

        def m(k):
            return ', '.join(f'{a} {_pct(g[g.arm == a][k].mean())}' for a in ARMS)
        v.append('<p><b>On real tracks the limit moves to about 2&ndash;3&nbsp;mm, and '
                 'production is far from it.</b> A real single track is not described '
                 'perfectly by the forward model, so splitting it gains far more '
                 '&Delta;&chi;&sup2; than splitting its perfect-model twin, and the split '
                 'threshold at a 1&nbsp;% false-split rate rises 10&ndash;30-fold. The '
                 'mismatch grows with the track&rsquo;s charge. On well-modelled real '
                 'overlays at '
                 f'3&ndash;16&nbsp;mm r.m.s. separation: perfect-model twin {m("twin")}; the '
                 f'ideal fit on the real windows {m("real")}; production {m("prod")}'
                 + (f'; fixed production {m("fixed")}' if g.fixed.notna().any() else '')
                 + ' (unweighted bin means). The fixed chain is compared at a false-split '
                 'rate on real single tracks no higher than production&rsquo;s.</p>')
    v.append('</section>')
    sec.append(''.join(v))

    # ---- R1
    if 'r1' in L:
        fig_asimov(L['r1'], fd)
        sec.append('<section><h2>R1 · the information limit</h2>'
                   '<p>Two tracks with the same slope and t0, generated noise-free by the '
                   'forward model under the run_145 bundle, 1450 ADC&middot;bin each, a flat '
                   'profile to 840&nbsp;ns. The best single track fitted to that window leaves '
                   '&lambda;, the expected &Delta;&chi;&sup2; of a perfect analysis at the '
                   'run&rsquo;s noise (13.3&nbsp;ADC). It collapses like ~d&#8308; near zero and '
                   'saturates once the tracks stop overlapping.</p>'
                   + figure_html('r1_asimov', 'Expected Δχ² between one and two tracks. '
                                 'Inclined tracks spread each depth slice over fewer strips '
                                 'per unit charge, so the same d carries ~50× less information.')
                   + '</section>')

    # ---- R2 + R4 synthetic
    if len(S):
        fig_synth(S, fd)
        parts = ['<section><h2>R2 &amp; R4 · the same synthetic planes, ideal against '
                 'production</h2>'
                 '<p><b>Ideal</b>: pure &chi;&sup2;, one- and two-track fits from the truth '
                 'and from a broad start set, threshold at the 99th percentile of '
                 '&Delta;&chi;&sup2; on 400 synthetic singles per cell. <b>Production</b>: '
                 '<code>wft.reco</code>&rsquo;s own fit, probe, trigger and joint fit on the '
                 'same planes (same seeds), cut to a production window, threshold '
                 f'<code>TWO_TRACK_F</code>&nbsp;=&nbsp;{PROD_F:g}. <b>Fixed</b>: production '
                 'with <code>WFT_TWO_TRACK_SCALE=two</code> and '
                 '<code>WFT_TWO_TRACK_SEARCH=grid</code>. A pair counts when both fitted '
                 'lines lie within 1&nbsp;mm r.m.s. of distinct true lines at the same '
                 'absolute times.</p>',
                 figure_html('r4_synth_ladder', 'Efficiency to resolve a pair against '
                             'separation, 100 pairs per point.')]
        for arm in ARMS:
            for tan in (0.0, 0.3):
                parts.append(f'<h3>chamber {arm}, tan&nbsp;&theta;&nbsp;=&nbsp;{tan:g}</h3>'
                             + synth_table(S, arm, tan))
        parts.append('</section>')
        sec.append(''.join(parts))

    # ---- R3
    if T is not None:
        fig_real(T, fd)
        N = fig_null(clean, fd)

        def rows_of(TT):
            return [tr([a, f'{r.lo:g}–{r.hi:g}', int(r.n), _pct(r.twin), _pct(r.real),
                        _pct(r.fixed), _pct(r['prod'])]) for a in ARMS
                    for _i, r in TT[TT.arm == a].sort_values('lo').iterrows()]
        rows, rows_all = rows_of(T), rows_of(Ta)
        roc_html = ''
        if 'scan' in L:
            D = fig_roc(L['scan'], L['pick'], fd)
            rr = []
            for a in ARMS:
                for v, F in (('current_f0', PROD_F), ('fixed_f0', L['pick'][a])):
                    h = D[(D.arm == a) & (D.variant == v) & (D.F == F)].iloc[0]
                    rr.append(tr([a, 'production' if v == 'current_f0' else 'fixed',
                                  f'{F:g}', _pct(h.fsr, 1), _pct(h.eff)]))
            roc_html = ('<h3>At equal false-split rate</h3>'
                        '<p>One production run per chain at threshold 0 records every '
                        'split attempt; any threshold is then applied offline (the '
                        'corroborated threshold kept at 0.4&nbsp;F, as in production). '
                        'False splits: fraction of 600 real single donors per view that '
                        'get an accepted split in either view.</p>'
                        + figure_html('r3_roc', 'Real pairs resolved (all windows, all '
                                      'separations) against real single tracks split, as '
                                      'the threshold F is scanned. Circles: the operating '
                                      'points compared in the table.')
                        + table(['chamber', 'chain', 'F', 'singles split',
                                 'pairs resolved'], rr))
        keep_txt = '; '.join(f'{a}: {_pct(kept.get((a, 1), np.nan))} of singles, '
                             f'{_pct(kept.get((a, 2), np.nan))} of pairs' for a in ARMS)
        nrows = [tr([r.arm, r.population, int(r.n), f'{r["median"]:.1f}', f'{r.q99:.1f}'])
                 for _i, r in N.iterrows()]
        sec.append('<section><h2>R3 &amp; R4 · real overlays</h2>'
                   '<p>Two clean single donors of one chamber, file tag and trigger phase, '
                   'coincident in this view (|&Delta;t0|&nbsp;&lt;&nbsp;30&nbsp;ns), overlaid '
                   'with <code>--overlay replace</code> semantics. Each pair has a '
                   '<b>perfect-model twin</b>: both donors refitted alone on their own windows, '
                   'rebuilt from those fits and profiles by the forward model, with white noise '
                   'at each strip&rsquo;s pedestal level. Twin against real isolates model '
                   'mismatch, pair by pair. Thresholds come from each population&rsquo;s own '
                   'single donors. Production is the full chain (seeding, pairing, rescue, '
                   'joint fit at 300/120) on the same overlay; a pair counts when both donors '
                   'have a candidate within 1&nbsp;mm r.m.s. in this view. <b>Fixed '
                   'production</b> adds <code>WFT_TWO_TRACK_SCALE=two</code>, '
                   '<code>WFT_TWO_TRACK_SEARCH=grid</code>, no trigger and every candidate, '
                   '<b>with its threshold raised until it splits real single tracks no more '
                   'often than production does</b> ('
                   + ', '.join(f'{a}: F&nbsp;=&nbsp;{F:g}' for a, F in L.get('pick', {}).items())
                   + '; see the trade-off below).</p>'
                   '<p><b>Well-modelled windows only.</b> The windows are 61 strips wide, so '
                   'some hold charge the donor does not explain (another cluster, '
                   'saturation). A window is kept when its donor(s), refitted alone on it, '
                   f'reach &chi;&sup2;/dof&nbsp;&lt;&nbsp;{DONOR_CHI2DOF_MAX:g}: {keep_txt}. '
                   'The unrestricted numbers are in the second table; there the thresholds '
                   'are set by that foreign charge, not by the track.</p>'
                   + figure_html('r3_real_ladder', 'Pairs resolved in one view against '
                                 'separation, well-modelled windows.')
                   + table(['chamber', 'r.m.s. sep [mm]', 'pairs', 'twin, ideal',
                            'real, ideal', 'fixed production', 'production'], rows)
                   + '<h3>All windows, unrestricted</h3>'
                   + table(['chamber', 'r.m.s. sep [mm]', 'pairs', 'twin, ideal',
                            'real, ideal', 'fixed production', 'production'], rows_all)
                   + roc_html
                   + '<h3>The price of a split on real single tracks</h3>'
                   + figure_html('r3_null', 'Δχ² gained by the ideal two-track fit on '
                                 'well-modelled single tracks. Dotted: the 99th percentile, '
                                 'i.e. the threshold.')
                   + table(['chamber', 'population', 'windows', 'median Δχ²',
                            '99th pct'], nrows)
                   + '</section>')
    else:
        sec.append('<section><h2>R3 · real overlays</h2><p>Not yet run '
                   '(<code>two_track_limit real</code>).</p></section>')

    sec.append(validation_html())
    sec.append(ladder_html(L, fd))
    sec.append(
        '<section><h2>What this does not rule out</h2><ul>'
        '<li><b>The toy is one plane, equal charges, tied t0.</b> Unequal charges, '
        'a t0 offset and the second view all change the answer; the x/y pairing loss '
        '(the largest remaining loss at &ge;&nbsp;24&nbsp;mm on the bench) is not in it.</li>'
        '<li><b>tan&nbsp;0 is not in the data.</b> No clean donor has |tan&nbsp;&theta;| '
        '&lt; 0.05; chamber C&rsquo;s bundle puts a vertical track on one strip '
        '(&sigma;<sub>p0</sub>&nbsp;=&nbsp;0.039&nbsp;mm, not revalidated), so its tan&nbsp;0 '
        'column is the model&rsquo;s, not the chamber&rsquo;s.</li>'
        '<li><b>The ideal fit is ideal only in its search.</b> It still uses the forward '
        'model, so any shared model error (kernel, template, v<sub>drift</sub> prior) is in '
        'both the twin and the fit.</li>'
        '<li><b>The contract is measured on one sub-run</b> (run_145 stat090_0000, seven '
        'tags). Its clean-single sample is small: 0.66&nbsp;% of C&rsquo;s 1&nbsp;057 is 7 '
        'events, so neighbouring F values are not statistically distinct. The pick is a '
        'threshold, not a measurement of the false-split rate.</li>'
        '<li><b>Real triggers have no truth.</b> <code>split-ab</code> bounds the cost of '
        'a threshold (singles split, tracks lost). The gain (pairs resolved) comes only '
        'from the overlay bench.</li>'
        '<li><b>The fixes are opt-in.</b> They pass the contract, but nothing in production '
        'uses them until the bundles and condor environment carry them and the full pass is '
        'rerun.</li>'
        '<li><b>Donor truth is a fit.</b> A donor whose single-track fit is wrong gives a '
        'wrong label; the twin inherits it too.</li>'
        '</ul></section>')

    html = (f'<!doctype html><html lang="en"><head>{head("Two-track limit")}</head><body>'
            '<header><h1>Two tracks in one plane: the real limit, and how far we are '
            'from it</h1>'
            f'<p class="sub">run_145 stat090_0000, chambers A and C &middot; built '
            f'{dt.datetime.now():%Y-%m-%d %H:%M}</p></header>'
            f'<main>{"".join(sec)}</main>'
            '<footer><p>Generated by <code>sept26_prelim_analysis/'
            'make_two_track_limit_report.py</code> from <code>two_track_limit.py</code>. '
            'Handoff: <code>HANDOFF_TWO_TRACK_LIMIT.md</code>; working log: '
            '<code>TWO_TRACK_FIT_LOG.md</code>.</p></footer></body></html>')
    p = od / 'report.html'
    p.write_text(html)
    print(f'[report] wrote {p}')
    return p


if __name__ == '__main__':
    build()
