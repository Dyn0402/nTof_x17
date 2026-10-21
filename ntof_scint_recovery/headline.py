#!/usr/bin/env python3
"""
headline.py -- the final answer of the flash-recovery study: from when after the γ-flash a
scintillator pulse can be reconstructed accurately, and how biased and how well resolved
(in amplitude, energy and time) it is against time since the flash.

    python -m ntof_scint_recovery.headline

Three inputs, each answering one part:

1. **Reconstruction** (``inject.py`` -> ``inject_analysis.py``): pulses of known amplitude and
   time injected into the real flash tails and reconstructed.  Efficiency, amplitude bias and
   resolution, time bias and resolution; for every pulse (what an analysis gets) and for clean
   ones (the flash tail alone, without the beam-induced pile-up).
2. **Detector gain** (``gain_ref.py``): the amplitude spectra's own quantiles over their
   1-10 ms values.  The injection adds pulses after the photodetector, so it cannot see a gain
   sag; this can.
3. **Energy scale** (``gain_ref.py``): mV per MeVee per counter from the cosmic muons at
   30-32 ms and the muon calibration (ntof_calorimetry).

The energy bias of a reconstructed pulse = (1 + reconstruction bias) x gain - 1.

Writes ``out_dir('reco')``: ``headline.csv`` (t_rec per family, class, amplitude, criterion),
``performance.csv`` (bias / resolution / efficiency at fixed times, in mV, MeVee and ns),
``figures/*.png`` with a ``.csv`` beside each, and ``report.html``.
"""
from __future__ import annotations

import html
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ntof_scint_recovery.common import out_dir  # noqa: E402

FAMN = {'WAL': 'SiPM wall (per end)', 'PSS': 'plastic', 'LIQ': 'liquid'}
#: the amplitude that stands for "a lepton" in each family: ~1 MIP
REF_AMP = {'WAL': 30.0, 'PSS': 200.0, 'LIQ': 100.0}
TIMES = (1, 2, 3, 5, 10, 20, 50, 100, 200)
METHOD = 'multi'
WALL_MIP_MEV = 0.47
#: per-end wall MIP (mV) from the muon calibration (ntof_calorimetry/wp, sum of ends / 2), arm B
#: scaled from the in-situ cosmic ratio B/C (energy_scale.csv)
WALL_CAL_SUM_MPV = {'A': 43.68, 'C': 61.34, 'D': 59.09}


def energy_scale() -> dict:
    """Family mean and range: plastics mV/MeVee, walls mV per end per MIP."""
    E = pd.read_csv(out_dir('gain') / 'energy_scale.csv')
    p = E[E.fam == 'PSS'].copy()
    ratio = np.nanmedian(p.cosmic_med_mV / p.cal_mip_mV)
    p['use'] = np.where(np.isfinite(p.cal_mv_per_mevee), p.cal_mv_per_mevee, p.mv_per_mevee / ratio)
    w = E[E.fam == 'WAL'].copy()
    w['arm'] = w.counter.str[3]
    insitu = w.groupby('arm').cosmic_med_mV.median()
    mip_end = {a: WALL_CAL_SUM_MPV[a] / 2 for a in WALL_CAL_SUM_MPV}
    mip_end['B'] = insitu['B'] / insitu['C'] * mip_end['C']
    return dict(pss=dict(zip(p.counter, p.use)), pss_mean=float(p.use.mean()), pss_lo=float(p.use.min()),
                pss_hi=float(p.use.max()), pss_insitu_over_cal=float(ratio),
                wal=mip_end, wal_mean=float(np.mean(list(mip_end.values()))),
                wal_lo=float(min(mip_end.values())), wal_hi=float(max(mip_end.values())))


def to_energy(fam: str, mv: float, ES: dict) -> str:
    if fam == 'PSS':
        return f'{mv / ES["pss_mean"]:.2f} MeVee'
    if fam == 'WAL':
        m = mv / ES['wal_mean']
        return f'{m:.2f} MIP ({m * WALL_MIP_MEV:.2f} MeV)'
    return f'{mv:.0f} mV'


def at_time(g: pd.DataFrame, t: float, col: str):
    r = g[(g.t_lo_us <= t) & (g.t_hi_us > t)]
    return float(r[col].iloc[0]) if len(r) else np.nan


#: before this the spectrum is fast-neutron recoils (Geant4): its shape cannot separate gain from mix
GAIN_FROM_US = 2.5


def gain_at(G: pd.DataFrame, pop: str, cls: str, t: float) -> float:
    """The detector gain at t: q99 ratio of the population.  NaN before GAIN_FROM_US.  Walls: capped
    at 1 (their q99 sits a few % above 1 early because piled pulses are bigger: mix, not gain)."""
    if t < GAIN_FROM_US:
        return np.nan
    g = G[(G['pop'] == pop) & (G.cls == cls)].sort_values('t_us')
    g = g[np.isfinite(g.gain) & (g.t_us < 2e4) & (g.t_us > 1)]
    if not len(g):
        return np.nan
    v = float(np.interp(np.log(t), np.log(g.t_us), g.gain))
    return min(v, 1.0) if pop == 'wall bar' else v


def first_from(g: pd.DataFrame, ok: np.ndarray) -> float:
    bad = np.nonzero(~ok)[0]
    if not len(bad):
        return float(g.t_lo_us.iloc[0])
    if bad[-1] == len(ok) - 1:
        return np.nan
    return float(g.t_lo_us.iloc[bad[-1] + 1])


PSS_POP, WAL_POP = 'plastic (combined)', 'wall (no sag resolved)'
GAIN_POP = {'PSS': PSS_POP, 'WAL': WAL_POP}


def combine(G: pd.DataFrame) -> pd.DataFrame:
    """Add the two gain estimates used downstream.  Plastics: the mean of the q99 ratios of
    the coincident (behind a wall hit) and the singles populations, half their difference as
    the uncertainty (they are made by different particle mixes, so their spread is the mix
    systematic).  Walls: no sag the data resolve -- gain 1, with |q99 - 1| of the wall bars
    as the uncertainty (their q99 sits above 1 early: piled pulses are bigger)."""
    a = G[G['pop'] == 'plastic|wall'].set_index(['cls', 't_us'])
    b = G[G['pop'] == 'plastic singles'].set_index(['cls', 't_us'])
    j = a[['t_lo_us', 't_hi_us']].copy()
    j['gain'] = (a.gain + b.gain) / 2
    j['gain_unc'] = (a.gain - b.gain).abs() / 2
    j['pop'] = PSS_POP
    w = G[G['pop'] == 'wall bar'].set_index(['cls', 't_us'])
    k = w[['t_lo_us', 't_hi_us']].copy()
    k['gain'] = 1.0
    k['gain_unc'] = (w.gain - 1).abs()
    k['pop'] = WAL_POP
    G = G.copy()
    G['gain_unc'] = np.nan
    return pd.concat([G, j.reset_index(), k.reset_index()], ignore_index=True)


def gain_unc_at(G: pd.DataFrame, pop: str, cls: str, t: float) -> float:
    if t < GAIN_FROM_US or pop is None:
        return np.nan
    g = G[(G['pop'] == pop) & (G.cls == cls) & (G.t_us > 1) & (G.t_us < 2e4)].sort_values('t_us')
    g = g[np.isfinite(g.gain_unc)]
    return float(np.interp(np.log(t), np.log(g.t_us), g.gain_unc)) if len(g) else np.nan


def gain_times(G: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pop, cls), g in G.groupby(['pop', 'cls']):
        g = g[(g.t_us > GAIN_FROM_US) & (g.t_us < 1e4) & np.isfinite(g.gain)].sort_values('t_us')
        for lvl in (0.9, 0.95, 0.98):
            rows.append(dict(pop=pop, cls=cls, level=lvl,
                             t_us=first_from(g, (g.gain >= lvl).to_numpy())))     # a sag: one-sided
    return pd.DataFrame(rows)


def figs(S, G, M, R4, ES, fdir):
    fdir.mkdir(parents=True, exist_ok=True)
    out = []
    s = S[S.method == METHOD]
    cols = {'par': '#1f5fa8', 'ded': '#c8601a'}
    # 1. accurately reconstructed fraction (within 10 %) and efficiency, per family, all amplitudes
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    rows = []
    for k, fam in enumerate(('WAL', 'PSS', 'LIQ')):
        amps = sorted(s[s.fam == fam].amp_mV.unique())
        cm = plt.cm.viridis(np.linspace(0.05, 0.85, len(amps)))
        for A, c in zip(amps, cm):
            g = s[(s.fam == fam) & (s.cls == 'ded') & (s.amp_mV == A) & (s.tbin > 0)].sort_values('t_us')
            ax[k].plot(g.t_us, g.good10, '-o', ms=3, color=c, label=f'{A:g} mV ≈ {to_energy(fam, A, ES)}')
            for r in g.itertuples():
                rows.append(dict(fam=fam, cls='ded', amp_mV=A, t_us=r.t_us, good10=r.good10, good5=r.good5, eff=r.eff,
                                 f_piled=r.f_piled))
        gp = s[(s.fam == fam) & (s.cls == 'par') & (s.amp_mV == REF_AMP[fam]) & (s.tbin > 0)].sort_values('t_us')
        ax[k].plot(gp.t_us, gp.good10, '--', color='k', lw=1.5, label=f'{REF_AMP[fam]:g} mV, parasitic')
        ax[k].axhline(0.9, color='grey', ls=':')
        ax[k].set_xscale('log')
        ax[k].set_xlabel('time since the γ-flash at the detector [µs]')
        ax[k].set_title(FAMN[fam])
        ax[k].legend(fontsize=7, loc='lower right')
    ax[0].set_ylabel('fraction reconstructed within 10 % (dedicated)')
    fig.tight_layout()
    fig.savefig(fdir / 'good10_vs_t.png', dpi=110)
    pd.DataFrame(rows).to_csv(fdir / 'good10_vs_t.csv', index=False)
    plt.close(fig)
    out.append(('good10_vs_t.png', 'Pulses reconstructed within 10 % of their true amplitude',
                'Injected pulses of every amplitude (colours) against time since the flash, dedicated pulses; dashed: '
                'the ~1-MIP amplitude at parasitic intensity. Below the dotted 90 % line the reconstruction is not '
                'yet accurate for a typical pulse. Includes pile-up with real beam-induced pulses.'))
    # 2. amplitude bias: reconstruction, gain, total (reference amplitude)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    rows = []
    for k, fam in enumerate(('WAL', 'PSS', 'LIQ')):
        for cls in ('par', 'ded'):
            g = s[(s.fam == fam) & (s.cls == cls) & (s.amp_mV == REF_AMP[fam]) & (s.tbin > 0)].sort_values('t_us')
            ax[k].plot(g.t_us, 100 * g.bias, '-o', ms=3, color=cols[cls], label=f'reconstruction, {cls}')
            pop = GAIN_POP.get(fam)
            if pop:
                gg = np.array([gain_at(G, pop, cls, t) for t in g.t_us])
                tot = (1 + g.bias.to_numpy()) * gg - 1
                ax[k].plot(g.t_us, 100 * (gg - 1), ':', color=cols[cls], label=f'detector gain (data), {cls}')
                ax[k].plot(g.t_us, 100 * tot, '-', lw=2.5, color=cols[cls], alpha=0.5, label=f'total, {cls}')
            else:
                gg = tot = np.full(len(g), np.nan)
            for r, a, b in zip(g.itertuples(), gg, tot):
                rows.append(dict(fam=fam, cls=cls, amp_mV=REF_AMP[fam], t_us=r.t_us, reco_bias=r.bias, gain=a,
                                 total_bias=b))
        ax[k].axhspan(-3, 3, color='green', alpha=0.1)
        ax[k].set_xscale('log')
        ax[k].set_ylim(-60, 20)
        ax[k].set_xlabel('time since the γ-flash at the detector [µs]')
        ax[k].set_title(f'{FAMN[fam]}, {REF_AMP[fam]:g} mV ≈ {to_energy(fam, REF_AMP[fam], ES)}')
        ax[k].legend(fontsize=7, loc='lower right')
    ax[0].set_ylabel('amplitude bias [%]')
    fig.tight_layout()
    fig.savefig(fdir / 'bias_vs_t.png', dpi=110)
    pd.DataFrame(rows).to_csv(fdir / 'bias_vs_t.csv', index=False)
    plt.close(fig)
    out.append(('bias_vs_t.png', 'Amplitude (energy) bias against time',
                'Reconstruction bias (median, injection), the detector gain from the data (q99 of the amplitude spectrum '
                'over its 1-10 ms value; plastics behind a wall hit, wall bars) and their product. Green band ±3 %. '
                'The liquids have no gain reference: their spectrum shape changes too much.'))
    # 3. resolution: amplitude (relative) and time
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    rows = []
    for k, fam in enumerate(('WAL', 'PSS', 'LIQ')):
        amps = sorted(s[s.fam == fam].amp_mV.unique())
        cm = plt.cm.viridis(np.linspace(0.05, 0.85, len(amps)))
        for A, c in zip(amps, cm):
            g = s[(s.fam == fam) & (s.cls == 'ded') & (s.amp_mV == A) & (s.tbin > 0)].sort_values('t_us')
            ax[0, k].plot(g.t_us, 100 * g.r_s68, '-o', ms=3, color=c, label=f'{A:g} mV')
            ax[1, k].plot(g.t_us, g.d_s68, '-o', ms=3, color=c)
            ctl = s[(s.fam == fam) & (s.cls == 'ded') & (s.amp_mV == A) & (s.tbin == 0)]
            if len(ctl):
                ax[0, k].axhline(100 * ctl.r_s68.iloc[0], color=c, ls=':', lw=1)
                ax[1, k].axhline(ctl.d_s68.iloc[0], color=c, ls=':', lw=1)
            for r in g.itertuples():
                rows.append(dict(fam=fam, amp_mV=A, t_us=r.t_us, r_s68=r.r_s68, d_s68_ns=r.d_s68,
                                 r_s68_clean=r.r_s68_clean, d_s68_clean_ns=r.d_s68_clean))
        ax[0, k].set_yscale('log')
        ax[1, k].set_yscale('log')
        ax[0, k].set_title(FAMN[fam])
        ax[1, k].set_xscale('log')
        ax[1, k].set_xlabel('time since the γ-flash at the detector [µs]')
        ax[0, k].legend(fontsize=7)
    ax[0, 0].set_ylabel('amplitude resolution σ68/A [%]')
    ax[1, 0].set_ylabel('time resolution σ68 [ns]')
    fig.tight_layout()
    fig.savefig(fdir / 'resolution_vs_t.png', dpi=110)
    pd.DataFrame(rows).to_csv(fdir / 'resolution_vs_t.csv', index=False)
    plt.close(fig)
    out.append(('resolution_vs_t.png', 'Amplitude and time resolution against time (dedicated)',
                'Half the 16-84 % range of A_fit/A and of t_fit - t0, all injected pulses that were found. Dotted: the same '
                'before the flash (noise only). The excess is pile-up with beam-induced pulses; clean pulses alone are in '
                'performance.csv.'))
    # 4. pile-up probability
    fig, ax = plt.subplots(figsize=(8, 4.8))
    rows = []
    for fam, c in (('WAL', '#7a52b8'), ('PSS', '#c8601a'), ('LIQ', '#2f8a5b')):
        for cls, ls in (('ded', '-'), ('par', '--')):
            g = s[(s.fam == fam) & (s.cls == cls) & (s.amp_mV == REF_AMP[fam]) & (s.tbin > 0)].sort_values('t_us')
            ax.plot(g.t_us, g.f_piled, ls, color=c, label=f'{FAMN[fam]}, {cls}')
            rows += [dict(fam=fam, cls=cls, t_us=r.t_us, f_piled=r.f_piled) for r in g.itertuples()]
    ax.set_xscale('log')
    ax.set_xlabel('time since the γ-flash at the detector [µs]')
    ax.set_ylabel('P(a real pulse ≥ 5 % of A overlaps the pulse)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fdir / 'pileup_vs_t.png', dpi=110)
    pd.DataFrame(rows).to_csv(fdir / 'pileup_vs_t.csv', index=False)
    plt.close(fig)
    out.append(('pileup_vs_t.png', 'Pile-up probability',
                'Fraction of injected ~1-MIP pulses that land within a pulse width (template above 5 % of peak) of a real '
                'pulse at least 5 % of their size. This is the beam-induced occupancy; it scales with the pulse rate.'))
    # 5. gain test: quantile ratios
    Q = pd.read_csv(out_dir('gain') / 'gain_quantiles.csv')
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for k, pop in enumerate(('plastic|wall', 'plastic singles', 'wall bar')):
        g = Q[(Q['pop'] == pop) & (Q.cls == 'ded') & (Q.t_us > 1) & (Q.t_us < 2e4)].sort_values('t_us')
        for qn, c in zip(('q50', 'q75', 'q90', 'q95', 'q99'), plt.cm.plasma(np.linspace(0.05, 0.8, 5))):
            ax[k].plot(g.t_us, g[qn], '-o', ms=3, color=c, label=qn)
        ax[k].axhline(1, color='k', lw=0.8)
        ax[k].set_xscale('log')
        ax[k].set_ylim(0.2, 1.8)
        ax[k].set_title(f'{pop}, dedicated')
        ax[k].set_xlabel('time since the γ-flash at the detector [µs]')
        ax[k].legend(fontsize=7)
    ax[0].set_ylabel('quantile / its 1-10 ms value')
    fig.tight_layout()
    fig.savefig(fdir / 'gain_quantiles.png', dpi=110)
    Q.to_csv(fdir / 'gain_quantiles.csv', index=False)
    plt.close(fig)
    out.append(('gain_quantiles.png', 'Gain or particle mix? The spectrum quantiles',
                'Each quantile of the amplitude spectrum over its 1-10 ms value. A gain change moves all of them by the same '
                'factor; a change of particle mix moves some. Plastics behind a wall hit: all quantiles drop together '
                'before ~10 µs (a gain sag), after ~20 µs q50 and q99 are back while q75-q95 stay low to ~0.3 ms '
                '(the mix).'))
    # 6. Geant4 mix
    try:
        R = pd.read_csv(out_dir('g4mix') / 'g4_rates.csv')
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
        for fam, c in (('wall', '#7a52b8'), ('plastic', '#c8601a'), ('liquid', '#2f8a5b')):
            g = R[R.fam == fam]
            t = np.sqrt(g.t_lo_us * g.t_hi_us)
            ax[0].plot(t, g.f_proton, '-', color=c, label=f'{fam}: proton recoils')
            ax[1].plot(t, g.g4_rate_per_us, '-', color=c, label=f'{fam}: Geant4 (per counter)')
            ax[1].plot(t, g.data_rate_per_us, '--', color=c, label=f'{fam}: data (per channel)')
        ax[0].set_xscale('log')
        ax[0].set_ylabel('fraction of clusters > 0.1 MeVee')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_ylabel('pulses per µs per bunch (dedicated)')
        for a in ax:
            a.set_xlabel('time since the γ-flash at the detector [µs]')
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fdir / 'g4_mix.png', dpi=110)
        R.to_csv(fdir / 'g4_mix.csv', index=False)
        plt.close(fig)
        out.append(('g4_mix.png', 'Geant4: what arrives, and how often',
                    'Left: the share of proton recoils (fast neutrons) among deposits above 0.1 MeVee; the rest are e± from γ. '
                    'Right: the pulse rate, Geant4 (apparatus only, EAR2 neutron beam, 8.5×10¹² ppp) against the data. The '
                    'Geant4 wall is a whole arm (20 bars), the data a single bar end.'))
    except FileNotFoundError:
        pass
    return out


def data_rate(fam: str, cls: str) -> pd.DataFrame:
    """Measured pulse rate per live channel (pulses/µs/bunch above 4 mV, production runs) on the
    fine log-time grid of the pulse histograms."""
    global _H
    if _H is None:
        from ntof_scint_recovery.coinc_analysis import load
        _H = load(out_dir('pulsehist_v3'))
    H = _H
    te, ae = H['t_edges_us'], H['a_edges']
    chans = [tuple(c) for c in H['channels']]
    dets = {'WAL': range(4), 'PSS': range(4, 8), 'LIQ': range(8, 12)}[fam]
    k = 1 if cls == 'ded' else 0
    live = [i for i, c in enumerate(chans) if c[0] in dets and H['HA'][k, i].sum() > 0]
    n = H['HA'][k, live][:, :100, ae[:-1] >= 4.0].sum(axis=(0, 2))
    t = np.sqrt(te[:100] * te[1:101])
    return pd.DataFrame(dict(t_us=t, rate=n / H['nb'][k] / len(live) / np.diff(te[:101])))


_H = None


def occupancy_model(s: pd.DataFrame) -> pd.DataFrame:
    """1 - good10 = 1 - exp(-a R(t)) with R the measured pulse rate per channel, fitted on the
    injection bins from FIT_FROM_US to the end of the head block; then carried to 10 ms with the
    measured rate.  t_good90_model = first time from which the predicted good10 stays >= 0.9."""
    rows = []
    for (fam, cls, A), g in s[s.tbin > 0].groupby(['fam', 'cls', 'amp_mV']):
        R = data_rate(fam, cls)
        g = g[(g.t_us >= FIT_FROM_US[fam]) & np.isfinite(g.good10)]
        if len(g) < 3:
            continue
        r = np.interp(np.log(g.t_us), np.log(R.t_us), R.rate)
        y = -np.log(np.clip(g.good10.to_numpy(), 1e-3, 1))
        a = float(np.sum(r * y) / np.sum(r * r))                     # least squares through 0
        pred = np.exp(-a * R.rate.to_numpy())
        ok = (pred >= 0.9) | (R.t_us.to_numpy() < 1)
        bad = np.nonzero(~ok & (R.t_us.to_numpy() < 1e4))[0]
        t90 = float(R.t_us.iloc[bad[-1] + 1]) if len(bad) and bad[-1] + 1 < len(R) else np.nan
        resid = float(np.sqrt(np.mean((np.exp(-a * r) - g.good10.to_numpy()) ** 2)))
        rows.append(dict(fam=fam, cls=cls, amp_mV=A, a_us=a, fit_rms=resid, t_good90_model_us=t90,
                         rate_at_t90=float(np.interp(np.log(t90), np.log(R.t_us), R.rate)) if np.isfinite(t90) else np.nan,
                         n_fit=len(g)))
    return pd.DataFrame(rows)


#: the occupancy model is fitted where the flash tail itself no longer matters (clean pulses unbiased)
FIT_FROM_US = {'WAL': 10.0, 'PSS': 5.0, 'LIQ': 5.0}


def main() -> int:
    S = pd.read_csv(out_dir('inject') / 'inject_summary.csv')
    T = pd.read_csv(out_dir('inject') / 'inject_trec.csv')
    G = combine(pd.read_csv(out_dir('gain') / 'gain_curve.csv'))
    ES = energy_scale()
    GT = gain_times(G)
    o = out_dir('reco')
    # headline: per family, class, amplitude
    s = S[S.method == METHOD]
    rows = []
    for (fam, cls, A), g in s[s.tbin > 0].groupby(['fam', 'cls', 'amp_mV']):
        g = g.sort_values('tbin')
        tr = T[(T.method == METHOD) & (T.fam == fam) & (T.cls == cls) & (T.amp_mV == A)].set_index('criterion').t_rec_us
        pop = GAIN_POP.get(fam)
        g95 = GT[(GT['pop'] == pop) & (GT.cls == cls) & (GT.level == 0.95)].t_us
        g95 = float(g95.iloc[0]) if pop and len(g95) else np.nan
        t_acc = tr.get('all_good10_0.9', np.nan)
        rows.append(dict(fam=fam, cls=cls, amp_mV=A, energy=to_energy(fam, A, ES),
                         t_tail_us=tr.get('nominal', np.nan), t_tail_loose_us=tr.get('loose', np.nan), t_good10_90_us=t_acc,
                         t_good5_90_us=tr.get('all_good5_0.9', np.nan), t_gain95_us=g95,
                         t_headline_us=np.nanmax([t_acc, g95]) if np.isfinite(t_acc) else np.nan,
                         head_end_us=float(g.t_hi_us.iloc[-1])))
    Hd = pd.DataFrame(rows)
    OM = occupancy_model(s)
    OM.to_csv(o / 'occupancy_model.csv', index=False)
    Hd = Hd.merge(OM[['fam', 'cls', 'amp_mV', 't_good90_model_us', 'a_us', 'fit_rms']], on=['fam', 'cls', 'amp_mV'], how='left')
    # beyond the head block, the model stands in for the injection
    Hd['t_headline_us'] = np.where(np.isfinite(Hd.t_headline_us), Hd.t_headline_us,
                                   np.nanmax(np.c_[Hd.t_good90_model_us, Hd.t_gain95_us], axis=1))
    Hd['headline_from_model'] = ~np.isfinite(Hd.t_good10_90_us)
    Hd.to_csv(o / 'headline.csv', index=False)
    GT.to_csv(o / 'gain_times.csv', index=False)
    # performance at fixed times
    prow = []
    for (fam, cls, A), g in s.groupby(['fam', 'cls', 'amp_mV']):
        pop = GAIN_POP.get(fam)
        scale = ES['pss_mean'] if fam == 'PSS' else (ES['wal_mean'] / WALL_MIP_MEV if fam == 'WAL' else np.nan)
        for t in TIMES:
            b = at_time(g, t, 'bias')
            gn = gain_at(G, pop, cls, t) if pop else np.nan
            prow.append(dict(fam=fam, cls=cls, amp_mV=A, E_MeV=A / scale, t_us=t,
                             eff=at_time(g, t, 'eff'), good10=at_time(g, t, 'good10'), good5=at_time(g, t, 'good5'),
                             f_piled=at_time(g, t, 'f_piled'),
                             reco_bias=b, gain=gn, gain_unc=gain_unc_at(G, pop, cls, t), total_bias=(1 + b) * gn - 1 if pop else np.nan,
                             r_s68=at_time(g, t, 'r_s68'), sigma_E_MeV=at_time(g, t, 'r_s68') * A / scale,
                             r_s68_clean=at_time(g, t, 'r_s68_clean'), bias_clean=at_time(g, t, 'bias_clean'),
                             d_bias_ns=at_time(g, t, 'd_bias_ns'), d_s68_ns=at_time(g, t, 'd_s68'),
                             d_s68_clean_ns=at_time(g, t, 'd_s68_clean'),
                             ctrl_r_s68=float(g[g.tbin == 0].r_s68.iloc[0]), ctrl_d_s68_ns=float(g[g.tbin == 0].d_s68.iloc[0])))
    P = pd.DataFrame(prow)
    P.to_csv(o / 'performance.csv', index=False)
    F = figs(S, G, P, None, ES, o / 'figures')
    write_report(o, Hd, P, GT, ES, F)
    pd.set_option('display.width', 250)
    print(Hd.round(2).to_string(index=False))
    print(GT.round(2).to_string(index=False))
    return 0


def fmt_t(us):
    if not np.isfinite(us):
        return '—'
    return f'{us:.1f} µs' if us < 10 else (f'{us:.0f} µs' if us < 1000 else f'{us / 1e3:.1f} ms')


def write_report(o, Hd, P, GT, ES, F):
    def tab(df, cols, names):
        th = ''.join(f'<th>{html.escape(n)}</th>' for n in names)
        trs = ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in r) + '</tr>' for r in df[cols].itertuples(index=False))
        return f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'
    h = Hd.copy()
    for c in ('t_tail_us', 't_good10_90_us', 't_good5_90_us', 't_gain95_us', 't_headline_us', 'head_end_us'):
        h[c] = h[c].map(fmt_t)
    ref = h[h.apply(lambda r: r.amp_mV == REF_AMP[r.fam], axis=1)]
    p = P[(P.cls == 'ded') & P.apply(lambda r: r.amp_mV == REF_AMP[r.fam], axis=1)].copy()
    for c in ('reco_bias', 'gain', 'total_bias', 'r_s68', 'good10', 'f_piled', 'eff'):
        p[c] = p[c].map(lambda v: '—' if not np.isfinite(v) else (f'{v:.2f}' if c == 'gain' else f'{100 * v:+.1f} %' if 'bias' in c else f'{100 * v:.0f} %'))
    for c in ('sigma_E_MeV', 'd_s68_ns'):
        p[c] = p[c].map(lambda v: '—' if not np.isfinite(v) else f'{v:.2f}')
    gt = GT[GT.cls == 'ded'].copy()
    gt['t_us'] = gt.t_us.map(fmt_t)
    figs_html = ''.join(f'<figure><img src="figures/{f}" style="max-width:100%"><figcaption><b>{html.escape(t)}.</b> '
                        f'{html.escape(c)}</figcaption></figure>' for f, t, c in F)
    doc = f'''<!doctype html><html><head><meta charset="utf-8"><title>Flash recovery: reconstruction</title>
<style>body{{font-family:Helvetica,Arial,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#1c2230;background:#fff}}
table{{border-collapse:collapse;margin:12px 0;font-size:14px}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
th{{background:#f0eee8}}figure{{margin:24px 0}}figcaption{{font-size:14px;color:#444}}</style></head><body>
<h1>Scintillator pulses after the γ-flash: from when, and how well</h1>
<p><b>Verdict.</b> See the table: the time from which a ~1-MIP pulse is reconstructed within 10 % in at least 90 % of cases
(beam-induced pile-up included) <i>and</i> the detector gain is back within 5 %. The flash tail itself (baseline, shape,
clipping) stops mattering much earlier ("tail" column, clean pulses); after that the limits are the beam-induced pulse
rate (pile-up) and, on the plastics, a PMT gain sag of ~10 µs.</p>
{tab(ref, ['fam', 'cls', 'amp_mV', 'energy', 't_tail_us', 't_good10_90_us', 't_gain95_us', 't_headline_us', 'head_end_us'],
     ['family', 'intensity', 'amplitude [mV]', 'energy', 'tail recovered (clean)', '≥90 % within 10 %', 'gain ≥ 0.95', 'headline', 'measured to'])}
<p>"—" = not reached within the unsuppressed head block (walls ~40 µs, liquids ~20-150 µs, plastics ~200 µs), the only
part of the trace where pulses can be injected.</p>
<h2>Performance at fixed times (dedicated, ~1 MIP)</h2>
{tab(p, ['fam', 't_us', 'eff', 'good10', 'f_piled', 'reco_bias', 'gain', 'total_bias', 'r_s68', 'sigma_E_MeV', 'd_s68_ns'],
     ['family', 't [µs]', 'found', 'within 10 %', 'piled', 'reco bias', 'gain', 'energy bias', 'σ68(A)/A', 'σ68(E) [MeV(ee)]', 'σ68(t) [ns]'])}
<p>Energy: plastics {ES['pss_mean']:.0f} mV/MeVee on average ({ES['pss_lo']:.0f}-{ES['pss_hi']:.0f} over the 8 bars, muon
calibration; in-situ cosmics at 30-32 ms read {ES['pss_insitu_over_cal']:.2f} of it); wall ends {ES['wal_mean']:.0f} mV per MIP
({ES['wal_lo']:.0f}-{ES['wal_hi']:.0f}), MIP = {WALL_MIP_MEV} MeV in 3 mm. Liquids: no usable energy scale.</p>
<h2>Gain recovery times (data, dedicated)</h2>
{tab(gt, ['pop', 'level', 't_us'], ['population', 'gain level', 'from'])}
<h2>Figures</h2>{figs_html}
<h2>What this does not rule out</h2><ul>
<li>The gain reference assumes the top of the spectrum (q99) is set by the same particles at every time. Geant4 of the
apparatus says the population above 0.1 MeVee is e± from capture γ from ~2 µs on, with a stable shape to ~1 ms; but the
data spectrum is harder than the simulation (the hall and the in-beam γ are not simulated), so a slow change of the hard
component would read as gain.</li>
<li>Before ~2 µs the plastics and liquids see fast-neutron proton recoils; the spectrum there cannot separate gain from mix.</li>
<li>Injection is possible only inside the unsuppressed head block; later times are judged from the trend and the rates.</li>
<li>Injected pulses use the family template; per-channel shape differences are common to the control and cancel to first order.</li>
<li>The pile-up depends on the pulse rate: the numbers hold for this apparatus at these intensities.</li></ul>
</body></html>'''
    (o / 'report.html').write_text(doc)


if __name__ == '__main__':
    sys.exit(main())
