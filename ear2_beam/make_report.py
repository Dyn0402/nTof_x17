#!/usr/bin/env python3
"""
make_report.py -- the EAR2 beam at our station: figures + report.html.

Reads optical_transport.py's outputs in /media/dylan/data/x17/ear2_beam
(summary_<variant>.csv, profile_<variant>.npz) and the two flux files, and
writes <out>/report.html with figures/ beside it.

    python -m ear2_beam.make_report
"""
from __future__ import annotations

import datetime as dt
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'sept26_prelim_analysis'))
import figstyle as fs  # noqa: E402

OUT = Path('/media/dylan/data/x17/ear2_beam')
FIG = OUT / 'figures'
LAMBDA = Path.home() / 'CLionProjects/MX17_Full_Geant/data/lamda2DvsEn_EAR2.root'
FLUX3 = Path.home() / 'CLionProjects/MX17_Full_Geant/data/fluxEAR2-Ph3_in_different_units.root'
FLUX4 = REPO / 'ear2_beam/data/Final_evaluated_flux_Phase4_EAR2.root'
#: PPACMon beam-interception factors at 19.95 m, thermal, small collimator
#: (Pavon-Rodriguez et al., EPJ A 61 (2025), arXiv:2505.00042, Fig. 14 top, read off the plot)
BIF_MEAS = {1.5: 0.215, 2.0: 0.355, 3.0: 0.655, 4.0: 0.875}
VARIANTS = {'july2021': 'small collimator, as designed (ear2_july2021.inp)',
            'survey2017': 'small, 1st collimator at the 2017 survey (-1.4, -1.1) cm',
            'miss2mm': 'small, 2nd-collimator exit 2 mm off (n_TOF sensitivity case)',
            'fission': 'BIG second collimator (60 mm exit)'}
COL = {'july2021': fs.INK, 'survey2017': '#0072B2', 'miss2mm': '#009E73', 'fission': fs.COPPER}


def load():
    S = {v: pd.read_csv(OUT / f'summary_{v}.csv') for v in VARIANTS if (OUT / f'summary_{v}.csv').exists()}
    P = {v: np.load(OUT / f'profile_{v}.npz') for v in VARIANTS if (OUT / f'profile_{v}.npz').exists()}
    return S, P


def lambda_cdf():
    """What the old model used: Lambda2D's row at 25 meV as a 'radius' CDF (mm)."""
    import uproot
    v, ex, ey = uproot.open(LAMBDA)['Lambda2D'].to_numpy()
    E = np.sqrt(ex[:-1] * ex[1:])
    i = np.argmin(abs(np.log10(E) - np.log10(0.0253)))
    c = np.cumsum(v[i])
    return 10 * ey[1:], c / c[-1]


def profile_cdf(h, e_mm):
    c = 0.5 * (e_mm[1:] + e_mm[:-1])
    X, Z = np.meshgrid(c, c, indexing='ij')
    w = h.ravel()
    mx, mz = (w * X.ravel()).sum() / w.sum(), (w * Z.ravel()).sum() / w.sum()
    r = np.hypot(X.ravel() - mx, Z.ravel() - mz)
    o = np.argsort(r)
    return r[o], np.cumsum(w[o]) / w.sum()


def fig_wrong(S, P):
    fs.use()
    fig, ax = fs.figure(fs.FIG)
    rl, cl = lambda_cdf()
    ax.plot(rl, cl, color=fs.TRACK, lw=1.8, label='Lambda2D read as a radius (old model)')
    e = 10 * P['july2021']['edges_cm']
    rows = []
    for L, ls in ((19.5, '-'), (19.95, '--')):
        r, c = profile_cdf(P['july2021'][f'beam_{L:g}'], e)
        ax.plot(r, c, color=fs.INK, ls=ls, lw=1.6, label=f'EAR2 transport, {L:g} m')
        rows.append(pd.DataFrame(dict(model=f'transport_{L:g}', r_mm=r[::50], cum=c[::50])))
    ax.axvline(10.0, color=fs.MUTED, lw=1, ls=':')
    ax.text(10.3, 0.97, '³He bore (r = 10 mm)', color=fs.MUTED, fontsize=9, va='top')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('distance from the beam centre (mm)')
    ax.set_ylabel('fraction of the beam inside')
    ax.legend(loc='lower right', frameon=False)
    fs.title(ax, 'The old beam was the resolution function: ~5× too narrow',
             'thermal neutrons; the old curve is Lambda2D (λ, cm) at 25 meV')
    rows.append(pd.DataFrame(dict(model='lambda2d_25meV', r_mm=rl[::30], cum=cl[::30])))
    fs.save(fig, FIG / 'old_vs_real', data=pd.concat(rows))


def fig_maps(P):
    import matplotlib.pyplot as plt
    fs.use()
    Ls = (19.0, 19.5, 19.95, 20.5)
    fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.9), sharey=True, constrained_layout=True)
    e = 10 * P['july2021']['edges_cm']
    rows = []
    for ax, L in zip(axs, Ls):
        h = P['july2021'][f'beam_{L:g}']
        h = h / h.max()
        ax.pcolormesh(e, e, h.T, cmap='viridis', vmin=0, vmax=1, rasterized=True)
        ax.add_patch(plt.Circle((0, 0), 10.0, fill=False, color='w', lw=1.0, ls='--'))
        ax.set_aspect('equal')
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_title(f'{L:g} m', fontsize=10)
        ax.set_xlabel('x (mm)')
        c = 0.5 * (e[1:] + e[:-1])
        rows.append(pd.DataFrame(dict(L=L, x_mm=c, profile_x=h.sum(1) / h.sum(1).max())))
    axs[0].set_ylabel('z (mm)')
    fig.suptitle('Thermal beam (14–124 meV) at four heights, small collimator; dashed: the ³He bore',
                 fontsize=10.5, x=0.01, ha='left')
    fs.save(fig, FIG / 'maps', data=pd.concat(rows))


def fig_bif(S):
    fs.use()
    fig, ax = fs.figure(fs.HALF)
    s = S['july2021']
    s = s[(s.band == 'beam') & (s.L == 19.95)].iloc[0]
    d = np.array(sorted(BIF_MEAS))
    sim = np.array([s[f'bif_{x:g}cm'] for x in d])
    ax.plot(d, [BIF_MEAS[x] for x in d], 'o', color=fs.INK, ms=7, label='PPACMon (measured)')
    ax.plot(d, sim, 's', mfc='none', color=fs.TRACK, ms=9, mew=1.5, label='this transport')
    ax.set_xlabel('sample diameter (cm), centred')
    ax.set_ylabel('beam interception factor')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', frameon=False)
    fs.title(ax, 'Validation at 19.95 m', 'thermal, small collimator')
    fs.save(fig, FIG / 'bif_validation',
            data=pd.DataFrame(dict(d_cm=d, measured=[BIF_MEAS[x] for x in d], transport=sim)))


def fig_height(S):
    fs.use()
    fig, ax = fs.figure(fs.FIG)
    rows = []
    for v, s in S.items():
        g = s[s.band == 'beam'].sort_values('L')
        ax.plot(g.L, g.bif_2cm, '-o', color=COL[v], ms=4, lw=1.5, label=VARIANTS[v])
        rows.append(g[['L', 'bif_2cm', 'r50_mm', 'r90_mm', 'fwhm_x_mm', 'fwhm_z_mm', 'cx_mm', 'cz_mm']].assign(variant=v))
    ax.axvspan(19.45, 20.0, color=fs.GRID, zorder=0)
    ax.text(19.47, 0.03, 'our station:\n19.5 (repo) … 19.95 (n_TOF ref.)', fontsize=8.5, color=fs.MUTED)
    ax.set_xlabel('flight path (m)')
    ax.set_ylabel('fraction of the beam on a 20 mm bore')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', frameon=False, fontsize=8.5)
    fs.title(ax, 'How much of the beam the ³He bore sees, by height and collimator',
             'centred bore; thermal; the old model gave 0.95')
    fs.save(fig, FIG / 'bore_fraction_vs_height', data=pd.concat(rows))


def flux_ratio():
    import uproot
    p3 = uproot.open(FLUX3)['flux_n_pulse_isolet_100bpd'].to_numpy()
    p4 = uproot.open(FLUX4)['h_flux_ear2'].to_numpy()

    def n_in(h, lo, hi):
        v, e = h
        a, b = np.clip(e[:-1], lo, hi), np.clip(e[1:], lo, hi)
        m = b > a
        return float((v[m] * np.log(b[m] / a[m])).sum())

    rows = []
    for lab, lo, hi in (('beam window 4–12 ms (14–124 meV)', 0.0138, 0.124),
                        ('late 20–75 ms (0.35–5 meV)', 3.5e-4, 4.97e-3),
                        ('all below 1 keV', 1e-3, 1e3), ('1–10 eV', 1, 10)):
        a, b = n_in(p3, lo, hi), n_in(p4, lo, hi)
        rows.append(dict(window=lab, ph3=a, ph4=b, ratio=b / a))
    return pd.DataFrame(rows)


def table(df, fmt=None):
    fmt = fmt or {}
    h = '<table><tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns) + '</tr>'
    for _, r in df.iterrows():
        h += '<tr>' + ''.join(f'<td>{fmt.get(c, "{}").format(r[c]) if not isinstance(r[c], str) else html.escape(r[c])}</td>'
                              for c in df.columns) + '</tr>'
    return h + '</table>'


def build():
    S, P = load()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_wrong(S, P)
    fig_maps(P)
    fig_bif(S)
    fig_height(S)
    F = flux_ratio()
    s = S['july2021']
    b = s[s.band == 'beam'].set_index('L')
    head = pd.DataFrame([dict(height_m=L, r50_mm=b.loc[L, 'r50_mm'], fwhm_mm=0.5 * (b.loc[L, 'fwhm_x_mm'] + b.loc[L, 'fwhm_z_mm']),
                              r99_mm=b.loc[L, 'r99_mm'], bore_20mm=b.loc[L, 'bif_2cm'], dia_30mm=b.loc[L, 'bif_3cm'],
                              cone_mrad_per_cm=b.loc[L, 'slope_x_per_cm_mrad'])
                         for L in (19.0, 19.5, 19.95, 20.5)])
    var = pd.concat([g[(g.band == 'beam') & (g.L.isin([19.5, 19.95]))].assign(variant=VARIANTS[v])
                     for v, g in S.items()])[['variant', 'L', 'cx_mm', 'cz_mm', 'r50_mm', 'bif_2cm']]
    bands = pd.read_csv(OUT / 'ear2_profile_july2021_L19.3.csv')
    bands = bands[bands.band % 4 == 2][['e_lo', 'e_hi', 'r50_mm', 'fwhm_x_mm', 'bif_2cm', 'cx_mm', 'cz_mm',
                                        'slope_x_per_cm_mrad']]
    css = ('body{font-family:system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1b2430;'
           'line-height:1.5}table{border-collapse:collapse;margin:8px 0 18px;font-size:14px}td,th{border:1px solid '
           '#d5dae1;padding:3px 8px;text-align:right}th{background:#f2f4f7}img{max-width:100%}.verdict{background:'
           '#f6f0f7;border-left:4px solid #8a3f8f;padding:10px 14px}.cap{color:#6a7583;font-size:14px}')
    f3 = {c: '{:.3f}' for c in ('bore_20mm', 'dia_30mm', 'bif_2cm', 'ratio')}
    f1 = {c: '{:.1f}' for c in ('r50_mm', 'fwhm_mm', 'r99_mm', 'cx_mm', 'cz_mm', 'fwhm_x_mm', 'cone_mrad_per_cm', 'slope_x_per_cm_mrad')}
    fmt = {**f1, **f3, 'height_m': '{:g}', 'L': '{:g}', 'e_lo': '{:.3g}', 'e_hi': '{:.3g}', 'ph3': '{:.3g}', 'ph4': '{:.3g}'}
    b195, b1995 = b.loc[19.5], b.loc[19.95]
    fis = S.get('fission')
    fis_line = ''
    if fis is not None:
        fb = fis[fis.band == 'beam'].set_index('L')
        fis_line = (f' With the <b>big</b> collimator the beam would be ~{0.5 * (fb.loc[19.5, "fwhm_x_mm"] + fb.loc[19.5, "fwhm_z_mm"]):.0f} mm '
                    f'FWHM and the bore would see {fb.loc[19.5, "bif_2cm"]:.2f}.')
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>EAR2 beam at our station</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{css}</style></head><body>
<h1>The EAR2 neutron beam at our station</h1>
<p class="cap">Generated {dt.date.today().isoformat()} by <code>ear2_beam/make_report.py</code>. Inputs: n_TOF phase-4 FLUKA phase space
(<code>/eos/experiment/ntof/simul/phase4/data_fluka_4.5.1</code>, 400 of 3003 runs, 5.0 M neutrons), n_TOF collimation files
(<code>simul/transport/input</code>), the Phase-4 evaluated flux (<code>2021_Commissioning/flux/EAR2</code>).</p>
<div class="verdict"><p><b>Verdict.</b> Every beam footprint we used until 14 Oct 2026 (the Geant4 neutron gun, the capsule-pose model)
took the beam's radius from <code>Lambda2D</code>, which is the n_TOF <b>resolution function</b> (λ, the moderation length in cm;
<code>transport.cc</code> HIST_LAMBDA), not a profile. The real thermal beam, from n_TOF's own phase space through the EAR2
collimation by the method of n_TOF's optical transport code, reproduces the measured PPACMon interception factors at 19.95 m
to 1–2 %. At 19.5 m it has a half-beam radius of {b195.r50_mm:.1f} mm (old model: 2.5 mm), FWHM {0.5 * (b195.fwhm_x_mm + b195.fwhm_z_mm):.0f} mm,
and ends sharply at r ≈ {b195.r99_mm:.0f} mm. A centred 20 mm ³He bore sees <b>{b195.bif_2cm:.2f}</b> of the beam at 19.5 m and
<b>{b1995.bif_2cm:.2f}</b> at 19.95 m (old model: 0.95).{fis_line} Below 1 keV the profile does not depend on energy. The beam is centred
on the pipe axis to 1–3 mm in every misalignment case n_TOF considered; a 10 mm offset would need the defining aperture 1 cm off.</p></div>
<h2>Headline numbers (small collimator, thermal)</h2>{table(head, fmt)}
<p class="cap">bore_20mm / dia_30mm: fraction of the beam inside a centred 20 / 30 mm disk. cone: ray angle per cm of distance from
the axis (the beam diverges from an apex ~2.5 m below the station), on top of a local spread of 2–3 mrad.</p>
<h2>What changed</h2><img src="figures/old_vs_real.png" alt="cumulative beam fraction vs radius, old vs real">
<p class="cap">The old model put half the beam inside 2.5 mm; the real beam puts half inside 10–12.5 mm.</p>
<h2>Validation against the measurement</h2><img src="figures/bif_validation.png" alt="BIF measured vs transport">
<p class="cap">PPACMon values read off Fig. 14 of Pavón-Rodríguez et al., EPJ A 61 (2025), arXiv:2505.00042 (aligned on the beam centre).</p>
<h2>The beam with height</h2><img src="figures/maps.png" alt="2D beam maps at four heights">
<img src="figures/bore_fraction_vs_height.png" alt="bore fraction vs height">
<p class="cap">The beam grows ~4 mm in half-radius per metre, so where the capsule sits along the beam matters: our analysis code
uses 19.5 m, n_TOF's capture reference position is 19.95 m. The capsule's real height is not known.</p>
<h2>Energy dependence (19.3 m, the Geant4 gun plane for a capsule at 19.5 m)</h2>{table(bands, fmt)}
<h2>Collimation variants</h2>{table(var, fmt)}
<p class="cap">cx, cz: beam centroid against the nominal axis. The FLUKA source alone puts it ~1 mm off; known misalignments add 1–3 mm.</p>
<h2>The flux: Phase 4, not Phase 3</h2>{table(F, fmt)}
<p class="cap">Neutrons per pulse (7×10¹² protons). The Geant4 gun and the capsule model used the Phase-3 (pre-2021 target) evaluation.</p>
<h2>What this does not rule out</h2><ul>
<li><b>Which second collimator we ran with.</b> The small (capture) one is n_TOF's capture standard and the one measured above; it is
assumed, not confirmed. The run's raw headers are no longer on EOS; the n_TOF logbook or coordinators can confirm.</li>
<li><b>The capsule's flight path</b> (19.5 vs 19.95 m changes the bore fraction by a third).</li>
<li><b>Hardware in the hall</b> not in the transport: the in-hall lead collimator (used with the small collimator), air scattering
(adds a faint halo), the station frame. The PPACMon agreement says none of it changes the core at 19.95 m.</li>
<li><b>The FLUKA source</b> is one simulation (FLUKA 4.5.1); the measured profile shows small asymmetries it does not reproduce exactly.</li>
<li>The orientation of n_TOF's (x, z) against our chamber frame is unknown, so the 1–3 mm offsets cannot be signed in our frame.</li>
</ul></body></html>'''
    (OUT / 'report.html').write_text(page)
    print('wrote', OUT / 'report.html')


if __name__ == '__main__':
    build()
