#!/usr/bin/env python3
"""make_gas_scatter.py -- Paper I v2: what the one-straight-track assumption misses for few-MeV electrons.

    .venv/bin/python papers/paper1_v2/figs/make_gas_scatter.py [--out papers/paper1_v2/figs/out/gas_scatter] [--n 20000]

Toy, no data. A particle enters the drift gas at normal incidence and crosses the measured 28.2 mm gap of
Ar/iC4H10 95/5 (1 atm, 20 C; X0 = 20.40 g/cm^2, rho = 1.703 mg/cm^3, X0 = 119.7 m, so the gap is 2.36e-4 X0;
mcs_common). Multiple Coulomb scattering is a Gaussian random walk of the projected angle (independent kicks on
0.1 mm steps) whose total width over the gap is the Highland theta0, so (y, theta) at the exit have the standard
covariance theta0^2 L^2/3, theta0^2 L/2, theta0^2. Energy loss in the gap (~70 keV) is ignored.

CAVEATS, stated in the paper text too: Highland is quoted by the PDG to 11 % only above 1e-3 X0; at 2.4e-4 X0 it is
at the edge of its validity. A particle makes only ~37 elastic scatters in the gap (Lynch-Dahl Omega), so the
real angle distribution is a Gaussian core plus single-scatter (Rutherford) tails, and the tails are NOT included
here: the rms values below describe the core. Cross-check: the Lynch & Dahl Moliere-based width (F = 0.98) is ~5 %
below Highland at every energy here (written to the json).

Quantities (one projection, rms over tracks):
  fit - entry : the straight line fitted to the trajectory with uniform weights along z (as a TPC samples it,
                ionisation uniform in depth, all charge weighted equally), against the direction at entry;
  exit - entry: theta0 of the gap (= Highland by construction);
  deviation   : rms over z and tracks of the trajectory's transverse distance from its fitted line.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', '..')]
import paper_style as ps  # noqa: E402
import mcs_common as mc  # noqa: E402

L = mc.G_DRIFT
DZ = 0.1
PITCH = 0.78
SIG_POS = 0.12
MEAS_ANG = 1.0              # deg, chamber resolution on muons (0.99 X / 0.90 Y)
T_MU = 4000.0               # MeV kinetic, a typical cosmic muon (mean ~4 GeV at sea level)
T_TAB = (3, 4, 5, 8, 12, 16)


def walk(theta0, n, rng, nz=None):
    """Trajectories y(z) [mm] on z = 0..L (entry direction along z, y(0) = 0). Kicks applied at step midpoints."""
    nz = nz or int(round(L / DZ))
    z = np.linspace(0, L, nz + 1)
    dz = L / nz
    kicks = rng.normal(0, theta0 * np.sqrt(dz / L), (n, nz))
    th = np.cumsum(kicks, axis=1)                         # angle after each kick
    # kick at the start of each step (step k: angle th[k]) -> y increments
    y = np.concatenate([np.zeros((n, 1)), np.cumsum(np.tan(th) * dz, axis=1)], axis=1)
    return z, y, th[:, -1]


def fit_lines(z, y):
    """Uniform-weight least-squares line per trajectory: slope, intercept, residuals."""
    zc = z - z.mean()
    slope = (y - y.mean(axis=1, keepdims=True)) @ zc / (zc @ zc)
    icpt = y.mean(axis=1) - slope * z.mean()
    res = y - (icpt[:, None] + slope[:, None] * z[None, :])
    return slope, icpt, res


def stats(T, n, rng, m=mc.ME):
    x = mc.x_over_X0('gas', L)
    th0 = float(mc.highland(x, T, m))
    z, y, th_exit = walk(th0, n, rng)
    slope, _, res = fit_lines(z, y)
    ld, om = mc.lynch_dahl('gas', L, T, m)
    return dict(T_MeV=T, p_MeV=round(float(mc.kin(T, m)[0]), 3), theta0_highland_deg=np.degrees(th0),
                exit_minus_entry_rms_deg=float(np.degrees(np.sqrt(np.mean(th_exit ** 2)))),
                fit_minus_entry_rms_deg=float(np.degrees(np.sqrt(np.mean(np.arctan(slope) ** 2)))),
                fit_over_theta0=float(np.sqrt(np.mean(np.arctan(slope) ** 2)) / th0),
                deviation_rms_mm=float(np.sqrt(np.mean(res ** 2))),
                deviation_max_rms_mm=float(np.sqrt(np.mean(np.max(np.abs(res), axis=1) ** 2))),
                theta0_lynch_dahl_deg=np.degrees(ld), lynch_dahl_over_highland=ld / th0, omega_scatters=om)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'gas_scatter'))
    ap.add_argument('--n', type=int, default=20000)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ps.use()
    rng = np.random.default_rng(20261110)

    Ts = np.arange(3.0, 16.01, 0.5)
    S = [stats(T, a.n, rng) for T in Ts]
    tab = {f'{T}MeV': {k: (round(v, 5) if isinstance(v, float) else v) for k, v in stats(float(T), 4 * a.n, rng).items()}
           for T in T_TAB}
    mu = stats(T_MU, a.n, rng, m=mc.MMU)

    fig = plt.figure(figsize=(ps.FULL[0], 2.75))
    axa = fig.add_axes([0.01, 0.17, 0.31, 0.76])
    axb = fig.add_axes([0.42, 0.17, 0.22, 0.76])
    axc = fig.add_axes([0.76, 0.17, 0.22, 0.76])

    # (a) trajectories, bundles offset in y
    cols = {'mu': '#555555', 16: '#56B4E9', 8: ps.DET_COLOR['det3'], 3: ps.ACCENT}
    rows = [('mu', mc.MMU, T_MU, r'$\mu$, 4 GeV'), (16, mc.ME, 16.0, '16 MeV'), (8, mc.ME, 8.0, '8 MeV'),
            (3, mc.ME, 3.0, '3 MeV')]
    rr = np.random.default_rng(5)
    off_step = 2.6
    for i, (key, m, T, lab) in enumerate(rows):
        off = -i * off_step
        th0 = float(mc.highland(mc.x_over_X0('gas', L), T, m))
        z, y, _ = walk(th0, 5, rr)
        for k in range(y.shape[0]):
            axa.plot(z, off + y[k], color=cols[key], lw=0.9, alpha=0.9, zorder=3)
        axa.plot([0, L], [off, off], color=ps.MUTED, lw=0.6, ls=':', zorder=2)
        axa.text(0.4, off + 0.75, lab, ha='left', va='bottom', fontsize=7, color=cols[key])
        if key == 3:
            # the straight line fitted to the most-deflected trajectory
            j = int(np.argmax(np.abs(y[:, -1])))
            s_, c_, _ = fit_lines(z, y[j:j + 1])
            axa.plot(z, off + c_[0] + s_[0] * z, color=ps.MODEL, lw=1.0, ls='--', zorder=4)
            axa.text(L - 0.3, off + c_[0] + s_[0] * L + 0.25, 'fitted line', fontsize=7, color=ps.MODEL, va='bottom',
                     ha='right')
    axa.set_xlim(0, L)
    axa.set_ylim(-3 * off_step - 1.2, 1.6)
    axa.set_yticks([])
    axa.set_xlabel('depth in the drift gap (mm)')
    axa.spines['left'].set_visible(False)
    axa.grid(False)
    # scale bar for the transverse axis: 1 mm
    xb = L - 1.0
    axa.plot([xb, xb], [0.4, 1.4], color=ps.INK, lw=1.2, clip_on=False, zorder=6)
    # stretch factor (data units per inch, x over y)
    fig.canvas.draw()
    bb = axa.get_window_extent()
    sx = (axa.get_xlim()[1] - axa.get_xlim()[0]) / bb.width
    sy = (axa.get_ylim()[1] - axa.get_ylim()[0]) / bb.height
    stretch = sx / sy
    axa.text(xb - 0.5, 0.9, f'1 mm\n(transverse ×{stretch:.1f})', fontsize=7, ha='right', va='center', color=ps.INK,
             linespacing=0.95)
    axa.set_xticks([0, 10, 20, 28.2])
    axa.set_xticklabels(['0', '10', '20', '28.2'])

    # (b) angles
    fe = [s['fit_minus_entry_rms_deg'] for s in S]
    ee = [s['exit_minus_entry_rms_deg'] for s in S]
    hl = [s['theta0_highland_deg'] for s in S]
    ldl = [s['theta0_lynch_dahl_deg'] for s in S]
    axb.plot(Ts, hl, color=ps.MODEL, lw=1.1, label='exit − entry ($\\theta_0$)')
    axb.plot(Ts, ldl, color=ps.MODEL, lw=0.8, ls=':', label='$\\theta_0$, Lynch–Dahl')
    axb.plot(Ts, fe, color=ps.ACCENT, lw=1.4, label='fit − entry')
    axb.axhline(MEAS_ANG, color=ps.DET_COLOR['det3'], lw=0.9, ls='--')
    axb.text(15.8, MEAS_ANG + 0.05, 'chamber resolution\n(muons)', fontsize=7, color=ps.DET_COLOR['det3'],
             ha='right', va='bottom', linespacing=0.95)
    axb.set_xlim(3, 16)
    axb.set_ylim(0, 2.6)
    axb.set_xlabel('kinetic energy (MeV)')
    axb.set_ylabel('rms angle, one projection (°)')
    axb.legend(loc='upper right', fontsize=7, handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)

    # (c) transverse deviation from the fitted line
    dv = [s['deviation_rms_mm'] for s in S]
    axc.plot(Ts, dv, color=ps.ACCENT, lw=1.4, label='rms deviation\nfrom fitted line')
    axc.axhline(PITCH, color=ps.MUTED, lw=0.9, ls='-.')
    axc.text(15.8, PITCH * 1.1, 'strip pitch 0.78 mm', fontsize=7, color=ps.MUTED, ha='right', va='bottom')
    axc.axhline(SIG_POS, color=ps.DET_COLOR['det3'], lw=0.9, ls='--')
    axc.text(15.8, SIG_POS * 1.1, 'position resolution', fontsize=7, color=ps.DET_COLOR['det3'], ha='right',
             va='bottom')
    axc.set_yscale('log')
    axc.set_xlim(3, 16)
    axc.set_ylim(0.005, 2.0)
    axc.set_xlabel('kinetic energy (MeV)')
    axc.set_ylabel('transverse deviation (mm)')
    axc.legend(loc='lower left', fontsize=7, handlelength=1.6, borderaxespad=0.2)

    for ax, lab, dx in ((axa, '(a)', 0.0), (axb, '(b)', -0.30), (axc, '(c)', -0.30)):
        ax.text(dx, 1.07, lab, transform=ax.transAxes, fontsize=ps.BASE_PT, fontweight='bold', va='top', ha='left')
    ps.save(fig, a.out)

    g = mc.MAT['gas']
    num = dict(
        gas=dict(mixture='Ar/iC4H10 95/5 by volume, 1 atm, 20 C', rho_g_cm3=round(g['rho'], 6),
                 mass_fraction_Ar=round(g['w_ar'], 4), X0_g_cm2=round(g['X0'], 3),
                 X0_m=round(mc.X0_cm('gas') / 100, 2),
                 inputs='PDG: Ar X0 19.55 g/cm2, rho 1.662 mg/cm3; isobutane X0 45.23 g/cm2, rho 2.489 mg/cm3'),
        gap_mm=L, gap_x_over_X0=round(mc.x_over_X0('gas', L), 7),
        model='Gaussian random walk of the projected angle, total width = Highland theta0 over the gap; '
              'normal incidence; no single-scatter tails; no energy loss',
        stretch_factor_panel_a=round(stretch, 2),
        energies=tab,
        muon_4GeV={k: (round(v, 6) if isinstance(v, float) else v) for k, v in mu.items()},
        fit_over_theta0_mean=round(float(np.mean([s['fit_over_theta0'] for s in S])), 4),
        analytic_fit_over_theta0=round(float(np.sqrt(13 / 35)), 4),
        notes='fit - entry = uniform-weight line fit vs entry direction; analytic ratio sqrt(13/35)=0.609 for a '
              'continuous random walk: var(slope) = (36/L^6) * int int u(L-u) v(L-v) min(u,v) du dv / L * theta0^2.',
    )
    with open(a.out + '.json', 'w') as f:
        json.dump(num, f, indent=1)
    print(json.dumps({k: num[k] for k in ('gas', 'gap_x_over_X0', 'fit_over_theta0_mean', 'stretch_factor_panel_a')},
                     indent=1))
    for k, v in tab.items():
        print(k, {kk: v[kk] for kk in ('theta0_highland_deg', 'fit_minus_entry_rms_deg', 'deviation_rms_mm',
                                       'lynch_dahl_over_highland', 'omega_scatters')})
    print('mu', mu['theta0_highland_deg'], mu['fit_minus_entry_rms_deg'], mu['deviation_rms_mm'])


if __name__ == '__main__':
    main()
