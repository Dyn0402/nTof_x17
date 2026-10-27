#!/usr/bin/env python3
"""stackup_induction.py -- item 5 (ii): the sizes of the candidate X/Y footprint mechanisms from the board stack-up.

Stack (docs/detector_spec/spec.py, from MX17_Geant AsBuiltSpec + gerbers): mesh | 150 um gas | resistive strips (ESL,
550 um on 800 um, along the coordinate the Y view measures, 10 um) | 70.5 um kapton + glue (eps ~3.4) | L4 pads
0.68 mm on 0.78 mm, EACH pad routed by a via to an X strip (L6) or a Y strip (L5).  Both views therefore read the
SAME pad plane at the SAME depth: the 'deeper layer sees a wider footprint' mechanism has no lever (the buried L5/L6
lines sit under the pads and see the charge only through the 0.1 mm pad gaps).

1. Static induction footprint of a charge on the resistive surface (charge between two grounded planes, layered
   dielectric), 1-D projection, Fourier form  g(k) = e1 / [sinh(k d1) (e1 coth(k d1) + e2 coth(k d2))];  integrated
   over 0.78 mm pads for landing positions across a pad, and matched to the model's Gaussian-in-a-box -> the
   Gaussian-equivalent sigma (same for both views: the pad plane is shared).
2. RC spreading in the resistive sheet, D measured per chamber (bundle rc_D_y; SPS det4 2.2-3.0e-4 mm^2/ns): variance
   2 D s in both directions until the charge meets the strip edges (X only).  Box limit w^2/12, fill time
   tau = w^2 / (pi^2 D) (lowest mode of diffusion on a segment with reflecting ends), and 2 D s at one 60 ns sample.
3. Resistive-strip quantisation in X: the landed charge is confined to its 550 um strip; asymptotic extra variance
   w^2/12 (+ the gap funnelling, <= (250 um)^2/12 on the 31 % of landings that fall in a gap).
Writes results/stackup_induction.json and figures/stackup_mechanisms.png.
"""
import json, os
import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
PITCH = 0.78
W_ESL, P_ESL = 0.55, 0.80
D_RC = {'SPS det4 (run_71)': 2.75e-4, 'det2': 4.67e-4, 'det3': 3.44e-4, 'det4': 2.2e-4, 'det6': 5.90e-4, 'det7': 6.26e-4}


def profile_1d(d1, d2, e1=3.4, e2=1.0, L=20.0, n=2 ** 15):
    """1-D (projected) induced charge density on the pad plane for a unit charge on the resistive surface [1/mm]."""
    x = (np.arange(n) - n // 2) * (L / n)
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    ka = np.abs(k); ka[0] = 1e-9
    g = e1 / (np.sinh(ka * d1) * (e1 / np.tanh(ka * d1) + e2 / np.tanh(ka * d2)))
    f = np.real(np.fft.fftshift(np.fft.ifft(g))) / (L / n)
    return x, f, float(g[0] if np.isfinite(g[0]) else e1 / d1 / (e1 / d1 + e2 / d2))


def pad_fractions(x, f, x0s, npad=2):
    dx = x[1] - x[0]; c = np.cumsum(f) * dx
    out = []
    for x0 in x0s:
        row = []
        for i in range(-npad, npad + 1):
            lo, hi = i * PITCH - PITCH / 2 - x0, i * PITCH + PITCH / 2 - x0
            row.append(np.interp(hi, x, c) - np.interp(lo, x, c))
        out.append(row)
    F = np.array(out)
    return F / F.sum(1, keepdims=True)


def gauss_fractions(s, x0s, npad=2):
    out = []
    for x0 in x0s:
        out.append([0.5 * (erf((i * PITCH + PITCH / 2 - x0) / (np.sqrt(2) * s)) - erf((i * PITCH - PITCH / 2 - x0) / (np.sqrt(2) * s)))
                    for i in range(-npad, npad + 1)])
    F = np.array(out)
    return F / F.sum(1, keepdims=True)


def sigma_equiv(Fref, x0s):
    r = minimize_scalar(lambda s: ((gauss_fractions(s, x0s) - Fref) ** 2).sum(), bounds=(0.01, 0.8), method='bounded')
    return float(r.x)


def main():
    x0s = np.linspace(-PITCH / 2, PITCH / 2, 41)
    res = {'stack': dict(insulator_um=70.5, eps_insulator=3.4, gas_gap_um=150, pad_mm=0.68, pitch_mm=PITCH,
                         esl_width_mm=W_ESL, esl_pitch_mm=P_ESL, esl_width_status='open (from memory)'),
           'induction': {}, 'rc': {}}
    for d1 in (0.050, 0.0705, 0.090, 0.120):
        x, f, q = profile_1d(d1, 0.150)
        F = pad_fractions(x, f, x0s)
        s = sigma_equiv(F, x0s)
        nb = float(F[len(x0s) // 2, [1, 3]].sum())
        res['induction'][f'{d1*1e3:.1f}um'] = dict(sigma_equiv_mm=s, neighbour_share_centred=nb, frac_on_pads=q)
        print(f'insulator {d1*1e3:5.1f} um: Gaussian-equivalent induction footprint {s:.3f} mm, centred landing ±1 share {nb:.3f}, '
              f'{100*q:.0f} % of the charge induced on the pads')
    # single-plane (no mesh, no dielectric contrast) Cauchy for reference
    xc = np.linspace(-10, 10, 2 ** 15); fc = 0.0705 / np.pi / (xc ** 2 + 0.0705 ** 2)
    res['induction']['cauchy_single_plane_70.5um'] = dict(sigma_equiv_mm=sigma_equiv(pad_fractions(xc, fc, x0s), x0s))
    print(f'  (single grounded plane, Cauchy gamma = 70.5 um: {res["induction"]["cauchy_single_plane_70.5um"]["sigma_equiv_mm"]:.3f} mm)')
    b2 = W_ESL ** 2 / 12
    for k, D in D_RC.items():
        tau = W_ESL ** 2 / (np.pi ** 2 * D)
        res['rc'][k] = dict(D=D, var_60ns=2 * D * 60, sigma_60ns=float(np.sqrt(2 * D * 60)), box_var=b2, box_sigma=float(np.sqrt(b2)),
                            fill_tau_ns=tau, t_reach_box_ns=b2 / (2 * D))
        print(f'{k:18s} D {D:.2e}: 2Ds at 60 ns {2*D*60:.4f} mm² (σ {np.sqrt(2*D*60):.3f}), X box w²/12 {b2:.4f} (σ {np.sqrt(b2):.3f}), '
              f'fill τ {tau:.0f} ns, 2Ds reaches the box at {b2/(2*D):.0f} ns')
    gapfrac = (P_ESL - W_ESL) / P_ESL
    res['quantisation'] = dict(gap_fraction=gapfrac, gap_funnel_var_upper=gapfrac * (P_ESL - W_ESL) ** 2 / 12, box_var=b2)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, 'results', 'stackup_induction.json'), 'w'), indent=1)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x, f, _ = profile_1d(0.0705, 0.150)
    ax[0].plot(x, f / f.max(), label='induced on pads (70.5 µm kapton, 150 µm gas to mesh)')
    s = res['induction']['70.5um']['sigma_equiv_mm']
    ax[0].plot(x, np.exp(-0.5 * (x / s) ** 2), '--', label=f'Gaussian-equivalent σ = {s:.2f} mm')
    for i in range(-2, 3):
        ax[0].axvspan(i * PITCH - 0.34, i * PITCH + 0.34, color='0.9', zorder=0)
    ax[0].set_xlim(-1.6, 1.6); ax[0].set_xlabel('x − landing [mm]'); ax[0].set_ylabel('relative density')
    ax[0].set_title('Static induction: identical for X and Y (same pad plane)'); ax[0].legend(fontsize=8)
    s_ = np.linspace(0, 400, 401)
    for k, D in D_RC.items():
        if k in ('SPS det4 (run_71)', 'det3', 'det7'):
            l, = ax[1].plot(s_, 2 * D * s_, label=f'Y (along): 2Ds, {k}')
            ax[1].plot(s_, b2 * (1 - np.exp(-2 * D * s_ / b2)), '--', color=l.get_color(), label=f'X (across, bounded), {k}')
    ax[1].axhline(b2, color='k', lw=0.6); ax[1].axvline(60, color='k', lw=0.6, ls=':')
    ax[1].text(62, 0.002, 'one sample', fontsize=8); ax[1].text(250, b2 * 1.05, 'w²/12 (550 µm strip)', fontsize=8)
    ax[1].set_ylim(0, 0.15); ax[1].set_xlabel('time since landing s [ns]'); ax[1].set_ylabel('RC variance added [mm²]')
    ax[1].set_title('Resistive-sheet spreading: same D, X stops at the strip edge'); ax[1].legend(fontsize=7)
    fig.tight_layout(); os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    fig.savefig(os.path.join(HERE, 'figures', 'stackup_mechanisms.png'), dpi=130)


if __name__ == '__main__':
    main()
