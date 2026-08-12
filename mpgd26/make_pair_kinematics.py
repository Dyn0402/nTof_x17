#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pair_kinematics.py -- the "ideal case" e+e- kinematics figure for the
target slide of the MPGD2026 talk.

    ../.venv/bin/python make_pair_kinematics.py            # figure, from the reduction
    ../.venv/bin/python make_pair_kinematics.py --reduce    # re-derive the reduction

WHAT IT SHOWS.  Generator-level (truth) kinematics of the e+e- pair from
n + 3He -> 4He*(20.58 MeV), de-exciting either through the hypothesised X17 or
through internal pair conversion (IPC).  Left panel: the single-lepton kinetic
energy spectrum.  Right panel: the pair opening angle.  Nothing here has passed
through any material -- no capsule wall, no window, no multiple scattering, no
detector acceptance.  That is the point: this is the picture the real target
degrades, which is what the following slide is about.

PROVENANCE -- this is a Geant4 campaign product, not a hand-made curve.  The
numbers come from the production primary generator
`MX17_Full_Geant/src/X17PrimaryGenerator.cc`, read back out of the primary
vertices in `EventAction.cc` (branches `em_ke`, `ep_ke`, `openingAngle` of
`EventTree`) and written for EVERY generated event -- no trigger, no gate:

  campaign   /eos/experiment/ntof/data/x17/full_sim/pairs_thermal_trig_2cm
             (10^7 events, X17/IPC 50/50, vertices from the thermal
             self-shielding library; the "trig" in the name refers to the
             trigger STUDY the campaign fed, not to a cut on this tree)
  extract    MX17_Full_Geant/scripts/extract_signal_openingangle.py  (on lxplus)
             -> analysis/al_pair/signal_openingangle.npz  (64 MB, gitignored)
  reduce     this file, --reduce  ->  data/ideal_pair_kinematics.csv  (committed)

The reduction is a histogram, so the figure regenerates offline from the repo
with no lxplus and no 64 MB blob.  `--reduce` needs the npz; point at it with
`--npz` if it is somewhere else.

MODEL CAVEAT, to be said out loud if anyone asks about the IPC curve.  The
generator's IPC is a two-body decay of a virtual photon whose invariant mass is
drawn from dN/dMee ~ 1/Mee over [2me, 20.58 MeV], decaying isotropically in its
own rest frame.  That is the standard shape-level stand-in; it carries NO
E0/M1/E2 multipole angular correlation and no nuclear form factor, so the IPC
opening-angle and energy-sharing curves are a modelling baseline.  The X17 curve
has no such caveat -- a 16.8 MeV/c2 boson from a 20.58 MeV transition is pure
two-body kinematics, and the flat energy box below is exact.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import plotstyle as P  # noqa: E402

OUT = os.path.join(HERE, 'slides', 'assets', 'img')
REDUCTION = os.path.join(HERE, 'data', 'ideal_pair_kinematics.csv')
DEFAULT_NPZ = os.path.join(os.path.expanduser('~'), 'CLionProjects',
                           'MX17_Full_Geant', 'analysis', 'al_pair',
                           'signal_openingangle.npz')

# The generator's constants (SimConfig defaults, X17PrimaryGenerator.cc).
E_TRANSITION = 20.58        # MeV, 4He* excitation
M_X17 = 16.8                # MeV/c2, the generator's boson mass
ME = 0.51099895             # MeV

# Binning of the reduction.
KE_BINS = np.round(np.arange(0.0, 20.8 + 1e-9, 0.2), 3)      # 104 edges
TH_BINS = np.round(np.arange(0.0, 180.0 + 1e-9, 2.0), 3)     # 91 edges

CHANNELS = ('x17', 'ipc')


# --------------------------------------------------------------------------- #
# Closed-form checks -- the reduction is asserted against these, so a silently
# broken input cannot become a slide.
# --------------------------------------------------------------------------- #

def x17_ke_box() -> tuple[float, float]:
    """Exact lab kinetic-energy limits of one lepton from X17 -> e+e-.

    The boson is monoenergetic (E = E_transition, the 4He recoil is negligible),
    so its boost is fixed; the decay is isotropic in its rest frame, hence the
    lab energy is UNIFORM between these two edges.
    """
    gamma = E_TRANSITION / M_X17
    beta = np.sqrt(E_TRANSITION ** 2 - M_X17 ** 2) / E_TRANSITION
    e_star = M_X17 / 2.0
    p_star = np.sqrt(e_star ** 2 - ME ** 2)
    return (gamma * (e_star - beta * p_star) - ME,
            gamma * (e_star + beta * p_star) - ME)


def x17_theta_min() -> float:
    """Minimum opening angle -- the symmetric configuration, both legs at 90
    degrees to the boost axis in the boson rest frame.  Each leg then carries
    transverse momentum p* and longitudinal p_parent/2."""
    p_star = np.sqrt((M_X17 / 2.0) ** 2 - ME ** 2)
    p_long = np.sqrt(E_TRANSITION ** 2 - M_X17 ** 2) / 2.0
    return 2.0 * np.degrees(np.arctan2(p_star, p_long))


# --------------------------------------------------------------------------- #
# Reduction
# --------------------------------------------------------------------------- #

def reduce_npz(npz_path: str, out_path: str) -> None:
    if not os.path.exists(npz_path):
        sys.exit(f'--reduce needs the Geant4 npz; not found at {npz_path}\n'
                 '  regenerate it on lxplus with MX17_Full_Geant/scripts/'
                 'extract_signal_openingangle.py, or pass --npz')
    d = np.load(npz_path)

    rows = []
    stats = {}
    for ch in CHANNELS:
        ke = np.concatenate([d[f'{ch}_em'], d[f'{ch}_ep']])     # both legs
        th = d[f'{ch}_theta']
        soft = np.minimum(d[f'{ch}_em'], d[f'{ch}_ep'])
        tot = d[f'{ch}_em'] + d[f'{ch}_ep']

        hk, _ = np.histogram(ke, bins=KE_BINS, density=True)
        ht, _ = np.histogram(th, bins=TH_BINS, density=True)
        for lo, hi, v in zip(KE_BINS[:-1], KE_BINS[1:], hk):
            rows.append(('lepton_ke', ch, lo, hi, v))
        for lo, hi, v in zip(TH_BINS[:-1], TH_BINS[1:], ht):
            rows.append(('opening_angle', ch, lo, hi, v))

        stats[ch] = dict(
            n=th.size,
            ke_min=ke.min(), ke_max=ke.max(),
            th_min=th.min(), th_med=float(np.median(th)),
            th_q25=float(np.percentile(th, 25)),
            th_q75=float(np.percentile(th, 75)),
            soft_min=soft.min(), soft_med=float(np.median(soft)),
            tot_mean=float(tot.mean()), tot_std=float(tot.std()),
        )

    # How much IPC sits under the X17 angular peak -- the separation problem,
    # and the number the right-hand panel annotates.
    stats['ipc']['leak'] = float((d['ipc_theta'] > stats['x17']['th_min']).mean())
    stats['x17']['leak'] = 1.0

    # --- sanity gates, against closed form ------------------------------- #
    lo, hi = x17_ke_box()
    assert abs(stats['x17']['ke_min'] - lo) < 0.01, stats['x17']['ke_min']
    assert abs(stats['x17']['ke_max'] - hi) < 0.01, stats['x17']['ke_max']
    for ch in CHANNELS:
        # both channels de-excite the SAME 20.58 MeV state: the pair's total
        # kinetic energy is fixed at E_transition - 2 me for every event.
        assert abs(stats[ch]['tot_mean'] - (E_TRANSITION - 2 * ME)) < 1e-3
        assert stats[ch]['tot_std'] < 1e-6, stats[ch]['tot_std']
    # the X17 box must be FLAT: relative rms of the interior bins
    box = np.array([v for kind, ch, a, b, v in rows
                    if kind == 'lepton_ke' and ch == 'x17'
                    and a >= lo + 0.2 and b <= hi - 0.2])
    assert box.std() / box.mean() < 0.02, box.std() / box.mean()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as fh:
        # Header comments are written by hand, not through csv.writer: the stat
        # lines are one long field and the writer would quote them.
        fh.write('# ideal-case e+e- truth kinematics, '
                 'n+3He -> 4He*(20.58 MeV) -> e+e-\n')
        fh.write(f'# source: {os.path.basename(npz_path)} <- Geant4 '
                 'pairs_thermal_trig_2cm EventTree (generator truth, ungated)\n')
        fh.write('# reduce: mpgd26/make_pair_kinematics.py --reduce\n')
        for ch in CHANNELS:
            s = stats[ch]
            fh.write(f'# {ch}: n={s["n"]} ke_lo={s["ke_min"]:.3f} '
                     f'ke_hi={s["ke_max"]:.3f} theta_min={s["th_min"]:.2f} '
                     f'theta_med={s["th_med"]:.2f} theta_q25={s["th_q25"]:.1f} '
                     f'theta_q75={s["th_q75"]:.1f} soft_min={s["soft_min"]:.3f} '
                     f'soft_med={s["soft_med"]:.3f} sum_ke={s["tot_mean"]:.4f} '
                     f'above_x17_thr={s["leak"]:.4f}\n')
        w = csv.writer(fh)
        w.writerow(['kind', 'channel', 'lo', 'hi', 'density'])
        for r in rows:
            w.writerow([r[0], r[1], f'{r[2]:g}', f'{r[3]:g}', f'{r[4]:.6g}'])

    print(f'  -> {out_path}')
    for ch in CHANNELS:
        s = stats[ch]
        print(f'  {ch}: n={s["n"]}  KE [{s["ke_min"]:.2f},{s["ke_max"]:.2f}] MeV  '
              f'theta med {s["th_med"]:.1f} deg (min {s["th_min"]:.1f}, '
              f'IQR {s["th_q25"]:.0f}-{s["th_q75"]:.0f})  '
              f'softer leg >= {s["soft_min"]:.2f} MeV')
    ipc_over = float((d['ipc_theta'] > stats['x17']['th_min']).mean())
    print(f'  IPC leaking above the X17 angular threshold '
          f'({stats["x17"]["th_min"]:.1f} deg): {100 * ipc_over:.1f} %')
    print(f'  closed form: X17 box [{lo:.3f},{hi:.3f}] MeV, '
          f'theta_min {x17_theta_min():.2f} deg')


# --------------------------------------------------------------------------- #
# Loading the reduction
# --------------------------------------------------------------------------- #

def load_reduction(path: str) -> tuple[dict, dict]:
    """-> ({(kind, channel): (edges, density)}, {channel: {stat: value}})"""
    if not os.path.exists(path):
        sys.exit(f'reduction missing: {path}\n'
                 '  build it once with:  make_pair_kinematics.py --reduce')
    acc: dict[tuple[str, str], list] = {}
    meta: dict[str, dict] = {}
    with open(path) as fh:
        for raw in fh:
            if raw.startswith('#'):
                # "# x17: n=... theta_med=... " -> parsed for the annotations
                body = raw.lstrip('#').strip()
                if ':' not in body:
                    continue
                key, _, rest = body.partition(':')
                if key.strip() not in CHANNELS:
                    continue
                got = {}
                for tok in rest.split():
                    if '=' not in tok:
                        continue
                    k, _, v = tok.partition('=')
                    got[k] = v
                meta[key.strip()] = got
                continue
            parts = raw.strip().split(',')
            if len(parts) != 5 or parts[0] == 'kind':
                continue
            kind, ch, lo, hi, dens = parts
            acc.setdefault((kind, ch), []).append((float(lo), float(hi), float(dens)))
    out = {}
    for key, vals in acc.items():
        vals.sort()
        edges = np.array([v[0] for v in vals] + [vals[-1][1]])
        out[key] = (edges, np.array([v[2] for v in vals]))
    return out, meta




# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #

X17_C = P.ACCENT                 # the deck accent -- the signal
IPC_C = P.DET_COLOR['A']         # blue -- the irreducible physics background


def _step(ax, edges, dens, color, fill=False, lw=2.0, ls='-'):
    """Histogram outline drawn as a polyline, so the two channels overlay
    cleanly without matplotlib's bar edges fighting each other."""
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(dens, 2)
    if fill:
        ax.fill_between(x, 0, y, color=color, alpha=0.20, lw=0, zorder=2)
    ax.plot(x, y, color=color, lw=lw, ls=ls, zorder=3, solid_joinstyle='miter')


def fig_pair_kinematics(red: dict, meta: dict, out_path: str) -> None:
    ke_lo = float(meta['x17']['ke_lo'])
    ke_hi = float(meta['x17']['ke_hi'])
    th_min = float(meta['x17']['theta_min'])
    th_med_x = float(meta['x17']['theta_med'])
    th_med_i = float(meta['ipc']['theta_med'])
    soft_min = float(meta['x17']['soft_min'])
    sum_ke = float(meta['x17']['sum_ke'])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # ---------------- A: single-lepton kinetic energy ------------------- #
    ax = axes[0]
    e_i, d_i = red[('lepton_ke', 'ipc')]
    e_x, d_x = red[('lepton_ke', 'x17')]
    _step(ax, e_i, d_i, IPC_C, lw=2.2)
    _step(ax, e_x, d_x, X17_C, fill=True, lw=2.6)

    top = max(d_x.max(), d_i.max())
    ax.set_ylim(0, top * 1.42)
    ax.set_xlim(0, 20.6)

    for edge in (ke_lo, ke_hi):
        ax.plot([edge, edge], [0, top * 1.06], color=X17_C, lw=1.1, ls=':', zorder=1)
    ax.annotate(f'{ke_lo:.1f}', xy=(ke_lo, top * 1.08), color=X17_C, fontsize=10.5,
                fontweight='bold', ha='center', va='bottom')
    ax.annotate(f'{ke_hi:.1f} MeV', xy=(ke_hi, top * 1.08), color=X17_C,
                fontsize=10.5, fontweight='bold', ha='center', va='bottom')

    # Sits low inside the box: at x ~ 6.5 the IPC curve is well above this, so the
    # label clears both curves. Do not put it at mid-height -- it lands on IPC.
    P.end_label(ax, 6.5, top * 0.26,
                'X17 (16.8 MeV/c²)\nflat box, hard edges', X17_C, ha='center')
    P.end_label(ax, 18.6, d_i[np.searchsorted(e_i, 18.6) - 1] + top * 0.10,
                'IPC\n(1/M$_{ee}$ model)', IPC_C, ha='center')

    ax.set_xlabel('kinetic energy of one lepton  [MeV]')
    ax.set_ylabel('probability / MeV')
    P.strip(ax)
    # Subtitles must stay short: at this figure width a long left-hand subtitle
    # runs under the right panel's title. Full statement lives in the footer.
    P.title(ax, 'The X17 boxes the energy sharing in',
            f'both legs pooled · the pair always carries {sum_ke:.2f} MeV')

    # ---------------- B: opening angle --------------------------------- #
    ax = axes[1]
    a_i, b_i = red[('opening_angle', 'ipc')]
    a_x, b_x = red[('opening_angle', 'x17')]
    topb = max(b_x.max(), b_i.max())
    ax.axvspan(th_min, 180.0, color=X17_C, alpha=0.07, zorder=0)
    _step(ax, a_i, b_i, IPC_C, lw=2.2)
    _step(ax, a_x, b_x, X17_C, fill=True, lw=2.6)

    ax.set_xlim(0, 180)
    ax.set_ylim(0, topb * 1.30)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])

    ax.plot([th_min, th_min], [0, topb * 1.16], color=X17_C, lw=1.1, ls=':', zorder=1)
    ax.annotate(f'kinematic threshold {th_min:.0f}°',
                xy=(th_min, topb * 1.18), color=X17_C, fontsize=10.5,
                fontweight='bold', ha='right', va='bottom')
    P.end_label(ax, 143, topb * 0.42, f'X17\nmedian {th_med_x:.0f}°', X17_C,
                ha='center')
    P.end_label(ax, 46, b_i[np.searchsorted(a_i, 46) - 1] + topb * 0.10,
                f'IPC\nmedian {th_med_i:.0f}°', IPC_C, ha='center')

    ax.set_xlabel('e⁺e⁻ opening angle  [deg]')
    ax.set_ylabel('probability / deg')
    P.strip(ax)
    P.title(ax, 'and that is what puts the pair at large angle',
            'the observable the micro-TPC is built to measure')

    fig.subplots_adjust(wspace=0.24)
    P.note(fig,
           'IDEAL CASE — Geant4 generator truth, read off the primary vertices before any transport: no capsule '
           'wall, no window, no multiple scattering, no acceptance. Campaign pairs_thermal_trig_2cm, '
           f'{int(meta["x17"]["n"]) + int(meta["ipc"]["n"]):,} events, X17/IPC 50/50 by construction (NOT the '
           'physical branching ratio — the two curves are shapes, normalised separately). X17 at 16.8 MeV/c² is '
           'exact two-body kinematics; the IPC curve is the standard dN/dM$_{ee}$ ∝ 1/M$_{ee}$ virtual-photon '
           f'stand-in with isotropic decay, carrying no E0/M1/E2 multipole correlation. Softer leg ≥ {soft_min:.1f} MeV '
           'for X17 by construction. Regenerate: mpgd26/make_pair_kinematics.py',
           y=-0.03)
    P.save(fig, out_path)


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reduce', action='store_true',
                    help='re-derive data/ideal_pair_kinematics.csv from the Geant4 npz')
    ap.add_argument('--npz', default=DEFAULT_NPZ, help='source npz for --reduce')
    ap.add_argument('--out', default=os.path.join(OUT, 'ideal_pair_spectrum.png'))
    args = ap.parse_args()

    P.use()
    if args.reduce:
        reduce_npz(args.npz, REDUCTION)
    red, meta = load_reduction(REDUCTION)
    fig_pair_kinematics(red, meta, args.out)


if __name__ == '__main__':
    main()
