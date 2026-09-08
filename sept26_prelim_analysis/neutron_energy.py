#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neutron_energy.py -- time since the flash -> neutron kinetic energy, and what it costs.

Stage 3's ``e_neutron_keV`` is a declared null.  This is the conversion it
needs, plus the resolution that conversion actually delivers, so that when the
column is filled nobody has to guess how much of a spectrum is real.

THE CHAIN, AND WHERE IT IS BROKEN.  Three of the four pieces already exist:

  flight path      19.5 m, EAR2 (``MX17_Simulation/dead_time_sim.py``,
                   ``neutron_energy_vs_flight_time.py``)
  conversion       relativistic, below; cross-checked against
                   ``neutron_energy_vs_flight_time.time_s_to_energy_eV``
  the time base    ALREADY CALIBRATED. ``ntof_processing/flash_timing`` measures
                   t_flash(bunch) = tof_PKUP(bunch) + C, C ~ -1708 ns per
                   channel, good to 0.5 ns run-to-run within an epoch and
                   3.2 ns per bunch. And ``ntof_dream_merge.ntof_io.read_bunches``
                   already returns ``t_since_flash_ns = tof - tflash`` against a
                   REPAIRED per-bunch tflash.
  the slim         **THE GAP.** ``slim.py`` computes ``t_since_flash_ns`` --
                   it matches DREAM to n_TOF on exactly that quantity -- and
                   then does not write it. The output carries ``tof`` (raw,
                   1.004e6 to 7.505e7 ns, an acquisition window identical
                   across all three detector families) and ``dt_ns`` (the
                   match residual), but neither ``tflash`` nor ``BunchNumber``,
                   so t_since_flash cannot be reconstructed from a slim file.

So this is not a measurement problem.  It is one branch, and the patch is three
lines in ``pass2_hits`` (see :func:`slim_patch`).  Applying it needs the slims
regenerated, which needs EOS.

WHAT THE WINDOW COVERS -- inferred, with corroboration, not read.  ``tflash``
is not in the slim, but all three detector families share the same tof floor at
1.00408e6 ns and the same ceiling at 7.505e7 ns, which is an acquisition window
rather than physics.  Taking the floor AS the flash (the window opens on it)
gives t_since_flash = tof - 1.00408e6 and an energy range of

    tof 1.005e6 ns  ->     2.4 MeV
    tof 1.01e6      ->    57   keV
    tof 2e6         ->     2.0 eV
    tof 5e6-8e6     ->   125-41 meV     <- the busiest bins in the file
    tof 7.5e7       ->     0.36 meV

and the corroboration is that the busy region lands on the thermal peak, which
is where the run_55 analysis independently reported the He-3(n,p) capture
flood.  So the window is not the limitation -- it spans MeV to sub-meV.  This
stays an inference until one bunch's tflash is read from the raw n_TOF files,
which needs EOS.

WHAT THE RESOLUTION WILL BE.  E = (gamma-1) m c^2 with beta = L/(c t), so
non-relativistically E ~ 1/t^2 and dE/E = 2 dt/t: the fractional energy error is
twice the fractional timing error, and it gets *worse* at high energy because t
gets short.  :func:`resolution_table` evaluates it for the timing budget the
flash calibration quotes, so the honest statement of what a spectrum can
resolve is available before the spectrum is.

    python -m sept26_prelim_analysis.neutron_energy
"""
from __future__ import annotations

import numpy as np

#: EAR2 flight path.  Not a new number: ``MX17_Simulation/dead_time_sim.py``
#: carries it as "m, EAR2 flight path" and ``neutron_energy_vs_flight_time.py``
#: uses the same value.
FLIGHT_PATH_M = 19.5

C_M_S = 299792458.0
EV_TO_J = 1.602176634e-19
M_N_KG = 1.67492749804e-27
REST_ENERGY_EV = M_N_KG * C_M_S ** 2 / EV_TO_J      # 939.565e6 eV

#: The timing budget, from ntof_processing/flash_timing/data/
#: flash_timing_calibration.json -- these are the numbers that set the energy
#: resolution, so they are quoted rather than assumed.
TIMING_NS = {
    'per_bunch_sigma': 3.2,
    'intensity_walk': 5.0,
    'epoch_07_11_vs_07_16': 3.8,
    'within_epoch_run_to_run': 0.5,
    'transport_to_end_of_campaign': 1.2,
}


def energy_eV(t_since_flash_ns, distance_m: float = FLIGHT_PATH_M):
    """Neutron kinetic energy [eV] from time since the gamma flash [ns].

    Relativistic throughout: beta = L/(c t), E = (gamma - 1) m c^2.  Times that
    imply beta >= 1 (a hit at or before the flash) return NaN rather than a
    clipped value -- such a hit is not a neutron and should not be given an
    energy.
    """
    t = np.asarray(t_since_flash_ns, float) * 1e-9
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = distance_m / (C_M_S * t)
    beta = np.where((beta > 0) & (beta < 1.0), beta, np.nan)
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    return (gamma - 1.0) * REST_ENERGY_EV


def energy_keV(t_since_flash_ns, distance_m: float = FLIGHT_PATH_M):
    return energy_eV(t_since_flash_ns, distance_m) / 1e3


def time_ns(energy_eV_, distance_m: float = FLIGHT_PATH_M):
    """Inverse: time since the flash [ns] for a given kinetic energy [eV]."""
    e = np.asarray(energy_eV_, float)
    gamma = 1.0 + e / REST_ENERGY_EV
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    return distance_m / (beta * C_M_S) * 1e9


def resolution_table(energies_eV=(1e-2, 1e0, 1e2, 1e4, 1e6, 1e7),
                     dt_ns: float | None = None,
                     distance_m: float = FLIGHT_PATH_M):
    """dE/E from the timing budget, per energy.

    dE/E = 2 dt/t non-relativistically, and the exact form is used here.  The
    point of the table is that the fractional error grows with energy: at
    10 MeV the flight time is ~450 ns and a few ns of timing is already a
    percent-level effect, while in the eV region it is negligible.
    """
    dt = float(dt_ns if dt_ns is not None else
               np.hypot(TIMING_NS['per_bunch_sigma'], TIMING_NS['intensity_walk']))
    rows = []
    for e in np.atleast_1d(np.asarray(energies_eV, float)):
        t = float(time_ns(e, distance_m))
        e_lo = float(energy_eV(t + dt, distance_m))
        e_hi = float(energy_eV(t - dt, distance_m)) if t > dt else np.nan
        rows.append(dict(energy_eV=e, t_ns=t, dt_ns=dt,
                         dE_over_E=abs(e_hi - e_lo) / (2 * e)
                         if np.isfinite(e_hi) else np.nan))
    return rows


def slim_patch() -> str:
    """The three-line change that unblocks this, as a statement of intent."""
    return """\
ntof_processing/slim_pipeline/slim.py, pass2_hits():

  1. add 't_since_flash_ns' to the `cols` dict initialiser (~line 424)
  2. beside the other per-hit appends (~line 462), carry it through:
         cols['t_since_flash_ns'].append(a['t_since_flash_ns'][pick[k]])
     -- note it indexes the SOURCE array with pick[k], like 'tof' does, not
     the prediction side
  3. add it to the uproot output dict (~line 483):
         't_since_flash_ns': hits['t_since_flash_ns'].astype(np.float64),

`a['t_since_flash_ns']` is already in scope -- pass2_hits matches on it. The
value is float64 like `tof`, and against the REPAIRED tflash, which is the
whole point of computing it upstream rather than here.

Then the slims must be regenerated (EOS), and build_tracks can join
t_since_flash_ns per (subrun, event_id) and fill e_neutron_keV with
neutron_energy.energy_keV.
"""


def main() -> int:
    print(f'EAR2 flight path {FLIGHT_PATH_M} m, neutron rest energy '
          f'{REST_ENERGY_EV / 1e6:.3f} MeV\n')
    print('flight time vs energy:')
    print(f'  {"energy":>12} {"time since flash":>18}')
    for e in (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e4, 1e6, 1e7):
        t = float(time_ns(e))
        u = f'{t / 1e6:.3f} ms' if t > 1e6 else (
            f'{t / 1e3:.2f} us' if t > 1e3 else f'{t:.1f} ns')
        lab = (f'{e:.0e} eV' if e < 1e3 else
               f'{e / 1e3:.0f} keV' if e < 1e6 else f'{e / 1e6:.0f} MeV')
        print(f'  {lab:>12} {u:>18}')
    dt = np.hypot(TIMING_NS['per_bunch_sigma'], TIMING_NS['intensity_walk'])
    print(f'\nenergy resolution for dt = {dt:.1f} ns '
          f'(per-bunch {TIMING_NS["per_bunch_sigma"]} ns and intensity walk '
          f'{TIMING_NS["intensity_walk"]} ns, added in quadrature):')
    print(f'  {"energy":>12} {"t (ns)":>12} {"dE/E":>10}')
    for r in resolution_table():
        lab = (f'{r["energy_eV"]:.0e} eV' if r['energy_eV'] < 1e3 else
               f'{r["energy_eV"] / 1e3:.0f} keV' if r['energy_eV'] < 1e6 else
               f'{r["energy_eV"] / 1e6:.0f} MeV')
        print(f'  {lab:>12} {r["t_ns"]:>12.1f} {100 * r["dE_over_E"]:>9.3f} %')
    print('\n--- what is missing ---')
    print(slim_patch())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
