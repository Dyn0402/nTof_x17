"""
Canonical detector geometry and run configuration for MX17 simulation scripts.

Edit this file to update detector dimensions or run parameters.
All simulation scripts import geometry from here so changes propagate everywhere.

Geometry source: Geant4 MX17_Full_Geant/SimConfig.hh
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MX17_Simulator import SimConfig, GaussianSpectrum, X17PhysicsSpectrum, IPCPhysicsSpectrum

# ── Run parameters ────────────────────────────────────────────────────────────
N_EVENTS  = 200_000   # readout windows for acceptance study
N_WORKERS = 16         # parallel workers (set to 1 to disable multiprocessing)
SEED      = 42

# ── Signal angular spectrum (full relativistic X17 → e+e- kinematics) ────────
SIGNAL_SPECTRUM = X17PhysicsSpectrum(m_x17_mev=16.8, E_transition_mev=20.58)

# ── IPC background spectrum (full relativistic IPC kinematics) ────────────────
IPC_SPECTRUM = IPCPhysicsSpectrum(E_transition_mev=20.58)

# ── Active-volume geometry (from Geant4 SimConfig.hh) ────────────────────────
#
#  Detector stack per arm (origin → outward):
#    MM:         25.0 cm,  38×34 cm  (u×v)
#    Scint wall: 24.92 cm, 48×48 cm,  3 mm PVT  ("SiPM wall")
#
#  Standard (Config A): SW → LS-1 → LS-2 → back_scint
#  Back-first (Config B): SW → back_scint → LS-1 → LS-2
#
GEO = dict(
    # Micromegas
    detector_distance      = 25.0,
    detector_size          = (38.0, 34.0),    # active area: u × v [cm]
    mm_drift_gap           = 3.0,
    gap_mm_to_scint        = 2.77,            # MM back-face → SW centre [cm]

    # Trigger scintillator wall ("SiPM wall")
    scint_wall_size        = (48.0, 48.0),    # u × v [cm]
    scint_wall_thickness   = 0.3,

    # Liquid scintillator layers (both configs)
    gap_scint_to_liq       = 3.44,            # SW centre → LS-1 centre [cm] (standard only)
    liq_scint_size         = (45.0, 45.0),
    liq_scint_thickness    = 2.0,
    liq_scint_2_size       = (45.0, 45.0),
    liq_scint_2_thickness  = 2.0,
    gap_ls1_to_ls2         = 2.26,

    # Back plastic scintillator (2 bars per arm)
    back_scint_size_u      = 20.0,            # each bar: u [cm]
    back_scint_size_v      = 30.0,            # each bar: v [cm]
    back_scint_thickness   = 2.0,
    back_scint_gap         = 0.3,             # inter-bar gap [cm]
    gap_ls2_to_backscint   = 2.30,            # LS-2 centre → BS centre (standard) [cm]
    gap_scint_to_backscint = 3.15,            # SW centre → BS centre (back_first) [cm]
    gap_backscint_to_ls1   = 2.36,            # BS centre → LS-1 centre (back_first) [cm]

    # He-3 pressurised capsule (cylinder axis = Y = beam)
    he_capsule_source      = True,
    he_radius_cm           = 1.5,
    he_half_length_cm      = 2.5,
)

# ── Geant4-derived detector response ──────────────────────────────────────────
# Produced by: MX17_Full_Geant/scripts/analyze_pairs.py --export-response <json>
# Provides energy-dependent multiple-scattering direction smearing (dominated
# by the He-3 target walls) and calorimetric invariant-mass response, replacing
# the perfect directions / flat-Gaussian inv_mass_sigma used previously.
#
# ⚠ The current file was derived from the OLD-geometry Geant4 dataset
#   (22 cm arms, 300 bar He-3, 0.9 mm CFRP wall, ArIso).  Regenerate it after
#   the new-geometry production and update this path.
# Set G4_RESPONSE = None to recover the legacy (unsmeared) behaviour.
G4_RESPONSE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "geant4_response_old_geom.json")

# ── Shared base for the acceptance study (signal pairs, He capsule) ────────────
_ACCEPTANCE_BASE = dict(
    **GEO,
    signal_spectrum      = SIGNAL_SPECTRUM,
    background_spectrum  = IPCPhysicsSpectrum(E_transition_mev=20.58),
    n_signal             = 1.0,
    n_random             = 0.0,
    n_background_pairs   = 0.0,
    n_events             = N_EVENTS,
    # Realistic MM time resolution (30 ns) with the coincidence window opened
    # to 200 ns ≈ 4.7σ of the pair Δt (σ_Δt = √2·30 ns), so essentially no
    # genuine X17 pair is lost to timing.  (A 60 ns window costs ~18%.)
    coincidence_window   = 200.0,
    time_spread          = 2000.0,
    spatial_resolution   = 0.5,
    time_resolution      = 30.0,
    seed                 = SEED,
    # Geant4 response: MS direction smearing + calorimetric mass
    g4_response_path       = G4_RESPONSE,
    g4_direction_estimator = 'first',   # dir @ 1st MM hit (realistic MM measurement)
    g4_energy_method       = 'ls',      # LS-only calorimetry
    g4_energy_corrected    = True,      # apply upstream-loss correction
)

# ── Config A: standard stack order (Geant4 default) ──────────────────────────
cfg_A = SimConfig(
    **_ACCEPTANCE_BASE,
    stack_order          = 'standard',
    trigger_second_layer = 'liq_scint_1',
)

# ── Config B: back_scint placed before liquid scintillators ──────────────────
cfg_B = SimConfig(
    **_ACCEPTANCE_BASE,
    stack_order          = 'back_scint_first',
    trigger_second_layer = 'back_scint',
)
