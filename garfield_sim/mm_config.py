"""
mm_config.py — Central configuration for Micromegas gain simulation
====================================================================
Edit this file to change gases, voltage ranges, pressures, or geometry.
All other scripts import from here.
"""

import os

# ── Output directories ────────────────────────────────────────────────────────

# Where gas tables (.gas files) are cached
GAS_DIR = os.path.join(os.path.dirname(__file__), "gas_tables")

# Where simulation results (JSON + numpy) are written
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

os.makedirs(GAS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Detector geometry ─────────────────────────────────────────────────────────

GAP_CM   = 0.0150   # Amplification gap: 150 µm mesh-to-resistive-layer
TEMP_K   = 293.15   # Room temperature

# ── Pressure conditions ───────────────────────────────────────────────────────
# Atmospheric pressure at altitude h (m) via barometric formula:
#   P(h) = 101325 * exp(-h / 8500)  [Pa]   (scale height ~8500 m)
# Convert Pa → Torr: 1 Pa = 0.00750062 Torr

def altitude_to_torr(h_m):
    """Convert altitude in metres to atmospheric pressure in Torr."""
    import math
    p_pa = 101325.0 * math.exp(-h_m / 8500.0)
    return p_pa * 0.00750062

PRESSURES = {
    "Saclay_160m": altitude_to_torr(160),   # CEA Saclay
    "CERN_450m":   altitude_to_torr(450),   # CERN Meyrin
}

# ── Gas mixtures ──────────────────────────────────────────────────────────────
# Each entry:
#   "label"      : human-readable name used in filenames and plots
#   "components" : list of (garfield_name, fraction_percent) — must sum to 100
#   "penning"    : dict with keys:
#       "mode"   : "auto"  → call EnablePenningTransfer() (built-in table)
#                  "manual" → call EnablePenningTransfer(rP, 0., noble_gas)
#       "rP"     : transfer probability (only used if mode="manual")
#       "gas"    : noble gas name for manual mode (e.g. "he")
#
# Notes on Penning:
#   Ar/iC4H10:  Built-in parameterisation exists (Sahin et al., JINST 5 2010).
#               Use mode="auto"; Garfield++ interpolates rP(c,p).
#   He/C2H6:    NOT in Garfield++ built-in table.
#               He 2³S metastable (19.8 eV) >> IP(C2H6) ≈ 11.5 eV, so
#               Penning transfer is energetically very favourable.
#               We use rP = 0.40 as a reasonable conservative estimate;
#               this should be validated against measured gain curves.
#               Literature on similar He+hydrocarbon mixtures (e.g. He/iC4H10)
#               suggests rP in the range 0.3–0.6.

GAS_MIXTURES = [
    {
        "label":      "He_C2H6_96p5_3p5",
        "components": [("he", 96.5), ("c2h6", 3.5)],
        "penning":    {"mode": "manual", "rP": 0.40, "gas": "he"},
    },
    {
        "label":      "Ar_iC4H10_95_5",
        "components": [("ar", 95.0), ("ic4h10", 5.0)],
        "penning":    {"mode": "auto"},
    },
]

# ── Voltage scan ──────────────────────────────────────────────────────────────
# Mesh voltage range (V). The E-field in the gap = V / GAP_CM.
# 400 V / 0.015 cm = 26667 V/cm
# 550 V / 0.015 cm = 36667 V/cm
V_MIN      =  400    # V
V_MAX      =  550    # V
V_STEP     =    5    # V  (use 10 for a quick first pass, 5 for full resolution)

import numpy as np
VOLTAGES = np.arange(V_MIN, V_MAX + V_STEP, V_STEP)   # inclusive of V_MAX

# ── Magboltz gas table settings ───────────────────────────────────────────────
# E-field grid for gas table. Must bracket the amplification fields we scan.
# Amplification field range: ~26 kV/cm – 37 kV/cm.
# We go a bit wider so Magboltz never needs to extrapolate.
E_GRID_MIN_VCM  =  5_000   # V/cm  (low end, not used in gain but needed for table)
E_GRID_MAX_VCM  = 60_000   # V/cm  (well above our max amp field)
E_GRID_NPTS     =     20   # number of log-spaced points

# Number of Magboltz collision sets (x10^7). Higher = more accurate but slower.
# 10 → ~1-2% precision, takes ~5-15 min per gas/pressure combo.
# Use 5 for quick checks, 10 for production.
MAGBOLTZ_NCOLL  =     10

# ── Avalanche simulation settings ────────────────────────────────────────────
N_EVENTS        =    200   # electrons to simulate per (gas, pressure, voltage) point

# ── Timing checkpoint ─────────────────────────────────────────────────────────
# After this many events in a voltage step, print an ETA.
TIMING_CHECKPOINT = 10
