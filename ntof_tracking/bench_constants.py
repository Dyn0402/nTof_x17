#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_constants.py — everything the June 2026 cosmic-bench characterization
established about the MX17 chambers, packaged for beam-side reconstruction.

Sources of record (in mx_june_cosmic_qa/): DET3_WEEKEND_ANALYSIS.md,
MICROTPC_RUNBOOK.md §0/§0b, PAPER_STATUS.md, PLAN_38/39/42 results.
Bench conditions: Ar/iso 95/5 (+~1% H2O, ~1% air), drift 1000 V over 30 mm,
DREAM 32 samples x 60 ns. Beam conditions DIFFER (gas, sampling 20 ns,
300-400 sample windows) — see TRANSFERABILITY notes on each block.
"""
import numpy as np

# ---------------------------------------------------------------------------
# geometry / DAQ — design properties, transfer as-is
# ---------------------------------------------------------------------------
PITCH_MM = 0.78          # strip pitch, both planes, all chambers
N_STRIPS_PLANE = 512     # channels per plane (one FEU = one plane on bench)
DRIFT_GAP_MM = 30.0      # mechanical drift gap (det A/B/C/D July fleet)
                         # NB May-test mx17_3 had a 16 mm gap — always check
                         # the chamber build sheet / run_config.

# bench DREAM sampling. BEAM RUNS DIFFER: May/July nTOF runs use
# sample_period = 20 ns and 300-400 samples (6-8 us windows). Read
# n_samples_per_waveform & sample_period from each run's run_config.json.
BENCH_SAMPLE_NS = 60.0
FTST_NS = 10.0           # DREAM fine-timestamp step (3-bit, 100 MHz clock);
                         # applied upstream: hit `time` is already
                         # trigger-phase corrected (PLAN_42).

# ---------------------------------------------------------------------------
# thresholds & selection (bench-tuned; revisit amplitude thresholds at beam —
# July quick-look QA used 400 ADC against beam noise)
# ---------------------------------------------------------------------------
THR_HIT = 100.0          # strip counts toward a cluster above this (ADC)
A_LEAD_MIN = 300.0       # signature features need a lead strip above this
SPARK_VETO_STRIPS = 50   # >50 strips firing = discharge; veto the event/window
GAP_THRESHOLD_MM = 12.0  # 1-D position-gap clustering split
MIN_STRIPS = 3           # min strips per reconstructable cluster (4 pre-unshare
                         # convention in some scripts; 3 is the floor)

# angle-analysis windows (all in tan space)
TAN_LO, TAN_HI, TAN_STEP = 0.06, 0.44, 0.04   # extent-slope fit band
TAN_HI_SIG = 0.27        # cap for regressed-angle extent fits (feature
                         # saturation above ~15 deg)
TAN_SWITCH = 0.09        # hybrid rule: regression below ~5 deg, time fit above
SAT_DEG = 10.0           # T_sat = median cluster duration for |theta|>10 deg
MAX_TAN = 0.7
PLATEAU_TAN = (0.12, 0.55)

FEATS_HITS6 = ('tot_lead', 'q_frac', 'n_raw', 'a_asym', 'a_lead', 't_delay')

# ---------------------------------------------------------------------------
# resistive charge sharing — DESIGN PROPERTY, transfers across chambers
# (det3 vs det2 agree; gas-independent to first order: it is the strip RC
# network). Keyed by mx17 detector number. (c1, c2) = +-1 / +-2 neighbour
# fraction; y-plane has the larger +-2 reach (strips run along y).
# Neighbour delay ~ +69 ns (about +1 bench sample; in ns it should persist
# at 20 ns beam sampling — verify once with beam waveforms).
# ---------------------------------------------------------------------------
CSHARE = {
    3: {'x': (0.449, 0.052), 'y': (0.516, 0.151)},   # measured (det3, 26)
    2: {'x': (0.430, 0.060), 'y': (0.520, 0.200)},   # measured (det2, 29)
    6: {'x': (0.449, 0.052), 'y': (0.516, 0.151)},   # NOT measured: det3 copy
    7: {'x': (0.449, 0.052), 'y': (0.516, 0.151)},   # NOT measured: det3 copy
    4: {'x': (0.449, 0.052), 'y': (0.516, 0.151)},   # NOT measured: det3 copy
}
UNSHARE_ALPHA = 0.5      # prompt fraction of the mixed unsharing kernel
NEIGHBOUR_DELAY_NS = 69.0

# ---------------------------------------------------------------------------
# X/Y charge balance (PLAN_38): f = qX/(qX+qY) is NARROW and flat in
# position & angle -> use as an X<->Y pairing discriminant in busy windows.
# Central value is per-chamber (assembly), width ~0.07 for all.
# ---------------------------------------------------------------------------
F_BALANCE = {
    3: dict(med=0.487, s68=0.07),    # measured (490 V resist)
    2: dict(med=0.531, s68=0.07),    # measured (525 V resist)
    6: dict(med=0.50, s68=0.09),     # NOT measured — placeholder, remeasure
    7: dict(med=0.50, s68=0.09),     # in situ (one histogram per run)
    4: dict(med=0.50, s68=0.09),
}

# ---------------------------------------------------------------------------
# drift velocity — DOES NOT TRANSFER (bench gas Ar/iso 95/5 + H2O; beam gas
# differs and varies). What transfers is the METHOD:
#   v = extent_slope / T_sat  with a calibrated angle abscissa (script 21/35)
# The bench Ar/iso+1%H2O curve is kept for reference/sanity only.
# Beam gas candidates have Garfield tables: garfield_sim/results/
#   {Ar_iC4H10_95_5, He_C2H6_96p5_3p5, Ne_iC4H10_95_5_rP0xx}_CERN_450m.json
# ---------------------------------------------------------------------------
BENCH_V_GEOM_UM_NS = {   # det3, E = HV/3.0 cm, M3 v2 (700 V pt from scan)
    233.3: 23.31, 300.0: 29.67, 333.3: 34.30, 366.7: 35.53,
}
BENCH_T_SAT_NS = 691.0           # det3 @1000 V (recorded column ~23.4 mm)
BENCH_RECORDED_COLUMN_MM = 23.4  # < 30 mm gap: attachment-truncated (O2);
                                 # beam gas => remeasure via extent slope.

# ---------------------------------------------------------------------------
# angular performance achieved on the bench (targets / sanity anchors)
# ---------------------------------------------------------------------------
BENCH_ANGLES = dict(
    hybrid_s68_deg=1.8,          # uniform, all angles, 97-99 % coverage
    regression_only_plateau=2.1,  # hit-level regression alone
    production_plateau=6.9,      # raw anchored time fit (sharing-biased)
    det2_frozen_transfer=2.85,   # det3 model frozen onto det2 (-15 %)
)

# position estimators (needs waveforms for the sub-pitch ones)
POSITION = dict(
    prod_s68_mm=(0.85, 0.91),        # earliest-strip anchor, all angles (x,y)
    early_charge_low_angle=(0.61, 0.72),  # theta<5 deg, EARLY_K=2 raw samples
    combo_n_switch=9,                # early-charge if n_strips<=9 else prod
    early_k_samples_bench=2,         # = 120 ns of charge at 60 ns sampling;
                                     # at 20 ns beam sampling use ~6 samples
                                     # to integrate the same 120 ns
)

# timing (PLAN_42) — hit `time` is ABSOLUTE vs the trigger (ftst-corrected,
# 30 % constant-fraction, sub-sample interpolated)
TIMING = dict(
    sigma_single_strip_ns=38.9,
    sigma_detector_ns=33.1,          # walk-corrected floor 29.0
    sigma_absolute_ns=37.7,          # detector-dominated budget
    interplane_walk_ns_per_asym=-100.0,  # small S/N correction (33->29 ns)
)

# edge / fringe field (script 32): fiducialize ANGLE analyses away from the
# edge; hit-mode POSITION is robust to the edge.
EDGE_ANGLE_FIDUCIAL_MM = 25.0        # minimum; 40 mm conservative
EDGE_EFF_TURNON_MM = 25.0            # efficiency 0->96 % over this band

# sparks (PLAN_39): NO post-spark dead time; only the in-spark coincidence
# loss. Veto the window, keep the neighbours.
SPARK = dict(rate_hz={3: 0.33, 2: None, 6: None, 7: 1.58},
             crossing_frac={3: 0.044, 2: 0.055, 6: 0.240, 7: 0.319})

# ---------------------------------------------------------------------------
# detector identity map: bench letter/number <-> July beam FEUs
# July convention (ntof_july_analysis/README.md): mx17_A/B/C/D with
# FEUs A:[3,4] B:[5,6] C:[7,8] D:[1,2]. WHICH FEU IS X vs Y MUST BE READ
# from the run's run_config.json detectors[].dream_feus (or autodetected
# from the strip map) — never assumed.
# ---------------------------------------------------------------------------
DETECTORS = {
    'A': dict(mx17=3, july_feus=(3, 4), bench_feus=(7, 8),
              bench='best performer: eff 92.9 % (5 mm fid), hybrid 1.75 deg'),
    'B': dict(mx17=2, july_feus=(5, 6), bench_feus=(6, 8),
              bench='healthy: eff 90.6 %, hybrid self 2.47 deg'),
    'C': dict(mx17=6, july_feus=(7, 8), bench_feus=None,
              bench='spark-limited on bench (24 %); angular r=0.86'),
    'D': dict(mx17=7, july_feus=(1, 2), bench_feus=None,
              bench='most spark-limited (32 %); good core sigma'),
}


def magboltz_v_of_E(gas_json_path):
    """Load a garfield_sim/results/*.json drift table and return an
    interpolator E [V/cm] -> v [um/ns]. Schema: check the file — the
    bench-era files store lists of E and v; adapt if the beam-gas files
    differ."""
    import json
    with open(gas_json_path) as f:
        d = json.load(f)
    # bench-era convention: {'E_V_cm': [...], 'v_um_ns': [...]} or similar —
    # be tolerant of the two known layouts.
    for ek, vk in (('E_V_cm', 'v_um_ns'), ('E', 'v'), ('fields', 'velocities')):
        if ek in d and vk in d:
            E = np.asarray(d[ek], float)
            v = np.asarray(d[vk], float)
            return lambda x: np.interp(x, E, v)
    raise KeyError(f'unrecognized drift-table schema in {gas_json_path}: '
                   f'{list(d.keys())[:8]}')
