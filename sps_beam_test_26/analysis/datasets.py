#!/usr/bin/env python3
"""Flat-mount det4 datasets, and their conditions, defined once.

This file exists for the same reason `mapping_check/run61_conditions.py` does:
the run_61 scans were once labelled with a single mount angle in two
copy-pasted scripts that then drifted, and half of each combined curve turned
out to be at the other angle.  Nothing here is allowed to be restated in an
analysis script.

A **condition** is a (mount, gas, drift, resist) tuple.  Per the repo rule a
calibration used outside its condition is a silent error, so every plateau
below carries all four, and plateaus are never merged across them.

Times are seconds since midnight of the dataset's `base_date`, so run_63 --
which crosses midnight -- stays monotonic.

Authority for each field:
  mount/gas      RUN_TIMELINE.md epochs (NOT run_config.json, which is stale
                 for gas, mount and start_time on this whole campaign)
  sub-run bounds <run>/dream_daq.log
  HV plateaus    the recovered hv_monitor.csv traces, cross-checked against
                 our own scan logs
  ZS sigma       the `_thr.prg` header copied into the sub-run directory
  peaking_ns     the DREAM shaping time -- a hardware constant, see below

**`peaking_ns` is 180 for every dataset here and that is correct.**  Several
sub-runs are named `cfg_gain3.0_peaktime200_opt`, `cfg_gain4.5_peaktime100`,
`cfg_gain3.0_peaktime50` and so on.  Those gains and peaking times are **P2's
VMM settings** -- banco named their runs after what *they* were scanning.  det4
reads out through the DREAM chain, which was untouched: verified 2026-08-05 by
md5-ing every `P2B_Beam.cfg` copy staged from runs 55/57/61/62/63/64/66/68/70
(29 files).  Each `Feu * Dream * <n>` register line has exactly one value
across all of them (register 1 = `0x081f 0xd023` everywhere), and the only
lines that differ between any two configs are `Sys NbOfSamples` (32/64),
`Sys PedRun Threshold` (5.00/4.00 sigmas) and `Sys DaqRun Time` (banco's
sub-run length).  Do NOT "correct" peaking_ns from a sub-run name.

Across the whole campaign the only det4 settings we ever changed were the two
HV channels (8:8 drift, 12:2 resist -- driven outside banco's DAQ, which is
why they read `null` in run_config's `hvs`), the sample count, and the ZS
threshold / RAW-vs-ZS.
"""
from __future__ import annotations


def hms(day, h, m, s=0.0):
    """Seconds since midnight of day 0 of the dataset."""
    return day * 86400 + h * 3600 + m * 60 + s


STAGE_ROOT = "/media/dylan/data/x17/sps_run53_det4_check/staging/"

DATASETS = {
    # ------------------------------------------------------------------ run 56
    "run56_m70V": dict(
        stage=STAGE_ROOT + "run_56_m70V/",
        base_date="2026-08-01",
        mount="flat", gas="Ar/CO2/iso 95/3/2",
        n_samples=64, sample_ns=60, zs_sigma=5.0, peaking_ns=180,
        ped="EicP2Bt_pedestals_pedthr_260801_12H19_000_03.fdf",
        # (sub-run, stem, t0, [file indices])
        subruns=[("meshscan_m70V",
                  "EicP2Bt_meshscan_m70V_datrun_260801_15H47_",
                  hms(0, 15, 47, 25.058), ["000", "001"])],
        # (label, t_lo, t_hi, drift_V, resist_V)
        plateaus=[("590V", hms(0, 15, 47, 25), hms(0, 15, 52, 50), 700, 590.0),
                  ("625V", hms(0, 15, 52, 57), hms(0, 15, 59, 34), 700, 624.7)],
        z_det4=1120.0,
        note="highest voltage det4 ever ran flat; resist steps inside the sub-run",
    ),

    # ------------------------------------------------------------------ run 63
    # run_63 straddles a zone access, so it is TWO conditions and must never be
    # treated as one.  The H4 TAX beam stopper (XTAX_022_023:POSITION_MEAS,
    # logged only by the mx17-daq NXCALS client -- see the memory note on the
    # two feeds) dates it to the second:
    #
    #   moving  00:37:16 - 00:40:10     closing
    #   BLOCKED 00:40:11 - 00:57:55     the access, 17.7 min
    #   moving  00:57:56 - 01:00:49     opening
    #   open    01:00:50 ->             beam back
    #
    # det4 was rotated from 25.64 deg back to FLAT during that access.  What
    # first looked like a "beam dip" inside operating_01 is the stopper, not a
    # machine fault.
    "run63_rot25": dict(
        stage=STAGE_ROOT + "run_63/",
        base_date="2026-08-02",
        mount="25.64 deg", gas="Ar/CF4/iso 88/10/2",
        n_samples=64, sample_ns=60, zs_sigma=4.0, peaking_ns=180,
        ped="EicP2Bt_pedestals_pedthr_260802_15H04_000_03.fdf",
        subruns=[
            ("operating_00", "EicP2Bt_operating_00_datrun_260802_23H53_",
             hms(0, 23, 53, 9.027), ["000", "001", "002", "003", "004", "005",
                                     "006"]),
            ("operating_01", "EicP2Bt_operating_01_datrun_260803_00H23_",
             hms(1, 0, 23, 25.276), ["000", "001", "002"]),
        ],
        plateaus=[
            # the drift ladder at fixed resist 769.8 V, pre-access.  Scan C
            # (short dwells) then scan D (8 min each).  Everything from d225 on
            # is inside the stopper window and has little or no beam.
            ("d675", hms(0, 23, 52, 59), hms(0, 23, 56, 25), 675.5, 769.8),
            ("d575", hms(0, 23, 56, 29), hms(1, 0, 1, 41), 575.3, 769.8),
            ("d475", hms(1, 0, 1, 44), hms(1, 0, 5, 24), 475.3, 769.8),
            ("d625", hms(1, 0, 6, 14), hms(1, 0, 14, 16), 625.3, 769.8),
            ("d525", hms(1, 0, 14, 20), hms(1, 0, 22, 20), 525.2, 769.8),
            ("d425", hms(1, 0, 22, 24), hms(1, 0, 30, 27), 425.2, 769.8),
            ("d325", hms(1, 0, 30, 29), hms(1, 0, 37, 16), 325.1, 769.8),
        ],
        z_det4=1120.0,
        note="PRE-access, so 25.64 deg -- measured det(A) 1.1132 (1/cos = "
             "26.1 deg) against 1.009 for a genuinely flat mount.  A drift "
             "ladder at fixed resist; the flat-geometry kernel argument does "
             "NOT apply here, this needs the full wft forward fit with w != 0.",
    ),

    "run63_flat": dict(
        stage=STAGE_ROOT + "run_63/",
        base_date="2026-08-02",
        mount="flat", gas="Ar/CF4/iso 88/10/2",
        n_samples=64, sample_ns=60, zs_sigma=4.0, peaking_ns=180,
        ped="EicP2Bt_pedestals_pedthr_260802_15H04_000_03.fdf",
        subruns=[
            # operating_02 straddles the stopper: only its tail (from 01:00:50)
            # is flat AND has beam.  Not staged locally yet.
            ("operating_02", "EicP2Bt_operating_02_datrun_260803_00H53_",
             hms(1, 0, 53, 41.226), ["000", "001", "002", "003", "004"]),
            ("operating_03", "EicP2Bt_operating_03_datrun_260803_01H24_",
             hms(1, 1, 24, 6.364), ["000", "001", "002", "003", "004"]),
        ],
        plateaus=[
            # flat, beam back at 01:00:50, drift and resist both untouched
            # since 00:55:40 -- one single condition for 53.4 min.
            ("flat700", hms(1, 1, 0, 50), hms(1, 1, 54, 11), 700.4, 769.8),
        ],
        z_det4=1120.0,     # refit; run_config.json's 1155 is stale
        note="POST-access: flat, at the operating point, 53.4 min of beam, "
             "ZS one step looser (4 sigma) than run_56.  This is the dataset "
             "the flat charge-spreading measurement wanted.  No drift lever "
             "in THIS run -- but 'every drift scan ran before the access' is "
             "WRONG (corrected 2026-08-05 late): run_68 and run_70 are flat "
             "CF4 drift scans taken AFTER it, run_68's at 64 samples.  "
             "See EXTRACTION_2026-08-05b.md section 8.",
    ),
    # ------------------------------------------------------------------ run 71
    # The RAW run, taken 2026-08-03 05:22-05:52 specifically to kill the zero
    # suppression systematic that limited tau_s and c2 in run_56/run_63.
    #
    #   zero_suppress = False          no channel or sample censoring at all
    #   pedestal_subtraction = False   raw per-channel baselines -> --zs-baseline 0
    #   common_noise_subtraction=False on-FEU CM off -> CNS must be done in software
    #   64 x 60 ns, latency 32         unchanged, so the pulse still sits ~sample 25
    #
    # Mount is unchanged from run63_flat: the H4 TAX stopper was open
    # continuously 01:00:50 -> 06:03:08, so no access intervened and det4 is
    # still flat.  Resist held 769.8 V throughout; drift stepped 700 -> 450 ->
    # 275 V, which is the diffusion lever with the resistive layer untouched.
    #
    # NB the sub-run is named `cfg_gain4.5_peaktime50` -- that is P2's label for
    # their own config scan, not ours.  The Dream register in the run's cfg is
    # `Feu * Dream * 1 0x081f 0xd023`, code (0xd023>>4)&0xF = 2 = 180 ns, the
    # same shaping as run_56/run_63.  Trust the register, not the sub-run name.
    "run71_raw": dict(
        stage=STAGE_ROOT + "run_71/",
        base_date="2026-08-03",
        mount="flat", gas="Ar/CF4/iso 88/10/2",
        n_samples=64, sample_ns=60, zs_sigma=5.0, peaking_ns=180,
        raw=True,
        ped="EicP2Bt_pedestals_pedthr_260802_15H04_000_03.fdf",
        subruns=[("cfg_gain4.5_peaktime50",
                  "EicP2Bt_cfg_gain4.5_peaktime50_datrun_260803_05H22_",
                  hms(0, 5, 22, 26.908),
                  # banco's pipeline produced the LAST groups first, so these
                  # are what has uRWELL combined_hits; 000-022 (the bulk of the
                  # 700 V block) are not processed yet.
                  # 002-007 pulled from EOS 2026-08-04: the 700 V drift block
                  # (05:24-05:29, well inside the raw700 plateau), det4-only.
                  ["002", "003", "004", "005", "006", "007",
                   "023", "024", "025", "026", "027", "028", "029", "030",
                   "031", "032", "033", "034", "035"])],
        plateaus=[
            # 700 V: only file 023's leading ~20 s survives in the staged set
            ("raw700", hms(0, 5, 22, 30), hms(0, 5, 41, 55), 700.5, 769.8),
            ("raw450", hms(0, 5, 42, 31), hms(0, 5, 47, 21), 450.2, 769.8),
            ("raw275", hms(0, 5, 47, 41), hms(0, 5, 52, 26), 275.0, 769.8),
        ],
        z_det4=1100.0,
        note="RAW, no zero suppression -- the run that turns tau_s and c2 from "
             "acceptance-limited into measurements.  Two drift points at fixed "
             "resist give the invariance test: alpha (diffusion) should move, "
             "beta (the kernel) should not.",
    ),
}

# ------------------------------------------------------------------ run 60
# The gas-transition overnight: Ar/CO2/iso -> Ar/CF4/iso switched during the
# 20:24-21:10 access (TAX-dated, GAS_FLUSH_TIMELINE.md), run started 21:20.
# At ~2 ln/h into ~4.6 L the mixture exchanges with tau ~ 2.3-4.6 h, so the
# 24 x 30 min sub-runs ARE the flush transient. SPS FTARGET stopped ~04:50
# (spill record), so overnight_15..23 are beamless.  HV fixed: drift 700.5 V,
# resist 649.75 V.  ZS 5 sigma (21H12 prg header), 64 x 60 ns.
_R60_T0 = {  # sub-run -> start (s since 2026-08-01 midnight), from dream_daq.log
    i: hms(0, 21, 20, 5) + i * (30 * 60 + 15) for i in range(24)
}
DATASETS["run60_flush"] = dict(
    stage=STAGE_ROOT + "run_60/",
    base_date="2026-08-01",
    mount="25.64 deg", gas="Ar/CO2->CF4 TRANSITION (see GAS_FLUSH_TIMELINE)",
    n_samples=64, sample_ns=60, zs_sigma=5.0, peaking_ns=180,
    ped="EicP2Bt_pedestals_pedthr_260801_21H12_000_03.fdf",
    subruns=[(f"overnight_{i:02d}",
              f"EicP2Bt_overnight_{i:02d}_datrun_*_",   # datrun stamp = SUB-RUN
              _R60_T0[i],                               # start; decoder globs
              [f"{j:03d}" for j in range(8)])
             for i in range(15)],                    # 15..23 beamless, skip
    plateaus=[(f"ov{i:02d}", _R60_T0[i], _R60_T0[i] + 30 * 60, 700.5, 649.75)
              for i in range(15)],
    z_det4=1120.0,
    note="THE FLUSH TRANSIENT: 0.6-8.1 h after the gas switch, fixed HV, "
         "fixed mount 25.64 deg. Do not use as either mixture; use to "
         "measure tau_flush.",
)

# ------------------------------------------------------------------ run 59
# The last CO2-mixture dataset: 64 samples, started 20:00:54, beam KILLED at
# 20:24:07 by the gas-change access (TAX record) -- so ~22 min of beam in
# detE_long_00 and nothing usable after.  The CO2-side anchor for the flush
# fit, 40 min before the switch, same 25.64 deg mount as run_60.
# ZS nominally 3 sigma from 18:12 (ZS_TIMELINE); the prg copy in the run dir
# is the stale 16H21 5-sigma set -- verify from the data.
DATASETS["run59_co2"] = dict(
    stage=STAGE_ROOT + "run_59/",
    base_date="2026-08-01",
    mount="25.64 deg", gas="Ar/CO2/iso 95/3/2",
    # the 18:12 threshold reload reused the 16H21 pedestal data (the
    # pedestals_08-01-26_18-12-15 dir on EOS contains the 16H21 fdf)
    n_samples=64, sample_ns=60, zs_sigma=3.0, peaking_ns=180,
    ped="EicP2Bt_pedestals_pedthr_260801_16H21_000_03.fdf",
    subruns=[("detE_long_00", "EicP2Bt_detE_long_00_datrun_260801_20H02_",
              hms(0, 20, 2, 0), [f"{j:03d}" for j in range(20)])],
    plateaus=[("co2", hms(0, 20, 2, 0), hms(0, 20, 24, 7), 700.5, 649.75)],
    z_det4=1120.0,
    note="beam dies 20:24:07; HV assumed same as run_60's (verify from "
         "hv_monitor.csv before quoting)",
)

# ------------------------------------------------------------------ run 61
# The unanalyzed tail of run_61: meshscan_m70V..m100V (17:34-18:57) are NOT
# scan points -- the resist scan ended 17:14 and the HV then sat at the
# OPERATING POINT (drift 700.2 V, resist 769.8 V, hv_monitor ptp < 0.2 V
# after m70V's opening ramp) for ~80 min at 25.64 deg in CF4.  This is the
# high-statistics inclined operating dataset: the 4th (highest-field) plateau
# for the wft ladder fit and the X-tilt closure.  m100V loses beam 18:53:21
# (SPS linac; M100V_PARTIAL.md -- normalise by ~67 %).
DATASETS["run61_op25"] = dict(
    stage=STAGE_ROOT + "run_61/",
    base_date="2026-08-02",
    mount="25.64 deg", gas="Ar/CF4/iso 88/10/2",
    n_samples=32, sample_ns=60, zs_sigma=4.0, peaking_ns=180,
    ped="EicP2Bt_pedestals_pedthr_260802_15H04_000_03.fdf",
    subruns=[
        ("meshscan_m70V", "EicP2Bt_meshscan_m70V_datrun_260802_17H34_",
         hms(0, 17, 34, 16), [f"{j:03d}" for j in range(8)]),
        ("meshscan_m80V", "EicP2Bt_meshscan_m80V_datrun_260802_17H55_",
         hms(0, 17, 55, 48), [f"{j:03d}" for j in range(8)]),
        ("meshscan_m90V", "EicP2Bt_meshscan_m90V_datrun_260802_18H17_",
         hms(0, 18, 17, 13), [f"{j:03d}" for j in range(8)]),
        ("meshscan_m100V", "EicP2Bt_meshscan_m100V_datrun_260802_18H38_",
         hms(0, 18, 38, 48), [f"{j:03d}" for j in range(8)]),
    ],
    plateaus=[
        # one condition; m70V's first ~2 min are the resist ramp 580->770
        ("op25", hms(0, 17, 36, 30), hms(0, 18, 53, 21), 700.2, 769.8),
    ],
    z_det4=1120.0,
    note="25.64 deg AT the operating point (same HV as run63_flat/run71) -- "
         "the inclined high-stat block. Beam dies 18:53:21.",
)

# ------------------------------------------------------------------ run 66
# OUR flat resist (gain) scan, 780 -> 405 V in 25 V steps, 205 s/point, drift
# held 700.5 V.  Written by our own driver (`det4_resist_scan_780_400.csv`,
# archived under records/scan_logs_late/run_66/), so the windows below come
# from the driver, not from a monitor trace.
#
# ⚠ The sub-run is called `cfg_gain4.5_peaktime200_opt`.  That is P2's VMM
# configuration, NOT ours -- see the module docstring.  det4's Dream shaping is
# identical here to every other run of the campaign.
#
# WHY THIS DATASET MATTERS: kernel gain-invariance is currently established
# over run_56's flat 590 -> 625 V, a 6 % swing.  This is a factor ~1.9 in
# resist voltage in the SAME flat geometry, where the sharing kernel is
# measurable directly rather than through a track fit.  It is the widest gain
# lever the campaign has at normal incidence.
#
# The scan runs 03:00:34 -> 03:56:35 and crosses into run_67 (which is NOT
# staged: its plateaus are 555 -> 405 V, far below det4's 769.8 V operating
# point, where little gain is expected).  Only the nine plateaus that fall
# inside run_66's sub-run window (03:00:29 - 03:30:38) are listed; r580 is
# truncated by the sub-run boundary.
#
# Beam: FTARGET extraction covers 02:13:14 - 04:00:45 at ~1240e10 per spill,
# so every plateau here has beam.  (Contrast runs 68/69, whose flat drift scan
# sat entirely inside the 04:00:45 - 04:59:19 beam-off gap and is worthless.)
DATASETS["run66_flat_resist"] = dict(
    stage=STAGE_ROOT + "run_66/",
    base_date="2026-08-03",
    mount="flat", gas="Ar/CF4/iso 88/10/2",
    n_samples=32, sample_ns=60, zs_sigma=4.0, peaking_ns=180,
    ped="EicP2Bt_pedestals_pedthr_260802_15H04_000_03.fdf",
    subruns=[
        ("cfg_gain4.5_peaktime200_opt",
         "EicP2Bt_cfg_gain4.5_peaktime200_opt_datrun_260803_03H00_",
         hms(0, 3, 0, 29.812), [f"{j:03d}" for j in range(4)]),
    ],
    plateaus=[
        # (label, t_lo, t_hi, drift_V, resist_V) -- +15 s settle, -3 s before
        # the next step.  RESIST is the scanned quantity here, drift is held.
        ("r780", hms(0, 3, 0, 49), hms(0, 3, 3, 56), 700.5, 779.75),
        ("r755", hms(0, 3, 4, 14), hms(0, 3, 7, 21), 700.5, 755.0),
        ("r730", hms(0, 3, 7, 39), hms(0, 3, 10, 46), 700.5, 730.0),
        ("r705", hms(0, 3, 11, 4), hms(0, 3, 14, 11), 700.5, 705.0),
        ("r680", hms(0, 3, 14, 29), hms(0, 3, 17, 36), 700.5, 680.0),
        ("r655", hms(0, 3, 17, 54), hms(0, 3, 21, 1), 700.5, 655.0),
        ("r630", hms(0, 3, 21, 19), hms(0, 3, 24, 26), 700.5, 629.75),
        ("r605", hms(0, 3, 24, 44), hms(0, 3, 27, 51), 700.5, 605.0),
        ("r580", hms(0, 3, 28, 9), hms(0, 3, 30, 35), 700.5, 580.0),
    ],
    z_det4=1120.0,
    note="OUR flat resist/gain scan 780->580 V at fixed drift 700.5 V, in the "
         "flat CF4 epoch (same conditions as run63_flat/run71 otherwise). The "
         "widest gain lever at normal incidence; run_56's flat kernel "
         "invariance covered only 590->625 V.",
)

# ------------------------------------------------------------------ run 70
# OUR flat CF4 drift scan, 600 -> 100 V at fixed resist 769.75 V.  Together
# with run_68/69 (which are beamless, see below) this is what falsifies the
# long-standing "no drift lever in the flat data" claim -- see
# FLAT_CF4_RUN63.md section 4, retracted 2026-08-05.
#
# ⚠ sub-run name `cfg_gain3.0_peaktime50` is P2's VMM config, not ours.
#
# ⚠ 32 SAMPLES = 1.92 us window, and the CF4 drift ladder is 2.0-2.5 us long,
# so this dataset CANNOT give v(E) -- t90 rails at the window edge exactly as
# run61_rot15_ladder does.  What it gives is the charge / mesh-transparency
# lever at NORMAL incidence, the flat counterpart of run_61's 15 deg version.
#
# Windows MEASURED from the sub-run's own hv_monitor.csv (1,816 rows at 1 Hz),
# which is cleaner than the scan CSV; +15 s settle, -3 s before the step.
#
# Beam: FTARGET resumes 04:59:19 after an hour-long gap and runs to 06:00:31,
# so the whole scan has beam -- but the FIRST 700 V dwell starts at 04:50:44,
# i.e. 8.6 min before beam returns, so d700's window below starts at 04:59:35
# and is short.  d700b is the return leg at the end of the scan: same
# condition, independent slice, so the two are an internal consistency check.
DATASETS["run70_flat_drift"] = dict(
    stage=STAGE_ROOT + "run_70/",
    base_date="2026-08-03",
    mount="flat", gas="Ar/CF4/iso 88/10/2",
    n_samples=32, sample_ns=60, zs_sigma=4.0, peaking_ns=180,
    ped="EicP2Bt_pedestals_pedthr_260802_15H04_000_03.fdf",
    subruns=[
        ("cfg_gain3.0_peaktime50",
         "EicP2Bt_cfg_gain3.0_peaktime50_datrun_260803_04H50_",
         hms(0, 4, 50, 44.906), [f"{j:03d}" for j in range(3)]),
    ],
    plateaus=[
        ("d700", hms(0, 4, 59, 35), hms(0, 5, 1, 0), 700.4, 769.75),
        ("d600", hms(0, 5, 1, 22), hms(0, 5, 5, 15), 600.4, 769.75),
        ("d500", hms(0, 5, 5, 37), hms(0, 5, 7, 50), 500.4, 769.75),
        ("d400", hms(0, 5, 8, 10), hms(0, 5, 10, 26), 400.2, 769.75),
        ("d300", hms(0, 5, 10, 47), hms(0, 5, 13, 1), 300.0, 769.75),
        ("d200", hms(0, 5, 13, 22), hms(0, 5, 15, 35), 200.0, 769.75),
        ("d100", hms(0, 5, 15, 56), hms(0, 5, 18, 11), 100.0, 769.75),
        ("d700b", hms(0, 5, 19, 30), hms(0, 5, 21, 9), 700.5, 769.75),
    ],
    z_det4=1120.0,
    note="OUR flat CF4 drift scan 600->100 V at the operating resist. NO v(E) "
         "(32 samples, window-railed) -- this is the charge/transparency "
         "lever at normal incidence. d700/d700b bracket the scan at the same "
         "condition and should agree.",
)

# ------------------------------------------------------------------ run 62
# THE SECOND 25.64 deg CF4 DRIFT LADDER -- an independent repeat of
# run63_rot25, taken 90 minutes earlier, and absent from every analysis
# document until 2026-08-05 (late).  run_62 appears exactly once in the whole
# record (GAS_FLUSH_TIMELINE §5, and only as "the gas is fully exchanged by
# then"); RUN_TIMELINE's "what is analysed" table does not list it at all.
#
# Conditions are the SAME as run63_rot25 -- 25.64 deg, Ar/CF4/iso, 64 samples,
# resist held 769.75 V -- so this is a genuine reproducibility check on the
# v(E) curve, not a new measurement.  The scan is scripted and timestamped
# (`det4_drift_scan_A_700_100.csv`, archived under
# records/scan_logs_late/run_62/): 7 points, 700 -> 100 V, 375 s each.
#
# Only scan A overlaps data-taking.  Scan B (650 -> 50 V, 22:50-23:36) ran
# after the last sub-run with FEU03 in it -- driftscan_gap250V and up are
# 66 kB stubs -- so scan B has no det4 data and is NOT represented here.
#
# The d500 window straddles the gap150V/gap200V sub-run boundary (gap150V ends
# 22:23:49, gap200V starts 22:24:09); plateau_of labels by wall time, so the
# window below simply loses those 20 s.  d300/d200 are far enough down the
# field that the ladder overruns the 3.84 us window, exactly as run63_rot25's
# low-field points do -- lower bounds, not measurements.
DATASETS["run62_rot25_ladder"] = dict(
    stage=STAGE_ROOT + "run_62/",
    base_date="2026-08-02",
    mount="25.64 deg", gas="Ar/CF4/iso 88/10/2",
    n_samples=64, sample_ns=60, zs_sigma=4.0, peaking_ns=180,
    ped="EicP2Bt_pedestals_pedthr_260802_15H04_000_03.fdf",
    subruns=[
        ("driftscan_gap150V", "EicP2Bt_driftscan_gap150V_datrun_260802_22H02_",
         hms(0, 22, 2, 34.275), [f"{j:03d}" for j in range(4)]),
        ("driftscan_gap200V", "EicP2Bt_driftscan_gap200V_datrun_260802_22H24_",
         hms(0, 22, 24, 9.857), [f"{j:03d}" for j in range(2)]),
    ],
    plateaus=[
        # scan-A dwells, +15 s settle, -3 s before the step
        ("d700", hms(0, 22, 7, 23), hms(0, 22, 13, 15), 700.5, 769.75),
        ("d600", hms(0, 22, 13, 33), hms(0, 22, 19, 30), 600.5, 769.75),
        ("d500", hms(0, 22, 19, 48), hms(0, 22, 25, 45), 500.5, 769.75),
        ("d400", hms(0, 22, 26, 3), hms(0, 22, 32, 0), 400.25, 769.75),
        ("d300", hms(0, 22, 32, 18), hms(0, 22, 38, 15), 300.25, 769.75),
        ("d200", hms(0, 22, 38, 33), hms(0, 22, 44, 30), 200.0, 769.75),
        # d100's dwell starts 22:44:33 but gap200V ends 22:45:23 -- ~50 s of
        # data, below ladder_span's 500-hit floor.  Deliberately omitted.
    ],
    z_det4=1120.0,
    note="the SECOND 25.64 deg CF4 drift ladder, same conditions as "
         "run63_rot25 (resist 769.75 V, 64 samples) but different drift "
         "points (700/600/500/400/300/200 vs 675/625/575/525/475/425/325). "
         "The one true reproducibility check on the beam v(E) curve.",
)

# The 15.465 deg drift ladder -- run_61's m20V + the FIRST m30V pass.
#
# `det4_drift_scan.log` is a scripted, fully time-stamped scan (resist held
# 750.0 V, 10 drift points, 5 min each), so these windows come from the driver
# itself rather than from the monitor trace: each entry below is the logged
# "arrived at" +10 s to the logged "point done" -5 s.
#
#   1  700.5 V  13:13:06   6  350.0 V  13:38:31
#   2  630.2    13:18:11   7  280.0    13:43:36
#   3  560.2    13:23:16   8  210.0    13:48:41
#   4  490.2    13:28:21   9  140.0    13:53:46
#   5  420.2    13:33:26  10   70.0    13:58:51  (killed 14:00:46)
#
# THE TRAP THIS ENTRY EXISTS FOR: `meshscan_m30V` names TWO different sub-runs
# 20 minutes apart in wall-clock terms but two hours and one ACCESS apart in
# condition -- datrun 260802_13H46 is the 15.465 deg ladder tail, datrun
# 260802_16H08 is at 25.64 deg with the resist scan creeping underneath it.
# Only the 13H46 stem is listed here.  Never glob `meshscan_m30V_*`.
#
# Point 10 (70 V) is 91 s of data and ends in the 14:00:46 end-of-run
# power-off that took drift to 23.5 V and resist to 642.5 V; the window stops
# at the sub-run boundary, but treat that point as indicative only.
DATASETS["run61_rot15_ladder"] = dict(
    stage=STAGE_ROOT + "run_61/",
    base_date="2026-08-02",
    mount="15.465 deg", gas="Ar/CF4/iso 88/10/2",
    # the _thr.prg copied in is the Sat 21H12 set, header "5.000000 sigmas";
    # RUN_TIMELINE §3 has det E at 3 sigma from Sat 18:12 and the header was
    # stale for run_57 for exactly that reason, so this is the one field here
    # that is NOT independently confirmed.  It only sets the analyzer's
    # software threshold; the span estimator's amp >= 150 gate sits far above
    # either value, so no ladder number depends on it.
    n_samples=32, sample_ns=60, zs_sigma=5.0, peaking_ns=180,
    ped="EicP2Bt_pedestals_pedthr_260801_21H12_000_03.fdf",
    subruns=[
        ("meshscan_m20V", "EicP2Bt_meshscan_m20V_datrun_260802_13H15_",
         hms(0, 13, 15, 52.090), [f"{j:03d}" for j in range(7)]),
        ("meshscan_m30V", "EicP2Bt_meshscan_m30V_datrun_260802_13H46_",
         hms(0, 13, 46, 19.679), ["000", "001", "002"]),
    ],
    plateaus=[
        # d700 is clipped at the front: the sub-run only starts at 13:15:52,
        # 2.8 min into the point
        ("d700", hms(0, 13, 15, 52), hms(0, 13, 18, 1), 700.5, 750.0),
        ("d630", hms(0, 13, 18, 21), hms(0, 13, 23, 6), 630.2, 750.0),
        ("d560", hms(0, 13, 23, 26), hms(0, 13, 28, 11), 560.2, 750.0),
        ("d490", hms(0, 13, 28, 31), hms(0, 13, 33, 16), 490.2, 750.0),
        ("d420", hms(0, 13, 33, 36), hms(0, 13, 38, 21), 420.2, 750.0),
        ("d350", hms(0, 13, 38, 41), hms(0, 13, 43, 26), 350.0, 750.0),
        ("d280", hms(0, 13, 43, 46), hms(0, 13, 48, 31), 280.0, 750.0),
        ("d210", hms(0, 13, 48, 51), hms(0, 13, 53, 36), 210.0, 750.0),
        ("d140", hms(0, 13, 53, 56), hms(0, 13, 58, 41), 140.0, 750.0),
        ("d070", hms(0, 13, 59, 1), hms(0, 14, 0, 32), 70.0, 750.0),
    ],
    z_det4=1120.0,
    note="the 15.465 deg CF4 drift ladder -- same gas as run63_rot25's 25.64 "
         "deg ladder but a different mount angle AND a different resist "
         "(750.0 vs 769.8 V), so agreement of the two v(E) curves is a real "
         "cross-check of how the forward fit handles the inclination, not a "
         "repeat measurement.",
)

# ------------------------------------------------------------------ run 55
# The FLAT CO2 drift scan (pre-rotation, Sat 14:25-14:55): drift 700/600/500/
# 400 V at resist 549.8 V, 600 s/point, killed at the 5th point by the
# end-of-run power-off.  FLAT_CF4_RUN63.md said "no drift lever in the flat
# data" -- true of the CF4 era only; THIS is the flat drift lever, in the
# CO2 mixture.  Plateau windows from the hv_monitor drift trace (the 12-min
# sub-run boundaries do not line up with the 10-min points).
DATASETS["run55_flatdrift"] = dict(
    stage=STAGE_ROOT + "run_55/",
    base_date="2026-08-01",
    mount="flat", gas="Ar/CO2/iso 95/3/2",
    n_samples=64, sample_ns=60, zs_sigma=5.0, peaking_ns=180,
    ped="EicP2Bt_pedestals_pedthr_260801_12H19_000_03.fdf",
    subruns=[
        ("meshscan_m00V", "EicP2Bt_meshscan_m00V_datrun_*_",
         hms(0, 14, 15, 23), [f"{j:03d}" for j in range(8)]),
        ("meshscan_m10V", "EicP2Bt_meshscan_m10V_datrun_*_",
         hms(0, 14, 27, 49), [f"{j:03d}" for j in range(8)]),
        ("meshscan_m20V", "EicP2Bt_meshscan_m20V_datrun_*_",
         hms(0, 14, 40, 15), [f"{j:03d}" for j in range(8)]),
        ("meshscan_m30V", "EicP2Bt_meshscan_m30V_datrun_*_",
         hms(0, 14, 52, 42), [f"{j:03d}" for j in range(8)]),
    ],
    plateaus=[
        # MEASURED 2026-08-05 from the four staged meshscan_*/hv_monitor.csv
        # (channel 8:8 = drift, 12:2 = resist), 2,434 rows at 1 Hz covering
        # 14:13:43-14:55:05 with no gap.  The dwells are:
        #   700.3 V  14:13:43-14:25:45 (722 s)   resist 549.75 throughout
        #   600.3 V  14:25:48-14:35:51 (603 s)
        #   500.3 V  14:35:54-14:46:08 (614 s)
        #   400.1 V  14:46:11-14:55:05 (534 s, cut by the 14:57 power-off)
        # Windows below take a +15 s settle margin and stop 3 s before the
        # step.  The PREVIOUS windows (detE_scan.log cadence, 600 s from
        # 14:25) each ran 20-45 s PAST the step into the next, slower point --
        # which is a drift-time contamination of exactly the kind the span
        # estimator is sensitive to.  Do not restore them.
        ("d700", hms(0, 14, 13, 58), hms(0, 14, 25, 42), 700.3, 549.75),
        ("d600", hms(0, 14, 26, 3), hms(0, 14, 35, 48), 600.3, 549.75),
        ("d500", hms(0, 14, 36, 9), hms(0, 14, 46, 5), 500.3, 549.75),
        ("d400", hms(0, 14, 46, 26), hms(0, 14, 55, 2), 400.1, 549.75),
    ],
    z_det4=1120.0,
    note="flat CO2 drift ladder -- prompt-diffusion alpha(E), v(E) in the "
         "CO2 mixture, and the ZS-era kernel-vs-field cross-check. Plateau "
         "windows MEASURED from hv_monitor 2026-08-05 (see above); the 400 V "
         "point is short (534 s, ended by the power-off).",
)

# ------------------------------------------------------------- runs 57 + 58
# THE SATURDAY 25.64 deg DRIFT LADDER, in the CO2 mixture -- the mixture twin
# of run63_rot25.  17 points, drift 700 -> 10 V at fixed resist 669.75 V,
# 18:04-19:45, spanning run_57 driftscan_gap350V -> run_58 operating_02.
# Staged 2026-08-05 (wave 3), HIGH-FIELD HALF ONLY: run_58 is 86 GB of FEU03
# against 37 GB of free disk, and its points (500 V and below) are where the
# ladder starts running off the 3.84 us window anyway.  The four staged points
# are 243/226/208/191 V/cm -- deliberately the same range as run63_rot25's top
# four (235/217/200/182 V/cm), so the two mixtures compare point for point.
#
# HV windows MEASURED 2026-08-05 from the run_57+run_58 hv_monitor traces
# (58,115 rows, channel 8:8 drift / 12:2 resist, no gap 16:35 -> 09:26+1):
#   700.3 V  18:00:00-18:15:00   resist 674 -> 669.75 (settling from the
#                                17:00-18:03 resist ladder)
#   650.3 V  18:15:03-18:20:07   resist 669.75 from here on
#   600.3 V  18:20:09-18:25:22
#   550.2 V  18:25:25-18:29:00   ended by the sub-run boundary
#   [HV off 18:30:47-18:32:53 = the end-of-run power-off between 57 and 58]
#   500.3 V  18:33:43-18:40:17   resist mean 623 -- still RECOVERING from the
#                                power-off, so run_58's first point is NOT
#                                gain-comparable with the rest.  Recorded here
#                                because it is not obvious from the setpoints.
#   450/400/350/300/250/200/150/100/70/40/20/10 V, ~314 s each, to 19:45:19
#
# ZS CAVEAT -- the one thing the _thr.prg cannot tell you.  Both sub-run
# directories carry the 16H21 pedestal run's _thr.prg, whose header says
# "5.000000 sigmas".  It is STALE: our own script dropped det E to 2 sigma at
# 18:04 and raised it to 3 sigma at 18:12 without a new pedestal run, and
# RUN_TIMELINE §3's rate table is the evidence (FEU3 2.08 -> 23.76 MB/s across
# exactly that boundary).  So the 700 V plateau straddles a threshold change
# and the window below starts at 18:12:30 to keep every plateau at 3 sigma;
# 650/600/550 are clean.  Do not "fix" the window back to the full dwell.
DATASETS["run57_rot25_co2"] = dict(
    stage=STAGE_ROOT + "run_57/",
    base_date="2026-08-01",
    mount="25.64 deg", gas="Ar/CO2/iso 95/3/2",
    n_samples=64, sample_ns=60, zs_sigma=3.0, peaking_ns=180,
    ped="EicP2Bt_pedestals_pedthr_260801_16H21_000_03.fdf",
    subruns=[
        ("driftscan_gap350V", "EicP2Bt_driftscan_gap350V_datrun_260801_18H04_",
         hms(0, 18, 4, 26.287), [f"{j:03d}" for j in range(17)]),
        ("driftscan_gap400V", "EicP2Bt_driftscan_gap400V_datrun_260801_18H16_",
         hms(0, 18, 16, 55.880), [f"{j:03d}" for j in range(11)]),
    ],
    plateaus=[
        # +15 s settle after each step, stop 3 s before the next
        ("d700", hms(0, 18, 12, 30), hms(0, 18, 14, 57), 700.3, 669.75),
        ("d650", hms(0, 18, 15, 18), hms(0, 18, 20, 4), 650.3, 669.75),
        ("d600", hms(0, 18, 20, 24), hms(0, 18, 25, 19), 600.3, 669.75),
        ("d550", hms(0, 18, 25, 40), hms(0, 18, 28, 57), 550.2, 669.75),
    ],
    z_det4=1120.0,
    note="Saturday's 25.64 deg drift ladder in Ar/CO2/iso -- the CO2 twin of "
         "run63_rot25, same four top fields.  Resist 669.75 V throughout (NOT "
         "run_60's 649.75).  The 500 V and lower points live in run_58 and are "
         "NOT staged: 86 GB, and 500 V is where run_58's resist was still "
         "recovering from the 18:30 power-off.",
)

#: plateaus of run_63 that live in sub-runs we did NOT stage (operating_00/02)
UNSTAGED_NOTE = (
    "run_63 drift points 675/575/475/625/525 sit in operating_00 and the "
    "125 V tail runs into operating_02; neither is staged locally yet."
)


def get(name):
    d = DATASETS[name]
    return d


def plateau_of(name, t):
    """Label of the plateau containing time t, or '' -- vectorised."""
    import numpy as np
    d = DATASETS[name]
    out = np.full(np.shape(t), "", dtype="<U12")
    for label, lo, hi, _, _ in d["plateaus"]:
        out[(t >= lo) & (t < hi)] = label
    return out


if __name__ == "__main__":
    for k, d in DATASETS.items():
        print(f"\n=== {k}  ({d['mount']}, {d['gas']}, ZS {d['zs_sigma']} sigma)")
        print(f"    {d['note']}")
        for lab, lo, hi, dr, re in d["plateaus"]:
            print(f"    {lab:>6}  drift {dr:6.1f}  resist {re:6.1f}   "
                  f"{(hi-lo)/60:5.1f} min")
