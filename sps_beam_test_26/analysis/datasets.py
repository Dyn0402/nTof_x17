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
             "here -- every drift scan ran before the access.",
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
