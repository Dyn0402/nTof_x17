#!/usr/bin/env python3
"""The run_61 measurement conditions, defined once.

This file exists because the same set of HV plateau windows was previously
copy-pasted into `resist_scan_*.py` and `gain_scan_*.py`, and the two copies
drifted: both ended up labelled "15.465 deg" even though the mount was rotated
back to 25.64 deg during the 14:00-16:06 gap, so half the points in each
combined curve were at the other angle. Anything that needs run_61's HV points
imports them from here.

A **condition** is a (mount angle, gas) pair. Per the repo rule, a calibration
or a scan used outside its condition is a silent error, so each condition owns
its own output directory and its own alignment fit — the two are never merged
into one curve.

Directory naming on the data disk is `<mount>_<gas>__<runs>`:

    flat_ArCO2iso_95-3-2__run53-56          flat,      Ar/CO2/iso 95/3/2
    rot25_ArCO2iso_95-3-2__run57            25.64 deg, Ar/CO2/iso 95/3/2
    rot15_ArCF4iso_88-10-2__run61_1214-1400 15.465deg, Ar/CF4/iso 88/10/2
    rot25_ArCF4iso_88-10-2__run61_1606on    25.64 deg, Ar/CF4/iso 88/10/2

Gas tracked P2's own line throughout (we ran the same mixture on our own line):
Ar/CO2/iso **95/3/2** from the Friday-night install until the ~21:00 access on
2026-08-01, Ar/CF4/iso **88/10/2** from then on. run_60 (21:20-09:26) was taken
while that exchange was still flushing and belongs to neither.

Mount angle is a rotation **about the vertical axis, right-hand rule**,
rotating the detector down; `DAQ_DETE_ROT_Y` / `DETE_ROT_Y_DEG` carry it.

`run_config.json` disagrees with all of this and is wrong — see
`sps_beam_test_26/analysis/RUN_TIMELINE.md`.
"""
import datetime as dt

BASE = "/home/dylan/x17/sps_run53_det4_check/"
PAIRED = BASE + "paired_npz/"

# Wall-clock start of each run_61 sub-run, from dream_daq.log. The paired npz
# caches store event timestamps relative to their own sub-run start, so this is
# what puts them on one absolute axis -- the HV scans' dwell points do not line
# up with the ~21 min sub-run boundaries.
SUBRUN_T0 = {
    "m10V": dt.datetime(2026, 8, 2, 12, 45, 22),
    "m40V": dt.datetime(2026, 8, 2, 16, 29, 53),
    "m50V": dt.datetime(2026, 8, 2, 16, 51, 19),
    "m60V": dt.datetime(2026, 8, 2, 17, 12, 50),
}

Z = 1120.0          # det4 plane, mm, in the uRWELL reference frame
DRIFT_HELD_V = 700.0

CONDITIONS = [
    dict(
        key="rot15_ArCF4iso_88-10-2__run61_1214-1400",
        mount_deg=15.465,
        gas="Ar/CF4/iso 88/10/2",
        label=("Ar/CF4/iso 88/10/2, 15.465 deg -- run 61 12:14-14:00, "
               "drift ~700 V, resist 725-790 V (unscripted creep)"),
        # No scripted ladder here: nothing scanned det4's resist on purpose
        # before the drift scan. Every stable plateau in the hv_monitor resist
        # trace (card 12 ch 2, 2 V tolerance, >=20 s dwell) is cut as its own
        # point, including the short ~30-45 s ramp steps between the two long
        # (~9 min) dwells at 764.8 V and 789.9 V.
        points={
            "m10V": [
                (724.8, "12:45:23", "12:46:04"),
                (729.7, "12:46:05", "12:48:58"),
                (734.8, "12:48:59", "12:49:26"),
                (739.8, "12:49:28", "12:50:10"),
                (744.8, "12:50:11", "12:50:53"),
                (750.0, "12:50:54", "12:51:23"),
                (754.8, "12:51:24", "12:52:06"),
                (764.8, "12:52:20", "13:01:38"),
                (769.8, "13:01:39", "13:02:10"),
                (774.8, "13:02:11", "13:02:39"),
                (779.7, "13:02:40", "13:03:08"),
                (785.0, "13:03:10", "13:03:37"),
                (789.9, "13:03:38", "13:13:00"),
            ],
        },
    ),
    dict(
        key="rot25_ArCF4iso_88-10-2__run61_1606on",
        mount_deg=25.64,
        gas="Ar/CF4/iso 88/10/2",
        label=("Ar/CF4/iso 88/10/2, 25.64 deg -- run 61 16:06 on, "
               "drift held 700 V, resist 720->580 V scripted"),
        # A real scripted ladder: det4_resist_scan_720_580.log/.csv in
        # runs/run_61/ on banco. 29 points, 720->580 V, 5 V steps, 60 s dwell.
        # Windows are [settled, point-done] from that log. Points that straddle
        # a sub-run boundary appear in both sub-runs and are merged by voltage.
        points={
            "m40V": [
                (719.8, "16:42:58", "16:43:58"),
                (715.0, "16:44:03", "16:45:03"),
                (710.0, "16:45:08", "16:46:08"),
                (705.0, "16:46:13", "16:47:13"),
                (699.8, "16:47:18", "16:48:18"),
                (694.8, "16:48:23", "16:49:23"),
                (690.0, "16:49:28", "16:50:28"),
                (684.8, "16:50:33", "16:50:58"),   # m40V ends 16:50:58
            ],
            "m50V": [
                (684.8, "16:51:19", "16:51:33"),   # continuation
                (680.0, "16:51:38", "16:52:38"),
                (675.0, "16:52:43", "16:53:43"),
                (669.8, "16:53:48", "16:54:48"),
                (665.0, "16:54:53", "16:55:53"),
                (659.8, "16:55:58", "16:56:58"),
                (654.8, "16:57:03", "16:58:03"),
                (649.8, "16:58:08", "16:59:08"),
                (644.8, "16:59:13", "17:00:13"),
                (639.8, "17:00:18", "17:01:18"),
                (634.8, "17:01:23", "17:02:23"),
                (629.8, "17:02:28", "17:03:28"),
                (624.8, "17:03:33", "17:04:33"),
                (619.8, "17:04:38", "17:05:38"),
                (614.8, "17:05:43", "17:06:43"),
                (610.0, "17:06:48", "17:07:48"),
                (605.0, "17:07:53", "17:08:53"),
                (599.8, "17:08:58", "17:09:58"),
                (595.0, "17:10:03", "17:11:03"),
                (590.0, "17:11:08", "17:12:08"),
                (584.8, "17:12:13", "17:12:30"),   # m50V ends 17:12:30
            ],
            "m60V": [
                (584.8, "17:12:50", "17:13:13"),   # continuation
                (580.0, "17:13:18", "17:14:18"),
            ],
        },
    ),
]

# Pedestal sets differ across the two conditions -- condition 1 was decoded
# against 2026-08-01 21:12, condition 2 against 2026-08-02 15:04. Absolute ADC
# is therefore NOT comparable between them; only the trend within each is.
PEDESTAL_SET = {
    "rot15_ArCF4iso_88-10-2__run61_1214-1400": "pedestals_08-01-26_21-11-02",
    "rot25_ArCF4iso_88-10-2__run61_1606on": "pedestals_08-02-26_15-02-43",
}


def parse_t(s, ref):
    """"HH:MM:SS" -> datetime on the same day as `ref` (a sub-run start)."""
    h, m, sec = s.split(":")
    return dt.datetime(ref.year, ref.month, ref.day, int(h), int(m), int(sec))


def outdir(cond):
    return BASE + cond["key"] + "/"


# Which cache to use for gain work. Every cache carries `h_amp`, but only some
# carry `h_sat` -- m10V was paired before the extractor learned to store the
# saturation flag, so it has a separate `_gain` re-extraction; the later
# sub-runs got it in their only pass. Check with
# `set(np.load(p).files) >= {"h_amp", "h_sat"}` before adding a new one.
GAIN_NPZ = {
    "m10V": "pair_m10V_gain.npz",
    "m40V": "pair_m40V.npz",
    "m50V": "pair_m50V.npz",
    "m60V": "pair_m60V.npz",
}


def sources(cond, gain=False):
    """[(npz path, sub-run start, [(V, t0, t1), ...])] for this condition.

    `gain=True` picks the cache variant that also carries the saturation flag.
    """
    return [(PAIRED + (GAIN_NPZ[sr] if gain else "pair_%s.npz" % sr),
             SUBRUN_T0[sr], pts)
            for sr, pts in cond["points"].items()]
