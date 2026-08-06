#!/usr/bin/env python3
"""
probe_penning.py — re-probe Garfield++'s built-in Penning transfer table.

    source setup_garfield.sh
    python3 probe_penning.py

The rP values and the "not implemented" behaviours recorded in mm_config.py and
README.md are properties of whichever Garfield++ we run, not of the gas. They
were originally probed against the LCG_108 Garfield (commit 6fb94b35). Re-run
this after any change to MX17_GARFIELD_PIN and reconcile the output with the
comments in mm_config.py — a silent change here would bias every gain
comparison, because auto mode simulates rP = 0 for a mixture Garfield does not
know, without failing.

Prints one line per mixture: whether EnablePenningTransfer() succeeded and, if
it did, the rP it selected.
"""

import ctypes
import sys

PRESSURE_TORR = 720.8   # CERN 450 m, as used for the original probe
TEMP_K = 293.0

# (label, [(gas, percentage), ...])
MIXTURES = [
    ("Ar/CO2 99/1",             [("ar", 99.), ("co2", 1.)]),
    ("Ar/CO2 97/3",             [("ar", 97.), ("co2", 3.)]),
    ("Ar/CO2 95/5",             [("ar", 95.), ("co2", 5.)]),
    ("Ar/CO2 93/7",             [("ar", 93.), ("co2", 7.)]),
    ("Ar/CO2 90/10",            [("ar", 90.), ("co2", 10.)]),
    ("Ar/CO2 85/15",            [("ar", 85.), ("co2", 15.)]),
    ("Ar/CO2 80/20",            [("ar", 80.), ("co2", 20.)]),
    ("Ar/CO2 70/30",            [("ar", 70.), ("co2", 30.)]),
    ("Ar/iC4H10 95/5",          [("ar", 95.), ("ic4h10", 5.)]),
    ("Ar/iC4H10 90/10",         [("ar", 90.), ("ic4h10", 10.)]),
    ("Ar/CO2/iC4H10 93/5/2",    [("ar", 93.), ("co2", 5.), ("ic4h10", 2.)]),
    ("Ar/CF4 90/10",            [("ar", 90.), ("cf4", 10.)]),
    ("Ne/CF4 90/10",            [("ne", 90.), ("cf4", 10.)]),
    ("Ne/C2H6 90/10",           [("ne", 90.), ("c2h6", 10.)]),
    ("Ne/CF4/C2H6 80/10/10",    [("ne", 80.), ("cf4", 10.), ("c2h6", 10.)]),
]


def main():
    import ROOT
    import Garfield  # noqa: F401  (loads the dictionary)

    print(f"Garfield++ probe at {PRESSURE_TORR} Torr, {TEMP_K} K\n")
    print(f"{'mixture':<26} {'EnablePenningTransfer':<22} rP")
    print("-" * 62)

    for label, comps in MIXTURES:
        gas = ROOT.Garfield.MediumMagboltz()
        args = []
        for name, frac in comps:
            args += [name, frac]
        gas.SetComposition(*args)
        gas.SetPressure(PRESSURE_TORR)
        gas.SetTemperature(TEMP_K)
        gas.Initialise(True)

        ok = gas.EnablePenningTransfer()

        # GetPenningTransfer(i, r, lambda) reports the value actually stored on
        # excitation level i; the double& outputs come back through ctypes.
        rp_txt = "-"
        if ok:
            r, lam = ctypes.c_double(0.), ctypes.c_double(0.)
            if gas.GetPenningTransfer(0, r, lam):
                rp_txt = f"{r.value:.3f}"
            else:
                rp_txt = "(set)"
        print(f"{label:<26} {str(bool(ok)):<22} {rp_txt}", flush=True)

    print("\nReconcile against the tables in mm_config.py and README.md.")


if __name__ == "__main__":
    sys.exit(main())
