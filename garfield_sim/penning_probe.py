#!/usr/bin/env python3
"""Probe what EnablePenningTransfer() (auto) does for several mixtures."""
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)
import Garfield

CASES = [
    ("Ar/iC4H10 95/5",       [("ar", 95.), ("ic4h10", 5.)]),
    ("Ar/iC4H10 98/2",       [("ar", 98.), ("ic4h10", 2.)]),
    ("Ar/CO2/iC4H10 93/5/2", [("ar", 93.), ("co2", 5.), ("ic4h10", 2.)]),
    ("Ar/CO2 70/30",         [("ar", 70.), ("co2", 30.)]),
]

for name, comps in CASES:
    print("=" * 70)
    print("CASE:", name)
    gas = ROOT.Garfield.MediumMagboltz()
    flat = [x for pair in comps for x in pair]
    gas.SetComposition(*flat)
    gas.SetTemperature(293.15)
    gas.SetPressure(720.8)
    ok = gas.EnablePenningTransfer()
    print("  EnablePenningTransfer() ->", ok)
    gas.PrintGas()
