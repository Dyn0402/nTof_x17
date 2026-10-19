#!/usr/bin/env python3
"""Concentration dependence of Garfield's built-in Penning parameterisations."""
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)
import Garfield, re, io, sys

def probe(comps):
    gas = ROOT.Garfield.MediumMagboltz()
    flat = [x for pair in comps for x in pair]
    gas.SetComposition(*flat)
    gas.SetTemperature(293.15)
    gas.SetPressure(720.8)
    ok = gas.EnablePenningTransfer()
    return ok

for q in [1., 2., 3., 5., 7., 10., 15., 20., 30.]:
    print(f"### Ar/CO2 {100-q}/{q}", flush=True)
    probe([("ar", 100 - q), ("co2", q)])
for q in [2., 5., 10., 20.]:
    print(f"### Ar/iC4H10 {100-q}/{q}", flush=True)
    probe([("ar", 100 - q), ("ic4h10", q)])
