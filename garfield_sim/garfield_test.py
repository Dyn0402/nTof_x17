"""
garfield_test.py
================
Incremental test script for Garfield++ Python (PyROOT) interface.
Run with:
    python3 garfield_test.py

Prerequisites:
    source /path/to/root/bin/thisroot.sh
    source $GARFIELD_HOME/install/share/Garfield/setupGarfield.sh

The script runs 5 tests of increasing complexity:
  1. Import ROOT and Garfield library
  2. Instantiate MediumMagboltz and set a gas composition
  3. Run a short Magboltz gas table calculation (~30 s for Ar/CO2 at 1 atm)
  4. Set up a uniform-field geometry with ComponentConstant + Sensor
  5. Run a single-electron avalanche with AvalancheMicroscopic and print gain

Tests 1-4 are quick. Test 5 takes a few seconds per electron (microscopic tracking).
If tests 1-4 pass but test 5 hangs, your gas table E-field range probably doesn't
cover the amplification field — see the note in test 3.
"""

import sys
import time

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────
# TEST 1: Imports
# ──────────────────────────────────────────────────────────────
section("TEST 1: Import ROOT and Garfield")

try:
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)   # suppress graphics pop-ups
    print(f"{PASS} import ROOT  (version {ROOT.__version__})")
except Exception as e:
    print(f"{FAIL} import ROOT failed: {e}")
    sys.exit(1)

try:
    import Garfield
    print(f"{PASS} import Garfield")
except Exception as e:
    print(f"{FAIL} import Garfield failed: {e}")
    print("      → Did you source setupGarfield.sh before running?")
    sys.exit(1)

# Also confirm the namespace is accessible
try:
    _ = ROOT.Garfield.MediumMagboltz
    print(f"{PASS} ROOT.Garfield namespace accessible")
except Exception as e:
    print(f"{FAIL} ROOT.Garfield namespace not found: {e}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# TEST 2: Instantiate MediumMagboltz, set gas
# ──────────────────────────────────────────────────────────────
section("TEST 2: MediumMagboltz — set gas composition")

try:
    gas = ROOT.Garfield.MediumMagboltz()
    # Ar/CO2 70/30 — a very standard well-understood mixture
    gas.SetComposition("ar", 70., "co2", 30.)
    gas.SetTemperature(293.15)   # K (room temperature)
    gas.SetPressure(760.)        # Torr (1 atm)
    print(f"{PASS} MediumMagboltz created")
    print(f"      Gas: {gas.GetName()}")
    print(f"      Temperature: {gas.GetTemperature():.1f} K")
    print(f"      Pressure:    {gas.GetPressure():.1f} Torr")
except Exception as e:
    print(f"{FAIL} MediumMagboltz setup failed: {e}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# TEST 3: Run Magboltz (short gas table)
# ──────────────────────────────────────────────────────────────
section("TEST 3: Magboltz gas table calculation (~30-120 s)")

print(f"{INFO} Setting E-field grid: 1 kV/cm – 200 kV/cm, 20 points (log)")
print(f"{INFO} nColl = 5 (x10^7 collisions) — low precision, fast for testing")
print(f"      (For production use nColl >= 10)")
print(f"      Running Magboltz, please wait...\n")

try:
    # E range in V/cm. The upper end (200 kV/cm) covers typical Micromegas
    # amplification fields. For your Ne/Ar/He mixtures you can extend this.
    gas.SetFieldGrid(1.e3, 2.e5, 20, True)   # Emin, Emax, nPoints, logSpacing
    t0 = time.time()
    gas.GenerateGasTable(5)     # nColl x 10^7 collisions
    elapsed = time.time() - t0
    print(f"\n{PASS} Magboltz finished in {elapsed:.1f} s")

    # Read back the Townsend coefficient at a representative field
    # ElectronTownsend returns alpha in cm^-1 via output argument (PyROOT style)
    # We check it's non-zero at a high field (should clearly be in gain regime)
    E_test = 50000.   # 50 kV/cm — well into avalanche territory for Ar/CO2
    alpha = ROOT.Double(0.)
    eta   = ROOT.Double(0.)
    gas.ElectronTownsend(E_test, 0., 0., 0., 0., alpha)
    gas.ElectronAttachment(E_test, 0., 0., 0., 0., eta)
    print(f"      At E = {E_test/1e3:.0f} kV/cm:")
    print(f"        Townsend α = {float(alpha):.1f} cm⁻¹")
    print(f"        Attachment η = {float(eta):.4f} cm⁻¹")
    if float(alpha) > 0:
        print(f"{PASS} Non-zero Townsend coefficient — gas table looks good")
    else:
        print(f"{FAIL} Townsend coefficient is zero — something went wrong with Magboltz")
        print(f"      Check that gfortran is installed and HEED_DATABASE is set.")
except Exception as e:
    print(f"\n{FAIL} Magboltz failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# TEST 4: ComponentConstant + Sensor (uniform field geometry)
# ──────────────────────────────────────────────────────────────
section("TEST 4: ComponentConstant + Sensor")

# Micromegas amplification gap geometry:
#   z = 0      → anode (readout strip)
#   z = gap    → mesh
# Electrons drift from mesh toward anode (in -z direction),
# so the electric field points in +z (Ex=0, Ey=0, Ez > 0 in Garfield convention
# where field points from anode to mesh, and electrons drift opposite to E).
# 
# Actually in Garfield++ the drift direction is determined by the sign of the
# field relative to the electron charge, so we just set a field in +z and place
# the electron at z = gap, drifting toward z = 0.

GAP   = 0.0128    # cm = 128 µm — typical Micromegas amplification gap
E_AMP = 50000.    # V/cm amplification field (50 kV/cm)

try:
    cmp = ROOT.Garfield.ComponentConstant()

    # Set the active volume (bounding box)
    cmp.SetArea(-1., -1., 0.,   # xmin, ymin, zmin
                 1.,  1., GAP)  # xmax, ymax, zmax

    # Uniform field in +z direction (electrons drift toward z=0, the anode)
    cmp.SetElectricField(0., 0., E_AMP)

    # Attach the gas medium
    cmp.SetMedium(gas)

    print(f"{PASS} ComponentConstant created")
    print(f"      Gap: {GAP*1e4:.0f} µm, E = {E_AMP/1e3:.0f} kV/cm")

    # Build sensor
    sensor = ROOT.Garfield.Sensor()
    sensor.AddComponent(cmp)
    sensor.SetArea(-1., -1., 0., 1., 1., GAP)

    print(f"{PASS} Sensor created and configured")

    # Quick sanity check: ask the sensor for the field at the midpoint
    ex = ROOT.Double(0.); ey = ROOT.Double(0.); ez = ROOT.Double(0.)
    mid = GAP / 2.
    status = ROOT.Long(0)
    sensor.ElectricField(0., 0., mid, ex, ey, ez, status)
    print(f"      Field at gap midpoint: Ez = {float(ez):.0f} V/cm (status={int(status)})")
    if abs(float(ez) - E_AMP) < 1.0:
        print(f"{PASS} Field value correct")
    else:
        print(f"{FAIL} Field value unexpected — check ComponentConstant setup")

except Exception as e:
    print(f"{FAIL} Component/Sensor setup failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# TEST 5: AvalancheMicroscopic — single electron, compute gain
# ──────────────────────────────────────────────────────────────
section("TEST 5: AvalancheMicroscopic — single electron avalanche")

print(f"{INFO} Drifting a single electron through the {GAP*1e4:.0f} µm gap")
print(f"      at {E_AMP/1e3:.0f} kV/cm. Expect gain >> 1 for Ar/CO2.")
print(f"      (This may take 5-30 s for one electron at high gain)\n")

try:
    aval = ROOT.Garfield.AvalancheMicroscopic()
    aval.SetSensor(sensor)

    # Seed electron at the mesh (top of gap), with 0 initial energy
    # x0, y0, z0, t0, e0 (eV)
    x0, y0, z0, t0, e0 = 0., 0., GAP, 0., 0.

    t0_wall = time.time()
    aval.AvalancheElectron(x0, y0, z0, t0, e0)
    elapsed = time.time() - t0_wall

    # GetAvalancheSize: ne = electrons, ni = ions at end
    ne = ROOT.Long(0)
    ni = ROOT.Long(0)
    aval.GetAvalancheSize(ne, ni)

    ne_val = int(ne)
    ni_val = int(ni)

    print(f"      Avalanche size: {ne_val} electrons, {ni_val} ions")
    print(f"      Wall time: {elapsed:.1f} s")

    if ne_val > 1:
        print(f"{PASS} Avalanche completed, gain = {ne_val} (single-event, statistical)")
    elif ne_val == 1:
        print(f"{INFO} Only 1 electron — possible that no ionisation occurred at this field.")
        print(f"      Try increasing E_AMP or check the gas table E-field coverage.")
    else:
        print(f"{FAIL} No electrons survived — check geometry and gas table.")

    # Also check electron endpoint — should be near z=0 (anode)
    npts = aval.GetNumberOfElectronEndpoints()
    if npts > 0:
        xe = ROOT.Double(0.); ye = ROOT.Double(0.); ze = ROOT.Double(0.)
        te = ROOT.Double(0.); ee = ROOT.Double(0.)
        xs = ROOT.Double(0.); ys = ROOT.Double(0.); zs = ROOT.Double(0.)
        ts = ROOT.Double(0.); es = ROOT.Double(0.)
        stat = ROOT.Long(0)
        aval.GetElectronEndpoint(0, xs, ys, zs, ts, es, xe, ye, ze, te, ee, stat)
        print(f"      First endpoint: z_end = {float(ze)*1e4:.1f} µm (should be ~0)")

except Exception as e:
    print(f"{FAIL} AvalancheMicroscopic failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
section("SUMMARY")
print("All 5 tests completed. If you saw [PASS] on each, Garfield++ is")
print("working correctly for the gain simulation use case.")
print()
print("Next steps:")
print("  - Save the gas table:  gas.WriteGasFile('ar_co2_70_30_1atm.gas')")
print("  - Load it later with:  gas.LoadGasFile('ar_co2_70_30_1atm.gas')")
print("  - Loop over E fields and pressures to build a gain curve")
print()
