import ROOT, ctypes, math
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True); ROOT.gErrorIgnoreLevel = ROOT.kError
import Garfield

k=1.380649e-23; T=293.15
for lab,P in (("Ar_CO2_70_30_hiE",720.81),("Ne_CF4_C2H6_80_10_10_hiE",720.81)):
    g=ROOT.Garfield.MediumMagboltz(); g.LoadGasFile(f"gas_tables/{lab}_CERN_450m.gas")
    g.SetTemperature(T); g.SetPressure(P)
    N=P*133.322/(k*T)*1e-6
    print(f"\n=== {lab} (50 um gap) ===")
    print(f"{'V':>5} {'E kV/cm':>9} {'E/N Td':>8} {'alpha':>9} {'eta':>7} {'(a-e)*d':>9}")
    prev=None
    for V in (200,250,300,350,400,450,500,550,600,700,800):
        E=V/0.0050
        a=ctypes.c_double(0.); e=ctypes.c_double(0.)
        g.ElectronTownsend(0.,0.,E,0.,0.,0.,a); g.ElectronAttachment(0.,0.,E,0.,0.,0.,e)
        ad=(a.value-e.value)*0.0050
        flag=""
        if prev is not None and a.value < prev: flag="  <-- NON-MONOTONIC"
        prev=a.value
        print(f"{V:5d} {E/1e3:9.1f} {E/N/1e-17:8.1f} {a.value:9.1f} {e.value:7.2f} {ad:9.2f}{flag}")
