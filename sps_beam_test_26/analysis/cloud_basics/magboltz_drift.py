#!/usr/bin/env python3
"""magboltz_drift.py -- drift velocity and diffusion (D_T, D_L) for the bench and
beam gases, dry and with water, over the drift-field range.

One mixture per process; results to results/magboltz_<tag>.json.
    source garfield_sim/setup_garfield.sh; python3 magboltz_drift.py <tag>
"""
import ctypes, json, os, sys

MIX = {
    # bench: Ar/iC4H10 95/5 at Saclay pressure, water replacing argon
    'bench_dry':   (745.83, [('ar', 95.0), ('ic4h10', 5.0)]),
    'bench_w0p25': (745.83, [('ar', 94.75), ('ic4h10', 5.0), ('h2o', 0.25)]),
    'bench_w0p5':  (745.83, [('ar', 94.5), ('ic4h10', 5.0), ('h2o', 0.5)]),
    'bench_w1':    (745.83, [('ar', 94.0), ('ic4h10', 5.0), ('h2o', 1.0)]),
    'bench_w2':    (745.83, [('ar', 93.0), ('ic4h10', 5.0), ('h2o', 2.0)]),
    # beam: Ar/CF4/iC4H10 88/10/2 at CERN pressure
    'beam_dry':    (720.8, [('ar', 88.0), ('cf4', 10.0), ('ic4h10', 2.0)]),
    'beam_w0p5':   (720.8, [('ar', 87.5), ('cf4', 10.0), ('ic4h10', 2.0), ('h2o', 0.5)]),
    'beam_w1p7':   (720.8, [('ar', 86.3), ('cf4', 10.0), ('ic4h10', 2.0), ('h2o', 1.7)]),
    'beam_w3':     (720.8, [('ar', 85.0), ('cf4', 10.0), ('ic4h10', 2.0), ('h2o', 3.0)]),
    # oxygen (air ingress) on top of the run_71 water: attachment test
    'beam_w1p7_o0p05': (720.8, [('ar', 86.25), ('cf4', 10.0), ('ic4h10', 2.0), ('h2o', 1.7), ('o2', 0.05)]),
    'beam_w1p7_o0p1':  (720.8, [('ar', 86.2), ('cf4', 10.0), ('ic4h10', 2.0), ('h2o', 1.7), ('o2', 0.1)]),
    'beam_w1p7_o0p2':  (720.8, [('ar', 86.1), ('cf4', 10.0), ('ic4h10', 2.0), ('h2o', 1.7), ('o2', 0.2)]),
    'bench_w0p5_o0p05': (745.83, [('ar', 94.45), ('ic4h10', 5.0), ('h2o', 0.5), ('o2', 0.05)]),
}
FIELDS = (60, 92, 100, 150, 200, 243, 250, 300, 400)
# amplification region: ~450-560 V over the 150 um gap
AMP_FIELDS = (25000, 30000, 35000, 40000)


def main(tag, amp=False):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    import Garfield  # noqa
    p, comp = MIX[tag]
    g = ROOT.Garfield.MediumMagboltz()
    args = []
    for n, f in comp:
        args += [n, f]
    g.SetComposition(*args)
    g.SetTemperature(293.15)
    g.SetPressure(p)
    pts = []
    for E in (AMP_FIELDS if amp else FIELDS):
        g.SetFieldGrid(E, E, 1, False)
        g.GenerateGasTable(5)
        vx, vy, vz = (ctypes.c_double() for _ in range(3))
        g.ElectronVelocity(0, 0, -E, 0, 0, 0, vx, vy, vz)
        dl, dt = ctypes.c_double(), ctypes.c_double()
        g.ElectronDiffusion(0, 0, -E, 0, 0, 0, dl, dt)
        eta = ctypes.c_double()
        g.ElectronAttachment(0, 0, -E, 0, 0, 0, eta)
        # v_um_ns kept in the original (Garfield cm/ns x 1e3 = 10 um/ns) units for
        # continuity with the first batch; v_true_um_ns is the real value
        pts.append(dict(E_Vcm=E, v_um_ns=abs(vz.value) * 1e3, v_true_um_ns=abs(vz.value) * 1e4,
                        DT_um_rtcm=dt.value * 1e4, DL_um_rtcm=dl.value * 1e4,
                        eta_per_cm=eta.value,
                        lambda_mm=(10.0 / eta.value) if eta.value > 0 else None))
        print(tag, pts[-1], flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                       f'magboltz_{tag}{"_amp" if amp else ""}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(tag=tag, pressure_torr=p, comp=comp, points=pts), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], amp=len(sys.argv) > 2 and sys.argv[2] == 'amp')
