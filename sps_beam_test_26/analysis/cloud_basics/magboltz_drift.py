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
FIELDS = (60, 75, 92, 100, 108, 142, 150, 200, 243, 250, 300, 400)

# air ingress grid: '<base>_w<water %>_a<air %>' with 'p' for the decimal point,
# e.g. beam_w1p7_a0p05.  Air is N2/O2/Ar 78.08/20.95/0.93; water and air both
# replace argon.  Magboltz takes at most six gases: base (3) + h2o + n2 + o2.
BASES = {
    'beam':  (720.8, [('ar', 88.0), ('cf4', 10.0), ('ic4h10', 2.0)]),     # Ar/CF4/iso, CERN
    'co2':   (720.8, [('ar', 95.0), ('co2', 3.0), ('ic4h10', 2.0)]),      # Ar/CO2/iso, CERN
    'bench': (745.83, [('ar', 95.0), ('ic4h10', 5.0)]),                   # Ar/iso, Saclay
}
AIR = (('n2', 78.08), ('o2', 20.95), ('ar', 0.93))


def air_mix(tag):
    base, w, a = tag.split('_')
    w = float(w[1:].replace('p', '.')); a = float(a[1:].replace('p', '.'))
    p, comp = BASES[base]
    frac = dict(comp)
    frac['ar'] -= w + a
    for g, f in AIR:
        frac[g] = frac.get(g, 0.0) + a * f / 100.0
    if w > 0:
        frac['h2o'] = w
    return p, [(g, f) for g, f in frac.items() if f > 0]
# amplification region: ~450-560 V over the 150 um gap
AMP_FIELDS = (25000, 30000, 35000, 40000)


def main(tag, amp=False, fields=None, ncoll=5):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    import Garfield  # noqa
    p, comp = MIX[tag] if tag in MIX else air_mix(tag)
    g = ROOT.Garfield.MediumMagboltz()
    args = []
    for n, f in comp:
        args += [n, f]
    g.SetComposition(*args)
    g.SetTemperature(293.15)
    g.SetPressure(p)
    pts = []
    for E in (fields or (AMP_FIELDS if amp else FIELDS)):
        g.SetFieldGrid(E, E, 1, False)
        g.GenerateGasTable(ncoll)
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
                       f'magboltz_{tag}{"_amp" if amp else ""}'
                       f'{f"_E{fields[0]:g}_c{ncoll}" if fields and len(fields) == 1 else ""}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(tag=tag, pressure_torr=p, comp=comp, ncoll=ncoll, points=pts), open(out, 'w'), indent=1)


if __name__ == '__main__':
    # magboltz_drift.py <tag> [amp]                      -- the FIELDS list, 5e7 collisions
    # magboltz_drift.py <tag> E=<field> [c=<ncoll>]      -- one field, high statistics (one condor job)
    kw = dict(a.split('=', 1) for a in sys.argv[2:] if '=' in a)
    main(sys.argv[1], amp='amp' in sys.argv[2:],
         fields=[float(kw['E'])] if 'E' in kw else None, ncoll=int(kw.get('c', 5)))
