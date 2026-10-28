"""T3: measure_templates' algorithm on the v4c simulation (MX17_Geant, det3 response: wet gas, sigma0 420 um, bG 0.48,
reconstructed ions f_ion 0.90, DREAM 166 ns beta 0.75), per known angle, next to the data's |tan| bins (t1_det3.json).
Same cuts (2 brightest strips, 500 < amp < 3550, peak not at the window edge), same t50 alignment.
Input: /media/dylan/data/claude_tmp/v3wet/v4c_th{00,x05,..,x30}/ (laptop).  Usage: python -I t3_sim_template.py"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from t1_tan_amp import shape_stats, t50, GRID, SAT, HERE
from wft.io import FeuReader

SIM = '/media/dylan/data/claude_tmp/v3wet'
ANG = {'th00': 0, 'thx05': 5, 'thx10': 10, 'thx15': 15, 'thx20': 20, 'thx30': 30}


def sim_pulses(path, nmax=3000):
    r = FeuReader(path); out = []
    for eid, ftst, W in r.iter_events():
        W = np.asarray(W, np.float32); ns = W.shape[1]
        for i in np.argsort(W.max(axis=1))[::-1][:2]:
            w = W[i]; a = float(w.max()); ipk = int(np.argmax(w))
            if a < 500 or a > SAT or ipk < 6 or ipk > ns - 12:
                continue
            c = t50(w)
            if np.isfinite(c):
                out.append((a, np.interp(GRID, (np.arange(ns) - c) * 60.0, w / a, left=np.nan, right=np.nan)))
        if len(out) > 2 * nmax:
            break
    return out


if __name__ == '__main__':
    res = {}
    for tag, deg in ANG.items():
        for f in sorted(glob.glob(f'{SIM}/v4c_{tag}/sim_decoded_*_0[78].root')):
            feu = f.rsplit('_', 1)[1][:2]
            P = sim_pulses(f)
            if len(P) < 25:
                print(tag, feu, 'too few', len(P)); continue
            t, st = shape_stats([p[1] for p in P])
            st['tan'] = float(np.tan(np.radians(deg))); res[f'{tag}_feu{feu}'] = dict(st, shape=t.tolist())
            print(f"{tag:6s} feu{feu} tan {st['tan']:.3f} n={st['n']:5d} rise {st['rise']:4.0f} FWHM {st['fwhm']:4.0f} "
                  f"peak {st['peak']:4.0f} under {st['under']:+.3f} t800 {st['tail800']:+.3f}")
    json.dump(res, open(f'{HERE}/results/t3_sim_v4c.json', 'w'))
