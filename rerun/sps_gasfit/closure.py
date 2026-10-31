#!/usr/bin/env python3
"""closure.py [--out results/closure.json] -- does gasfit's grid fit return the composition it was given?

Toy stacks from the same model (predict.current_field (x) the run71 shaper of cloud_basics beam_comp_fit, normalised
like the data), Gaussian noise at the data's typical error, at the CF4 ladder fields (one group) and at a single
243 V/cm field (one stack: the run63 flat700 / run59 situation).  Truth on and off the grid nodes.  Also: the
water-gap degeneracy (toy made with a 30 mm gap, fitted with 28.8 mm).
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gasfit as F                       # noqa: E402

PAR = np.array([53.28221060991366, 17.178563090580035, 0.07642548168786019, 5223.267139626077])


def toy(Mo, comp, E, err, rng):
    I = Mo.current(comp, E)
    s, off = F.P.shaped(I, PAR)
    c = np.interp(F.T, F.P.FINE + off - 110.0, s)   # t0 of the run71 fit with this shaper (cloud_basics)
    nm = (F.T >= F.NORM[0]) & (F.T <= F.NORM[1])
    y = c / c[nm].mean()
    return dict(t=F.T, y=y + rng.normal(0, err, len(y)), e=np.full(len(y), err), E=E, zsd=None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'closure.json'))
    a = ap.parse_args(); rng = np.random.default_rng(1)
    out = []
    for gap_true, gap_fit in ((28.8, 28.8), (30.0, 28.8)):
        Mt = F.Model('beam', 'air_merged', gap_true, 0.0); Mf = F.Model('beam', 'air_merged', gap_fit, 0.0)
        W = np.linspace(Mf.G.W.min(), Mf.G.W.max(), 13); A = np.linspace(0, Mf.G.A.max(), 14)
        for comp in ((1.55, 0.07), (1.47, 0.11)):
            for name, Es in (('ladder', [243.2, 208.5, 173.8, 139.0, 104.3]), ('single243', [243.2]), ('raw3', [243.2, 156.3, 95.5])):
                # Es are fields at 28.8 mm; the drift voltage is what was set, the true field is V / gap_true
                data = []
                for E in Es:
                    V = E * 2.88
                    d = toy(Mt, comp, V / (gap_true / 10.0), 0.01, rng); d['V'] = V; data.append(d)
                chi = F.grid_scan(Mf, data, PAR, W, A)
                ndf = sum(int(((d['t'] >= F.WIN[0]) & (d['t'] <= F.WIN[1])).sum()) - 2 for d in data) - 2
                S = F.summarise(chi, W, A, ndf)
                r = dict(gap_true=gap_true, gap_fit=gap_fit, truth=comp, set=name, water=S['water'], air=S['air'],
                         water_68=S['water_68'], air_68=S['air_68'], chi2=S['chi2'], ndf=S['ndf'])
                out.append(r); print(r, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
