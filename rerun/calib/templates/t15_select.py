"""T15 selection test: choose the template stretch s per chamber and view by chi2 alone on one half of the held-out
events, then measure the angle change on the other half -- what a calibration node fitting s would deliver.
Halves by event id (cache order): choose on ids >= SPLIT, evaluate on ids < SPLIT, and the reverse.
Usage: python -I t15_select.py DIR  (DIR holds the t15 <det>_<arm>.npz)  -> results/t15/t15_select.json"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t5_compare import angerr, s68
HERE = os.path.dirname(os.path.abspath(__file__))
ST = {'base': 1.0, 'st076': 0.76, 'st080': 0.80, 'st084': 0.84, 'st088': 0.88, 'st092': 0.92, 'st096': 0.96, 'st104': 1.04}


def main(d):
    rng = np.random.default_rng(5); out = {}
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        arms = {a: np.load(f'{d}/{det}_{a}.npz')['per'] for a in ST if os.path.exists(f'{d}/{det}_{a}.npz')}
        for vi, v in enumerate('xy'):
            tabs = {a: {int(r[0]): r for r in p[p[:, 1] == vi]} for a, p in arms.items()}
            ids = sorted(set.intersection(*[set(t) for t in tabs.values()]))
            split = ids[len(ids) // 2]
            for name, (ch, ev) in {'A': (lambda i: i >= split, lambda i: i < split), 'B': (lambda i: i < split, lambda i: i >= split)}.items():
                ci = [i for i in ids if ch(i)]; ei = [i for i in ids if ev(i)]
                x2 = {a: float(np.median([t[i][4] / max(t[i][5], 1) for i in ci])) for a, t in tabs.items()}
                pick = min(x2, key=x2.get)
                RA = np.array([tabs[pick][i] for i in ei]); RB = np.array([tabs['base'][i] for i in ei])
                dA, _ = angerr(RA); dB, _ = angerr(RB)
                boots = [s68(dA[j]) - s68(dB[j]) for j in (rng.integers(0, len(ei), len(ei)) for _ in range(300))]
                out[f'{det}_{v}_{name}'] = dict(pick=pick, s=ST[pick], ds68=float(s68(dA) - s68(dB)), err=float(np.std(boots)),
                                               n=len(ei), chi2_choose={a: round(c, 2) for a, c in x2.items()})
                print(f'{det} {v} {name}: chi2 picks s={ST[pick]:.2f} -> held-out half ds68 {s68(dA) - s68(dB):+.3f} ± {np.std(boots):.3f} (n {len(ei)})')
    json.dump(out, open(f'{HERE}/results/t15/t15_select.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
