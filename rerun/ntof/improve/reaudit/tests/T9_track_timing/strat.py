"""T9 task 4: stratified correlations of the scale-form residual r*sign(j) (|j| >= 0.1) with the main predictors."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prep import D, cosmic_long
from corr import rho, clipq

plt.rcParams.update({'font.size': 10.5})
MAIN = ['dT90', 'dTend', 'dt0pin', 'du90', 'dt0xy', 'nstr']
J = pd.read_parquet(D / 'joined.parquet', columns=['n_cand_x_a', 'n_cand_y_a', 'n_cand_x_c', 'n_cand_y_c'])


def main():
    L = cosmic_long()
    rows = []
    for (arm, p), F in L.items():
        F = F.copy()
        F['ncand'] = J[f'n_cand_{p}_{arm.lower()}'].to_numpy()
        G = F[F.clean & (F.j.abs() >= 0.1)].copy()
        y = (G.r * np.sign(G.j)).to_numpy()
        qq = np.nanquantile(G.lnq, [1 / 3, 2 / 3])
        strata = {'all': np.ones(len(G), bool),
                  'centre(<125)': (G.edge < 125).to_numpy(), 'edge(>=125)': (G.edge >= 125).to_numpy(),
                  'q low': (G.lnq < qq[0]).to_numpy(), 'q mid': ((G.lnq >= qq[0]) & (G.lnq < qq[1])).to_numpy(),
                  'q high': (G.lnq >= qq[1]).to_numpy(),
                  '|j| 0.1-0.25': (G.j.abs() < 0.25).to_numpy(), '|j| >= 0.25': (G.j.abs() >= 0.25).to_numpy(),
                  'ncand=1': (G.ncand == 1).to_numpy(), 'ncand>1': (G.ncand > 1).to_numpy(),
                  'chi2 < median': (G.chi2 < G.chi2.median()).to_numpy(),
                  'chi2 >= median': (G.chi2 >= G.chi2.median()).to_numpy()}
        for f in MAIN:
            x = clipq(G[f].to_numpy(float))
            for sn, m in strata.items():
                rr = rho(x[m], y[m])
                # also within fixed nstr (the footprint) for the timing predictors: nstr = 4..7 separately, averaged
                rows.append(dict(arm=arm, view=p, pred=f, stratum=sn, n=rr['n'], spear=rr['spear'],
                                 spear_e=rr['spear_e'], pear=rr['pear']))
            if f != 'nstr':
                vals = []
                for k in range(3, 12):
                    m = (G.nstr == k).to_numpy()
                    if m.sum() > 300:
                        vals.append((rho(x[m], y[m])['spear'], m.sum()))
                if vals:
                    v, w = np.array(vals).T
                    rows.append(dict(arm=arm, view=p, pred=f, stratum='fixed nstr (weighted)', n=int(w.sum()),
                                     spear=float(np.average(v, weights=w)), spear_e=1.03 / np.sqrt(w.sum())))
    T = pd.DataFrame(rows)
    T.to_csv(D / 'strata.csv', index=False)
    P = T.pivot_table(index=['pred', 'stratum'], columns=['arm', 'view'], values='spear').round(3)
    pd.set_option('display.width', 220, 'display.max_rows', 300)
    print(P.to_string())


if __name__ == '__main__':
    main()
