"""T9 task 2: per-track correction from the timing predictors, held out by run parity; frozen on the beam."""
import json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '3')
sys.path.insert(0, '/tmp/t9pkgs')
sys.path.insert(1, '/home/dylan/PycharmProjects/nTof_x17')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prep import D, PRED, cosmic_long, beam_long, miss_mm
from rerun.ntof.improve.track_quality import pointing

plt.rcParams.update({'font.size': 10.5})
TIMING = ['dTend', 'dT90', 'dT50', 'du50', 'du90', 'duend', 'dt0xy', 'dtc', 'dt0pin', 'dt0pin_o']
SETS = {'timing': TIMING, 'all': PRED}


def widths(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    m = np.median(r)
    a = np.abs(r - m)
    return dict(n=len(r), mad=float(1.4826 * np.median(a)), w68=float(np.quantile(a, 0.68)),
                w95=float(np.quantile(a, 0.95)), rms_clip=float(np.std(r[np.abs(r) < 0.15])))


class Lin:
    """r = b0 + sum_k beta_k z_k + sum_k gamma_k z_k s  (z = standardised, clipped predictor; s = measured slope).
    No bare s term (that would learn the shrinkage of the cosmic angle prior).  Fit: least squares, 3 sigma clip x3."""

    def __init__(self, feats):
        self.f = feats

    def design(self, F):
        Z = np.column_stack([np.clip((F[f].to_numpy(float) - self.mu[f]) / self.sd[f], -4, 4) for f in self.f])
        Z = np.nan_to_num(Z)
        s = F.s.to_numpy(float)[:, None]
        return np.c_[np.ones(len(F)), Z, Z * s]

    def fit(self, F):
        self.mu = {f: float(np.nanmedian(F[f])) for f in self.f}
        self.sd = {f: float(1.4826 * np.nanmedian(np.abs(F[f] - self.mu[f])) or np.nanstd(F[f]) or 1.0) for f in self.f}
        X, y = self.design(F), F.r.to_numpy(float)
        keep = np.ones(len(y), bool)
        for _ in range(3):
            self.b = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
            res = y - X @ self.b
            sig = 1.4826 * np.median(np.abs(res[keep]))
            keep = np.abs(res) < 3 * sig
        return self

    def predict(self, F):
        return self.design(F) @ self.b

    def to_json(self):
        k = len(self.f)
        return dict(features=self.f, centre=self.mu, scale=self.sd, b0=float(self.b[0]),
                    beta=dict(zip(self.f, map(float, self.b[1:1 + k]))),
                    gamma_times_slope=dict(zip(self.f, map(float, self.b[1 + k:]))),
                    use='tan_corr = s - (b0 + sum beta_k z_k + sum gamma_k z_k s), z_k = clip((f_k - centre)/scale, -4, 4)')


def gbt(F, feats):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(loss='absolute_error', learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                      min_samples_leaf=100, early_stopping=True, validation_fraction=0.2,
                                      random_state=1)
    return m.fit(F[feats].to_numpy(float), F.r.to_numpy(float))


def main():
    L = cosmic_long()
    rows, coef = [], {}
    for (arm, p), F in L.items():
        par = F.run.to_numpy() % 2
        for tr in (0, 1):
            Ftr = F[(par == tr) & F.clean.to_numpy()]
            te = par != tr
            for sample in ('clean', 'loose'):
                Fte = F[te & F[sample].to_numpy()]
                base = widths(Fte.r)
                rows.append(dict(arm=arm, view=p, train=tr, sample=sample, model='none', **base))
                for name, fs in SETS.items():
                    m = Lin(fs).fit(Ftr)
                    rc = Fte.r - m.predict(Fte)
                    rows.append(dict(arm=arm, view=p, train=tr, sample=sample, model=f'lin_{name}', **widths(rc)))
                for name, fs in (('gbt_slope_only', ['s']), ('gbt_slope_pos', ['s', 'x_local', 'y_local']),
                                 ('gbt_timing', TIMING + ['s']), ('gbt_all', PRED + ['s', 'x_local', 'y_local'])):
                    key = (arm, p, tr, name)
                    if key not in coef:
                        coef[key] = gbt(Ftr, fs)
                    rc = Fte.r - coef[key].predict(Fte[fs].to_numpy(float))
                    rows.append(dict(arm=arm, view=p, train=tr, sample=sample, model=name, **widths(rc)))
    T = pd.DataFrame(rows)
    T.to_csv(D / 'heldout_cosmic.csv', index=False)
    S = T.groupby(['arm', 'view', 'sample', 'model'])[['n', 'mad', 'w68', 'w95']].mean().reset_index()
    pd.set_option('display.width', 220, 'display.max_rows', 300)
    print(S.round(5).to_string())

    # frozen coefficients from ALL clean cosmic lines, per chamber and view
    frozen = {}
    for (arm, p), F in L.items():
        for name, fs in SETS.items():
            frozen[f'{arm}_{p}_{name}'] = Lin(fs).fit(F[F.clean]).to_json()
    json.dump(frozen, open(D / 'coefficients.json', 'w'), indent=1)

    # beam: x view, frozen cosmic coefficients
    Bl = beam_long()
    rng = np.random.default_rng(7)
    rb = []
    for arm, F in Bl.items():
        for name, fs in [('none', None)] + list(SETS.items()):
            if fs is None:
                s = F.s.to_numpy()
            else:
                m = Lin(fs)
                j = frozen[f'{arm}_x_{name}']
                m.mu, m.sd = j['centre'], j['scale']
                m.b = np.r_[j['b0'], [j['beta'][f] for f in fs], [j['gamma_times_slope'][f] for f in fs]]
                s = F.s.to_numpy() - m.predict(F)
            for sel_name, sel in (('all', np.ones(len(F), bool)), ('abs_s>=0.1', np.abs(F.s.to_numpy()) >= 0.1),
                                  ('abs_s>=0.25', np.abs(F.s.to_numpy()) >= 0.25)):
                mi = miss_mm(F[sel], s[sel])
                # null: shuffled displacements (track_quality convention)
                disp = (-(F.p0_z.to_numpy() + 4.5) * s)[sel]
                pos = F.p0_x.to_numpy()[sel]
                mi0 = pos + disp[rng.permutation(len(disp))] + 10.9
                pt = pointing(mi, mi0)
                r = (s - F.s_exp.to_numpy())[sel]
                core = widths(r[np.abs(r) < 0.15])
                rb.append(dict(arm=arm, model=name, sel=sel_name, n=pt['n'], f15=pt['f1'], exc15=pt['exc1'],
                               exc45=pt['exc2'], w68_mm=pt['w68'], tail=pt['tail'], tan_core_mad=core['mad'],
                               tan_core_w68=core['w68']))
    TB = pd.DataFrame(rb)
    TB.to_csv(D / 'beam_frozen.csv', index=False)
    print(TB.round(4).to_string())

    # figure: widths before / after, cosmics (clean held out) and beam
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    Sc = S[S['sample'] == 'clean']
    mods = ['none', 'lin_timing', 'lin_all', 'gbt_slope_only', 'gbt_timing', 'gbt_all']
    labs = [f'{a} {v}' for a in 'AC' for v in 'xy']
    for i, mo in enumerate(mods):
        y = [Sc[(Sc.arm == a) & (Sc.view == v) & (Sc.model == mo)].mad.iloc[0] * 1e3 for a in 'AC' for v in 'xy']
        axs[0].bar(np.arange(4) + (i - 2.5) * 0.13, y, 0.13, label=mo)
    axs[0].set_xticks(range(4), labs)
    axs[0].set_ylabel('held-out MAD sigma of r (x1000)')
    axs[0].set_ylim(15, 32)
    axs[0].legend(fontsize=7.5)
    axs[0].set_title('cosmic A-C lines, clean, other run parity')
    Tb = TB[TB.sel == 'abs_s>=0.1']
    for i, mo in enumerate(['none', 'timing', 'all']):
        axs[1].bar(np.arange(2) + (i - 1) * 0.25, [Tb[(Tb.arm == a) & (Tb.model == mo)].exc15.iloc[0] for a in 'AC'],
                   0.25, label=mo)
    axs[1].set_xticks(range(2), ['A', 'C'])
    axs[1].set_ylabel('capsule-pointing excess within 15 mm')
    axs[1].set_title('late beam, x view, |s| >= 0.1, frozen cosmic fit')
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(D / 'correction_gain.png', dpi=120)


if __name__ == '__main__':
    main()
