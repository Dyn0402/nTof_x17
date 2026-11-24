"""r4_05_inject: does a window cut at ~1700 ns turn a golden shape into the observed 500 V shape? Scale the q_u90 of a
golden-shape set up to the 500 V level and cap it (hard cap, and a pessimistic soft cap that starts at 1300 ns)."""
import sys; sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import r4_lib as L, numpy as np, json
rng = np.random.default_rng(5)
out = {}
for src, N in ((L.GOLD1, None), (L.SC + '/drift_scan_resist_490V_drift_700V', None)):
    d0 = L.load_events(src)
    k = 1372.0 / d0.q.mean()
    for amp in (1.0, 1.33):          # 1.33 = (1+beta_500)/(1+beta_1000) if the gap shape stays the same, beta 1.28 against 0.68
        for cap in ('none', 'hard1700', 'soft1300'):
            d = d0.copy()
            lq = np.log(d.q.values)
            lq = np.mean(lq) + amp * (lq - np.mean(lq)) if False else lq  # keep noise as is
            # amplify only the shape: add (amp-1) * (fitted shape part) 
            A = L.design(d0, ('walk', 'tan')); c, _ = L.rfit(A, d0.lq.values)
            shape = A[:, 3:6] @ c[3:6]
            lq = d0.lq.values + (amp - 1) * shape + np.log(k)
            q = np.exp(lq)
            if cap == 'hard1700': q = np.minimum(q, 1699.)
            if cap == 'soft1300': q = q / (1 + (q / 1750.) ** 8) ** (1 / 8) if False else np.where(q < 1300, q, 1300 + 400 * np.tanh((q - 1300) / 400))
            d['q'] = q; d['lq'] = np.log(q)
            f = L.fit_shape(d, nboot=60)
            key = f'{src.split("/")[-1][-12:]} amp{amp} {cap}'
            out[key] = [f['c'][3:6].tolist(), f['err'][3:6].tolist(), float((q > 1650).mean())]
            print(f'{key:38s} guu {f["c"][3]:+.3f} gww {f["c"][4]:+.3f}+-{f["err"][4]:.3f} guw {f["c"][5]:+.3f}  frac>1650 {(q>1650).mean():.3f}', flush=True)
json.dump(out, open(f'{L.OUT}/inject.json', 'w'))
