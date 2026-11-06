#!/usr/bin/env python3
"""h2c.py -- H2c: the forward model fitted to a hit list (HITS_DAQ_PLAN §4).

A hit-only readout keeps, per strip, one (t, a) from family V (emulate.py).  H2c uses every piece of physics the
waveform fit W uses -- the pass kernel (build_matrix_rc: footprint, diffusion, RC spreading, the electronics
template) -- but only the information V keeps: for a trial track (p0, w, t0) and charge scale Q the model window is
generated, the SAME emulator V is run on it, and the predicted hit list is compared with the observed one:

  cost = sum over strips hit in both   ((t_obs - t_pred) / s_t)^2 + ((a_obs - a_pred) / (f_a a_obs + 3 noise))^2
       + P_miss x (strips hit in only one of the two)

The depth profile q (one charge per 60 ns drift bin):
  qmode 'uniform'  a uniform column from the mesh to the cathode (q_k = Q inside gap / v): biased, because real
                   profiles are not uniform (closure on model windows: w 0.94 X / 0.92 Y; with the true q: 1.000)
  qmode 'amp'      solved from the hit amplitudes at every trial track: with the track fixed, each strip's peak height
                   is linear in q (the design matrix row at the strip's model peak sample), so q = NNLS of the observed
                   amplitudes with a second-difference smoothness penalty (a hit list has ~one amplitude per depth bin).
                   Two passes: the peak samples of the first solution define the rows of the second.
  qmode 'mean'     a fixed depth shape times Q: the mean normalised profile, a calibration constant per view and run
                   condition (h2c_run.py --qcal: the waveform fit's q on the even event ids; only odd ids are scored).
  qmode 'ampmean'  as 'amp', but the regulariser pulls q toward c x the mean shape (c a free NNLS unknown) instead of
                   toward zero curvature: the event's own amplitudes set the Landau clusters, the shape sets the rest.
The fit is Nelder-Mead over (p0, w, t0[, ln Q for 'uniform']), started from the hit ladder (u_h, earliest hit) with
the start offsets learned once from model windows (no reference, no waveform information from the event).
"""
import numpy as np
from scipy.optimize import minimize

import emulate as E

S_T = 15.0          # ns: per-strip time scale of the cost (the order of the measured per-strip time scatter)
F_A = 0.15          # relative amplitude tolerance
P_MISS = 4.0        # cost of a strip hit in data but not in the model or vice versa (~2 sigma)


class H2C:
    def __init__(self, wm, hyper, view, family, gap_mm, v_um_ns, K, nsig=5.0, ideal=None, qmode='amp', lam=3.0,
                 qshape=None):
        self.wm, self.hyper, self.view, self.fam = wm, hyper, view, family
        self.qmode, self.lam = qmode, lam
        D2 = np.zeros((max(K - 2, 0), K))
        for i in range(K - 2):
            D2[i, i:i + 3] = (1.0, -2.0, 1.0)
        self.D2 = D2
        self.nsig, self.ideal = nsig, ideal
        T = gap_mm / (v_um_ns / 1000.0)                       # drift time across the gap [ns]
        uk = (np.arange(K) + 0.5) * wm.DT
        self.qshape = np.clip((T - np.arange(K) * wm.DT) / wm.DT, 0.0, 1.0)  # partial last bin
        if qmode in ('mean', 'ampmean'):
            if qshape is None:
                raise ValueError(f'qmode {qmode} needs a calibrated qshape')
            qs = np.clip(np.asarray(qshape, float)[:K], 0.0, None)
            self.qshape = qs * (self.qshape.sum() / max(qs.sum(), 1e-12))  # same total as the uniform column
        self.K = K
        self.off = dict(dt0=0.0, dp0=0.0)                     # start offsets, set by learn_offsets

    def emulate(self, Wm, noise):
        f = self.fam
        if f == 'dream':
            return E.dream(Wm, noise, nsig=self.nsig)
        if f == 'fd':
            return E.fd(Wm, noise, nsig=self.nsig)
        if f == 'ideal':
            return self.ideal(Wm, noise, nsig=self.nsig)
        if f == 'binary':
            return E.binary(Wm, noise, nsig=self.nsig, phase=12.5)
        return getattr(E, f)(Wm, noise, nsig=self.nsig)

    def model(self, pos, p0, w, t0, lnQ, gain):
        q = np.exp(lnQ) * self.qshape
        return self.wm.model_waveforms(self.view, pos, p0, w, t0, q, self.hyper) * gain[:, None]

    def solve_q(self, B, n, gain, noise, obs):
        """q >= 0 from the observed hit amplitudes (see module doc).  B: (n*NSAMP, K) design matrix (no gain)."""
        from scipy.optimize import nnls
        ns = B.shape[0] // n
        Bs = B.reshape(n, ns, -1) * gain[:, None, None]
        hit = np.flatnonzero(np.isfinite(obs['peak']))
        if not len(hit):
            return None
        a = obs['peak'][hit]
        sa = F_A * np.abs(a) + 3 * noise[hit]
        q = np.full(B.shape[1], max(np.nansum(a), 1.0) / max(self.qshape.sum(), 1.0)) * self.qshape
        for _ in range(2):
            Mq = Bs @ q                                            # (n, ns)
            kk = np.argmax(Mq[hit], axis=1)
            A = Bs[hit, kk, :]                                     # rows at each hit strip's model peak sample
            sc = np.median(np.abs(A[A > 0])) if (A > 0).any() else 1.0
            r = np.sqrt(self.lam) * sc / np.median(sa)
            if self.qmode == 'ampmean':
                # unknowns [q, c]: data rows [A, 0]; prior rows r (q_k - c shape_k)
                K = A.shape[1]
                lhs = np.vstack([np.hstack([A / sa[:, None], np.zeros((len(a), 1))]),
                                 r * np.hstack([np.eye(K), -self.qshape[:, None]])])
                rhs = np.r_[a / sa, np.zeros(K)]
                q = nnls(lhs, rhs, maxiter=400)[0][:K]
                continue
            lhs = np.vstack([A / sa[:, None], r * self.D2])
            rhs = np.r_[a / sa, np.zeros(self.D2.shape[0])]
            q, _ = nnls(lhs, rhs, maxiter=200)
        return q

    def cost(self, x, pos, gain, noise, obs):
        if self.qmode in ('amp', 'ampmean'):
            p0, w, t0 = x[:3]
            B = self.wm.build_matrix(self.view, pos, p0, w, t0, self.hyper)
            q = self.solve_q(B, len(pos), gain, noise, obs)
            if q is None:
                return 1e9
            Wm = (B @ q).reshape(len(pos), -1) * gain[:, None]
        else:
            Wm = self.model(pos, *x, gain)
        pr = self.emulate(Wm, noise)
        ho, hp = np.isfinite(obs['t']), np.isfinite(pr['t'])
        both = ho & hp
        c = np.sum(((obs['t'][both] - pr['t'][both]) / S_T) ** 2)
        ao = obs['a'][both]
        if self.fam != 'binary':
            c += np.sum(((ao - pr['a'][both]) / (F_A * np.abs(ao) + 3 * noise[both])) ** 2)
        c += P_MISS * np.sum(ho ^ hp)
        return c

    def start(self, pos, obs, u_h):
        k = np.flatnonzero(np.isfinite(obs['t']))
        if not len(k):
            return None
        j = k[np.argmin(obs['t'][k])]
        w0 = u_h / 1.2 if np.isfinite(u_h) else 0.0          # the hits ladder reads ~1.2x too steep (P0)
        Q0 = np.nansum(obs['a']) / max(self.qshape.sum(), 1.0)
        return np.array([pos[j] + self.off['dp0'], w0, obs['t'][j] + self.off['dt0'], np.log(max(Q0, 1.0))])

    def fit(self, pos, gain, noise, obs, u_h, n_start=3):
        x0 = self.start(pos, obs, u_h)
        if x0 is None:
            return None
        best = None
        if self.qmode in ('amp', 'ampmean'):
            x0 = x0[:3]
        for fw in ([1.0, 0.6, 1.5][:n_start]):
            xs = x0.copy()
            xs[1] *= fw
            steps = [[0.4, 0, 0, 0], [0, 0.003, 0, 0], [0, 0, 40.0, 0], [0, 0, 0, 0.3]]
            simplex = np.array([xs] + [xs + np.array(st[:len(xs)]) for st in steps[:len(xs)]])
            r = minimize(self.cost, xs, args=(pos, gain, noise, obs), method='Nelder-Mead',
                         options=dict(initial_simplex=simplex, maxfev=500, xatol=1e-4, fatol=1e-3))
            if best is None or r.fun < best.fun:
                best = r
        return dict(p0=best.x[0], w=best.x[1], t0=best.x[2], lnQ=best.x[3] if len(best.x) > 3 else np.nan,
                    cost=float(best.fun), nfev=int(best.nfev))

    def learn_offsets(self, planes, n=200):
        """median (t0 - earliest hit time) and (p0 - its strip position) on the pass MODEL windows of n planes: the
        emulator's own response to the model, no reference and no event waveform beyond the hit list."""
        dts, dps = [], []
        for p in planes[:n]:
            if p.get('M') is None:
                continue
            h = self.emulate(p['M'].astype(float), p['noise'].astype(float))
            k = np.flatnonzero(np.isfinite(h['t']))
            if not len(k):
                continue
            j = k[np.argmin(h['t'][k])]
            dts.append(p['t0'] - h['t'][j])
            dps.append(p['p0'] - float(p['pos'][j]))
        if dts:
            self.off = dict(dt0=float(np.median(dts)), dp0=float(np.median(dps)))
        return self.off
