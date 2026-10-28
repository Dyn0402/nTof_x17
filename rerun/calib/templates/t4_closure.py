"""T4: closure of the template measurement on the simulation, and the corrected joint fit.

Truth: the v4c per-unit-charge response = DREAM(166 ns, beta 0.75) (x) the reconstructed longitudinal current
(results/h_long_v4c.npy, from MX17_Geant ions.reconstructed_longitudinal, meshfield pooled point, K0 2.05, 490 V:
f_ion 0.900, ion charge 50/90/99 % by 107/191/222 ns).  (The resistive-sheet self kernel of the strip is NOT in it.)
Measured: t3's sim templates per angle (t3_sim_v4c.json) and t1's data templates per |tan| bin.
Joint fit: template(tan) = unit-peak, t50-aligned h (x) box(L / (v tan)), box with fractional width (smooth in L),
h = DREAM(t_peak, beta) (x) [(1-f) delta + f * the v4c ion shape stretched by s].  Free t_peak, beta, f, s, L.
Usage: python -I t4_closure.py"""
import json, os, pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/CLionProjects/MX17_Geant'))
from response.dream.shaper import DreamShaper
from t1_tan_amp import CACHE, TAN_BINS, pulses, GRID, HERE
from t2_shaper_model import stats

DT = 2.0
T = np.arange(0, 4000, DT)
HL = np.load(f'{HERE}/results/h_long_v4c.npy')          # (t_ns, h) on 0.2 ns, unit area
V = dict(det2=39.3, det3=35.8, det4=33.0, det6=24.2, det7=36.8, sim=37.18)
FIT = (GRID >= -300) & (GRID <= 1300)
_SH = {}


def shaper(tp, beta):
    k = (round(tp, 2), round(beta, 3))
    if k not in _SH:
        _SH[k] = DreamShaper(pzc_residual=beta, t_peak_ns=tp, dt_ns=DT, t_max_ns=4000.0).h
    return _SH[k]


def ion_current(f, s):
    """unit-area current on T: (1-f) prompt spike + f * the v4c ion part stretched in time by s."""
    t, h = HL[:, 0], HL[:, 1]
    hi = np.where(t > 1.0, h, 0.0); hi /= hi.sum() * (t[1] - t[0])
    cur = np.interp(T, t * s, hi, right=0.0) / s * f
    cur[0] += (1 - f) / DT
    return cur


def response(tp, beta, f, s):
    y = np.convolve(ion_current(f, s) * DT, shaper(tp, beta))[:len(T)]
    return y


def smooth_box(h, width_ns):
    """h convolved with a unit-area box of fractional width (linear in width, so the fit sees a gradient)."""
    n = width_ns / DT
    k = int(np.floor(n)); fr = n - k
    ker = np.ones(k + 1); ker[-1] = fr if k >= 1 else 1.0
    if k < 1:
        ker = np.array([1.0])
    return np.convolve(h, ker / ker.sum())[:len(h)]


def align(h):
    h = h / h.max(); i = np.argmax(h >= 0.5)
    t50 = T[i - 1] + (0.5 - h[i - 1]) / (h[i] - h[i - 1]) * DT
    return np.interp(GRID, T - t50, h, left=0.0, right=np.nan)


def joint_fit(data, tans, v, pin=None):
    def model(p):
        tp, beta, f, s, L = p
        h = response(tp, beta, f, s)
        return [align(np.mean([smooth_box(h, L * 1000 / (v * tn)) for tn in tb], axis=0)) for tb in tans]

    def resid(p):
        if pin:
            p = [p[0], p[1], pin[0], pin[1], p[2]]
        return np.concatenate([np.nan_to_num((m - d)[FIT]) for m, d in zip(model(p), data)])
    if pin:
        r = least_squares(resid, [166, .3, 1.0], bounds=([80, 0, .1], [300, 1, 4]), x_scale=[20, .2, .3], diff_step=1e-3)
        p = [r.x[0], r.x[1], pin[0], pin[1], r.x[2]]
    else:
        best = None
        for f0, s0 in ((0.9, 1.0), (0.5, 1.0), (0.9, 0.5)):
            rr = least_squares(resid, [166, .3, f0, s0, 1.0], bounds=([80, 0, 0, .2, .1], [300, 1, 1, 4, 4]),
                               x_scale=[20, .2, .2, .3, .3], diff_step=1e-3)
            if best is None or rr.cost < best.cost:
                best = rr
        r, p = best, list(best.x)
    return p, float(np.sqrt(np.mean(r.fun ** 2)))


def report(name, p, rms, extra=''):
    h = response(*p[:4]); st = stats(T, h)
    print(f'{name:10s} t_peak {p[0]:5.1f} beta {p[1]:.2f} f_ion {p[2]:.2f} ion-stretch {p[3]:.2f} L {p[4]:.2f} mm '
          f'rms {rms:.4f} | h: FWHM {st["fwhm"]:4.0f} rise {st["rise"]:4.0f} peak-t50 {st["peak_after_t50"]:4.0f} '
          f'under {st["under"]:+.3f} {extra}')
    return dict(t_peak=p[0], beta=p[1], f_ion=p[2], ion_stretch=p[3], L_mm=p[4], rms=rms, h_stats=st)


if __name__ == '__main__':
    out = {}
    truth = response(166.0, 0.75, 0.9006, 1.0); st = stats(T, truth)
    print(f'sim TRUTH h (166 ns, beta .75, v4c ions): FWHM {st["fwhm"]:.0f} rise {st["rise"]:.0f} '
          f'peak-t50 {st["peak_after_t50"]:.0f} under {st["under"]:+.3f}')
    out['sim_truth'] = st
    # sim closure: X view (feu07) at thx10..30
    sim = json.load(open(f'{HERE}/results/t3_sim_v4c.json'))
    data, tans = [], []
    for tag in ('thx10', 'thx15', 'thx20', 'thx30'):
        r = sim[f'{tag}_feu07']; data.append(np.array(r['shape'])); tans.append([r['tan']])
    p, rms = joint_fit(data, tans, V['sim'])
    out['sim_free'] = report('sim free', p, rms)
    p, rms = joint_fit(data, tans, V['sim'], pin=(0.9006, 1.0))
    out['sim_pinned_ions'] = report('sim pin', p, rms, '(ions pinned to truth)')
    # data, X and Y, |tan| >= 0.15 bins
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        ev = pickle.load(open(CACHE.format(det), 'rb'))
        for plane in 'xy':
            P = pulses(ev, plane); tn = np.array([q[0] for q in P]); S = np.array([q[2] for q in P])
            data, tans = [], []
            for lo, hi in TAN_BINS[2:]:
                m = (tn >= lo) & (tn < hi)
                if m.sum() < 25:
                    continue
                d = np.nanmedian(S[m], axis=0); d -= np.nanmedian(d[GRID < -250]); d /= np.nanmax(d)
                data.append(d); tans.append(np.quantile(tn[m], np.linspace(0.05, 0.95, 7)))
            if len(data) < 2:
                continue
            p, rms = joint_fit(data, tans, V[det])
            out[f'{det}_{plane}'] = report(f'{det} {plane}', p, rms, f'({len(data)} bins)')
    json.dump(out, open(f'{HERE}/results/t4_closure.json', 'w'), indent=1)
