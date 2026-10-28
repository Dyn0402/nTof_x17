"""T2: the physical response the simulation uses (MX17_Geant response/: DREAM PZC + zeta 0.75 Sallen-Key, residual CSA
pole beta, ion rectangle of transit T carrying f_ion of the charge + prompt electron spike), in the template's
own shape statistics (t50-aligned, unit peak), against the data's tan -> infinity extrapolation (t1_extrap.json).
Usage: python -I t2_shaper_model.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/CLionProjects/MX17_Geant'))
from response.dream.shaper import DreamShaper

HERE = os.path.dirname(os.path.abspath(__file__))


def response(t_peak=180.0, beta=0.75, T_ion=340.0, f_ion=0.908, dt=1.0, tmax=6000.0):
    sh = DreamShaper(pzc_residual=beta, t_peak_ns=t_peak, dt_ns=dt, t_max_ns=tmax)
    t = np.arange(0, tmax, dt)
    cur = np.zeros_like(t); cur[0] = (1 - f_ion) / dt
    cur[t < T_ion] += f_ion / T_ion
    return t, sh.apply(cur * 1e9, dt)            # apply() multiplies by dt*1e-9


def stats(t, h):
    h = h / h.max(); ipk = int(np.argmax(h))
    def first(th):
        i = np.argmax(h >= th); return t[i - 1] + (th - h[i - 1]) / (h[i] - h[i - 1]) * (t[i] - t[i - 1])
    t50 = first(0.5); r10 = first(0.1); r90 = first(0.9)
    j = ipk + np.argmax(h[ipk:] < 0.5); t50b = t[j - 1] + (h[j - 1] - 0.5) / (h[j - 1] - h[j]) * (t[j] - t[j - 1])
    return dict(fwhm=t50b - t50, rise=r90 - r10, peak_after_t50=t[ipk] - t50, under=float(h.min()),
                at800=float(np.interp(t50 + 800, t, h)))


if __name__ == '__main__':
    rows = []
    for tp in (158, 166, 180):
        for beta in (0.0, 0.75):
            for T, f in ((0.0, 0.0), (340.0, 0.908), (300.0, 0.908), (340.0, 0.5)):
                s = stats(*response(tp, beta, T if T else 1.0, f))
                rows.append(dict(t_peak=tp, beta=beta, T_ion=T, f_ion=f, **s))
                print(f"t_peak {tp} beta {beta:.2f} ion T {T:3.0f} f {f:.3f}: FWHM {s['fwhm']:5.0f} rise10-90 {s['rise']:4.0f} "
                      f"peak-t50 {s['peak_after_t50']:4.0f} under {s['under']:+.3f} h(t50+800) {s['at800']:+.3f}")
    ext = json.load(open(f'{HERE}/results/t1_extrap.json'))
    print('data, tan->inf FWHM:', {d: {p: round(v[p]['fwhm_inf']) for p in v} for d, v in ext.items()})
    json.dump(rows, open(f'{HERE}/results/t2_shaper_model.json', 'w'), indent=1)
