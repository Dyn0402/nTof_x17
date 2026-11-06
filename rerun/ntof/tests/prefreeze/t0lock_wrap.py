"""Run ntof_reco with wft.model._fit_plane_lm wrapped: optional t0-start shift (T0SHIFT env), and a chi2(t0) profile
at the fit's (p0, w) written per fit to T0LOG (jsonl)."""
import json, os, sys
import numpy as np
sys.path[:0] = ['/home/dylan/PycharmProjects/nTof_x17/rerun/ntof', '/home/dylan/PycharmProjects/nTof_x17']
import wft.model as wm
SH = float(os.environ.get('T0SHIFT', '0'))
LOG = os.environ['T0LOG']
orig = wm._fit_plane_lm
DTS = [-5, -2.5, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2.5, 5]
def wrapped(W, noise, pos, sat, plane, p0_init, w_init, t0_init, hyper, t0_prior, pivot):
    r = orig(W, noise, pos, sat, plane, p0_init, w_init, t0_init + SH, hyper, t0_prior, pivot)
    if np.isfinite(r['chi2']):
        prof = [float(wm._vp_eval(plane, W, noise, pos, sat, np.array([r['p0'], r['w'], r['t0'] + d]), hyper, True,
                                  t0_prior)[0]) for d in DTS]
        with open(f'{LOG}.{os.getpid()}', 'a') as f:
            f.write(json.dumps(dict(plane=plane, t0=r['t0'], t0_init=float(t0_init), chi2=r['chi2'], prof=prof)) + '\n')
    return r
wm._fit_plane_lm = wrapped
if os.environ.get('SMOOTH_TMPL'):
    from scipy.interpolate import CubicSpline
    _uc = wm.use_calibration
    def use_cal(cal):
        _uc(cal)
        g = wm.TGRID; gf = np.arange(g[0], g[-1] + 1e-9, 1.0)
        wm.TMPL = {p: CubicSpline(g, wm.TMPL[p])(gf) for p in wm.TMPL}
        wm.TGRID = gf
        wm._rc_cache.clear()
    wm.use_calibration = use_cal
sys.argv = ['ntof_reco.py'] + sys.argv[1:]
import runpy
runpy.run_path('/home/dylan/PycharmProjects/nTof_x17/rerun/ntof/ntof_reco.py', run_name='__main__')
