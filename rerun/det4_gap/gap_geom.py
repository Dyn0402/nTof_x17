#!/usr/bin/env python3
"""gap_geom.py <npz dir> [--out results/sps_gap.json] -- det4's drift gap from the strip extent of SPS rotated-mount tracks.

No drift velocity, no pulse model.  A track through the rotated chamber crosses the gap g at the mount angle theta, so
in the projected view its ionisation column covers L = g * tan(theta) of strips, from the mesh end (earliest arrival)
to the cathode end (latest).  Per event (the npz of sps_extent.py, the pass's own builder at +-32 strips):

  1. the projected view's strips with a peak >= PK_MIN; a robust (Theil-Sen) line peak-time vs position through the
     strips >= PK_LINE; strips more than T_TOL samples off the line are pile-up / noise and count as zero charge;
  2. the line must be inclined (|slope| >= SLOPE_MIN samples/mm) and both column ends inside the readout window
     (first strip peak >= T_FIRST, last <= n_samples - T_LAST), else the far end may be cut by the window;
  3. the window-integral charge of the kept strips is fitted with an attenuated box blurred by the footprint,
         q(y) = A exp(k (y - a)) [Phi((y - a)/s) - Phi((y - b)/s)],
     s fixed to the flat-mount footprint of the same gas (fit of the same strip charge on the flat row's events).
     The half-height edges of a symmetric blur sit at a and b whatever s is, and the free k (attachment, gain
     variation along the column, ZS losses at the faint end) keeps the far edge from being read inward;
  4. L = |b - a|; the gap is median(L) / tan(theta), the error the bootstrap of the median plus the mount angle.

Every row is quoted on its own and per gas and angle: geometry is the same in every gas and field, attachment and
the time window are not, so their agreement is the test.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

PK_MIN = 60.0          # ADC, ~6 x the strip noise
PK_LINE = 150.0
T_TOL = 4              # samples off the time-position line
SLOPE_MIN = 0.4        # samples / mm: 28 mm in ~2-3 us over ~13 mm of strips is ~2-3 samples / mm
T_FIRST = 2
T_LAST = 3
MIN_STRIPS = 6
CONTAIN_MM = 1.6       # both fitted edges at least ~2 strips inside the window, or the extent is cut
#: mount angles (deg): the config value, and the measured yaw of run57 (det4-sps-strip-map memory: alignment cos(phi)
#: gives 25.4 against the config's 25.64); the half difference is carried as the angle error
MOUNT = {25.64: (25.52, 0.12), 15.465: (15.465, 0.12)}


def theil_sen(x, y):
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]; ok = np.abs(dx) > 1e-6
    b = np.median((y[j] - y[i])[ok] / dx[ok]) if ok.any() else 0.0
    return np.median(y - b * x), b


def box_model(p, y, s):
    A, k, a, b = p
    lo, hi = min(a, b), max(a, b)
    return A * np.exp(np.clip(k * (y - a), -30, 30)) * (ndtr((y - lo) / s) - ndtr((y - hi) / s))


def event_strips(d, v, i):
    m = np.isfinite(d[f'{v}_pos'][i]) & ~d[f'{v}_bad'][i]
    return d[f'{v}_pos'][i][m].astype(float), d[f'{v}_q'][i][m].astype(float), d[f'{v}_pk'][i][m].astype(float), \
        d[f'{v}_tpk'][i][m].astype(float)


def footprint_sigma(d, v):
    """Flat-mount footprint: Gaussian fit of each event's strip charge around its brightest strip (+-6 strips)."""
    out = []
    for i in range(len(d['eid'])):
        y, q, pk, t = event_strips(d, v, i)
        if len(y) < 5 or pk.max() < 300:
            continue
        c = y[np.argmax(pk)]; m = np.abs(y - c) <= 6 * 0.78
        yy, qq = y[m], np.where(pk[m] >= PK_MIN, q[m], 0.0)
        f = lambda p: p[0] * np.exp(-0.5 * ((yy - p[1]) / p[2]) ** 2) - qq
        try:
            r = least_squares(f, [qq.max(), c, 0.8], bounds=([0, c - 3, 0.2], [np.inf, c + 3, 4.0]))
            out.append(r.x[2])
        except Exception:
            pass
    return float(np.median(out)), float(np.std(out) / np.sqrt(max(len(out), 1))), len(out)


def event_extent(d, v, i, s, nsmp):
    y, q, pk, t = event_strips(d, v, i)
    use = pk >= PK_LINE
    if use.sum() < MIN_STRIPS:
        return None
    a0, b0 = theil_sen(y[use], t[use])
    if abs(b0) < SLOPE_MIN:
        return None
    keep = (pk >= PK_MIN) & (np.abs(t - (a0 + b0 * y)) <= T_TOL)
    if keep.sum() < MIN_STRIPS:
        return None
    tk = t[keep]
    # only the start is cut on: a cut on the last arrival (tk.max() inside the window) keeps exactly the events whose
    # far end is invisible and reads L short (v2, run57: 20.3 / 22.7 mm at d600 / d650); the window's effect is read
    # off L against the start time instead (summarise: the early-start plateau)
    if tk.min() < T_FIRST:
        return None
    qq = np.where(keep, q, 0.0)
    yk = y[keep]
    # the mesh end is the earliest-arriving end
    y_mesh = yk[np.argmin(tk)]; y_cath = yk[np.argmax(tk)]
    p0 = [np.median(qq[keep]), 0.0, y_mesh, y_cath]
    span = abs(y_cath - y_mesh) + 10
    lo = [0, -2.0, min(y_mesh, y_cath) - 8, min(y_mesh, y_cath) - 8]
    hi = [np.inf, 2.0, max(y_mesh, y_cath) + 8, max(y_mesh, y_cath) + 8]
    try:
        r = least_squares(lambda p: box_model(p, y, s) - qq, p0, bounds=(lo, hi), loss='soft_l1',
                          f_scale=max(np.median(qq[keep]) * 0.3, 1.0))
    except Exception:
        return None
    A, k, a, b = r.x
    if not np.isfinite(r.x).all() or abs(b - a) > span:
        return None
    if min(a, b) - y.min() < CONTAIN_MM or y.max() - max(a, b) < CONTAIN_MM:
        return None
    # k oriented mesh -> cathode (negative = the charge falls toward the cathode: attachment, ZS losses)
    k_mc = k * np.sign(y_cath - y_mesh) if y_cath != y_mesh else 0.0
    return dict(L=abs(b - a), k=k, k_mc=float(k_mc), n=int(keep.sum()), t0=float(tk.min()), t1=float(tk.max()), slope=float(b0),
                L_raw=abs(y_cath - y_mesh))


def boot_median(x, n=400, seed=1):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    return float(np.std([np.median(rng.choice(x, len(x))) for _ in range(n)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                       'results', 'sps_gap.json'))
    ap.add_argument('--keys', help='comma list (default: every rotated row in DIR)')
    ap.add_argument('--views', default='xy')
    ap.add_argument('--pk', help='PK_LINE,PK_MIN override (ADC)')
    ap.add_argument('--sigma', help='reuse flat-row footprints: co2x,co2y,cf4x,cf4y in mm (else fitted)')
    a = ap.parse_args()
    files = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(a.dir, '*.npz'))}
    global PK_LINE, PK_MIN
    if a.pk:
        PK_LINE, PK_MIN = (float(x) for x in a.pk.split(','))
    if a.keys:
        keep = set(a.keys.split(',')) | {'run55_flatdrift__d700', 'run63_flat__flat700'}
        files = {k: f for k, f in files.items() if k in keep}
    flat = {'co2': 'run55_flatdrift__d700', 'cf4': 'run63_flat__flat700'}
    res = dict(method=__doc__.split('\n\n')[0], pk_line=PK_LINE, pk_min=PK_MIN, rows={})
    sig = {}
    if a.sigma:
        c2x, c2y, c4x, c4y = (float(x) for x in a.sigma.split(','))
        sig = dict(co2=dict(x=(c2x, 0, 0), y=(c2y, 0, 0)), cf4=dict(x=(c4x, 0, 0), y=(c4y, 0, 0)))
    for gas, key in flat.items():
        if key in files and gas not in sig:
            d = np.load(files[key])
            sig[gas] = {v: footprint_sigma(d, v) for v in 'xy'}
            print(f'{key}: footprint sigma x {sig[gas]["x"][0]:.3f}  y {sig[gas]["y"][0]:.3f} mm', flush=True)
    res['footprint'] = sig
    for key, f in sorted(files.items()):
        d = np.load(f)
        mount = float(d['mount_deg'])
        if mount == 0.0:
            continue
        gas = 'co2' if ('co2' in key or key.startswith(('run55', 'run56', 'run57', 'run59'))) else 'cf4'
        nsmp = 32 if key.startswith(('run61', 'run66', 'run70')) else 64
        th, dth = MOUNT[mount]
        tan, dtan = np.tan(np.radians(th)), np.radians(dth) / np.cos(np.radians(th)) ** 2
        row = dict(gas=gas, mount_deg=th, n_samples=nsmp, n_events=int(len(d['eid'])))
        for v in a.views:
            s = sig.get(gas, sig.get('cf4', {})).get(v, (0.8, 0, 0))[0]
            E = [e for i in range(len(d['eid'])) if (e := event_extent(d, v, i, s, nsmp)) is not None]
            if len(E) < 30:
                row[v] = dict(n=len(E)); continue
            L = np.array([e['L'] for e in E]); t0 = np.array([e['t0'] for e in E]); t1 = np.array([e['t1'] for e in E])
            early = t0 <= np.percentile(t0, 50)
            # the column duration of the least-truncated events (earliest quarter of starts), and the start cut that
            # keeps a column of that duration inside the window
            q1 = t0 <= np.percentile(t0, 25)
            dur = float(np.percentile((t1 - t0)[q1], 90))
            t_cut = nsmp - 1 - T_LAST - dur
            fits = t0 <= t_cut
            bins = np.unique(np.percentile(t0, [0, 20, 40, 60, 80, 100]))
            by_t0 = [dict(t0_lo=float(lo), t0_hi=float(hi), n=int(((t0 >= lo) & (t0 <= hi)).sum()),
                          gap=float(np.median(L[(t0 >= lo) & (t0 <= hi)]) / tan)) for lo, hi in zip(bins[:-1], bins[1:])
                     if ((t0 >= lo) & (t0 <= hi)).sum() >= 20]
            Lf = L[fits] if fits.sum() >= 30 else L[q1]
            g = np.median(Lf) / tan
            dg = np.hypot(boot_median(Lf) / tan, g * dtan / tan)
            row[v] = dict(n=len(E), n_fits_window=int(fits.sum()), dur90_samples=dur, t0_cut=float(t_cut),
                          L_med=float(np.median(Lf)), gap=float(g), gap_err=float(dg), gap_all=float(np.median(L) / tan),
                          by_t0=by_t0,
                          gap_early_half=float(np.median(L[early]) / tan),
                          gap_late_half=float(np.median(L[~early]) / tan),
                          gap_rawends=float(np.median([e['L_raw'] for e in E]) / tan),
                          k_med=float(np.median([e['k'] for e in E])),
                          k_mc_med=float(np.median([e['k_mc'] for e in E])),
                          slope_med=float(np.median([e['slope'] for e in E])),
                          L_p16_p84=[float(np.percentile(L, 16)), float(np.percentile(L, 84))], footprint_sigma=s)
            print(f'{key} {v}: n {len(E)} (in window {fits.sum()}, dur90 {dur:.0f} smp, t0 <= {t_cut:.0f})  '
                  f'gap {g:.2f} +- {dg:.2f}  all {np.median(L) / tan:.2f}  by t0 '
                  + ' '.join(f'[{b["t0_lo"]:.0f}-{b["t0_hi"]:.0f}] {b["gap"]:.1f}' for b in by_t0)
                  + f'  k(mesh->cathode) {row[v]["k_mc_med"]:+.3f}/mm', flush=True)
        res['rows'][key] = row
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
