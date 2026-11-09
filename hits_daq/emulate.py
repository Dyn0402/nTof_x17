"""emulate.py -- hit-DAQ emulators on one plane's window (Arm D and Arm S of HITS_DAQ_PLAN).

A window is (n_strip, n_sample) ADC at SNS = 60 ns per sample, with a per-strip noise sigma.  Every emulator turns each
strip's pulse into at most one hit (t [ns], a, flags) -- the information a hit-only front end keeps -- and returns a
dict of per-strip arrays (NaN where the strip has no hit):

  t       the hit time [ns] (the family's estimator)
  a       the hit amplitude (peak height, ToT or 1 for binary), the weight a hits tier may use
  peak    the parabolic peak height (always filled where there is a hit; for QA / cross-family comparison)
  nb      True where the strip was read only through neighbour logic (below threshold)

Families (papers/paper2/hits_daq/DAQ_SURVEY.md "Which algorithm families exist"):

  V0  dream     our WaveformAnalyzer.cpp: boxcar matched-filter gate (MF samples) above NSIG x the gate noise, time =
                30 % leading edge of the parabolic peak, amplitude = parabolic peak.  Pedestal + common-mode subtracted.
  V1  peak      peak time + peak amplitude (VMM3a time-at-peak, its designed mode).  Analog front end: no common-mode
                subtraction (raw - ped), threshold on the shaped pulse, one hit per channel, neighbour enable.
  V2  thr       threshold-crossing time + peak amplitude (VMM3a time-at-threshold; TIGER ToA + E-branch;
                STS-XYTER).  Same front-end conditions as V1.
  V3  tot       threshold-crossing time + time-over-threshold as the amplitude (TIGER ToT mode, Timepix3).
  V4  fd        sampled + per-strip Fermi-Dirac rising-edge fit, time at half height (the APV25 / SAMPA practice).
  V5  binary    threshold crossing quantised to a 25 ns bunch-crossing clock, no amplitude (VFAT3-like).
  V6  ideal     best single time per strip: the strip's waveform fitted by the measured single-electron template
                (amplitude, t0; grid + parabola), i.e. a matched-filter time.  Upper bound for one-time-per-strip.
  V7  cw        sampled: charge-weighted mean time of the first pulse above threshold, peak amplitude.  One time per strip that
                uses the whole pulse, the estimator the literature's cluster times are closest to (needs samples: W, as V0/V4).
  V8  totmid    midpoint of the threshold crossings, ToT as the amplitude (analog logic: R, as V3).  Not in FAMILIES (the
                default of tiers.py, whose old tables lack them): ask for them with --families cw,totmid.

Sub-sample times come from a monotone cubic (PCHIP) upsampling of the 60 ns samples to 1 ns.  The DREAM shaping is
kept (it cannot be undone from 60 ns samples): V1-V3 and V5 emulate the hit LOGIC of those front ends at DREAM shaping.
"""
import os

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

SNS = 60.0
FINE = 1.0
GATE_K = 0.68  # production rms_gate / rms (see dream)


def _parab(y, k):
    if k <= 0 or k >= len(y) - 1:
        return float(k), float(y[k])
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    d = float(np.clip(d, -0.5, 0.5))
    return k + d, float(y1 - 0.25 * (y0 - y2) * d)


def _fine(y):
    ts = np.arange(len(y)) * SNS
    tf = np.arange(0.0, ts[-1] + FINE / 2, FINE)
    return tf, PchipInterpolator(ts, y)(tf)


def _first_up(tf, yf, thr, start=0):
    """first upward crossing of thr at or after index start (linear between fine points); (t, index) or (nan, -1)."""
    above = yf >= thr
    idx = np.flatnonzero(above[start + 1:] & ~above[start:-1])
    if above[start]:
        return tf[start], start
    if not len(idx):
        return np.nan, -1
    j = start + idx[0]
    t = tf[j] + (thr - yf[j]) / (yf[j + 1] - yf[j]) * (tf[j + 1] - tf[j])
    return t, j + 1


def _empty(n):
    return dict(t=np.full(n, np.nan), a=np.full(n, np.nan), peak=np.full(n, np.nan), nb=np.zeros(n, bool))


def _neighbours(out, fired, values_fn, n):
    """VMM neighbour enable: every strip adjacent (in window order = position order) to a fired strip is read too."""
    nbr = np.zeros(n, bool)
    nbr[:-1] |= fired[1:]
    nbr[1:] |= fired[:-1]
    for i in np.flatnonzero(nbr & ~fired):
        r = values_fn(i)
        if r is not None:
            out['t'][i], out['a'][i], out['peak'][i] = r
            out['nb'][i] = True


def dream(W, noise, nsig=5.0, mf=5, frac=0.3, gate_k=None):
    """V0.  gate_k: the gate noise (sigma of the 5-sample boxcar of the noise, production's per-channel `rms_gate`) in
    units of the per-sample noise; scalar or one value per strip.  Default GATE_K = 0.68, measured from production hits
    (rms_gate = boxcar peak / significance) on det3 Run 1 and Run 2: per channel 0.645-0.72, not the white-noise
    1/sqrt(mf) = 0.447, because the noise is time-correlated.  validate_v0.py measures it per channel."""
    n, ns = W.shape
    out = _empty(n)
    gk = np.broadcast_to(np.asarray(GATE_K if gate_k is None else gate_k, float), (n,))
    ker = np.ones(mf) / mf
    for i in range(n):
        y = W[i]
        g = np.convolve(y, ker, mode='same')
        if not np.any(g >= nsig * gk[i] * noise[i]):
            continue
        k = int(np.argmax(y))
        kk, A = _parab(y, k)
        if A <= 0:
            continue
        thr = frac * A
        j = k
        while j > 0 and y[j - 1] >= thr:
            j -= 1
        t = 0.0 if j == 0 else (j - 1 + (thr - y[j - 1]) / (y[j] - y[j - 1])) * SNS
        out['t'][i], out['a'][i], out['peak'][i] = t, A, A
    return out


def _analog(R, noise, nsig, mode, neighbour=True):
    """V1 / V2 / V3 / V5 on an analog-like window R (raw - ped, no common mode): threshold nsig x noise on the pulse,
    first pulse only (one hit per channel per event)."""
    n, ns = R.shape
    out = _empty(n)
    fired = np.zeros(n, bool)
    cache = {}

    def strip(i, need_cross):
        tf, yf = cache.get(i) or cache.setdefault(i, _fine(R[i]))
        thr = nsig * noise[i]
        tc, jc = _first_up(tf, yf, thr)
        if need_cross and jc < 0:
            return None
        # the first pulse: from the crossing to where it falls back below thr (a neighbour: the whole window)
        if jc >= 0:
            je = _pulse_end(yf, jc, thr)
            k = jc + int(np.argmax(yf[jc:je + 1]))
        else:
            k = int(np.argmax(yf))
        A = float(yf[k])
        if A <= 0:
            return None
        if mode == 'peak':
            return tf[k], A, A
        if jc < 0:                       # neighbour below threshold: no crossing; time at peak, flagged by nb
            return tf[k], (0.0 if mode == 'tot' else (1.0 if mode == 'binary' else A)), A
        if mode == 'thr':
            return tc, A, A
        if mode == 'tot':
            below = np.flatnonzero(yf[jc:] < thr)
            te = tf[jc + below[0]] if len(below) else tf[-1]
            return tc, te - tc, A
        if mode == 'binary':
            return tc, 1.0, A
        if mode == 'cw':                 # the fired pulse only: [crossing, back below threshold]
            y = yf[jc:je + 1]
            return float(np.sum(tf[jc:je + 1] * y) / np.sum(y)), A, A
        if mode == 'totmid':
            below = np.flatnonzero(yf[jc:] < thr)
            te = tf[jc + below[0]] if len(below) else tf[-1]
            return 0.5 * (tc + te), te - tc, A
        raise ValueError(mode)

    for i in range(n):
        r = strip(i, True)
        if r is not None:
            out['t'][i], out['a'][i], out['peak'][i] = r
            fired[i] = True
    if neighbour and mode in ('peak', 'thr'):
        _neighbours(out, fired, lambda i: strip(i, False), n)
    return out


def _pulse_end(yf, j0, thr):
    below = np.flatnonzero(yf[j0:] < thr)
    return j0 + below[0] if len(below) else len(yf) - 1


def peak(R, noise, nsig=5.0, neighbour=True):
    return _analog(R, noise, nsig, 'peak', neighbour)


def thr(R, noise, nsig=5.0, neighbour=True):
    return _analog(R, noise, nsig, 'thr', neighbour)


def tot(R, noise, nsig=5.0):
    return _analog(R, noise, nsig, 'tot', False)


def cw(W, noise, nsig=5.0):
    return _analog(W, noise, nsig, 'cw', False)


def totmid(R, noise, nsig=5.0):
    return _analog(R, noise, nsig, 'totmid', False)


def binary(R, noise, nsig=5.0, bx=25.0, phase=None, rng=None):
    out = _analog(R, noise, nsig, 'binary', False)
    ph = (rng or np.random.default_rng(0)).uniform(0, bx) if phase is None else phase
    k = np.isfinite(out['t'])
    out['t'][k] = np.floor((out['t'][k] - ph) / bx) * bx + ph + bx / 2
    out['a'][k] = 1.0
    return out


def _fd(t, q0, qm, tf, s):
    return q0 + qm / (1.0 + np.exp(-(t - tf) / s))


def fd(W, noise, nsig=5.0):
    """V4: Fermi-Dirac fit to the rising edge (samples from the pulse start to the peak), time at half height (tf)."""
    n, ns = W.shape
    out = _empty(n)
    ts = np.arange(ns) * SNS
    for i in range(n):
        y = W[i]
        k = int(np.argmax(y))
        if y[k] < nsig * noise[i] or k < 2:
            continue
        kk, A = _parab(y, k)
        j = k
        while j > 0 and y[j - 1] > 0.1 * A:
            j -= 1
        lo = max(0, j - 2)
        tt, yy = ts[lo:k + 1], y[lo:k + 1]
        t50 = np.interp(0.5 * A, yy[:np.argmax(yy) + 1], tt[:np.argmax(yy) + 1]) if len(yy) > 1 else ts[k]
        try:
            p, _ = curve_fit(_fd, tt, yy, p0=(0.0, A, t50, 25.0), maxfev=400,
                             bounds=([-5 * noise[i], 0, ts[0] - SNS, 3.0], [5 * noise[i], 3 * A, ts[-1], 300.0]))
            t = p[2]
        except Exception:
            t = t50
        out['t'][i], out['a'][i], out['peak'][i] = t, A, A
    return out


class Ideal:
    """V6: per-strip template fit.  tmpl(t) is the measured unit response on a fine grid (tgrid [ns]); the strip is
    fitted as a * tmpl(t - t0) by a t0 scan (2 ns grid) with the linear a solved at each t0, then a parabola."""

    def __init__(self, tgrid, tmpl, ns, step=2.0):
        self.tgrid, self.tmpl = np.asarray(tgrid, float), np.asarray(tmpl, float)
        ts = np.arange(ns) * SNS
        self.t0s = np.arange(-600.0, ts[-1], step)
        self.B = np.stack([np.interp(ts - t0, self.tgrid, self.tmpl, left=0, right=0) for t0 in self.t0s])
        self.bb = np.einsum('ij,ij->i', self.B, self.B)

    def __call__(self, W, noise, nsig=5.0):
        n = W.shape[0]
        out = _empty(n)
        for i in range(n):
            y = W[i]
            k = int(np.argmax(y))
            if y[k] < nsig * noise[i]:
                continue
            by = self.B @ y
            a = by / np.maximum(self.bb, 1e-12)
            c = by * a                                   # chi2 reduction
            j = int(np.argmax(c))
            jj, _ = _parab(c, j)
            t0 = np.interp(jj, np.arange(len(self.t0s)), self.t0s)
            out['t'][i], out['a'][i], out['peak'][i] = t0, a[j], _parab(y, k)[1]
        return out


FAMILIES = ('dream', 'peak', 'thr', 'tot', 'fd', 'binary', 'ideal')
CODES = dict(dream='V0', peak='V1', thr='V2', tot='V3', fd='V4', binary='V5', ideal='V6', cw='V7', totmid='V8')


PASS1_EOS = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'


def bundle_path(meta, override=None):
    """The window file's calibration bundle.  extract_windows.py records the bundle as it was in the extraction job
    (a copy of the row's Analysis_pass1 bundle in the condor scratch dir, '<scratch>/Analysis/<run>/<sub>/<det>/wft/
    calib_bundle'); when that path is gone, the same row's bundle on EOS Analysis_pass1 is used."""
    if override:
        return override
    b = meta['bundle']
    if os.path.exists(os.path.join(b, 'bundle.json')) or '/Analysis/' not in b:
        return b
    return os.path.join(PASS1_EOS, b.split('/Analysis/', 1)[1])
