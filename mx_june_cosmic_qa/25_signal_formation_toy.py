#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25_signal_formation_toy.py

Signal-formation toy MC for the det3 micro-TPC: decide which velocity
estimator is unbiased, by generating waveform-level events with KNOWN v_true
and running the exact estimators used on data.

Measured facts to reproduce (long run, 1000 V):
  - shaper rise scale t_peak - t_CFD ~ 140 ns, amplitude-independent
  - cluster extent floor ~3 mm (x) at theta~0; extent slope 23.2 mm/unit-tan
  - full-cluster time-span plateau T_sat ~ 690 ns; core(30%) span ~ 555 ns
  - hits-like ridge (anchored amp-weighted) v ~ 28; waveform CFD core-OLS
    ridge v ~ 30; geometry extent-slope/T_sat v ~ 34
The question: which of these numbers equals v_true when the SAME estimators
run on simulated events?

Physics in the toy: ionisation clusters along the track (Poisson),
exponential gain fluctuations, attachment thinning exp(-z/lambda),
transverse + longitudinal diffusion ~ sqrt(z), prompt capacitive
cross-coupling to +-1/+-2 neighbours, CR-RC^2 shaping (peak 2*tau), 60 ns
sampling with random trigger phase, noise, threshold.

Usage: ../.venv/bin/python 25_signal_formation_toy.py [--vtrue=34] [--nev=4000]
Prints estimator results per v_true; optional scan over several v_true.
"""
import sys
import numpy as np

RNG = np.random.default_rng(17)

# ---------------- fixed detector/electronics parameters ----------------
PITCH = 0.78          # mm
GAP = 30.0            # mm drift gap
N_SAMP, DT = 32, 60.0
TAU_SH = 70.0         # ns  -> CR-RC^2 peaking at 2*tau = 140 ns
NOISE = 12.0          # ADC per sample
THR_HIT = 100.0       # ADC: strip recorded as a hit (matches data floor)
THR_WF = 150.0        # ADC: waveform time extraction threshold (as in 24)
CORE_FRAC = 0.30
MIN_STRIPS = 4
N_STRIPS = 128

# ---------------- tunable physics (defaults ~ best-fit gas) ----------------
LAMBDA_ATT = 15.0     # mm attachment length
DT_DIFF = 0.0287      # mm/sqrt(mm) transverse (287 um/sqrt(cm))
DL_DIFF = 0.020       # mm/sqrt(mm) longitudinal
CLUSTERS_PER_MM = 3.5
GAIN_SCALE = 260.0    # ADC per avalanche-electron-ish (tuned to core amp ~1000)
XT1, XT2 = 0.12, 0.025   # prompt capacitive cross-talk fractions

TS = np.arange(N_SAMP) * DT


def shaper(t):
    """CR-RC^2, unit peak amplitude, peak at 2*tau."""
    x = np.maximum(t, 0.0) / TAU_SH
    return (x ** 2) * np.exp(2.0 - x) / 4.0


def gen_event(v_true, tan_th):
    """Return dict of per-strip waveforms for one track."""
    x0 = RNG.uniform(20, N_STRIPS * PITCH - 20)
    t0 = RNG.uniform(200, 500)                      # trigger phase
    n_cl = RNG.poisson(CLUSTERS_PER_MM * GAP * np.sqrt(1 + tan_th**2))
    z = RNG.uniform(0, GAP, n_cl)
    q = RNG.exponential(1.0, n_cl) + 0.4            # cluster charge x gain fluct
    alive = RNG.random(n_cl) < np.exp(-z / LAMBDA_ATT)
    z, q = z[alive], q[alive]
    if len(z) < 5:
        return None
    x = x0 + z * tan_th + RNG.normal(0, DT_DIFF * np.sqrt(z))
    t = t0 + z * 1000.0 / v_true + RNG.normal(0, DL_DIFF * np.sqrt(z) * 1000.0 / v_true)
    s_idx = np.round(x / PITCH).astype(int)

    wf = np.zeros((N_STRIPS, N_SAMP))
    for k, frac in [(0, 1.0), (1, XT1), (-1, XT1), (2, XT2), (-2, XT2)]:
        si = s_idx + k
        ok = (si >= 0) & (si < N_STRIPS)
        for j in np.where(ok)[0]:
            wf[si[j]] += frac * q[j] * GAIN_SCALE * shaper(TS - t[j])
    wf += RNG.normal(0, NOISE, wf.shape)
    return wf, x0, t0


UNSHARE = False


def unshare(wf, c1, c2):
    """Exact inverse of the prompt sharing kernel, per sample (banded solve)."""
    from scipy.linalg import solve_banded
    n = wf.shape[0]
    ab = np.zeros((5, n))
    ab[0, 2:] = c2; ab[1, 1:] = c1; ab[2, :] = 1.0
    ab[3, :-1] = c1; ab[4, :-2] = c2
    return solve_banded((2, 2), ab, wf)


def cfd_time(w):
    ipk = int(np.argmax(w))
    a = w[ipk]
    if a < THR_WF or ipk == 0:
        return np.nan, a
    for i in range(1, ipk + 1):
        lvl = 0.5 * a
        if w[i] >= lvl > w[i - 1]:
            return DT * (i - 1 + (lvl - w[i - 1]) / (w[i] - w[i - 1])), a
    return np.nan, a


def analyse_events(v_true, n_ev, tan_lo=0.06, tan_hi=0.55):
    """Simulate and run the data estimators. Angles drawn cos^2-ish."""
    S_prod, S_cfd, tans = [], [], []
    exts, spans, spans_core, exts_core, atans = [], [], [], [], []
    floors = []
    n_done = 0
    while n_done < n_ev:
        tan_th = np.tan(RNG.normal(0, 0.35))
        if abs(tan_th) > 0.9:
            continue
        g = gen_event(v_true, tan_th)
        if g is None:
            continue
        wf, x0, t0 = g
        if UNSHARE:
            wf = unshare(wf, XT1, XT2)
        amax = wf.max(axis=1)
        hit = np.where(amax >= THR_HIT)[0]
        if len(hit) < MIN_STRIPS:
            continue
        # largest contiguous-ish cluster (gap > 2.0 mm breaks)
        breaks = np.where(np.diff(hit) * PITCH > 2.0)[0]
        groups = np.split(hit, breaks + 1)
        cl = max(groups, key=len)
        if len(cl) < MIN_STRIPS:
            continue
        n_done += 1
        pos = cl * PITCH
        amp = amax[cl]
        tcfd = np.array([cfd_time(wf[s])[0] for s in cl])
        # hits-like per-strip time: same CFD here stands in for the pulse fit
        okt = np.isfinite(tcfd)
        if okt.sum() < MIN_STRIPS:
            continue

        # (1) production-like: anchored at earliest, amplitude-weighted
        i0 = np.nanargmin(np.where(okt, tcfd, np.inf))
        dx, dtt = pos - pos[i0], tcfd - tcfd[i0]
        w = np.where(okt, amp, 0.0)
        den = np.sum(w * dx * dx)
        if den > 0:
            m = np.sum(w * np.where(okt, dx * dtt, 0.0)) / den
            if m != 0:
                S_prod.append(1000.0 / m)
                tans.append(tan_th)
        # (2) CFD core OLS
        mcore = (amp >= CORE_FRAC * amp.max()) & okt
        if mcore.sum() >= 3 and np.ptp(pos[mcore]) > 0:
            m = np.polyfit(pos[mcore], tcfd[mcore], 1)[0]
            if m != 0:
                S_cfd.append(1000.0 / m)
        else:
            S_cfd.append(np.nan)
        # geometry ingredients
        atans.append(abs(tan_th))
        exts.append(np.ptp(pos))
        spans.append(np.nanmax(tcfd[okt]) - np.nanmin(tcfd[okt]))
        if mcore.sum() >= 3:
            exts_core.append(np.ptp(pos[mcore]))
            spans_core.append(np.ptp(tcfd[mcore]))
        else:
            exts_core.append(np.nan); spans_core.append(np.nan)
        if abs(tan_th) < 0.05:
            floors.append(np.ptp(pos) + PITCH)

    tans = np.array(tans)
    atans = np.array(atans)

    def ridge(S):
        S = np.array(S)
        vs = []
        n = min(len(S), len(tans))
        for lo, hi in [(tan_lo, tan_hi), (-tan_hi, -tan_lo)]:
            m = (tans[:n] > lo) & (tans[:n] < hi) & np.isfinite(S[:n])
            if m.sum() < 50:
                continue
            x, y = tans[:n][m], S[:n][m]
            for _ in range(4):
                p = np.polyfit(x, y, 1)
                r = y - np.polyval(p, x)
                s = 1.4826 * np.median(np.abs(r - np.median(r)))
                keep = np.abs(r - np.median(r)) < 3 * s
                x, y = x[keep], y[keep]
            vs.append(np.polyfit(x, y, 1)[0])
        return np.mean(vs) if vs else np.nan

    # geometry estimator: extent-vs-tan slope over median profile / T_sat
    def geom(ext, span):
        ext, span = np.asarray(ext, float), np.asarray(span, float)
        ctr, med = [], []
        for b0 in np.arange(0.06, 0.44, 0.04):
            m = (atans >= b0) & (atans < b0 + 0.04) & np.isfinite(ext)
            if m.sum() >= 30:
                ctr.append(b0 + 0.02); med.append(np.median(ext[m]))
        if len(ctr) < 4:
            return np.nan, np.nan, np.nan
        slope = np.polyfit(ctr, med, 1)[0]
        msat = (atans > np.tan(np.radians(10))) & np.isfinite(span)
        tsat = np.median(span[msat])
        return slope * 1000.0 / tsat, slope, tsat

    v_geom_full, zs_full, ts_full = geom(exts, spans)
    v_geom_core, zs_core, ts_core = geom(exts_core, spans_core)
    return dict(v_true=v_true,
                v_prod=ridge(S_prod), v_cfd_core=ridge(S_cfd),
                v_geom_full=v_geom_full, z_slope_full=zs_full, t_sat_full=ts_full,
                v_geom_core=v_geom_core, z_slope_core=zs_core, t_sat_core=ts_core,
                floor_mm=np.median(floors) if floors else np.nan)


def main():
    global LAMBDA_ATT, XT1, XT2, DT_DIFF, UNSHARE
    UNSHARE = '--unshare' in sys.argv
    vt = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--vtrue=')), None)
    nev = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--nev=')), 4000)
    LAMBDA_ATT = next((float(a.split('=')[1]) for a in sys.argv
                       if a.startswith('--lam=')), LAMBDA_ATT)
    XT1 = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--xt1=')), XT1)
    XT2 = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--xt2=')), XT2)
    DT_DIFF = next((float(a.split('=')[1]) for a in sys.argv
                    if a.startswith('--dt=')), DT_DIFF)
    vlist = [vt] if vt else [28.0, 31.0, 34.0, 37.0]
    print(f'config: lam={LAMBDA_ATT} xt1={XT1} xt2={XT2} dt={DT_DIFF} '
          f'unshare={UNSHARE}')
    print(f'{"v_true":>7} {"v_prod":>7} {"v_cfd":>7} {"v_geo_f":>8} {"v_geo_c":>8} '
          f'{"z_slp_f":>8} {"T_sat_f":>8} {"floor":>6}')
    for v in vlist:
        r = analyse_events(v, nev)
        print(f'{r["v_true"]:7.1f} {r["v_prod"]:7.2f} {r["v_cfd_core"]:7.2f} '
              f'{r["v_geom_full"]:8.2f} {r["v_geom_core"]:8.2f} '
              f'{r["z_slope_full"]:8.1f} {r["t_sat_full"]:8.0f} {r["floor_mm"]:6.1f}')


if __name__ == '__main__':
    main()
