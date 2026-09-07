#!/usr/bin/env python3
"""
kernel_lib.py -- shared machinery for the angled-mount kernel measurement.

The question: does the charge-spreading kernel measured on a FLAT det4 still
hold when the same detector, same gas, same resist voltage and same zero
suppression is rotated to 25.64 deg?  And does its drift ladder give a second
lever on diffusion?

The dataset that makes this an A/B rather than a comparison across campaigns:
run_63 contains BOTH mounts, 90 minutes apart, either side of one H4 TAX
access (`FLAT_CF4_RUN63.md`).  Same run number, same gas bottle, resist held at
769.8 V throughout, ZS at 4 sigma in both.  Only the mount angle and the drift
field differ.

    run63_rot25  operating_00/_01   25.64 deg   drift 425 / 325 V   (d425, d325)
    run63_flat   operating_02/_03   flat        drift 700 V         (flat700)

WHICH VIEW IS TILTED, and how we know.  Not from the alignment matrix -- det4
is rolled +90.2 deg, so the strips called "x" in `det4_sps_map` measure the
detector's Y coordinate and the roll makes the row norms of `A` misleading.
Measured instead, from the data: fit peak-time against strip position event by
event and look at the SIGNED median slope.  A tilted view has a coherent drift
ladder and a nonzero signed slope; a normal-incidence view has slopes that
scatter about zero.

    flat700   x -4.4 ns/mm    y  -0.4 ns/mm     <- neither, as expected
    25.64 deg x -0.5 ns/mm    y -198.2 ns/mm    <- the Y VIEW is the ladder

-198 ns/mm implies v * tan(theta) = 5.05 um/ns, i.e. v = 10.5 um/ns at
108-142 V/cm, which is where this wet CF4 mixture should be
(`RAW_RUN71_REANALYSIS` measured 13-15 um/ns at 233 V/cm).  The ladder is real.

So:
  * the **X view is at normal incidence in BOTH mounts** -> the kernel A/B.
    Any change there is mount or drift field, never track geometry.
  * the **Y view is the 25.64 deg ladder** -> the geometry lever, and a
    depth-resolved handle, because at an angle depth maps onto STRIP.

CENSORING.  Both arms are zero-suppressed at 4 sigma.  `TWOGAS_HEADON` F3 sized
this: at identical conditions a ZS arm reads ~6 % low on the +-1 PEAK ratio and
~18 % low on the +-1 AREA ratio against RAW.  Every comparison here is
ZS-to-ZS at the same threshold, which is the only way it is allowed to be done.
Absolute numbers are not comparable to the RAW arms; the flat-vs-angled
DIFFERENCE is.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = '/home/dylan/PycharmProjects/nTof_x17'
for _p in (os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment'),
           os.path.join(REPO, 'sps_beam_test_26', 'analysis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from det4_sps_map import POSITION_MM, VIEW, PITCH_MM        # noqa: E402

STAGE = '/media/dylan/data/x17/sps_run53_det4_check/staging/run_63/'
SNS = 60.0
NSAMP = 64
GAP_MM = 30.0                    # det4 drift gap
TILT_DEG = 25.64

ARMS = {
    'flat700':  dict(f='wf_run63_flat.npz',      plateau='flat700',
                     drift_V=700.4, mount='flat',     tilted=None),
    'rot_d425': dict(f='wf_run63_operating.npz', plateau='d425',
                     drift_V=425.2, mount='25.64 deg', tilted='y'),
    'rot_d325': dict(f='wf_run63_operating.npz', plateau='d325',
                     drift_V=325.1, mount='25.64 deg', tilted='y'),
    'rot_d225': dict(f='wf_run63_operating.npz', plateau='d225',
                     drift_V=225.0, mount='25.64 deg', tilted='y'),
}


def load_arm(name):
    """Dense per-event waveforms for one arm.

    Returns dict with, per view, a list of (positions, W) where W is
    (n_strip, NSAMP) with zero-suppressed samples filled as 0 -- filling with
    0 rather than dropping is what keeps a weak neighbour from being silently
    promoted by survivorship (the `charge_sharing` note's own rule).
    """
    a = ARMS[name]
    z = np.load(STAGE + a['f'], allow_pickle=True)
    ev, ch, samp, amp = z['ev'], z['ch'], z['samp'], z['amp']
    plat = z['ev_plateau']
    keep_ev = np.flatnonzero(plat == a['plateau'])
    m = np.isin(ev, keep_ev)
    ev, ch, samp, amp = ev[m], ch[m], samp[m], amp[m]
    order = np.lexsort((samp, ch, ev))
    ev, ch, samp, amp = ev[order], ch[order], samp[order], amp[order]

    out = {'x': {}, 'y': {}, 'meta': dict(a), 'pX': z['ev_pX'], 'pY': z['ev_pY']}
    # split into per-event blocks
    bnd = np.r_[0, np.flatnonzero(np.diff(ev)) + 1, len(ev)]
    for i in range(len(bnd) - 1):
        s, e = bnd[i], bnd[i + 1]
        eid = int(ev[s])
        cc, ss, aa = ch[s:e], samp[s:e], amp[s:e]
        for v in ('x', 'y'):
            k = VIEW[cc] == v
            if k.sum() < 3:
                continue
            c2, s2, a2 = cc[k], ss[k], aa[k]
            uch, inv = np.unique(c2, return_inverse=True)
            W = np.zeros((len(uch), NSAMP), np.float32)
            W[inv, s2] = a2
            pos = POSITION_MM[uch]
            o = np.argsort(pos)
            out[v][eid] = (pos[o], W[o], uch[o])
    return out


def ladder_slope(events, min_strips=5, amp_min=60.0):
    """Signed median dt/dx over events -- the tilted-view discriminator."""
    sl = []
    for pos, W, _ in events.values():
        pk = W.max(axis=1)
        k = pk > amp_min
        if k.sum() < min_strips or np.ptp(pos[k]) < 2:
            continue
        t = W[k].argmax(axis=1) * SNS
        sl.append(np.polyfit(pos[k], t, 1)[0])
    return (float(np.median(sl)) if sl else np.nan,
            float(np.median(np.abs(sl))) if sl else np.nan, len(sl))


def parabolic_peak(w, i):
    if i <= 0 or i >= len(w) - 1:
        return float(i)
    a, b, c = float(w[i - 1]), float(w[i]), float(w[i + 1])
    den = a - 2 * b + c
    return i + (0.5 * (a - c) / den if den < 0 else 0.0)


def neighbour_stack(events, q0lo=200.0, q0hi=3000.0, nrel=14,
                    require_adjacent=True):
    """The campaign's peak-aligned neighbour estimator.

    Per event: the leading strip (largest peak) inside the amplitude gate, then
    its physical +-1, +-2 neighbours -- selected by POSITION, never by channel
    number, because det4's inverted connectors put physically adjacent strips
    127 FEU channels apart (`det4_sps_map` docstring).

    Returns per-offset median peak-aligned stacks (normalised to the leading
    strip's peak) plus the event-wise +-1 peak-time shift.  Absent neighbours
    count as an all-zero trace, and the detection fraction is reported so the
    censoring is visible rather than hidden.
    """
    aligned, shifts, detn, detd = {}, {1: [], -1: []}, {}, {}
    widths = []
    n_used = 0
    for pos, W, _ in events.values():
        pk = W.max(axis=1)
        lead = int(np.argmax(pk))
        q0 = pk[lead]
        if not (q0lo <= q0 <= q0hi):
            continue
        ipk = int(np.argmax(W[lead]))
        if ipk < 3 or ipk > NSAMP - 4:
            continue
        n_used += 1
        c_par = parabolic_peak(W[lead], ipk)
        cols = np.arange(ipk - nrel, ipk + nrel + 1)
        ok = (cols >= 0) & (cols < NSAMP)
        for d in (-3, -2, -1, 0, 1, 2, 3):
            want = pos[lead] + d * PITCH_MM
            j = int(np.argmin(np.abs(pos - want)))
            hit = abs(pos[j] - want) < 0.4 * PITCH_MM
            detd[d] = detd.get(d, 0) + 1
            row = np.zeros(2 * nrel + 1, np.float64)
            if hit:
                detn[d] = detn.get(d, 0) + 1
                row[ok] = W[j][cols[ok]] / q0
                if abs(d) == 1 and require_adjacent:
                    jp = int(np.argmax(W[j]))
                    # kept per SIDE.  At normal incidence both sides are the
                    # same +60 ns RC delay; on a drift ladder the two sides are
                    # ANTISYMMETRIC (one rung earlier, one later), so the
                    # +1/-1 shift pair is itself a geometry discriminator and
                    # pooling the two signs would average it away.
                    shifts[d].append((parabolic_peak(W[j], jp) - c_par) * SNS)
            aligned.setdefault(d, []).append(row)
        widths.append(int((pk > 0.2 * q0).sum()))

    out = {'n_events': n_used,
           'shift_p1_ns': (float(np.median(shifts[1])) if shifts[1] else np.nan),
           'shift_m1_ns': (float(np.median(shifts[-1])) if shifts[-1] else np.nan),
           'width_20pct': float(np.median(widths)) if widths else np.nan,
           'n_shift': len(shifts[1]) + len(shifts[-1])}
    out['pm1_shift_ns'] = float(np.nanmean([out['shift_p1_ns'],
                                            out['shift_m1_ns']]))
    out['shift_asym_ns'] = float(out['shift_p1_ns'] - out['shift_m1_ns'])
    # 20 %-trimmed mean, not the median: with absent strips entered as zeros
    # (which is the only way to keep a weak neighbour from being promoted by
    # survivorship) a +-2 detection fraction of ~0.33 drives the MEDIAN to
    # exactly zero and destroys the observable.  trim20 is also what the
    # campaign's own stacks use, so these numbers stay comparable.
    def trim20(a):
        a = np.sort(np.asarray(a, float), axis=0)
        k = int(round(0.10 * len(a)))
        return a[k:len(a) - k].mean(axis=0) if len(a) - 2 * k > 0 else a.mean(axis=0)

    st = {d: trim20(v) for d, v in aligned.items() if len(v) >= 20}
    if 0 in st:
        a0, p0 = np.sum(st[0]), np.max(st[0])
        for d in sorted(st):
            out[f'pk_{d:+d}'] = float(np.max(st[d]) / p0)
            out[f'area_{d:+d}'] = float(np.sum(st[d]) / a0)
            out[f'detfrac_{d:+d}'] = float(detn.get(d, 0) / max(detd.get(d, 1), 1))
    out['stacks'] = {str(d): st[d].tolist() for d in sorted(st)}
    out['t_rel_ns'] = ((np.arange(2 * nrel + 1) - nrel) * SNS).tolist()
    return out


def sym(o, key, d=1):
    """Symmetrised +-d observable."""
    a, b = o.get(f'{key}_+{d}'), o.get(f'{key}_-{d}')
    v = [q for q in (a, b) if q is not None and np.isfinite(q)]
    return float(np.mean(v)) if v else np.nan


# --------------------------------------------------------------------------- #
# depth-resolved lateral width -- the diffusion / film separation
# --------------------------------------------------------------------------- #
def telescope_map(arm_data, view, min_ev=200, q0lo=200.0):
    """Fit the view's strip coordinate against the telescope prediction.

    Not assumed: det4 is rolled +90 deg, so which detector axis a view measures
    is a fit, not a convention.  For the X view it comes out
    pred = 1.001*pX + const with a 0.57 mm MAD -- i.e. the X strips measure pX
    one-for-one.

    Two things this has to get right or the residual blows up to 5-9 mm:
      * **single cluster only.** 7.6 % of events carry a second, separated
        group of strips (beam pile-up in the H4 spill -- the same population
        `RAW_RUN71_REANALYSIS` §1 found).  A charge-weighted centroid over both
        lands between them.
      * **leading strip, not the centroid**, as the position estimator, for the
        same reason.
    """
    P, Q, C = [], [], []
    for eid, (pos, W, _) in arm_data[view].items():
        pk = W.max(axis=1)
        i = int(np.argmax(pk))
        if pk[i] < q0lo:
            continue
        grp = pos[pk > 0.2 * pk[i]]
        if len(grp) > 1 and (np.diff(np.sort(grp)) > 1.6).any():
            continue                                   # >1 cluster: skip
        px, py = arm_data['pX'][eid], arm_data['pY'][eid]
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        P.append(px); Q.append(py); C.append(pos[i])
    if len(C) < min_ev:
        return None
    M = np.column_stack([P, Q, np.ones(len(C))])
    C = np.array(C)
    c = np.linalg.lstsq(M, C, rcond=None)[0]
    for _ in range(3):
        r = C - M @ c
        s_ = 1.4826 * np.median(np.abs(r - np.median(r)))
        k = np.abs(r - np.median(r)) < 3 * s_
        if k.sum() < min_ev:
            break
        c = np.linalg.lstsq(M[k], C[k], rcond=None)[0]
    r = C - M @ c
    return c, float(1.4826 * np.median(np.abs(r - np.median(r)))), len(C)


def width_vs_time(arm_data, view, coef, q0lo=200.0, q0hi=3000.0, max_mm=3.9):
    """RMS lateral spread of the charge in each 60 ns sample, about the
    TELESCOPE-predicted impact point.

    At normal incidence the whole ionisation column sits at one transverse
    position, so sample index maps to drift DEPTH and this is sigma(z) --
    diffusion, plus whatever the film has already spread, plus the avalanche.
    Referencing to the external telescope rather than to det4's own centroid is
    what stops it being circular.

    Returned per sample: charge-weighted rms, the summed charge, and how many
    strips contributed (the zero-suppression censoring, made visible).
    """
    num = np.zeros(NSAMP)
    den = np.zeros(NSAMP)
    qtot = np.zeros(NSAMP)
    nstrip = np.zeros(NSAMP)
    nev = 0
    for eid, (pos, W, _) in arm_data[view].items():
        pk = W.max(axis=1)
        i = int(np.argmax(pk))
        q0 = pk[i]
        if not (q0lo <= q0 <= q0hi):
            continue
        grp = pos[pk > 0.2 * q0]
        if len(grp) > 1 and (np.diff(np.sort(grp)) > 1.6).any():
            continue                                   # pile-up: skip
        px, py = arm_data['pX'][eid], arm_data['pY'][eid]
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        pred = coef[0] * px + coef[1] * py + coef[2]
        d = pos - pred
        k = np.abs(d) <= max_mm
        if k.sum() < 2:
            continue
        nev += 1
        A = np.clip(W[k], 0, None)                 # (n_strip, NSAMP)
        dd = d[k][:, None]
        num += (A * dd ** 2).sum(axis=0)
        den += A.sum(axis=0)
        qtot += A.sum(axis=0)
        nstrip += (A > 0).sum(axis=0)
    sig = np.sqrt(np.divide(num, den, out=np.full(NSAMP, np.nan), where=den > 0))
    return dict(sigma_mm=sig.tolist(), charge=qtot.tolist(),
                nstrip=(nstrip / max(nev, 1)).tolist(), n_events=nev,
                t_ns=(np.arange(NSAMP) * SNS).tolist())
