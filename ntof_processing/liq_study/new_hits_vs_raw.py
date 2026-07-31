#!/usr/bin/env python3
"""T4 per-hit: DOES NOT WORK. Kept as the record of why, and as the diagnostic.

*** Do not quote numbers from this script. Its control fails. ***

The intent was to classify each v12-only liquid hit against the raw waveform.
Two observables were tried -- a joint template fit, and a model-free local rise
-- and BOTH score hits that v11 and v12 agree on (real by construction) no
better than the new hits: 2-4 % of matched controls clear the 5-sigma bar. When
a known-good control fails, the method is wrong, not the data.

The underlying obstacle is real and is the useful output of this file: **PSA
`tof` and the raw sample index do not correspond per hit.** What is solid is
that the bunch identification is right and that there is a stable per-detector
lag (`calibrate_lag`, +21 to +29 ns). What is not is that only ~20 % of
unambiguous large raw pulses have a hit at that lag, and single-pulse overlays
show hits on flat baseline. Until someone establishes what `tof` marks on a
fitted pulse, no per-hit raw-vs-PSA classification from this repo means
anything.

Use `raw_pulse_budget.py` instead: it answers the same question by counting,
which only needs a hit to land in the right ~1000 ns block.

The rest of this docstring describes the method as designed, for whoever picks
it up once the alignment is understood.

The cheap proxies in `new_hits.py` come out ambiguous, and they were always
going to: 95 % of the new hits are splits of existing pulses, so the
time-of-flight profile of the new population is inherited from the real pulses
it was split off and cannot discriminate. The amplitude spectrum leans the other
way -- the new hits crowd the amp-50 elimination cut 3-4x more than pre-existing
ones -- which is the fake-population signature, but is also exactly what a
genuinely resolved shoulder on a big pulse's tail looks like.

So this goes to the waveforms. Rather than classify a few hundred overlays by
eye, it asks the question the eye is trying to answer, quantitatively:

    GIVEN the pulses both processings already agree on, does adding a pulse at
    the new hit's time explain the raw samples significantly better?

Method, per new hit:
  * pull the raw block containing it, baseline-subtract, sign-flip;
  * take every v12 hit within +-WIN ns as the pulse list, build a design matrix
    of the measured template shifted to each of those times, and solve the
    JOINT linear least squares for all amplitudes at once;
  * read the new pulse's own amplitude and its standard error from the
    covariance, sigma_k = sigma_noise * sqrt((A^T A)^-1_kk).

`significance = amplitude / sigma_k` is the answer. A real pulse (class a or b)
has a positive, significant amplitude even after its neighbours are given every
chance to absorb it. A finder re-firing on an existing pulse's smooth tail
(class c) does not: with the neighbour in the fit there is nothing left for it,
so its amplitude collapses toward zero or goes negative.

The joint fit matters. Fitting the new pulse alone would credit it with its
neighbour's charge and call everything real -- which is how a split gets
mistaken for a recovery.

CONTROL, and the reason to trust the number: the identical test runs on hits
BOTH processings found. Those are real by construction, so their significance
distribution is the scale against which the new hits are read. If matched hits
did not pass, the method would be broken, not the hits.

Alignment is done by explicit index arithmetic, never by `correlate`, because
two separate bugs in this repo have been correlate-alignment bugs (REVIEW.md
Section 5).

    python new_hits_vs_raw.py <psa_v11.root> <psa_v12.root> <raw.bin> [raw.bin ...]
"""
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

TREES = ('LIQA', 'LIQB', 'LIQC', 'LIQD')
T_LO, T_HI = 1_000_000, 18_000_000
PRE, POST = 20, 200           # template: peak at index PRE
WIN = 300.0                   # ns of neighbours included in the joint fit
TOL = 3.0                     # ns, same-pulse tolerance (as in new_hits.py)
NEAR = 150.0                  # ns, split-vs-recovery boundary
PAD = 32768                   # zero-suppression fill
MAX_PER_TREE = 4000           # sampled new hits per tree
SIG_REAL, SIG_FAKE = 5.0, 3.0


def build_template(blocks):
    rows = []
    for s in blocks:
        base = np.median(np.concatenate([s[:50], s[-50:]]))
        d = -(s - base)
        i = int(np.argmax(d))
        a = d[i]
        if a < 300 or i < PRE + 5 or i + POST + 5 > len(d):
            continue
        if (s > 60000).any():                 # under-range wrap: shape is junk
            continue
        rms = np.std(s[:50] - np.median(s[:50]))
        before = d[:max(0, i - 8)]
        if before.size and before.max() > max(0.10 * a, 5 * rms):
            continue
        after = d[i + 8:]
        if after.size and (after - np.minimum.accumulate(after)).max() > max(0.06 * a, 5 * rms):
            continue
        rows.append(d[i - PRE:i + POST] / a)
    if len(rows) < 30:
        return None
    m = np.median(np.array(rows), axis=0)
    return m / m.max()


def baseline_and_noise(s):
    """Robust baseline and sample noise of a ZERO-SUPPRESSED block.

    The obvious "median and std of the first 30 samples" is wrong here and gets
    the whole test wrong: zero suppression means a block STARTS on a pulse, so
    those 30 samples are signal. It over-estimates the noise by an order of
    magnitude, and then every hit -- including ones both processings found --
    looks insignificant.

    Baseline: these detectors are negative-going, so the quiet level is the HIGH
    side of the sample distribution, not the middle. Noise: the median absolute
    difference between neighbouring samples, which the pulses only contribute to
    during their rise and fall and so survives a block that is mostly pulse.
    """
    base = float(np.percentile(s, 90))
    dif = np.diff(s)
    mad = float(np.median(np.abs(dif - np.median(dif))))
    rms = 1.4826 * mad / np.sqrt(2.0)
    return base, rms


def calibrate_lag(blocks, tree, b_psa, t_psa, lags=np.arange(-40, 161)):
    """Sample lag between the PSA's reported `tof` and the raw pulse PEAK.

    They are NOT the same instant, and the difference is per detector: measured
    +28 ns on LIQA and +21 ns on LIQD. Assuming they coincide -- which is the
    obvious thing to do, since both are "ns within the acquisition window" --
    puts every raw lookup on the flat baseline in front of the pulse and makes
    real hits look like nothing is there. That is the third alignment bug of
    this shape in this repo (REVIEW.md Section 5), so it is measured here rather
    than assumed, from the data itself, every run.

    Measured by stacking the raw trace around every PSA hit and taking the lag
    of the peak of the average.
    """
    prof = np.zeros(len(lags))
    n = 0
    for (det, bunch), blks in blocks.items():
        if det != tree:
            continue
        s = slice(*np.searchsorted(b_psa, [bunch, bunch + 1]))
        tt = t_psa[s]
        for start, samp in blks:
            real = samp != PAD
            sr = samp[real]
            if len(sr) < 200:
                continue
            d = np.where(real, np.percentile(sr, 90) - samp, 0.0)
            sel = tt[(tt >= start - lags[0] + 5) &
                     (tt < start + len(samp) - lags[-1] - 5)]
            for t in sel:
                i = int(round(t)) - start
                prof += d[i + lags[0]:i + lags[-1] + 1]
                n += 1
    if n < 200:
        return None, n
    return int(lags[int(np.argmax(prof))]), n


def local_rise(d, rms, t, back=8, fwd=2):
    """How much does the waveform RISE into this hit, in units of baseline noise?

    This is classes (a)/(b) vs (c) of ../archive/PRE_SHIP_TESTS.md T4 stated as a number,
    and it is deliberately model-free and LOCAL, because the joint template fit
    that would otherwise answer it is ill-conditioned at 24 ns pulse spacing
    (REVIEW.md Section 7 predicted this; the control below measures it).

    A pulse arriving at t makes the trace climb from the trough between it and
    its predecessor up to its own peak. Anything sitting on the smooth decay of
    an earlier pulse does not climb at all: the trace is monotonically falling
    through t, so the trough IS the sample at t and the rise is ~0.

      (a) recovery : big rise, no pre-existing hit nearby
      (b) split    : big rise, pre-existing hit within 150 ns -- a real shoulder
      (c) fake     : no rise; the finder re-fired on a tail or on noise
    """
    i = int(round(t))
    n = len(d)
    if i - back < 0 or i + fwd + 1 > n:
        return np.nan
    peak = float(np.max(d[max(0, i - 1):i + fwd + 1]))
    trough = float(np.min(d[i - back:i + 1]))
    return (peak - trough) / rms


def joint_significance(d, rms, t_local, k, tmpl):
    """Significance of pulse k when ALL pulses in t_local are fitted together."""
    n = len(d)
    cols = []
    for t in t_local:
        c = np.zeros(n)
        lo = int(round(t)) - PRE
        a0, a1 = max(0, lo), min(n, lo + len(tmpl))
        if a1 <= a0:
            return np.nan, np.nan
        c[a0:a1] = tmpl[a0 - lo:a1 - lo]
        cols.append(c)
    A = np.column_stack(cols)
    if A.shape[1] >= n:
        return np.nan, np.nan
    G = A.T @ A
    try:
        Ginv = np.linalg.inv(G + 1e-9 * np.eye(len(G)) * np.trace(G) / len(G))
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    amp = Ginv @ (A.T @ d)
    var = float(Ginv[k, k])
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan
    return float(amp[k]), float(amp[k] / (rms * np.sqrt(var)))


def load_psa(path, tree):
    a = uproot.open(path)[tree].arrays(['BunchNumber', 'tof', 'amp'], library='np')
    m = (a['tof'] >= T_LO) & (a['tof'] < T_HI)
    b, t, amp = a['BunchNumber'][m], a['tof'][m].astype(float), np.abs(a['amp'][m])
    o = np.lexsort((t, b))
    return b[o], t[o], amp[o]


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    v11_p, v12_p = sys.argv[1], sys.argv[2]

    # ---- raw side: blocks keyed by (tree, bunch) -----------------------------
    blocks, cur = {}, None
    for raw in sys.argv[3:]:
        for _o, tag, _v, pay in iter_banks(raw):
            if tag == 'EVEH':
                cur = parse_eveh(pay)['words'][1]
            elif tag == 'ACQC' and cur is not None:
                det, _c, blks = parse_acqc(pay, with_samples=True)
                if det not in TREES:
                    continue
                for start, s in blks:
                    if T_LO <= start < T_HI and len(s) > 40:
                        blocks.setdefault((det, cur), []).append(
                            (int(start), s.astype(float)))
    have = sorted({b for _d, b in blocks})
    print(f'raw: {len(blocks)} (tree,bunch) block groups over bunches '
          f'{min(have) if have else "-"}-{max(have) if have else "-"}\n')

    rng = np.random.default_rng(20260729)
    grand = {}
    for tree in TREES:
        src = [s for (d, _b), v in blocks.items() if d == tree for _, s in v]
        tmpl = build_template(src)
        if tmpl is None:
            print(f'{tree}: too few clean pulses for a template')
            continue
        b11, t11, _a11 = load_psa(v11_p, tree)
        b12, t12, a12 = load_psa(v12_p, tree)
        lag, nlag = calibrate_lag(blocks, tree, b12, t12)
        if lag is None:
            print(f'{tree}: only {nlag} hits land in the raw chunks -- cannot '
                  f'calibrate the tof->peak lag')
            continue
        print(f'{tree}: tof -> raw peak lag = {lag:+d} ns (from {nlag} hits)')

        rows = {'new': [], 'ctl': []}
        for (det, bunch), blks in blocks.items():
            if det != tree:
                continue
            s11 = slice(*np.searchsorted(b11, [bunch, bunch + 1]))
            s12 = slice(*np.searchsorted(b12, [bunch, bunch + 1]))
            o11, o12 = t11[s11], t12[s12]
            amp12 = a12[s12]
            if o12.size == 0:
                continue
            if o11.size:
                j = np.clip(np.searchsorted(o11, o12), 0, o11.size - 1)
                j0 = np.clip(j - 1, 0, o11.size - 1)
                dmin = np.minimum(np.abs(o12 - o11[j]), np.abs(o12 - o11[j0]))
            else:
                dmin = np.full(o12.size, np.inf)
            is_new = dmin > TOL

            for start, s in blks:
                end = start + len(s)
                real = s != PAD
                if real.sum() < 40:
                    continue
                base, rms = baseline_and_noise(s[real])
                d = np.where(real, base - s, 0.0)
                if not np.isfinite(rms) or rms <= 0:
                    continue
                inblk = (o12 >= start) & (o12 < end)
                if not inblk.any():
                    continue
                tl_all = o12[inblk] - start
                newl = is_new[inblk]
                for k in np.flatnonzero(newl):
                    near = np.abs(tl_all - tl_all[k]) <= WIN
                    idx = np.flatnonzero(near)
                    kk = int(np.searchsorted(idx, k))   # k is always in idx
                    _amp, sig = joint_significance(d, rms, tl_all[near], kk, tmpl)
                    rise = local_rise(d, rms, tl_all[k] + lag)
                    if np.isfinite(rise):
                        gap = (np.min(np.abs(o11 - o12[inblk][k]))
                               if o11.size else np.inf)
                        rows['new'].append((rise, sig, gap,
                                            amp12[inblk][k]))
                # control: an equal number of MATCHED hits from the same blocks
                mat = np.flatnonzero(~newl)
                if mat.size:
                    for k in rng.choice(mat, size=min(len(mat),
                                                      max(1, int(newl.sum()))),
                                        replace=False):
                        near = np.abs(tl_all - tl_all[k]) <= WIN
                        idx = np.flatnonzero(near)
                        kk = int(np.searchsorted(idx, k))
                        _amp, sig = joint_significance(d, rms, tl_all[near], kk, tmpl)
                        rise = local_rise(d, rms, tl_all[k] + lag)
                        if np.isfinite(rise):
                            rows['ctl'].append((rise, sig, np.nan,
                                                amp12[inblk][k]))
            if len(rows['new']) > MAX_PER_TREE:
                break

        if len(rows['new']) < 100:
            print(f'{tree}: only {len(rows["new"])} new hits landed in the raw '
                  f'chunks -- not enough')
            continue
        new = np.array(rows['new'])
        ctl = np.array(rows['ctl']) if rows['ctl'] else np.empty((0, 4))
        rise_n, sig_n, gap_n = new[:, 0], new[:, 1], new[:, 2]
        print(f'=== {tree} ===   {len(new)} new hits, {len(ctl)} matched controls')
        print(f'  CONTROL, hits BOTH processings found (real by construction):')
        print(f'    local rise      : median {np.median(ctl[:, 0]):6.1f} sigma_noise, '
              f'{np.mean(ctl[:, 0] >= SIG_REAL):5.1%} above {SIG_REAL:.0f}')
        print(f'    joint-fit sig.  : median {np.median(ctl[:, 1]):6.1f}, '
              f'{np.mean(ctl[:, 1] >= SIG_REAL):5.1%} above {SIG_REAL:.0f}   '
              f'<- ill-conditioned, do not use')
        real = rise_n >= SIG_REAL
        fake = rise_n < SIG_FAKE
        print(f'  NEW hits, local rise: median {np.median(rise_n):.1f} sigma_noise')
        print(f'    (a)+(b) REAL   >={SIG_REAL:.0f} sigma : {real.mean():6.1%}')
        print(f'    ambiguous  {SIG_FAKE:.0f}-{SIG_REAL:.0f} sigma : '
              f'{np.mean(~real & ~fake):6.1%}')
        print(f'    (c) FAKE       < {SIG_FAKE:.0f} sigma : {fake.mean():6.1%}')
        sp = gap_n < NEAR
        for lab, m in (('    of which split   ', sp), ('    of which recovery', ~sp)):
            if m.sum() > 20:
                print(f'{lab} n={m.sum():6d}  real {np.mean(rise_n[m] >= SIG_REAL):6.1%}'
                      f'  fake {np.mean(rise_n[m] < SIG_FAKE):6.1%}')

        # THE test. An absolute rise threshold is not calibratable here -- even
        # hits both processings found only clear 5 sigma a few per cent of the
        # time, because most liquid pulses genuinely sit on a predecessor's
        # tail. What IS calibratable is the comparison: at the SAME reported
        # amplitude, does a new hit look like a matched hit in the raw data?
        # A fake sitting on a smooth tail has no rise at all, so a fake
        # population must sit BELOW the matched hits of the same amplitude.
        print('  MATCHED-AMPLITUDE COMPARISON (the test that is calibratable)')
        print(f'    {"amp bin":>14s} {"n new":>7s} {"n matched":>10s} '
              f'{"median rise: new":>17s} {"matched":>9s} {"ratio":>7s}')
        ratios, wts = [], []
        edges = [50, 70, 100, 150, 250, 500, 1e9]
        for lo, hi in zip(edges[:-1], edges[1:]):
            mn = (new[:, 3] >= lo) & (new[:, 3] < hi)
            mc = (ctl[:, 3] >= lo) & (ctl[:, 3] < hi)
            if mn.sum() < 30 or mc.sum() < 30:
                continue
            rn, rc = np.median(rise_n[mn]), np.median(ctl[mc, 0])
            print(f'    {lo:6.0f}-{hi:<7.0f} {mn.sum():7d} {mc.sum():10d} '
                  f'{rn:17.2f} {rc:9.2f} {rn / rc:7.2f}')
            ratios.append(rn / rc)
            wts.append(mn.sum())
        rat = float(np.average(ratios, weights=wts)) if ratios else np.nan
        print(f'    -> new hits carry {rat:.2f}x the local rise of matched hits '
              f'of the same amplitude')
        grand[tree] = (real.mean(), fake.mean(), len(new),
                       float(np.mean(ctl[:, 0] >= SIG_REAL)) if len(ctl) else np.nan,
                       rat)

    if grand:
        w = np.array([v[2] for v in grand.values()], float)
        r = np.array([v[0] for v in grand.values()])
        fk = np.array([v[1] for v in grand.values()])
        print(f'\n{"=" * 62}\nT4 verdict, yield-weighted over {int(w.sum())} '
              f'new hits')
        ct = np.array([v[3] for v in grand.values()])
        rt = np.array([v[4] for v in grand.values()])
        print(f'  absolute 5-sigma rate: new {np.average(r, weights=w):.1%} vs '
              f'matched-hit control {np.average(ct, weights=w):.1%}')
        print(f'    (the absolute threshold is NOT usable -- the control fails '
              f'it too)')
        print(f'\n  MATCHED-AMPLITUDE rise ratio, new / matched : '
              f'{np.average(rt, weights=w):.2f}')
        print( '    ~1.0  -> the new hits are indistinguishable from hits both '
               'processings\n            found, at the same amplitude: they are '
               'real')
        print( '    <<1.0 -> the new hits sit on smooth tails where matched hits '
               'do not:\n            they are finder re-fires and the STEP SIZE '
               'change should be reverted')
    return 0


if __name__ == '__main__':
    sys.exit(main())
