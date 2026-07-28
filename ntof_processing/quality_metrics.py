#!/usr/bin/env python3
"""Timing and amplitude QUALITY of a reprocessed file -- not just how many hits.

Hit count and matcher efficiency say how much signal a UserInput recovers. They
say nothing about whether those hits are well measured, and a change that
resolves more pileup can do so by reporting worse-determined pulses. These are
the two quality axes, both computed from the file alone with no external truth.

TIMING
  T1  same-bar top<->bottom dt.  Each wall bar is read out at both ends
      (detn = 2g+1 and 2g+2 for bar g), so a through-going particle appears in
      both. The width of that dt peak is the wall's own timing resolution,
      convolved with the transit-time spread along the bar -- the same physics
      in every variant, so the comparison is fair. ~2 M pairs per partial.
  T2  wall<->plastic prompt coincidence dt, plastic required to be MIP-like
      (amp > PLASTIC_MIP). The core width is the combined resolution of the two
      legs. This is the "do the wall and plastic timings agree" metric.

AMPLITUDE
  A1  MIP peak in sqrt(amp_top * amp_bottom) of a coincident bar, MIP-tagged by
      a large plastic hit. The geometric mean of the two ends cancels the
      exponential attenuation along the bar, so it is position-independent.
      Reported as peak position and FWHM/peak -- narrower is better measured.

EVERY number here is ACCIDENTAL-SUBTRACTED. Both detectors run at high rate, so
"nearest hit within a window" is dominated by chance coincidences: without
subtraction T2 comes out ~39 ns wide and A1 shows no MIP peak at all, both of
which are the accidental background rather than the physics. The subtraction is
the standard one -- take the same distribution from an off-time sideband and
subtract it -- so what is quoted is the genuine coincidence.

Widths come from the FWHM of the background-subtracted peak (/2.355), which
needs no fit to converge.

    python quality_metrics.py label=a.root,b.root [label2=...]
"""
import sys

import numpy as np
import uproot

ARMS = 'ABCD'
LATE_NS = 1e6              # well clear of the flash and its recovery
PLASTIC_MIP = 1000.0       # amp cut that makes a plastic hit a MIP tag
SIDE_LO, SIDE_HI = 300.0, 1300.0    # off-time sideband, same width as the core
CORE = 300.0
COINC_NS = 20.0            # prompt window for the MIP tag


def peak_width(d, span=CORE, nbins=240):
    """Centre and sigma of an accidental-subtracted dt peak."""
    d = d[np.isfinite(d)]
    if d.size < 200:
        return np.nan, np.nan, 0
    h, e = np.histogram(d, bins=nbins, range=(-span, span))
    c = 0.5 * (e[1:] + e[:-1])
    # flat accidental level from the outer thirds
    edge = np.abs(c) > 0.66 * span
    bg = np.median(h[edge]) if edge.any() else 0.0
    hs = h.astype(float) - bg
    if hs.max() <= 0:
        return np.nan, np.nan, int(hs.sum())
    half = hs.max() / 2
    above = np.flatnonzero(hs > half)
    if above.size < 2:
        return float(c[hs.argmax()]), np.nan, int(hs.sum())
    fwhm = float(c[above[-1]] - c[above[0]])
    return float(c[hs.argmax()]), fwhm / 2.355, int(max(hs.sum(), 0))


def load(files, tree, branches):
    out = {}
    for p in files:
        f = uproot.open(p)
        if tree not in {k.split(';')[0] for k in f.keys()}:
            continue
        a = f[tree].arrays(branches, library='np')
        for k, v in a.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()} if out else None


def nearest_dt(t_a, t_b):
    if t_a.size == 0 or t_b.size == 0:
        return np.array([]), np.array([], dtype=int)
    j = np.searchsorted(t_b, t_a)
    j0 = np.clip(j - 1, 0, t_b.size - 1)
    j1 = np.clip(j, 0, t_b.size - 1)
    d0, d1 = t_a - t_b[j0], t_a - t_b[j1]
    take = np.abs(d0) <= np.abs(d1)
    return np.where(take, d0, d1), np.where(take, j0, j1)


def analyse(label, files):
    print(f'\n{"=" * 76}\n{label}')
    res = {}
    tb_all, wp_all = [], []
    mip_on, mip_off = [], []

    for arm in ARMS:
        w = load(files, f'WAL{arm}',
                 ['BunchNumber', 'detn', 'tof', 'tflash', 'amp'])
        p = load(files, f'PSS{arm}', ['BunchNumber', 'tof', 'tflash', 'amp'])
        if w is None:
            continue
        wt_all = w['tof'] - w['tflash']
        m = wt_all > LATE_NS
        wt, wd, wb, wa = wt_all[m], w['detn'][m], w['BunchNumber'][m], w['amp'][m]

        pt = pb = None
        if p is not None:
            t = p['tof'] - p['tflash']
            mp = (t > LATE_NS) & (p['amp'] > PLASTIC_MIP)
            pt, pb = t[mp], p['BunchNumber'][mp]

        for b in np.unique(wb):
            mb = wb == b
            tb_, db_, ab_ = wt[mb], wd[mb], wa[mb]
            tp = np.sort(pt[pb == b]) if pt is not None else np.array([])
            for g in range(4):
                st = np.argsort(tb_[db_ == 2 * g + 1])
                sb = np.argsort(tb_[db_ == 2 * g + 2])
                ot, at = tb_[db_ == 2 * g + 1][st], ab_[db_ == 2 * g + 1][st]
                ob, ab2 = tb_[db_ == 2 * g + 2][sb], ab_[db_ == 2 * g + 2][sb]
                if ot.size == 0 or ob.size == 0:
                    continue
                d, j = nearest_dt(ot, ob)
                tb_all.append(d)
                # bar objects: the two ends within the T1 core
                sel = np.abs(d) < 20
                if not sel.any() or tp.size == 0:
                    continue
                tmean = 0.5 * (ot[sel] + ob[j[sel]])
                gmean = np.sqrt(np.abs(at[sel] * ab2[j[sel]]))
                o = np.argsort(tmean)
                tmean, gmean = tmean[o], gmean[o]
                dp, _ = nearest_dt(tmean, tp)
                wp_all.append(dp)
                mip_on.append(gmean[np.abs(dp) < COINC_NS])
                off = (np.abs(dp) > SIDE_LO) & (np.abs(dp) < SIDE_LO + 2 * COINC_NS)
                mip_off.append(gmean[off])

    tb = np.concatenate(tb_all) if tb_all else np.array([])
    wp = np.concatenate(wp_all) if wp_all else np.array([])

    c, s, n = peak_width(tb)
    res['T1_tb_sigma'] = s
    print(f'  T1 wall top<->bottom dt : centre {c:7.2f} ns  sigma {s:6.2f} ns  '
          f'(peak n={n:,})')
    c, s, n = peak_width(wp)
    res['T2_wp_sigma'] = s
    res['T2_wp_centre'] = c
    print(f'  T2 wall<->plastic    dt : centre {c:7.2f} ns  sigma {s:6.2f} ns  '
          f'(peak n={n:,})   [plastic amp > {PLASTIC_MIP:.0f}]')

    on = np.concatenate(mip_on) if mip_on else np.array([])
    off = np.concatenate(mip_off) if mip_off else np.array([])
    if on.size > 500:
        hi = np.percentile(on, 99.5)
        bins = np.linspace(0, hi, 120)
        ho, _ = np.histogram(on, bins=bins)
        hf, _ = np.histogram(off, bins=bins)
        # the off-time window is 2x as wide as the prompt one
        hs = ho.astype(float) - 0.5 * hf
        ctr = 0.5 * (bins[1:] + bins[:-1])
        # ignore the first bins, where the amplitude threshold shapes the spectrum
        k = max(3, int(0.05 * len(ctr)))
        pk_i = k + int(np.argmax(hs[k:]))
        pk = float(ctr[pk_i])
        half = hs[pk_i] / 2
        above = np.flatnonzero(hs > half)
        fwhm = float(ctr[above[-1]] - ctr[above[0]]) if above.size > 1 else np.nan
        res['A1_mip_peak'] = pk
        res['A1_mip_relwidth'] = fwhm / pk if pk else np.nan
        res['A1_mip_n'] = float(hs.sum())
        print(f'  A1 MIP sqrt(top*bot)    : peak {pk:7.0f} ADC  FWHM {fwhm:7.0f}  '
              f'FWHM/peak {fwhm / pk:5.2f}   (prompt {on.size:,}, '
              f'accidental {0.5 * off.size:,.0f})')
    else:
        print(f'  A1 MIP: too few coincident bars ({on.size})')
    return res


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    out = {}
    for spec in sys.argv[1:]:
        lab, files = spec.split('=', 1)
        out[lab] = analyse(lab, files.split(','))
    if len(out) > 1:
        labs = list(out)
        print(f'\n{"=" * 76}\nrelative to {labs[0]}')
        print('  (lower sigma = better timing; lower FWHM/peak = better amplitude)')
        for k in ('T1_tb_sigma', 'T2_wp_sigma', 'A1_mip_relwidth',
                  'A1_mip_peak', 'A1_mip_n'):
            base = out[labs[0]].get(k)
            if base is None or not np.isfinite(base) or base == 0:
                continue
            row = '   '.join(
                f'{l}: {out[l].get(k, float("nan")):8.2f} '
                f'({out[l].get(k, float("nan")) / base - 1:+5.1%})'
                for l in labs[1:])
            print(f'  {k:18s} base {base:8.2f}   {row}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
