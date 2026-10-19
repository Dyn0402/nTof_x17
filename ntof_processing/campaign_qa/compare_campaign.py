#!/usr/bin/env python3
"""Compare the runs WE processed against the runs n_TOF processed themselves.

The comparison is only meaningful on bunches that had protons.  A partial can
be dominated by empty PS pulses -- PKUP amplitude 0, index PulseIntensity 0 --
and those bunches have no gamma flash, so tflash is 0 or garbage and the hit
rate collapses.  That is the beam, not the processing, and mixing them in makes
our runs look broken when they are not (run 224692's first partial is 75 %
empty).  So every number below is computed on BEAM bunches only, defined as
PKUP amplitude > 0 in that partial.

Per run, on beam bunches:
  rate      hits per 1e12 protons, per tree -- intensity-normalised, so runs at
            different beam intensity compare directly
  flash     modal tflash per tree, and the fraction of beam bunches more than
            150 ns off it.  The broken July processing sat at 37-85 % on PSS.
  offset    prompt-coincidence peak of large PSS hits and of LIQ hits against
            the same arm's wall hits, after removing each tree's modal tflash.
            The broken processing sat at -375/+25/-325/-325 ns on A/B/C/D.
  amp       median hit amplitude per tree, which catches a gain or template
            change that the rates would not show

Usage:
    python compare_campaign.py ours=<dir>[,<dir>...] official=<dir>[,...] \
                               [--partials=1] [--json=out.json]

Each <dir> is a run directory holding run<run>_NNNN.root partials.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import uproot

ARMS = 'ABCD'
TREES = ([f'WAL{a}' for a in ARMS] + [f'PSS{a}' for a in ARMS]
         + [f'LIQ{a}' for a in ARMS])
PLASTIC_MIP = 1000.0     # amp above which a plastic hit is a MIP-like tag
COINC = 300.0            # +-window for the prompt-coincidence search, ns
FLASH_TOL = 150.0        # a bunch is "off flash" beyond this, ns


def partials(d, npart):
    d = Path(d)
    ps = sorted(d.glob('run[0-9]*_[0-9]*.root'),
                key=lambda q: int(q.stem.split('_')[-1]))
    return ps[:npart]


def mode_ns(v, binw=10.0):
    v = v[np.isfinite(v) & (v > 0)]
    if not v.size:
        return np.nan
    h, e = np.histogram(v, bins=np.arange(0.0, 20000.0, binw))
    return float(e[h.argmax()] + binw / 2)


def nearest_dt(t_a, t_b):
    """For each t_a, the signed time to the nearest t_b (both sorted)."""
    if t_a.size == 0 or t_b.size == 0:
        return np.array([])
    j = np.searchsorted(t_b, t_a)
    j0, j1 = np.clip(j - 1, 0, t_b.size - 1), np.clip(j, 0, t_b.size - 1)
    d0, d1 = t_a - t_b[j0], t_a - t_b[j1]
    return np.where(np.abs(d0) <= np.abs(d1), d0, d1)


def peak_of(d, span=COINC, binw=2.0):
    """Accidental-subtracted peak position of a dt distribution."""
    d = d[np.isfinite(d) & (np.abs(d) < span)]
    if d.size < 100:
        return np.nan, 0
    h, e = np.histogram(d, bins=int(2 * span / binw), range=(-span, span))
    c = 0.5 * (e[1:] + e[:-1])
    bg = np.median(h[np.abs(c) > 0.66 * span])
    hs = h.astype(float) - bg
    if hs.max() <= 0:
        return np.nan, 0
    return float(c[hs.argmax()]), int(hs.max())


def analyse_run(run, files):
    """All metrics for one run, from its first `files` partials."""
    out = {'run': run, 'files': [str(f) for f in files]}
    beam_bunches, protons, n_bunch_tot = set(), 0.0, 0

    # ---- which bunches had protons, and how many
    for p in files:
        f = uproot.open(p)
        pk = f['PKUP'].arrays(['BunchNumber', 'amp', 'PulseIntensity'], library='np')
        good = pk['amp'] > 0
        beam_bunches.update(int(b) for b in pk['BunchNumber'][good])
        protons += float(np.nansum(pk['PulseIntensity'][good]))
        n_bunch_tot += pk['BunchNumber'].size
    out['bunches_total'] = n_bunch_tot
    out['bunches_beam'] = len(beam_bunches)
    out['frac_empty'] = 1.0 - len(beam_bunches) / max(n_bunch_tot, 1)
    out['protons_1e12'] = protons / 1e12
    out['mean_intensity'] = protons / max(len(beam_bunches), 1)

    if not beam_bunches:
        out['note'] = 'no beam bunches in the sampled partials'
        return out

    # ---- per tree, on beam bunches only
    data, trees = {}, {}
    for t in TREES:
        cols = {}
        for p in files:
            f = uproot.open(p)
            if t not in {k.split(';')[0] for k in f.keys()}:
                continue
            a = f[t].arrays(['BunchNumber', 'tflash', 'tof', 'amp'], library='np')
            keep = np.isin(a['BunchNumber'], list(beam_bunches))
            for k, v in a.items():
                cols.setdefault(k, []).append(v[keep])
        if not cols:
            continue
        a = {k: np.concatenate(v) for k, v in cols.items()}
        data[t] = a
        bn, tf = a['BunchNumber'], a['tflash']
        first = np.unique(bn, return_index=True)[1]
        tfb = tf[first]
        m = mode_ns(tfb)
        bad = float(np.mean(np.abs(tfb - m) > FLASH_TOL)) * 100 if np.isfinite(m) else np.nan
        trees[t] = {
            'hits': int(bn.size),
            'beam_bunches_seen': int(np.unique(bn).size),
            'hits_per_1e12p': bn.size / (protons / 1e12) if protons else np.nan,
            'tflash_mode_ns': m,
            'flash_bad_pct': bad,
            'tflash_zero_pct': float(np.mean(tfb <= 0)) * 100,
            'median_amp': float(np.nanmedian(a['amp'])) if bn.size else np.nan,
            'median_tof_us': float(np.nanmedian(a['tof'])) / 1e3 if bn.size else np.nan,
        }
    out['trees'] = trees

    # ---- cross-detector flash consistency, per arm
    # tof is already flash-subtracted by the PSA, but each tree carries its own
    # tflash; removing the tree's MODAL tflash puts every tree on one time base,
    # so a residual offset between PSS/LIQ and the wall is a real disagreement
    # about where the flash is.
    off = {}
    for arm in ARMS:
        w, ps, lq = data.get(f'WAL{arm}'), data.get(f'PSS{arm}'), data.get(f'LIQ{arm}')
        if w is None:
            continue
        mw = trees[f'WAL{arm}']['tflash_mode_ns']
        for tag, other in (('pss', ps), ('liq', lq)):
            if other is None:
                continue
            mo = trees[f'{tag.upper()}{arm}']['tflash_mode_ns']
            peaks, weights = [], []
            # bunch by bunch, so hits from different bunches never pair up
            common = np.intersect1d(np.unique(w['BunchNumber']),
                                    np.unique(other['BunchNumber']))
            dts = []
            for b in common[:40]:
                tw = np.sort(w['tof'][w['BunchNumber'] == b] + (w['tflash'][w['BunchNumber'] == b] - mw))
                sel = other['BunchNumber'] == b
                if tag == 'pss':
                    sel &= other['amp'] > PLASTIC_MIP
                to = other['tof'][sel] + (other['tflash'][sel] - mo)
                if tw.size and to.size:
                    dts.append(nearest_dt(np.sort(to), tw))
            if dts:
                d = np.concatenate(dts)
                pk, h = peak_of(d)
                off[f'{tag}{arm}'] = {'peak_ns': pk, 'height': h, 'n': int(d.size)}
    out['offsets'] = off
    return out


def fmt(runs, label):
    print(f'\n{"#" * 78}\n# {label}\n{"#" * 78}')
    hdr = (f'{"run":>7} {"bunch":>6} {"empty%":>7} {"1e12p":>8} '
           + ' '.join(f'{t:>8}' for t in ('WALA', 'WALB', 'PSSA', 'PSSC', 'LIQA', 'LIQD')))
    print('\nhits per 1e12 protons (beam bunches only)')
    print(hdr)
    print('-' * len(hdr))
    for r in runs:
        if 'trees' not in r:
            print(f'{r["run"]:>7}  {r.get("note", "no data")}')
            continue
        cells = ' '.join(f'{r["trees"].get(t, {}).get("hits_per_1e12p", float("nan")):8.0f}'
                         for t in ('WALA', 'WALB', 'PSSA', 'PSSC', 'LIQA', 'LIQD'))
        print(f'{r["run"]:>7} {r["bunches_beam"]:6d} {r["frac_empty"] * 100:7.1f} '
              f'{r["protons_1e12"]:8.1f} {cells}')

    print('\nmodal tflash per tree (ns) and % of beam bunches >150 ns off it')
    hdr2 = f'{"run":>7} ' + ' '.join(f'{t:>16}' for t in ('WALA', 'WALB', 'PSSA', 'PSSC', 'LIQA'))
    print(hdr2)
    print('-' * len(hdr2))
    for r in runs:
        if 'trees' not in r:
            continue
        cells = ' '.join(
            f'{r["trees"].get(t, {}).get("tflash_mode_ns", float("nan")):9.0f}/'
            f'{r["trees"].get(t, {}).get("flash_bad_pct", float("nan")):5.1f}%'
            for t in ('WALA', 'WALB', 'PSSA', 'PSSC', 'LIQA'))
        print(f'{r["run"]:>7} {cells}')

    print('\nprompt-coincidence offset vs the same arm wall (ns; target |peak| < 25)')
    keys = [f'{t}{a}' for t in ('pss', 'liq') for a in ARMS]
    hdr3 = f'{"run":>7} ' + ' '.join(f'{k:>7}' for k in keys)
    print(hdr3)
    print('-' * len(hdr3))
    for r in runs:
        o = r.get('offsets', {})
        if not o:
            continue
        cells = ' '.join(f'{o.get(k, {}).get("peak_ns", float("nan")):7.0f}' for k in keys)
        print(f'{r["run"]:>7} {cells}')

    print('\nmedian hit amplitude per tree')
    hdr4 = f'{"run":>7} ' + ' '.join(f'{t:>8}' for t in
                                     ('WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSC', 'LIQA', 'LIQD'))
    print(hdr4)
    print('-' * len(hdr4))
    for r in runs:
        if 'trees' not in r:
            continue
        cells = ' '.join(f'{r["trees"].get(t, {}).get("median_amp", float("nan")):8.1f}'
                         for t in ('WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSC', 'LIQA', 'LIQD'))
        print(f'{r["run"]:>7} {cells}')


def main():
    npart, outjson, groups = 1, None, {}
    for a in sys.argv[1:]:
        if a.startswith('--partials='):
            npart = int(a.split('=', 1)[1])
        elif a.startswith('--json='):
            outjson = a.split('=', 1)[1]
        else:
            label, _, spec = a.partition('=')
            groups.setdefault(label, []).extend(spec.split(','))

    result = {}
    for label, dirs in groups.items():
        rows = []
        for d in dirs:
            m = re.search(r'(\d{6})', Path(d).name)
            run = int(m.group(1)) if m else -1
            ps = partials(d, npart)
            if not ps:
                print(f'{label} {d}: no partials')
                continue
            try:
                rows.append(analyse_run(run, ps))
            except Exception as e:
                print(f'{label} {run}: FAILED {type(e).__name__}: {e}')
        rows.sort(key=lambda r: r['run'])
        result[label] = rows
        fmt(rows, label)

    if outjson:
        Path(outjson).write_text(json.dumps(result, indent=1, default=float))
        print(f'\nwrote {outjson}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
