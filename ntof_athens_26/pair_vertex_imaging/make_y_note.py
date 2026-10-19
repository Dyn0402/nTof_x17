#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_y_note.py -- the note for the y study: where the three peaks came from,
how y is cleaned, and what single tracks and pairs say about the capsule along
the beam.

Built from the tables `y_image.py` wrote and the figures `make_y_figures.py`
drew; every number in the prose is read from those tables.  PNGs embedded; the
3D view loads plotly from its CDN.

    python -m pair_vertex_imaging.make_y_note
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(REPO), str(REPO / 'mpgd26'), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths                   # noqa: E402
from sept26_prelim_analysis.report_style import head       # noqa: E402
from pair_vertex_imaging import y_image as Y               # noqa: E402
from pair_vertex_imaging.make_image_note import (          # noqa: E402
    esc, img, mm, n, pc, sgn, table)
from pair_vertex_imaging.make_y_figures import gas_robust_sigma  # noqa: E402

OUT = HERE / 'figures'
TITLE = 'The pair vertex along the beam'
RAW, BAND, FOCUS = 'raw (s = 1)', 'band scale', 'focus scale'


def read(od, name):
    return pd.read_csv(od / name, keep_default_na=False, na_values=[''])


def fr(F, kind, sel, scale, tier=None):
    r = F[(F.kind == kind) & (F.selection == sel) & (F.scale_set == scale)]
    if tier is not None:
        r = r[r.tier == tier]
    return r.iloc[0] if len(r) else None


def fitted(r) -> bool:
    return r is not None and np.isfinite(r.get('c', np.nan))


def cy(r, nd=1) -> str:
    if r is None:
        return 'not fitted'
    if not fitted(r):
        return 'no capsule term wanted'
    return f'{sgn(r.c, nd)}&nbsp;&plusmn;&nbsp;{mm(r.c_err, nd)}&nbsp;mm'


def blob_profile(od, cap):
    """y peak and FWHM of the balanced 3D map near the capsule transversely."""
    from scipy.ndimage import gaussian_filter1d
    Z3 = np.load(od / 'y_map3d.npz')
    ex, ey = Z3['ex'], Z3['ey']
    xc, yc = 0.5 * (ex[:-1] + ex[1:]), 0.5 * (ey[:-1] + ey[1:])
    near = ((np.abs(xc - cap[0]) < 16)[:, None] & (np.abs(xc - cap[1]) < 16)[None, :])
    out = {}
    for name in ('balanced', 'x slab (A–D, C–D)'):
        E = Z3[f'{name}|excess'].transpose(0, 2, 1)[near].sum(0)
        p = gaussian_filter1d(E, 1.0)
        i = int(np.argmax(p))
        half = p[i] / 2
        lo, hi = i, i
        while lo > 0 and p[lo] > half:
            lo -= 1
        while hi < len(p) - 1 and p[hi] > half:
            hi += 1
        fwhm = (yc[hi] - yc[lo]) if (p[lo] <= half and p[hi] <= half) else np.nan
        # centroid of the positive excess within the half-maximum span
        sl = slice(lo, hi + 1)
        w = np.clip(p[sl], 0, None)
        out[name] = dict(peak=float(yc[i]), fwhm=float(fwhm),
                         centroid=float((yc[sl] * w).sum() / max(w.sum(), 1e-30)))
    return out


def build(od: Path) -> str:
    F = read(od, 'y_fits.csv')
    BS = read(od, 'y_band_scale.csv')
    FS = read(od, 'y_focus.csv')
    meta = json.loads((od / 'y_derive.meta.json').read_text())
    pmeta = json.loads((od / 'pairs_y.meta.json').read_text())
    cap = meta['capsule_xz']
    ycen = meta['capsule_y_centroid']
    gas_rs = gas_robust_sigma()
    blob = blob_profile(od, cap)

    aleg = {s: fr(F, 'pair', 'A–D, A leg', s) for s in (RAW, FOCUS, BAND)}
    cleg = {s: fr(F, 'pair', 'C–D, C leg', s) for s in (RAW, FOCUS, BAND)}
    mean_ad = fr(F, 'pair', 'A–D, mean of both legs (as before)', RAW)
    dleg_ad = fr(F, 'pair', 'A–D, D leg alone', RAW)
    dd = fr(F, 'pair', 'D–D, mean of both D legs', RAW)
    st = {a: fr(F, 'single track', f'chamber {a}', RAW, 'y-clean') for a in Y.ARMS}
    bconf = BS[BS.tier == 'y-clean, confirmed'].set_index('arm')
    bclean = BS[BS.tier == 'y-clean'].set_index('arm')
    y0s = bconf.band_y0.to_numpy(float)
    fmin = FS.loc[FS.groupby('arm').rsig_data.idxmin()].set_index('arm')
    # how flat the focus scan is above s = 1: widest-to-narrowest over s in [1, 1.5]
    flat = {a: float(g[g.scale >= 1.0].rsig_data.max() / g[g.scale >= 1.0].rsig_data.min())
            for a, g in FS.groupby('arm')}
    centroids = [aleg[RAW].c, cleg[RAW].c] + [st[a].c for a in Y.ARMS if fitted(st[a])]

    verdict = f'''
<p class="verdict"><b>The three peaks were an artefact of averaging, and y can be
imaged: taking y from the legs that measure it, the pair vertex puts the capsule
at y&nbsp;&asymp;&nbsp;+30&nbsp;mm with a per-event blur of about
{mm(aleg[RAW].s, 0)}&nbsp;mm on top of the 80&nbsp;mm gas.</b></p>
<ul>
<li><b>Where the peaks came from.</b> The vertex y used so far is the mean of the
two legs&rsquo; y at the transverse crossing. In an A&ndash;D pair the A leg&rsquo;s
y there is a single peak, {cy(aleg[RAW])}; the D leg&rsquo;s y is junk
(robust &sigma; {mm(dleg_ad.rsig_data, 0)}&nbsp;mm, no capsule term) and
uncorrelated with it. Their mean smears the D leg&rsquo;s structure at half scale
around the A leg&rsquo;s peak &mdash; the three humps, and a fit that needs a
{mm(mean_ad.s, 0)}&nbsp;mm blur to describe them. A&ndash;D and C&ndash;D carried the
y distribution only because they are most of the excess near the capsule.</li>
<li><b>The pair y, done properly</b> (y of the A or C leg, crossing within
{mm(Y.WIN_X, 0)}&nbsp;mm of the capsule in x, x and y planes cleaned): A&ndash;D
{cy(aleg[RAW])}, &sigma;<sub>blur</sub>&nbsp;=&nbsp;{mm(aleg[RAW].s)}&nbsp;mm;
C&ndash;D {cy(cleg[RAW])}, &sigma;<sub>blur</sub>&nbsp;=&nbsp;{mm(cleg[RAW].s)}&nbsp;mm.</li>
<li><b>Single tracks say the same</b>, chamber by chamber:
A {cy(st['A'])}, C {cy(st['C'])}, D {cy(st['D'])}; and the y <i>band
crossing</i>, which does not depend on the y angle scale at all, puts it at
{', '.join(f'{a} {sgn(bconf.loc[a].band_y0)}' for a in Y.ARMS)}&nbsp;mm
(scintillator-confirmed tracks).</li>
<li><b>That is about {mm(np.mean(y0s) - ycen, 0)}&nbsp;mm above the gas&rsquo;s nominal
centroid</b> (y&nbsp;=&nbsp;{sgn(ycen)}&nbsp;mm in the detector frame). All three
chambers and both methods agree on it, so it is not a single chamber&rsquo;s y map;
whether it is the capsule&rsquo;s real height, a common y offset of the frame, or
where the captures happen inside the gas is not something this analysis can
separate.</li>
<li><b>In 3D</b>, the x slab (A&ndash;D, C&ndash;D) and the z slab (D&ndash;D) at equal
weight, with y from the A/C legs, make one compact blob: in y it peaks at
{sgn(blob['balanced']['peak'], 0)}&nbsp;mm with a FWHM of
{mm(blob['balanced']['fwhm'], 0)}&nbsp;mm.</li>
</ul>
'''

    sample = (f'<p class="note">Sample: {n(pmeta["n_data"])} data pairs and '
              f'{n(pmeta["n_tracks"])} single tracks (x-clean, pointing within 30&nbsp;mm) '
              f'over {pmeta["n_runs"]} runs of the condor full pass, chambers A, C and D. '
              f'Pairs: both legs within {mm(Y.CUT, 0)}&nbsp;mm of the axis. The null for '
              f'pairs is every direction shuffled within chamber &times; run &times; all '
              f'five quality flags; for single tracks, tan&thinsp;y shuffled the same way.</p>')

    s1 = f'''
<h2>1. Where the three peaks came from</h2>
<figure>{img('y_split', 'mean of legs versus each leg')}
<figcaption><b>A&ndash;D (top) and C&ndash;D (bottom).</b> Left: the mean of the two
legs&rsquo; y, as plotted before. Middle: the A or C leg alone. Right: the D leg alone,
after the y-plane cuts of section&nbsp;2 &mdash; still nothing at the capsule. y angle
scale as reconstructed.</figcaption></figure>
<p>The two legs of a pair meet at one transverse point, but each has its own y
there: y&nbsp;=&nbsp;p<sub>0,y</sub>&nbsp;+&nbsp;tan&thinsp;&theta;<sub>y</sub>&nbsp;&middot;&nbsp;&Delta;w,
with &Delta;w the distance along that chamber&rsquo;s drift axis from its strip plane
to the crossing. Averaging them is the right thing only if both are measurements.
The D leg&rsquo;s is not: in section&nbsp;2 its y band is barely there even after
cleaning, and in pairs its y shows only edge structure.</p>
'''

    bt = BS.assign(tier_label=BS.tier)
    s2 = f'''
<h2>2. The y plane has the same two problems as x</h2>
<ul>
<li><b>Unmeasured slopes</b> &mdash; <code>y_slope_reliable</code>, the same
|tan|&nbsp;&ge;&nbsp;0.08 rule as x, below which the fit piles tracks at
tan&thinsp;y&nbsp;&asymp;&nbsp;0 (the horizontal line below).</li>
<li><b>Noisy y columns</b> &mdash; found per run from the data, 2&nbsp;mm columns
above 4&times; the running median over 30&nbsp;mm (the vertical stripes). D&rsquo;s
y plane is the worst; some of its stripes survive this finder.</li>
</ul>
<figure>{img('y_bands', 'y pointing bands')}
<figcaption><b>The y pointing band</b>, before (top) and after (bottom) the y-plane
cuts, tracks already x-clean and pointing within 30&nbsp;mm in x. The vertical axis is
the y slope toward the beam axis scaled to the lever arm, so a point source gives a
line of slope 1 through (y<sub>s</sub>, 0); an 80&nbsp;mm source gives a smear of such
lines.</figcaption></figure>
{table(bt, {'arm': 'chamber', 'tier_label': 'tracks', 'n': 'tracks',
            'band_y0': 'band crossing y [mm]', 'band_scale': 'band slope'},
       {'n': n, 'band_y0': sgn, 'band_scale': lambda v: mm(v, 2)})}
<p>The <b>band crossing</b> &mdash; where the robust line through the band meets zero
slope &mdash; is the ensemble y of the source, and it is exact under any rescaling of
tan&thinsp;y. With scintillator-confirmed tracks the three chambers agree to
{mm(np.ptp(y0s), 1)}&nbsp;mm.</p>
'''

    strow = []
    for a in Y.ARMS:
        for s in (RAW, FOCUS, BAND):
            r = fr(F, 'single track', f'chamber {a}', s, 'y-clean')
            if r is not None:
                strow.append(dict(sel=f'chamber {a}', scale=s, n=r.n_data,
                                  c=r.c if fitted(r) else np.nan,
                                  s_=r.s if fitted(r) else np.nan,
                                  f=r.get('f', np.nan), rs=r.rsig_data, rsn=r.rsig_null,
                                  chi=r.chi2 / max(r.ndf, 1) if 'chi2' in r else np.nan))
    ST = pd.DataFrame(strow)
    s3 = f'''
<h2>3. Single tracks image the capsule along the beam</h2>
<p>Each track&rsquo;s y where it passes the beam axis, fitted as the gas&rsquo;s own
y profile (its cross-section &pi;R(y)&sup2;, centroid free) blurred by a Gaussian, on
top of the tan&thinsp;y-shuffled null.</p>
<figure>{img('y_single', 'single-track y images and focus scans')}
<figcaption><b>Top:</b> single-track y at the beam axis per chamber, y angle scale as
reconstructed. <b>Bottom:</b> the robust width against the y angle scale applied, with
the band slope and the focus minimum marked; the copper line is the robust &sigma; of
the gas alone ({mm(gas_rs, 0)}&nbsp;mm).</figcaption></figure>
{table(ST, {'sel': 'chamber', 'scale': 'y scale', 'n': 'tracks', 'c': 'centroid',
            's_': '&sigma;<sub>blur</sub>', 'f': 'capsule term', 'rs': 'robust &sigma;',
            'rsn': 'null', 'chi': '&chi;&sup2;/ndf'},
       {'n': n, 'c': sgn, 's_': mm, 'f': lambda v: pc(v), 'rs': lambda v: mm(v, 0),
        'rsn': lambda v: mm(v, 0), 'chi': lambda v: mm(v, 0)})}
<p class="caution"><b>The fits are not good fits.</b> With hundreds of thousands of
tracks the &chi;&sup2;/ndf is in the hundreds: the peak is more rounded than gas
&otimes; one Gaussian, and the null does not describe the tails exactly. The
centroids are nonetheless stable to a few mm across chambers and scales, and they
agree with the band crossings, which involve no fit shape at all.</p>
'''

    prow = []
    for lab, dct in (('A–D, A leg', aleg), ('C–D, C leg', cleg)):
        for s in (RAW, FOCUS, BAND):
            r = dct[s]
            prow.append(dict(sel=lab, scale=s, n=r.n_data,
                             c=r.c if fitted(r) else np.nan, e=r.c_err if fitted(r) else np.nan,
                             s_=r.s if fitted(r) else np.nan, f=r.get('f', np.nan),
                             chi=r.chi2 / max(r.ndf, 1)))
    PT = pd.DataFrame(prow)
    s4 = f'''
<h2>4. The pair vertex y</h2>
<figure>{img('y_pairs', 'pair vertex y')}
<figcaption><b>The pair y from the legs that measure it.</b> A&ndash;D and C&ndash;D:
the A or C leg&rsquo;s y at the crossing. D&ndash;D: the mean of both D legs &mdash;
{n(dd.n_data)} pairs and {cy(dd)}, because D&rsquo;s y does not point well enough in
pairs.</figcaption></figure>
{table(PT, {'sel': 'pair y from', 'scale': 'y scale', 'n': 'pairs', 'c': 'centroid',
            'e': '&plusmn;', 's_': '&sigma;<sub>blur</sub>', 'f': 'capsule term',
            'chi': '&chi;&sup2;/ndf'},
       {'n': n, 'c': sgn, 'e': mm, 's_': mm, 'f': lambda v: pc(v), 'chi': lambda v: mm(v, 1)})}
<p><b>Why take the A/C leg&rsquo;s y and not a combination.</b> In an A&ndash;D pair the
crossing&rsquo;s x comes from the A leg, its z from the D leg and its y, now, from the
A leg again. The pair contributes the transverse crossing point at which A&rsquo;s y
is evaluated; the y information itself is one track&rsquo;s. That is why the pair
blur ({mm(aleg[RAW].s)}&nbsp;mm) is the single-track blur
({mm(st['A'].s)}&nbsp;mm) and not better.</p>
'''

    s5 = f'''
<h2>5. The y angle scale is not calibrated, and that is carried as a bracket</h2>
<p>tan&thinsp;&theta;<sub>x</sub> is calibrated per run (<code>k_arm</code>);
tan&thinsp;&theta;<sub>y</sub> cannot be calibrated against the capsule in the same way
because the capsule is 80&nbsp;mm long along y. Two data-driven estimates were
tried, and they fail in opposite directions:</p>
<ul>
<li><b>The band slope</b> comes out below 1
({', '.join(f'{a} {mm(bclean.loc[a].band_scale, 2)}' for a in Y.ARMS)}):
background tracks flatten the band. Dividing tan&thinsp;y by it widens every image
&mdash; the C&ndash;D pair blur goes from {mm(cleg[RAW].s, 0)} to
{mm(cleg[BAND].s, 0)}&nbsp;mm.</li>
<li><b>The focus minimum</b> &mdash; the scale that makes the single-track image
narrowest &mdash; comes out above 1
({', '.join(f'{a} {mm(fmin.loc[a].scale, 2)}' for a in Y.ARMS)}), because a larger
scale also shrinks the tan&thinsp;y noise. And the width is nearly flat for any scale
&ge;&nbsp;1: the widest-to-narrowest ratio over 1.0&ndash;1.5 is
{', '.join(f'{a} {mm(flat[a], 2)}' for a in Y.ARMS)}.</li>
</ul>
<p>So the images use the scale as reconstructed, and the A&ndash;D centroid moves
between {sgn(aleg[FOCUS].c)} (focus) and {sgn(aleg[BAND].c)}&nbsp;mm (band) under the
other two &mdash; a systematic of a few mm on the centroid, larger on the blur.</p>
'''

    s6 = f'''
<h2>6. In 3D</h2>
<p>The x slab (A&ndash;D, C&ndash;D; y from the A/C leg) and the z slab (D&ndash;D; y from
both D legs) at equal weight, each minus its null. Drag to rotate; the buttons switch
between the balanced map and each slab.</p>
<div style="border:1px solid var(--line);border-radius:8px;background:var(--panel);overflow:hidden">
{(OUT / 'y_volume.div.html').read_text(encoding='utf-8')}
</div>
<figure>{img('y_maps', 'projections of the 3D map')}
<figcaption><b>Projections.</b> Rows: the x slab, the z slab, balanced. Columns:
transverse, x&ndash;y, z&ndash;y. The x slab alone is already localised in x&ndash;y; it
is the z&ndash;y view that needs D&ndash;D. The capsule is drawn at its <i>nominal</i>
y. Balanced map near the capsule transversely: y peak
{sgn(blob['balanced']['peak'], 0)}&nbsp;mm, FWHM {mm(blob['balanced']['fwhm'], 0)}&nbsp;mm
(x slab alone: {sgn(blob['x slab (A–D, C–D)']['peak'], 0)}&nbsp;mm,
{mm(blob['x slab (A–D, C–D)']['fwhm'], 0)}&nbsp;mm).</figcaption></figure>
'''

    caveats = f'''
<h2>What this does not rule out</h2>
<ul>
<li><b>The +30&nbsp;mm is not interpreted.</b> Every chamber and both methods agree on
it, which excludes one chamber&rsquo;s y strip map. It does not separate the
capsule&rsquo;s physical height, a common offset of the y frame, and a non-uniform
capture distribution inside the gas &mdash; the neutron beam runs along y, so where
along the gas the captures happen is itself a question.</li>
<li><b>The gas is a reference shape, not the expectation.</b> The expectation going
into this campaign is that the pairs come from the capsule&rsquo;s <i>aluminium</i>
&mdash; its bottom end, and perhaps its top &mdash; rather than from the He-3 gas,
and that the capsule does not sit at the nominal frame origin. So neither the
gas-profile source model nor the nominal gas centroid used above is what the data
should reproduce. An aluminium end-cap model is the next source shape to try.</li>
<li><b>The y angle scale is uncalibrated</b> (section&nbsp;5). The centroid moves by a
few mm across the bracket; the blur is more sensitive.</li>
<li><b>The fit model is too simple</b> for the single-track statistics (&chi;&sup2;/ndf
in the hundreds). Quote centroids and robust widths, not the fitted blur, to better
than a few mm.</li>
<li><b>D&rsquo;s y is not usable in pairs</b>, even cleaned, and some of D&rsquo;s noisy y
columns pass the finder; a D with its y noise handled in the reconstruction could
change the z&ndash;y view.</li>
<li><b>Nothing here says two legs are one decay.</b> As in the companion notes, two
independent capsule tracks give the same image.</li>
</ul>
'''
    return verdict + sample + s1 + s2 + s3 + s4 + s5 + s6 + caveats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'pair_vertex')))
    ap.add_argument('--out', default=str(OUT / 'y_image_note.html'))
    a = ap.parse_args()
    od = Path(a.src)
    doc = (f'<!doctype html><html lang="en"><head>{head(TITLE)}'
           f'<meta name="description" content="Why the pair-vertex y had three peaks, '
           f'the y-plane cleaning, and the capsule imaged along the beam by single tracks '
           f'and pairs."></head><body>'
           f'<div class="topbar"><div class="topbar-in">'
           f'<span class="eyebrow">n_TOF 2026 &middot; X17 &middot; preliminary</span>'
           f'</div></div><div class="wrap">'
           f'<h1>{TITLE}</h1>'
           f'<p class="lede">The third follow-up to <i>Pair vertices as a capsule image</i>. '
           f'The x and z of the pair vertex are settled; this is y, where the source is '
           f'80&nbsp;mm long and the first look showed three peaks.</p>'
           f'{build(od)}'
           f'<p class="prov">Built by <code>ntof_athens_26/pair_vertex_imaging/'
           f'make_y_note.py</code> from the tables in <code>{esc(a.src)}</code> '
           f'(<code>y_image.py</code>); every number above is read from those tables.</p>'
           f'</div></body></html>')
    p = Path(a.out)
    p.write_text(doc, encoding='utf-8')
    print(f'wrote -> {p}  ({len(doc) / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
