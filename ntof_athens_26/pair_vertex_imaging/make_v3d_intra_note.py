#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_v3d_intra_note.py -- the note for the last two studies of the pair-vertex
series: every clean pair in one 3D density (`vertex3d.py`), and whether two
tracks in ONE chamber share a vertex (`intra_vertex.py`, including its
``--multiplicity`` study of why that cannot be tested yet).

Built from the tables those modules wrote; every number in the prose is read
from them.  PNGs embedded; the 3D view loads plotly from its CDN.

    python -m pair_vertex_imaging.make_v3d_intra_note
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
for p in (str(REPO), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths                   # noqa: E402
from sept26_prelim_analysis.report_style import head       # noqa: E402
from pair_vertex_imaging import vertex_image as VI         # noqa: E402
from pair_vertex_imaging.make_image_note import (          # noqa: E402
    esc, img, mm, n, pc, sgn, table)

OUT = HERE / 'figures'
TITLE = 'Two-track vertices in 3D, and same-chamber pairs'
HANDOFF = 'sept26_prelim_analysis/HANDOFF_INTRA_TWO_TRACK_RECO.md'


def read(od, name):
    return pd.read_csv(od / name, keep_default_na=False, na_values=[''])


def tight_hint(od: Path, cap) -> pd.DataFrame:
    """Both legs within 10 mm of the capsule in x at its depth: |dy| < 40 mm, and
    the legs' y correlation, real against each null."""
    d = pd.read_parquet(od / 'pairs_intra.parquet')
    d = d[~d.clone]
    x1, x2 = d.xm_c + 0.5 * d.dx_c, d.xm_c - 0.5 * d.dx_c
    y1, y2 = d.ym_c + 0.5 * d.dy_c, d.ym_c - 0.5 * d.dy_c
    m = (np.abs(x1) < 10) & (np.abs(x2) < 10)
    rows = []
    for (arm, kind), g in d[m].groupby(['arm', 'kind'], observed=True):
        k = int((np.abs(g.dy_c) < 40).sum())
        # 'ycorr', not 'corr': a row's .corr is pandas' Series.corr method
        rows.append(dict(arm=arm, kind=kind, n=len(g), k=k, frac=k / max(len(g), 1),
                         err=np.sqrt(max(k, 1)) / max(len(g), 1),
                         ycorr=float(np.corrcoef(y1[g.index], y2[g.index])[0, 1])))
    return pd.DataFrame(rows)


def build(od: Path) -> str:
    cap = VI.capsule_centre()
    C = read(od, 'v3d_classes.csv').set_index('cls')
    M = read(od, 'v3d_measure.csv')
    IS = read(od, 'intra_summary.csv')
    MU = read(od, 'intra_multiplicity.csv')
    SE = read(od, 'intra_twotrack_separation.csv')
    mmeta = json.loads((od / 'intra_multiplicity.meta.json').read_text())
    H = tight_hint(od, cap)

    def mrow(cls, kind='data'):
        return M[(M.cls == cls) & (M['map'] == kind)].iloc[0]

    def irow(arm, kind):
        return IS[(IS.arm == arm) & (IS.kind == kind)].iloc[0]

    def murow(arm, sample):
        return MU[(MU.arm == arm) & (MU['sample'] == sample)].iloc[0]

    allr, bal, baln, dd = mrow('all'), mrow('balanced'), mrow('balanced', 'noise'), mrow('D–D')
    xs_share = C.loc['A–D, C–D'].excess / C.loc['all'].excess
    a1, a2 = murow('A', '1 track in the chamber'), murow('A', '2 tracks')
    c1, c2 = murow('C', '1 track in the chamber'), murow('C', '2 tracks')
    ar, am = irow('A', 'real'), irow('A', 'mixed')
    cr, cm = irow('C', 'real'), irow('C', 'mixed')
    ha = H.set_index(['arm', 'kind'])
    # the separation table counts selected TRACKS in two-track chambers, once per
    # view; each view's rows sum to the same track total
    closey = SE[(SE.sep_hi <= 12.0) & (SE.view == 'y')].groupby('arm').n.sum()
    closex = SE[(SE.sep_hi <= 12.0) & (SE.view == 'x')].groupby('arm').n.sum()
    twotot = SE[SE.view == 'y'].groupby('arm').n.sum()
    far = SE[(SE.sep_lo >= 80.0)].set_index(['arm', 'view'])
    near = SE[(SE.sep_lo == 16.0)].set_index(['arm', 'view'])

    verdict = f'''
<p class="verdict"><b>A 3D density of every clean pair works once the pair
classes are weighted by what they measure &mdash; and whether two tracks in one
chamber share a vertex cannot be tested with the current reconstruction, because a
second track in the chamber breaks both.</b></p>
<ul>
<li><b>Summing every clean pair shows one class.</b> A&ndash;D and C&ndash;D carry
{pc(xs_share)} of the excess and image only x, so the raw sum is their slab: it
peaks {mm(allr.peak_dist_from_capsule)}&nbsp;mm from the single-track capsule
position and never localises in z.</li>
<li><b>Weighted equally, the x slab (A&ndash;D, C&ndash;D) and the z slab
(D&ndash;D) cross at the capsule:</b> peak at x&nbsp;=&nbsp;{sgn(bal.peak_x, 0)},
z&nbsp;=&nbsp;{sgn(bal.peak_z, 0)}&nbsp;mm, {mm(bal.peak_dist_from_capsule)}&nbsp;mm
from it, {mm(bal.box_sigma)}&sigma; in a 12&nbsp;mm box against
{mm(baln.box_sigma)}&sigma; at the largest fluctuation of the same construction with
no source. A&ndash;A, C&ndash;C and A&ndash;C add nothing above noise.</li>
<li><b>Same-chamber pairs show no common vertex</b> &mdash; their legs agree no better
than two tracks from different triggers (event-mixed) &mdash; <b>but the tests are
blind.</b> A track that shares its chamber with a second one has a y at the capsule
{mm(a2.rsig_y_at_capsule / a1.rsig_y_at_capsule, 1)}&times; worse in A
({mm(a1.rsig_y_at_capsule, 0)} &rarr; {mm(a2.rsig_y_at_capsule, 0)}&nbsp;mm) and
{mm(c2.rsig_y_at_capsule / c1.rsig_y_at_capsule, 1)}&times; worse in C, its fits use
2&ndash;3&times; the strips at {mm(a2.med_chi2dof_y / a1.med_chi2dof_y, 0)}&times; the
&chi;&sup2;/dof, and it is worst for two tracks far apart. And two tracks closer than
12&nbsp;mm on the strip plane are essentially never reconstructed as two: of
{n(twotot.get('A', 0))} selected tracks in two-track A chambers,
{int(closey.get('A', 0))} has its partner within 12&nbsp;mm in y and
{int(closex.get('A', 0))} in x ({int(closey.get('C', 0))} and
{int(closex.get('C', 0))} of {n(twotot.get('C', 0))} in C).</li>
<li><b>Recovering these is a reconstruction task</b>, handed off in
<code>{HANDOFF}</code>.</li>
</ul>
'''

    s1 = f'''
<h2>1. One 3D density of every clean pair</h2>
<p>Every pair with both legs slope-measured and outside noisy columns (in x) and
within 60&nbsp;mm of the beam axis, as its transverse crossing with y the mean of the
legs&rsquo; y. Each class minus its own no-source null (directions shuffled within
chamber &times; run &times; quality flags), the null normalised to the data in the
corners of the transverse window, where both |x&minus;c<sub>x</sub>| and
|z&minus;c<sub>z</sub>| exceed 35&nbsp;mm.</p>
{table(C.drop(index='balanced', errors='ignore').reset_index(), {'cls': 'class', 'n_pairs': 'pairs', 'n_in_window': 'in window',
                         'bkg_fraction': 'null share', 'excess': 'excess'},
       {'n_pairs': n, 'n_in_window': lambda v: n(v) if np.isfinite(v) else '&mdash;',
        'bkg_fraction': lambda v: pc(v), 'excess': lambda v: n(v) if v > 10 else mm(v, 2)})}
<p>Which direction each class localises is set by its geometry: in A&ndash;D and
C&ndash;D the crossing&rsquo;s x is the A/C leg and its z the D leg; two D lines run
along x, so D&ndash;D fixes z; two A lines or two C lines run along z and fix neither
well. A class on its own therefore images a slab, not a point. The
&ldquo;balanced&rdquo; map gives the x slab and the z slab unit weight each.</p>
<figure>{img('v3d_projections', 'projections by class')}
<figcaption><b>Data minus null, per class and summed.</b> Columns: transverse,
x&ndash;y, z&ndash;y. The y columns here use the mean of both legs, which the y note
shows is contaminated by the D leg &mdash; the y study replaces it.</figcaption></figure>
<figure>{img('v3d_significance', 'significance maps')}
<figcaption><b>The transverse image as a significance</b> (excess over its Poisson
error in 12&nbsp;mm boxes): raw sum, balanced, and balanced with no source.
</figcaption></figure>
<figure>{img('v3d_profiles', 'slices through the capsule')}
<figcaption><b>Slices through the capsule</b>, each class stacked, against the same
construction with no source (grey dashed).</figcaption></figure>
<div style="border:1px solid var(--line);border-radius:8px;background:var(--panel);overflow:hidden">
{(OUT / 'v3d_volume.div.html').read_text(encoding='utf-8')}
</div>
<p class="note">The interactive view: balanced map first, each class on the buttons.
D&ndash;D alone peaks {mm(dd.peak_dist_from_capsule)}&nbsp;mm from the capsule at
{mm(dd.box_sigma)}&sigma;.</p>
'''

    itab = IS[['arm', 'kind', 'n_noclone', 'rsig_dx_c', 'rsig_dy_c', 'f_dy15',
               'n_well', 'f_zagree30', 'spearman_zx_zy', 'med_sep']]
    s2 = f'''
<h2>2. Do two tracks in one chamber share a vertex?</h2>
<p><b>The right null is event mixing, not the shuffle.</b> The shuffled null has no
source at all; it answers &ldquo;do these tracks point anywhere&rdquo;. Two legs from
<i>different</i> triggers that each pass the same cuts still each point at the capsule,
so the mixed null answers the question asked here: &ldquo;are these two tracks more
vertex-like than two independent capsule tracks&rdquo;.</p>
<p>Legs: A&ndash;A or C&ndash;C, slope measured and not noisy in <i>both</i> views,
transverse miss from the measured capsule position below 30&nbsp;mm. Pairs whose legs
share an x or a y cluster (clones) are removed first; there were
{pc(IS.clone_frac.max(), 2)} at most.</p>
<ol>
<li><b>Same point at the capsule&rsquo;s depth</b>: &Delta;x and &Delta;y between the
legs at z&nbsp;=&nbsp;c<sub>z</sub>.</li>
<li><b>The two views agree on depth</b>: the x views cross at z<sub>x</sub>, the y
views at z<sub>y</sub>; for a real vertex these are one depth measured twice. Only
pairs with an angle difference above 0.15 in both views.</li>
<li><b>Where agreeing pairs&rsquo; vertices are</b>, real against mixed.</li>
</ol>
{table(itab, {'arm': 'chamber', 'kind': 'sample', 'n_noclone': 'pairs',
              'rsig_dx_c': '&Delta;x robust &sigma;', 'rsig_dy_c': '&Delta;y robust &sigma;',
              'f_dy15': '|&Delta;y|&nbsp;&lt;&nbsp;15&nbsp;mm', 'n_well': 'depth-conditioned',
              'f_zagree30': '|z<sub>x</sub>&minus;z<sub>y</sub>|&nbsp;&lt;&nbsp;30&nbsp;mm',
              'spearman_zx_zy': 'Spearman &rho;(z<sub>x</sub>, z<sub>y</sub>)',
              'med_sep': 'median 3D closest approach'},
       {'n_noclone': n, 'rsig_dx_c': mm, 'rsig_dy_c': lambda v: mm(v, 0),
        'f_dy15': lambda v: pc(v, 1), 'n_well': n, 'f_zagree30': lambda v: pc(v, 1),
        'spearman_zx_zy': lambda v: f'{v:+.2f}', 'med_sep': lambda v: mm(v, 0)})}
<p><b>No test separates real from mixed.</b> A&ndash;A &Delta;y robust &sigma; is
{mm(ar.rsig_dy_c, 0)} against {mm(am.rsig_dy_c, 0)}&nbsp;mm, C&ndash;C
{mm(cr.rsig_dy_c, 0)} against {mm(cm.rsig_dy_c, 0)}; the two views&rsquo; depths are
uncorrelated in every sample. The one hint is in A&ndash;A with both legs within
10&nbsp;mm of the capsule in x: |&Delta;y|&nbsp;&lt;&nbsp;40&nbsp;mm in
{pc(ha.loc[('A', 'real')].frac)} of {n(ha.loc[('A', 'real')].n)} real pairs against
{pc(ha.loc[('A', 'mixed')].frac)} of {n(ha.loc[('A', 'mixed')].n)} mixed, legs&rsquo; y
correlation {mm(ha.loc[('A', 'real')].ycorr, 2)} against
{mm(ha.loc[('A', 'mixed')].ycorr, 2)} &mdash; about
{mm((ha.loc[('A', 'real')].frac - ha.loc[('A', 'mixed')].frac) / np.hypot(ha.loc[('A', 'real')].err, ha.loc[('A', 'mixed')].err), 1)}&sigma;, on a
few hundred pairs, and nothing like it in C.</p>
<figure>{img('intra_test1', 'test 1')}
<figcaption><b>Test 1.</b> Real, mixed and shuffled.</figcaption></figure>
<figure>{img('intra_test2', 'test 2')}
<figcaption><b>Test 2.</b> A real vertex would put pairs on the diagonal.</figcaption></figure>
<figure>{img('intra_test3', 'test 3')}
<figcaption><b>Test 3.</b> Where the few agreeing pairs sit, real against mixed.</figcaption></figure>
'''

    mut = MU[['arm', 'sample', 'n', 'frac', 'rsig_y_at_capsule', 'rsig_x_at_capsule',
              'med_y_strips', 'med_x_strips', 'med_chi2dof_y', 'med_chi2dof_x', 'f_ncand_y_gt1']]
    s3 = f'''
<h2>3. Why those tests are blind: a second track breaks the first</h2>
<p>For a common vertex, &Delta;y at the capsule&rsquo;s depth should be about
&radic;2 times one track&rsquo;s y resolution &mdash; roughly
{mm(np.sqrt(2) * a1.rsig_y_at_capsule, 0)}&nbsp;mm in A &mdash; and far narrower than
for independent tracks. The mixed sample above has {mm(am.rsig_dy_c, 0)}&nbsp;mm. The
reason is the tracks themselves:</p>
{table(mut, {'arm': 'chamber', 'sample': 'gated tracks in the chamber', 'n': 'tracks',
             'frac': 'share', 'rsig_y_at_capsule': 'y at capsule, robust &sigma;',
             'rsig_x_at_capsule': 'x at capsule, robust &sigma;',
             'med_y_strips': 'y strips', 'med_x_strips': 'x strips',
             'med_chi2dof_y': '&chi;&sup2;/dof y', 'med_chi2dof_x': '&chi;&sup2;/dof x',
             'f_ncand_y_gt1': '&gt;&nbsp;1 y candidate'},
       {'n': n, 'frac': lambda v: pc(v, 1), 'rsig_y_at_capsule': lambda v: mm(v, 0),
        'rsig_x_at_capsule': mm, 'med_y_strips': lambda v: mm(v, 0),
        'med_x_strips': lambda v: mm(v, 0), 'med_chi2dof_y': mm, 'med_chi2dof_x': mm,
        'f_ncand_y_gt1': lambda v: pc(v)})}
<p>Campaign-wide ({n(mmeta['n_tracks'])} selected tracks). Multiplicity counts every
gated track of the chamber in the trigger, before quality cuts. Selection is the same
as section&nbsp;2, so these are the legs the tests used.</p>
<figure>{img('intra_multiplicity', 'track quality against multiplicity')}
<figcaption><b>Left:</b> y at the capsule&rsquo;s depth by the number of tracks in the
chamber. <b>Middle and right:</b> for chambers with exactly two tracks, resolution and
strip count against the two tracks&rsquo; separation on the strip plane, with the
single-track value dotted and the seed-clustering gap (12&nbsp;mm) in copper.
</figcaption></figure>
<p><b>Three things this shows, and one it does not.</b></p>
<ul>
<li><b>Both views degrade, y far more than x</b>: y by 3&ndash;4&times;, x by about
1.4&times;. The strip counts and &chi;&sup2;/dof rise in both.</li>
<li><b>It gets worse as the two tracks move apart, not closer</b> &mdash; the opposite of
overlapping charge. The closest resolvable pairs (16&ndash;24&nbsp;mm apart,
{n(near.loc[('A', 'y')].n)} A and {n(near.loc[('C', 'y')].n)} C tracks) have y at
{mm(near.loc[('A', 'y')].rsig, 0)} and {mm(near.loc[('C', 'y')].rsig, 0)}&nbsp;mm with
{mm(near.loc[('A', 'y')].med_strips, 0)} strips; 80&ndash;400&nbsp;mm apart it is
{mm(far.loc[('A', 'y')].rsig, 0)}&nbsp;mm (A) and {mm(far.loc[('C', 'y')].rsig, 0)}&nbsp;mm
(C), with {mm(far.loc[('A', 'y')].med_strips, 0)} and
{mm(far.loc[('C', 'y')].med_strips, 0)} strips in the y fit. Damage that scales with the
distance between two tracks points at a window or a plane fit that spans or mixes them.</li>
<li><b>Close tracks are not reconstructed as two at all.</b> Of
{n(twotot.get('A', 0))} selected tracks in two-track A chambers,
{int(closey.get('A', 0))} has its partner within 12&nbsp;mm in y and
{int(closex.get('A', 0))} in x; for C, {int(closey.get('C', 0))} and
{int(closex.get('C', 0))} of {n(twotot.get('C', 0))}. The seed
clustering joins hits closer than 12&nbsp;mm, so two tracks that close become one
candidate &mdash; lost from the pair sample, not mis-measured in it.</li>
<li><b>What it does not show is which mechanism</b> &mdash; a wrong x&harr;y assignment
between two time-coincident candidates, candidate windows contaminated by the other
track, or intrinsically messier events (showers, &delta;-rays). Separating those needs
a sample with known truth; the handoff sets one out.</li>
</ul>
'''

    s4 = f'''
<h2>4. What happens next</h2>
<p><code>{HANDOFF}</code> hands off the reconstruction work: how <code>wft</code>
builds tracks today and where two tracks in one chamber can go wrong in it, a
waveform-overlay truth bench to tell the mechanisms apart, the reconstruction options
to try, and the acceptance criteria &mdash; including that single-track events must
not move. Once two-track events reconstruct as well as single ones, the three tests of
section&nbsp;2 run unchanged (<code>python -m pair_vertex_imaging.intra_vertex</code>).</p>
'''

    caveats = '''
<h2>What this does not rule out</h2>
<ul>
<li><b>A common vertex in same-chamber pairs.</b> The null result is a statement about
the tests&rsquo; sensitivity, not about the physics.</li>
<li><b>The balanced 3D map is a back-projection</b>: the right centre and a cross-shaped
spread, not a deconvolved image. A reconstruction that treats each pair as a
measurement along one direction (ML-EM with per-class response) would sharpen it.</li>
<li><b>The source shape.</b> The pairs are expected to come from the capsule&rsquo;s
aluminium (its bottom end, perhaps its top), not the gas, and the capsule is not
expected at the nominal origin. Nothing here fits a source shape; positions are
measured relative to the single-track capsule position.</li>
<li><b>Chamber B</b> is excluded throughout (no usable angle), so B&ndash;B pairs and
the B&ndash;D opposing class are absent.</li>
</ul>
'''
    return verdict + s1 + s2 + s3 + s4 + caveats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'pair_vertex')))
    ap.add_argument('--out', default=str(OUT / 'v3d_intra_note.html'))
    a = ap.parse_args()
    od = Path(a.src)
    doc = (f'<!doctype html><html lang="en"><head>{head(TITLE)}'
           f'<meta name="description" content="Every clean pair in one 3D density, and '
           f'whether two tracks in one chamber share a vertex."></head><body>'
           f'<div class="topbar"><div class="topbar-in">'
           f'<span class="eyebrow">n_TOF 2026 &middot; X17 &middot; preliminary</span>'
           f'</div></div><div class="wrap">'
           f'<h1>{TITLE}</h1>'
           f'<p class="lede">The fourth note of the pair-vertex series, after <i>Pair vertices '
           f'as a capsule image</i>, <i>Where the z of the pair image went</i> and <i>The pair '
           f'vertex along the beam</i>.</p>'
           f'{build(od)}'
           f'<p class="prov">Built by <code>ntof_athens_26/pair_vertex_imaging/'
           f'make_v3d_intra_note.py</code> from the tables in <code>{esc(a.src)}</code> '
           f'(<code>vertex3d.py</code>, <code>intra_vertex.py</code>); every number above is '
           f'read from those tables.</p>'
           f'</div></body></html>')
    p = Path(a.out)
    p.write_text(doc, encoding='utf-8')
    print(f'wrote -> {p}  ({len(doc) / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
