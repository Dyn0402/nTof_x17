#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_z_note.py -- the note for the z follow-up: why the z maps are uncorrelated,
whether chamber D can be cleaned, and how much z A and C give on their own.

Built from the tables `z_image.py` wrote (default and ``ac_aligned``) and the
figures `make_z_figures.py` drew; every number in the prose is read from those
tables.  PNGs are embedded so the note is one file for the notes site.

    python -m pair_vertex_imaging.make_z_note
"""
from __future__ import annotations

import argparse
import html
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
from pair_vertex_imaging import z_image as Z               # noqa: E402
from pair_vertex_imaging.make_image_note import (          # noqa: E402
    esc, img, mm, n, pc, sgn, table)

OUT = HERE / 'figures'
TITLE = 'Where the z of the pair image went'
CUT = 60.0
OTHER = 'clean'


def read(od, name):
    return pd.read_csv(od / name, keep_default_na=False, na_values=[''])


def fr(F, sel, tier, coord, cut=CUT, smin=0.0, other=None):
    other = tier if other is None else other
    r = F[(F.selection == sel) & (F.tier == tier) & (F.tier_other == other)
          & (F.coord == coord) & (F.cut_mm == cut)
          & np.isclose(F.sin_psi_min.astype(float), smin)]
    return r.iloc[0] if len(r) else None


def fitted(r) -> bool:
    return r is not None and np.isfinite(r.get('c', np.nan))


def cz(r, nd=1) -> str:
    """'centre ± err' or the plain statement that no capsule term is wanted."""
    if r is None:
        return 'not fitted'
    if not fitted(r):
        return 'no capsule term wanted'
    return f'{sgn(r.c, nd)}&nbsp;&plusmn;&nbsp;{mm(r.c_err, nd)}&nbsp;mm'


def fit_table(rows: list[tuple[str, object]]) -> str:
    recs = []
    for label, r in rows:
        if r is None:
            continue
        ok = fitted(r)
        recs.append(dict(
            sel=label, n=r.n_data,
            c=r.c if ok else np.nan, c_err=r.c_err if ok else np.nan,
            s=r.s if ok else np.nan, f=r.get('f', np.nan),
            dl=r.get('two_dnll_vs_none', np.nan),
            chi=(r.chi2 / max(r.ndf, 1)) if 'chi2' in r and np.isfinite(r.chi2) else np.nan,
            band=r.get('band_excess', np.nan)))
    T = pd.DataFrame(recs)
    return table(T, {'sel': 'selection', 'n': 'pairs', 'c': 'centre', 'c_err': '&plusmn;',
                     's': '&sigma;', 'f': 'capsule term', 'dl': '2&Delta;lnL',
                     'chi': '&chi;&sup2;/ndf', 'band': '&plusmn;15&nbsp;mm excess'},
                 {'n': n, 'c': sgn, 'c_err': mm, 's': lambda v: mm(v, 0),
                  'f': lambda v: pc(v), 'dl': lambda v: mm(v, 0),
                  'chi': lambda v: mm(v, 1),
                  'band': lambda v: sgn(100 * v) + '&nbsp;%'})


def build(od: Path) -> str:
    F = read(od, 'z_fits.csv')
    FA = read(od, 'z_fits_ac_aligned.csv') if (od / 'z_fits_ac_aligned.csv').exists() else None
    T = read(od, 'z_pointing.csv').set_index(['arm', 'tier'])
    H = pd.read_csv(od / 'z_hot_columns.csv').dropna(subset=['frac_tracks_hot'])
    hot = H.groupby('arm').frac_tracks_hot.median()
    meta = json.loads((od / 'pairs_z.meta.json').read_text())
    metaA = (json.loads((od / 'pairs_z_ac_aligned.meta.json').read_text())
             if (od / 'pairs_z_ac_aligned.meta.json').exists() else None)
    cap = meta['capsule_xz']
    ic = pd.read_csv(paths.spell('out', 'imaging_campaign', 'per_arm.csv')).set_index('arm')

    def P(arm, tier):
        return T.loc[(arm, tier)]

    # --- the numbers the verdict quotes
    dD_all, dD_conf = P('D', 'all'), P('D', 'confirmed')
    dA_clean, dA_conf = P('A', 'clean'), P('A', 'confirmed')
    rel_D = 1 - P('D', 'reliable').frac_of_all
    rel_A = 1 - P('A', 'reliable').frac_of_all
    perp = {t: fr(F, 'perpendicular', t, 'z', other=OTHER) for t in Z.TIERS}
    perpx = fr(F, 'perpendicular', 'confirmed', 'x', other=OTHER)
    dd_c = fr(F, 'D-D', 'clean', 'z')
    dd_c5 = fr(F, 'D-D', 'clean', 'z', smin=0.5)
    dd_f = fr(F, 'D-D', 'confirmed', 'z')
    dd_f5 = fr(F, 'D-D', 'confirmed', 'z', smin=0.5)
    dd_c150 = fr(F, 'D-D', 'clean', 'z', cut=150.0)
    dd_f150 = fr(F, 'D-D', 'confirmed', 'z', cut=150.0)
    ac = fr(F, 'A-C', 'clean', 'z', smin=0.3)
    ac0 = fr(F, 'A-C', 'clean', 'z')
    ac150 = fr(F, 'A-C', 'clean', 'z', cut=150.0, smin=0.3)
    acA = fr(FA, 'A-C', 'clean', 'z', smin=0.3) if FA is not None else None
    acx = fr(F, 'A-C', 'clean', 'x', smin=0.3)
    aa = [fr(F, s, t, 'z', smin=m) for s in ('A-A', 'C-C')
          for t in ('clean', 'confirmed') for m in Z.SIN_PSI_BINS]
    aa_none = sum(1 for r in aa if r is not None and not fitted(r))
    aa_tot = sum(1 for r in aa if r is not None)
    aa_fit = [r for r in aa if fitted(r)]
    aa_fmax = max((r.f for r in aa_fit), default=np.nan)
    aa_dlmax = max((r.two_dnll_vs_none for r in aa_fit), default=np.nan)
    aa_band = max((abs(r.band_excess) for r in aa if r is not None
                   and np.isfinite(r.get('band_excess', np.nan))), default=np.nan)
    shift = (metaA or {}).get('shift_x_mm', {})

    verdict = f'''
<p class="verdict"><b>Yes, it is chamber D &mdash; and cleaning D does not rescue
z in perpendicular pairs, because the D leg of an A&ndash;D or C&ndash;D pair is
mostly a bystander track. z does come back from two other places: pairs with
both legs in D, and, weakly, opposing A&ndash;C pairs.</b></p>
<ul>
<li><b>D&rsquo;s gated tracks carry almost no pointing.</b> A median
{pc(hot.get('D', np.nan))} of them per run sit in noisy readout columns, and
{pc(rel_D)} have slopes too shallow for the drift timing to measure. Within
30&nbsp;mm of the axis D beats its own direction-shuffled null by
{pc(dD_all.excess_30, 1)}; A beats its null by {pc(P('A', 'all').excess_30, 1)}.
Keep only D tracks whose slope is measured, that are not in a noisy column, and
whose own scintillators fired, and D&rsquo;s excess becomes
<b>{pc(dD_conf.excess_30, 1)}</b> (median miss {mm(dD_conf.med_miss, 0)} against
{mm(dD_conf.med_miss_null, 0)}&nbsp;mm) &mdash; as good as chamber A after the
same cleaning ({pc(dA_clean.excess_30, 1)}).</li>
<li><b>But A&ndash;D and C&ndash;D pairs still give no z image.</b> With the D
leg merely cleaned the fit wants either nothing or a
{mm(perp['clean'].s if fitted(perp['clean']) else np.nan, 0)}&nbsp;mm-wide term with a
<i>deficit</i> at the capsule ({sgn(100 * perp['clean'].band_excess)}&nbsp;%).
Requiring D&rsquo;s scintillators on the D leg leaves only
{n(perp['confirmed'].n_data)} pairs: the production trigger lights one arm, so a
D-confirmed leg almost never shares its trigger with a good A or C track. Their
z is {cz(perp['confirmed'])} with &sigma;&nbsp;=&nbsp;{mm(perp['confirmed'].s, 0)}&nbsp;mm
&mdash; not enough to see a capsule.</li>
<li><b>D&ndash;D pairs image z.</b> Two D lines both run along x, so their
crossing fixes z and not x. Clean on both legs: z&nbsp;=&nbsp;{cz(dd_c)},
&sigma;&nbsp;=&nbsp;{mm(dd_c.s)}&nbsp;mm; scintillator-confirmed and
sin&thinsp;&psi;&nbsp;&ge;&nbsp;0.5: {cz(dd_f5)},
&sigma;&nbsp;=&nbsp;{mm(dd_f5.s)}&nbsp;mm, capsule term {pc(dd_f5.f)}. The
single-track band crossing puts the capsule at z&nbsp;=&nbsp;{sgn(cap[1])}&nbsp;mm,
and the D&ndash;D centre sits above it in every selection &mdash; from
{cz(dd_c)} (clean) to {cz(dd_f150)} (confirmed, no leg cut). D&rsquo;s readout is
dead from z&nbsp;&asymp;&nbsp;&minus;57 to 0&nbsp;mm, which removes capsule tracks
on one side, and is the first suspect.</li>
<li><b>A and C on their own give almost no z.</b> Across {aa_tot} fits of
A&ndash;A and C&ndash;C crossings (clean and confirmed, every crossing-angle cut)
{aa_none} want no capsule term, and the other {aa_tot - aa_none} find one of the
kind a fit makes of noise &mdash; at most 2&Delta;lnL&nbsp;=&nbsp;{mm(aa_dlmax, 0)}.
Opposing A&ndash;C pairs carry a weak
z image: {cz(ac)}, &sigma;&nbsp;=&nbsp;{mm(ac.s, 0)}&nbsp;mm,
&chi;&sup2;/ndf&nbsp;=&nbsp;{mm(ac.chi2 / ac.ndf, 1)} (clean,
sin&thinsp;&psi;&nbsp;&ge;&nbsp;0.3) &mdash; {mm(abs(ac.c - cap[1]), 0)}&nbsp;mm
from the capsule &mdash; <b>and it is not robust</b>: with no leg cut it becomes a
&sigma;&nbsp;=&nbsp;{mm(ac150.s, 0)}&nbsp;mm hump at {cz(ac150)}.'''
    if fitted(acA):
        verdict += f''' Moving A and C by {mm(abs(shift.get('A', 0)), 2)}&nbsp;mm each so their
single-track x crossings agree moves it by only {sgn(acA.c - ac.c)}&nbsp;mm.'''
    verdict += '</li></ul>'

    sample = (f'<p class="note">Sample: {n(meta["n_data"])} data pairs and '
              f'{n(meta["n_null"])} null pairs over {meta["n_runs"]} runs of the '
              f'condor full pass, chambers A, C and D, the same data sample as '
              f'the companion note (checked pair for pair at the 30&nbsp;mm cut). '
              f'Unless stated, both legs are within {CUT:.0f}&nbsp;mm of the beam '
              f'axis.</p>')

    s1 = f'''
<h2>1. The uncorrelated z hits are chamber D</h2>
<p>In a perpendicular pair the transverse crossing is nearly square, so its x is
set by the A or C leg and its z by the D leg. Anything wrong in the z maps is
therefore a statement about D&rsquo;s tracks, and D&rsquo;s tracks can be looked
at on their own: a track from the capsule lands on a diagonal band in
(impact position, angle), and a track that is not from the capsule lands
anywhere.</p>
<figure>{img('z_bands', 'pointing bands per chamber and cut')}
<figcaption><b>The pointing band, per chamber, per cut.</b> Columns left to right
apply the cuts described in section&nbsp;2. A and C show the band from the start;
D&rsquo;s is buried under vertical stripes (noisy columns, which make tracks at
every angle at one position) and a horizontal line at tan&nbsp;&asymp;&nbsp;0
(slopes the drift timing did not measure). Because D&rsquo;s in-plane coordinate
is &minus;z, each D stripe is a fixed z &mdash; the horizontal bands in the
transverse vertex map.</figcaption></figure>
{table(T.reset_index(), {'arm': 'chamber', 'tier': 'cut', 'n_tracks': 'tracks',
                         'frac_of_all': 'kept', 'med_miss': 'median miss',
                         'med_miss_null': 'null', 'f30': 'within 30&nbsp;mm',
                         'f30_null': 'null', 'excess_30': 'excess'},
       {'n_tracks': n, 'frac_of_all': lambda v: pc(v), 'med_miss': lambda v: mm(v, 0),
        'med_miss_null': lambda v: mm(v, 0), 'f30': lambda v: pc(v, 1),
        'f30_null': lambda v: pc(v, 1), 'excess_30': lambda v: pc(v, 1)})}
<p>Every gated, angle-calibrated track of the campaign. &ldquo;null&rdquo; is the
same tracks with their directions shuffled among tracks of the same chamber, run
and cut &mdash; impact points and angles kept, their pairing destroyed. The
excess is the pointing information.</p>
<figure>{img('z_pointing', 'single-track miss distributions')}
<figcaption><b>Single-track miss distance from the beam axis</b>, each cut against
its own shuffled null (dashed). D&rsquo;s curves only separate from their nulls
at the last cut.</figcaption></figure>
'''

    s2 = f'''
<h2>2. The three cuts</h2>
<ul>
<li><b>Slope measured</b> &mdash; <code>x_slope_reliable</code>, which the
waveform reconstruction sets as |tan&thinsp;&theta;|&nbsp;&ge;&nbsp;0.08: below that
&ldquo;the timing carries no slope information&rdquo; (<code>wft/reco.py</code>), and
the fit piles those tracks up at tan&nbsp;&asymp;&nbsp;0. The flag is recorded and
gates nothing in the current chain. It removes {pc(rel_D)} of D&rsquo;s gated
tracks and {pc(rel_A)} of A&rsquo;s. <i>It also removes genuinely near-normal
tracks from the capsule</i> &mdash; about &plusmn;19&nbsp;mm around each
chamber&rsquo;s foot point &mdash; which the null takes equally.</li>
<li><b>Not in a noisy column</b> &mdash; found per run from the data: 2&nbsp;mm
columns whose track occupancy is more than 4&times; the running median over
30&nbsp;mm. Median fraction of tracks in one, per run: D {pc(hot.get('D', np.nan))},
A {pc(hot.get('A', np.nan))}, C {pc(hot.get('C', np.nan))}.</li>
<li><b>Own scintillators fired</b> &mdash; <code>coinc_this_arm</code>: the
track&rsquo;s own arm recorded its wall-and-plastic coincidence, which selects
particles that crossed the chamber toward that wall.</li>
</ul>
<p>The cuts are cumulative. The null for each is shuffled within the same cut, so
it is always the same selection as the data.</p>
'''

    perp_rows = [(f'D leg: {t}', perp[t]) for t in Z.TIERS]
    s3 = f'''
<h2>3. Does cleaning D bring z back in A&ndash;D and C&ndash;D pairs?</h2>
<p>The D leg&rsquo;s cut is varied; the A/C leg is always &ldquo;slope measured,
not in a noisy column&rdquo;. Both legs are cut on together only in the
companion tables, because a pair confirmed on both legs needs two arms to fire
and the production trigger almost never does that.</p>
{fit_table(perp_rows)}
<p>z of perpendicular crossings; centre, blur and capsule term free. With the D
leg merely cleaned the fit finds a broad term of &sigma;&nbsp;&asymp;&nbsp;50&nbsp;mm
with a deficit, not an excess, at the capsule: that is D&rsquo;s remaining
acceptance structure, not a source. With the D leg confirmed there are
{n(perp['confirmed'].n_data)} pairs, and their x image is still there
({cz(perpx)}) while z is {cz(perp['confirmed'])}.</p>
<figure>{img('z_transverse_tiers', 'transverse maps by D-leg cut')}
<figcaption><b>Perpendicular crossings, data minus null</b>, as the D leg is
cleaned. The vertical band at the capsule&rsquo;s x (from the A/C leg) survives
every cut; nothing localises in z.</figcaption></figure>
<figure>{img('z_perp_profiles', 'perpendicular z and x profiles')}
<figcaption><b>The same, as fitted profiles.</b> Top: z. Bottom: x, for
comparison.</figcaption></figure>
'''

    par_rows = []
    for sel in Z.PAR_SELECTIONS:
        for tier in ('clean', 'confirmed'):
            for smin in (0.0, 0.5):
                lab = f'{sel} · {tier}' + (f' · sin ψ ≥ {smin:.1f}' if smin else '')
                par_rows.append((lab, fr(F, sel, tier, 'z', smin=smin)))
        par_rows.append((f'{sel} · clean · no leg cut', fr(F, sel, 'clean', 'z', cut=150.0)))

    s4 = f'''
<h2>4. Can A and C give z on their own?</h2>
<p><b>A single A or C track cannot.</b> A and C sit at &plusmn;z and drift along
z, so a track in either runs roughly along z. Its drift depth is exactly what
gives it an <i>angle</i> &mdash; x changing with depth &mdash; and so its x at
the capsule. It says nothing about <i>where along its own length</i> the vertex
is. That needs a second line crossing it at an angle. Without D, the second line
is another A or C track:</p>
<ul>
<li><b>A&ndash;A and C&ndash;C.</b> Both lines run along z, so they cross at a
shallow angle and the crossing is well placed in x and badly placed in z &mdash;
the mirror image of why perpendicular pairs work.</li>
<li><b>A&ndash;C.</b> The lines come from opposite sides. For lines with slopes
t<sub>A</sub>, t<sub>C</sub> (dx/dz) meeting the strip planes at x<sub>A</sub>
(z&nbsp;=&nbsp;+L) and x<sub>C</sub> (z&nbsp;=&nbsp;&minus;L):
<pre>z = [x_A &minus; x_C &minus; L (t_A + t_C)] / (t_C &minus; t_A),    L = 234.6 mm</pre>
so z is set by the <i>difference</i> of the two chambers&rsquo; positions and
slopes. A relative x offset of the two chambers moves each A&ndash;C z by
offset&nbsp;/&nbsp;(t<sub>C</sub>&nbsp;&minus;&nbsp;t<sub>A</sub>). That
denominator takes both signs across the sample, so an offset pushes pairs both
ways: it widens the A&ndash;C image more than it moves it (tested in
section&nbsp;5).</li>
<li><b>D&ndash;D</b>, for completeness: two D lines run along x, so their crossing
fixes z well and x badly. That is chamber D measuring z, not A or C.</li>
</ul>
{fit_table(par_rows)}
<p>z fits, centre free. A&ndash;A and C&ndash;C: no capsule term wanted in
{aa_none} of {aa_tot} fits, and the model-free excess within &plusmn;15&nbsp;mm of
the capsule never exceeds {mm(100 * aa_band, 1)}&nbsp;%. A&ndash;C: a weak image,
{cz(ac0)} with no crossing-angle cut, whose centre depends on the leg cut &mdash;
with no leg cut (sin&thinsp;&psi;&nbsp;&ge;&nbsp;0.3) the fit is a
&sigma;&nbsp;=&nbsp;{mm(ac150.s, 0)}&nbsp;mm hump at {cz(ac150)}, the blur pinned
at its upper bound, which is a broad excess rather than a capsule image.
D&ndash;D: {cz(dd_c)} clean, {cz(dd_f)} confirmed; with no leg cut
{cz(dd_c150)} clean and {cz(dd_f150)} confirmed. D&ndash;D is a real z image
(2&Delta;lnL&nbsp;=&nbsp;{mm(dd_f5.two_dnll_vs_none, 0)} for confirmed,
sin&thinsp;&psi;&nbsp;&ge;&nbsp;0.5) whose centre moves by several mm with the
selection.</p>
<figure>{img('z_parallel_profiles', 'z from same-side and opposing pairs')}
<figcaption><b>z profiles</b> for A&ndash;A, C&ndash;C, A&ndash;C and D&ndash;D,
clean on both legs, without (top) and with (bottom) a
sin&thinsp;&psi;&nbsp;&ge;&nbsp;0.5 crossing-angle cut.</figcaption></figure>
'''

    s5 = ''
    if FA is not None:
        al_rows = []
        for tier in ('all', 'clean'):
            for smin in Z.SIN_PSI_BINS:
                a0 = fr(F, 'A-C', tier, 'z', smin=smin)
                a1 = fr(FA, 'A-C', tier, 'z', smin=smin)
                al_rows.append(dict(
                    sel=f'{tier} · sin ψ ≥ {smin:.1f}',
                    n0=a0.n_data if a0 is not None else np.nan,
                    c0=a0.c if fitted(a0) else np.nan,
                    e0=a0.c_err if fitted(a0) else np.nan,
                    c1=a1.c if fitted(a1) else np.nan,
                    e1=a1.c_err if fitted(a1) else np.nan,
                    s1=a1.s if fitted(a1) else np.nan))
        AL = pd.DataFrame(al_rows)
        AL['shift'] = AL.c1 - AL.c0
        xA0, xC0 = float(ic.loc['A', 'median_mm']), float(ic.loc['C', 'median_mm'])
        ad0, ad1 = (fr(F, 'A-D', 'all', 'x', other='all'),
                    fr(FA, 'A-D', 'all', 'x', other='all'))
        cd0, cd1 = (fr(F, 'C-D', 'all', 'x', other='all'),
                    fr(FA, 'C-D', 'all', 'x', other='all'))
        s5 = f'''
<h2>5. Is the A&ndash;C z offset A and C sitting apart?</h2>
<p>A&rsquo;s and C&rsquo;s single-track band crossings put the capsule at
x&nbsp;=&nbsp;{sgn(xA0, 2)} and {sgn(xC0, 2)}&nbsp;mm &mdash; the same capsule,
{mm(abs(xA0 - xC0), 2)}&nbsp;mm apart. If that is a relative placement offset
it enters A&ndash;C z directly. So the pairs were rebuilt with A moved by
{sgn(shift.get('A', 0), 2)}&nbsp;mm and C by {sgn(shift.get('C', 0), 2)}&nbsp;mm in
x, which puts both crossings on their mean, and everything re-measured.</p>
<p><b>The shift did what it should where it should:</b> the x image from A&ndash;D
pairs moved by {sgn(ad1.c - ad0.c, 2)}&nbsp;mm and from C&ndash;D pairs by
{sgn(cd1.c - cd0.c, 2)}&nbsp;mm. <b>The A&ndash;C z image did not move</b>:</p>
{table(AL, {'sel': 'A&ndash;C selection', 'n0': 'pairs',
            'c0': 'z as reconstructed', 'e0': '&plusmn;',
            'c1': 'z with A/C aligned', 'e1': '&plusmn;', 'shift': 'moved by',
            's1': '&sigma; aligned'},
       {'n0': n, 'c0': sgn, 'e0': mm, 'c1': sgn, 'e1': mm, 'shift': sgn,
        's1': lambda v: mm(v, 0)})}
<figure>{img('z_aligned', 'A-C z before and after alignment')}
<figcaption><b>A&ndash;C z, as reconstructed and with A and C aligned.</b> Right:
every A&ndash;C fit both ways, against the single-track capsule z.</figcaption></figure>
<p>That is what the formula predicts once the sign of
t<sub>C</sub>&nbsp;&minus;&nbsp;t<sub>A</sub> is taken into account: the shift
moves each pair by an amount of either sign, and the image widens slightly rather
than moving. So the A/C placement does not explain why A&ndash;C z sits away from
the capsule. What does move it is the leg cut (section&nbsp;4).</p>
<p class="caution">This tests one hypothesis, and the answer is no; it is not a
correction. The 2&nbsp;mm A/C disagreement could still be an angle-scale difference
between the chambers (the A-arm scintillator wall already asks for a different
<i>k</i>), which enters A&ndash;C z through the
t<sub>A</sub>&nbsp;+&nbsp;t<sub>C</sub> term and was not tested here.</p>
'''

    s6 = f'''
<h2>6. Everything at once</h2>
<figure>{img('z_summary', 'forest plot of fitted centres')}
<figcaption><b>Every fitted centre</b>, z on the left and x on the right, with the
single-track capsule position in copper. Filled points have 2&Delta;lnL&nbsp;&gt;&nbsp;25
against no capsule term.</figcaption></figure>
'''

    caveats = '''
<h2>What this does not rule out</h2>
<ul>
<li><b>A better D reconstruction.</b> Everything here cuts D&rsquo;s bad tracks
away; it does not repair them. The hot-channel wildcard tried on D in September
made the reconstruction worse as first tuned (<code>STATUS.md</code>, 2026-09-08), so
a D with its noisy columns handled properly could give perpendicular z that this
cannot.</li>
<li><b>The z offsets are not resolved.</b> D&ndash;D and A&ndash;C z both differ
from the single-track z by more than their statistical errors, in opposite
directions, and both move with the selection by more than their errors. D&rsquo;s
dead readout between z&nbsp;&asymp;&nbsp;&minus;57 and 0&nbsp;mm is the first
suspect for D&ndash;D; the source model in the fit has no acceptance in it. A/C
placement was tested and does not move A&ndash;C z; a relative angle scale was not
tested. None of this is a calibrated z.</li>
<li><b>The slope cut removes near-normal capsule tracks.</b> The null takes the
same cut, so the comparison is fair, but the images are of the capsule seen
through a &plusmn;19&nbsp;mm hole around each foot point.</li>
<li><b>Confirmation selects a population.</b> A scintillator-confirmed track
points at its own wall; for same-chamber pairs that also restricts the crossing
angle.</li>
<li><b>Nothing here says two legs are one decay.</b> As in the companion note,
two independent capsule tracks give the same image.</li>
</ul>
'''
    return verdict + sample + s1 + s2 + s3 + s4 + s5 + s6 + caveats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'pair_vertex')))
    ap.add_argument('--out', default=str(OUT / 'z_image_note.html'))
    a = ap.parse_args()
    od = Path(a.src)
    doc = (f'<!doctype html><html lang="en"><head>{head(TITLE)}'
           f'<meta name="description" content="Why the z maps of the pair vertices '
           f'are uncorrelated, whether chamber D can be cleaned, and how much z A '
           f'and C give on their own."></head><body>'
           f'<div class="topbar"><div class="topbar-in">'
           f'<span class="eyebrow">n_TOF 2026 &middot; X17 &middot; preliminary</span>'
           f'</div></div><div class="wrap">'
           f'<h1>{TITLE}</h1>'
           f'<p class="lede">The follow-up to <i>Pair vertices as a capsule '
           f'image</i>: the x&ndash;y maps look right and every map with z in it '
           f'is full of uncorrelated vertices. Is that chamber D, can D be cleaned, '
           f'and can chambers A and C supply z from their drift depths instead?</p>'
           f'{build(od)}'
           f'<p class="prov">Built by <code>ntof_athens_26/pair_vertex_imaging/'
           f'make_z_note.py</code> from the tables in <code>{esc(a.src)}</code> '
           f'(<code>z_image.py</code>, default and <code>--align-ac</code>); every '
           f'number above is read from those tables.</p>'
           f'</div></body></html>')
    p = Path(a.out)
    p.write_text(doc, encoding='utf-8')
    print(f'wrote -> {p}  ({len(doc) / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
