#!/usr/bin/env python3
"""
make_pair_angle_slides.py -- the relative-angle section of the two-track deck.

Answers "what relative angle do the modelled pairs have, does angle help, and
what angle do real pairs have?" from `pair_angle.py`'s outputs
(``~/x17/sept26_prelim/two_track_limit/pair_angle/``). Called by
`make_two_track_deck.build`; every slide checks that its input exists, and the
synthetic slide says "pending" until ``pair_angle oracle`` has finished.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import slidedoc as sd
from slidedoc import BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN, INK, MUT, RULE

from sept26_prelim_analysis import pair_angle as PA

ARM_COL = dict(A=BLUE, C=ORANGE)
PCT_TICKS = [(v / 100, f'{v}%') for v in (0, 25, 50, 75, 100)]
#: divergence ladder colours, parallel (dark) to strongly diverging (light)
D_COL = {0.0: INK, 0.5: '#3b4f86', 1.0: BLUE, 2.0: PURPLE, 4.0: GOLD, 8.0: ORANGE}


def _gap_frac() -> float:
    return PA.GAP_MM / PA.L_STRIP


# --------------------------------------------------------------------------- #
def s_geometry(D):
    W, H = 1000, 640
    cx, cy = 120, 300
    xc, xs = 780, 880           # cathode, strips
    o = [f'<rect x="{xc}" y="50" width="{xs - xc}" height="500" fill="#eef1f6" stroke="{RULE}"/>',
         sd.T(xc - 8, 40, 'cathode', 20, MUT, 'end'), sd.T(xs + 8, 40, 'strips', 20, MUT, 'start'),
         sd.T((xc + xs) / 2, 578, '30 mm drift gap', 20, MUT),
         f'<circle cx="{cx}" cy="{cy}" r="30" fill="#f3e3c4" stroke="{GOLD}" stroke-width="2"/>',
         sd.T(cx, cy + 62, '³He capsule', 22, INK), sd.T(cx, cy + 88, 'r = 10 mm', 20, MUT)]
    # common-vertex pair: exaggerated opening angle
    for yend, c in ((262, BLUE), (318, ORANGE)):
        k = (yend - cy) / (xs - cx)
        o.append(sd.line(cx, cy, xs + 50, cy + k * (xs + 50 - cx), c, 4))
    o.append(sd.T(470, 250, 'opening angle θ', 22, INK, 'middle', 600,
                  tip='The pair’s opening angle. For two legs of one vertex it is also their relative '
                      'angle inside the chamber: straight lines keep their angle.'))
    o.append(sd.arrow(xs + 70, 262, xs + 70, 318, INK, 2, 9) + sd.arrow(xs + 70, 318, xs + 70, 262, INK, 2, 9))
    o.append(sd.T(xs + 84, 296, 'd = Lθ', 22, INK, 'start', 600))
    o.append(sd.arrow(cx, 600, xs, 600, MUT, 2, 10) + sd.arrow(xs, 600, cx, 600, MUT, 2, 10))
    o.append(sd.T((cx + xs) / 2, 630, f'L = {PA.L_STRIP:.0f} mm, beam axis to strips', 21, MUT,
                  tip='MM_DIST_Z 204.5 mm (mylar face) + W_STRIP 30.1 mm, ntof_tracking/reco/geometry.py. '
                      'Drawn not to scale: the gap is exaggerated 4×, the angle much more.'))
    # unrelated pair crossing in the gap
    o.append(sd.line(xc - 60, 420, xs + 40, 520, GREY, 3.5, '9 6'))
    o.append(sd.line(xc - 60, 515, xs + 40, 430, GREY, 3.5, '9 6'))
    o.append(sd.T(xc - 70, 470, 'unrelated tracks:', 21, MUT, 'end'))
    o.append(sd.T(xc - 70, 496, 'any relative angle', 21, MUT, 'end'))
    schem = sd.svg(W, H, ''.join(o), 'pair geometry')

    f = _gap_frac()
    ds = (1, 3, 6, 12, 24, 50)
    rows = [[f'{d:g} mm', f'{np.degrees(d / PA.L_STRIP):.2f}°', f'{f * d:.2f} mm'] for d in ds]
    tips = [f'{PA.GAP_MM:.0f}·d/L = {f:.3f} × {d:g} mm' for d in ds]
    tab = sd.table(['separation d', 'θ = d / L', 'divergence over 30 mm'], rows, size=24,
                   widths=[200, 170, 280], align=['right', 'right', 'right'], tips=tips)
    right = sd.col(
        sd.p('Two legs of one vertex are straight lines from the same point. Their angle '
             'inside the chamber <i>is</i> the opening angle, and it fixes their separation: '
             f'<b>d ≈ Lθ</b>. Across the drift gap the separation changes by only '
             f'<b>{100 * f:.0f} %</b>.', 26),
        tab,
        sd.callout('So for a common-vertex pair, <b>“close” and “parallel” are the same thing</b>. '
                   'A sharp angle between close tracks needs two vertices: an unrelated pair.', BLUE, 25),
        gap=24, w=640)
    body = sd.title('A common vertex makes close pairs parallel',
                    'Relative angle and separation are one variable for a pair from the capsule.')
    body += sd.row(schem, right, gap=40)
    D.slide('angle-geometry', body, f'''
<p>The question this section answers: the bench bins pairs by separation, but two tracks at a sharp relative angle should be much easier to split than two parallel ones. What angle do the modelled pairs have, how much does angle help, and what angle do real pairs have?</p>
<p>Geometry first. A pair from one vertex at distance L from the strip plane, with opening angle θ, lands d ≈ Lθ apart. Both legs are straight, so their relative angle inside the chamber is θ too. Over the {PA.GAP_MM:.0f} mm gap, which is nearer the source, the separation shrinks by a factor (L − 30)/L. The change is {100 * f:.1f} % of d: 1.5 mm at 12 mm, 0.4 mm at 3 mm.</p>
<p>Off-centre vertices change this little. The capsule is 10 mm in radius, so in the x view a vertex shifts the pointing by at most ±10 mm out of {PA.L_STRIP:.0f}. Along the beam (y view) the capsule is 80 mm long, but both legs of one pair share the vertex, so its position does not enter their relative angle.</p>
<p>Not modelled: multiple scattering between the vertex and the chamber. Scattering in the capsule wall changes the opening angle, but the scattered legs still come from (nearly) one point, so d ≈ Lθ still holds. Only scattering close to the chamber (window, gas) breaks it.</p>''',
            short='Angle: geometry')


# --------------------------------------------------------------------------- #
def s_oracle(D):
    od = PA.out_dir()
    f = od / 'r5_oracle.parquet'
    if not f.exists():
        part = od / 'r5_oracle.partial.parquet'
        body = sd.title('With truth known: does angle help? (pending)',
                        'Synthetic oracle at fixed mesh separation and controlled divergence is running.')
        body += sd.callout(f'<code>pair_angle oracle</code> has not finished '
                           f'({len(pd.read_parquet(part)) if part.exists() else 0} pairs so far). '
                           'Rebuild the deck when it has.', GOLD, 26)
        D.slide('angle-oracle', body, '', short='Angle: oracle')
        return
    T = PA.oracle_table(pd.read_parquet(f))
    As = pd.read_csv(od / 'r5_asimov.csv') if (od / 'r5_asimov.csv').exists() else None
    div_tick = [(0, '0'), (0.5, '0.5'), (1, '1'), (1.5, '1.5'), (2, '2'), (3, '3')]
    panels = []
    for arm in ('A', 'C'):
        P = sd.Plot(810, 560, x=(0, 3), y=(0, 1), title=f'chamber {arm}',
                    xlabel='separation at the strips [mm]', ylabel='pairs resolved' if arm == 'A' else '')
        P.xticks(div_tick).yticks(PCT_TICKS)
        for Dv, c in D_COL.items():
            g = T[(T.arm == arm) & np.isclose(T.D, Dv)].sort_values('d')
            tips = [f'diverging by {Dv:g} mm over the column\nd = {d:g} mm: {100 * e:.0f} % of {n}'
                    f'\nr.m.s. separation {r:.2f} mm' for d, e, n, r in zip(g.d, g.eff, g.n, g.rms)]
            if As is not None:
                lam = As[(As.arm == arm) & np.isclose(As.D, Dv)].set_index('d').lam
                tips = [t + f'\nnoise-free Δχ² λ = {lam.get(d, np.nan):,.0f}' for t, d in zip(tips, g.d)]
            P.line(list(g.d), list(g.eff), c, 4 if Dv == 0 else 3.2, None, r=5.5, tips=tips,
                   tip=f'diverging by {Dv:g} mm')
        g = T[(T.arm == arm) & (T.D < 0)].sort_values('d')
        tips = [f'crossing at mid-column (D = {Dv:g} mm)\nd = {d:g} mm: {100 * e:.0f} % of {n}'
                for d, Dv, e, n in zip(g.d, g.D, g.eff, g.n)]
        P.line(list(g.d), list(g.eff), RED, 3.2, '9 6', r=5.5, tips=tips, marker='open',
               tip='crossing: the separation reverses over the column')
        f_ = _gap_frac()
        P.raw(sd.T(P.X(1.5), P.Y(0.06), f'common vertex: divergence ≈ {f_:.2f} d (≤ {3 * f_:.1f} mm here)',
                   20, GREEN, tip='A pair from one point diverges by 30/L ≈ 0.13 of its separation, so over '
                                  'this whole x range it sits on the parallel (darkest) curve.'))
        panels.append(P.svg(f'oracle {arm}'))
    leg = sd.legend([(f'diverging {Dv:g} mm', c) for Dv, c in D_COL.items()]
                    + [('crossing', RED, 'dash')], size=22)
    # headline from the table: smallest divergence resolving ≥ 90 % at d = 0 in both chambers
    z = T[np.isclose(T.d, 0) & (T.D > 0)]
    ok = z[z.eff >= 0.9].groupby('arm').D.min()
    dmin = float(ok.max()) if len(ok) == 2 else np.nan
    par = T[np.isclose(T.D, 0)].set_index(['arm', 'd']).eff
    ttl = (f'Angle does help: a {dmin:g} mm divergence resolves pairs that touch at the strips'
           if dmin == dmin else 'Angle helps at the smallest separations')
    sub = ('Perfect forward model, ideal fit, white noise at 13.3 ADC, view x, tan 0.3. Track b diverges from a '
           'by D over the drift column. 40 pairs per point; threshold at 1 % false splits.')
    body = sd.title(ttl, sub) + leg + sd.row(*panels, gap=44)
    p05 = ', '.join(f'{a} {100 * par.get((a, 0.5), np.nan):.0f} %' for a in ('A', 'C'))
    D.slide('angle-oracle', body, f'''
<p>This is R2’s oracle with one change: the second track is tilted relative to the first. Its slope differs by D / 840 ns, so the two lines separate by D more at the far end of the 840 ns column than at the strips. D = −2d is a symmetric crossing at mid-column. Truth is known, the fitter is ideal (one- and two-track fits from the truth and from a broad start set), and the threshold is R2’s own 99th percentile of Δχ² on 400 synthetic singles at tan 0.3 (view x), so this ladder sits on R2’s and is directly comparable.</p>
<p>Parallel pairs (D = 0) reproduce R2: {p05} at 0.5 mm. Divergence adds information at every separation. Its effect is largest where the pair touches at the strips. Above ~2 mm everything is resolved anyway.</p>
<p>Noise-free Δχ² (hover a point): at d = 0, λ grows from ~3 (D = 1 mm) to ~40–60 (2 mm) and ~250–330 (4 mm). The 1 % threshold is ~25.</p>
<p>The catch is the green line. A common-vertex pair diverges by only 13 % of its separation, so on this axis it lives on the parallel curve. The angle carries a lot of information, but the pairs we want do not have much of it.</p>
<p>Source: <code>pair_angle.py asimov</code>, <code>oracle --n 40</code> → <code>r5_asimov.csv</code>, <code>r5_oracle.parquet</code>.</p>''',
            foot='R5 · pair_angle.py. Hover a point for k/n, the r.m.s. separation and the noise-free Δχ².',
            short='Angle: oracle')


# --------------------------------------------------------------------------- #
def s_pointing(D):
    od = PA.out_dir()
    Dn = pd.read_parquet(od / 'donors.parquet')
    PT = pd.read_csv(od / 'pointing.csv').set_index(['arm', 'view'])
    panels = []
    for v, lab in (('x', 'x view (across the beam)'), ('y', 'y view (along the beam)')):
        P = sd.Plot(810, 540, x=(0, 400), y=(-0.65, 0.65), title=lab,
                    xlabel='track position at the strips [mm]', ylabel='fitted tan θ' if v == 'x' else '')
        P.xticks([(t, str(t)) for t in range(0, 401, 100)]).yticks(
            [(t, f'{t:+.1f}' if t else '0') for t in (-0.6, -0.3, 0, 0.3, 0.6)])
        for arm in ('C', 'A'):
            g = Dn[Dn.arm == arm]
            r = PT.loc[(arm, v)]
            P.scatter(list(g[f'{v}_p0']), list(g[f'{v}_tan_theta']), ARM_COL[arm], 3.5, 0.3,
                      tip=f'chamber {arm}, {len(g)} clean single tracks')
            xs = np.percentile(g[f'{v}_p0'], [3, 97])
            P.line(list(xs), list(r.slope * xs + r.slope * -r.foot), ARM_COL[arm], 3, '10 6', markers=False,
                   tip=f'chamber {arm}: slope 1/{r.L_eff:.0f} mm, foot at {r.foot:.0f} mm, '
                       f'scatter ±{r.resid_rsig:.3f} (robust σ), r = {r["corr"]:.2f}')
        a, c = PT.loc[('A', v)], PT.loc[('C', v)]
        P.raw(sd.T(P.X(390), P.Y(0.56), f'scatter about the line: A ±{a.resid_rsig:.2f}, C ±{c.resid_rsig:.2f}',
                   21, INK, 'end'))
        panels.append(P.svg(f'pointing {v}'))
    leg = sd.legend([('chamber A', BLUE), ('chamber C', ORANGE), ('fitted pointing line', INK, 'dash')])
    ax, ay = PT.loc[('A', 'x')], PT.loc[('A', 'y')]
    body = sd.title(f'Tracks point at the capsule: slope is tied to position in x, loosely in y',
                    'The bench’s donors: clean single tracks, run_145 stat090_0000, both slopes measured, '
                    'pointing within 30 mm of the beam axis.')
    body += leg + sd.row(*panels, gap=44)
    D.slide('angle-pointing', body, f'''
<p>Every donor of the overlay bench is shown, in the model’s own units (the fitted tan θ = w / v that the forward model uses, before the per-arm angle scale k). The line is a straight-line fit of tan θ against position. Its slope is 1/L<sub>eff</sub>: A x 1/{ax.L_eff:.0f} mm, C x 1/{PT.loc[("C", "x")].L_eff:.0f} mm. That is flatter than the geometric 1/{PA.L_STRIP:.0f} mm because fitted slopes are compressed by charge sharing (the angle scale k, CLAUDE.md and campaign_angle).</p>
<p><b>x view:</b> scatter ±{ax.resid_rsig:.3f} (A), ±{PT.loc[("C", "x")].resid_rsig:.3f} (C). That is about the single-track slope resolution, plus at most ±10 mm/{PA.L_STRIP:.0f} from the capsule radius. The slope is essentially fixed by the position.</p>
<p><b>y view:</b> scatter ±{ay.resid_rsig:.2f} (A), ±{PT.loc[("C", "y")].resid_rsig:.2f} (C), four times wider. The pointing cut is a distance to the beam <i>axis</i>, so along the beam the vertex is free: the 80 mm capsule and anything else on the axis. Two donors from different triggers at the same y position can therefore differ in slope by ~0.2–0.3.</p>
<p>That matters for the next slide. The bench pairs donors from <b>different</b> triggers, so their vertices differ. A real pair shares one vertex.</p>''',
            foot='intra_bench donors.parquet · pair_angle.py data → pointing.csv. Robust σ = half the 16–84 % range.',
            short='Angle: pointing')


# --------------------------------------------------------------------------- #
def s_bench_angle(D):
    od = PA.out_dir()
    P3 = pd.read_parquet(od / 'r3_pairs_angle.parquet')
    EV = pd.read_parquet(od / 'bench_events_angle.parquet')
    f_ = _gap_frac()
    # left: divergence vs mesh separation, A, both views
    S = sd.Plot(760, 560, x=(0, 24), y=(0, 14), title='chamber A overlay pairs, per view',
                xlabel='separation at the strips [mm]', ylabel='divergence over 30 mm  [mm]')
    S.xticks([(t, str(t)) for t in (0, 6, 12, 18, 24)]).yticks([(t, str(t)) for t in (0, 3, 6, 9, 12)])
    for v, c in (('y', PURPLE), ('x', BLUE)):
        g = P3[(P3.arm == 'A') & (P3.plane == v)]
        S.scatter(list(g.sep), list(g['div']), c, 5, 0.45,
                  tip=f'view {v}: {len(g)} pairs; median divergence {g["div"].median():.1f} mm')
    S.line([0, 24], [0, 24 * f_], GREEN, 4, None, markers=False,
           tip=f'common vertex: divergence = {f_:.3f} × separation (geometric)')
    S.band([0, 24], [0, 0], [1.9, 1.9], GREY, 0.12,
           tip='±1.9 mm: the r.m.s. a divergence estimate carries from the two donors’ own slope '
               'errors (√2 × 30 mm × 0.045).')
    S.raw(sd.T(S.X(23.5), S.Y(24 * f_) - 12, 'common vertex', 21, GREEN, 'end', 600))
    S.raw(sd.T(S.X(0.5), S.Y(1.9) - 8, 'slope-error floor', 20, MUT, 'start'))
    # right: ideal fit and fixed chain on A, < 1 / 1-3 / 3-12 mm, by divergence class
    P3 = P3.assign(cls=np.where(P3['div'] < 1.0, 'lt1', np.where(P3['div'] < 3.0, 'mid', 'ge3')))
    bands = [(0, 1, '&lt; 1 mm'), (1, 3, '1–3 mm'), (3, 12, '3–12 mm')]
    cls_col = {'lt1': INK, 'mid': BLUE, 'ge3': GOLD}
    cls_lab = {'lt1': 'divergence &lt; 1 mm', 'mid': '1–3 mm', 'ge3': '≥ 3 mm'}
    panels = []
    for key, ttl in (('real_ok', 'ideal fit on the real windows'), ('fixed_ok', 'fixed chain')):
        B = sd.Plot(440, 420, x=(-0.5, 2.5), y=(0, 1), title=ttl, margin=(24, 16, 80, 92 if key == 'real_ok' else 24),
                    ylabel='resolved, chamber A' if key == 'real_ok' else '')
        B.yticks(PCT_TICKS if key == 'real_ok' else [(t, '') for t, _ in PCT_TICKS])
        for i, (lo, hi, bl) in enumerate(bands):
            B.raw(sd.T(B.X(i), B.y0 + B.ph + 34, bl.replace('&lt;', '<'), 21, INK))
            for j, k in enumerate(('lt1', 'mid', 'ge3')):
                g = P3[(P3.arm == 'A') & (P3.sep >= lo) & (P3.sep < hi) & (P3.cls == k)]
                if len(g) < 5:
                    continue
                e = float(g[key].astype(float).mean())
                B.vbar((B.X(i) + (j - 1) * 34,), e, 30, cls_col[k],
                       tip=f'{ttl}, {bl.replace("&lt;", "<")} at the strips, {cls_lab[k].replace("&lt;", "<")}\n'
                           f'{int(g[key].astype(float).sum())}/{len(g)} = {100 * e:.0f} % (x and y views)')
        panels.append(B.svg(ttl))
    leg = sd.legend([(cls_lab[k], cls_col[k], 'box') for k in ('lt1', 'mid', 'ge3')], size=22)
    # event-level: share of coincident bench overlays below 12 mm with either view diverging >= 3 mm
    ev = EV[(EV.cls == 'coincident')]
    sep = np.minimum(ev.sep_x, ev.sep_y)
    close = ev[sep < 12]
    frac3 = float((close[['div_x', 'div_y']].max(axis=1) >= 3).mean())
    body = sd.title('Bench pairs diverge more than a common-vertex pair would',
                    'Real-overlay pairs (R3, well-modelled windows), divergence from the two donors’ fitted slopes.')
    note = sd.callout(f'At event level, <b>{100 * frac3:.0f} %</b> of coincident bench overlays closer than 12 mm '
                      'have a view diverging by ≥ 3 mm, mostly y. A pair from one vertex diverges by ≤ 1.5 mm there.',
                      GOLD, 23)
    body += sd.row(S.svg('divergence vs separation'),
                   sd.col(leg, sd.row(*panels, gap=8), note, gap=10), gap=36)
    g = P3[P3.arm == 'A']
    def eff(key, lo, hi, k):
        h = g[(g.sep >= lo) & (g.sep < hi) & (g.cls == k)]
        return f'{100 * h[key].astype(float).mean():.0f} % ({len(h)})'
    D.slide('angle-bench', body, f'''
<p>Left: every per-view pair of the R3 real-overlay bench in chamber A, at its separation at the strips, with the divergence 30 mm × |Δtan θ| from the two donors’ own fits. The green line is where a pair from one vertex would sit (30/L = {f_:.3f} × d). x-view pairs (blue) scatter about it within the slope-error floor. y-view pairs (purple) spread far above it, because the two donors come from different points along the beam.</p>
<p>Right: the same pairs, resolved or not, split by divergence. Pairs that touch at the strips (&lt; 1 mm) are resolved by the ideal fit {eff("real_ok", 0, 1, "lt1")} when parallel and {eff("real_ok", 0, 1, "ge3")} when diverging ≥ 3 mm (n in brackets). The data says the same as the oracle: angle helps exactly where separation fails. A common-vertex pair is in the parallel class, so <b>below ~3 mm the bench average is optimistic for genuine pairs</b>; above 3 mm the classes agree.</p>
<p><b>A side finding about the fixed chain.</b> At 3–12 mm it resolves {eff("fixed_ok", 3, 12, "lt1")} of parallel pairs but only {eff("fixed_ok", 3, 12, "ge3")} of strongly diverging ones, where the ideal fit gets {eff("real_ok", 3, 12, "ge3")}. The loss is in the y view. The cause is plausibly the grid search, which seeds <i>parallel</i> line pairs (a common tan); a relative-tan dimension in the grid is the obvious fix. It matters for unrelated pairs, not for common-vertex ones.</p>
<p>Caveat: each divergence is estimated from two fitted slopes, with an r.m.s. error of about 1.9 mm, so the &lt; 1 mm and 1–3 mm classes mix. The ≥ 3 mm class is mostly genuinely diverging.</p>''',
            short='Angle: bench')


# --------------------------------------------------------------------------- #
def s_real(D):
    od = PA.out_dir()
    T = pd.read_csv(od / 'intra_a_sep_angle.csv')
    Sc = pd.read_parquet(od / 'intra_a_scatter.parquet')
    # left: separation distribution, real vs mixed (all pairs)
    H = sd.Plot(760, 540, x=(0, 520), y=(0, 0.05), title='where the pairs land',
                xlabel='separation in the plane [mm]', ylabel='fraction of pairs per 10 mm')
    H.xticks([(t, str(t)) for t in (0, 100, 200, 300, 400, 500)]).yticks(
        [(t, f'{100 * t:.0f}%') for t in (0, 0.01, 0.02, 0.03, 0.04, 0.05)])
    for mixed, c, lab in ((True, GREY, 'mixed across triggers'), (False, BLUE, 'same trigger')):
        g = T[(T.sel == 'all') & (T.mixed == mixed)].sort_values('lo')
        edges = list(g.lo) + [g.hi.iloc[-1]]
        dens = list(g.frac / (g.hi - g.lo) * 10)
        xs_, ys_ = sd.step_xy(edges, dens)
        H.raw(sd.poly([H.X(a) for a in xs_], [H.Y(b) for b in ys_], c, 3.5, None if not mixed else '9 6'))
        for lo, hi, fr, n, dd in zip(g.lo, g.hi, g.frac, g.n, dens):
            H.raw(f'<rect class="hit" x="{H.X(lo):.1f}" y="{H.Y(dd):.1f}" width="{H.X(hi) - H.X(lo):.1f}" '
                  f'height="{H.Y(0) - H.Y(dd):.1f}" fill="transparent"'
                  f'{sd.tipattr(f"{lab}, {lo:g}–{hi:g} mm: {n:,} pairs ({100 * fr:.2f} %)")}/>')
    g0 = T[(T.sel == 'all') & (T.lo < 24)]
    nr = int(g0[~g0.mixed].n.sum())
    fm = float(g0[g0.mixed].frac.sum())
    nall = int(T[(T.sel == 'all') & ~T.mixed].n_sel.iloc[0])
    H.raw(sd.T(H.X(30), H.Y(0.047), f'below 24 mm: {nr} of {nall:,} real pairs', 21, BLUE, 'start', 600))
    H.raw(sd.T(H.X(30), H.Y(0.0435), f'(mixing predicts {100 * fm:.1f} %)', 21, MUT, 'start'))
    # right: opening angle vs separation, slope+pointing, real vs mixed
    A = sd.Plot(800, 540, x=(0, 520), y=(0, 90), title='opening angle (both slopes measured, both pointing)',
                xlabel='separation in the plane [mm]', ylabel='opening angle [°]')
    A.xticks([(t, str(t)) for t in (0, 100, 200, 300, 400, 500)]).yticks([(t, f'{t}°') for t in (0, 30, 60, 90)])
    for mixed, c in ((True, GREY), (False, BLUE)):
        g = Sc[Sc.mixed == mixed]
        A.scatter(list(g.sep_plane_mm), list(g.open_deg), c, 3.2, 0.28 if mixed else 0.4,
                  tip=f'{"mixed" if mixed else "same-trigger"} pairs, thinned to {len(g)}')
    for mixed, c, lab in ((True, GREY, 'mixed'), (False, BLUE, 'same trigger')):
        g = T[(T.sel == 'slope+pointing') & (T.mixed == mixed) & (T.n >= 10)].sort_values('lo')
        mids = list((g.lo + g.hi) / 2)
        tips = [f'{lab}, {lo:g}–{hi:g} mm: median {m:.1f}°, 10–90 % {a:.0f}–{b:.0f}° ({n:,} pairs)'
                for lo, hi, m, a, b, n in zip(g.lo, g.hi, g.q50, g.q10, g.q90, g.n)]
        A.line(mids, list(g.q50), c, 4, None if not mixed else '9 6', r=6, tips=tips, tip=f'{lab}: median')
    xs = np.linspace(0, 2 * PA.L_STRIP * np.tan(np.radians(42)), 40)
    A.line(list(xs), list(np.degrees(2 * np.arctan(xs / 2 / PA.L_STRIP))), GREEN, 4, None, markers=False,
           tip='common vertex on the beam axis: θ = 2 atan(d / 2L)')
    A.raw(sd.T(A.X(xs[-1]) + 8, A.Y(84) + 8, 'common vertex', 21, GREEN, 'start', 600))
    leg = sd.legend([('same trigger', BLUE), ('mixed across triggers (no shared vertex)', GREY, 'dash'),
                     ('common vertex, θ = d / L', GREEN)])
    sl = T[(T.sel == 'slope') & ~T.mixed]
    body = sd.title('Real pairs in chamber A are unrelated tracks at wide angles',
                    f'Every pair of gated tracks in one chamber-A trigger, all post-access production runs '
                    f'({nall:,} pairs), against pairs mixed across triggers.')
    body += leg + sd.row(H.svg('separation'), A.svg('opening angle'), gap=40)
    cs = pd.read_csv(PA.paths.out('det_a_intra') / 'census.csv').set_index('selection')
    D.slide('angle-real', body, f'''
<p>Source: <code>det_a_intra</code>, which pairs every gated, angle-calibrated chamber-A track within a trigger, with a cross-trigger mixed sample as the geometric null (no shared vertex possible). The angles are the production chain’s, from waveforms.</p>
<p><b>Same-trigger pairs look like mixed pairs.</b> They have the same separation distribution and the same opening angles at every separation: median {cs.loc["all", "median_open_deg"]:.0f}° for all pairs, against {cs.loc["mixed", "median_open_deg"]:.0f}° mixed. Among pairs with both slopes measured, {cs.loc["slope", "median_open_deg"]:.0f}° against {cs.loc["slope+mixed", "median_open_deg"]:.0f}°. Only {100 * cs.loc["all", "frac_vertex"]:.3f} % of same-trigger pairs vertex near the axis, against {100 * cs.loc["mixed", "frac_vertex"]:.3f} % of mixed ones. So the two-track sample is dominated by two unrelated particles in one trigger. <b>The intuition that pairs are rarely parallel holds for them:</b> they are far from it.</p>
<p><b>No real pair closer than 24 mm survives production</b> ({nr} of {nall:,}), where mixing predicts {100 * fm:.1f} %. That is the production loss this whole study is about. The data therefore cannot show the angle of close real pairs; it can only show what the population is made of.</p>
<p>Pointing pairs (right) follow the common-vertex line only loosely. The pointing cut is a distance to the beam <i>axis</i> (≤ 30 mm), so a pair can “point” from two different places along the beam. That is the same y-view freedom as on the pointing slide.</p>''',
            foot='det_a_intra pairs.parquet (production chain, chamber A) · pair_angle.py data. Right panel thinned to 1 500 pairs per sample.',
            short='Angle: real pairs')


# --------------------------------------------------------------------------- #
def s_ipc(D):
    od = PA.out_dir()
    F = pd.read_csv(od / 'ipc_intra_sep_fine.csv')
    C = pd.read_csv(od / 'ipc_intra_sep.csv')
    P = sd.Plot(1000, 600, x=(0, 120), y=(0, 1), title='same-chamber IPC pairs: cumulative separation',
                xlabel='separation at the strips [mm]', ylabel='fraction closer than this')
    P.xticks([(t, str(t)) for t in (0, 12, 24, 50, 75, 100, 120)]).yticks(PCT_TICKS)
    P.band([0, 12], [0, 0], [1, 1], RED, 0.08, tip='< 12 mm: the band where the bench says close pairs are hard')
    P.band([12, 24], [0, 0], [1, 1], GOLD, 0.08, tip='12–24 mm')
    cols = dict(M1=BLUE, E0=PURPLE)
    for kind in ('M1', 'E0'):
        g = F[F.kind == kind].sort_values('lo')
        cum = np.cumsum(g.frac.to_numpy())
        xs = list(g.hi)
        tips = [f'{kind}: {100 * c:.1f} % of same-chamber pairs closer than {x:g} mm' if i % 3 == 2 else None
                for i, (x, c) in enumerate(zip(xs, cum))]
        P.line([0] + xs, [0] + list(cum), cols[kind], 4, None, markers=False,
               tip=f'{kind} internal pair conversion, ipc_born')
        P.points([x for x, t in zip(xs, tips) if t], [c for c, t in zip(cum, tips) if t], cols[kind], 4,
                 tips=[t for t in tips if t])
    P.raw(sd.T(P.X(6), P.Y(0.94), '< 12 mm', 21, RED, 'middle', 600))
    c = C.set_index(['kind', 'lo'])
    def below(kind, x):
        return float(C[(C.kind == kind) & (C.hi <= x)].frac_of_intra.sum())
    rows = [[k, f'{100 * c.loc[(k, 0), "intra_of_all"]:.0f} %', f'{100 * below(k, 6):.1f} %',
             f'{100 * below(k, 12):.1f} %', f'{100 * below(k, 24):.1f} %'] for k in ('M1', 'E0')]
    tab = sd.table(['', 'in one chamber', '&lt; 6 mm', '&lt; 12 mm', '&lt; 24 mm'], rows, size=24,
                   widths=[60, 190, 120, 130, 130], align=['left', 'right', 'right', 'right', 'right'])
    m12 = 100 * below('M1', 12)
    side = sd.col(
        sd.p('Where would a genuine pair land? The IPC continuum from <code>ipc_born</code> (the '
             'expected spectrum, CLAUDE.md), thrown from the capsule centre with an isotropic pair axis. '
             'Both legs must cross one strip plane.', 25),
        tab,
        sd.callout(f'Same-chamber IPC pairs are mostly <b>well separated</b>: only {m12:.0f} % (M1) are closer '
                   'than 12 mm. Those few are parallel, and those are the ones the bench average flatters.', BLUE, 24),
        gap=22, w=600)
    body = sd.title(f'Genuine pairs are rarely close: {m12:.0f} % of same-chamber M1 pairs are within 12 mm',
                    'Expected IPC opening-angle law through a point-source toy of the four strip planes.')
    body += sd.legend([('M1 (the measured 55 μb channel)', BLUE), ('E0 (no real photon)', PURPLE)]) + \
        sd.row(P.svg('ipc separation'), side, gap=48)
    D.slide('angle-ipc', body, f'''
<p>This is a prior, not data. For the two s-wave channels open after the flash veto, <code>ipc_born.sample('M1')</code> and <code>e0_lab()</code> give the opening-angle law with its weights. Leg 1 is thrown isotropically and leg 2 at the opening angle with a random azimuth, from the centre of the capsule. A pair counts as same-chamber when both legs cross the same 380 × 340 mm strip plane at L = {PA.L_STRIP:.0f} mm. Pinwheel offsets, dead strips, capsule size and efficiency are ignored.</p>
<p>The X17 itself never lands in one chamber: at 20.6 MeV its opening angle is at least 2 asin(m/E) ≈ 110°, against ~90° for the largest pair one plane can hold.</p>
<p>What the toy leaves out and the real sample may have: <b>external conversions</b> of the 20.6 MeV M1 photon in the capsule wall or the chamber window. Those pairs are extremely collimated (θ ~ m<sub>e</sub>/E, a few mm apart) and exactly parallel. If they matter for an analysis, they live in the hardest corner of this study.</p>
<p>Source: <code>pair_angle.py data</code> → <code>ipc_intra_sep.csv</code>, <code>ipc_intra_sep_fine.csv</code>.</p>''',
            foot='Prior, not a measurement: ipc_born (M1, E0) through pair_angle.ipc_toy. Hover the curves for cumulative fractions.',
            short='Angle: IPC prior')


def add_slides(D):
    od = PA.out_dir()
    if not (od / 'pointing.csv').exists():
        return
    s_geometry(D)
    s_oracle(D)
    s_pointing(D)
    s_bench_angle(D)
    s_real(D)
    s_ipc(D)
