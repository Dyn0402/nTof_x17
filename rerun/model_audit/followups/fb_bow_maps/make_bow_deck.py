#!/usr/bin/env python3
"""Slide note: the MX17 chamber bow (FB follow-up of the model audit, 2026-10-27), figure first.

Reads results/*.json written by xcheck.py, profiles.py, gap_route.py, gate_bow.py, physmodel.py and summary.py, so
re-running those and then this rebuilds every number and chart.  Rows of the consistency DAG (make_fb_dag.py) appear
as soon as their tables exist (build_fb_tables.py); until then the consistency slide says which are pending.

    .venv/bin/python rerun/model_audit/followups/fb_bow_maps/make_bow_deck.py
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
        rerun/model_audit/followups/fb_bow_maps/deck/mx17-chamber-bow.html --slug mx17-chamber-bow --force --deploy
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deckkit import sd, J, fin, hmap, plain_ticks, DCOL, XC, YC  # noqa: E402
from deckkit import BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN, INK, MUT  # noqa: E402
from fbcommon import DETS, DATASETS, FB_ROWS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'deck', 'mx17-chamber-bow.html')
S = J('summary.json')
term = sd.term
SRC = 'rerun/model_audit/followups/fb_bow_maps/'


def gold():
    return [S[d] for d in DETS]


# --------------------------------------------------------------------------- #
def s_cover(D):
    nmu = sum(S[d]['n'] for d in DETS)
    dmin = min(S[d]['depth_mm'] for d in DETS); dmax = max(S[d]['depth_mm'] for d in DETS)
    body = (sd.kicker('MX17 · cosmic bench · model-audit follow-up FB · 27 Oct 2026') +
            '<h1 style="font-size:70px;font-weight:600;line-height:1.08;letter-spacing:-1.5px;color:#eef0f3">'
            f'The readout boards are bowed: on every chamber the mesh centre sits '
            f'{dmin:.1f}–{dmax:.1f} mm below its rim, and the drift lines tilt with it</h1>' +
            sd.row(sd.bignum('5', 'chambers, both strip views', sd.DBLUE, 'det2, det3, det4, det6, det7 on the June bench'),
                   sd.bignum(f'{nmu:,}', 'muons with an M3 track', sd.DGREEN, 'golden rows, M3 recipe χ² < 1, NClus = 4'),
                   sd.bignum('3', 'routes that see it', sd.DGREY,
                             'positions vs M3 · fitted drift tilt · narrowest cluster (no fit at all)'),
                   sd.bignum('1', 'route that cannot', sd.DRED, 'the column duration: its X and Y readings disagree'),
                   gap=48))
    notes = f"""
<p><b>Verdict.</b> The bow is real at the level of "the readout plane is not flat by 1–3 mm and the drift direction
follows its slope"; it is seen by three routes that do not share their systematics, with the sign the geometry
predicts, on every chamber. It is <b>static per chamber</b>: 19 runs (other subruns, other nights, both bench slots, other drift and
mesh voltages) give each chamber the same depth within ~2σ, and the drift-tilt shape measured on the golden run transfers
to the other runs with only the rigid tilt re-measured. Which load makes it (gas overpressure, mesh tension, weight)
cannot be split with the data we have.</p>
<p><b>What it changes:</b> the angle conversion (a position-dependent drift tilt, ±0.6 µm/ns across the chamber) and the
position of the mesh plane (±tan·h, up to ~0.5 mm at tan 0.3). A correction measured from positions alone predicts most of
the angle improvement on held-out tracks.</p>
<p>Source: <code>{SRC}REPORT.md</code>; generator <code>{SRC}make_bow_deck.py</code>.</p>"""
    D.slide('cover', body, notes, dark=True, short='cover')


def side_view(w=1660, h=620):
    """Exaggerated side view: flat cathode, dished mesh, field lines, a vertical and an inclined track."""
    o = []
    x0, x1 = 120, 1100
    yc, ym0 = 110, 470                      # cathode, mesh at the rim
    sag = 70                                # drawn sag (exaggerated ~20x)
    xs = np.linspace(x0, x1, 60)
    mid = 0.5 * (x0 + x1); half = 0.5 * (x1 - x0)
    ym = ym0 + sag * (1 - ((xs - mid) / half) ** 2)
    o.append(sd.line(x0, yc, x1, yc, INK, 4))
    o.append(sd.T(x0 - 14, yc + 8, 'cathode', 22, MUT, 'end'))
    o.append(sd.line(x0, ym0, x1, ym0, GREY, 2, dash='10 8'))
    o.append(sd.T(x0 + 8, ym0 - 12, 'flat plane the alignment assumes', 20, GREY, 'start'))
    o.append(sd.poly(list(xs), list(ym), BLUE, 5))
    o.append(sd.T(x0 - 14, ym0 + 8, 'mesh', 22, BLUE, 'end'))
    o.append(sd.T(mid + 120, ym0 + sag + 46, 'readout board sags: centre lower, further from the cathode', 22, BLUE))
    # field lines: normal to the mesh at the bottom, vertical at the cathode (flat-cathode case)
    for xf in np.linspace(x0 + 60, x1 - 60, 9):
        yb = ym0 + sag * (1 - ((xf - mid) / half) ** 2)
        slope = -2 * sag * (xf - mid) / half ** 2          # dy/dx of the drawn mesh (y down)
        tilt = slope                                        # dx/dy of the normal ~ slope
        ys = np.linspace(yc, yb, 12)
        f = (ys - yc) / (yb - yc)                           # 0 at cathode, 1 at mesh
        xx = xf + tilt * (yb - yc) * (f ** 2) * 0.5 - tilt * (yb - yc) * 0.5
        o.append(sd.poly(list(xx), list(ys), GOLD, 2, dash='6 6'))
    o.append(sd.T(x1 - 40, yc + 40, 'drift lines (dashed) leave the', 21, GOLD, 'end'))
    o.append(sd.T(x1 - 40, yc + 66, 'mesh at right angles to it', 21, GOLD, 'end'))
    # an inclined track meeting the low mesh
    ta, tb = 430, 560
    o.append(sd.line(ta, yc - 40, tb + 10, ym0 + sag + 8, RED, 4))
    o.append(sd.T(ta - 6, yc - 48, 'muon (M3 track)', 22, RED, 'end'))
    # the right column: what each route sees
    cx = 1190
    items = [('P', 'positions', 'the track meets a low mesh later: it lands tan × h away from where M3 points'),
             ('A', 'fitted drift tilt', 'charge arrives displaced along the tilted drift lines: a vertical track looks '
                                        'inclined, by an amount that changes sign across the chamber'),
             ('W', 'narrowest cluster', 'the thinnest cluster is the track parallel to the drift lines, not the '
                                        'vertical one (amplitudes only, no fit)'),
             ('T', 'column duration', 'a deeper gap takes longer to drift through (only if the cathode stays flat)')]
    y = 40
    for k, name, txt in items:
        o.append(f'<circle cx="{cx + 22}" cy="{y + 14}" r="22" fill="{INK}"/>')
        o.append(sd.T(cx + 22, y + 22, k, 24, '#fff', 'middle', 600))
        o.append(sd.T(cx + 60, y + 22, name, 24, INK, 'start', 600))
        words, line_, lines = txt.split(), '', []
        for wd in words:
            if len(line_ + wd) > 40:
                lines.append(line_); line_ = ''
            line_ += wd + ' '
        lines.append(line_)
        for i, L in enumerate(lines):
            o.append(sd.T(cx + 60, y + 52 + 26 * i, L.strip(), 20, MUT, 'start'))
        y += 70 + 26 * len(lines)
    return sd.svg(w, h, ''.join(o), 'side view of a bowed chamber')


def s_setup(D):
    body = (sd.title('What a bowed readout board does to a micro-TPC, and four ways to see it',
                     'Side view, sag exaggerated ~20×. Throughout: hover points and cells for values, counts and errors.') +
            side_view())
    notes = """
<p><b>Frame.</b> M3 z points up; the chambers lie readout-side down with the cathode on top (slide 8 pins both), so
electrons drift down onto the mesh. h(x, y) is the height of the mesh surface above the flat plane the alignment
assumes; a dish has h lowest at the centre.</p>
<p><b>P, positions.</b> A track of slope tan = dx/dz crosses a mesh lowered by −h at a point shifted by tan·h. In every
position cell the slope of the residual (fitted p0 − M3 prediction) against tan is h there. Uses only fitted positions and
M3.</p>
<p><b>A, drift tilt.</b> The fit measures w = dx/dt of the arriving charge. If the drift lines leave the mesh at right angles
to it, charge from height d lands d·dh/dx away, so w gains w0 = v·r_f·dh/dx, with r_f = 1 if the cathode follows the bow
(drift lines tilted all the way up) and r_f = ½ if the cathode stays flat (the tilt falls linearly from the mesh to the
cathode). For a dish, dh/dx changes sign across the chamber: w0 ramps from negative to positive.</p>
<p><b>W, narrowest cluster.</b> The seed cluster (strips above threshold) is narrowest for the track that runs along the
drift lines, at tan = −w0/v, not at tan = 0. This needs no fitted angle and no hit times, only which strips fired.</p>
<p><b>T, column duration.</b> If the cathode stays flat, the gap is deeper at the centre by the dish depth and the
column of charge takes longer to arrive. The ladder slope (w against M3 tan, a pure velocity measurement) separates a
deeper gap from a slower gas.</p>"""
    D.slide('setup', body, notes, short='what a bow does')


def s_positions(D):
    pw, ph = 800, 600
    panels = []
    for p, axis in (('x', 'x'), ('y', 'y')):
        P = sd.Plot(pw, ph, x=(20, 380), y=(-1.5, 4.0), xlabel=f'track {axis} at mid-gap [mm]',
                    ylabel='mesh height h − h(centre) [mm]', title=f'{p.upper()} strips')
        P.xticks(plain_ticks(50, 350, 100)).yticks(plain_ticks(-1, 4, 1, '{:+g}'))
        P.hline(0, GREY, dash='4 4')
        for d in DETS:
            pr = J(f'profiles_{d}.json')[p]
            pr = dict(pr, h=[y if abs(y - np.polyval(pr['h_par'], 0.0)) < 4 else np.nan for y in pr['h']])
            pos, h, he = np.array(pr['pos']), np.array(pr['h']), np.array(pr['h_err'])
            c = np.polyval(pr['h_par'], 0.0)
            tips = [f'{d} {p.upper()}: {x:.0f} mm\nh − h(centre) = {y - c:+.2f} ± {e:.2f} mm\nn = {int(n)}'
                    for x, y, e, n in zip(pos, h, he, pr['n'])]
            P.line(list(pos), [y - c for y in h], DCOL[d], w=3, tips=tips)
        panels.append(P.svg(f'h profile {p}'))
    dmin = min(S[d]['depth_mm'] for d in DETS); dmax = max(S[d]['depth_mm'] for d in DETS)
    leg = sd.legend([(d, DCOL[d]) for d in DETS])
    body = (sd.title(f'Positions alone: the mesh centre sits {dmin:.1f}–{dmax:.1f} mm below the rim, on all five chambers',
                     'h = slope of the position residual (fit − M3) against the M3 track slope, in 9 bins across the chamber') +
            sd.row(*panels, gap=44) + leg)
    rows = ''.join(f'<tr><td>{d}</td><td>{S[d]["slot"]}</td><td>{S[d]["depth_mm"]:.2f} ± {S[d]["depth_err"]:.2f}</td>'
                   f'<td>{S[d]["kbar"] * 100:.0f}</td><td>{S[d]["n"]:,}</td></tr>' for d in DETS)
    notes = f"""
<p>Per bin a Huber line residual = a + h·tan over |tan| &lt; 0.45 (residuals beyond 3 mm of the median dropped); errors from
30 bootstrap resamples. Curves are shifted so the parabola fitted to them is zero at 200 mm. Depth below = ½·k̄·(150 mm)²
from the 2-D fit of slide 4 (k̄ = mean of the four curvature terms, error inflated by their scatter): the height difference
between the centre and a point 150 mm out along one axis; the corners are twice that.</p>
<table><tr><th>chamber</th><th>bench slot</th><th>depth [mm]</th><th>curvature [1/km]</th><th>muons</th></tr>{rows}</table>
<p>The outermost bins (within ~30 mm of an edge) also carry a truncation effect (tracks leaving the active area
through the side lose part of their column), so the fits use 30–370 mm.</p>
<p>Code: <code>{SRC}profiles.py</code>, <code>{SRC}xcheck.py</code>.</p>"""
    D.slide('positions', body, notes, foot='Golden rows of the June bench, finalize reconstruction (promoted phys+mask bundles), '
            'M3 recipe χ² < 1 and NClus = 4.', short='P: positions')


def s_maps(D):
    cm = sd.DIVERGE
    tiles = []
    for d in DETS:
        X = J(f'xcheck_{d}.json')
        col = []
        for p in 'xy':
            M = np.array(X[f'cells_{p}']['h'], float)
            M = M - np.nanmedian(M[1:4, 1:4][np.isfinite(M[1:4, 1:4])]) if np.isfinite(M[1:4, 1:4]).any() else M
            col.append(hmap(M, f'{d} · {p.upper()} strips', -3, 3, cm, w=300, h=300, unit='mm',
                            errs=X[f'cells_{p}']['h_err'], ns=X[f'cells_{p}']['n'],
                            xlabel='x [mm]' if p == 'y' else '', ylabel='y [mm]' if d == 'det2' else '',
                            margin=(16, 10, 50 if p == 'y' else 24, 70 if d == 'det2' else 20)))
        tiles.append(sd.col(*col, gap=6, w=300 if d != 'det2' else 300))
    cb = sd.colorbar(150, 560, -3, 3, [(-3, '−3'), (-1.5, '−1.5'), (0, '0'), (1.5, '+1.5'), (3, '+3')],
                     label='h [mm]', cmap=cm)
    pulls = np.concatenate([np.abs(S[d]['xy_pull'][:2]) for d in DETS])
    body = (sd.title('Two strip layers, two telescope projections, one surface: the dish is seen in both directions by both views',
                     'Mesh height in 70 mm cells from the X-strip residuals (top) and, independently, the Y-strip residuals '
                     '(bottom); blue = lower') +
            sd.row(*tiles, cb, gap=22))
    notes = f"""
<p>Each view measures the mesh height with its own strips and its own M3 projection (x residual against M3 tan_x, y
residual against tan_y), so the two rows of maps share neither the readout layer nor the telescope coordinate. A real
surface must look the same in both, including along the axis a view does not measure (the X strips see the dish along
y through where the tracks cross, not through their own coordinate).</p>
<p>Quantitatively (2-D quadratic fit, event level): the curvature along x from the X view vs from the Y view, and along y
likewise, agree with |pull| ≤ {np.max(pulls):.1f} (median {np.median(pulls):.1f}). The one systematic pattern: the
curvature along a view's <i>own</i> axis reads 20–40 % larger than the same curvature seen by the other view (kww from Y
is larger than kww from X on all five chambers). Edge truncation acts along the own axis only, so the other-axis numbers
are the conservative ones; they are still 1.0–2.2 mm per (100 mm)².</p>
<p>Cells: Huber line residual vs tan in each 70 mm cell (≥ 60 tracks, ≥ 10 of each sign of tan); hover for value,
bootstrap error and count. Maps are shifted to zero at the median of the inner 3 × 3 cells.</p>"""
    D.slide('maps', body, notes, short='P: maps, both views')


def s_xy_pinned(D):
    pw, ph = 800, 600
    A = sd.Plot(pw, ph, x=(0, 3.2), y=(0, 3.2), xlabel='curvature from the X strips [mm per (100 mm)²]',
                ylabel='curvature from the Y strips [mm per (100 mm)²]', title='The two views')
    A.xticks(plain_ticks(0, 3, 1)).yticks(plain_ticks(0, 3, 1))
    A.raw(sd.line(A.X(0), A.Y(0), A.X(3.2), A.Y(3.2), GREY, 2, dash='8 6'), back=True)
    B = sd.Plot(pw, ph, x=(0, 3.2), y=(0, 3.2), xlabel='curvature, fitted t0 [mm per (100 mm)²]',
                ylabel='curvature, t0 pinned per trigger class', title='Free vs pinned t0')
    B.xticks(plain_ticks(0, 3, 1)).yticks(plain_ticks(0, 3, 1))
    B.raw(sd.line(B.X(0), B.Y(0), B.X(3.2), B.Y(3.2), GREY, 2, dash='8 6'), back=True)
    for d in DETS:
        X = J(f'xcheck_{d}.json')
        for k, mk in (('kuu', 'circle'), ('kww', 'diamond')):
            xv, yv = X['pos_x'][k], X['pos_y'][k]
            xe, ye = X['pos_x'][k + '_err'], X['pos_y'][k + '_err']
            A.raw(sd.line(A.X(xv - xe), A.Y(yv), A.X(xv + xe), A.Y(yv), DCOL[d], 2))
            A.raw(sd.line(A.X(xv), A.Y(yv - ye), A.X(xv), A.Y(yv + ye), DCOL[d], 2))
            A.points([xv], [yv], DCOL[d], r=9, marker=mk,
                     tips=[f'{d}, curvature along {"x" if k == "kuu" else "y"}\nX view {xv:.2f} ± {xe:.2f}\n'
                           f'Y view {yv:.2f} ± {ye:.2f}'])
            for p in 'xy':
                f, q = X[f'pos_{p}'][k], X[f'pospin_{p}'].get(k, np.nan)
                B.points([f], [q], DCOL[d], r=8, marker=mk,
                         tips=[f'{d} {p.upper()} view, along {"x" if k == "kuu" else "y"}\nfitted t0 {f:.2f}\npinned t0 {q:.2f}'])
    leg = sd.row(sd.legend([(d, DCOL[d], 'dot') for d in DETS]), sd.p('circles: curvature along x · diamonds: along y', 24, MUT), gap=40)
    rat = np.array([S[d]['kbar_pinned'] / S[d]['kbar'] for d in DETS])
    body = (sd.title('The surface is the same from both views, and does not come from a sliding t0',
                     'Curvature terms of the 2-D fit h(x, y) = … + ½ kxx x² + ½ kyy y²; dashed = equality') +
            sd.row(A.svg('xy'), B.svg('pinned'), gap=44) + leg)
    notes = f"""
<p><b>Left.</b> 20 numbers: for each chamber, the curvature along x and along y, measured once from the X strips and
once from the Y strips. They scatter about the diagonal; the own-axis excess (slide 4 Details) is the visible pattern.</p>
<p><b>Right: could a t0 error fake a height?</b> The fit's t0 and p0 slide together along the track: a t0 error δt moves
p0 by w·δt = v·tan·δt, which reads as a height v·δt. If δt depended on position (per-chip timing, a column-shape
effect), that would fake h(x, y). Moving every p0 along its own track to the median t0 of its trigger-time class (ftst)
removes any such slide. The dish survives: the pinned curvature is {np.min(rat):.2f}–{np.max(rat):.2f} × the fitted one
(chamber means), with more noise (t0 jitter of 50–100 ns adds 2–4 mm × tan of position noise).</p>"""
    D.slide('xy', body, notes, short='P: views agree, t0')


def s_angle(D):
    tiles = []
    for p in 'xy':
        row = []
        for d in DETS:
            pr = J(f'profiles_{d}.json')[p]
            pos = np.array(pr['pos']); w0 = np.array(pr['w0']); we = np.array(pr['w0_err'])
            c = np.nanmedian(w0)
            lo = np.array(pr['w0_pred']['0.5']); hi = np.array(pr['w0_pred']['1.0'])
            P = sd.Plot(316, 330, x=(20, 380), y=(-1.6, 1.6), xlabel=f'{p} [mm]' if p == 'y' else '',
                        ylabel='w0 − median [µm/ns]' if d == 'det2' else '',
                        margin=(14, 10, 56 if p == 'y' else 30, 74 if d == 'det2' else 30),
                        title=f'{d} · {p.upper()}')
            P.xticks([(100, '100'), (300, '300')]).yticks([(-1, '−1'), (0, '0'), (1, '+1')])
            P.band(list(pos), list(np.minimum(lo, hi)), list(np.maximum(lo, hi)), DCOL[d], alpha=0.22,
                   tip='bow predicted from the positions: flat cathode (r_f = ½) to parallel bow (r_f = 1)')
            tips = [f'{d} {p.upper()} {x:.0f} mm\nw0 − median = {y - c:+.2f} ± {e:.2f} µm/ns' for x, y, e in zip(pos, w0, we)]
            P.line(list(pos), [y - c for y in w0], INK, w=2.5, r=5, tips=tips)
            row.append(P.svg(f'w0 {d} {p}'))
        tiles.append(sd.row(*row, gap=10))
    rf = np.array([S[d]['r_f1d'] for d in DETS]).ravel()
    inb = int(np.sum((rf > 0.4) & (rf < 1.2)))
    leg = sd.legend([('measured w0 (angle route)', INK), ('predicted from positions, r_f = ½ … 1', GREY, 'box')])
    body = (sd.title(f'The drift lines tilt the way the positions predict: right sign on 10/10 views, right size on {inb}/10',
                     'w0 = offset of the fitted transverse speed w for a track M3 calls vertical, in 9 bins across the chamber') +
            sd.col(*tiles, gap=4) + leg)
    rows = ''.join(f'<tr><td>{d}</td><td>{S[d]["r_f1d"][0]:.2f}</td><td>{S[d]["r_f1d"][1]:.2f}</td>'
                   f'<td>{S[d]["r_f2d"][0]:.2f}</td><td>{S[d]["r_f2d"][1]:.2f}</td></tr>' for d in DETS)
    notes = f"""
<p>The band is computed from slide 3's parabola alone: w0_pred(p) = b·r_f·dh/dp, with b the local ladder slope (the
drift velocity seen by the fit) and r_f between ½ (cathode flat) and 1 (cathode bowed with the board). Nothing in the band
is fitted to the angles. The sign follows from the frame (z up, cathode up): a dish (h'' &gt; 0) must make w0 rise with
position. It does, on all ten views. That sign is predicted, not chosen.</p>
<p>r_f = measured dw0/dp ÷ (b·h''), per view:</p>
<table><tr><th>chamber</th><th>X (1-D)</th><th>Y (1-D)</th><th>X (2-D fit)</th><th>Y (2-D fit)</th></tr>{rows}</table>
<p>Most views sit at 0.4–0.9, i.e. a mostly flat cathode (the Kapton foil is stretched on the stiff 30 mm frame, the
board is not backed behind the active area). det6 X reads 1.9 in 1-D (0.85 in 2-D): its 1-D position curvature is the
smallest; the angle ramp is normal.</p>
<p><b>What an M3 artefact cannot do:</b> M3 errors that come from bowed telescope planes shift a track by tan·g and vanish
for vertical tracks, so they cannot make w0 (defined at tan = 0) ramp. A differential plane-scale error between the M3
pairs could, but it would need ~14 % of strip-pitch error and would show as a 6 % position scale at the DUT (slide 9).</p>"""
    D.slide('angle', body, notes, short='A: drift tilt')


def s_width(D):
    pw, ph = 800, 560
    panels = []
    for p in 'xy':
        P = sd.Plot(pw, ph, x=(20, 380), y=(-60, 60), xlabel=f'track {p} at mid-gap [mm]',
                    ylabel='track slope of the narrowest cluster [×10⁻³]', title=f'{p.upper()} strips')
        P.xticks(plain_ticks(50, 350, 100)).yticks(plain_ticks(-60, 60, 20, '{:+g}'))
        P.hline(0, GREY, dash='4 4')
        for d in DETS:
            X = J(f'xcheck_{d}.json'); pr = J(f'profiles_{d}.json')[p]
            R = np.array(X[f'width_{p}'], float)
            P.line(list(R[:, 0]), list(1e3 * R[:, 1]), DCOL[d], w=0, r=8,
                   tips=[f'{d} {p.upper()} {x:.0f} mm: narrowest at tan = {1e3 * y:+.0f} ± {1e3 * e:.0f} ×10⁻³\nn = {int(n)}'
                         for x, y, e, n in R])
            pos = np.array(pr['pos']); w0 = np.array(pr['w0']); b = pr['b_med']
            P.line(list(pos), list(-1e3 * (w0 - np.nanmedian(w0)) / b), DCOL[d], w=2.5, dash='8 6', markers=False,
                   tip=f'{d}: −w0/v from the fitted drift tilt (slide 6)')
        panels.append(P.svg(f'width {p}'))
    ws = [S[d]['width'] for d in DETS]
    nsame = sum(1 for w in ws for v in w if v and np.sign(v['slope_per_m']) == np.sign(v['pred_per_m']))
    leg = sd.legend([(d, DCOL[d], 'dot') for d in DETS] + [('dots: narrowest seed cluster (amplitudes only)', INK, 'dot'),
                                                            ('dashed: −w0/v from the fit', INK, 'dash')])
    body = (sd.title(f'With no fit at all, the narrowest cluster leans across the chamber the same way ({nsame}/10 views)',
                     'Per position bin: the M3 slope at which the seed cluster (strips above threshold) is narrowest') +
            sd.row(*panels, gap=44) + leg)
    rows = ''.join(f'<tr><td>{d}</td>' + ''.join(
        f'<td>{v["slope_per_m"]:+.2f}</td><td>{v["pred_per_m"]:+.2f}</td>' if v else '<td>–</td><td>–</td>'
        for v in S[d]['width']) + '</tr>' for d in DETS)
    notes = f"""
<p>A track parallel to the local drift lines puts all its charge on the fewest strips. If the drift lines are vertical
that is the vertical track (tan = 0); if they lean by α, it is the track at tan = −α. Per position bin (6 bins, tracks
above the 30th charge percentile, |tan| &lt; 0.25) the seed width n_seed is fitted with a rounded V,
n = a + c·√((tan − t*)² + 0.02²), on a grid of t*; errors from 20 bootstrap resamples. n_seed is the seeder's count of
strips above threshold: amplitudes only, no hit times and nothing from the waveform fit.</p>
<p>Slope of t* across the chamber [per m] against the prediction −(dw0/dp)/v:</p>
<table><tr><th>chamber</th><th>X width</th><th>X pred.</th><th>Y width</th><th>Y pred.</th></tr>{rows}</table>
<p>Same sign on {nsame}/10; the width route reads somewhat steeper than the prediction on most views (the strip-count
estimator is coarse: one strip is 0.78 mm, the V vertex moves by a fraction of a strip).</p>"""
    D.slide('width', body, notes, short='W: narrowest cluster')


def sign_diagram(w=1660, h=640):
    """Left: the bench stack in M3 z (to scale in z). Right: one chamber as it sits on its rail board."""
    o = []
    z0, z1, y0, y1 = 0, 1350, 600, 30          # z range -> slide y (up = smaller y)
    Y = lambda z: y0 + (z - z0) / (z1 - z0) * (y1 - y0)
    xl, xr = 140, 520
    o.append(sd.arrow(80, Y(0), 80, Y(1330), INK, 3))
    o.append(sd.T(66, Y(700), 'M3 z [mm], up', 22, INK, rot=-90))
    for nm, z in (('M3 top pair', 1302), ('', 1185), ('M3 bottom pair', 144), ('', 24)):
        o.append(f'<rect x="{xl}" y="{Y(z) - 5:.0f}" width="{xr - xl}" height="10" fill="{GREY}"/>')
        o.append(sd.T(xr + 14, Y(z) + 7, f'{nm} z = {z}' if nm else f'z = {z}', 20, MUT, 'start'))
    for nm, z, c in (('upper slot P2', 702, BLUE), ('lower slot P1', 232, BLUE)):
        o.append(f'<rect x="{xl + 30}" y="{Y(z + 30):.0f}" width="{xr - xl - 60}" height="{Y(z) - Y(z + 30):.0f}" '
                 f'fill="{c}" fill-opacity=".18" stroke="{c}" stroke-width="2"/>')
        o.append(sd.T(xr + 14, Y(z) + 2, f'{nm}: mesh at z = {z}', 20, c, 'start', 600))
    o.append(sd.line(300, Y(1330), 360, Y(-10), RED, 3))
    o.append(sd.T(296, Y(1340) - 4, 'cosmic muon (down)', 20, RED, 'end'))
    o.append(sd.T(330, y0 + 34, 'ray files: Z_Up = 1302, Z_Down = 24', 20, INK))
    # right: the chamber
    cx0, cx1 = 900, 1580
    yc, ym, yb = 150, 420, 470
    o.append(f'<rect x="{cx0 - 30}" y="{ym}" width="{cx1 - cx0 + 60}" height="{yb - ym}" fill="#3d8b5a" fill-opacity=".35"/>')
    o.append(sd.T(0.5 * (cx0 + cx1), ym + 34, 'readout board (1.7 mm PCB + 5 mm Rohacell)', 20, INK))
    o.append(f'<rect x="{cx0 - 60}" y="{yb}" width="{cx1 - cx0 + 120}" height="16" fill="{GREY}"/>')
    o.append(sd.T(0.5 * (cx0 + cx1), yb + 44, 'rail board of the slot (top face at z = 227 / 697 mm)', 20, MUT))
    o.append(sd.line(cx0, yc, cx1, yc, INK, 4))
    o.append(sd.T(cx1 + 40, yc + 8, 'cathode', 22, INK, 'start'))
    xs = np.linspace(cx0, cx1, 50); mid = 0.5 * (cx0 + cx1); half = 0.5 * (cx1 - cx0)
    o.append(sd.poly(list(xs), list(ym - 4 + 26 * (1 - ((xs - mid) / half) ** 2) - 26), BLUE, 4))
    o.append(sd.T(cx1 + 40, ym - 18, 'mesh', 22, BLUE, 'start'))
    for xa in (1040, 1240, 1440):
        o.append(sd.arrow(xa, yc + 30, xa, ym - 40, GOLD, 3))
    o.append(sd.T(1240, yc + 110, 'electrons drift DOWN', 22, GOLD, 'middle', 600))
    o.append(sd.line(980, yc - 60, 1120, ym + 10, RED, 3))
    o.append(sd.T(1240, 580, 'later charge comes from higher up: the fitted slope w has the sign', 21, INK))
    o.append(sd.T(1240, 608, 'of the M3 slope on every chamber (kw > 0) → cathode up, readout down', 21, INK))
    return sd.svg(w, h, ''.join(o), 'bench stack and chamber orientation')


def s_sign(D):
    body = (sd.title('Which way: the mesh centre is lower, away from the cathode, from the telescope axis and how the chambers sit',
                     'Each sign is tied to a geometry record or a measurement against the telescope, not to the bow fit') +
            sign_diagram())
    notes = """
<p><b>The chain.</b> (1) M3 z points up: the ray files carry Z_Up = 1302 mm and Z_Down = 24 mm, the heights of the top
and bottom M3 planes, and the upper DUT slot (P2, z = 702) sits above the lower (P1, z = 232); bench levels in
<code>mpgd26/geometry.py</code>. (2) The chambers lie cathode-up: each rests on a rail board with its mesh 5 mm above the
board top, and, as a measurement, the fitted micro-TPC slope w has the same sign as the M3 slope on every chamber
(kw &gt; 0 in every bundle), which for tan = dx/dz_up means the drift depth grows upward. (3) h = d(residual)/d(tan) with
tan = dx/dz_up is a height along z_up: h lowest at the centre = the mesh centre sits below its rim, further from the
cathode. (4) Drift lines normal to such a mesh lean outward-down, so w0 must rise with position: observed on 10/10 views
(slide 6). That last step is a prediction checked by the data, not an input.</p>
<p>All three candidate loads push the centre the same way on the bench: the board's weight (lying horizontal, readout
side down), gas overpressure (pushes the readout board outward = down) and residual mesh tension (contracts the mesh
face, curling the board concave towards the mesh: centre down relative to the clamped rim).</p>"""
    D.slide('sign', body, notes, short='sign')


def s_exclusions(D):
    bu = max(max(abs(S[d]['scale_bu']), abs(S[d]['scale_bw'])) for d in DETS)
    rat = np.array([S[d]['kbar_pinned'] / S[d]['kbar'] for d in DETS])
    rows = [
        ['M3 plane-scale error (a "telescope" mode)',
         'needs a 6 % position scale at the DUT to make the w0 ramp',
         f'measured: ≤ {bu:.2f} % (residual vs position)', '<b style="color:#2f8a5b">excluded</b>'],
        ['t0 sliding along the track',
         'would vanish when p0 is moved to a pinned t0',
         f'pinned keeps {np.min(rat) * 100:.0f}–{np.max(rat) * 100:.0f} % of the curvature', '<b style="color:#2f8a5b">excluded as the cause</b>'],
        ['Fringe field at the frame (a lens)',
         'decays as exp(−πs/30 mm): e-fold 10 mm from the edge',
         'the ramp is linear over 300 mm, flat cathode-to-mesh', '<b style="color:#2f8a5b">excluded</b>'],
        ['M3 planes bowed themselves',
         'shifts every track by tan·g: fakes h, but no w0 at tan = 0',
         'w0 ramp and width route are immune; a common part of h cannot be split off',
         '<b style="color:#c8601a">bounds the size, not the sign</b>'],
        ['Selection / edge truncation',
         'acts along each view\'s own axis near the edges',
         'own-axis curvature 20–40 % above other-axis; both positive', '<b style="color:#c8601a">a 20–40 % systematic</b>'],
    ]
    tips = ['bu, bw of the event-level fit residual = a + bu·U + bw·W + tan·h(U, W): a scale error ε gives bu = ε·100 mm',
            'p0 moved along its own track to the median t0 of its ftst class',
            'Laplace: a perturbation at the side wall of a parallel-plate gap of depth d decays as exp(−π s/d)',
            'for a track measured with M3 planes all displaced by g(x, y) in z, every M3 position shifts by tan·g',
            'see slide 4 Details']
    body = (sd.title('What else could make it: each alternative makes a prediction the data reject, except one that only bounds the size',
                     'Alternatives to a bowed readout board and the test that answers each') +
            sd.table(['alternative', 'what it would do', 'measured', 'verdict'], rows, size=24,
                     widths=[380, 520, 520, 240], align=['left', 'left', 'left', 'left'], tips=tips))
    notes = f"""
<p><b>M3 plane scale.</b> A tan error proportional to position (what a linear w0 ramp would need from the telescope) requires
M3 to mis-scale positions differently at its upper and lower plane pairs. With 1150 mm between the pairs and a ramp of
~1×10⁻⁴ per mm, that is ~14 % of differential pitch, impossible for etched strips, and it would leave a ~6 % position
scale at a DUT between the pairs. The residual scale terms bu, bw are ≤ {bu:.2f} mm per 100 mm on every chamber.</p>
<p><b>M3 planes bowed.</b> If all four M3 planes sagged by the same g(x, y), every reconstructed track would be displaced by
tan·g at every z, with its slope unchanged: the position route would read h_DUT − g. This cannot make the angle (w0) or
width routes move, so those two prove the chamber's own drift lines tilt. But a common g could change the size of the
position-route dish; the ratio r_f (slide 6) would then absorb it. Within r_f between ½ and 1, the chamber's own dish is
between ~0.5× and ~1.3× the position number.</p>"""
    D.slide('exclusions', body, notes, short='alternatives')


def s_column(D):
    tiles = []
    for d in ('det2', 'det3', 'det7'):
        G = J(f'gap_{d}.json')
        col = []
        for p in 'xy':
            M = np.array(G[p]['uend_q']['gap'], float)
            col.append(hmap(M, f'{d} · {p.upper()} strips', 27, 33, sd.VIRIDIS, w=300, h=300, unit='mm', fmt='{:.1f}',
                            errs=G[p]['uend_q']['gap_err'], xlabel='x [mm]' if p == 'y' else '',
                            ylabel='y [mm]' if d == 'det2' else '',
                            margin=(16, 10, 50 if p == 'y' else 24, 70 if d == 'det2' else 20)))
        tiles.append(sd.col(*col, gap=6, w=300))
    cb = sd.colorbar(140, 560, 27, 33, [(27, '27'), (29, '29'), (31, '31'), (33, '33')], label='gap [mm]')
    # predicted vs measured gap curvature, both views, all chambers
    P = sd.Plot(560, 600, x=(-4, 4), y=(-4.5, 4.5), xlabel='−h curvature (prediction if cathode flat)',
                ylabel='gap curvature from T × b', title='Same column, two views')
    P.xticks(plain_ticks(-4, 4, 2, '{:+g}')).yticks(plain_ticks(-4, 4, 2, '{:+g}'))
    P.raw(sd.line(P.X(-4), P.Y(-4), P.X(4), P.Y(4), GREY, 2, dash='8 6'), back=True)
    P.hline(0, GREY, dash='3 5', w=1)
    for d in DETS:
        G = J(f'gap_{d}.json')
        for p, c in (('x', XC), ('y', YC)):
            for k, kk in ((3, 'h_kuu'), (4, 'h_kww')):
                gv = G[p]['uend_q']['gap_fit'][k]; ge = G[p]['uend_q']['gap_fit_err'][k]
                hv = -G[p][kk]
                P.raw(sd.line(P.X(hv), P.Y(max(gv - ge, -4.5)), P.X(hv), P.Y(min(gv + ge, 4.5)), c, 2))
                P.points([hv], [max(min(gv, 4.4), -4.4)], c, r=7,
                         tips=[f'{d} {p.upper()} view, along {"x" if k == 3 else "y"}\n−h curvature {hv:.2f}\n'
                               f'gap curvature {gv:.2f} ± {ge:.2f}'])
    leg = sd.legend([('X-strip column', XC, 'dot'), ('Y-strip column', YC, 'dot')])
    body = (sd.title('The column duration cannot settle it: X and Y time the same column and disagree by more than the effect',
                     'Gap = column duration × local ladder slope (v), 70 mm cells, charge-matched contained tracks; '
                     'right: all chambers') +
            sd.row(*tiles, cb, sd.col(P.svg('gap vs h'), leg, gap=8, w=560), gap=18))
    notes = f"""
<p><b>Why T × b.</b> The column duration T is gap / v. A v(x, y) map (gas, field) and a gap map both move T. The ladder
slope b (fitted w against M3 tan) is v alone: the micro-TPC slope does not depend on how deep the gap is. So T × b is the
gap with v(x, y) divided out. This is how the gap is separated from FA's v(x, y): the gap uses the ladder only as a
local v and needs no v model.</p>
<p><b>What it shows.</b> On det2 and det3 the X column reads longer in the middle, as a dish over a flat cathode predicts
(1.5–2 mm), and the 2026-07-30 gap study saw the same thing (det2 "flat with a ~2 mm rim droop", det3 "deepest at the
centre"). But X and Y, which time one and the same column, disagree: the fitted gap curvatures of the two views differ by
1–4 mm per (100 mm)², more than the 1–2 predicted. The column end depends on amplitude, on the Y-side resistive tail and
on the 60 ns depth grid, all of which vary across a chamber. With q_uend or (q_u90 − q_u50), with or without charge
matching, the picture is the same. So the column route neither confirms nor excludes the bow; it is listed as a limit,
not as evidence.</p>
<p>Code: <code>{SRC}gap_route.py</code>. The model-free strip-summed column (FA's colsum) is the better estimator if it
gets per-cell statistics.</p>"""
    D.slide('column', body, notes, short='T: column duration')


#: rows whose ANGLE reconstruction is broken that night (ladder slope far below v): positions only
BAD_ANGLE = {'det3_o23': 'ladder slope 4.4 µm/ns instead of ~30: the degraded 6-23 bottom-slot night (registry: skeptical)',
             'det2_o17': 'angle s68 ~10°: the 6-17 night with det2\'s half-dead X (registry: skeptical)'}


def s_consistency(D):
    W, H = 1660, 540
    order = sorted([n for n in DATASETS if n in S], key=lambda n: (DETS.index(S[n]['det']), n != S[n]['det'], n))
    P = sd.Plot(W, H, x=(-0.5, len(order) - 0.5), y=(0, 4.0), xlabel='', ylabel='dish depth at 150 mm [mm]',
                margin=(24, 20, 170, 100))
    P.yticks(plain_ticks(0, 4, 1))
    chi = {}
    for i, n in enumerate(order):
        s = S[n]; c = DCOL[s['det']]
        mk = 'circle' if s['slot'] == 'top' else 'diamond'
        P.raw(sd.line(P.X(i), P.Y(max(s['depth_mm'] - s['depth_err'], 0)), P.X(i), P.Y(min(s['depth_mm'] + s['depth_err'], 4)), c, 3))
        P.points([i], [min(s['depth_mm'], 3.95)], c, r=10, marker=mk,
                 tips=[f'{n}: {s["note"]}\n{s["slot"]} slot, {s["n"]:,} muons\ndepth {s["depth_mm"]:.2f} ± {s["depth_err"]:.2f} mm'])
        P.raw(sd.T(P.X(i), P.y0 + P.ph + 26, n.replace('_', ' ') + (' (golden)' if n == s['det'] else ''), 18, INK, 'end', rot=-35))
        chi.setdefault(s['det'], []).append((s['depth_mm'], s['depth_err']))
    pulls, names_p = [], []
    for d, v in chi.items():
        x = np.array(v); w = 1 / x[:, 1] ** 2; m = np.sum(w * x[:, 0]) / np.sum(w)
        pulls += list((x[:, 0] - m) / x[:, 1])
    pn = [(abs(pl), n) for pl, n in zip(pulls, [n for d in chi for n in order if S[n]['det'] == d])]
    n2 = sum(1 for a_, _ in pn if a_ < 2)
    worst = max(pn)
    leg = sd.row(sd.legend([(d, DCOL[d], 'dot') for d in DETS]),
                 sd.p('circles: top bench slot (z = 702 mm) · diamonds: bottom slot (z = 232 mm)', 24, MUT), gap=40)
    body = sd.title(f'Static per chamber: {n2} of {len(order)} runs (both slots, other nights, other HV) within 2σ of the '
                    f'chamber mean; one outlier, {worst[1].replace("_", " ")} at {worst[0]:.1f}σ',
                    'Dish depth from positions, each run with its own alignment and M3 rays, the chamber\'s golden bundle') + \
        P.svg('consistency') + leg
    rows = ''.join(f'<tr><td>{n}</td><td>{S[n]["det"]}</td><td>{S[n]["slot"]}</td><td>{S[n]["note"]}</td>'
                   f'<td>{S[n]["n"]:,}</td><td>{S[n]["depth_mm"]:.2f} ± {S[n]["depth_err"]:.2f}</td></tr>' for n in order)
    bad = ''.join(f'<li>{k}: {v}</li>' for k, v in BAD_ANGLE.items())
    notes = f"""
<p>Each row: same reconstruction code and the chamber's promoted golden bundle (as the pass's vrefit --keep-v: positions
and w in µm/ns do not depend on the bundle's v), its OWN waveform alignment and M3 rays (rr1, v2). Rows built by
<code>{SRC}make_fb_dag.py</code> on condor and <code>build_fb_tables.py</code>. Error bars are the four-curvature-term
mean, inflated by the terms' scatter.</p>
<table><tr><th>row</th><th>chamber</th><th>slot</th><th>run</th><th>muons</th><th>depth [mm]</th></tr>{rows}</table>
<p>Largest deviation: det3 on 6-25 (bottom slot, 500 V drift) at 0.9 mm, 2σ below the det3 mean; det3 on 6-23, also in
the bottom slot, reads 1.7 mm, so the slot itself does not change the dish. The small HV rows (380–715 muons) scatter
more but agree. Positions need no drift reconstruction, so the position route works even on nights whose angles are
broken:</p><ul>{bad}</ul>"""
    D.slide('consistency', body, notes, short='consistency')


def s_transfer(D):
    T = S.get('_transfer') or {}
    keys = [k for k in T if k.split('->')[1] not in BAD_ANGLE]
    labs = [f'{k.split("->")[1].replace("_", " ")} {p.upper()}' for k in keys for p in 'xy']
    P = sd.Plot(1660, 560, x=(-0.5, len(labs) - 0.5), y=(0, 2.2), ylabel='angle s68 on that run [deg]',
                margin=(24, 20, 130, 100))
    P.yticks(plain_ticks(0, 2, 0.5))
    arms = [('const', 'one w0, one kw, refit on the run', GREY), ('static_plus_tilt', 'golden shape + this run\'s tilt', GREEN),
            ('refit_full', 'this run\'s own full 2-D map', BLUE)]
    g = []
    for i, (k, p) in enumerate([(k, p) for k in keys for p in 'xy']):
        for j, (a, nm, c) in enumerate(arms):
            v = T[k][p][a]['s68_deg']
            P.vbar((P.X(i) + (j - 1) * 13,), min(v, 2.2), 12, c, tip=f'{k} {p.upper()} · {nm}: {v:.3f}°, n = {T[k][p]["n"]:,}')
        P.raw(sd.T(P.X(i), P.y0 + P.ph + 24, labs[i], 17, INK, 'end', rot=-40))
        c0, st, own = (T[k][p][a]['s68_deg'] for a, _, _ in arms)
        if c0 - own > 0.02:
            g.append((c0 - st) / (c0 - own))
    body = (sd.title(f'The golden shape transfers: {100 * np.median(g):.0f} % of each run\'s own-map gain, re-measuring only the tilt',
                     'Held-out angle resolution on each consistency run; the bow shape comes from the golden run only') +
            P.svg('transfer') + sd.legend([(nm, c, 'box') for _, nm, c in arms]))
    notes = f"""
<p>This is the proposed bundle split, tested: the drift-tilt surface A(x, y) (constant removed) measured on the chamber's
golden row is applied unchanged to another run; only the rigid tilt (constant) and kw are re-measured on that run,
two-fold held out. It does as well as a full map refit on the run itself on most rows, including det3 in the other bench
slot two days earlier (6-25: X 1.159 → 0.989°, own map 0.987°). Excluded from the chart: the two nights whose angle
reconstruction is broken (previous slide's Details). Code: <code>{SRC}gate_bow.py --transfer</code>.</p>"""
    D.slide('transfer', body, notes, short='transfer test')


def s_plausible(D):
    ph = S['_phys']
    dep = [S[d]['depth_mm'] for d in DETS]
    rows = [('gravity, edges free to rotate', ph['gravity_simply_supported_dip150_mm'], GREY, 'board + Rohacell weight, 33 Pa'),
            ('gravity, edges clamped', ph['gravity_clamped_dip150_mm'], GREY, ''),
            ('mesh pre-tension 10 N/cm, edges free', ph['mesh_10Ncm_depth_rim150_mm'], PURPLE, '5–15 N/cm: 0.4–1.2 mm'),
            ('gas overpressure 1 mbar, edges free', ph['dip150_per_mbar_simply_supported'], BLUE, 'pushes the board outward = down'),
            ('gas overpressure 3 mbar, edges clamped', 3 * ph['dip150_per_mbar_clamped'], BLUE, ''),
            ('measured, mean of 5 golden rows', float(np.mean(dep)), RED, f'range {min(dep):.1f}–{max(dep):.1f} mm')]
    hb = sd.hbars([(r[0], r[1], r[2], None, r[3]) for r in rows], 2.4, width=700, label_w=560, fmt=lambda v: f'{v:.2f} mm')
    body = (sd.title('Plausible: a 1.7 mm board spanning 40 cm with nothing behind it bows by millimetres under a few mbar',
                     'Dip of the readout board between its centre and a point 150 mm out, per load (Kirchhoff plate; '
                     'plausibility only, not a fit)') + hb)
    notes = f"""
<p>From the CAD (MX17_Geant <code>design/GEOMETRY_FROM_CAD.md</code>): the readout PCB is 1.70 mm, backed by 5 mm Rohacell,
clamped between the 30 mm Al gas frame (410 mm aperture) and an 8 mm Al support plate with a 402 mm through-aperture,
so nothing stiff sits behind the active area. Plate: FR4 E = 20 GPa, ν = 0.3, D = {ph['D_Nm']:.1f} N·m (Rohacell has
~1/300 of FR4's modulus and no second skin, so the PCB alone is taken), span 0.402 m. Square-plate coefficients
(Timoshenko): centre deflection 0.00406 q a⁴/D simply supported, 0.00126 clamped.</p>
<p>Gas overpressure needed for 2 mm: {ph['dp_for_2mm_simply_supported_mbar']:.1f} mbar (free) to
{ph['dp_for_2mm_clamped_mbar']:.1f} mbar (clamped); a bubbler-limited flow gives mbar-level overpressure. Gravity alone is
0.1–0.4 mm. A mesh laminated under 5–15 N/cm, acting on one face of the board, gives 0.4–1.2 mm over ±150 mm if the edges
can rotate, ~0 if fully clamped. The measured 1.5–2.3 mm is inside this range, with gas pressure the only load that can
reach it alone. At n_TOF the chambers stand vertical: the gravity part is gone, the pressure and tension parts are not.</p>"""
    D.slide('plausible', body, notes, short='plausibility')


def s_gate(D):
    G = S['_gate']
    arms = [('const', 'one w0, one kw (today)', GREY), ('map1d', '1-D angle map (ws_widthtilt)', PURPLE),
            ('bow2d', '2-D bow surface + scale', BLUE), ('phys', 'shape from positions + 2 numbers', GREEN)]
    labels = [f'{d} {p.upper()}' for d in DETS for p in 'xy']
    P = sd.Plot(1100, 600, x=(-0.5, len(labels) - 0.5), y=(0, 2.1), ylabel='angle resolution s68 [deg]',
                margin=(24, 20, 70, 100))
    P.yticks(plain_ticks(0, 2, 0.5))
    bw = 18
    for i, (d, p) in enumerate([(d, p) for d in DETS for p in 'xy']):
        for k, (a, nm, c) in enumerate(arms):
            v = G[d][p][a]['s68_deg']
            P.vbar((P.X(i) + (k - 1.5) * (bw + 3),), v, bw, c, tip=f'{d} {p.upper()} · {nm}\ns68 = {v:.3f}°\nn = {G[d][p]["n"]:,} held out')
        P.raw(sd.T(P.X(i), P.y0 + P.ph + 30, labels[i], 20, INK))
    gains = {a: [] for a, _, _ in arms}
    for d in DETS:
        for p in 'xy':
            for a, _, _ in arms:
                gains[a].append(1 - G[d][p][a]['s68_deg'] / G[d][p]['const']['s68_deg'])
    pos = [(G[d][p]['pos_s68_mm']['align'], G[d][p]['pos_s68_mm']['height']) for d in DETS for p in 'xy']
    Q = sd.Plot(520, 560, x=(0.4, 0.85), y=(0.4, 0.85), xlabel='position s68, flat mesh [mm]',
                ylabel='with the height map [mm]', title='Position', margin=(24, 20, 92, 96))
    Q.xticks(plain_ticks(0.4, 0.8, 0.2, '{:.1f}')).yticks(plain_ticks(0.4, 0.8, 0.2, '{:.1f}'))
    Q.raw(sd.line(Q.X(0.4), Q.Y(0.4), Q.X(0.85), Q.Y(0.85), GREY, 2, dash='8 6'), back=True)
    for (a, b), lab in zip(pos, labels):
        Q.points([a], [b], DCOL[lab.split()[0]], r=8, tips=[f'{lab}: {a:.3f} → {b:.3f} mm'])
    leg = sd.legend([(nm, c, 'box') for _, nm, c in arms])
    gp = np.median(gains['phys']) * 100; gb = np.median(gains['bow2d']) * 100; gm = np.median(gains['map1d']) * 100
    body = (sd.title(f'Corrected for the bow, angles sharpen by a median {gb:.0f} % on held-out tracks; '
                     f'the positions alone predict {gp:.0f} %',
                     'Two-fold cross-validation on the golden rows: map from half the events, resolution on the other half') +
            sd.row(sd.col(P.svg('gate'), leg, gap=6, w=1100),
                   sd.col(Q.svg('pos'), sd.legend([(d, DCOL[d], 'dot') for d in DETS], size=20), gap=4, w=520), gap=40))
    tr = S.get('_transfer')
    trs = ''
    if tr:
        trs = ''.join(f'<tr><td>{k}</td>' + ''.join(f'<td>{tr[k][p]["const"]["s68_deg"]:.3f} / {tr[k][p]["static_plus_tilt"]["s68_deg"]:.3f} / '
                                                       f'{tr[k][p]["refit_full"]["s68_deg"]:.3f}</td>' for p in 'xy') + '</tr>'
                      for k in tr)
        trs = ('<p><b>Transfer test</b> (the proposed bundle split): the static shape measured on the golden row, applied to '
               'another run of the chamber with only the rigid tilt and the scale re-measured there. Columns: s68 [deg] with '
               'one w0/kw / golden shape + run tilt / that run\'s own full 2-D map.</p>'
               f'<table><tr><th>golden → row</th><th>X</th><th>Y</th></tr>{trs}</table>')
    notes = f"""
<p>The angle map acts after the waveform fit (<code>wft.reco.tan_of</code> converts the fitted w and p0 into tan), so the gate
is exact on the per-event tables: no refit is needed. Training half = even event ids, test = odd, then swapped.
Coordinates are the production ones (the fit's own p0 + 15 mm × tan, both views), not M3's.</p>
<ul><li><b>const</b>: w0 and kw refit on the training half (today's bundle content).</li>
<li><b>map1d</b>: w0(p) and the angle scale s(p) in 12 bins of the view's own coordinate (the ws_widthtilt angle_map,
in <code>wft/</code> but in no bundle). Median gain {gm:.0f} %.</li>
<li><b>bow2d</b>: w = A(x, y) + B·tan·(1 + s_x x + s_y y), A quadratic in both coordinates (9 numbers). Median gain {gb:.0f} %.</li>
<li><b>phys</b>: A = c0 + v·r_f·∂h/∂p with h(x, y) fitted to the training half's <i>positions</i>; only c0 and r_f come
from the angles. Median gain {gp:.0f} %.</li></ul>
<p>The 1-D map wins on several Y views because it also carries a binned angle scale s(p) (a local v map, FA's
subject), which the quadratic arms model only linearly. Position: subtracting tan·h(x, y) improves the position s68
by 4–15 % (right panel).</p>{trs}
<p><b>With the decided reconstruction (FC, main cef45df):</b> one σ0 = 0.30 mm for both views, per-event σ0 choice off,
pinned t0, 20 ns grid. FC applied the cross-validated bow map in that configuration and it still sharpens the angle:
det2 Y 1.36 → 1.07°, det3 Y 1.01 → 0.88°, det4 Y 1.61 → 1.35°, det6 Y 1.88 → 1.41°, det7 Y 1.65 → 1.36°; X views 0–14 %
(det4 X and det7 X ~0), and it removed det6 X's −8 % steep-track compression
(<code>rerun/model_audit/followups/fc_sigma0_trade/results/fc_trade.json</code>, arm s31). The tables gated here were
made with the promoted bundles (old σ0 handling); the bow observables (p0, w) move negligibly with σ0.</p>
<p>Code: <code>{SRC}gate_bow.py</code>; results <code>results/gate_bow.json</code>, <code>gate_transfer.json</code>.</p>"""
    D.slide('gate', body, notes, short='correction gate')


def s_close(D):
    body = (sd.kicker('what this does not rule out, and what goes into the reconstruction') +
            sd.row(sd.col(sd.p('Not ruled out', 34, '#eef0f3', 600),
                          sd.p('· a common bow of the M3 planes adding to (or taking from) the position-route depth: the '
                               'chamber\'s own dish is 0.5–1.3× the quoted number', 26, sd.DMUT),
                          sd.p('· the mechanism: gas overpressure, mesh tension and weight all push the same way; no '
                               'pressure or torque record exists', 26, sd.DMUT),
                          sd.p('· what the boards do at n_TOF (vertical, other flow): the bench map must not be '
                               'transferred without an in-situ check', 26, sd.DMUT),
                          sd.p('· a direct gap measurement: the column route is limited by its own X–Y systematics', 26, sd.DMUT),
                          gap=18),
                   sd.col(sd.p('Into the bundle (spec to FE)', 34, '#eef0f3', 600),
                          sd.p('· per chamber: a static bow block, the height surface h(x, y) and the drift-tilt '
                               'surface A(x, y) with its constant removed', 26, sd.DMUT),
                          sd.p('· per run: only the rigid tilt (w0) and kw, as today', 26, sd.DMUT),
                          sd.p('· tan_of uses both coordinates of the track; the mesh point of a track is (p0, z + h)', 26, sd.DMUT),
                          sd.p('· gate: the held-out table gate here, then FE\'s full gate; bench only', 26, sd.DMUT),
                          gap=18), gap=80))
    notes = f"""<p>FE spec in <code>{SRC}REPORT.md</code> ("Spec for FE").</p>"""
    D.slide('close', body, notes, dark=True, short='not ruled out')


def main():
    D = sd.Deck('MX17 chamber bow', 'The MX17 readout boards are bowed by 1-3 mm: evidence, sign, consistency, correction')
    s_cover(D); s_setup(D); s_positions(D); s_maps(D); s_xy_pinned(D); s_angle(D); s_width(D); s_sign(D)
    s_exclusions(D); s_column(D); s_consistency(D); s_plausible(D); s_gate(D); s_transfer(D); s_close(D)
    D.write(OUT, note_meta=dict(title='MX17 chamber bow', summary='The readout boards of all five MX17 chambers are bowed by '
                                '1.5-2.3 mm (centre lower): three independent routes, sign from external facts, consistency '
                                'across runs, a plate-model plausibility check and a held-out gate of the correction.',
                                tags='x17, mx17, model audit', date='2026-10-27'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
