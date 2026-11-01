#!/usr/bin/env python3
"""make_deck.py -- the slide note for det4 at SPS H4 measured on one small working region (results/region.json).

    python3 make_deck.py [--out deck/mx17-sps-region.html]
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py deck/mx17-sps-region.html --slug mx17-sps-region --force --deploy

Every number on the slides is read from results/region.json and results/trigger_ref.json (region_analysis.py,
trigger_ref.py), so re-running the analysis and this script updates the note.
"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402
import numpy as np     # noqa: E402

R = json.load(open(os.path.join(HERE, 'results', 'region.json')))
TR = json.load(open(os.path.join(HERE, 'results', 'trigger_ref.json')))
RX, RY = R['region']['x'], R['region']['y']
R2 = R['regions']['R2 (cross-check: X band 201-214)']
RES = R['resolution']
M = R['maps']
EFF = R['efficiency']
HOT = [(154.6, 158.7, 'ch 372–376: oscillating channel and its in-phase neighbours'), (249.9, 250.8, 'ch 510: oscillating channel')]
WIN = (99.84, 299.52)
UM = 1e3


def L(items, **k):
    """legend from (colour, label) pairs"""
    return sd.legend([(b, a, *c) for a, b, *c in items], **k)


def um(x):
    return f'{x * UM:.0f}'


def res(tag, est, region='R'):
    return RES[tag][f'{region}|{est}']


# headline numbers
r63 = {e: res('run63', e) for e in ('p0', 'p_mid', 'c_pos')}
X_Q = r63['p_mid']['X']                 # X: the time-aware estimator (p0 + 7.5 mm tan)
Y_Q = r63['p0']['Y']                    # Y: the wft fit p0 (sps_audit's best Y estimator)
X_B2F2 = X_Q['res_b2f2']; X_B2 = X_Q['res_b2']; Y_B2 = Y_Q['res_b2']
E_BOTH = EFF['eff_region_both']
N_R = r63['p_mid']['n']
F_C = R['f_common']


def pm(o):
    lo, hi = o['sys_range']
    return f'{um(o["sigma_det4"])} <span style="font-size:0.6em">+{um(hi - o["sigma_det4"])} / −{um(o["sigma_det4"] - lo)}</span>'


# ---------------------------------------------------------------- slide 1: cover
def cover(D):
    body = (sd.kicker('det4 (mx17_E) at SPS H4 · run 63 · Ar/CF4/iso 88/10/2 · resist 700 V · 243 V/cm')
            + sd.title(f'One working 10 × 24 mm² patch of det4: σ<sub>Y</sub> ≈ {um(Y_Q["sigma_det4"])} µm, '
                       f'σ<sub>X</sub> ≈ {um(X_Q["sigma_det4"])} µm, {100 * E_BOTH[0]:.0f} % efficient',
                       'Resolution, efficiency, footprint and timing measured where det4 works, instead of across a beam spot '
                       'that crosses dead stripes, three gain bands and three telescope zones', )
            + sd.row(sd.bignum(f'{N_R:,}', 'telescope tracks with det4 in the patch', sd.DBLUE, 'run 63 flat700'),
                     sd.bignum(f'{EFF["eff_tracks"]:,}', 'telescope tracks in the efficiency map', sd.DGREEN, 'every single-cluster track'),
                     sd.bignum('10 × 24', 'mm² selected: X 150–160, Y 222–246', sd.DGREY, 'of a 200 × 200 mm² readout window'),
                     gap=64))
    notes = f'''<p>The whole-beam-spot SPS numbers (<code>rerun/sps_audit</code>) average over everything det4 had at H4: dead
amplification stripes, live bands of different gain (lead-strip charge ×1.6 between them) and the three pitch zones of the
uRWELL telescope's back plane. On det4 X those zones land on different live bands, so the free 3-zone fit that should
separate det4 from the telescope mixes the two and fails (χ²/1 = 7–88). This note picks <b>one region</b> where det4 works
in both views and measures everything there.</p>
<p>Code: <code>rerun/sps_region/</code> (<code>region_analysis.py</code>, <code>eff_job.py</code>, <code>trigger_ref.py</code>,
this deck <code>make_deck.py</code>). Inputs: the final SPS pass telescope tables
(<code>x17rerun/sps_audit/tel/F*</code>, code 3ac0e79), the footprint tables, the Paper I timing strip sums
(code 48244778), and a new run-63 efficiency table with every telescope track (<code>x17rerun/sps_region/eff/</code>).</p>'''
    D.slide('cover', body, notes, dark=True, short='cover')


# ---------------------------------------------------------------- slide 2: the map
def map_slide(D):
    xe, ye = np.array(M['xe']), np.array(M['ye'])
    n, b = np.array(M['eff_n']), np.array(M['eff_both'])
    # 2 mm cells for the zoom
    k = 2
    nx, ny = (len(xe) - 1) // k, (len(ye) - 1) // k
    cells = []
    for i in range(nx):
        for j in range(ny):
            nn = n[i * k:(i + 1) * k, j * k:(j + 1) * k].sum(); bb = b[i * k:(i + 1) * k, j * k:(j + 1) * k].sum()
            if nn < 15:
                continue
            x0, x1, y0, y1 = xe[i * k], xe[(i + 1) * k], ye[j * k], ye[(j + 1) * k]
            cells.append((x0, x1, y0, y1, bb / nn, f'X {x0:.0f}–{x1:.0f}, Y {y0:.0f}–{y1:.0f} mm\nefficiency (both views) '
                          f'{100 * bb / nn:.0f} %\n{int(bb)} of {int(nn)} telescope tracks'))
    # left: the whole chamber, with the instrumented window and the beam spot
    W = sd.Plot(600, 640, x=(0, 400), y=(0, 400), xlabel='det4 X [mm] (measured by the X strips)', ylabel='det4 Y [mm]',
                margin=(20, 20, 92, 100), title='the chamber')
    W.xticks([(v, str(v)) for v in (0, 100, 200, 300, 400)]).yticks([(v, str(v)) for v in (0, 100, 200, 300, 400)])
    W.rect(0, 398.6, 18, 380, sd.GREY, 2, tip='MX17 active area X 0–398.6, Y 18–380 mm (common/mx17_active_area.py)')
    W.rect(WIN[0], WIN[1], WIN[0], WIN[1], sd.INK, 2.5, dash='10 6',
           tip='read out at H4: connectors x_3–x_6 and y_3–y_6 (FEU 3, 4 × 64 strips per view, 0.78 mm pitch)')
    W.cells(cells, 0, 1, sd.VIRIDIS)
    W.rect(RX[0], RX[1], RY[0], RY[1], sd.RED, 3, tip='selected region R')
    W.text(WIN[0] + 4, WIN[1] - 14, 'read-out window', 21, sd.INK)
    W.text(6, 392, 'active area', 21, sd.GREY)
    W.text(150, 132, 'telescope', 21, sd.MUT); W.text(150, 112, 'beam spot', 21, sd.MUT)
    # right: zoom with stripes, hot channels, zones, R and R2
    Z = sd.Plot(860, 640, x=(140, 245), y=(150, 266), xlabel='det4 X [mm]', ylabel='det4 Y [mm]',
                margin=(20, 130, 92, 100), title='zoom on the beam spot')
    Z.xticks([(v, str(v)) for v in range(140, 246, 20)]).yticks([(v, str(v)) for v in range(150, 262, 20)])
    Z.cells(cells, 0, 1, sd.VIRIDIS)
    for lo, hi, t in HOT:
        if hi > 150:
            Z.raw(f'<rect x="{Z.X(140):.1f}" y="{Z.Y(hi):.1f}" width="{Z.X(245) - Z.X(140):.1f}" height="{max(Z.Y(lo) - Z.Y(hi), 3):.1f}" '
                  f'fill="{sd.ORANGE}" fill-opacity="0.45"{sd.tipattr(t)}/>')
    for z in M['zones_det4']['Y']:
        if 150 < z['det4_lo'] < 262:
            Z.raw(sd.line(Z.X(140), Z.Y(z['det4_lo']), Z.X(245), Z.Y(z['det4_lo']), sd.GOLD, 2.5, '10 6'))
        Z.text(246, (max(z['det4_lo'], 150) + min(z['det4_hi'], 262)) / 2, f'Y zone {z["pitch"]:g} mm', 20, sd.GOLD, 'start', 600)
    for z in M['zones_det4']['X']:
        if 140 < z['det4_hi'] < 245:
            Z.raw(sd.line(Z.X(z['det4_hi']), Z.Y(150), Z.X(z['det4_hi']), Z.Y(262), sd.GOLD, 2.5, '10 6'))
    for z in M['zones_det4']['X']:
        xm = (max(z['det4_lo'], 140) + min(z['det4_hi'], 245)) / 2
        Z.text(xm, 260, f'X zone {z["pitch"]:g} mm', 20, sd.GOLD, 'middle', 600)
    Z.rect(RX[0], RX[1], RY[0], RY[1], sd.RED, 4, tip=f'R: X {RX[0]:.0f}–{RX[1]:.0f}, Y {RY[0]:.0f}–{RY[1]:.0f} mm\n{N_R} telescope tracks with det4')
    Z.rect(R2['x'][0], R2['x'][1], R2['y'][0], R2['y'][1], sd.RED, 2.5, dash='8 6',
           tip='R2: cross-check region in the second live band (lower gain, X telescope zone 1.5 mm)')
    Z.text(RX[0] + 0.5, RY[1] + 1.5, 'R', 28, sd.RED, weight=700)
    Z.text(R2['x'][0] + 0.5, R2['y'][1] + 2, 'R2', 24, sd.RED, weight=600)
    Z.text(166, 205, 'dead', 20, '#ffffff', 'middle'); Z.text(166, 199, 'stripe', 20, '#ffffff', 'middle')
    cb = sd.colorbar(130, 470, 0, 1, [(0, '0'), (0.5, '0.5'), (1, '1')], 'efficiency')
    leg = L([(sd.RED, 'R (solid), R2 (dashed)'), (sd.ORANGE, 'oscillating Y channels'), (sd.GOLD, 'telescope back-plane zone edges')], size=21)
    body = (sd.title('The patch R: det4\'s best live band, one telescope zone',
                     'det4 local frame; colour = fraction of telescope tracks det4 sees in both views (run 63, 2 mm cells); R2 = cross-check')
            + sd.row(W.svg('whole chamber'), Z.svg('zoom on the beam spot'), sd.col(cb, leg, gap=24), gap=16, align='start'))
    notes = f'''<p><b>How the map is made.</b> Every single-cluster four-view uRWELL track (<code>eff_job.py</code>, all
{EFF["eff_tracks"]:,} tracks in the run-63 flat700 plateau) is projected to det4 with the same telescope→det4 affine the
resolution uses. A track counts as seen if det4 has a local maximum ≥ 50 ADC (5 × noise) within ±2 mm of the prediction in
X and in Y. The vertical dark bands are det4's dead amplification stripes (fixed, measured on the bench too). At H4 the
chamber was rolled 90°, so in the hall they ran horizontally.</p>
<p><b>Why R = X {RX[0]:.0f}–{RX[1]:.0f}, Y {RY[0]:.0f}–{RY[1]:.0f} mm:</b></p>
<ol><li>X: the live band 148–162 mm is the widest with the highest gain (lead-strip charge ~330 ADC against ~210 in the 201–214 band).
R keeps ≥ 2 mm from both stripe edges.</li>
<li>Y ≥ 222 mm puts both telescope coordinates in the back plane's 1.0 mm pitch zone (edge at Y ≈ 220 mm), so the
telescope pointing is the same for every track in R.</li>
<li>Clear of the oscillating Y channels: ch 372–376 (154.6–158.7 mm) and ch 510 (250.4 mm). Y ends at 246 mm.</li>
<li>In the densest part of the spot: {N_R:,} telescope tracks with a det4 record.</li></ol>
<p>R2 (X {R2["x"][0]:.0f}–{R2["x"][1]:.0f} mm, same Y) is the same recipe in the second band. It is used only as a cross-check.</p>'''
    D.slide('map', body, notes, short='detector map')


# ---------------------------------------------------------------- slide 3: profile across X
def profile_slide(D):
    pe = EFF['eff_profile_x']; px = M['profile_x']
    P = sd.Plot(1040, 620, x=(140, 240), y=(0, 1.05), xlabel='det4 X [mm] (telescope-predicted)',
                ylabel='efficiency, both views (Y 222–246 mm)', margin=(30, 110, 92, 104))
    P.xticks([(v, str(v)) for v in range(140, 241, 20)]).yticks([(v, f'{v:.1f}') for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0)])
    P.y2(1 / 600, [(0, '0'), (100, '100'), (200, '200'), (300, '300'), (400, '400'), (500, '500'), (600, '600')], 'lead-strip charge X [ADC]')
    P.raw(f'<rect x="{P.X(RX[0]):.1f}" y="{P.y0}" width="{P.X(RX[1]) - P.X(RX[0]):.1f}" height="{P.ph}" fill="{sd.RED}" fill-opacity="0.10"/>', back=True)
    P.raw(f'<rect x="{P.X(R2["x"][0]):.1f}" y="{P.y0}" width="{P.X(R2["x"][1]) - P.X(R2["x"][0]):.1f}" height="{P.ph}" fill="{sd.RED}" fill-opacity="0.05"/>', back=True)
    P.line([q['x'] for q in pe], [q['eff'] for q in pe], sd.GREEN, 3, r=4,
           tips=[f'X {q["x"]:.1f} mm: {100 * q["eff"]:.0f} % of {q["n"]} tracks' for q in pe])
    qx = [(q['x'], q['q_x']) for q in px if q['q_x'] is not None and q['agree_x'] > 0.5]
    P.line([a for a, _ in qx], [b / 600 for _, b in qx], sd.BLUE, 3, r=4, dash='6 4',
           tips=[f'X {a:.1f} mm: median lead-strip charge {b:.0f} ADC (tracks det4 reads correctly)' for a, b in qx])
    for z in M['zones_det4']['X']:
        if 140 < z['det4_hi'] < 240:
            P.vline(z['det4_hi'], sd.GOLD, '10 6', 2.5)
        P.text((max(z['det4_lo'], 140) + min(z['det4_hi'], 240)) / 2, 1.0 if z['pitch'] != 0.5 else 0.94, f'zone {z["pitch"]:g} mm', 21, sd.GOLD, 'middle')
    P.text(RX[0] + 5, 0.06, 'R', 28, sd.RED, 'middle', 700); P.text((R2['x'][0] + R2['x'][1]) / 2, 0.06, 'R2', 24, sd.RED, 'middle', 600)
    leg = L([(sd.GREEN, 'efficiency (left axis)'), (sd.BLUE, 'gain: lead-strip charge (right axis)'), (sd.GOLD, 'telescope zone edge')], size=22)
    side = sd.col(sd.callout('Each live band sits in a different telescope zone, so the whole-spot zone fit compares bands, not zones.', sd.GOLD, 26),
                  sd.callout('R: the highest gain, full efficiency, one zone.', sd.RED, 26), gap=24, w=520)
    body = (sd.title('Across the stripes: efficiency, gain and telescope zone change together',
                     'run 63; tracks with Y in R; 1 mm bins of telescope-predicted det4 X')
            + sd.row(sd.col(P.svg('profile across X'), leg, gap=8), side, gap=48))
    notes = '''<p>This is why the whole-spot X resolution could not be decomposed: the telescope's back-plane zone is set by the
track's y on the uRWELL, which at H4 maps onto det4 X. The 1.0 mm zone covers the high-gain band 148–162 mm, the 1.5 mm zone
the lower-gain band 201–214 mm and the 0.5 mm zone only the narrow band at 232 mm. A 3-zone fit of det4 X therefore fits gain
differences as if they were telescope pitch: χ²/1 = 7–88 in <code>rerun/sps_audit</code>. Inside one region the zone is
fixed and the gain is uniform, so the problem goes away. The telescope term then has to come from somewhere else (next slides).</p>
<p>Gain: median charge of the X lead strip for tracks det4 reads within ±1.5 mm of the telescope. It falls from ~400 to
~280 ADC across the 148–162 band and is ~210 ADC in the 201–214 band.</p>'''
    D.slide('profile', body, notes, short='why this patch')


# ---------------------------------------------------------------- slide 4: telescope term
def telescope_slide(D):
    fc = R['f_cal']
    rows = []
    for est, lab in (('p0', 'fit p0'), ('p_mid', 'p_mid'), ('c_pos', 'centroid')):
        for band in ('band 150-160', 'band 201-214'):
            z = {q['pitch']: q for q in fc[est][band]['zones']}
            f = fc[est][band]['fit']
            rows.append([lab, band.replace('band ', 'X '), f'{um(z[1.0]["sigma_res"])} ({z[1.0]["n"]})', f'{um(z[1.5]["sigma_res"])} ({z[1.5]["n"]})',
                         f'{f["f"]:.3f} ± {f["f_err"]:.3f}', f'{um(f["sigma_det4"])} ± {um(f["sigma_det4_err"])}'])
    tab = sd.table(['estimator', 'det4 band', 'Y res. zone 1.0 [µm] (n)', 'zone 1.5 [µm] (n)', 'f = σ<sub>back</sub>/pitch', 'det4 Y [µm]'], rows, 22)
    # cartoon: residual^2 vs pitch^2
    P = sd.Plot(620, 520, x=(0, 2.5), y=(0, 0.16), xlabel='back-plane pitch² [mm²]', ylabel='σ²(residual) [mm²]', margin=(20, 20, 92, 110))
    P.xticks([(0.25, '0.25'), (1, '1'), (2.25, '2.25')]).yticks([(0, '0'), (0.05, '0.05'), (0.1, '0.10'), (0.15, '0.15')])
    wb = (1120 / 1370) ** 2; wf = (1 - 1120 / 1370) ** 2 / 12
    for est, col in (('p0', sd.BLUE), ('c_pos', sd.GREEN)):
        f = fc[est]['band 150-160']['fit']; z = [q for q in fc[est]['band 150-160']['zones'] if q['used']]
        P.points([q['pitch'] ** 2 for q in z], [q['sigma_res'] ** 2 for q in z], col, 9,
                 tips=[f'{est}, Y zone {q["pitch"]} mm: σ {um(q["sigma_res"])} ± {um(q["err"])} µm, n {q["n"]}' for q in z])
        xs = np.linspace(0, 2.5, 30)
        P.line(list(xs), [f['sigma_det4'] ** 2 + wf + wb * f['f'] ** 2 * x for x in xs], col, 2, markers=False, dash='6 4')
    P.text(0.08, 0.145, 'intercept = det4²', 21, sd.INK); P.text(0.08, 0.130, 'slope = (z/z_back)² f²', 21, sd.INK)
    leg = L([(sd.BLUE, 'fit p0'), (sd.GREEN, 'charge centroid')], size=21)
    body = (sd.title(f'The telescope term: f = {F_C["f"]:.3f} ({F_C["range"][0]:.3f}–{F_C["range"][1]:.3f}) from det4 Y inside each live band',
                     'R has one telescope zone, so f is measured on det4 Y over the whole band, where two zones are populated')
            + sd.row(sd.col(P.svg('zone fit'), leg, gap=6), sd.col(tab, sd.p(
                f'Pointing at det4 in zone 1.0 with f = {F_C["f"]:.3f}: <b>{um(X_Q["pointing"])} µm</b> in each coordinate. '
                'The six fits agree within 0.03 in f, so the telescope behaves the same in both bands.', 24), gap=24), gap=48))
    notes = f'''<p>Model (as <code>sps_audit/analyze.py</code>): σ²<sub>res</sub> = σ²<sub>det4</sub> + (z/z<sub>back</sub>)²(f·pitch)²
+ (1 − z/z<sub>back</sub>)²(1 mm/√12)², with z = 1120 mm (det4) and z<sub>back</sub> = 1370 mm. Inside one det4 X band, det4 Y
crosses the 1.5 and 1.0 mm zones of the back plane's x coordinate. The 0.5 mm zone (Y 164–173) has only 50–270 tracks there,
at the edge of the beam and of the band, so it is not used. Two zones and two unknowns: the fit is exact (χ² = 0), and its
errors come from the residual widths. It assumes det4 Y is uniform along Y within the band. The gain profile along Y is
flat there (±10 %).</p>
<p>f is a property of the telescope, so one value, the mean of the six fits, is used for every estimator: f =
{F_C["f"]:.3f}, systematic range {F_C["range"][0]:.3f}–{F_C["range"][1]:.3f}. The whole-spot fits of <code>sps_audit</code> gave
0.25–0.34; their upper end comes from the failed X fits (previous slide). The X view's telescope coordinate is the back
plane's y, assumed to have the same f. The check on the next slide supports this: selecting 2-strip telescope clusters
removes the same ~200 µm (in quadrature) from both views.</p>'''
    D.slide('telescope', body, notes, short='telescope term')


# ---------------------------------------------------------------- slide 5: resolution
def resolution_slide(D):
    bins = np.arange(-1.5, 1.5001, 0.05); c = (bins[:-1] + bins[1:]) / 2
    panels = []
    for v, o, lab in (('X', X_Q, 'X (p_mid)'), ('Y', Y_Q, 'Y (fit p0)')):
        keep = (bins[:-1] >= -1.0 - 1e-9) & (bins[1:] <= 1.0 + 1e-9)
        h = np.array(o['hist'], float)[keep]; h2 = np.array(o['hist_b2'], float)[keep]
        h /= h.max(); h2 /= h2.max()
        eb = list(bins[:-1][keep]) + [bins[1:][keep][-1]]
        P = sd.Plot(800, 560, x=(-1.0, 1.0), y=(0, 1.08), xlabel=f'det4 {v} − telescope [mm]', ylabel='tracks (peak = 1)' if v == 'X' else '',
                    margin=(20, 20, 92, 90), title=f'det4 {lab} in R')
        P.xticks([(-1, '−1'), (-0.5, '−0.5'), (0, '0'), (0.5, '0.5'), (1, '1')]).yticks([(0, '0'), (0.5, '0.5'), (1, '1')])
        xs, ys = sd.step_xy(eb, list(h))
        P.raw(sd.poly([P.X(a) for a in xs], [P.Y(b) for b in ys], sd.GREY, 2.5), back=True)
        xs, ys = sd.step_xy(eb, list(h2))
        P.raw(sd.poly([P.X(a) for a in xs], [P.Y(b) for b in ys], sd.BLUE, 3.5))
        P.text(-0.95, 1.0, f'all tracks: σ {um(o["sigma_res"])} µm', 22, sd.GREY, weight=600)
        P.text(-0.95, 0.92, f'2-strip back cluster: σ {um(o["res_b2"][0])} µm', 22, sd.BLUE, weight=600)
        P.text(-0.95, 0.80, f'det4 alone: {um(o["sigma_det4"])} µm', 26, sd.RED, weight=700)
        panels.append(P.svg(f'{v} residual'))
    leg = L([(sd.GREY, 'every telescope track (single cluster per plane)'), (sd.BLUE, 'telescope back-plane cluster of 2 strips (interpolated)')], size=22)
    body = (sd.title(f'In R det4 resolves σ<sub>X</sub> = {um(X_Q["sigma_det4"])} µm and σ<sub>Y</sub> = {um(Y_Q["sigma_det4"])} µm; '
                     f'≤ {um(X_B2F2[0])} / {um(Y_B2[0])} µm with no telescope model',
                     f'run 63 flat700; residual to the uRWELL track at z = 1120 mm; core σ of a Gaussian in ±1.2 mm; other view within ±1.5 mm')
            + sd.row(*panels, gap=44) + leg)
    rows = []
    for est, lab in (('p0', 'wft fit p0'), ('p_mid', 'p_mid (p0 + 7.5 mm · tan)'), ('c_pos', 'charge centroid')):
        o = r63[est]
        for v in ('X', 'Y'):
            q = o[v]
            rows.append([lab, v, f'{um(q["sigma_res"])} ± {um(q["err"])}', f'{um(q["res_b2"][0])} ± {um(q["res_b2"][1])} ({q["res_b2"][2]})',
                         f'{um(q["res_b2f2"][0])}' if np.isfinite(q['res_b2f2'][0]) else '–',
                         f'{um(q["sigma_det4"])} ± {um(q["stat"])}', f'{um(q["sys_range"][0])}–{um(q["sys_range"][1])}'])
    tab = sd.table(['estimator', 'view', 'residual, all [µm]', 'residual, 2-strip back (n)', '2-strip back + front', 'det4 (stat) [µm]', 'f range'], rows, 20)
    notes = f'''<p><b>Two ways to read the telescope out of the residual.</b></p>
<ol><li><b>Model-free upper bound.</b> When the back-plane cluster of the coordinate in question has 2 strips, its position is
interpolated and the telescope pointing is much better than for a 1-strip (binary) cluster. The residual of that subsample
is det4 ⊕ (a smaller) telescope, so it bounds det4 from above: X ≤ {um(X_B2[0])} ± {um(X_B2[1])} µm
({X_B2[2]} tracks), and ≤ {um(X_B2F2[0])} ± {um(X_B2F2[1])} µm ({X_B2F2[2]}) when the front cluster is 2-strip too.
Y ≤ {um(Y_B2[0])} ± {um(Y_B2[1])} µm ({Y_B2[2]}). (The front x plane has too few 2-strip clusters for a Y b2f2 bound.)</li>
<li><b>Subtracted.</b> σ<sub>det4</sub> = √(σ²<sub>res</sub> − P²) on all tracks, P = {um(X_Q["pointing"])} µm (zone 1.0,
f = {F_C["f"]:.3f}). The f range gives the systematic (table, last column).</li></ol>
<p>Consistency: the f model predicts the telescope pointing of the 2-strip subsample, P<sub>b2</sub> = √(P² − (σ²<sub>all</sub>
− σ²<sub>b2</sub>)). It comes out positive and similar in both views: X {um(X_Q["P_b2_implied"])} µm, Y
{um(Y_Q["P_b2_implied"])} µm. So the X/Y difference belongs to det4, not to the telescope.</p>
{tab}
<p><b>Estimator choice.</b> Y: all three agree within the f systematic (p0 is quoted, as in <code>sps_audit</code>). X: the
charge centroid and p0 are ~2× worse than p_mid ({um(r63["c_pos"]["X"]["sigma_det4"])} and {um(r63["p0"]["X"]["sigma_det4"])} µm).
The X centroid is worse even for 3-strip clusters, so the narrower X footprint does not explain it. A depth-dependent shift of
the X charge (the X ±1 neighbour asymmetry, <code>paper1_spsfigs</code>) is a candidate, but it is not demonstrated.
p_mid reads the track 7.5 mm below the mesh, where the p0–angle correlation vanishes (chosen on the whole-spot run 63 in
<code>sps_audit</code>, not tuned here).</p>
<p>The alignment is a robust affine refitted on R's own tracks (<code>sps_audit/analyze.seeded_fit</code>). The residual is
flat across R: four Y quarters give X {", ".join(um(q["X"][0]) for q in r63["p_mid"]["uniformity"])} µm and Y
{", ".join(um(q["Y"][0]) for q in r63["p0"]["uniformity"])} µm (raw residuals).</p>'''
    D.slide('resolution', body, notes, short='resolution')


# ---------------------------------------------------------------- slide 6: consistency
def consistency_slide(D):
    items = [('run 63 R', res('run63', 'p_mid')['X'], res('run63', 'p0')['Y'], 'run 63 flat700, CF4, ZS 64, R'),
             ('run 63 R2', res('run63', 'p_mid', 'R2 (cross-check: X band 201-214)')['X'], res('run63', 'p0', 'R2 (cross-check: X band 201-214)')['Y'],
              'run 63, the second live band (lower gain; X in telescope zone 1.5)'),
             ('run 66 R', res('run66', 'p_mid')['X'], res('run66', 'p0')['Y'], 'run 66 r755+r780, CF4, ZS 32, R'),
             ('run 53 R', res('run53', 'c_pos')['X'], res('run53', 'c_pos')['Y'], 'run 53, Ar/CO2/iso 95/3/2, ZS 32, R (centroid: no v, so no p_mid)')]
    P = sd.Plot(1100, 640, x=(-0.5, len(items) - 0.5), y=(0, 0.42), ylabel='σ [mm]', margin=(30, 20, 92, 104))
    P.xticks([(i, it[0]) for i, it in enumerate(items)]).yticks([(v, f'{v:.1f}') for v in (0, 0.1, 0.2, 0.3, 0.4)])
    for i, (lab, X, Y, desc) in enumerate(items):
        for dx, o, col, v in ((-0.12, X, sd.BLUE, 'X'), (0.12, Y, sd.ORANGE, 'Y')):
            lo, hi = o['sys_range']
            P.raw(sd.line(P.X(i + dx), P.Y(lo), P.X(i + dx), P.Y(hi), col, 3))
            P.points([i + dx], [o['sigma_det4']], col, 9, tips=[f'{desc}\n{v}: det4 {um(o["sigma_det4"])} µm (f range {um(lo)}–{um(hi)}); '
                                                              f'residual {um(o["sigma_res"])}, 2-strip back {um(o["res_b2"][0])} µm'])
            if np.isfinite(o['res_b2'][0]):
                P.points([i + dx], [o['res_b2'][0]], col, 9, marker='open', tips=[f'{desc}\n{v}: upper bound (2-strip back cluster) {um(o["res_b2"][0])} µm'])
    P.text(1.5, 0.39, 'R in run 63 is the best place on det4', 24, sd.INK, 'middle', 700)
    leg = L([(sd.BLUE, 'X (p_mid; run 53: centroid)'), (sd.ORANGE, 'Y (p0; run 53: centroid)')], size=22)
    leg2 = sd.p('● det4 after subtracting the telescope (bar: f range)   ○ upper bound, 2-strip telescope clusters', 22, sd.MUT)
    body = (sd.title('Same recipe elsewhere: the lower-gain band and the other runs are worse, as expected',
                     'det4 intrinsic σ per view; the same telescope f everywhere')
            + sd.row(sd.col(P.svg('consistency'), leg, leg2, gap=6), sd.col(
                sd.callout('R2 (lower gain): X and Y both worse, so the patch choice matters.', sd.RED, 24),
                sd.callout('run 66 at the same field: X a little worse with half the samples (ZS 32).', sd.BLUE, 24),
                sd.callout('run 53 (Ar/CO2, no drift velocity): only the centroid is usable; Y agrees with run 63.', sd.GREY, 24), gap=20, w=480), gap=40))
    notes = '''<p>R2 is X 202–213 mm, the same Y range, in the second live band: lead-strip charge ~210 ADC against ~330, and its X
coordinate sits in the telescope's 1.5 mm zone (pointing ~0.33 mm, so its X subtraction is larger and less certain). Its
det4 values are higher in both views. The resolution follows the local gain, so the whole-spot numbers are averages over
regions of different quality.</p>
<p>run 66: the two highest-gain resist plateaus (755, 780 V), 32 samples (the drift column is cut), 243 V/cm. run 53: Ar/CO2/iso
at a lower gain (lead charge ~110 ADC), no drift velocity in the registry, so the wft fit and p_mid are not meaningful
there; the charge centroid is shown. Its X centroid is bad, as it is in run 63; its Y centroid agrees with run 63's.</p>'''
    D.slide('consistency', body, notes, short='cross-checks')


# ---------------------------------------------------------------- slide 7: efficiency
def efficiency_slide(D):
    cells = EFF['eff_cells']
    P = sd.Plot(520, 700, x=(RX[0], RX[1]), y=(RY[0], RY[1]), xlabel='det4 X [mm]', ylabel='det4 Y [mm]', margin=(20, 20, 92, 100), title='R in 5 × 6 mm cells')
    P.xticks([(150, '150'), (155, '155'), (160, '160')]).yticks([(v, str(v)) for v in (222, 228, 234, 240, 246)])
    P.cells([(q['x'][0], q['x'][1], q['y'][0], q['y'][1], q['eff'], f'{100 * q["eff"]:.1f} % of {q["n"]} tracks') for q in cells], 0.8, 1.0, sd.VIRIDIS, gap=1)
    for q in cells:
        P.text((q['x'][0] + q['x'][1]) / 2, (q['y'][0] + q['y'][1]) / 2 - 0.5, f'{100 * q["eff"]:.0f} %', 24, '#ffffff', 'middle', 700)
    cb = sd.colorbar(130, 560, 0.8, 1.0, [(0.8, '80 %'), (0.9, '90 %'), (1.0, '100 %')], 'efficiency')
    labels = [('both views, ±2 mm', E_BOTH[0], sd.GREEN), ('X view, ±2 mm', EFF['eff_region_x'][0], sd.BLUE), ('Y view, ±2 mm', EFF['eff_region_y'][0], sd.ORANGE),
              ('both views, ±1 mm', EFF['eff_region_both_win1'], sd.GREY), ('both views, ±3 mm', EFF['eff_region_both_win3'], sd.GREY),
              ('any signal ≥ 50 ADC, both views', EFF['eff_region_any_signal'], sd.GREY)]
    B = sd.Plot(760, 560, x=(0.7, 1.0), y=(-0.5, len(labels) - 0.5), xlabel='fraction of telescope tracks in R', margin=(20, 30, 92, 380))
    B.xticks([(0.7, '70 %'), (0.8, '80 %'), (0.9, '90 %'), (1.0, '100 %')])
    for i, (lab, v, col) in enumerate(labels):
        y = len(labels) - 1 - i
        B.raw(f'<rect x="{B.X(0.7):.1f}" y="{B.Y(y + 0.32):.1f}" width="{B.X(v) - B.X(0.7):.1f}" height="{B.Y(y - 0.32) - B.Y(y + 0.32):.1f}" fill="{col}"'
              f'{sd.tipattr(f"{lab}: {100 * v:.1f} %")}/>')
        B.raw(sd.T(B.X(0.7) - 12, B.Y(y) + 8, lab, 22, sd.INK, 'end'))
        B.raw(sd.T(B.X(v) + 8, B.Y(y) + 8, f'{100 * v:.1f} %', 22, sd.INK, 'start', 600))
    body = (sd.title(f'det4 sees {100 * E_BOTH[0]:.1f} ± {100 * E_BOTH[1]:.1f} % of the telescope tracks in R, in both views at once',
                     f'run 63; {E_BOTH[2]:,} single-cluster telescope tracks predicted in R; det4 local maximum ≥ 50 ADC near the prediction')
            + sd.row(P.svg('efficiency cells'), cb, B.svg('efficiency definitions'), gap=24, align='center'))
    notes = f'''<p>Denominator: every telescope track with exactly one cluster in each of the four uRWELL plane-views, inside the
run-63 flat700 plateau (±20 s from its edges), predicted in R by the run-63 telescope→det4 affine. Numerator: det4 has a local
maximum of the per-strip peak amplitude ≥ 50 ADC within ±2 mm of the prediction, in X and in Y (oscillating and coupled channels
excluded, the final SPS rules). New table: <code>eff_job.py</code> → <code>x17rerun/sps_region/eff/</code>
({EFF["eff_tracks"]:,} tracks).</p>
<p>It is a <b>lower bound</b> on det4's efficiency. The single-cluster telescope track is another particle, or out of time, in
part of the events: in <code>sps_audit</code> 7–13 % of single-cluster tracks have front and back planes more than 3 mm apart.
det4 has some signal in both views in {100 * EFF["eff_region_any_signal"]:.1f} % of these events, and the efficiency grows with the
matching window ({100 * EFF["eff_region_both_win1"]:.0f} / {100 * E_BOTH[0]:.0f} / {100 * EFF["eff_region_both_win3"]:.0f} % at
±1 / 2 / 3 mm), as expected if the missing part is telescope tails and pile-up rather than det4 inefficiency.</p>
<p>Uniformity: the eight cells range {100 * min(q["eff"] for q in EFF["eff_cells"]):.0f}–{100 * max(q["eff"] for q in EFF["eff_cells"]):.0f} %.
The X half 155–160 mm is ~3 points lower. The gain also falls there towards the stripe edge at 162.5 mm (slide 3).</p>'''
    D.slide('efficiency', body, notes, short='efficiency')


# ---------------------------------------------------------------- slide 8: footprint
def footprint_slide(D):
    fz = R['footprint']['run63_zs']; fr = R['footprint']['run71_raw700']
    P = sd.Plot(800, 560, x=(0.5, 6.5), y=(0, 0.85), xlabel='strips in the cluster (above 50 ADC)', ylabel='fraction of events',
                margin=(20, 20, 92, 100), title='run 63 (ZS), tracks in R')
    P.xticks([(k, str(k)) for k in range(1, 7)]).yticks([(v, f'{v:.1f}') for v in (0, 0.2, 0.4, 0.6, 0.8)])
    for v, col, dx in (('x', sd.BLUE, -0.18), ('y', sd.ORANGE, 0.18)):
        h = np.array(fz['region'][v]['strips_hist'], float); h /= h.sum()
        hw = np.array(fz['whole spot'][v]['strips_hist'], float); hw /= hw.sum()
        for k in range(1, 7):
            P.vbar(k + dx, h[k], 30, col, tip=f'{v.upper()} in R: {100 * h[k]:.0f} % of events have {k} strips')
        P.points([k + dx for k in range(1, 7)], [hw[k] for k in range(1, 7)], col, 6, marker='open',
                 tips=[f'{v.upper()} whole spot: {100 * hw[k]:.0f} % with {k} strips' for k in range(1, 7)])
    rows = []
    for lab, f, n in (('run 63, ZS 64 (R)', fz['region'], fz['region']['n']), ('run 63, ZS 64 (whole spot)', fz['whole spot'], fz['whole spot']['n']),
                      ('run 71, RAW, no ZS (R)', fr['region'], fr['region']['n']), ('run 71, RAW (whole spot)', fr['whole spot'], fr['whole spot']['n'])):
        rows.append([lab, f'{n:,}', f'{f["x"]["strips_mean"]:.2f} / {f["y"]["strips_mean"]:.2f}', f'{f["x"]["lead_frac"]:.2f} / {f["y"]["lead_frac"]:.2f}',
                     f'{f["x"]["nb"]:.2f} / {f["y"]["nb"]:.2f}', f'{f["x"]["q_sum"]:.0f} / {f["y"]["q_sum"]:.0f}'])
    tab = sd.table(['sample', 'n', 'strips X / Y', 'lead fraction', 'neighbour ÷ lead', 'cluster charge [ADC]'], rows, 21)
    leg = L([(sd.BLUE, 'X'), (sd.ORANGE, 'Y')], size=22)
    body = (sd.title(f'Footprint in R: {fr["region"]["x"]["strips_mean"]:.1f} (X) and {fr["region"]["y"]["strips_mean"]:.1f} (Y) strips without zero '
                     f'suppression; each neighbour carries {fr["region"]["x"]["nb"]:.2f} / {fr["region"]["y"]["nb"]:.2f} of the lead',
                     'cluster = strips ≥ 50 ADC within ±4 samples of the lead strip\'s peak; medians; 0.78 mm pitch')
            + sd.row(sd.col(P.svg('strip multiplicity'), sd.p('bars: R · open circles: whole beam spot', 21, sd.MUT), leg, gap=4),
                     sd.col(tab, sd.p(f'In R the charge is {fz["region"]["x"]["q_sum"] / fz["whole spot"]["x"]["q_sum"]:.1f}× the spot median, so '
                                      'fewer neighbours fall under the 50 ADC threshold and the cluster is one strip wider.', 24), gap=24), gap=40))
    notes = f'''<p>Quantities per event (<code>sps_cluster</code> via the pass tables): number of strips ≥ 50 ADC within ±4 samples
of the lead strip's peak (censored strips bridged); lead fraction = lead peak ÷ sum of the cluster's peaks; neighbour ratio =
each ±1 neighbour's peak ÷ the lead's (median over both sides). run 63: telescope-matched tracks in R (both views within ±1.5 mm),
{fz["region"]["n"]:,} events. run 71 RAW (raw700, 243 V/cm, no zero suppression): det4's own cluster position selects R, since the
footprint table has no telescope ({fr["region"]["n"]} of {fr["whole spot"]["n"]:,} events).</p>
<p>Without zero suppression (run 71) the footprint is threshold-limited only by the 50 ADC cut, so it is the closest to the
physical one. On ZS rows a suppressed neighbour reads 0, which raises the lead fraction and changes the neighbour ratio; on
run 63 the X neighbour ratio is {fz["region"]["x"]["nb"]:.2f} against Y {fz["region"]["y"]["nb"]:.2f}, mostly the ZS threshold
(<code>sps_audit</code> §5). The kernel itself (the ±1/±2 sharing in the reconstruction) is the pass calibration's.</p>
<p>In R the RAW sample is small ({fr["region"]["n"]} events), so its means carry ~±0.1 strips.</p>'''
    D.slide('footprint', body, notes, short='footprint')


# ---------------------------------------------------------------- slide 9: timing
def timing_slide(D):
    T = R['timing']
    raw = T['run71_raw__raw700']; zs = T['run63_flat__flat700']
    P = sd.Plot(1060, 600, x=(150, 6000, 'log'), y=(0, 300), xlabel='smaller of the two views\' cluster charge [ADC]',
                ylabel='per-view time resolution (X − Y)/√2 [ns]', margin=(30, 20, 92, 110))
    P.xticks([(200, '200'), (500, '500'), (1000, '1000'), (2000, '2000'), (5000, '5000')]).yticks([(v, str(v)) for v in (0, 50, 100, 150, 200, 250, 300)])
    P.band([150, 6000], [22, 22], [28, 28], sd.GREEN, 0.25, tip='bench per-view part, det2/det3/det7 golden rows: 22–28 ns (paper1_timing)')
    P.text(5800, 34, 'bench per-view 22–28 ns', 21, sd.GREEN, 'end')
    for data, est, col, dash, lab in ((raw, 'edge25', sd.BLUE, None, 'RAW run 71, edge'), (raw, 'cf20', sd.BLUE, '6 4', 'RAW run 71, CF 20 %'),
                                      (zs, 'edge25', sd.GREY, None, 'ZS run 63, edge')):
        for sample, mk in (('whole spot', 'circle'), ('region', 'diamond')):
            c = [q for q in data[est]['vs_charge'] if q['sample'] == sample]
            if not c:
                continue
            P.line([q['q_med'] for q in c], [q['sigma_xy'] for q in c], col if sample == 'whole spot' else sd.RED, 2.5 if sample == 'whole spot' else 0,
                   dash=dash, r=7, marker=mk, tips=[f'{lab}, {sample}: charge {q["q_lo"]:.0f}–{q["q_hi"]:.0f} ADC (median {q["q_med"]:.0f}), '
                                                    f'n {q["n"]}: {q["sigma_xy"]:.0f} ns' for q in c])
    e = raw['edge25']['region']; cfr = raw['cf20']['region']
    P.text(170, 285, f'in R (RAW): {e["sigma_xy"]:.0f} ± {e["sigma_xy_err"]:.0f} ns (edge), {cfr["sigma_xy"]:.0f} ± {cfr["sigma_xy_err"]:.0f} ns (CF 20 %)', 24, sd.RED, weight=700)
    leg = L([(sd.BLUE, 'RAW run 71 raw700: edge (solid), constant fraction 20 % (dashed)'), (sd.GREY, 'ZS run 63 flat700: edge'),
                     (sd.RED, '◆ tracks in R (same estimators)')], size=21)
    side = sd.col(sd.callout(f'Set by the signal size: at equal charge R and the whole spot agree. Above ~2500 ADC det4 times like the bench.', sd.BLUE, 24),
                  sd.callout(f'The trigger is not the limit: the uRWELL\'s own hit time has s68 = {TR["s68_mean"]:.0f} ns against it.', sd.GREEN, 24),
                  sd.callout('Event time (X+Y against the trigger) is not measured: the RAW sample has no telescope match.', sd.GREY, 24), gap=20, w=520)
    body = (sd.title(f'Timing in R: per-view {e["sigma_xy"]:.0f} ns (edge) / {cfr["sigma_xy"]:.0f} ns (constant fraction) without zero suppression',
                     'det4 X and Y strip sums, column-start estimators of Paper I §7; spread of t_X − t_Y divided by √2 (s68)')
            + sd.row(sd.col(P.svg('timing vs charge'), leg, gap=6), side, gap=36))
    rows = []
    for key, lab in (('run71_raw__raw700', 'run 71 raw700 (RAW)'), ('run71_raw__raw450', 'run 71 raw450 (RAW)'),
                     ('run63_flat__flat700', 'run 63 flat700 (ZS 64)'), ('run66_flat_resist__r780', 'run 66 r780 (ZS 32)')):
        t = T[key]
        rows.append([lab, f'{t["n_region"]} / {t["n_all"]}', f'{t["edge25"]["region"]["sigma_xy"]:.0f} ± {t["edge25"]["region"]["sigma_xy_err"]:.0f}',
                     f'{t["edge25"]["whole spot"]["sigma_xy"]:.0f}', f'{t["cf20"]["region"]["sigma_xy"]:.0f} ± {t["cf20"]["region"]["sigma_xy_err"]:.0f}',
                     f'{t["cf20"]["whole spot"]["sigma_xy"]:.0f}', f'{t["edge25"]["region"]["abs_raw"]:.0f}'])
    tab = sd.table(['row', 'n (R / all)', 'edge, R', 'edge, spot', 'CF20, R', 'CF20, spot', '(X+Y)/2 vs trigger, R'], rows, 20)
    notes = f'''<p><b>What is measured.</b> det4's X and Y strips read the same avalanches, so t<sub>X</sub> − t<sub>Y</sub> cancels the
particle time, the primary-cluster positions and the trigger. Its spread /√2 is the per-view part of the time resolution: the
readout and the estimator. On the bench the same quantity is 22–28 ns on the golden rows, and the event time there is 40–46 ns
because a common part of 35–40 ns (primary clusters near the mesh) adds to it (<code>rerun/paper1_timing</code>).</p>
<p>Estimators (<code>paper1_timing/estimators.py</code>): <b>edge</b> = a step convolved with the g_det4 electronics template,
fitted to the rising edge up to 25 % of the strip sum's maximum; <b>CF 20 %</b> = where the strip sum first crosses 20 % of its
maximum, linearly interpolated. Strip sums: <code>paper1_timing/sps_edge.py</code> (6000 events per row, ~17 s of each plateau;
both views clustered, charge ≥ 150 ADC). R is selected on det4's own cluster positions; there is no telescope in these tables.</p>
{tab}
<p><b>Trigger.</b> In the uRWELL telescope's hits (banco's combined_hits, run 63, nine files, interpolated hit times), the
largest hit per event has s68 = {TR["s68_mean"]:.1f} ns ({TR["s68_range"][0]:.0f}–{TR["s68_range"][1]:.0f} ns per file) against the
trigger. A trigger uniform in the 60 ns clock period alone would give ~20 ns, so the H4 trigger plus the DREAM time base
contribute at most ~19 ns (<code>trigger_ref.py</code>). The (X+Y)/2 spread against the trigger in R (last column, 140–250 ns) is
much larger than either the trigger or the per-view part. Most likely it is untagged pile-up in the 3.84 µs RAW window
(only 25 % of run-71 telescope tracks pass the cross-view fiducial in <code>sps_audit</code>), which X − Y does not see. It is
not quoted as det4's event time.</p>'''
    D.slide('timing', body, notes, short='timing')


# ---------------------------------------------------------------- slide 10: summary
def summary_slide(D):
    o = r63
    rows = [
        ['position resolution X', f'{um(X_Q["sigma_det4"])} µm (f range {um(X_Q["sys_range"][0])}–{um(X_Q["sys_range"][1])}), stat ±{um(X_Q["stat"])}',
         f'≤ {um(X_B2F2[0])} ± {um(X_B2F2[1])} µm model-free', 'p_mid, run 63, R'],
        ['position resolution Y', f'{um(Y_Q["sigma_det4"])} µm (f range {um(Y_Q["sys_range"][0])}–{um(Y_Q["sys_range"][1])}), stat ±{um(Y_Q["stat"])}',
         f'≤ {um(Y_B2[0])} ± {um(Y_B2[1])} µm model-free', 'fit p0 (centroid {0}, p_mid {1})'.format(um(o['c_pos']['Y']['sigma_det4']), um(o['p_mid']['Y']['sigma_det4']))],
        ['in strip pitches', f'X {X_Q["sigma_det4"] / 0.78:.2f}, Y {Y_Q["sigma_det4"] / 0.78:.2f} × 0.78 mm', '', ''],
        ['efficiency (both views)', f'{100 * E_BOTH[0]:.1f} ± {100 * E_BOTH[1]:.1f} %', f'X {100 * EFF["eff_region_x"][0]:.1f} %, Y {100 * EFF["eff_region_y"][0]:.1f} %',
         'lower bound (telescope purity)'],
        ['footprint (RAW, no ZS)', f'{R["footprint"]["run71_raw700"]["region"]["x"]["strips_mean"]:.1f} / {R["footprint"]["run71_raw700"]["region"]["y"]["strips_mean"]:.1f} strips (X / Y)',
         f'neighbour ÷ lead {R["footprint"]["run71_raw700"]["region"]["x"]["nb"]:.2f} / {R["footprint"]["run71_raw700"]["region"]["y"]["nb"]:.2f}', 'run 71 raw700'],
        ['per-view timing (RAW)', f'{R["timing"]["run71_raw__raw700"]["edge25"]["region"]["sigma_xy"]:.0f} ± {R["timing"]["run71_raw__raw700"]["edge25"]["region"]["sigma_xy_err"]:.0f} ns (edge)',
         f'{R["timing"]["run71_raw__raw700"]["cf20"]["region"]["sigma_xy"]:.0f} ± {R["timing"]["run71_raw__raw700"]["cf20"]["region"]["sigma_xy_err"]:.0f} ns (CF 20 %)', 'charge-limited'],
        ['angle at normal incidence', f'θ s68 X {o["p_mid"]["tan_x"]["theta_s68_deg"]:.1f}°, Y {o["p_mid"]["tan_y"]["theta_s68_deg"]:.1f}°', 'head-on (mount 0°): spread of the fit angle', 'wft fit, run-63 v'],
    ]
    tab = sd.table(['quantity (R, run 63 unless noted)', 'value', 'bound / split', 'note'], rows, 22)
    lim = sd.col(
        sd.p('<b>What this does not rule out</b>', 28, sd.DINK),
        sd.p('• The subtracted resolutions depend on the telescope f (0.25–0.28 measured in-band); the upper bounds do not.', 23, sd.DINK),
        sd.p('• X is worse than Y with every estimator; why is open (X ±1 asymmetry is a candidate, not shown).', 23, sd.DINK),
        sd.p('• Event time (the common part) is not measured at SPS; only the per-view part is.', 23, sd.DINK),
        sd.p('• One patch, one gas (Ar/CF4/iso 88/10/2): not n_TOF\'s gas or gain, and not the whole chamber.', 23, sd.DINK), gap=10)
    body = (sd.kicker('summary') + sd.title('det4 in its best 10 × 24 mm²', '', ) + sd.row(sd.card(f'<div style="color:{sd.INK}">{tab}</div>', w=1040), lim, gap=40))
    notes = '''<p>Decision for the papers: quote these numbers as "det4 in a fully working region at SPS", with the upper bounds,
or keep the whole-spot numbers (Y 190 ± 10 µm, X model-dependent). The region numbers are cleaner: one telescope zone, one gain, no
dead strips. They are also smaller, because the whole spot averages in lower-gain bands. Say which in the text.</p>
<p>Files: <code>rerun/sps_region/README.md</code> (method and numbers), <code>results/region.json</code>,
<code>results/trigger_ref.json</code>.</p>'''
    D.slide('summary', body, notes, dark=True, short='summary')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(HERE, 'deck', 'mx17-sps-region.html'))
    a = ap.parse_args()
    D = sd.Deck('det4 at SPS: one working patch', 'det4 resolution, efficiency, footprint and timing in one 10 × 24 mm² region at H4')
    cover(D); map_slide(D); profile_slide(D); telescope_slide(D); resolution_slide(D); consistency_slide(D)
    efficiency_slide(D); footprint_slide(D); timing_slide(D); summary_slide(D)
    D.write(a.out, note_meta=dict(title='det4 at SPS: one working patch',
                                  summary=f'det4 measured in a 10 × 24 mm² region at H4 (run 63): sigma_Y {um(Y_Q["sigma_det4"])} um, sigma_X '
                                          f'{um(X_Q["sigma_det4"])} um, {100 * E_BOTH[0]:.0f} % efficient, footprint and per-view timing',
                                  tags='x17, mx17, sps, resolution', date='2026-11-01'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
