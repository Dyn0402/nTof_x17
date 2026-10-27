#!/usr/bin/env python3
"""Slide note: every per-chamber map we have for the five June bench chambers (FOLLOWUPS item 8), each marked with
whether the reconstruction uses it today.  One slide per kind of map, the five chambers side by side.

Inputs: results/xcheck_<det>.json and gap_<det>.json (this directory), the golden per-event tables (common.load), the
promoted bundles (dead lists, w0/kw/v), ws_dead_charge/xy_gain_<det>.json (connector gains, F15) and f12_table.json
(chamber-side dead strips, F12), ws_v_w0/v_uncertainty.json (v in time quintiles), the noise-band census
(~/x17/cosmic_bench/satstudy/bands/patches.json) and the pre-rerun efficiency maps
(~/x17/cosmic_bench/Analysis/<row>/wft/efficiency/efficiency_map_sliding.png).  FA's v(x, y) / v(t) are shown when
followups/fa_gas_v_tilt/results/ has them, else marked pending.

    .venv/bin/python rerun/model_audit/followups/fb_bow_maps/make_maps_deck.py
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
        rerun/model_audit/followups/fb_bow_maps/deck/mx17-chamber-maps.html --slug mx17-chamber-maps --force --deploy
"""
import glob
import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deckkit import sd, J, hmap, plain_ticks, DCOL, XC, YC  # noqa: E402
from deckkit import BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN, INK, MUT  # noqa: E402
from fbcommon import DETS, ROOT, load, view_mask, bundle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
OUT = os.path.join(HERE, 'deck', 'mx17-chamber-maps.html')
WSD = os.path.join(REPO, 'rerun', 'model_audit', 'ws_dead_charge')
ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4',
        'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
BAND_KEYS = {'det2': 'g_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}
E5 = np.linspace(25, 375, 6)
FA = glob.glob(os.path.expanduser('~/PycharmProjects/nTof_x17/.claude/worktrees/*/rerun/model_audit/followups/fa_gas_v_tilt/results')) + \
    [os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fa_gas_v_tilt', 'results')]


def badge(used, text):
    c = {'yes': GREEN, 'partly': ORANGE, 'no': RED}[used]
    lab = {'yes': 'USED in reconstruction', 'partly': 'PARTLY used', 'no': 'NOT used in reconstruction'}[used]
    return (f'<div style="display:flex;gap:18px;align-items:center"><span style="background:{c};color:#fff;'
            f'font-size:24px;font-weight:600;padding:6px 16px;border-radius:8px">{lab}</span>'
            f'<span style="font-size:24px;color:{MUT}">{text}</span></div>')


def head(t, sub, used, utext):
    return sd.title(t, sub) + badge(used, utext)


def five(maps, cb):
    return sd.row(*[sd.col(m, gap=4, w=300) for m in maps], cb, gap=14)


def cellmap(fn, det, p):
    """generic 5x5 cell statistic over the golden table: fn(T, mask) -> value."""
    T = load(det)
    m = view_mask(T, p)
    u, w = T.u.to_numpy(), T.w.to_numpy()
    M = np.full((5, 5), np.nan); N = np.zeros((5, 5), int)
    for i in range(5):
        for j in range(5):
            k = m & (u >= E5[i]) & (u < E5[i + 1]) & (w >= E5[j]) & (w < E5[j + 1])
            N[i, j] = k.sum()
            if k.sum() >= 40:
                M[i, j] = fn(T, k)
    return M, N


def lab(d, first, bottom=True):
    return dict(xlabel='x [mm]' if bottom else '', ylabel='y [mm]' if first else '',
                margin=(16, 10, 50 if bottom else 24, 70 if first else 20))


# --------------------------------------------------------------------------- #
def s_cover(D):
    body = (sd.kicker('MX17 · June cosmic bench · model-audit follow-up FB, item 8 · 27 Oct 2026') +
            '<h1 style="font-size:70px;font-weight:600;line-height:1.08;color:#eef0f3">Every map we have of the five chambers, '
            'and which ones the reconstruction actually uses: four of eleven</h1>' +
            sd.row(sd.bignum('11', 'kinds of map', sd.DBLUE, 'geometry, drift, gain, efficiency, faults'),
                   sd.bignum('4', 'used today', sd.DGREEN, 'dead strips, and one number per view of tilt, scale, v'),
                   sd.bignum('5', 'chambers', sd.DGREY, 'golden row of each: det2, det3, det4, det6, det7'), gap=56))
    rows = ''.join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>' for a, b, c in (
        ('mesh height (bow)', 'no', 'FB: static per chamber, proposed bundle slot'),
        ('drift tilt w0(x, y)', 'one constant per view', 'angle_map code exists, in no production bundle'),
        ('angle scale / ladder slope', 'one kw per view', 'binned in angle_map, unused'),
        ('drift velocity v(x, y), v(t)', 'one v per view and run', 'FA'),
        ('column duration', 'no', 'diagnostic'),
        ('gain from charge', 'no (gainmap.npz all 1.0)', ''),
        ('X/Y connector gains', 'no', 'equal to ±2 %: nothing to correct (F15)'),
        ('efficiency', 'no', 'QA'),
        ('dead strips', 'yes', 'censored in the fit, bridged by the seeder'),
        ('noise bands', 'per event (band veto), not as a map', 'FE item 13'),
        ('t0 latency', 'free t0 in practice', 'FE item 7')))
    notes = f'<table><tr><th>map</th><th>used in reconstruction</th><th>owner / note</th></tr>{rows}</table>'
    D.slide('cover', body, notes, dark=True, short='cover')


def s_bow(D):
    rows = []
    for p in 'xy':
        maps = []
        for i, d in enumerate(DETS):
            X = J(f'xcheck_{d}.json')
            H = np.array(X[f'cells_{p}']['h'], float)
            H = H - np.nanmedian(H[1:4, 1:4])
            maps.append(hmap(H, f'{d} · {p.upper()}', -3, 3, sd.DIVERGE, w=300, h=270, unit='mm',
                             errs=X[f'cells_{p}']['h_err'], ns=X[f'cells_{p}']['n'], **lab(d, i == 0, p == 'y')))
        rows.append(sd.row(*[sd.col(m, w=300) for m in maps], gap=14))
    cb = sd.colorbar(140, 540, -3, 3, [(-3, '−3'), (0, '0'), (3, '+3')], label='h [mm]', cmap=sd.DIVERGE)
    body = (head('Mesh height: every chamber is dished, centre lower by 1.5–2.3 mm',
                 'Slope of the position residual against the M3 track slope, 70 mm cells, X view (top) and Y view; blue = lower',
                 'no', 'proposed: a static per-chamber surface in the bundle (spec in the bow note)') +
            sd.row(sd.col(*rows, gap=2), cb, gap=14))
    notes = '<p>Full analysis: https://dylan-neff.web.cern.ch/notes/mx17-chamber-bow.html</p>'
    D.slide('bow', body, notes, short='bow')


def s_w0(D):
    rows = []
    for p in 'xy':
        maps = []
        for i, d in enumerate(DETS):
            X = J(f'xcheck_{d}.json')
            M = np.array(X[f'cells_{p}']['w0'], float)
            maps.append(hmap(M, f'{d} · {p.upper()}', -1.2, 1.2, sd.DIVERGE, w=300, h=270, unit='µm/ns',
                             errs=X[f'cells_{p}']['w0_err'], ns=X[f'cells_{p}']['n'], **lab(d, i == 0, p == 'y')))
        rows.append(sd.row(*[sd.col(m, w=300) for m in maps], gap=14))
    cb = sd.colorbar(140, 540, -1.2, 1.2, [(-1.2, '−1.2'), (0, '0'), (1.2, '+1.2')], label='w0 [µm/ns]', cmap=sd.DIVERGE)
    body = (head('Drift tilt w0: a ramp across every chamber (X along x, Y along y) on top of one constant',
                 'Offset of the fitted transverse speed for a track M3 calls vertical, 70 mm cells', 'partly',
                 'one constant per view and run (bundle w0); the map is in no bundle') +
            sd.row(sd.col(*rows, gap=2), cb, gap=14))
    D.slide('w0', body, '<p>Cells: Huber line w = w0 + b·tan_ref. The constant part is the rigid tilt of the chamber against '
            'M3 (ws_v_w0, F9); the ramp is the bow (bow note, slide 6).</p>', short='tilt w0')


def s_scale(D):
    rows = []
    for p in 'xy':
        maps = []
        for i, d in enumerate(DETS):
            X = J(f'xcheck_{d}.json')
            M = np.array(X[f'cells_{p}']['b'], float)
            M = M / np.nanmedian(M)
            maps.append(hmap(M, f'{d} · {p.upper()}', 0.85, 1.15, sd.DIVERGE, w=300, h=270, unit='× chamber median',
                             fmt='{:.3f}', ns=X[f'cells_{p}']['n'], **lab(d, i == 0, p == 'y')))
        rows.append(sd.row(*[sd.col(m, w=300) for m in maps], gap=14))
    cb = sd.colorbar(140, 540, 0.85, 1.15, [(0.85, '0.85'), (1, '1'), (1.15, '1.15')], label='b / median', cmap=sd.DIVERGE)
    body = (head('Angle scale (ladder slope = local v × scale): flat to ±3 % on det2/det3, ±10–15 % on det4/det6',
                 'Slope of the fitted w against M3 tan per 70 mm cell, divided by the chamber median', 'partly',
                 'one kw and one v per view; v(x, y) is FA\'s, the bow does not change it') +
            sd.row(sd.col(*rows, gap=2), cb, gap=14))
    notes = ('<p>The ladder slope cannot tell a local drift velocity from a local angle scale; FA separates them '
             '(v(x, y) from the column duration and the ladder). det6 is ~10 % faster in its centre in both views, '
             'det4 Y ramps 0.82 → 1.04 along y.</p>')
    D.slide('scale', body, notes, short='angle scale')


def fa_files():
    out = {}
    for d in FA:
        for f in glob.glob(os.path.join(d, '*.json')):
            out[os.path.basename(f)] = f
    return out


def s_vt(D):
    U = json.load(open(os.path.join(ROOT, 'ws_v_w0', 'v_uncertainty.json')))
    P = sd.Plot(1100, 560, x=(0.5, 5.5), y=(0.84, 1.16), xlabel='fifth of the run (time order)',
                ylabel='ladder v / run mean', margin=(24, 20, 92, 104))
    P.xticks([(k, str(k)) for k in range(1, 6)]).yticks(plain_ticks(0.85, 1.15, 0.05, '{:.2f}'))
    P.hline(1, GREY, dash='4 4')
    for d in DETS:
        for p, dash in (('x', None), ('y', '8 6')):
            sp = U[d][p]['splits']['time']; v = np.array(sp['values']); e = np.array(sp['errors'])
            m = np.mean(v)
            P.line(list(range(1, 6)), list(v / m), DCOL[d], w=3, dash=dash,
                   tips=[f'{d} {p.upper()} quintile {k + 1}: {a:.2f} ± {b:.2f} µm/ns (n = {n})'
                         for k, (a, b, n) in enumerate(zip(v, e, sp['n']))])
    fa = fa_files()
    have = [k for k in fa if 'vt' in k.lower() or 'v_time' in k.lower() or 'vxy' in k.lower()]
    sub = ('FA results found: ' + ', '.join(have)) if have else 'FA\'s v(t) and v(x, y) calibration is PENDING; shown: ws_v_w0 time quintiles'
    leg = sd.row(sd.legend([(d, DCOL[d]) for d in DETS]), sd.p('solid X · dashed Y', 24, MUT), gap=40)
    body = (head('Drift velocity in time: det4 falls ~10 % over its day run, det6 rises ~12 % over the night',
                 sub, 'no', 'one v per view and run; v(t) and v(x, y) are FA\'s bundle slots (item 1)') + P.svg('vt') + leg)
    D.slide('vt', body, '<p>Ladder v per time quintile, from <code>ws_v_w0/v_uncertainty.json</code>. The column-duration '
            'route moves the same way (ws_v_w0): a real gas change after a short flush.</p>', short='v(t)')


def s_column(D):
    rows = []
    for p in 'xy':
        maps = []
        for i, d in enumerate(DETS):
            G = J(f'gap_{d}.json')
            M = np.array(G[p]['uend_q']['T'], float); M = M / np.nanmedian(M)
            maps.append(hmap(M, f'{d} · {p.upper()}', 0.9, 1.1, sd.VIRIDIS, w=300, h=270, unit='× median', fmt='{:.3f}',
                             **lab(d, i == 0, p == 'y')))
        rows.append(sd.row(*[sd.col(m, w=300) for m in maps], gap=14))
    cb = sd.colorbar(140, 540, 0.9, 1.1, [(0.9, '0.90'), (1, '1'), (1.1, '1.10')], label='T / median')
    body = (head('Column duration: longer in the middle on most chambers, but X and Y of one column disagree',
                 'Fitted column end after t0 (q_uend), charge-matched contained tracks, 70 mm cells', 'no',
                 'diagnostic only; carries gap / v, amplitude and Y-tail effects') + sd.row(sd.col(*rows, gap=2), cb, gap=14))
    D.slide('column', body, '<p>See the bow note, slide 10: the X–Y disagreement is larger than the 1–2 mm a dish would '
            'add, so this map is not used as evidence or as a correction.</p>', short='column')


def s_gain(D):
    rows = []
    for p in 'xy':
        maps = []
        for i, d in enumerate(DETS):
            def f(T, k, p=p):
                q = T[f'{p}_q_sum'].to_numpy()[k] / np.sqrt(1 + T['x_t'].to_numpy()[k] ** 2 + T['y_t'].to_numpy()[k] ** 2)
                return float(np.nanmedian(q))
            M, N = cellmap(f, d, p)
            maps.append(hmap(M / np.nanmedian(M), f'{d} · {p.upper()}', 0.5, 1.5, sd.VIRIDIS, w=300, h=270, unit='× median',
                             fmt='{:.2f}', ns=N, **lab(d, i == 0, p == 'y')))
        rows.append(sd.row(*[sd.col(m, w=300) for m in maps], gap=14))
    cb = sd.colorbar(140, 540, 0.5, 1.5, [(0.5, '0.5'), (1, '1'), (1.5, '1.5')], label='charge / median')
    body = (head('Gain from charge: ±20–40 % across a chamber, the same in both views, det4 striped',
                 'Median fitted track charge per unit path length per 70 mm cell, X view (top) and Y view, saturated tracks included',
                 'no', 'gainmap.npz is loaded but all 1.0 in every promoted bundle') + sd.row(sd.col(*rows, gap=2), cb, gap=14))
    notes = ('<p>Median of q_sum·cosθ over M3-matched fitted tracks. Saturation (16–69 % of tracks, F15) caps the bright '
             'end, so this compresses real gain differences. It is a proxy, not a calibration. X and Y read one avalanche, '
             'so the two rows should agree (F15: equal to ±2 %).</p>')
    D.slide('gain', body, notes, short='gain')


def s_conn(D):
    P = sd.Plot(1560, 520, x=(-0.5, 7.5), y=(-0.08, 0.08), xlabel='connector (64 strips = 49.9 mm, one DREAM chip)',
                ylabel='ln gain (X: solid, Y: dashed)', margin=(24, 20, 92, 110))
    P.xticks([(k, str(k)) for k in range(8)]).yticks(plain_ticks(-0.08, 0.08, 0.04, '{:+.2f}'))
    P.hline(0, GREY, dash='4 4')
    for d in DETS:
        g = json.load(open(os.path.join(WSD, f'xy_gain_{d}.json')))
        a, b = np.array(g['a']), np.array(g['b'])
        P.line(list(range(8)), list(a - a.mean()), DCOL[d], w=3,
               tips=[f'{d} X connector {k}: {100 * x:+.1f} %, n = {n}' for k, (x, n) in enumerate(zip(a - a.mean(), g['n_x']))])
        P.line(list(range(8)), list(b - b.mean()), DCOL[d], w=3, dash='8 6',
               tips=[f'{d} Y connector {k}: {100 * x:+.1f} %, n = {n}' for k, (x, n) in enumerate(zip(b - b.mean(), g['n_y']))])
    body = (head('Connector gains (X vs Y): equal to ±2–5 %, so nothing to correct',
                 'ln(qx/qy) = c + a[X connector] − b[Y connector], M3-matched unsaturated single tracks (ws_dead_charge F15)',
                 'no', 'applying them moves the charge-balance width by ≤ 0.004') + P.svg('conn') +
            sd.legend([(d, DCOL[d]) for d in DETS]))
    D.slide('conn', body, '<p>Only the X-to-Y ratio per connector is measurable this way (the avalanche is shared). '
            'Source: <code>rerun/model_audit/ws_dead_charge/xy_gain_*.json</code>.</p>', short='connector gains')


def eff_img(d):
    f = os.path.expanduser(f'~/x17/cosmic_bench/Analysis/{ROWS[d]}/wft/efficiency/efficiency_map_sliding.png')
    if not os.path.exists(f):
        return sd.p(f'{d}: no map', 22, MUT)
    im = Image.open(f).convert('RGB'); W, H = im.size
    im = im.crop((0, int(0.08 * H), int(0.335 * W), H))
    im = im.resize((int(im.width * 0.5), int(im.height * 0.5)), Image.LANCZOS)
    b = io.BytesIO(); im.quantize(colors=200).save(b, 'PNG', optimize=True)
    return sd.img(b.getvalue(), w=300, tip=f'{d}: {os.path.relpath(f, os.path.expanduser("~"))}')


def s_eff(D):
    body = (head('Efficiency (reconstructed within 5 mm of M3): det4 striped and blind in a corner, det6/det7 patchy',
                 'Sliding 25 mm kernel, M3 reference frame; pre-rerun reconstruction (August), not re-made on rr1',
                 'no', 'QA only') + sd.row(*[eff_img(d) for d in DETS], gap=14))
    D.slide('eff', body, '<p>These maps predate the rr1 re-decode and the phys bundles (file dates Aug 13–19); a 5 mm '
            'match radius reads containment as much as efficiency (memory note on the M3 match radius). They show where, '
            'not how much. Source: <code>~/x17/cosmic_bench/Analysis/&lt;row&gt;/wft/efficiency/</code>.</p>', short='efficiency')


def s_dead(D):
    F = json.load(open(os.path.join(WSD, 'f12_table.json')))
    P = sd.Plot(1600, 560, x=(0, 400), y=(-0.5, 9.5), xlabel='strip position [mm]', ylabel='', margin=(24, 20, 92, 170))
    P.xticks(plain_ticks(0, 400, 50))
    labels = []
    for r, (d, p) in enumerate([(d, p) for d in DETS for p in 'xy']):
        y = 9 - r
        labels.append((y, f'{d} {p.upper()}'))
        b = bundle(d)
        for ch in b['dead'].get(p, []):
            P.raw(f'<rect x="{P.X(ch * 0.78) - 2:.1f}" y="{P.Y(y + 0.35):.1f}" width="4" height="{P.Y(y - 0.35) - P.Y(y + 0.35):.1f}" '
                  f'fill="{RED}"{sd.tipattr(f"{d} {p.upper()} channel {ch} ({ch * 0.78:.1f} mm): in the bundle dead list")}/>')
        for pos in F.get(f'{d}_{p}', {}).get('positions', []):
            if pos.get('cls') == 'detector':
                P.raw(f'<circle cx="{P.X(pos["pos"]):.1f}" cy="{P.Y(y - 0.45):.1f}" r="4" fill="{INK}"'
                      f'{sd.tipattr(f"{d} {p.upper()} {pos[chr(112) + chr(111) + chr(115)]:.1f} mm: chamber-side fault (F12)")}/>')
    for y, t in labels:
        P.raw(sd.T(P.x0 - 14, P.Y(y) + 7, t, 21, INK, 'end'))
    leg = sd.legend([('in the bundle dead list (censored in the fit)', RED, 'box'),
                     ('chamber-side fault found across datasets (F12)', INK, 'dot')])
    body = (head('Dead strips: the bundle lists vs the faults that follow the chamber',
                 'Per view, strip position; red = what the reconstruction censors, black = what F12 found by position',
                 'yes', 'bundle dead list: censored (unknown charge) and bridged by the seeder') + P.svg('dead') + leg)
    D.slide('dead', body, '<p>F12 (ws_dead_charge): det6/det7 masks miss 15–18 chamber strips, det4 over-masks; n_TOF bundles '
            'carry no dead list. Gating the corrected masks is pending.</p>', short='dead strips')


def s_bands(D):
    pj = os.path.expanduser('~/x17/cosmic_bench/satstudy/bands/patches.json')
    B = json.load(open(pj)) if os.path.exists(pj) else {}
    P = sd.Plot(1600, 500, x=(0, 400), y=(0, 1.05), xlabel='strip position [mm]',
                ylabel='band occupancy (per-view max = 1)', margin=(24, 20, 92, 110))
    P.xticks(plain_ticks(0, 400, 50)).yticks(plain_ticks(0, 1, 0.25, '{:.2f}'))
    rates = []
    for d in DETS:
        k = BAND_KEYS[d]
        if k not in B:
            continue
        rates.append(f'{d}: {100 * B[k]["p_band"]:.1f} % of triggers')
        for p, dash in (('x', None), ('y', '8 6')):
            op = B[k]['occ_profile'][p]
            vals = [v for kk, v in op.items() if kk != 'pos'][0] if len(op) > 1 else None
            if vals is None:
                continue
            v = np.array(vals, float); pos = np.array(op['pos'], float)
            if v.max() > 0:
                v = v / v.max()
            n = 8
            vb = v[:len(v) // n * n].reshape(-1, n).mean(1); pb = pos[:len(pos) // n * n].reshape(-1, n).mean(1)
            P.line(list(pb), list(vb), DCOL[d], w=2.5, dash=dash, markers=False, tip=f'{d} {p.upper()} band occupancy')
    body = (head('Noise bands: fixed patches on det3 and det7, whole-plane bands on det2/det4/det6',
                 'Fraction of band triggers covering each strip (rerun/bands); hover a curve for its chamber', 'partly',
                 'band veto per event in the bench driver; band-strip censoring for every dataset is FE item 13') +
            P.svg('bands') + sd.row(sd.legend([(d, DCOL[d]) for d in DETS]), sd.p('solid X · dashed Y', 24, MUT), gap=40))
    D.slide('bands', body, '<p>Band rate: ' + '; '.join(rates) + '.</p><p>Hotspot census deferred until after reprocessing (item 13). Source: '
            '<code>~/x17/cosmic_bench/satstudy/bands/patches.json</code>, report notes/mx17-noise-bands.</p>', short='noise bands')


def main():
    D = sd.Deck('MX17 chamber maps', 'Every per-chamber map of the five June bench chambers, and which ones the reconstruction uses')
    for f in (s_cover, s_bow, s_w0, s_scale, s_vt, s_column, s_gain, s_conn, s_eff, s_dead, s_bands):
        f(D)
    D.write(OUT, note_meta=dict(title='MX17 chamber maps', summary='Gallery of every per-chamber map (bow, drift tilt, '
                                'angle scale, v(t), column, gain, connector gains, efficiency, dead strips, noise bands) of '
                                'the five June bench chambers, each marked whether the reconstruction uses it.',
                                tags='x17, mx17, model audit', date='2026-10-27'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
