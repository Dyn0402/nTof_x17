#!/usr/bin/env python3
"""make_deck.py -- the det3 loss categories as a slide note: what each efficiency-budget category looks like in the
waveforms, the physics behind it, and whether a better reconstruction could recover it; the saturated-strip treatment
in detail (Dylan 10-26).

    ../../.venv/bin/python summarize.py && ../../.venv/bin/python make_deck.py [--out PATH]
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --slug mx17-det3-loss-categories --force --deploy

Reads ~/x17/cosmic_bench/satstudy/summary.json (summarize.py), the waveform caches of satfit.py / examples.py and the
production candidate table; anything still running shows as "pending".
"""
import argparse, json, os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402
from olib import load  # noqa: E402
sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN, INK, MUT, RULE,  # noqa: E402
                      DBLUE, DRED, DGREY, DGREEN, DMUT, DINK, term, T, line, arrow, poly)

OUT = S.OUT
V_D = 35.8e-3          # mm/ns, the golden bundle's v
TS = 60.0
CAND = os.path.expanduser('~/x17/cosmic_bench/outliers/final/mx17_det3_saturday_scan_6-27-26/'
                          'long_run_resist_490V_drift_1000V/mx17_3/wft/final_phys/events.candidates.parquet')
#: colour per kind of loss, the same on every slide
KIND = dict(det=GREEN, cut='#8fd1a8', reco=BLUE, defn=PURPLE, phys=ORANGE, ref=GREY, hw=RED)
M3C, FITC, CANDC = '#ff4d5e', '#ffffff', '#5cf08d'
CLIPC, BOTC = '#ff3b3b', '#47b5ff'

G = dict(
    ray=('An M3 reference track (χ² < 1, 4 clusters, one per event) crossing det3’s active area, 6-27 Saturday-scan '
         'long run, 490 V. Interior = ≥ 30 mm from an unpassivated edge, ≥ 8 mm from a passivated one.'),
    detected='Both views fitted, |r| ≤ 5 mm from M3 at the mesh (raw strip frame), ≤ 2 competing strips per view.',
    floor=('Seeding (wft.seed): a strip counts only if its significance is ≥ 10 % of the plane’s strongest strip '
           '(SIG_REL_FLOOR); a cluster needs ≥ 3 such strips (MIN_STRIPS).'),
    veto='SPARK_VETO_HITS = 50: an event with more than 50 strips above the 10 % floor (X+Y) is dropped before any fit.',
    ndrop='n_dropped: strips in competing clusters. Production rejects a view with n_dropped > 2.',
    sat=('wft.model.SAT = 3550 ADC (pedestal + common-mode subtracted). Samples ≥ SAT are censored: out of the charge '
         'solve, plus a one-sided penalty if the model falls below them.'),
    footprint='Footprint: where a straight track through the 30 mm gap leaves charge — [ref, ref + 30·tan] ± 2.5 mm.',
    m3='M3: the 4-plane micromegas reference telescope; its track is extrapolated to det3’s mesh.',
)


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #
_WF = {}


def wf(eid):
    if not _WF:
        for f in ('wf_det3.pkl', 'wf_examples.pkl', 'wf_deck.pkl'):
            p = os.path.join(OUT, f)
            if os.path.exists(p):
                _WF.update(pickle.load(open(p, 'rb'))['wf'])
    if eid not in _WF:
        from qa_config import get_config
        from wft import io as wio
        cfg = get_config(S.KEY); e = {}
        for v, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
            for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
                r = wio.FeuReader(f)
                if eid in set(r.event_ids.tolist()):
                    for _e, _ft, W in r.iter_events({eid}):
                        e[v] = W.astype(np.float16); e[f'noise_{v}'] = r.noise.astype(np.float16)
        _WF[eid] = e
        p = os.path.join(OUT, 'wf_deck.pkl')
        old = pickle.load(open(p, 'rb'))['wf'] if os.path.exists(p) else {}
        old[eid] = e; pickle.dump(dict(wf=old), open(p, 'wb'))
    return _WF[eid]


_PM = {}


def posmap(v):
    if not _PM:
        from wft.io import strip_position_map
        from qa_config import get_config
        cfg = get_config(S.KEY); pm = strip_position_map(cfg)
        _PM.update(x=pm[cfg.MX17_FEU_X], y=pm[cfg.MX17_FEU_Y])
    return _PM[v]


_D = {}


def row(eid):
    if not _D:
        d = load(S.DET); _D['d'] = d.set_index('event_id', drop=False); _D['lad'] = d.attrs['ladder']
        R = pd.read_parquet(os.path.expanduser('~/x17/cosmic_bench/effbudget/rays_det3.parquet'))
        _D['R'] = R.set_index('event_id', drop=False)
    if eid in _D['d'].index:
        return _D['d'].loc[eid].to_dict()
    r = _D['R'].loc[eid]
    return dict(event_id=eid, ref_mesh_x=r.mesh_x, ref_mesh_y=r.mesh_y, ref_tan_x=r.tan_x, ref_tan_y=r.tan_y,
                x_t0=np.nan, y_t0=np.nan, x_ok=False, y_ok=False)


# --------------------------------------------------------------------------- #
# event display: strip position x drift time, one view
# --------------------------------------------------------------------------- #
def evd(eid, v, w=780, h=470, zoom=20.0, cen=None, title=None, fit=True, lines=(), mark=False, xlabel=True,
        t0=None, tmax=1920, margin=(24, 24, 84, 100)):
    r = row(eid); ev = wf(eid)
    W = ev[v].astype(np.float32); pos = posmap(v); ok = np.isfinite(pos)
    o = np.argsort(np.where(ok, pos, np.inf))[:ok.sum()]; P, Wv = pos[o], W[o]
    pref = r.get(f'ref_mesh_{v}', np.nan); tref = r.get(f'ref_tan_{v}', 0.0)
    if cen is None:
        cen = pref + 15 * tref if np.isfinite(pref) else P[np.argmax(Wv.max(1))]
    sel = np.abs(P - cen) < zoom
    vmax = max(float(np.percentile(Wv[sel], 99.5)) if sel.any() else 100, 60.0)
    pl = sd.Plot(w, h, x=(cen - zoom, cen + zoom), y=(tmax, 0), xlabel=f'{v} strip position [mm]' if xlabel else '',
                 ylabel='drift time [ns]', title=title, margin=margin)
    step = 10 if zoom <= 12 else (20 if zoom <= 30 else 40)
    pl.xticks([(t, f'{t:.0f}') for t in np.arange(np.ceil((cen - zoom) / step) * step, cen + zoom, step)])
    pl.yticks([(t, f'{t:.0f}') for t in range(0, tmax + 1, 500)])
    pl.raw(f'<rect x="{pl.x0}" y="{pl.y0}" width="{pl.pw}" height="{pl.ph}" fill="{sd.seq_color(0)}"/>', back=True)
    pitch = 0.778; cells = []
    for i in np.flatnonzero(sel):
        for k in range(Wv.shape[1]):
            val = float(Wv[i, k])
            if val < 0.04 * vmax:
                continue
            cells.append((P[i] - pitch / 2, P[i] + pitch / 2, k * TS, (k + 1) * TS, val, None))
    pl.cells(cells, 0, vmax)
    if mark:
        for i in np.flatnonzero(sel):
            for k in range(Wv.shape[1]):
                if Wv[i, k] >= S_SAT:
                    pl.rect(P[i] - pitch / 2, P[i] + pitch / 2, k * TS, (k + 1) * TS, CLIPC, 1.6)
                elif Wv[i, k] <= -200 and Wv[i].max() >= S_SAT:
                    pl.rect(P[i] - pitch / 2, P[i] + pitch / 2, k * TS, (k + 1) * TS, BOTC, 1.6)
    xlo, xhi = cen - zoom, cen + zoom

    def seg(p0, ww, ta, tb, col, wd, dash=None, tip=None):
        # inline SVG does not clip: keep the part of x = p0 + ww (t - ta), t in [ta, tb], inside the panel
        ts = np.linspace(ta, tb, 200); xs = p0 + ww * (ts - ta)
        k = (xs >= xlo) & (xs <= xhi) & (ts >= 0) & (ts <= tmax)
        if k.sum() >= 2:
            pl.line([xs[k][0], xs[k][-1]], [ts[k][0], ts[k][-1]], col, w=wd, dash=dash, markers=False, tip=tip)

    for (p0, ww, t0f, col, tip) in lines:
        seg(p0, ww, t0f, t0f + 30.0 / V_D, col, 9, tip=tip)
    t0v = t0 if t0 is not None else r.get(f'{v}_t0', np.nan)
    if not np.isfinite(t0v):
        t0v = 257.5 if v == 'x' else 302.5
    if np.isfinite(pref):
        seg(pref, tref * V_D, t0v, t0v + 30.0 / V_D, M3C, 3.5, '10 7',
            f'M3 extrapolated through the gap: mesh {pref:.1f} mm, tan {tref:+.3f}')
    if fit and r.get(f'{v}_ok', False):
        p0, ww, t0f = r[f'{v}_p0'], r[f'{v}_w'], r[f'{v}_t0']
        seg(p0, ww, t0f, t0f + 30.0 / V_D, FITC, 3,
            tip=f'production wft fit: p0 {p0:.1f} mm, χ²/dof {r[f"{v}_chi2"] / max(r[f"{v}_dof"], 1):.0f}')
    return pl.svg(f'event {eid} {v}')


S_SAT = 3550.0


def evd_legend(fit=True, extra=(), mark=False):
    it = [('M3 reference', M3C, 'dash')]
    if fit:
        it.append(('production fit', '#9aa3b3', 'line'))
    it += list(extra)
    if mark:
        it += [('sample ≥ 3550 (censored)', CLIPC, 'box'), ('sample ≤ −200 on a clipped strip (fitted as data)', BOTC, 'box')]
    return sd.legend(it, size=22)


def cbar(h=420):
    return sd.colorbar(120, h, 0, 1, [(0, '0'), (0.5, '½'), (1, 'max')], 'ADC / event max')


# --------------------------------------------------------------------------- #
# cartoons
# --------------------------------------------------------------------------- #
def chamber(w=760, h=430, kind='mip'):
    """Cross-section of the drift gap: cathode on top, mesh + strips below, a track and its ionisation."""
    o = []
    x0, x1, yc, ym = 40, w - 40, 40, h - 90
    o.append(f'<rect x="{x0}" y="{yc}" width="{x1 - x0}" height="{ym - yc}" fill="#eef3fa" stroke="none"/>')
    o.append(line(x0, yc, x1, yc, INK, 4)); o.append(T(x1 - 4, yc - 10, 'cathode (drift HV)', 20, MUT, 'end'))
    o.append(line(x0, ym, x1, ym, INK, 2, dash='6 4')); o.append(T(x1 - 4, ym - 10, 'mesh', 20, MUT, 'end'))
    for k in range(int((x1 - x0) / 22)):
        xs = x0 + 6 + 22 * k
        o.append(f'<rect x="{xs}" y="{ym + 26}" width="14" height="10" fill="#c2a14d"/>')
    o.append(T((x0 + x1) / 2, ym + 66, 'resistive strips → DREAM, 60 ns samples', 20, MUT))
    o.append(T(x0 + 10, (yc + ym) / 2, '30 mm drift', 20, MUT, 'start', rot=None))
    rng = np.random.default_rng(3)
    ax, bx = (x0 + x1) / 2 - 60, (x0 + x1) / 2 + 40
    o.append(line(ax, yc - 30, bx + (bx - ax) * 30 / (ym - yc), ym + 30, RED, 3))
    for k in range(26):
        f = rng.uniform(0.02, 0.98); xx = ax + (bx - ax) * f; yy = yc + (ym - yc) * f
        o.append(f'<circle cx="{xx + rng.normal(0, 2):.1f}" cy="{yy:.1f}" r="3.5" fill="{BLUE}"/>')
    o.append(arrow(ax + 150, yc + 60, ax + 150, yc + 130, BLUE, 2.5))
    o.append(T(ax + 160, yc + 100, 'electrons drift down at v ≈ 36 µm/ns', 20, BLUE, 'start'))
    o.append(T(ax - 10, yc - 8, 'muon', 22, RED, 'end'))
    return sd.svg(w, h, ''.join(o), 'drift gap cartoon')


def mini(kind, w=360, h=250):
    """A strip-position x drift-time cartoon of each loss category (time down, as the displays)."""
    o = [f'<rect x="0" y="0" width="{w}" height="{h}" rx="10" fill="#1d2433"/>']
    X = lambda f: 20 + f * (w - 40)
    Y = lambda f: 20 + f * (h - 40)
    trk = lambda a, b, col='#fde725', wd=7, op=1: o.append(
        f'<line x1="{X(a[0]):.0f}" y1="{Y(a[1]):.0f}" x2="{X(b[0]):.0f}" y2="{Y(b[1]):.0f}" stroke="{col}" '
        f'stroke-width="{wd}" stroke-linecap="round" opacity="{op}"/>')
    m3 = lambda a, b: o.append(f'<line x1="{X(a[0]):.0f}" y1="{Y(a[1]):.0f}" x2="{X(b[0]):.0f}" y2="{Y(b[1]):.0f}" '
                               f'stroke="{M3C}" stroke-width="3" stroke-dasharray="8 6"/>')
    blob = lambda c, r, col='#fde725': o.append(f'<circle cx="{X(c[0]):.0f}" cy="{Y(c[1]):.0f}" r="{r}" fill="{col}"/>')
    if kind == 'mip':
        trk((.45, .1), (.55, .9)); m3((.45, .05), (.55, .95))
    elif kind == 'discharge':
        for yy in (.25, .3, .62, .67):
            trk((.02, yy), (.98, yy), '#35b779', 9, .8)
        trk((.45, .1), (.5, .5), '#fde725', 6); m3((.45, .05), (.55, .95))
    elif kind == 'muon+other':
        trk((.3, .1), (.36, .9)); m3((.3, .05), (.36, .95)); blob((.82, .5), 26, '#35b779')
    elif kind == 'shower':
        trk((.4, .1), (.45, .9)); m3((.4, .05), (.45, .95))
        for k in range(6):
            trk((.43, .45), (.55 + .07 * k, .45 + .03 * np.sin(k)), '#35b779', 5)
    elif kind == 'multi':
        trk((.18, .1), (.24, .9), '#fde725', 6); m3((.18, .05), (.24, .95))
        trk((.62, .15), (.85, .85), '#fde725', 9)
        o.append(f'<line x1="{X(.6):.0f}" y1="{Y(.08):.0f}" x2="{X(.88):.0f}" y2="{Y(.93):.0f}" stroke="#fff" stroke-width="2.5"/>')
    elif kind == 'delta':
        trk((.4, .1), (.46, .9)); m3((.4, .05), (.46, .95))
        pts = [(.43, .5), (.52, .47), (.6, .52), (.66, .45), (.7, .53), (.73, .5)]
        o.append(poly([X(a) for a, b in pts], [Y(b) for a, b in pts], '#fde725', 6))
        o.append(f'<line x1="{X(.3):.0f}" y1="{Y(.15):.0f}" x2="{X(.75):.0f}" y2="{Y(.85):.0f}" stroke="#fff" stroke-width="2.5"/>')
    elif kind == 'blob':
        trk((.4, .1), (.46, .9), '#fde725', 4, .7); m3((.4, .05), (.46, .95)); blob((.52, .68), 22)
        o.append(f'<line x1="{X(.25):.0f}" y1="{Y(.1):.0f}" x2="{X(.62):.0f}" y2="{Y(.9):.0f}" stroke="#fff" stroke-width="2.5"/>')
    elif kind == 'twoclose':
        trk((.35, .1), (.5, .9), '#fde725', 6); trk((.45, .12), (.6, .88), '#fde725', 6); m3((.35, .05), (.5, .95))
        o.append(f'<line x1="{X(.42):.0f}" y1="{Y(.08):.0f}" x2="{X(.52):.0f}" y2="{Y(.92):.0f}" stroke="#fff" stroke-width="2.5"/>')
    return sd.svg(w, h, ''.join(o), f'{kind} cartoon')


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def pc(x, d=2):
    return 'pending' if x != x else f'{x:.{d}f} %'


def s_setup(D, Sm):
    body = (sd.title('Reading the displays: a track is a line in strip position × drift time',
                     'Each view (x, y) is one plane of strips; the drift gap turns depth into arrival time') +
            sd.row(sd.col(chamber(), sd.p('Charge from the top of the gap arrives last. An inclined track becomes a '
                                          'tilted line, a vertical one a column.', 24, MUT), w=780),
                   sd.col(evd(_ex['mip'], 'x', w=560, h=470, zoom=12, title='a detected track (x view)'),
                          evd_legend(), w=560),
                   sd.col(cbar(470), w=130), gap=40))
    notes = (f'<p>Every display in this note: colour = the strip’s pedestal- and common-mode-subtracted ADC in each '
             f'60 ns sample, scaled to the event’s 99.5th percentile; the red dashed line is the {term("M3", G["m3"])} '
             f'track drawn through the 30 mm gap at the bundle’s v = 35.8 µm/ns from the fitted t0; the grey/white line '
             f'is the production waveform fit (wft, phys bundle). Waveforms are QA here, not geometry: the fit is the '
             f'only position estimator.</p><p>Event {_ex["mip"]} is a random detected interior ray.</p>')
    D.slide('setup', body, notes, short='reading')


def s_spark(D, Sm):
    sp = Sm['sparks']; b = Sm['budget']['interior']; by = sp['by_cls']; n = sp['n']; med = sp['med']
    rec = sp.get('nospark')
    if rec:
        rtxt = {c: f'veto off: M3 among the candidates in {g["any_both"]}/{g["n"]}' for c, g in rec.items()}
    else:
        rtxt = {c: 'veto-off re-run pending' for c in by}
    cards = []
    for c, ex, txt in (('discharge', _ex['discharge'], 'the plane itself breaks down: whole chips light up at once '
                                                        'and ring negative'),
                       ('muon+other', _ex['muon+other'], 'a clean muon on M3 plus an unrelated deposit or a noisy '
                                                         'chip elsewhere in the plane'),
                       ('shower', _ex['shower'], 'a branching / sideways deposit grows out of the track at the '
                                                 'crossing')):
        m = med[c]
        tip = (f'{by[c]} of {n} interior “spark” rays. Median: {m["mult"]:.0f} strips above the floor, '
               f'{m["x_non"]:.0f} / {m["y_non"]:.0f} strips above SNR 6 (x / y), {m["x_nsatstrip"]:.0f} / '
               f'{m["y_nsatstrip"]:.0f} on the rail, {m["x_nneg"]:.0f} / {m["y_nneg"]:.0f} below −10 σ')
        cards.append(sd.col(sd.p(f'<b>{c}</b> · {by[c]} of {n} · {rtxt.get(c, "")}', 24),
                            evd(ex, 'y', w=520, h=420, zoom=45 if c != 'discharge' else 80, fit=False,
                                xlabel=True, margin=(10, 18, 80, 96)),
                            sd.p(txt, 22, MUT), w=520))
    body = (sd.title(f'“Spark” ({b["spark"]:.1f} %) is three things; only a quarter are discharges, and only '
                     f'“muon + other” is recoverable',
                     f'{term("> 50 strips above the floor", G["veto"])} drops the event before any fit · y view of one '
                     f'example each') + sd.row(*cards, gap=52) + evd_legend(fit=False))
    notes = ('<p>Split of all 126 interior rays in the spark category (sparks.py), from the full-plane waveforms: '
             '<b>discharge</b> = a view with &gt; 150 strips above SNR 6 or &gt; 100 strips swinging below −10 σ '
             '(chip-wide ringing); <b>muon+other</b> = in both views the charge on the M3 footprint is a single, '
             'track-wide cluster and anything else sits &gt; 5 mm away; <b>shower</b> = the rest (the cluster at the '
             'crossing is wide or branching). Checked by eye on random displays of each class.</p>'
             '<p>What a discharge looks like here: the strips of whole DREAM chips (64 channels) jump together in the '
             'same samples — horizontal bands in the display — with saturated strips and a negative swing on '
             'hundreds of channels. In most examples the muon’s own trace is visible on M3 <i>before</i> the band, '
             'consistent with the earlier finding that det3 sparks are muon-induced (4× more likely with a track).</p>'
             '<p>“muon+other” is mostly a clean track plus either a second particle far away or a slow baseline swing '
             'on a distant chip that crosses the 10 % floor (the floor is relative to the brightest strip, and a '
             'modest MIP makes it low). The 50-hit veto counts both planes together, so an inclined track (40+ strips '
             'in y) plus anything else trips it.</p>'
             '<p>Recovery test: the production reconstruction (reco_variant.py --no-spark-veto, finalize environment) '
             'on all 126, scored with the budget’s rule (both views, r ≤ 5 mm, n_dropped ≤ 2). ' +
             ('; '.join(f'{c}: {g["detected"]}/{g["n"]} detected as production would count it (r ≤ 5 mm AND '
                        f'n_dropped ≤ 2), {g["near"]} within 5 mm, chosen fit agreeing with M3 in both views in '
                        f'{g["win_both"]}, some candidate agreeing in both views in {g["any_both"]}'
                        for c, g in rec.items()) if rec else 'Pending.') +
             ('. The chosen fits that agree carry n_dropped = ' +
              (f'{min(rec["muon+other"]["nd_win"])}–{max(rec["muon+other"]["nd_win"])}' if rec and rec['muon+other'].get('nd_win') else '?') +
              ' (muon+other): the other deposit is a competing cluster, so with the veto gone they would still fall to '
              'the n_dropped &gt; 2 cut. Lifting the veto recovers them only together with the cut decision of the '
              '“found, cut” category; and like MULTI, ' + (f'{rec["muon+other"]["any_both"] - rec["muon+other"]["win_both"]} of '
              f'the {rec["muon+other"]["n"]}' if rec else '') + ' have M3 only among the runners-up.' if rec else '') + '</p>'
             '<p>A better rule than a hit count: veto on the discharge signature itself (chip-wide simultaneous '
             'bands, the negative swing), and let the candidate fit handle busy-but-clean events.</p>')
    D.slide('spark', body, notes, short='spark')


def s_delta(D, Sm):
    dl = Sm['delta']
    xi = 0.1535 * 0.4506 * 1.66e-3 * 3.0 * 1e3     # keV, Ar at 1 atm over 30 mm
    Tk = np.array([3, 5, 10, 20, 30, 50, 100.0])
    rng_mm = np.array([0.3, 0.7, 2.0, 6.8, 13, 33, 110.0])   # approx. CSDA path in Ar (ESTAR water scaled)
    pl = sd.Plot(700, 470, x=(3, 100, 'log'), y=(0.002, 0.2, 'log'), xlabel='δ-electron energy T [keV]',
                 ylabel='P(δ above T) per crossing', title='knock-on electrons in 30 mm of argon', margin=(10, 90, 84, 100))
    pl.xticks([(3, '3'), (10, '10'), (30, '30'), (100, '100')])
    pl.yticks([(0.002, '0.2 %'), (0.01, '1 %'), (0.05, '5 %'), (0.2, '20 %')])
    pl.line(list(Tk), list(xi / Tk), ORANGE, tips=[f'T > {t:.0f} keV: {100 * xi / t:.1f} % of tracks; CSDA path ≈ '
                                                    f'{r:g} mm' for t, r in zip(Tk, rng_mm)])
    for t, r in zip(Tk, rng_mm):
        pl.text(t * 1.08, xi / t * 1.25, f'{r:g} mm', 19, MUT, 'start')
    pl.hline(dl_frac(Sm) / 100, PURPLE, label=f'observed DELTA class {dl_frac(Sm):.2f} %', tip='C22 DELTA, interior rays',
             anchor='start', where='below')
    pl.text(3.4, 0.0032, 'labels: path length of a δ of that energy', 19, MUT)
    wd, wc = np.array(dl['width_delta']), np.array(dl['width_ctrl'])
    e = np.arange(0, 41, 4); hd, _ = np.histogram(np.clip(wd, 0, 39.9), e); hc, _ = np.histogram(np.clip(wc, 0, 39.9), e)
    p2 = sd.Plot(560, 470, x=(0, 40), y=(0, 0.8), xlabel='widest cluster in the view [mm]', ylabel='fraction of events',
                 title='cluster width', margin=(10, 18, 84, 90))
    p2.xticks([(t, str(t)) for t in (0, 10, 20, 30, 40)]).yticks([(t, f'{t:.1f}') for t in (0, .2, .4, .6, .8)])
    for k in range(len(hd)):
        p2.vbar(e[k] + 1.2, hc[k] / max(len(wc), 1), 14, GREY, tip=f'controls {e[k]}–{e[k + 1]} mm: {hc[k]}')
        p2.vbar(e[k] + 2.8, hd[k] / max(len(wd), 1), 14, ORANGE, tip=f'DELTA {e[k]}–{e[k + 1]} mm: {hd[k]}')
    ex = _ex['delta']
    body = (sd.title('δ-rays: a knock-on electron leaves its own track; the straight-line fit takes both',
                     f'{term("Rutherford knock-on rate", "dN/dT = ξ/T², ξ = 0.1535 MeV cm²/g · Z/A · ρx ≈ 0.34 keV for 30 mm of Ar (Ar-based gas, 1 atm)")} '
                     f'vs what det3 shows · event {ex}') +
            sd.row(sd.col(pl.svg('delta rate'), w=700), sd.col(evd(ex, 'y', w=420, h=470, zoom=18, margin=(10, 18, 84, 90)), w=420),
                   sd.col(p2.svg('width'), sd.legend([('DELTA class', ORANGE, 'box'), ('controls', GREY, 'box')], 22), w=560), gap=24) +
            evd_legend())
    notes = (f'<p><b>Physically:</b> the muon kicks an atomic electron hard. The rate of a knock-on above T falls as '
             f'ξ/T (ξ ≈ {xi:.2f} keV for 30 mm of argon at 1 atm), and its path length rises steeply with T: a 10 keV '
             f'δ goes ~2 mm (a blob), a 30 keV one ~1 cm, a 50 keV one ~3 cm (CSDA path in argon, approximate: ESTAR '
             f'water values scaled by the stopping-power ratio; the projected extent is shorter because the electron '
             f'scatters). So about 1 % of crossings carry a δ long enough to widen the cluster by ≳ 8 mm — the '
             f'classifier’s threshold — and the observed DELTA share is {dl_frac(Sm):.2f} %.</p>'
             f'<p><b>In the data:</b> a wide cluster that grows sideways out of the track at one depth, often curling, '
             f'often ending in a bright spot (where the electron stops and dE/dx peaks). The fit has one straight line '
             f'and a non-negative charge profile along it, so it tilts to pass through the δ’s charge.</p>'
             f'<p><b>Recoverable?</b> Partly, in principle: the muon’s part of the column is usually still straight and '
             f'on M3. The two-track fit does not do it (it looks for a second straight line: any candidate agrees in '
             f'{dl["twotrack_any"]:.0f} % of DELTA views with it on, vs {dl["finalize"]:.0f} % in production). A robust '
             f'fit that down-weights charge far from the line, or a track + free-form-blob model, could. Not tried here. '
             f'The gas mixture of the June bench was not recorded; ξ uses pure argon, which the few-% quencher changes '
             f'by a few %.</p>')
    D.slide('delta', body, notes, short='δ-rays')


def dl_frac(Sm):
    return Sm['budget']['interior']['far_by_cls'].get('DELTA', 0)


def s_sat_now(D, Sm):
    """What the fit does with a saturated strip, on one real strip."""
    ex, v, i = _ex['satstrip']
    W = wf(ex)[v].astype(np.float32)[i]
    t = np.arange(32) * TS
    pl = sd.Plot(1000, 600, x=(0, 1920), y=(-600, 4300), xlabel='time [ns]', ylabel='ADC (pedestal + CM subtracted)',
                 title=f'one saturated strip (event {ex}, {v} view)', margin=(10, 30, 84, 110))
    pl.xticks([(k, str(k)) for k in range(0, 1921, 480)]).yticks([(k, str(k)) for k in (-500, 0, 1000, 2000, 3000, 4000)])
    rail = float(W.max())
    pl.band([0, 1920], [S_SAT, S_SAT], [4300, 4300], RED, 0.08, tip='censored region: samples here are not fitted')
    pl.hline(S_SAT, RED, label=f'SAT = 3550: censor level', tip=G['sat'])
    pl.hline(rail, INK, '3 4', label=f'this strip’s rail ≈ {rail:.0f}', tip='4095 − pedestal − common mode', where='above', anchor='start')
    lo = float(W.min())
    pl.hline(lo, BLUE, '3 4', label=f'bottom rail ≈ {lo:.0f} (ADC 0) — fitted as data', where='below', anchor='start')
    tips = [f't = {tt:.0f} ns: {w:.0f} ADC — ' + ('censored (≥ 3550)' if w >= S_SAT else 'fitted as a measurement')
            for tt, w in zip(t, W)]
    pl.line(list(t + 30), list(W), INK, w=3, tips=tips, r=6)
    for tt, w in zip(t, W):
        if w >= S_SAT:
            pl.points([tt + 30], [w], RED, r=8)
        elif w <= -200:
            pl.points([tt + 30], [w], BLUE, r=8)
    steps = [dict(label='censor', sub='samples ≥ 3550 drop out of the charge solve (NNLS)', color=RED, tip=G['sat']),
             dict(label='one-sided penalty', sub='χ² += ((data − model)/σ)² only where the model is BELOW a censored sample',
                  color=RED),
             dict(label='everything else', sub='rising edge, falling edge, undershoot: fitted with the linear template',
                  color=BLUE),
             dict(label='not used', sub='WFT_SAT_SOLVE (Tobit active set) exists, off: charge 5–12 % low either way',
                  color=GREY, strike=True)]
    body = (sd.title('What the fit does with a saturated strip today: a fixed 3550 cut, one side only',
                     'wft.model.prep_plane / _solve_nnls, production setting; one real det3 strip') +
            sd.row(sd.col(pl.svg('saturated strip'), sd.legend([('censored sample', RED, 'dot'),
                                                                 ('fitted, but on the bottom rail', BLUE, 'dot'),
                                                                 ('fitted', INK, 'dot')], 22), w=1000),
                   sd.col(*[sd.card(sd.p(f'<b>{s["label"]}</b>', 24) + sd.p(s['sub'], 22, MUT), tip=s.get('tip'))
                            for s in steps], w=600), gap=48))
    notes = ('<p>Code: <code>wft/model.py</code>. <code>prep_plane</code> builds the mask <code>sat = W_raw ≥ SAT</code> '
             '(SAT = the bundle’s <code>sat_adc</code> = 3550, on the pedestal + CM subtracted samples before the gain '
             'division; dead channels are added to the same mask). <code>_solve_nnls</code> drops masked samples from '
             'the weighted NNLS for the depth-charge profile, then adds <code>Σ max(0, W − model)² / σ²</code> over '
             'the masked samples: the model is pushed up to the clip but never pulled down. The χ² dof count excludes '
             'them. <code>WFT_SAT_SOLVE=1</code> (opt-in, off in production) re-solves with the model held at the clip '
             'on samples where it falls below (a Tobit active set); FINDINGS §41 found it does not fix the 5–12 % charge '
             'deficit of saturated tracks, and that angle/position were unchanged by censoring on the bench '
             'resolution sample.</p>'
             '<p>Nothing else knows about saturation: the rest of a clipped strip’s samples — the rising edge, the '
             'falling edge after the plateau and the undershoot — are fitted with the same linear, amplitude-independent '
             'electronics template as an unsaturated strip, and so are the neighbours’ copies through the sharing kernel. '
             'Next slide: what the rail actually looks like.</p>')
    D.slide('sat-now', body, notes, short='saturation: now')


def s_sat_rail(D, Sm):
    R = Sm['rail']; e = np.array(R['hist_edges']); h = np.array(R['hist_counts'])
    pl = sd.Plot(800, 560, x=(3000, 4200), y=(0, h.max() * 1.15), xlabel='strip maximum [ADC]', ylabel='strips',
                 title='where strips stop (strip max > 3000)', margin=(10, 24, 84, 100))
    pl.xticks([(k, str(k)) for k in range(3000, 4201, 300)]).yticks([(k, f'{k:,}') for k in range(0, int(h.max() * 1.15), 500)])
    for k in range(len(h)):
        pl.vbar((e[k] + e[k + 1]) / 2, h[k], 22, RED if e[k] >= 3550 else GREY, tip=f'{e[k]:.0f}–{e[k + 1]:.0f}: {h[k]} strips')
    pl.vline(3550, RED, label='SAT = 3550')
    sd_ = Sm['sat']
    p2 = sd.Plot(800, 560, x=(-0.5, 4.5), y=(0, 1), xlabel='strips on the rail in the view (hits)', ylabel='fit disagrees with M3',
                 title='saturation vs failure, single-cluster views', margin=(10, 24, 84, 100))
    labs = ['0', '1–2', '3–5', '6–10', '> 10']
    p2.xticks([(k, l) for k, l in enumerate(labs)]).yticks([(t, f'{100 * t:.0f} %') for t in (0, .25, .5, .75, 1)])
    for vi, (v, col) in enumerate((('x', BLUE), ('y', ORANGE))):
        rows = [r for r in sd_[v]['bins'] if r['sel'] == 'clean']
        for k, r in enumerate(rows):
            if r['n'] >= 5:
                p2.vbar(k - 0.18 + 0.36 * vi, r['far'], 34, col, tip=f'{v}, {labs[k]} saturated strips: {r["k"]}/{r["n"]} '
                                                                    f'disagree ({100 * r["far"]:.1f} %)')
    sx, sy = sd_['x'], sd_['y']
    body = (sd.title('The rail is not at 3550, and the more strips sit on it, the more fits fail',
                     f'det3 waveforms ({R["n_events"]:,} events) · all golden-row fits, one-cluster views') +
            sd.row(sd.col(pl.svg('rail'), w=800),
                   sd.col(p2.svg('far vs nsat'), sd.legend([('x view', BLUE, 'box'), ('y view', ORANGE, 'box')], 22), w=800), gap=56))
    notes = (f'<p><b>Left:</b> the maximum of every strip above 3000 ADC in the satfit sample. The pile-up is the '
             f'ADC rail, 4095 − pedestal − common mode: it sits at 3650–3900 and varies per channel and per sample, '
             f'not at the bundle’s 3550. Within one clipped plateau the CM moves the clipped value by a median '
             f'{R["plateau_sag_q"]["50"]:.0f} ADC (90 %: {R["plateau_sag_q"]["90"]:.0f}); '
             f'{R["n_inside_plateau_below_sat"]:,} of {R["n_plateau_samples"]:,} samples inside a plateau '
             f'({100 * R["n_inside_plateau_below_sat"] / R["n_plateau_samples"]:.1f} %) fall below 3550 and are fitted as '
             f'exact measurements. Plateaus last a median {R["plateau_len_q"]["50"]:.0f} samples '
             f'({60 * R["plateau_len_q"]["50"]:.0f} ns).</p>'
             f'<p><b>The bottom rail.</b> After the plateau, 58 % of clipped strips swing down to ADC 0: the strip’s '
             f'max − min is 4100 ADC, the full 12-bit range (median undershoot {R["undershoot_after_clip_q"]["50"]:.0f} '
             f'ADC; unsaturated bright strips undershoot by a median {-100 * R["undershoot_ratio_unsat_q"]["50"]:.1f} % '
             f'of their peak). Those bottom-clipped samples are fitted as data too. Whether the electronics are still '
             f'linear there (a template undershoot of a 3–10× larger true pulse) or recovering from overload cannot be '
             f'told from these strips alone.</p>'
             f'<p><b>Right:</b> production fits, views with one hit cluster, by the number of saturated hits. Fraction '
             f'disagreeing with M3 (|dθ| &gt; 5° or |res| &gt; 3 mm). {100 * sx["frac_events_nsat3"]:.0f} % (x) and '
             f'{100 * sy["frac_events_nsat3"]:.0f} % (y) of views have ≥ 3 saturated strips. When a saturated fit fails, '
             f'it reads too STEEP in {100 * sx["too_steep"]:.0f} % (x) / {100 * sy["too_steep"]:.0f} % (y) of cases '
             f'(median |tan| excess {sx["dtan_med"]:.2f} / {sy["dtan_med"]:.2f}); χ²/dof median: failed '
             f'{sx["chi2dof"]["sat_far"]:.0f}, saturated-but-fine {sx["chi2dof"]["sat_ok"]:.0f}, unsaturated '
             f'{sx["chi2dof"]["unsat"]:.0f} (x).</p>'
             '<p>Saturation is correlated with everything else that is bright (δ-rays, blobs, two particles), so the '
             'right panel is not by itself the effect of the censoring rule: the next two slides separate them.</p>')
    D.slide('sat-rail', body, notes, short='saturation: the rail')


def s_sat_blob(D, Sm):
    exs = _ex['satblob']
    disp = [sd.col(evd(e, v, w=440, h=470, zoom=14, mark=True, title=f'event {e}, {v}', margin=(10, 18, 84, 96)), w=440)
            for e, v in exs]
    body = (sd.title('What a failed saturated track looks like: a faint MIP line plus one compact, very bright blob',
                     'FIT_SAT examples: red boxes = censored samples, blue = bottom-rail samples fitted as data') +
            sd.row(*disp, sd.col(mini('blob', 280, 220), sd.p('the blob is a short δ-electron stopping: '
                                                               '5–20 keV in 1–5 mm, 1–3× the whole MIP track’s charge '
                                                               'on 2–4 strips', 22, MUT), w=280), gap=26) +
            evd_legend(mark=True))
    notes = ('<p>Most FIT_SAT failures are not a saturated MIP column. The track itself is faint and on M3; the '
             'saturated strips belong to one compact deposit deep in the gap, often 1–4 mm off the line. A low-energy '
             'knock-on electron (δ) of 5–20 keV travels only 1–5 mm in argon, depositing as much charge as the whole '
             'MIP track (≈ 3 cm × ~2.5 keV/cm) on a few strips — so it saturates them. P(T &gt; 5 keV) ≈ 7 %, '
             'P(T &gt; 10 keV) ≈ 3 % per crossing (previous slide’s ξ/T), the same order as the 10 % of views with '
             '≥ 3 saturated strips.</p>'
             '<p>Why the fit fails: its model is one straight line with a non-negative charge profile along it. The '
             'blob holds most of the charge, so χ² is dominated by it and the line tilts to reach it (the "too steep" '
             'of the previous slide). The next slide tests whether the treatment of the clipped samples is to blame: it '
             'is not — even with every saturated strip removed the line still tilts, because the blob’s unsaturated '
             'edges and neighbours hold enough charge off the MIP line.</p>'
             '<p>So the fix aimed at this failure is the model, not the mask: a straight track + one free compact '
             'deposit (position, time, charge, its own width), which would let the MIP part fix the line.</p>')
    D.slide('sat-blob', body, notes, short='saturation: blobs')


def s_sat_arms(D, Sm):
    A = Sm['arms']
    names = dict(prod='production (≥ 3550)', low='censor ≥ 3300', both='≥ 3300 + bottom rail',
                 tail='strip after its clip', strip='drop saturated strips')
    short = dict(prod='production', low='≥ 3300', both='+ bottom', tail='after clip', strip='drop strips')
    order = ['prod', 'low', 'both', 'tail', 'strip']
    if not A:
        body = (sd.title('Alternative saturation rules: refit pending', 'satfit.py is refitting the 924 saturated events') +
                sd.p('pending', 30, MUT))
        D.slide('sat-arms', body, 'Pending: satfit.py fit.', short='saturation: rules')
        return
    arms = [a for a in order if f'sat|{a}' in A]
    pl = sd.Plot(820, 560, x=(-0.6, len(arms) - 0.4), y=(0, 1.1), ylabel='views agreeing with M3',
                 title='saturated events (≥ 3 saturated hits in a view)', margin=(10, 24, 120, 100))
    pl.xticks([(k, '') for k in range(len(arms))]).yticks([(t, f'{100 * t:.0f} %') for t in (0, .25, .5, .75, 1)])
    for k, a in enumerate(arms):
        s = A[f'sat|{a}']; c = A.get(f'ctrl|{a}')
        pl.vbar(k - 0.17, s['agree'], 60, RED if a == 'prod' else BLUE,
                tip=f'{names[a]}: {100 * s["agree"]:.1f} % of {s["n"]} saturated event-views agree; median χ²/dof {s["chi2dof"]:.0f}')
        if c:
            pl.vbar(k + 0.17, c['agree'], 60, GREY, tip=f'{names[a]}: {100 * c["agree"]:.1f} % of {c["n"]} unsaturated controls agree')
        pl.raw(T(pl.X(k), pl.y0 + pl.ph + 34, short[a], 20, INK))
        if a != 'prod' and f'switch|{a}' in A:
            sw = A[f'switch|{a}']
            pl.text(k, 1.03, f'+{sw["gained"]} / −{sw["lost"]}', 20, INK, 'middle',
                    tip=f'vs production, per saturated view: {sw["gained"]} start agreeing, {sw["lost"]} stop agreeing, of {sw["n"]}')
    best = max((a for a in arms if a != 'prod'), key=lambda a: A[f'sat|{a}']['agree'])
    gain = A[f'sat|{best}']['agree'] - A['sat|prod']['agree']
    rows = []
    for a in arms:
        sw = A.get(f'switch|{a}')
        rows.append([names[a], f'{100 * A[f"sat|{a}"]["agree"]:.1f} %',
                     f'+{sw["gained"]} / −{sw["lost"]}' if sw else '–', f'{A[f"sat|{a}"]["n_cens"]:.0f}',
                     f'{A[f"sat|{a}"]["chi2dof"]:.0f}'])
    body = (sd.title(f'Changing the censoring moves saturated views by {100 * gain:+.0f} points at most: the rail '
                     f'is not what misleads the fit',
                     f'{A["sat|prod"]["n"]} saturated views refitted; only the censor mask differs · '
                     f'agree = |dθ| ≤ 5° and |res| ≤ 3 mm') +
            sd.row(sd.col(pl.svg('arms'), sd.legend([('saturated, production', RED, 'box'), ('saturated, alternative', BLUE, 'box'),
                                                     ('unsaturated controls', GREY, 'box')], 22), w=820),
                   sd.col(sd.table(['rule', 'agree', 'vs prod', 'censored', 'χ²/dof'], rows, size=21,
                                   widths=[300, 110, 120, 110, 90],
                                   tips=['median censored samples per window; χ²/dof median over the views'] * len(rows)),
                          sd.callout('Even dropping every saturated strip leaves the agreement where it was: the '
                                     'blob’s <b>unsaturated</b> charge pulls the line.', RED, 24), w=780), gap=48))
    notes = ('<p>satfit.py: each saturated event (≥ 3 saturated hits in a view; 924 events) and 400 unsaturated controls, '
             'refitted with the production fitter (wft.reco.fit_plane, phys bundle, t0 prior, WFT_PIVOT_START=1, which '
             'reproduces the table to &lt; 0.05 mm) on a rebuilt window (the SNR &gt; 6 cluster under the production '
             'fit ± 3 strips). Only the censor mask changes between arms: <b>low</b> = censor ≥ 3300; <b>both</b> = '
             '≥ 3300 and, on a strip that clipped, every sample ≤ −200 dropped; <b>tail</b> = everything after a strip’s '
             'first clipped sample; <b>strip</b> = every sample of a clipped strip. Numbers above the bars: views that '
             'start / stop agreeing with M3 against production.</p>'
             '<p>Controls have no saturated sample, so only low/both can change them (the 3300 cut can catch a bright '
             'unsaturated sample).</p>'
             '<p>Reading it: the failures are 91 % angle and 74 % position failures in production, and their |dθ| 90th '
             'percentile stays at ~31° under every rule. The rules do lower χ²/dof (the clipped and bottom-rail '
             'samples were badly modelled), but the line does not move: the blob’s rising edges and its unsaturated '
             'neighbours (through the sharing kernel) carry enough charge off the MIP line to tilt it on their own. '
             'Any rule in the set agrees in 77 % of views (vs 71 %), so per-event rule shopping would not help either.</p>'
             '<p>What this does not test: an amplitude-dependent template, a per-channel rail from the pedestal file, '
             'or a track + blob model. The first two are about charge (FINDINGS §41), not geometry; the third is the '
             'one aimed at this failure.</p>')
    D.slide('sat-arms', body, notes, short='saturation: rules')


def s_fit(D, Sm):
    b = Sm['budget']['interior']['far_by_cls']
    disp = [sd.col(evd(e, v, w=440, h=470, zoom=16, title=f'event {e}, {v}', margin=(10, 18, 84, 96)), w=440) for e, v in _ex['fit']]
    body = (sd.title('“Clean, fit off” is mostly two close tracks the classifier could not separate',
                     f'FIT class, {b.get("FIT", 0):.2f} % of interior rays: no saturation, one cluster within 5 mm of M3') +
            sd.row(*disp, sd.col(mini('twoclose', 280, 220), sd.p('two tracks a few mm apart, or a V from one '
                                                                   'point: the fit lands between them', 22, MUT), w=280), gap=26) +
            evd_legend())
    notes = ('<p>The C22 classifier calls a view "clean" when nothing else sits more than 5 mm from the footprint and '
             'the cluster is not too wide. By eye most FIT events are two tracks within a few mm (parallel, or a V from '
             'a common vertex: a δ emitted forward, or a pair), plus some long-tailed single tracks. A single straight '
             'line goes between them.</p><p>Recoverable? This is the two-track fit’s regime, below its separation '
             'limit (notes/two-track-limit: the trigger fires on ~0 % of inclined pairs under 12 mm). With '
             'WFT_TWO_TRACK_FIT=1 any candidate agrees in 35 % vs 34 % without: no gain today. Small share.</p>')
    D.slide('fit', body, notes, short='clean fit off')


def s_cover(D, Sm):
    b = Sm['budget']['interior']; rc = Sm.get('recat') or {}; mt = Sm.get('multitrack') or {}
    lost = 100 - b['detected']
    rec = recoverable(Sm)
    sgl = (mt.get('classes') or {}).get('S', {})
    body = (sd.kicker('MX17 det3 · bench cosmics · efficiency loss categories') +
            sd.title(f'Of det3’s {lost:.1f} % interior loss, {rec["sure"]:.1f} points are recoverable without the '
                     f'reference, and another {rec["multi"]:.1f} are a second particle the chamber did reconstruct') +
            sd.row(sd.bignum(f'{b["n"]:,}', 'interior M3 tracks', DBLUE, 'det3 golden row, 6-27 long run, 490 V'),
                   sd.bignum(pc(b['detected'], 1), 'detected today', DGREEN, 'production finalize_phys table'),
                   sd.bignum(pc(100 * sgl.get('noband', float('nan')), 1), 'single-particle events, band veto on',
                             DGREEN, f'{100 * sgl.get("frac", float("nan")):.0f} % of the interior: the number to quote first'),
                   sd.bignum(f'+{rec["sure"]:.1f} pts', 'reference-free fixes, tested', DRED, ', '.join(rec['sure_parts'])),
                   gap=48) +
            sd.p('New on 10-26 pm: the noise bands are a chamber effect (follows the detector across FEUs and days), the '
                 'multi-particle events are reconstructed as well as single tracks and paired x↔y by charge, and '
                 'head-ons are recovered down to 5 % gain with doubled noise by a cross-view rule. Hover dotted words and '
                 'points for definitions and counts.', 26, DMUT))
    D.slide('cover', body, cover_notes(Sm, rec), dark=True, short='cover')


def recoverable(Sm):
    """Points (of interior rays) shown to be recoverable by a reference-free change, from the runs that finished."""
    b = Sm['budget']['interior']; n = b['n']; parts, sure = [], 0.0
    bf = Sm.get('band_fix')
    if bf:
        k = bf['det_var'] - bf['det_prod']
        sure += 100 * k / n; parts.append(f'band veto {100 * k / n:.2f}')
    sp = Sm['sparks'].get('nospark'); rcc = (Sm.get('recat') or {}).get('counts', {})
    if sp:
        # muon+other that are NOT bands: the veto off AND the n_dropped cut relaxed recovers them (win_both)
        g = sp.get('muon+other', {}); share = rcc.get('spark:muon+other', 0) / max(g.get('n', 1), 1)
        k = g.get('win_both', 0) * share
        sure += 100 * k / n; parts.append(f'spark veto {100 * k / n:.2f}')
    ff = (Sm.get('headon') or {}).get('ho20') or Sm['fit_failed'].get('min2')
    if ff:
        sure += 100 * ff['detected'] / n; parts.append(f'head-on rule {100 * ff["detected"] / n:.2f}')
    mt = Sm.get('multitrack') or {}
    M = (mt.get('classes') or {}).get('M', {})
    multi = 100 * M.get('frac', 0) * M.get('any', 0)
    return dict(sure=sure, sure_parts=parts, multi=multi)


def cover_notes(Sm, rec):
    return (f'<p>Sample: every M3 recipe track (χ² &lt; 1, 4 clusters, one ray per event) crossing det3’s interior, '
            f'golden row <code>sat_det3</code> (mx17_det3_saturday_scan_6-27-26 / long_run_resist_490V_drift_1000V), '
            f'production table <code>…/mx17_3/wft/final_phys/events.parquet</code> (phys bundle, PIVOT_START). The budget '
            f'is rerun/effbudget; the far classes are C22’s classifier (rerun/outliers/classify.py); the re-categorisation '
            f'is summarize.recat().</p>'
            f'<p>"Recoverable" counts only what a reference-free change was shown to recover on these very events '
            f'(re-running the production reconstruction with the change, then scoring against M3): '
            f'{"; ".join(rec["sure_parts"])}. The multi-particle share ({rec["multi"]:.1f} points: the M3 track is among '
            f'det3’s reconstructed tracks) is a matter of the efficiency definition, not of the reconstruction.</p>'
            '<p>Reproduce: <code>rerun/satstudy/</code> README, run order. Outputs in ~/x17/cosmic_bench/satstudy/.</p>')


RECAT = [  # (key, label, kind, tip)
    ('band', 'noise band', 'reco', 'A whole-window baseline offset on a patch of strips (slides “bands”) cut the event by '
                                   'n_dropped or tripped the spark veto.'),
    ('multi:other_picked', 'second particle: other one fitted', 'defn', 'det3 sees ≥ 2 particles; the M3 one is '
                                                                         'reconstructed but production picked the other.'),
    ('multi:m3_picked', 'second particle: M3 one fitted, cut', 'defn', 'det3 sees ≥ 2 particles and fitted the M3 one; '
                                                                      'the n_dropped cut rejected the event.'),
    ('multi:m3_not_found', 'second particle: merged in one view', 'phys', 'Two tracks a few mm apart in one projection: '
                                                                          'one cluster, one fit.'),
    ('far:DELTA', 'δ-ray on the track', 'phys', 'A wide cluster riding on the track.'),
    ('far:FIT_SAT', 'saturated blob, fit off', 'reco', 'A compact saturated δ blob tilts the line.'),
    ('spark:shower', 'spark veto: shower', 'phys', 'A branching deposit at the crossing.'),
    ('spark:discharge', 'spark veto: discharge', 'hw', 'The plane discharged.'),
    ('spark:muon+other', 'spark veto: muon + other', 'reco', 'A clean muon plus an unrelated deposit (bands removed).'),
    ('far:FIT', 'clean, fit off', 'reco', 'Mostly two tracks within a few mm.'),
    ('head_on', 'head-on x track', 'reco', 'The old “fit failed”: a 2-strip x cluster never seeded.'),
    ('far:POS_TAIL', 'position tail (≈ ⅔ bent M3)', 'ref', 'Right angle, wrong place; mostly M3 kinks.'),
    ('far:EDGE', 'edge (classifier)', 'ref', 'Footprint near an edge of the other coordinate.'),
    ('far:MULTI', 'second deposit, not particle-like', 'phys', 'C22 sees another deposit; det3’s particle count does not.'),
]


def s_budget(D, Sm):
    b = Sm['budget']['interior']; rc = Sm['recat']; pct = rc['pct']; cnt = rc['counts']
    rows = [(lab, pct.get(k, 0.0), KIND[kind], f'{tip}\n{pct.get(k, 0):.2f} % = {cnt.get(k, 0)} of {rc["n"]:,} interior rays')
            for k, lab, kind, tip in RECAT]
    rest = 100 - b['detected'] - sum(r[1] for r in rows)
    body = (sd.title(f'{100 - b["detected"]:.1f} % of interior tracks are lost: where each one went, re-categorised',
                     f'det3 interior, {b["n"]:,} M3 tracks, R = 5 mm · noise bands and second particles are now their own '
                     f'classes') +
            sd.row(sd.hbars(rows, 1.6, width=600, h=29, label_w=470, fmt=lambda v: f'{v:.2f} %', size=23),
                   sd.col(sd.legend([('recoverable by reconstruction', KIND['reco'], 'box'),
                                     ('physics at the crossing', KIND['phys'], 'box'),
                                     ('efficiency definition (two particles)', KIND['defn'], 'box'),
                                     ('detector (discharge)', KIND['hw'], 'box'),
                                     ('reference / geometry', KIND['ref'], 'box')], size=23),
                          sd.callout(f'Detected: <b>{b["detected"]:.1f} %</b>', GREEN, 30),
                          sd.p(f'not shown: {rest:.2f} % (charge elsewhere, other cuts)', 20, MUT), w=480), gap=36))
    notes = ('<p>summarize.recat(): first match wins, in the order detected, band, multi-particle, spark, the C22 far '
             'classes, head-on. <b>band</b>: the event carries a noise band (bands.py: ≥ 8 strips of a chip offset by '
             '&gt; 5σ and flat) and production lost it to the n_dropped cut or the 50-hit spark veto, and sparks.py '
             'does not call it a discharge. <b>multi-particle</b>: multitrack.py class M, ≥ 2 particle-like candidate '
             'clusters in a view (time extent ≤ 24 samples); the old “found, cut” second-particle events are '
             '“M3 one fitted, cut”. The spark row lost the bands it used to contain: '
             f'{cnt.get("spark:muon+other", 0)} of the {Sm["sparks"]["by_cls"].get("muon+other", 0)} “muon + other” '
             'remain.</p>')
    D.slide('budget', body, notes, short='budget')


# --------------------------------------------------------------------------- #
# noise bands
# --------------------------------------------------------------------------- #
def band_profiles(v, b, n=9, w=560, h=440):
    Z = np.load(os.path.join(OUT, 'bands_scan.npz'))
    import bands as Bd
    m0 = Z[f'm0_{v}'][:, 64 * b:64 * b + 64].astype(np.float32)
    ev = np.flatnonzero(Bd.band_events(Z, v)[:, b])
    q = np.flatnonzero(~Bd.band_events(Z, v).any(1))[:3]
    rng = np.random.default_rng(4); pick = rng.choice(ev, min(n, len(ev)), replace=False)
    pl = sd.Plot(w, h, x=(0, 63), y=(-3500, 3500), xlabel=f'channel in {v} chip {b}', ylabel='window mean − chip median [ADC]',
                 title=f'{v} chip {b}, before common mode', margin=(10, 18, 84, 110))
    pl.xticks([(t, str(t)) for t in (0, 16, 32, 48, 63)]).yticks([(t, f'{t:+d}' if t else '0') for t in (-3000, -1500, 0, 1500, 3000)])
    cols = [BLUE, ORANGE, PURPLE, RED, GOLD, '#2a9d8f', '#8d6e63', '#e76f51', '#6d597a']
    for i, k in enumerate(pick):
        y = m0[k] - np.median(m0[k])
        pl.line(np.arange(64), np.clip(y, -3500, 3500), cols[i % len(cols)], w=2, markers=False,
                tip=f'trigger {int(Z[f"eid_{v}"][k])}: a band event')
    for k in q:
        y = m0[k] - np.median(m0[k])
        pl.line(np.arange(64), y, INK, w=3, markers=False, tip='a trigger without a band')
    return pl.svg(f'{v} chip {b} profiles')


def s_bands_what(D, Sm):
    B = Sm.get('bands')
    if not B:
        return
    runs = B['runs']
    name = dict(sat_det3='det3 · 6-27 (golden)', g_det3_wknd='det3 · 6-27 night', g_det3='det3 · 6-22', g_det2='det2 · 6-22',
                g_det4='det4 · 6-24', g_det6_long='det6 · 6-26', g_det7_long='det7 · 6-26')
    rows = [(name.get(k, k), f'{r["x"]["feu"]}/{r["y"]["feu"]}', f'{100 * r["x"]["rate"]:.1f}', f'{100 * r["y"]["rate"]:.1f}',
             f'{100 * r["p_y_given_x"]:.0f}') for k, r in runs.items()]
    body = (sd.title('Noise bands: a flat baseline offset on a fixed patch of each chamber, in x and y at once',
                     'whole 1.9 µs window, ±50–3000 ADC, strip by strip · 3–4 % of all det3 triggers, up to 19 % on det7') +
            sd.row(sd.col(band_profiles('x', 0), w=560),
                   sd.col(band_profiles('y', 7), w=560),
                   sd.col(sd.table(['run', 'FEUs', 'x %', 'y %', 'y|x %'], rows, size=20),
                          sd.callout('det3 shows the same chips (x 0, y 5–7) on FEUs 7/8 and on FEUs 3/4, five days '
                                     'apart: it moves with the <b>chamber</b>, not the electronics', BLUE, 22), w=560),
                   gap=26))
    g = B['golden']
    notes = (f'<p>bands.py. A channel is offset in a trigger when its common-mode-subtracted window mean is &gt; 5σ of its '
             f'pedestal noise while its RMS about that mean stays &lt; 2σ: shifted, not pulsing. A band = ≥ 8 such '
             f'channels in one 64-channel chip. Left / middle: the pedestal-subtracted window mean of each channel minus '
             f'the chip median, before common mode, for random band triggers (colour) and three ordinary triggers '
             f'(black). The offset sits on a contiguous run of channels (x: 32–63, y: 0–25 of the chip), up to the ADC '
             f'rails; the block-median common mode removes most of it, which is why the displays show only ~100–300 ADC '
             f'(median offset after CM {g["x"]["amp_q"]["50"]:.0f} / {g["y"]["amp_q"]["50"]:.0f} ADC x / y, 99 % '
             f'{g["x"]["amp_q"]["99"]:.0f}). Both signs, equally ({100 * g["x"]["frac_positive"]:.0f} % positive in x).</p>'
             f'<p>Table: rate of triggers with a band per view, all bench chambers (golden row: every trigger; others: the '
             f'first 3 000 per file) and how often y has one when x does. On det3 the x band lives on strips x &lt; 50 mm '
             f'and the y band on y &gt; 250 mm: one corner of the chamber, seen by both strip layers.</p>')
    D.slide('bands', body, notes, short='bands: what')


def s_bands_not(D, Sm):
    B = Sm.get('bands')
    if not B:
        return
    nl = B.get('nulls', {}); dc = B.get('decay', {}); g = B['golden']
    lag = nl.get('p_band_given_spark_lag', {})
    hv = nl.get('hv_corr', {})
    rows = [
        ('time in the run (2.3 h)', f'{min(g["x"]["rate_per_10min"]) * 100:.1f}–{max(g["x"]["rate_per_10min"]) * 100:.1f} % per 10 min', 'steady'),
        ('a discharge 1–2 triggers before / after', ' / '.join(f'{100 * lag.get(k, float("nan")):.1f}' for k in ('-2', '-1', '1', '2')) + f' % vs {100 * nl.get("p_band", float("nan")):.1f} %', 'no'),
        ('time since the previous trigger (quartiles)', ' / '.join(f'{100 * x:.1f}' for x in nl.get('p_band_by_dt_quartile', [])) + ' %', 'no'),
        ('DREAM SCA cell of the window', f'χ² = {nl.get("sca_chi2_ndf", [float("nan")])[0]:.0f} / 31', 'no'),
        ('a period of 10–32 ms (mains)', f'R = {nl.get("period_R_max", float("nan")):.3f}, noise {nl.get("period_R_random", float("nan")):.3f}', 'no'),
        ('HV currents (per-minute rate vs imon)', f'|r| ≤ {max(abs(x) for x in hv.values()) if hv else float("nan"):.2f}', 'no'),
        ('in-window discharges, by chip', 'all chips alike', 'not where the bands are'),
        ('a decay inside the window', (f'median slope {1e3 * dc["x"]["rel_slope_q"]["50"]:+.3f} / ms of the offset' if dc else 'pending'), 'none: τ ≫ 100 µs'),
        ('common-mode coupling (per-channel slope)', 'removes nothing', 'not a CM gain'),
        ('the readout (FEU, cable, slot)', 'det3 on 7/8 and 3/4: same chips', 'no: the chamber'),
    ]
    body = (sd.title('What the bands are not: every handle in the data comes back null except the chamber itself',
                     'golden row, every trigger · x view (y gives the same answers)') +
            sd.row(sd.table(['tested against', 'measured', 'correlates?'], rows, size=22, widths=[520, 600, 380]), gap=20) +
            sd.p('Left standing: a slow, intermittent source fixed on each chamber that couples to both strip layers over a '
                 'patch of strips (det3: the corner x &lt; 50 mm, y &gt; 250 mm), e.g. a charging or intermittent '
                 'contact at the resistive-layer or mesh connection there. The chambers are gone, so the mechanism stays '
                 'a stated limitation; the data fix below does not depend on it.', 23, MUT))
    notes = ('<p>bands.py report → bands.json (nulls, decay). SCA cell: (DREAM timestamp // 6) mod 512, 32 bins. Period: '
             'largest Rayleigh R of the band-trigger timestamps over 4 401 periods 10–32 ms (tick = 10 ns: the run spans '
             '8 442 s in the HV log, 844·10⁹ ticks); “noise” is the expected maximum for random phases. Discharge: '
             '&gt; 150 strips above 6σ (x + y). HV: hv_monitor.csv of the run, every imon channel. In-window '
             'discharges: ≥ 8 strips of a chip above 3000 ADC — they are spread over all chips while the bands sit on x '
             'chip 0 and y chips 5–7. Decay: per band channel with |offset| &gt; 300 ADC, the least-squares slope of its '
             'waveform over the window divided by the offset; an exponential tail of time constant τ gives −1/τ. '
             'Common mode: fitting each channel’s residual against its chip’s common mode (all triggers) removes '
             'nothing — the offset is on a sub-range of the chip and independent of the chip’s own shift.</p>')
    D.slide('bands-not', body, notes, short='bands: not')


def s_bands_fix(D, Sm):
    B = Sm.get('bands'); bf = Sm.get('band_fix')
    if not B:
        return
    c = B['cost']; h = B['hits']
    t, f = c['True'], c['False']
    rows = [('no band', 100 * f['detected'], GREEN, f'{f["n"]:,} interior rays'),
            ('band in the event', 100 * t['detected'], RED, f'{t["n"]} interior rays ({100 * c["band_frac"]:.1f} %)')]
    if bf:
        rows.append(('band, veto (both ends)', 100 * bf['det_var'] / bf['n'], GREEN,
                     f'WFT_DROP_FLAT_HITS=1 on all {bf["n"]}: {bf["det_prod"]} → {bf["det_var"]}'))
    bl = Sm.get('band_fix_left')
    if bl:
        rows.append(('band, veto (first sample)', 100 * bl['det_var'] / bl['n'], DGREEN,
                     f'WFT_DROP_FLAT_HITS=left on all {bl["n"]}: {bl["det_prod"]} → {bl["det_var"]}; controls '
                     f'{bl["controls"]["det_var"]}/{bl["controls"]["n"]}, {bl["controls"]["views_moved"]} views moved'))
    ctxt = (f'controls: {bf["controls"]["det_var"]}/{bf["controls"]["n"]} detected with the veto '
            f'(production {bf["controls"]["det_prod"]}), {bf["controls"]["views_moved"]} view(s) moved' if bf else 'veto re-run pending')
    body = (sd.title('The bands cost 1.4 % of det3’s tracks; dropping hits that span the whole window recovers them',
                     'a particle’s signal lasts at most a drift column (~20 samples); a band strip is above threshold '
                     'from the first sample to the last') +
            sd.row(sd.col(sd.p('<b>Detected</b>, interior M3 tracks', 24),
                          sd.hbars(rows, 100, width=420, h=40, label_w=260, fmt=lambda v: f'{v:.1f} %', size=24), w=760),
                   sd.col(sd.table(['hits', 'n', 'truncated both sides'],
                                   [('on band strips', f'{h["band_hits"]:,}', pc(100 * h['band_both_trunc'], 1)),
                                    ('all other', f'{h["other_hits"]:,}', pc(100 * h['other_both_trunc'], 2)),
                                    ('other, significance > 20', '', pc(100 * h['strong_other_both_trunc'], 2))], size=22),
                          sd.callout(ctxt, BLUE, 22), w=700), gap=40))
    notes = (f'<p>Cost (bands.py report): {t["n"]} of {t["n"] + f["n"]:,} interior rays have a band in the trigger; '
             f'{100 * t["detected"]:.1f} % of them are detected against {100 * f["detected"]:.1f} % without. Where '
             f'they go: {", ".join(f"{k} {100 * v:.1f} %" for k, v in t["cats"].items())}. The band strips form a '
             f'competing cluster (n_dropped &gt; 2) or push the event over the 50-hit spark veto; the track itself is '
             f'usually fine (far rate unchanged). Note the loss does not depend on where the track is: the cut and the '
             f'veto are event-wide.</p>'
             f'<p>Fix (reco_variant.py --drop-flat): drop every hit with trunc_left AND trunc_right (above threshold at '
             f'the first and the last sample of the window) before seeding, so it neither clusters nor counts toward '
             f'the veto. That is {100 * h["band_both_trunc"]:.0f} % of the hits on band strips and '
             f'{100 * h["other_both_trunc"]:.2f} % of all others (these are mostly saturated or discharge strips).' +
             (f' Re-running production with it on all {bf["n"]} interior band events: {bf["det_prod"]} → '
              f'{bf["det_var"]} detected; by production category recovered: {bf.get("recovered_by_cat")}. {ctxt}.'
              if bf else ' Re-run pending.') + '</p>')
    D.slide('bands-fix', body, notes, short='bands: fix')


# --------------------------------------------------------------------------- #
# multi-particle events
# --------------------------------------------------------------------------- #
def s_multi(D, Sm):
    ms = Sm.get('multi_sub'); mt = Sm['multitrack']; tt = Sm.get('two_track') or {}
    if not ms:
        return
    sub = ms['sub']; q = ms['quality']; n = ms['n_interior']
    lab = dict(m3_picked='M3 particle fitted (cut by n_dropped)', other_picked='other particle fitted, M3 one also found',
               m3_not_found='M3 particle not found')
    rows = [(lab[k], 100 * sub[k]['n'] / n, c, f'{sub[k]["n"]} events; topology x×y {sub[k]["topo"]}; separation of the '
             f'other particle median {sub[k]["sep_q"]["50"]:.0f} mm')
            for k, c in (('m3_picked', KIND['defn']), ('other_picked', KIND['defn']), ('m3_not_found', KIND['phys']))]
    qrows = [(nm, f'{q[k]["x"]["res_sigma"]:.2f} / {q[k]["y"]["res_sigma"]:.2f}', f'{q[k]["x"]["dtheta_sigma"]:.2f} / {q[k]["y"]["dtheta_sigma"]:.2f}',
              f'{100 * q[k]["x"]["agree"]:.0f} / {100 * q[k]["y"]["agree"]:.0f} %')
             for k, nm in (('S', 'single particle'), ('M', 'multi-particle'))]
    ex = _ex['merged']
    disp = evd(ex[0], ex[1], w=520, h=430, zoom=22, title=f'M3 not found: event {ex[0]}, {ex[1]}', margin=(10, 18, 84, 96))
    body = (sd.title(f'{ms["n"]} events ({100 * ms["n"] / n:.1f} %) hold two particles; the M3 one is reconstructed as well '
                     f'as a single track in {100 * (sub["m3_picked"]["n"] + sub["other_picked"]["n"]) / ms["n"]:.0f} % of them',
                     'multi-particle = ≥ 2 particle clusters in a view, from det3 alone') +
            sd.row(sd.col(sd.hbars(rows, 1.4, width=360, h=36, label_w=440, fmt=lambda v: f'{v:.2f} %', size=22),
                          sd.table(['events', 'σ position x / y [mm]', 'σ angle x / y [°]', 'agrees x / y'], qrows, size=22),
                          w=900),
                   sd.col(disp, sd.p('two tracks ~5–20 mm apart in this view merge into one seed; the other view '
                                     'separates them', 21, MUT),
                          sd.callout((lambda r: f'joint two-track fit (WFT_TWO_TRACK_FIT): {r["m3_not_found"]["any_tt"]}/'
                                                f'{r["m3_not_found"]["n"]} of these recovered; {r["controls"]["split"]}/'
                                                f'{r["controls"]["n"]} single tracks split')(tt['reco_tt']) if tt.get('reco_tt')
                                     else 'two-track fit pending', BLUE, 20), w=520), gap=36) + evd_legend())
    ttn = ''
    for arm, nm in (('reco_tt', 'joint two-track fit (production settings)'), ('reco_ttg', 'two-track fit, grid search + '
                                                                                            'own-χ² scale, all candidates')):
        r = tt.get(arm)
        if r:
            ttn += (f'<li>{nm}: M3 track found (both views, any candidate) in M3-not-found events '
                    f'{r["m3_not_found"]["any_tt"]}/{r["m3_not_found"]["n"]} (production {r["m3_not_found"]["any_prod"]}); '
                    f'other-picked {r["other_picked"]["any_tt"]}/{r["other_picked"]["n"]}; controls: '
                    f'{r["controls"]["split"]} of {r["controls"]["n"]} single tracks split, {r["controls"]["views_moved"]} '
                    f'views moved.</li>')
    notes = (f'<p>multitrack.py sub. Quality: per view the particle candidate nearest M3 within 5 mm, robust σ of its '
             f'residual at the mesh and of its ladder-scaled angle difference, and the fraction agreeing with M3 '
             f'(|dθ| ≤ 5°, |res| ≤ 3 mm). The M3 particle in a two-particle event is fitted to the single-track resolution; '
             f'the slightly lower agreement comes from the closest pairs, whose charge overlaps the window.</p>'
             f'<p><b>M3 picked</b> ({sub["m3_picked"]["n"]}): the old “found, cut” second particle — production '
             f'chose correctly, the n_dropped cut threw the event away. <b>Other picked</b> '
             f'({sub["other_picked"]["n"]}): the other particle is brighter (charge ratio median '
             f'{sub["other_picked"]["qratio_med"]:.1f}) and the largest-cluster ranking prefers it; the M3 track is '
             f'among the candidates. Both are detected with multi-track credit. <b>M3 not found</b> '
             f'({sub["m3_not_found"]["n"]}): {sub["m3_not_found"]["topo"].get("1x2", 0) + sub["m3_not_found"]["topo"].get("2x1", 0)} '
             f'of them see one cluster in one view and two in the other — the two tracks overlap in that projection, '
             f'within the 12 mm seed gap, and one fit lands on the other track or between them.</p>'
             + (f'<p>Can the joint two-track fit (wft/TWO_TRACK_FIT_2026-09-16.md; its cross-view corroboration lowers '
                f'the split threshold exactly when the other view shows two tracks) split them?<ul>{ttn}</ul></p>' if ttn
                else '<p>Joint two-track fit re-run pending.</p>'))
    D.slide('multi', body, notes, short='two particles')


def s_pairing(D, Sm):
    toy = Sm.get('pairing_toy'); mt = Sm['multitrack']; ov = Sm.get('overlay')
    if not toy:
        return
    pr = mt['pairing']
    pl = sd.Plot(700, 470, x=(0.01, 3, 'log'), y=(0.4, 1.0), xlabel='|ln(Q₁/Q₂)|: how different the two tracks’ charges are',
                 ylabel='x↔y paired right', title='data-driven toy: real single tracks paired at random', margin=(10, 24, 84, 100))
    pl.xticks([(0.01, '0.01'), (0.1, '0.1'), (1, '1')]).yticks([(t, f'{100 * t:.0f} %') for t in (0.4, 0.6, 0.8, 1.0)])
    for nm, col in (('q', BLUE), ('q_u', GREEN), ('u', GREY)):
        bins = toy['by_ratio'][nm]
        xs = [np.sqrt(max(b['lo'], 0.01) * min(b['hi'], 3)) for b in bins]
        pl.line(xs, [b['right'] for b in bins], col, w=3,
                tips=[f'{b["lo"]}–{b["hi"]}: {100 * b["right"]:.1f} % of {b["n"]:,} pairs' for b in bins])
    pl.vline(toy['lq_sigma'], RED, label='x/y charge spread of one track')
    rows = [('toy, charge', pc(100 * toy['right']['q'], 1), f'{toy["n_pairs"]:,} pairs'),
            ('toy, charge + depth profile', pc(100 * toy['right']['q_u'], 1), ''),
            ('real two-track events, charge', pc(100 * pr['right']['q'], 1), f'{pr["n"]} events, M3 on one track'),
            ('real, + depth profile + t0', pc(100 * pr['right']['all'], 1), '')]
    if ov and ov.get('ge24'):
        rows += [('overlay ≥ 24 mm, charge', pc(100 * ov['ge24']['right_q'], 1), f'{ov["ge24"]["n"]} overlays, truth on both'),
                 ('overlay ≥ 24 mm, + profile', pc(100 * ov['ge24']['right_q_u'], 1), '')]
    body = (sd.title(f'x↔y pairing of two tracks by charge: {100 * pr["right"]["q"]:.0f} % right on real events, explained '
                     f'by a toy built from real single tracks', 'one particle leaves the same charge in x and y; two '
                     'particles rarely leave the same charge') +
            sd.row(sd.col(pl.svg('pairing vs ratio'), sd.legend([('charge only', BLUE, 'line'), ('charge + depth profile', GREEN, 'line'),
                                                                  ('depth profile only', GREY, 'line')], 21), w=700),
                   sd.col(sd.table(['sample', 'right', ''], rows, size=22), w=760), gap=40))
    ovn = ''
    if ov:
        ovn = ('<p><b>Overlay bench</b> (overlay.py; the bench version of intra_bench): two clean single tracks of the '
               'same file and trigger phase, t0 within 30 ns, summed in the waveforms; production seeding and fit '
               'unchanged; truth = each donor’s own fit. Per separation bin (min of x, y): ' +
               '; '.join(f'{b["lo"]}–{b["hi"]} mm: both found {100 * b["both_found"]:.0f} %' +
                         (f', paired right by charge {100 * b["right_q"]:.0f} %, + profile {100 * b["right_q_u"]:.0f} %, '
                          f'production gate {100 * b["prod_right"]:.0f} %' if b['right_q'] is not None else '')
                         for b in ov['bins']) + f'. {ov["n_single"]} single-donor controls reproduce production: '
               f'{100 * ov["single_ok"]:.0f} %.</p>')
    notes = (f'<p>pairing.py. The rule (wft.reco.xy_pair_cost’s form): of the two assignments take the one with the '
             f'lower summed cost, cost = Σ min(z², 25) of the x−y differences lq = ln(q_x/q_y), u50, u90 (charge-arrival '
             f'quantiles) and, for real events only, t0; medians and robust σ from clean single tracks (σ(lq) = '
             f'{toy["lq_sigma"]:.3f}: one track’s x and y charges agree to 5 %).</p>'
             f'<p><b>Toy</b>: {toy["n_singles"]:,} clean single tracks, calibration on odd event ids, test on even; '
             f'{toy["n_pairs"]:,} random pairs, each track keeping its own x and y. It is the infinite-separation limit '
             f'(no merging, no shared strips). The only failures are pairs whose charges agree to better than '
             f'~2σ(lq) — Landau fluctuations make that rare (12 % of pairs within 10 %). The depth profile is an '
             f'independent handle (80 % alone) that rescues half of those. t0 is left out of the toy (every donor has '
             f'its own trigger); on real events, where both tracks share one, t0 alone pairs 65 %.</p>'
             f'<p><b>Real</b>: the {pr["n"]} multi-particle events with exactly two particle clusters per view and the '
             f'M3 track on one of each; right = the M3-matched x and y end up together. The toy and the real events '
             f'agree ({100 * toy["right"]["q"]:.1f} vs {100 * pr["right"]["q"]:.1f} %).</p>' + ovn +
             '<p>Earlier work: wft/MULTITRACK_2026-09-14.md measured the same rule on the n_TOF run_145 overlay bench '
             '(beam seeder): swaps among time-coincident pairs 24 → 14 % (A) and 23 → 7 % (C) with pairing on. '
             'No Geant4 / Garfield study of the pairing exists; the response simulation (MX17_Geant) could make one.</p>')
    D.slide('pairing', body, notes, short='x↔y pairing')


# --------------------------------------------------------------------------- #
# head-on tracks
# --------------------------------------------------------------------------- #
def s_failed(D, Sm):
    F = Sm['fit_failed']; Ho = Sm.get('headon') or {}; ex = _ex['failed']
    hf = Ho.get('headon_feats', []); ff = (Ho.get('outside') or {}).get('fake_feats', [])
    pl = sd.Plot(600, 480, x=(1, 1000, 'log'), y=(1, 1000, 'log'), xlabel='brightest strip SNR', ylabel='2nd strip SNR',
                 title='2-strip clusters: head-ons vs added fakes', margin=(10, 24, 84, 100))
    pl.xticks([(1, '1'), (10, '10'), (100, '100'), (1000, '1000')]).yticks([(1, '1'), (10, '10'), (100, '100'), (1000, '1000')])
    if hf:
        pl.points([f['snr1'] for f in hf], [f['snr2'] for f in hf], GREEN, r=8,
                  tips=[f'event {f["event_id"]} x: head-on on M3, SNR {f["snr1"]:.0f} / {f["snr2"]:.0f} / {f["snr3"]:.0f}, '
                        f'{f["n10"]} strips > 10σ' for f in hf])
    if ff:
        pl.points([f['snr1'] for f in ff], [max(f['snr2'], 1.1) for f in ff], RED, r=8, marker='diamond',
                  tips=[f'event {f["event_id"]} {f["plane"]} at {f["p0"]:.0f} mm, M3 outside det3: SNR {f["snr1"]:.1f} / '
                        f'{f["snr2"]:.1f}, extent {f["ext"]} samples' for f in ff])
    pl.hline(20, BLUE, label='head-on rule: 2nd strip ≥ 20σ')
    ho = Ho.get('ho20'); m2 = Ho.get('min2'); out = Ho.get('outside'); hw = Sm.get('headon_wft')
    bt = Ho.get('by_tanx', [])
    tb = sd.table(['|tan θx|', 'rays', 'fit failed'], [(f'{r["lo"]:.2f}–{r["hi"]:.2f}', f'{r["n"]}', pc(100 * r['fit_failed'], 1))
                                                      for r in bt], size=21) if bt else ''
    st = (f'in wft (WFT_HEADON_SIG = 20 + WFT_HEADON_RESCUE): <b>{hw["det_var"]}/{hw["n"]}</b> recovered, '
          f'{hw["controls"]["views_moved"]} of {hw["controls"]["n"]} controls moved, empty chamber '
          f'{hw["outside"]["views_prod"]} → {hw["outside"]["views_new"]} fitted views'
          if hw else (f'<b>{ho["detected"]}/{ho["n"]}</b> recovered; {ho["controls"]["views_moved"]} of {ho["controls"]["n"]} '
                      f'controls moved' if ho else 'head-on re-run pending'))
    fk = (f'empty chamber ({out["n"]} triggers, M3 outside det3): MIN_STRIPS = 2 adds {out["min2"]["added_views"]} fits, '
          f'the head-on rule {out["ho20"]["added_views"]}' if out else '')
    body = (sd.title('Head-on x tracks (the old “fit failed”): a 2-strip rule recovers all of them, with no fakes',
                     f'all {F["n"]} interior rays: y fitted and on M3, x has no candidate · '
                     f'{100 * Ho.get("frac_tanx_lt_003", {}).get("failed", float("nan")):.0f} % have |tan θx| < 0.03 '
                     f'(detected: {100 * Ho.get("frac_tanx_lt_003", {}).get("detected", float("nan")):.0f} %)') +
            sd.row(sd.col(evd(ex, 'x', w=400, h=480, zoom=8, title=f'event {ex}, x', margin=(10, 18, 84, 96)), w=400),
                   sd.col(pl.svg('snr'), sd.legend([('head-on on M3 (fit failed)', GREEN, 'dot'),
                                                    ('fit only MIN_STRIPS = 2 adds, empty chamber', RED, 'box')], 21), w=600),
                   sd.col(tb, sd.callout(st, GREEN, 23), sd.callout(fk, BLUE, 21), w=500), gap=30))
    notes = ((f'<p>The y view is fitted and on M3; x has no candidate. A near-vertical x track puts its whole drift column '
              f'on 1–2 strips, and det3 x shares little charge (kernel c1 = 0.05), so the ±1 neighbours sit at 7–9 % of the '
              f'brightest strip: 15–50σ signals, but under the {term("10 % relative floor", G["floor"])}, which leaves 2 '
              f'strips, one short of MIN_STRIPS = 3. {100 * bt[0]["fit_failed"]:.0f} % of rays at |tan θx| &lt; 0.01 and '
              f'zero above 0.08.</p>' if bt else '') +
             (f'<p><b>Plain MIN_STRIPS = 2</b> recovers {m2["detected"]}/{m2["n"]}, but on {out["n"]} triggers whose M3 track '
              f'misses det3 it adds {out["min2"]["added_views"]} fits (edge noise bands, faint blips).</p>' if m2 and out else '') +
             (f'<p><b>Head-on rule</b> (reco_variant.py --headon-sig 20): MIN_STRIPS stays 3; a 2-strip cluster is admitted '
              f'when its weaker strip is ≥ 20σ. Head-ons have it at {min(f["snr2"] for f in hf):.0f}–'
              f'{max(f["snr2"] for f in hf):.0f}σ, the fakes at ≤ {max(f["snr2"] for f in ff):.0f}σ. '
              f'{ho["detected"]}/{ho["n"]} detected, {ho["controls"]["det"]}/{ho["controls"]["n"]} controls detected '
              f'(production {ho["controls"]["det_base"]}), {ho["controls"]["views_moved"]} control views moved; '
              f'{out["ho20"]["added_views"]} fits added on the empty-chamber set.</p>' if ho and out and hf and ff else ''))
    D.slide('failed', body, notes, short='head-on')


HO_RULES = [('prod', 'production', GREY, 'line'), ('ho20', '+ 2-strip ≥ 20σ', BLUE, 'line'),
            ('s1_7+qa', '+ 1 strip ≥ 7σ, charge-checked', ORANGE, 'line'), ('col_10+qa', '+ column finder, charge-checked', GREEN, 'line')]
HO_SHORT = {'prod': 'production', 'ho20': '2-strip', 's1_7+qa': '1-strip', 'col_10+qa': 'column'}


def headon_plot(H, v, nf, w=560, h=440):
    pl = sd.Plot(w, h, x=(0.04, 1.1, 'log'), y=(0.3, 1.0), xlabel='signal scale k (gain relative to 490 V)',
                 ylabel='head-ons found', title=f'{v} view, noise ×{nf}', margin=(10, 18, 84, 100))
    pl.xticks([(t, f'{t:g}') for t in (0.05, 0.1, 0.2, 0.5, 1)]).yticks([(t, f'{100 * t:.0f} %') for t in (0.4, 0.6, 0.8, 1.0)])
    for r, lab, col, _ in HO_RULES:
        ks = H['ks']; ys = [H['signal'][f'{v}|{nf}|{k:g}'][r] for k in ks]
        fk = [H['fake'][f'{v}|{nf}|{k:g}'][r] for k in ks]
        pl.line(ks, ys, col, w=3, tips=[f'{lab}, k = {k:g}: {100 * y:.1f} % found, fires on {100 * f:.2f} % of empty planes'
                                        for k, y, f in zip(ks, ys, fk)])
    return pl.svg(f'head-on {v} {nf}')


def s_headon_inj(D, Sm):
    H = Sm.get('headon2')
    if not H:
        return
    def fk(v, nf, r):
        return max(H['fake'][f'{v}|{nf}|{k:g}'][r] for k in H['ks'])
    rows = [(HO_SHORT[r], *(f'{100 * H["signal"][f"x|{nf}|{k}"][r]:.0f}' for nf, k in ((1, '1'), (1, '0.1'), (2, '0.1'), (2, '0.05'))),
             f'{100 * max(fk("x", 1, r), fk("x", 2, r)):.2f}') for r, lab, _, _ in HO_RULES]
    h3 = Sm.get('headon3') or {}
    e2e = (f'<b>in wft, end to end</b> (60 head-ons, both views at gain k, the worker fit): x, noise ×2, k = 0.1: '
           f'{100 * h3["x|2|0.1"]["prod"]:.0f} → {100 * h3["x|2|0.1"]["new"]:.0f} %; k = 0.05: {100 * h3["x|2|0.05"]["prod"]:.0f} → '
           f'{100 * h3["x|2|0.05"]["new"]:.0f} %; fakes {sum(round(r.get("fake_new", 0) * r.get("n_fake", 0)) for r in h3.values())}'
           f'/{sum(r.get("n_fake", 0) for r in h3.values())}' if h3 else 'wft end-to-end run pending')
    body = (sd.title('Head-ons down to 5 % gain with doubled noise: the other view’s time and charge make 1-strip seeds safe',
                     'det3 head-on tracks (|tan| < 0.03) injected at reduced gain into real empty-chamber noise; fakes on the '
                     'same noise') +
            sd.row(sd.col(headon_plot(H, 'x', 1), w=560), sd.col(headon_plot(H, 'x', 2), w=560),
                   sd.col(sd.p('x view, % found (fakes: % of empty planes)', 20, MUT),
                          sd.table(['rule', 'k 1', 'k .1', 'k .1 ×2σ', 'k .05 ×2σ', 'fakes'], rows, size=20),
                          sd.legend([(lab, col, st) for _, lab, col, st in HO_RULES], 19),
                          sd.callout(e2e, GREEN, 20), w=560), gap=24))
    c = H['calib']
    notes = (f'<p>headon2.py. Signal: {H["n"]["x"]} x and {H["n"]["y"]} y interior head-on rays (|tan| &lt; 0.03, the other '
             f'view fitted). The donor’s strips within ±4 mm of its M3 footprint are scaled by k and added onto a real '
             f'noise plane of an event whose M3 track misses det3 and in which production finds nothing (x: '
             f'{c["x|1"]["n_clean"]} of {c["x|1"]["n_planes"]}); noise ×2 (n_TOF after 23 July) adds a second noise '
             f'plane × √3 and takes hit significance against 2σ. Hits are emulated from the waveforms (significance = '
             f'1.3 · max / noise, threshold 5: matches mm_processor to r = 0.98–1.00). Found = a seed whose charge '
             f'centroid is within 2 mm of M3. Fakes: the same noise planes with nothing added, with the t0 and charge '
             f'(× k) of a random real track of the other view: the rule fires at all.</p>'
             f'<p>Rules stack on production (they act only when the plane has no seed): <b>2-strip ≥ 20σ</b> (the rule of '
             f'the previous slide); <b>1 strip ≥ 7σ</b>; <b>column finder</b>: per strip, the charge integrated over the '
             f'other view’s drift window [t0 − 120, t0 + 1020] ns against a baseline from outside it, Z ≥ 10, starting '
             f'inside the window and lasting ≥ 6 samples. <b>Charge-checked</b>: the rescued cluster’s window charge must '
             f'match the other view’s fitted charge, |ln(Q/(R·q_other))| ≤ 3σ, with R and σ from detected head-ons '
             f'(x: R = {c["x|1"]["r_q"]:.2f}, σ = {c["x|1"]["sigma_ln"]:.3f}) — the same x/y charge equality that pairs '
             f'two tracks. Without it the 1-strip rule fires on 1.9–2.8 % of empty planes; with it ≤ 0.03 %.</p>'
             + (lambda h3: (f'<p><b>End to end in wft</b> (headon3.py: both views injected at gain k, hits emulated, wft '
                            f'seeding with WFT_HEADON_SIG = 20, the worker fit with WFT_HEADON_RESCUE): ' +
                            '; '.join(f'{k}: production {100 * r["prod"]:.0f} % → {100 * r["new"]:.0f} % found, fakes '
                                      f'{100 * r.get("fake_prod", 0):.1f} → {100 * r.get("fake_new", 0):.1f} %'
                                      for k, r in h3.items()) + '.</p>') if h3 else '')(Sm.get('headon3')) +
             f'<p>y needs none of this at 490 V (its strips share more charge); at k ≤ 0.1 with doubled noise the same '
             f'rules recover y too (headon2.json). Not covered: tracks that are head-on in BOTH views (no other view to '
             f'lean on) and a busy plane (another particle), which is the multi-particle problem.</p>')
    D.slide('headon-inj', body, notes, short='head-on: extreme')


def s_rest(D, Sm):
    b = Sm['budget']['interior']; fb = b['far_by_cls']
    cards = [
        sd.card(sd.p(f'<b>Position tail: {fb.get("POS_TAIL", 0):.2f} %</b>', 26) +
                sd.p('Right angle, one clean cluster, > 3 mm from M3. About ⅔ are muons that scattered between the M3 '
                     'doublets (a kink a 4-layer χ² < 1 cannot see). The reference is wrong; nothing to recover.', 22, MUT), w=520),
        sd.card(sd.p(f'<b>Edge: {fb.get("EDGE", 0):.2f} %</b>', 26) +
                sd.p('Interior by the budget’s rule but near the edge of the OTHER coordinate (< 8 mm) or the field '
                     'distortion zone: a truncated or bent column. Geometry, not reconstruction.', 22, MUT), w=520)]
    body = (sd.title('The rest is the reference’s own error or the chamber edge', 'the remaining det3 interior categories') +
            sd.row(*cards, gap=40))
    D.slide('rest', body, '<p>C22 report (notes/mx17-outliers-vs-m3) has the evidence for the bent-track fraction '
                          '(kinkvtx.py: 0.65 ± 0.07 of the position tail) and the edge response.</p>', short='the rest')


def s_close(D, Sm):
    rc = Sm['recat']['pct']; rec = recoverable(Sm); bf = Sm.get('band_fix') or {}
    n = Sm['budget']['interior']['n']
    mt = Sm.get('multitrack') or {}; M = (mt.get('classes') or {}).get('M', {})
    rows = [('noise band', pc(rc.get('band', 0), 2), 'drop whole-window hits before seeding' +
             (f' ({bf["det_var"] - bf["det_prod"]} of {bf["n"]} recovered)' if bf else '')),
            ('second particle (3 sub-classes)', pc(sum(v for k, v in rc.items() if k.startswith('multi:')), 2),
             f'multi-track credit: M3 track found in {100 * M.get("any", float("nan")):.0f} %; quote apart'),
            ('spark veto (no discharge)', pc(rc.get('spark:muon+other', 0) + rc.get('spark:shower', 0), 2),
             'a discharge veto instead of a hit count; showers stay lost'),
            ('head-on x tracks', pc(rc.get('head_on', 0), 2), '2-strip ≥ 20σ now; 1-strip + column, charge-checked, for n_TOF'),
            ('saturated blobs', pc(rc.get('far:FIT_SAT', 0), 2), 'a track + blob model; censoring buys ≤ 2 % of these'),
            ('δ-rays, discharges, M3 kinks, edge', pc(sum(rc.get(k, 0) for k in ('far:DELTA', 'spark:discharge',
                                                                                 'far:POS_TAIL', 'far:EDGE')), 2), 'not recoverable')]
    body = (sd.kicker('summary') + sd.title('What a better reconstruction could buy, and what it cannot') +
            sd.table(['category', '% of interior', 'what it takes'], rows, size=25) +
            sd.p(f'<b>For the paper:</b> single-particle efficiency first, then the multi-particle class with any-track credit; '
                 f'x↔y pairing by charge as its own result. <b>Not ruled out:</b> the classes are heuristic (checked by eye '
                 f'on random examples); one chamber, one run (det3, 6-27, 490 V); the band mechanism is not identified.',
                 23, DMUT))
    D.slide('close', body, '', dark=True, short='summary')


def rec_part(Sm, k):
    n = Sm['budget']['interior']['n']
    ff = (Sm.get('headon') or {}).get('ho20') or Sm['fit_failed'].get('min2')
    return 100 * ff['detected'] / n if ff else float('nan')


# --------------------------------------------------------------------------- #
_ex = {}


def pick_examples(Sm):
    E = pickle.load(open(os.path.join(OUT, 'wf_examples.pkl'), 'rb'))['picks']
    _ex['mip'] = E['detected'][0]
    T = pd.read_parquet(os.path.join(OUT, 'sparks_det3.parquet'))
    _ex['discharge'] = 18645 if 18645 in set(T.event_id) else T[T.cls == 'discharge'].event_id.iloc[0]
    _ex['muon+other'] = 36694 if 36694 in set(T.event_id) else T[T.cls == 'muon+other'].event_id.iloc[0]
    _ex['shower'] = 41538 if 41538 in set(T.event_id) else T[T.cls == 'shower'].event_id.iloc[0]
    _ex['multi'] = 28078
    _ex['delta'] = 28537
    _ex['satblob'] = [(41373, 'x'), (46277, 'y'), (44721, 'y')]
    _ex['fit'] = [(4490, 'x'), (19676, 'y'), (3072, 'x')]
    _ex['failed'] = 43536
    _ex['merged'] = (46350, 'x')
    W = wf(330)['x'].astype(np.float32)
    _ex['satstrip'] = (330, 'x', int(np.argmax((W.max(1) >= S_SAT) * (W.min(1) < -200) * W.max(1)) if ((W.max(1) >= S_SAT) & (W.min(1) < -200)).any() else np.argmax(W.max(1))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(OUT, 'deck', 'det3-loss-categories.html'))
    a = ap.parse_args()
    Sm = json.load(open(os.path.join(OUT, 'summary.json')))
    row(int(pickle.load(open(os.path.join(OUT, 'wf_examples.pkl'), 'rb'))['picks']['detected'][0]))
    pick_examples(Sm)
    D = sd.Deck('det3 loss categories', 'What each det3 efficiency-loss category looks like, and whether a better '
                                         'reconstruction could recover it')
    for f in (s_cover, s_setup, s_budget, s_bands_what, s_bands_not, s_bands_fix, s_multi, s_pairing, s_failed,
              s_headon_inj, s_spark, s_delta, s_sat_now, s_sat_rail, s_sat_blob, s_sat_arms, s_fit, s_rest, s_close):
        f(D, Sm)
    import datetime as dt
    D.write(a.out, note_meta=dict(title='det3: where the lost tracks go, and what a better reconstruction could recover',
                                  summary='Each efficiency-loss category of det3 in the waveforms: sparks, second '
                                          'particles, δ-rays, saturated blobs, 2-strip seeds; the saturated-strip '
                                          'treatment in detail.',
                                  tags='X17, cosmic bench, micromegas, efficiency, reconstruction', date=dt.date.today().isoformat()))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
