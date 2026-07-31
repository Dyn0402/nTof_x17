"""37_srccal_slide_figs.py — wide, slide-ready figures for the two-source
plastic-calibration deck.

The per-run diagnostic grids from 34 and the 4-panel summary from 35 are working
figures: 22 panels of a 16:9 frame is unreadable from the back of a room. These
are 3-panel ~3:1 figures, one message each, sized to fill a beamer frame.

Outputs (figures/33_srccal/slides_*.png):
  slides_spectra     what the two sources look like, and the background subtraction
  slides_response    the calibration itself: 3 points, the line, the nonlinearity
  slides_detectors   what each detector type gives (plastic / liquid / wall)
  slides_controls    repeatability, the L/R map, the far-source control, pileup
Usage: python 37_srccal_slide_figs.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import edgefit
import srccal_runs as S

BASE = Path(__file__).parent
CACHE = BASE / 'cache'
CALIB = BASE / 'calib'
OUT = BASE / 'figures' / '33_srccal'
OUT.mkdir(parents=True, exist_ok=True)
BARS = [a + s for a in S.ARMS for s in 'LR']
WIDE = (16.5, 5.0)
SMOOTH = 2.0        # mV, display smoothing only


def cache(run):
    return np.load(CACHE / f'33_srccal_{run}.npz')


def sub_spectrum(run, tree, c, arm, key='lin_np'):
    """Background-subtracted, per-trigger spectrum of one channel."""
    z = cache(run)
    darks = S.dark_runs_for(arm)
    nt = int(z['n_triggers'])
    bkg = sum(cache(r)[f'{tree}_{key}'][c] for r in darks)
    nb = sum(int(cache(r)['n_triggers']) for r in darks)
    cen = 0.5 * (z['lin_edges'][:-1] + z['lin_edges'][1:])
    raw = z[f'{tree}_{key}'][c] / nt
    bg = (nt / nb) * bkg / nt
    return cen, raw, bg, raw - bg


def run_with(bar, src):
    for r in S.RUNS:
        if S.sources_in(r).get(src) == bar:
            return r
    return None


def edges_of(run, ch):
    d = json.loads((CALIB / f'srccal_edges_{run}.json').read_text())
    return {e['kevee']: e for e in d['channels'].get(ch, {}).get('edges', [])}


def sm(y):
    return edgefit.smooth(y, edgefit.kernel(SMOOTH / 0.25))


# ---------------------------------------------------------------- spectra ---
def fig_spectra():
    fig, ax = plt.subplots(1, 3, figsize=WIDE)
    # CL is the bar where all three edges come out cleanly — the honest
    # best case, and the only bar whose 1612 keVee point lands on the line.
    bar = 'CL'
    arm, c = bar[0], S.assumed_detn[bar[1]] - 1
    tree = f'PSS{arm}'

    for k, (src, col, lab, xhi) in enumerate((
            ('Cs137', 'seagreen', 'Cs-137, 662 keV $\\gamma$', 90),
            ('Y88', 'crimson', 'Y-88, 898 + 1836 keV $\\gamma$ (cascade)', 190))):
        run = run_with(bar, src)
        cen, raw, bg, s = sub_spectrum(run, tree, c, arm)
        a = ax[k]
        # LINEAR y: a Compton edge is a step in dN/dA and simply does not read
        # as one on a log axis. The turn-on below the valley is 100x the
        # continuum, so the y range is set from the continuum itself.
        ss = sm(s)
        valley = edgefit.valley_of(cen, ss)
        top = 1.35 * float(ss[np.argmin(abs(cen - valley))])
        a.step(cen, sm(raw), where='mid', color='0.55', lw=1.0,
               label='raw (source run)')
        a.step(cen, sm(bg), where='mid', color='0.8', lw=1.0,
               label='background (dark runs)')
        a.step(cen, ss, where='mid', color=col, lw=1.8, label='subtracted')
        for j, (E, e) in enumerate(sorted(edges_of(run, S.bar_key(bar)).items())):
            a.axvline(e['edge_mv'], color='k', lw=1.3, ls='--')
            a.annotate(f"{E:.0f} keVee\n{e['edge_mv']:.1f} mV",
                       (e['edge_mv'], top * (0.88 if j == 0 else 0.55)),
                       fontsize=8.5, ha='left', va='top', xytext=(5, 0),
                       textcoords='offset points')
        a.set_xlim(0, xhi)
        a.set_ylim(0, top)
        a.set_xlabel('pulse amplitude [mV]')
        a.set_ylabel('hits / trigger / 0.25 mV')
        a.set_title(f'({"ab"[k]}) {lab}\nbar {bar}, run {run[-6:]}', fontsize=10)
        a.legend(fontsize=7.5, loc='upper right')
        a.grid(alpha=0.25)

    a = ax[2]
    for bar in BARS:
        run = run_with(bar, 'Y88')
        arm, c = bar[0], S.assumed_detn[bar[1]] - 1
        cen, _, _, s = sub_spectrum(run, f'PSS{arm}', c, arm)
        a.step(cen, sm(s), where='mid', lw=1.2, label=bar)
        e = edges_of(run, S.bar_key(bar)).get(S.E_Y1)
        if e:
            a.plot([e['edge_mv']], [sm(s)[np.argmin(abs(cen - e['edge_mv']))]],
                   'k.', ms=7)
    a.set_yscale('log'); a.set_xlim(0, 150); a.set_ylim(1e-3, 30)
    a.set_xlabel('pulse amplitude [mV]')
    a.set_ylabel('hits / trigger / 0.25 mV')
    a.set_title('(c) All 8 bars, Y-88 — dots = fitted 699 keVee edge\n'
                'the same energy lands 29-65 mV apart', fontsize=10)
    a.legend(fontsize=7, ncol=2); a.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / 'slides_spectra.png', dpi=150)
    plt.close(fig)
    print('-> slides_spectra.png')


# --------------------------------------------------------------- response ---
def fig_response():
    cal = json.loads((CALIB / 'srccal_energy_calib.json').read_text())['channels']
    fig, ax = plt.subplots(1, 3, figsize=WIDE)

    a = ax[0]
    for bar in BARS:
        r = cal[S.bar_key(bar)]
        E = np.array([float(k) for k in r['points']])
        y = np.array([v['mv'] for v in r['points'].values()])
        dy = np.array([v['err'] for v in r['points'].values()])
        o = np.argsort(E)
        line, = a.plot(E[o], y[o], 'o-', ms=5, lw=1.2, label=bar)
        a.errorbar(E[o], y[o], yerr=dy[o], ls='none', color=line.get_color())
    for E, lab in ((S.E_CS, 'Cs 477'), (S.E_Y1, 'Y 699'), (S.E_Y2, 'Y 1612')):
        a.axvline(E, color='0.85', lw=0.8, zorder=0)
        a.text(E, 3, lab, fontsize=7, rotation=90, va='bottom')
    a.set_xlabel('Compton-edge energy [keVee]')
    a.set_ylabel('edge amplitude [mV]')
    a.set_title('(a) Measured response, 3 points per bar', fontsize=10)
    a.legend(fontsize=7, ncol=2); a.grid(alpha=0.25)

    # (b) nonlinearity: normalise each bar by a straight line through its two
    # CLEAN points (477 and 699), then ask where 1612 lands.
    a = ax[1]
    for bar in BARS:
        r = cal[S.bar_key(bar)]
        p = {float(k): v['mv'] for k, v in r['points'].items()}
        if S.E_CS not in p or S.E_Y1 not in p:
            continue
        slope = (p[S.E_Y1] - p[S.E_CS]) / (S.E_Y1 - S.E_CS)
        icept = p[S.E_CS] - slope * S.E_CS
        E = np.array(sorted(p))
        meas = np.array([p[e] for e in E])
        a.plot(E, meas / (slope * E + icept), 'o-', ms=5, lw=1.2, label=bar)
    a.axhline(1.0, color='k', lw=1.2)
    a.axhspan(0.95, 1.05, color='0.9', zorder=0)
    a.set_xlabel('Compton-edge energy [keVee]')
    a.set_ylabel('measured / (line through 477 & 699)')
    a.set_title('(b) Anchored on the two clean points, the 1612 keVee edge\n'
                'lands -22 % to +1 % — bar-dependent, so not a usable point',
                fontsize=10)
    a.legend(fontsize=7, ncol=2); a.grid(alpha=0.25)

    a = ax[2]
    hv = np.array([S.PLASTIC_HV_V[b] for b in BARS])
    g = np.array([cal[S.bar_key(b)]['mv_per_mevee_origin'] for b in BARS])
    a.scatter(hv, g, s=45, c='crimson', zorder=3)
    for b, x, y in zip(BARS, hv, g):
        a.annotate(b, (x, y), fontsize=8, xytext=(5, 3),
                   textcoords='offset points')
    a.set_xlabel('standing plastic PMT bias [V]  (unchanged since 2026-07-19)')
    a.set_ylabel('mV per MeVee')
    a.set_title('(c) Absolute scale vs HV — 2.3$\\times$ spread,\n'
                'and it does not follow the bias', fontsize=10)
    a.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / 'slides_response.png', dpi=150)
    plt.close(fig)
    print('-> slides_response.png')


# -------------------------------------------------------------- detectors ---
def fig_detectors():
    fig, ax = plt.subplots(1, 3, figsize=WIDE)
    bar, run = 'AR', run_with('AR', 'Y88')
    arm = 'A'

    a = ax[0]
    for c, lab, col in ((1, 'AR — source clamped on this bar', 'crimson'),
                        (0, 'AL — the other bar of the same arm', '0.45')):
        cen, _, _, s = sub_spectrum(run, 'PSSA', c, arm)
        a.step(cen, sm(s), where='mid', color=col, lw=1.5, label=lab)
    a.set_yscale('log'); a.set_xlim(0, 130); a.set_ylim(1e-3, 30)
    a.set_xlabel('amplitude [mV]'); a.set_ylabel('hits / trigger / 0.25 mV')
    a.set_title('(a) Plastic: both bars see it, the lit one 5$\\times$ more\n'
                '(light sharing, not a calibration point)', fontsize=10)
    a.legend(fontsize=7.5); a.grid(alpha=0.25)

    a = ax[1]
    cen, _, _, sY = sub_spectrum(run, 'LIQA', 0, 'A')
    runC = run_with('CR', 'Cs137')
    _, _, _, sC = sub_spectrum(runC, 'LIQC', 0, 'C')
    a.step(cen, sm(sY), where='mid', color='darkgreen', lw=1.6,
           label='LIQ A — Y-88 on this arm')
    a.step(cen, sm(sC), where='mid', color='0.6', lw=1.2,
           label='LIQ C — Cs-137 on this arm')
    e = edges_of(run, 'LIQA1').get(S.E_Y1)
    if e:
        a.axvline(e['edge_mv'], color='k', lw=1.2, ls='--')
        a.annotate(f"699 keVee\n{e['edge_mv']:.1f} mV", (e['edge_mv'], 2),
                   fontsize=8, xytext=(5, 0), textcoords='offset points')
    a.set_yscale('log'); a.set_xlim(0, 130); a.set_ylim(1e-3, 30)
    a.set_xlabel('amplitude [mV]'); a.set_ylabel('hits / trigger / 0.25 mV')
    a.set_title('(b) Liquid: a clean bump from Y-88,\nnothing from Cs-137 at 662 keV',
                fontsize=10)
    a.legend(fontsize=7.5); a.grid(alpha=0.25)

    a = ax[2]
    cen, raw, bg, s = sub_spectrum(run, 'WALA', 5, 'A')
    a.step(cen, sm(raw), where='mid', color='steelblue', lw=1.4,
           label='WAL A6, source run (raw)')
    a.step(cen, sm(bg), where='mid', color='0.75', lw=1.2,
           label='WAL A6, dark runs')
    a.step(cen, sm(s), where='mid', color='crimson', lw=1.4, label='subtracted')
    a.annotate('cosmic MIP bump — present in\nthe dark runs too, so it cancels',
               xy=(93, 3.5e-2), xytext=(40, 0.6), fontsize=8, ha='center',
               arrowprops=dict(arrowstyle='->', lw=1.0))
    a.set_yscale('log'); a.set_xlim(0, 130); a.set_ylim(1e-4, 5)
    a.set_xlabel('amplitude [mV]'); a.set_ylabel('hits / trigger / 0.25 mV')
    a.set_title('(c) SiPM wall: no source edge to fit\n'
                '(wall is in FRONT, source is on the plastic)', fontsize=10)
    a.legend(fontsize=7.5); a.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / 'slides_detectors.png', dpi=150)
    plt.close(fig)
    print('-> slides_detectors.png')


# --------------------------------------------------------------- controls ---
def fig_controls():
    cal = json.loads((CALIB / 'srccal_energy_calib.json').read_text())
    ctl = cal['controls']
    fig, ax = plt.subplots(1, 3, figsize=WIDE)

    # (a) the Cs-removed control: same bar, same Y source, Cs gone
    a = ax[0]
    cen, _, _, s588 = sub_spectrum('run224588', 'PSSA', 1, 'A')
    _, _, _, s596 = sub_spectrum('run224596', 'PSSA', 1, 'A')
    a.step(cen, sm(s588), where='mid', color='crimson', lw=1.5,
           label='224588: Y on AR, Cs on CR')
    a.step(cen, sm(s596), where='mid', color='navy', lw=1.5, ls='--',
           label='224596: Y on AR, Cs REMOVED')
    a.set_yscale('log'); a.set_xlim(0, 130); a.set_ylim(1e-3, 30)
    a.set_xlabel('amplitude [mV]'); a.set_ylabel('hits / trigger / 0.25 mV')
    lk = ctl['far_source_leakage']
    a.set_title(f"(a) Cs removed: shape unchanged to "
                f"{lk['shape_ratio_spread_pct']} %\n"
                f"only the rate moves ({lk['rate_ratio_588_over_596']:.2f}, "
                f"source re-placed)", fontsize=10)
    a.legend(fontsize=7.5); a.grid(alpha=0.25)

    # (b) detn <-> L/R: rate contrast, lit bar vs its partner
    a = ax[1]
    mp = ctl['detn_LR_map']['per_run']
    keys = sorted(mp, key=lambda k: k.split(':')[2])
    cs = [mp[k]['contrast'] for k in keys]
    labs = [k.split(':')[2] + '\n' + k.split(':')[1].replace('137', '') for k in keys]
    col = ['seagreen' if mp[k]['agrees'] else 'crimson' for k in keys]
    a.bar(range(len(keys)), cs, color=col)
    a.axhline(1, color='k', lw=1)
    a.set_xticks(range(len(keys)))
    a.set_xticklabels(labs, fontsize=6.5)
    a.set_yscale('log')
    a.set_ylabel('rate(lit bar) / rate(partner bar)')
    a.set_title(f"(b) One bar lit at a time names the map:\n"
                f"detn 1 = left, 2 = right "
                f"{ctl['detn_LR_map']['verdict']} "
                f"({ctl['detn_LR_map']['n_agree']}/"
                f"{ctl['detn_LR_map']['n_total']})", fontsize=10)
    a.grid(alpha=0.25, axis='y')

    # (c) pileup systematic on the edge position
    a = ax[2]
    shifts = {699: [], 1612: []}
    for run in S.RUNS:
        d = json.loads((CALIB / f'srccal_edges_{run}.json').read_text())
        for src, bar in S.sources_in(run).items():
            v = d['channels'].get(S.bar_key(bar), {})
            if v.get('role') != 'source_bar':
                continue
            for e in v.get('edges', []):
                if e.get('pileup_shift_mv') is None:
                    continue
                k = 699 if e['kevee'] == S.E_Y1 else (
                    1612 if e['kevee'] == S.E_Y2 else 477)
                if k in shifts:
                    shifts[k].append(100 * e['pileup_shift_mv'] / e['edge_mv'])
    a.hist([shifts[699], shifts[1612]], bins=np.arange(-10, 30, 2.5),
           color=['darkorange', 'crimson'], label=['699 keVee', '1612 keVee'])
    a.axvline(0, color='k', lw=1.2)
    a.set_xlabel('edge shift if pileup hits are KEPT [%]')
    a.set_ylabel('bars')
    a.set_title('(c) Pileup is a real systematic\n'
                '(30-36 % of hits on a lit bar carry the flag)', fontsize=10)
    a.legend(fontsize=7.5); a.grid(alpha=0.25, axis='y')

    fig.tight_layout()
    fig.savefig(OUT / 'slides_controls.png', dpi=150)
    plt.close(fig)
    print('-> slides_controls.png')


# ---------------------------------------------------------------- liquids ---
def fig_liquids():
    """Which of the four liquid cells calibrate, and why the others do not."""
    fig, ax = plt.subplots(1, 3, figsize=WIDE)

    # (a) each cell in its BEST Y-lit run
    best = {'A': 'run224596', 'B': 'run224590', 'C': 'run224592', 'D': 'run224594'}
    a = ax[0]
    for k, (arm, col) in enumerate(zip('ABCD',
                                       ('crimson', 'steelblue', '0.55', 'seagreen'))):
        run = best[arm]
        cen, _, _, s = sub_spectrum(run, f'LIQ{arm}', 0, arm)
        e = edges_of(run, f'LIQ{arm}1').get(S.E_Y1)
        lab = (f'LIQ{arm}  ({S.sources_in(run)["Y88"]})' if e
               else f'LIQ{arm}  ({S.sources_in(run)["Y88"]}) -- no edge')
        a.step(cen, sm(s), where='mid', color=col, lw=1.6, label=lab)
        if e:
            a.axvline(e['edge_mv'], color=col, lw=1.0, ls='--')
            a.annotate(f"{e['edge_mv']:.1f} mV", (e['edge_mv'], 4.5 / 2.4 ** k),
                       color=col, fontsize=8, xytext=(3, 0),
                       textcoords='offset points')
    a.set_yscale('log'); a.set_xlim(0, 70); a.set_ylim(1e-3, 8)
    a.set_xlabel('amplitude [mV]'); a.set_ylabel('hits / trigger / 0.25 mV')
    a.set_title('(a) All four cells, best Y-88 run\n'
                'dashed = fitted 699 keVee edge', fontsize=10)
    a.legend(fontsize=7.5); a.grid(alpha=0.25)

    # (b) it depends which BAR the source sits on -- and it does so for BOTH
    # sources, which is what makes it a property of the setup rather than of
    # one source's placement
    a = ax[1]
    x, w = np.arange(4), 0.2
    for k, (side, hatch) in enumerate((('L', ''), ('R', '//'))):
        for j, src in enumerate(('Y88', 'Cs137')):
            vals = []
            for arm in 'ABCD':
                dz = [cache(r) for r in S.dark_runs_for(arm)]
                drate = sum(float(z[f'LIQ{arm}_nhit'][0]) / int(z['n_triggers'])
                            for z in dz) / len(dz)
                runs = [r for r in S.RUNS
                        if S.sources_in(r).get(src) == arm + side]
                if not runs:
                    vals.append(0.0); continue
                r = float(np.mean([float(cache(q)[f'LIQ{arm}_nhit'][0])
                                   / int(cache(q)['n_triggers']) for q in runs]))
                vals.append(r / max(drate, 1e-9))
            a.bar(x + (2 * k + j - 1.5) * w, vals, w, hatch=hatch,
                  color='crimson' if src == 'Y88' else 'seagreen',
                  edgecolor='k', lw=0.4,
                  label=f'{"Y-88" if src == "Y88" else "Cs-137"} on the {side} bar')
    a.axhline(1, color='k', lw=1.2)
    a.set_xticks(x); a.set_xticklabels([f'LIQ{a_}' for a_ in 'ABCD'])
    a.set_ylabel('rate with source / rate in dark runs')
    a.set_title('(b) A and D answer to the R bar only --- for BOTH sources.\n'
                'B answers to both, C to neither. Not a distance effect.',
                fontsize=10)
    a.legend(fontsize=6.5); a.grid(alpha=0.25, axis='y')

    # (c) why LIQD needed the step model
    a = ax[2]
    for run, arm, col, lab in (('run224596', 'A', 'crimson', 'LIQA (bump forms)'),
                               ('run224594', 'D', 'seagreen', 'LIQD (no local max)')):
        cen, _, _, s = sub_spectrum(run, f'LIQ{arm}', 0, arm)
        ss = sm(s)
        norm = ss[np.argmin(abs(cen - 22))]
        a.step(cen, ss / norm, where='mid', color=col, lw=1.7, label=lab)
        e = edges_of(run, f'LIQ{arm}1').get(S.E_Y1)
        eb = {x['kevee']: x for x in
              (json.loads((CALIB / f'srccal_edges_{run}.json').read_text())
               ['channels'][f'LIQ{arm}1'].get('edges_bump_convention') or [])}
        if e:
            a.axvline(e['edge_mv'], color=col, lw=1.3, ls='--')
        if S.E_Y1 in eb:
            a.axvline(eb[S.E_Y1]['edge_mv'], color=col, lw=1.1, ls=':')
    a.set_xlim(5, 60); a.set_ylim(0, 1.6)
    a.set_xlabel('amplitude [mV]')
    a.set_ylabel('subtracted spectrum (scaled at 22 mV)')
    a.set_title('(c) Dashed = step centre (used), dotted = bump peak (22)\n'
                'the bump sits ~25 % low, and LIQD never forms one', fontsize=10)
    a.legend(fontsize=7.5); a.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / 'slides_liquids.png', dpi=150)
    plt.close(fig)
    print('-> slides_liquids.png')


if __name__ == '__main__':
    fig_spectra()
    fig_response()
    fig_detectors()
    fig_controls()
    fig_liquids()
