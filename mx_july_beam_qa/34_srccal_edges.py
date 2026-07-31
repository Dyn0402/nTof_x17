"""34_srccal_edges.py — Compton-edge extraction for the 2026-07-28 two-source
campaign (T2). Local script: it reads only the ~1 MB caches from 33.

For every run, every channel that a source illuminates is fitted:

  * the SOURCE BAR (PSS, the bar the source was centred on) — the primary
    measurement, one edge for Cs-137 (477 keVee), two for Y-88 (699 + 1612);
  * the OTHER BAR of the same arm — the source is centred on one bar now, so
    this bar's spectrum is the light-sharing / scattering measurement, reported
    but flagged `secondary`;
  * the 8 WAL channels and the LIQ vessel of each lit arm — same source, other
    detectors, bump model.

Background: for a channel on arm X, the template is the SUM of that same
channel's spectrum over the runs in which arm X carried no source at all
(4-5 runs each, `srccal_runs.dark_runs_for`), rate-normalised by trigger count.
This is a genuine same-day, same-DAQ-state, same-channel background — an
improvement on the 07-17 analysis, which could only use a different arm's
channel as a shape reference.

Pileup: the primary fit uses the pileup-vetoed histogram (`_lin_np`); the
all-hits histogram is fitted too, without bootstrap, and the difference is
recorded per edge as `pileup_shift_mv` — at these rates that shift is a real
systematic and is not assumed away.

Outputs:
  calib/srccal_edges_<run>.json
  figures/33_srccal/edges_<run>.png
Usage:
  python 34_srccal_edges.py [run_stem ...]      (default: all nine)
"""

import json
import sys
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
CALIB.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# Poisson-bootstrap counts. The statistical error these produce is 0.1-0.8 mV
# (0.4-2.6 %), far below the ~5 % convention systematic between the step-centre
# and half-height estimators, so there is nothing to buy above ~100 resamples —
# and each one is a pair of curve_fits, which is what the runtime is made of.
N_BOOT = {'PSS': 120, 'WAL': 40, 'LIQ': 40}


def load(run):
    p = CACHE / f'33_srccal_{run}.npz'
    if not p.exists():
        raise SystemExit(f'missing {p} — run 33_srccal_spectra.py first '
                         '(see lxplus/srccal.sub)')
    return np.load(p, allow_pickle=False)


def background(caches, arm, tree, c, key):
    """Summed dark-run spectrum of channel `c` of `tree`, plus the total trigger
    count behind it (the live-time normalisation)."""
    darks = S.dark_runs_for(arm)
    tot, ntrig = None, 0
    for r in darks:
        z = caches.get(r)
        if z is None or f'{tree}_{key}' not in z:
            continue
        h = z[f'{tree}_{key}'][c]
        tot = h if tot is None else tot + h
        ntrig += int(z['n_triggers'])
    return tot, ntrig, darks


def process_run(run, caches, only_source, priors, res=None):
    """Fit the channels illuminated by `only_source` in `run`.

    Called twice per run (Cs-137, then Y-88): the Cs-137 pass fills `priors`
    with each channel's 477 keVee edge, and the Y-88 pass consumes it, scaled by
    the energy ratio, as the position to fit its 699 keVee edge around. A clean
    single-gamma edge is a much better anchor than any feature search on a Y-88
    spectrum — see `edgefit.seed_candidates` for the three search rules this
    replaced and how each one failed.
    """
    z = caches[run]
    cen = 0.5 * (z['lin_edges'][:-1] + z['lin_edges'][1:])
    ntrig = int(z['n_triggers'])
    srcs = S.sources_in(run)

    res = res or {'run': run, 'n_triggers': ntrig, 'sources': srcs,
           'convention': (
               'edge_mv = fitted resolution-smeared step centre (PSS) or bump '
               'centre (WAL/LIQ), the same convention as 22_y88_edges.py so the '
               '07-17 and 07-28 campaigns compare directly. '
               'edge_mv_halfheight is the model-independent cross-check. '
               'The step model carries a sloped continuum under the edge: the '
               '1836 keV Compton continuum runs through the 699 keVee edge, and '
               "Y-88's two gammas are a CASCADE, so a source on the bar produces "
               'true-coincidence summing that adds counts above every '
               'single-gamma edge — which is why the 1612 keVee edge is marked '
               'secondary and the clean single-gamma Cs-137 edge is the better '
               'low-energy anchor. '
               'Background = same channel summed over the runs with no source '
               'on that arm, scaled by trigger count. Primary fit is on the '
               'pileup-vetoed spectrum; pileup_shift_mv = (all hits) - (veto). '
               f'Bootstrap {N_BOOT} Poisson resamples of signal AND background, '
               f'seed {edgefit.SEED}; error floored at half a bin (0.1 mV).'),
           'channels': {}}

    panels = []
    for src, bar in srcs.items():
        if src != only_source:
            continue
        arm = bar[0]
        lit_ch = S.bar_channel(bar)[1]
        for kind in ('PSS', 'LIQ', 'WAL'):
            tree = f'{kind}{arm}'
            if f'{tree}_lin' not in z:
                continue
            for c in range(S.NCH[kind]):
                role = ('source_bar' if (kind == 'PSS' and c + 1 == lit_ch)
                        else 'same_arm')
                panels.append((src, arm, kind, tree, c, role))

    for src, arm, kind, tree, c, role in panels:
        ch = f'{tree}{c + 1}'
        energies = S.EDGES_OF[src]
        sig_np = z[f'{tree}_lin_np'][c]
        sig_all = z[f'{tree}_lin'][c]
        bkg_np, nb_np, darks = background(caches, arm, tree, c, 'lin_np')
        bkg_all, nb_all, _ = background(caches, arm, tree, c, 'lin')
        scale = ntrig / nb_np if nb_np else 0.0

        # Y-88 fits start from this channel's own Cs-137 edge, scaled to
        # 699 keVee. Plastics only: the liquids have no usable Cs-137 spectrum
        # (the vessel sits behind the plastic and sees almost nothing at
        # 662 keV), so they keep the autonomous bump search.
        prior = None
        if kind == 'PSS' and ch in priors:
            src_p, mv_p = priors[ch]
            if src_p != src:                    # anchor from the other source
                prior = (mv_p * S.E_Y1 / S.E_CS if src == 'Y88'
                         else mv_p * S.E_CS / S.E_Y1)

        # Liquids are fitted with the STEP model, like the plastics. A Compton
        # edge is a step in dN/dA; the localised "bump" 22 fitted on these
        # channels is that step riding on the threshold turn-on, and its peak
        # sits ~25 % BELOW the half-height point. Two consequences, both
        # verified here: the step convention is consistent with the plastic
        # numbers, and it recovers LIQD, whose edge never forms a local maximum
        # at all so the bump model simply cannot see it. The bump fit is kept
        # alongside, because the 07-17 LIQ numbers are in that convention and
        # the transport comparison has to be like-for-like.
        model = 'PSS' if kind == 'LIQ' else kind
        r = edgefit.extract(cen, sig_np, bkg_np, scale, model, energies,
                            n_boot=N_BOOT[kind], prior=prior)
        r_all = edgefit.extract(cen, sig_all, bkg_all,
                                ntrig / nb_all if nb_all else 0.0,
                                model, energies, n_boot=0, prior=prior)
        r_bump = (edgefit.extract(cen, sig_np, bkg_np, scale, 'LIQ', energies,
                                  n_boot=N_BOOT[kind])
                  if kind == 'LIQ' else None)
        shift = {e['kevee']: e['edge_mv'] for e in r_all['edges']}
        for e in r['edges']:
            d = shift.get(e['kevee'])
            e['pileup_shift_mv'] = (round(d - e['edge_mv'], 2)
                                    if d is not None else None)

        nhit = int(z[f'{tree}_nhit'][c])
        res['channels'][ch] = dict(
            kind=kind, arm=arm, source=src, role=role,
            edges=r['edges'], valley_mv=r['valley'],
            n_excess=r['n_excess'], excess_over_bkg=r['excess_over_bkg'],
            hits_per_trigger=round(nhit / max(ntrig, 1), 1),
            n_hits=nhit,
            sat_frac=round(float(z[f'{tree}_nsat'][c]) / max(nhit, 1), 5),
            satuflag_frac=round(float(z[f'{tree}_nsatuflag'][c]) / max(nhit, 1), 5),
            pileup_frac=round(float(z[f'{tree}_npileup'][c]) / max(nhit, 1), 4),
            bkg_runs=darks, bkg_scale=round(scale, 4),
            prior_mv=round(prior, 2) if prior else None,
            model='step' if model == 'PSS' else 'bump',
            edges_bump_convention=(r_bump['edges'] if r_bump else None))
        res['channels'][ch]['_plot'] = (r, sig_np)

        # a source_bar edge becomes this channel's anchor for the other source
        if role == 'source_bar' and r['edges'] and ch not in priors:
            priors[ch] = (src, r['edges'][0]['edge_mv'])
    return res


def finish_run(run, caches, res):
    """Write the per-run figure + JSON once both source passes are in."""
    z = caches[run]
    cen = 0.5 * (z['lin_edges'][:-1] + z['lin_edges'][1:])
    panels = [(v['source'], v['arm'], v['kind'], f'{v["kind"]}{v["arm"]}',
               int(ch[-1]) - 1, v['role'])
              for ch, v in res['channels'].items()]
    figure(run, cen, res, panels)
    for v in res['channels'].values():
        v.pop('_plot', None)
    (CALIB / f'srccal_edges_{run}.json').write_text(json.dumps(res, indent=2))

    for src, bar in S.sources_in(run).items():
        ch = S.bar_key(bar)
        e = res['channels'].get(ch, {}).get('edges', [])
        print(f'  {run} {src:6s} on {bar} ({ch}): '
              + (', '.join(f"{x['kevee']:.0f}keVee={x['edge_mv']:.2f}"
                           f"+-{x['edge_mv_err']:.2f}mV" for x in e)
                 or 'NO EDGE FOUND'))
    print(f'  -> calib/srccal_edges_{run}.json, figures/33_srccal/edges_{run}.png')


def figure(run, cen, res, panels):
    n = len(panels)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow),
                             squeeze=False)
    axf = axes.flat
    for k, (src, arm, kind, tree, c, role) in enumerate(panels):
        ch = f'{tree}{c + 1}'
        v = res['channels'][ch]
        r, sig = v['_plot']
        ax = axf[k]
        ax.step(cen, sig, where='mid', color='0.8', lw=0.7, label='raw')
        ax.step(cen, r['sub'], where='mid', color='0.45', lw=0.7,
                label='bkg-subtracted')
        ax.plot(cen, r['sm'], color='steelblue', lw=1.0, alpha=0.8)
        for xx, yy in r['curves']:
            ax.plot(xx, yy, color='k', lw=1.3)
        for e in v['edges']:
            col = {477.34: 'seagreen', 698.63: 'darkorange',
                   1612.06: 'crimson'}.get(e['kevee'], 'purple')
            ax.axvline(e['edge_mv'], color=col, lw=1.3)
            ax.axvspan(e['edge_mv'] - e['edge_mv_err'],
                       e['edge_mv'] + e['edge_mv_err'], color=col, alpha=0.25)
            ax.text(e['edge_mv'], 1.5, f" {e['edge_mv']:.1f}", color=col,
                    fontsize=7, rotation=90, va='bottom')
        ax.axvline(v['valley_mv'], color='green', lw=0.6, ls=':')
        ax.set_yscale('log')
        top = max([e['edge_mv'] for e in v['edges']], default=60.0)
        ax.set_xlim(0, min(float(cen[-1]), max(2.4 * top, 120.0)))
        ax.set_ylim(bottom=1)
        star = '*' if role == 'source_bar' else ''
        ax.set_title(f'{ch}{star} [{src}] {v["hits_per_trigger"]:.0f} hit/trig',
                     fontsize=8)
        ax.tick_params(labelsize=6)
        if k >= n - ncol:
            ax.set_xlabel('amplitude [mV]', fontsize=7)
    for k in range(n, nrow * ncol):
        axf[k].axis('off')
    axf[0].legend(fontsize=6)
    src_txt = ', '.join(f'{k} on {v}' for k, v in S.sources_in(run).items())
    fig.suptitle(f'{run}: {src_txt} — Compton-edge extraction '
                 '(gray=raw, dark gray=background-subtracted, blue=smoothed, '
                 'black=fit; green=477, orange=699, red=1612 keVee; '
                 '* = source bar)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / f'edges_{run}.png', dpi=140)
    plt.close(fig)


def main():
    runs = sys.argv[1:] or S.RUNS
    caches = {r: load(r) for r in S.RUNS}      # all nine: dark templates need them
    priors, res = {}, {}
    # Cs-137 first, across ALL runs, so every bar's clean single-gamma anchor
    # exists before any Y-88 spectrum is touched. Then Y-88 anchored on it. Then
    # a repair pass: the two brightest bars (AL, BL — 4300-5400 hits/trigger,
    # 23-33 % pileup) can lose the Cs-137 edge into the threshold turn-on, and
    # for those the anchor runs the other way, from the Y-88 699 keVee edge.
    print('== pass 1: Cs-137 (the anchor) ==')
    for r in runs:
        res[r] = process_run(r, caches, 'Cs137', priors)
    print('== pass 2: Y-88 (anchored on Cs-137 where available) ==')
    for r in runs:
        res[r] = process_run(r, caches, 'Y88', priors, res[r])
    missing = [S.bar_key(b) for r in runs for s, b in S.sources_in(r).items()
               if s == 'Cs137' and not res[r]['channels'].get(
                   S.bar_key(b), {}).get('edges')]
    if missing:
        print(f'== pass 3: repair Cs-137 for {missing} (anchored on Y-88) ==')
        for r in runs:
            if any(S.bar_key(b) in missing for s, b in S.sources_in(r).items()
                   if s == 'Cs137'):
                res[r] = process_run(r, caches, 'Cs137', priors, res[r])
    print('   anchors: ' + ', '.join(f'{k}={v[1]:.2f}mV[{v[0]}]'
                                     for k, v in sorted(priors.items())))
    for r in runs:
        finish_run(r, caches, res[r])


if __name__ == '__main__':
    main()
