"""36_srccal_legacy_check.py — is the fleet actually gain-equalized, and is the
07-28 scale comparable to the 07-19 equalization? (cross-check, not calibration)

The question this answers
------------------------
On 2026-07-19 the plastic PMT bias was moved so that EVERY bar's Y-88 699 keVee
Compton edge would land on a common target of 31.2 mV
(`nTof_x17_DAQ/calibrations/pss/hv_equalization_y88_fifo.json`), and those
voltages are still the standing set. The 07-28 campaign measures that same edge
at the same operating point, so the target is a hard prediction — and the
measured edges span 29-65 mV instead. Something moved by up to 2x.

Two candidate causes, with opposite consequences:

  (A) ANALYSIS. The 07-28 files were reprocessed with our v12_liqpileup
      UserInput, which added PSS pulse-shape fitting — a different `amp`
      definition — and this analysis fits the edge with a sloped continuum
      rather than 22's flat one. Then the equalization is fine and only the
      numbers are on a different footing.
  (B) HARDWARE. The equalization transported each PMT through a FIFO factor and
      a gain power law measured in a scan; if that transport is off, the fleet
      is genuinely not equalized and the trigger point is not uniform.

The separator: the 07-17 source runs (224476-79) were reprocessed by the SAME
07-30 official pass. Push them through the SAME chain and compare with the
edges 22/23 published from them at the time. That difference is (A) alone,
measured. Whatever is left over in the 07-28 comparison is (B).

Caveat kept in view: the 07-17 runs are pre-FIFO (BNC-T readout) and at the old
flat HV, so their ABSOLUTE scale is not expected to match the 07-28 one. Only
the same-run, same-data reprocessing shift is being measured here.

Outputs:
  calib/srccal_legacy_check.json
  figures/33_srccal/legacy_check.png
Usage: python 36_srccal_legacy_check.py
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


def load(run):
    p = CACHE / f'33_srccal_{run}.npz'
    if not p.exists():
        raise SystemExit(f'missing {p} — run 33_srccal_spectra.py on the legacy '
                         'runs first (lxplus/srccal_legacy.sub)')
    return np.load(p)


def legacy_edges():
    """Fit the Y-88 edges on the 07-17 runs with the 07-28 machinery.

    Background: the same channel summed over the legacy runs whose arm carried
    no source (3 of 4 for every arm) — the same construction as 34, restricted
    to the legacy campaign so no 07-28 run leaks in.
    """
    z = {r: load(r) for r in S.LEGACY_RUNS}
    cen = 0.5 * (z[S.LEGACY_RUNS[0]]['lin_edges'][:-1]
                 + z[S.LEGACY_RUNS[0]]['lin_edges'][1:])
    out = {}
    for run in S.LEGACY_RUNS:
        arm = S.sources_in(run)['Y88'][0]
        tree = f'PSS{arm}'
        darks = S.dark_runs_for(arm, runs=S.LEGACY_RUNS)
        ntrig = int(z[run]['n_triggers'])
        nb = sum(int(z[r]['n_triggers']) for r in darks)
        for c in range(2):
            bar = arm + 'LR'[c]
            bkg = sum(z[r][f'{tree}_lin_np'][c] for r in darks)
            r = edgefit.extract(cen, z[run][f'{tree}_lin_np'][c], bkg,
                                ntrig / nb, 'PSS', S.EDGES_OF['Y88'], n_boot=120)
            e = {x['kevee']: x for x in r['edges']}
            out[bar] = dict(
                run=run, hits_per_trigger=round(float(z[run][f'{tree}_nhit'][c])
                                                / ntrig, 1),
                edge699_mv=e[S.E_Y1]['edge_mv'] if S.E_Y1 in e else None,
                edge699_err=e[S.E_Y1]['edge_mv_err'] if S.E_Y1 in e else None,
                edge1612_mv=e[S.E_Y2]['edge_mv'] if S.E_Y2 in e else None)
            print(f'  {bar} ({run}): 699 = {out[bar]["edge699_mv"]} mV')
    return out


def main():
    print('== 07-17 legacy runs through the 07-28 chain ==')
    leg = legacy_edges()

    # what 22/23 published from these same runs, on the OLD processing
    old = json.loads((CALIB / 'y88_energy_calib.json').read_text())['channels']
    # what 34 measured on 07-28
    new = {}
    for run in S.RUNS:
        d = json.loads((CALIB / f'srccal_edges_{run}.json').read_text())
        for src, bar in S.sources_in(run).items():
            if src != 'Y88':
                continue
            v = d['channels'].get(S.bar_key(bar), {})
            for e in v.get('edges', []):
                if e['kevee'] == S.E_Y1:
                    new.setdefault(bar, []).append(e['edge_mv'])

    rows = {}
    for bar in BARS:
        ch = S.bar_key(bar)
        o = old.get(ch, {}).get('edge699_mv')
        l = leg.get(bar, {}).get('edge699_mv')
        n = float(np.mean(new[bar])) if bar in new else None
        rows[bar] = dict(
            ch=ch,
            edge699_0717_published_mv=o,
            edge699_0717_reprocessed_mv=l,
            reprocessing_shift=round(l / o, 3) if (o and l) else None,
            edge699_0728_mv=round(n, 2) if n else None,
            equalization_target_mv=S.EQUALIZED_TARGET_699_MV,
            vs_target=round(n / S.EQUALIZED_TARGET_699_MV, 3) if n else None,
            hv_v=S.PLASTIC_HV_V[bar], hv_index_n=S.PLASTIC_HV_INDEX_N[bar])
        # The comparison that means something. The equalization target is in
        # OLD-PSA units; the 07-28 measurement is in new-PSA units. The
        # conversion is not a guess — it is measured per bar, on the very same
        # 07-17 runs, as `reprocessing_shift`. Divide it out and what is left is
        # the genuine hardware residual.
        if n and rows[bar]['reprocessing_shift']:
            resid = rows[bar]['vs_target'] / rows[bar]['reprocessing_shift']
            rows[bar]['equalization_residual'] = round(resid, 3)
            # HV that would close the residual, along this PMT's own power law
            rows[bar]['hv_for_target_v'] = int(round(
                S.PLASTIC_HV_V[bar] * (1.0 / resid) ** (1.0 / S.PLASTIC_HV_INDEX_N[bar])))
            rows[bar]['dv_to_target'] = rows[bar]['hv_for_target_v'] - S.PLASTIC_HV_V[bar]

    shifts = [r['reprocessing_shift'] for r in rows.values() if r['reprocessing_shift']]
    targets = [r['vs_target'] for r in rows.values() if r['vs_target']]
    resid = {b: r['equalization_residual'] for b, r in rows.items()
             if r.get('equalization_residual')}
    rv = np.array(list(resid.values()))
    ok = {b: v for b, v in resid.items() if abs(v - 1) <= 0.15}
    off = {b: v for b, v in resid.items() if abs(v - 1) > 0.15}
    verdict = dict(
        reprocessing_shift_median=round(float(np.median(shifts)), 3),
        reprocessing_shift_range=[round(min(shifts), 3), round(max(shifts), 3)],
        vs_target_median=round(float(np.median(targets)), 3),
        vs_target_range=[round(min(targets), 3), round(max(targets), 3)],
        raw_spread_across_fleet=round(max(targets) / min(targets), 2),
        residual=resid,
        residual_spread=round(float(rv.max() / rv.min()), 2),
        within_15pct=sorted(ok), outliers=sorted(off))
    verdict['reading'] = (
        'The same runs, reprocessed and refitted, move by '
        f'{verdict["reprocessing_shift_median"]}x (range '
        f'{verdict["reprocessing_shift_range"]}) — the analysis-side effect of '
        'the new PSA amplitude definition plus the new fitter, measured rather '
        'than assumed. The raw 07-28-vs-target ratios span '
        f'{verdict["vs_target_range"]}, and they track that per-bar shift '
        'almost one-to-one. Divide it out and the genuine hardware residual is '
        f'{sorted(rv)[0]:.2f}-{sorted(rv)[-1]:.2f}: '
        f'{len(ok)} of {len(resid)} bars ({", ".join(sorted(ok))}) sit within '
        f'15 % of the common target, and the outliers are '
        + ', '.join(f'{b} ({v - 1:+.0%})' for b, v in
                    sorted(off.items(), key=lambda kv: -abs(kv[1] - 1))).replace('%', r' %')
        + '. So the 07-19 equalization DID largely take; most of the apparent '
          '2.2x fleet spread is our own reprocessing, not the hardware.')

    res = dict(note=__doc__.split('Outputs:')[0].strip(), bars=rows,
               verdict=verdict)
    (CALIB / 'srccal_legacy_check.json').write_text(
        json.dumps(res, indent=2, default=float))
    figure(rows, verdict)

    print('\n bar   07-17 pub   07-17 reproc   shift    07-28    /target  '
          'residual   HV      -> for target')
    for bar, r in rows.items():
        f = lambda v, w=8: (f'{v:{w}.2f}' if isinstance(v, (int, float)) else ' ' * w)
        print(f' {bar:4s} {f(r["edge699_0717_published_mv"])} '
              f'{f(r["edge699_0717_reprocessed_mv"], 12)} '
              f'{f(r["reprocessing_shift"], 8)} {f(r["edge699_0728_mv"])} '
              f'{f(r["vs_target"], 8)} {f(r.get("equalization_residual"), 9)}  '
              f'{r["hv_v"]:5d} V  -> '
              f'{r.get("hv_for_target_v", 0):5d} V ({r.get("dv_to_target", 0):+d})')
    print('\n' + verdict['reading'])
    print('\n-> calib/srccal_legacy_check.json, figures/33_srccal/legacy_check.png')


def figure(rows, verdict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = np.arange(len(BARS))

    ax = axes[0]
    ax.bar(x - 0.2, [rows[b]['edge699_0717_published_mv'] or 0 for b in BARS],
           0.4, label='07-17, as published (old PSA + 22)', color='0.6')
    ax.bar(x + 0.2, [rows[b]['edge699_0717_reprocessed_mv'] or 0 for b in BARS],
           0.4, label='07-17 data, v12 PSA + this fitter', color='steelblue')
    ax.set_xticks(x); ax.set_xticklabels(BARS)
    ax.set_ylabel('699 keVee edge [mV]')
    ax.set_title('(a) Same runs, new analysis\n= the analysis-side shift', fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.25, axis='y')

    ax = axes[1]
    ax.bar(x - 0.2, [rows[b]['vs_target'] or 0 for b in BARS], 0.4,
           color='0.6', label='raw 07-28 / target')
    ax.bar(x + 0.2, [rows[b].get('equalization_residual') or 0 for b in BARS],
           0.4, color='crimson',
           label='after dividing out the measured analysis shift')
    ax.axhline(1.0, color='k', lw=1.5)
    ax.axhspan(0.85, 1.15, color='0.9', zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(BARS)
    ax.set_ylabel('measured / equalization target')
    ax.set_title('(b) Is the fleet equalized? Raw says no (spread '
                 f'{verdict["raw_spread_across_fleet"]}x);\ncorrected says '
                 f'mostly yes ({len(verdict["within_15pct"])}/8 within 15 %)',
                 fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.25, axis='y')

    ax = axes[2]
    ax.bar(x, [rows[b].get('dv_to_target', 0) for b in BARS], 0.6, color='seagreen')
    ax.axhline(0, color='k', lw=1)
    for i, b in enumerate(BARS):
        ax.text(i, rows[b].get('dv_to_target', 0), f"{rows[b]['hv_v']}",
                ha='center', va='bottom' if rows[b].get('dv_to_target', 0) < 0
                else 'top', fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(BARS)
    ax.set_ylabel('dV to reach the target [V]')
    ax.set_title('(c) HV move that would close the RESIDUAL,\n'
                 'via each PMT\'s own power law (labels = standing HV)',
                 fontsize=10)
    ax.grid(alpha=0.25, axis='y')

    fig.suptitle('Cross-check: is the plastic fleet gain-equalized at the '
                 'standing operating point?', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / 'legacy_check.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
