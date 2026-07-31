"""35_srccal_calib.py — absolute energy scale from the 2026-07-28 two-source
campaign (T3), plus the campaign's four built-in controls.

What the two-source design buys over the 07-17 Y-88 scan (`23_y88_energy_calib.py`):

  * THREE edges per plastic bar (477 / 699 / 1612 keVee) instead of two, so the
    scale is a fitted LINE, mV = a * E + b. The 07-17 analysis had to assume
    b = 0; here b is measured, and a non-zero b is exactly what a threshold- or
    baseline-induced scale error looks like. The through-origin slope is
    reported alongside so the two campaigns compare like for like.
  * every bar is illuminated DIRECTLY (source centred on it), so no
    "between the bars" light-sharing assumption enters the plastic scale.
  * Y-88 and Cs-137 run simultaneously on opposite arms, so the two energies
    are taken in the same DAQ state — a gain drift between source exposures
    cannot masquerade as nonlinearity.

Controls, all reported here:
  1. AR repeatability   — Y on AR in both 224588 and 224596.
  2. Cs leakage         — 224596 is Y-only; arm C's rate there vs its dark-run
                          mean bounds how much the far source contaminated the
                          Cs measurements in the other eight runs.
  3. detn <-> L/R map   — the source lights ONE bar, so the hot channel names
                          the mapping. Confirms or breaks the assumed
                          detn 1 = left / 2 = right.
  4. transport to 07-17 — per-channel gain ratio against
                          calib/y88_energy_calib.json (same convention, same
                          edge definitions), i.e. what the scintillator scale
                          did over the 11 days between the campaigns.

Outputs:
  calib/srccal_energy_calib.json
  figures/33_srccal/energy_calib.png
  SRCCAL_RESULTS_2026-07-28.md
Usage: python 35_srccal_calib.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import srccal_runs as S

BASE = Path(__file__).parent
CACHE = BASE / 'cache'
CALIB = BASE / 'calib'
OUT = BASE / 'figures' / '33_srccal'
OUT.mkdir(parents=True, exist_ok=True)

RATIO_Y = S.E_Y2 / S.E_Y1        # 2.307
RATIO_YCS = S.E_Y1 / S.E_CS      # 1.4636


def wls_line(E, y, dy, through_origin=False):
    """Weighted least squares. Returns (a, b, da, db) for y = a*E + b, or
    (a, 0, da, 0) if through_origin."""
    E, y, w = np.asarray(E, float), np.asarray(y, float), 1.0 / np.asarray(dy, float) ** 2
    if through_origin or len(E) < 2:
        a = float((w * E * y).sum() / (w * E * E).sum())
        return a, 0.0, float(np.sqrt(1.0 / (w * E * E).sum())), 0.0
    X = np.vstack([E, np.ones_like(E)]).T
    C = np.linalg.inv(X.T @ (w[:, None] * X))
    p = C @ (X.T @ (w * y))
    return float(p[0]), float(p[1]), float(np.sqrt(C[0, 0])), float(np.sqrt(C[1, 1]))


# An edge whose bootstrap error exceeds this fraction of its own position is
# reported but not calibrated on. It is the wall channels this removes: the
# source is clamped on the plastic and the wall sits in front of it, so a wall
# bump comes out at 20-40 % error (WALB5 38.5 +- 13.5 mV) — a real excess, but
# nothing to hang an energy scale on. The liquids' Y-88 bumps pass easily
# (LIQA1 26.45 +- 0.12) and their Cs-137 ones do not, which is the same story
# the raw rates tell.
MAX_REL_ERR = 0.15

# Energies the RECOMMENDED scale is fitted on. The 1612 keVee edge is measured
# and reported, but it is deliberately NOT fitted on: anchored on the two clean
# points it lands between -22 % and +1 % depending on the bar, and the 07-17
# campaign saw a comparable bar-to-bar spread (1612/699 = 1.81-2.61 there,
# 1.91-2.28 here) with a different geometry and a different fitter. Its
# bootstrap error is 0.3-0.9 %, so that spread is a systematic, not noise —
# which makes it a 20 %-level indicator, not a calibration point. Its errors
# are small enough that including it would dominate the weighted fit and drag
# the slope by ~10 %.
#
# Liquids get the 699 keVee edge ONLY. They do see the Cs-137 source (LIQA
# gains 1.66x with it on the near bar), but at their gain a 477 keVee edge
# would sit at ~23 mV against a threshold valley at ~16 mV — inside the
# turn-on — and the fit duly returns nonsense: 32.56 mV, i.e. ABOVE that same
# channel's 699 keVee edge at 33.73. Reported, not fitted on.
CALIBRATION_ENERGIES = {'PSS': (S.E_CS, S.E_Y1),
                        'LIQ': (S.E_Y1,),
                        'WAL': (S.E_Y1,)}


def collect():
    """channel -> list of measurements {kevee, mv, err, run, source, role},
    plus (liquids only) the same 699 keVee edges in the legacy bump convention."""
    per_ch, edges_json, bump = {}, {}, {}
    for run in S.RUNS:
        p = CALIB / f'srccal_edges_{run}.json'
        if not p.exists():
            raise SystemExit(f'missing {p} — run 34_srccal_edges.py first')
        d = json.loads(p.read_text())
        edges_json[run] = d
        for ch, v in d['channels'].items():
            for e in v['edges']:
                err = max(e['edge_mv_err'], 0.1)
                per_ch.setdefault(ch, []).append(dict(
                    kevee=e['kevee'], mv=e['edge_mv'], err=err,
                    used=bool(np.isfinite(err) and err < MAX_REL_ERR * e['edge_mv']),
                    halfheight=e['edge_mv_halfheight'],
                    pileup_shift=e.get('pileup_shift_mv'),
                    run=run, source=v['source'], role=v['role'], kind=v['kind'],
                    arm=v['arm']))
            # liquids only: the same edges in 22's bump convention, so the
            # 07-17 transport can be compared like-for-like (see 34). Only from
            # runs where the primary step fit ALSO found the edge: on a
            # no-signal spectrum (source on the far bar) the bump finder still
            # returns something, and it would drag the average.
            step_ok = any(e['kevee'] == S.E_Y1 for e in v['edges'])
            for e in (v.get('edges_bump_convention') or []):
                if step_ok and e['kevee'] == S.E_Y1 and np.isfinite(e['edge_mv_err']):
                    bump.setdefault(ch, []).append(
                        (e['edge_mv'], max(e['edge_mv_err'], 0.1)))
    return per_ch, edges_json, bump


def calibrate(per_ch):
    out = {}
    for ch, pts in sorted(per_ch.items()):
        kind = pts[0]['kind']
        # Plastics: only the runs where THIS bar carried the source define the
        # scale; the same-arm neighbour is a light-sharing measurement, not a
        # calibration point. Walls/liquids have no such distinction — they view
        # the source from a fixed position whichever bar it sits on, so all lit
        # runs count (the spread across them is the position systematic).
        use = [p for p in pts if p['role'] == 'source_bar'] if kind == 'PSS' else pts
        seen, use = len(use), [p for p in use if p['used']]
        if not use:
            out[ch] = dict(kind=kind, arm=pts[0]['arm'], n_points=0,
                           status=f'{seen} edge(s) seen, none within '
                                  f'{100 * MAX_REL_ERR:.0f}% error — not calibrated')
            continue
        # average repeated measurements of the same edge (e.g. AR: 588 + 596)
        byE = {}
        for p in use:
            byE.setdefault(p['kevee'], []).append(p)
        E, y, dy, spread = [], [], [], {}
        for k, g in sorted(byE.items()):
            w = 1.0 / np.array([p['err'] for p in g]) ** 2
            m = float((w * np.array([p['mv'] for p in g])).sum() / w.sum())
            e = float(np.sqrt(1.0 / w.sum()))
            if len(g) > 1:
                sp = float(np.std([p['mv'] for p in g], ddof=1))
                spread[k] = round(sp, 3)
                e = float(np.hypot(e, sp))       # inflate by the repeat spread
            E.append(k); y.append(m); dy.append(e)
        rec = dict(kind=kind, arm=pts[0]['arm'], n_points=len(E),
                   points={f'{k:.2f}': dict(mv=round(v, 3), err=round(d, 3),
                                            runs=[p['run'] for p in byE[k]])
                           for k, v, d in zip(E, y, dy)},
                   repeat_spread_mv=spread or None)
        # Split the measured points into the ones the scale is fitted on and the
        # ones only reported (see CALIBRATION_ENERGIES).
        fit_on = CALIBRATION_ENERGIES.get(kind, (S.E_CS, S.E_Y1))
        keep = [i for i, k in enumerate(E) if k in fit_on]
        extra = [i for i, k in enumerate(E) if k not in fit_on]
        if keep:
            E_all, y_all, dy_all = E, y, dy
            E = [E_all[i] for i in keep]
            y = [y_all[i] for i in keep]
            dy = [dy_all[i] for i in keep]
            rec['fitted_on_kevee'] = [round(k, 2) for k in E]
            rec['reported_not_fitted_kevee'] = [round(E_all[i], 2) for i in extra]
            rec['n_points'] = len(E)
        a0, _, da0, _ = wls_line(E, y, dy, through_origin=True)
        rec['mv_per_kevee_origin'] = round(a0, 5)
        rec['mv_per_mevee_origin'] = round(a0 * 1000, 3)
        rec['mv_per_kevee_origin_err'] = round(da0, 5)
        if len(E) >= 2:
            a, b, da, db = wls_line(E, y, dy)
            rec.update(mv_per_kevee=round(a, 5), mv_per_mevee=round(a * 1000, 3),
                       mv_per_kevee_err=round(da, 5),
                       offset_mv=round(b, 3), offset_mv_err=round(db, 3),
                       offset_significant=bool(abs(b) > 3 * db))
            resid = [round(float(v - (a * k + b)), 3) for k, v in zip(E, y)]
            rec['residual_mv'] = dict(zip([f'{k:.2f}' for k in E], resid))
            rec['nonlinearity_pct'] = round(
                100 * max(abs(np.array(resid)) / np.array(y)), 2)
            # With two edges the 2-parameter line passes exactly through both,
            # so the residuals are zero by construction and carry no
            # information. The linearity statement then lives entirely in the
            # measured edge RATIO against its expected value.
            rec['line_is_exact'] = bool(len(E) <= 2)
            # where the un-fitted point lands relative to the fitted line: the
            # response check, and the honest measure of what a 1612 keVee
            # calibration would have been worth
            for i in extra:
                pred = rec['mv_per_kevee'] * E_all[i] + rec['offset_mv'] \
                    if 'mv_per_kevee' in rec else a0 * E_all[i]
                rec.setdefault('check_vs_fitted_line', {})[f'{E_all[i]:.2f}'] = dict(
                    measured_mv=round(y_all[i], 2), predicted_mv=round(pred, 2),
                    ratio=round(y_all[i] / pred, 3))

        # free edge-ratio cross-checks (energy assignment, independent of gain)
        d = {k: v for k, v in zip(E_all if keep else E, y_all if keep else y)}
        if S.E_Y2 in d and S.E_Y1 in d:
            rec['ratio_1612_699'] = round(d[S.E_Y2] / d[S.E_Y1], 3)
            rec['ratio_1612_699_expected'] = round(RATIO_Y, 3)
        if S.E_Y1 in d and S.E_CS in d:
            rec['ratio_699_477'] = round(d[S.E_Y1] / d[S.E_CS], 3)
            rec['ratio_699_477_expected'] = round(RATIO_YCS, 3)
        out[ch] = rec
    return out


def compare_0717(cal, bump):
    """Per-channel gain ratio against the 07-17 Y-88 campaign, using the
    through-origin slope on both sides.

    The liquids are compared in 22's BUMP convention, not this analysis's step
    convention: 22 fitted a bump on those channels, and the two conventions
    differ by ~25 % on the same spectrum, which would swamp any real drift."""
    p = CALIB / 'y88_energy_calib.json'
    if not p.exists():
        return {'note': 'calib/y88_energy_calib.json absent — no transport'}
    old = json.loads(p.read_text())['channels']
    cmp_ = {}
    for ch, rec in cal.items():
        if not (ch in old and old[ch].get('mv_per_kevee')):
            continue
        if rec['kind'] == 'LIQ':
            if ch not in bump:
                continue
            w = 1.0 / np.array([e for _, e in bump[ch]]) ** 2
            mv = float((w * np.array([m for m, _ in bump[ch]])).sum() / w.sum())
            new, conv = mv / S.E_Y1, 'bump (as 22)'
        elif rec['n_points']:
            new, conv = rec['mv_per_kevee_origin'], 'step'
        else:
            continue
        r = new / old[ch]['mv_per_kevee']
        cmp_[ch] = dict(mv_per_kevee_0717=old[ch]['mv_per_kevee'],
                        mv_per_kevee_0728=round(new, 5), convention=conv,
                        ratio_0728_over_0717=round(r, 3),
                        flag='>10% change' if abs(r - 1) > 0.10 else 'ok')
    return cmp_


def controls(edges_json):
    """The four campaign controls, computed from the caches + edge JSONs."""
    z = {r: np.load(CACHE / f'33_srccal_{r}.npz') for r in S.RUNS}
    ctl = {}

    # 1. AR repeatability (Y on AR in 224588 and 224596)
    ch = S.bar_key('AR')
    rep = {}
    for run in ('run224588', 'run224596'):
        v = edges_json[run]['channels'].get(ch, {})
        rep[run] = {f'{e["kevee"]:.0f}': e['edge_mv'] for e in v.get('edges', [])}
    common = set(rep['run224588']) & set(rep['run224596'])
    ctl['AR_repeatability'] = dict(
        edges=rep,
        diff_pct={k: round(100 * (rep['run224596'][k] / rep['run224588'][k] - 1), 2)
                  for k in common} if common else {})

    # 2a. Far-source leakage, done the way the control run actually supports it:
    # arm A carries the SAME Y-88 source in 224588 (Cs also present, on CR) and
    # in 224596 (Cs removed). If the far Cs source leaked a spectral component
    # into arm A, the two spectra would differ in SHAPE, not just in rate.
    lin588 = z['run224588']['PSSA_lin_np'][1] / int(z['run224588']['n_triggers'])
    lin596 = z[S.CONTROL_RUN]['PSSA_lin_np'][1] / int(z[S.CONTROL_RUN]['n_triggers'])
    cen = 0.5 * (z['run224588']['lin_edges'][:-1] + z['run224588']['lin_edges'][1:])
    # Compare in 2 mV bands, not raw 0.25 mV bins: per-bin Poisson noise alone
    # spreads the ratio by ~15 % and would fake a shape difference.
    nb = int(round(2.0 / (cen[1] - cen[0])))
    trim = (len(cen) // nb) * nb
    band_c = cen[:trim].reshape(-1, nb).mean(1)
    b588 = lin588[:trim].reshape(-1, nb).sum(1)
    b596 = lin596[:trim].reshape(-1, nb).sum(1)
    band = (band_c > 15) & (band_c < 100)
    ratio = b588[band] / np.clip(b596[band], 1e-12, None)
    good = np.isfinite(ratio) & (b596[band] > 1e-3)
    ctl['far_source_leakage'] = dict(
        what='PSSA2 (Y-88 on AR) with the far Cs-137 present (224588) vs removed '
             '(224596): a leak would change the SHAPE, not only the rate',
        rate_ratio_588_over_596=round(float(lin588.sum() / lin596.sum()), 3),
        shape_ratio_median=round(float(np.median(ratio[good])), 3),
        shape_ratio_spread_pct=round(
            100 * float(np.std(ratio[good]) / np.median(ratio[good])), 1),
        verdict=('no spectral leak: the ratio is flat, the difference is overall '
                 'rate (source re-placement between runs)'
                 if float(np.std(ratio[good]) / np.median(ratio[good])) < 0.10
                 else 'SHAPE DIFFERS — investigate before trusting Cs runs'))

    # 2b. What the naive "control vs dark mean" comparison really shows: the
    # dark templates are NOT source-free. The ring order is A-D-C-B, and the
    # sources always sit on OPPOSITE arms, so the two unlit arms in any run are
    # each ADJACENT to both lit ones and pick up scatter. Arm C's dark runs
    # (sources on B and D) are its two neighbours; the Y-only control run is the
    # cleanest background this campaign has for it.
    neigh = {}
    for tree in ('PSSC', 'WALC', 'LIQC'):
        darks = [r for r in S.dark_runs_for('C') if r != S.CONTROL_RUN]
        dark_rate = np.sum([z[r][f'{tree}_nhit'] / int(z[r]['n_triggers'])
                            for r in darks], axis=0) / len(darks)
        ctrl_rate = z[S.CONTROL_RUN][f'{tree}_nhit'] / int(z[S.CONTROL_RUN]['n_triggers'])
        neigh[tree] = dict(
            neighbour_lit_dark_hits_per_trig=[round(float(x), 1) for x in dark_rate],
            control_run_hits_per_trig=[round(float(x), 1) for x in ctrl_rate],
            excess_of_dark_over_control_pct=[
                round(100 * float(d / max(c, 1e-9) - 1), 1)
                for c, d in zip(ctrl_rate, dark_rate)])
    ctl['dark_template_is_neighbour_lit'] = dict(
        note='the background template runs are not source-free — see 2b in the '
             'source. On a SOURCE BAR the background is ~3 % of the spectrum '
             '(excess/bkg ~30), so a 2x error in it moves the edge by <~1 %, '
             'below the convention systematic; it matters for the weaker '
             'same-arm and wall channels, where the control run is the better '
             'template.',
        per_tree=neigh)

    # 3. detn <-> L/R map: the lit bar must be the hot one
    mp = {}
    for run in S.RUNS:
        for src, bar in S.sources_in(run).items():
            tree = f'PSS{bar[0]}'
            rates = z[run][f'{tree}_nhit'] / int(z[run]['n_triggers'])
            hot = int(np.argmax(rates)) + 1
            assumed = S.assumed_detn[bar[1]]
            mp[f'{run}:{src}:{bar}'] = dict(
                rates_per_trig=[round(float(x), 1) for x in rates],
                hot_detn=hot, assumed_detn=assumed,
                agrees=bool(hot == assumed),
                contrast=round(float(max(rates) / max(min(rates), 1e-9)), 2))
    ok = sum(v['agrees'] for v in mp.values())
    ctl['detn_LR_map'] = dict(assumed=S.assumed_detn, per_run=mp,
                              n_agree=ok, n_total=len(mp),
                              verdict='CONFIRMED' if ok == len(mp)
                              else ('INVERTED' if ok == 0 else 'INCONSISTENT'))

    # 4. saturation / pileup health of the fitted channels
    health = {}
    for run in S.RUNS:
        for ch, v in edges_json[run]['channels'].items():
            if v['role'] != 'source_bar' and v['kind'] == 'PSS':
                continue
            if v['sat_frac'] > 1e-3 or v['pileup_frac'] > 0.30:
                health[f'{run}:{ch}'] = dict(sat_frac=v['sat_frac'],
                                             pileup_frac=v['pileup_frac'],
                                             hits_per_trigger=v['hits_per_trigger'])
    worst = sorted(health.items(), key=lambda kv: -kv[1]['pileup_frac'])[:5]
    ctl['flagged_channels'] = dict(
        n_flagged=len(health),
        criterion='sat_frac > 1e-3 or pileup_frac > 0.30',
        max_sat_frac=round(max([v['sat_frac'] for v in health.values()],
                               default=0.0), 6),
        worst_pileup={k: v['pileup_frac'] for k, v in worst},
        all=health)
    return ctl


def figure(cal, cmp_):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    # (a) plastic linearity: three edges per bar
    ax = fig.add_subplot(gs[0, 0])
    for ch, rec in sorted(cal.items()):
        if rec['kind'] != 'PSS' or not rec['n_points']:
            continue
        E = np.array([float(k) for k in rec['points']])
        y = np.array([v['mv'] for v in rec['points'].values()])
        dy = np.array([v['err'] for v in rec['points'].values()])
        o = np.argsort(E)
        ax.errorbar(E[o], y[o], yerr=dy[o], marker='o', ms=4, lw=1, label=ch)
        if 'mv_per_kevee' in rec:
            xx = np.linspace(0, 1700, 10)
            ax.plot(xx, rec['mv_per_kevee'] * xx + rec['offset_mv'],
                    lw=0.7, ls='--', alpha=0.6, color=ax.lines[-1].get_color())
    for E, lab in ((S.E_CS, 'Cs 477'), (S.E_Y1, 'Y 699'), (S.E_Y2, 'Y 1612')):
        ax.axvline(E, color='0.8', lw=0.7, zorder=0)
        ax.text(E, ax.get_ylim()[1], lab, fontsize=7, rotation=90, va='top')
    ax.set_xlabel('Compton edge [keVee]')
    ax.set_ylabel('edge amplitude [mV]')
    ax.set_title('Plastic bars: 3-point energy scale (dashed = fitted line)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)

    # (b) mV/MeVee per channel
    ax = fig.add_subplot(gs[0, 1])
    # one shared x axis for all three detector types, so a wall point sits under
    # its own label and not under a plastic's
    allch = [c for c in sorted(cal) if cal[c]['n_points']]
    xpos = {c: i for i, c in enumerate(allch)}
    for kind, col in (('PSS', 'crimson'), ('WAL', 'steelblue'), ('LIQ', 'darkgreen')):
        chs = [c for c in allch if cal[c]['kind'] == kind]
        if not chs:
            continue
        y = [cal[c]['mv_per_mevee_origin'] for c in chs]
        e = [cal[c]['mv_per_kevee_origin_err'] * 1000 for c in chs]
        ax.errorbar([xpos[c] for c in chs], y, yerr=e, ls='none', marker='o',
                    ms=4, color=col, label=f'{kind} (n={len(chs)})')
    ax.set_xticks(range(len(allch)))
    ax.set_xticklabels(allch, rotation=90, fontsize=6)
    ax.set_ylabel('mV per MeVee (through origin)')
    ax.set_title('Absolute scale per channel')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (c) offset / linearity residuals
    ax = fig.add_subplot(gs[1, 0])
    chs = [c for c in sorted(cal) if 'offset_mv' in cal[c]]
    ax.errorbar(range(len(chs)), [cal[c]['offset_mv'] for c in chs],
                yerr=[cal[c]['offset_mv_err'] for c in chs], ls='none',
                marker='s', ms=4, color='k')
    ax.axhline(0, color='crimson', lw=1)
    ax.set_xticks(range(len(chs)))
    ax.set_xticklabels(chs, rotation=90, fontsize=5)
    ax.set_ylabel('fitted offset b [mV]')
    ax.set_title('Zero-energy intercept (0 = the 07-17 through-origin assumption holds)')
    ax.grid(alpha=0.25)

    # (d) transport vs 07-17
    ax = fig.add_subplot(gs[1, 1])
    if isinstance(cmp_, dict) and cmp_ and 'note' not in cmp_:
        chs = sorted(cmp_)
        ax.plot(range(len(chs)), [cmp_[c]['ratio_0728_over_0717'] for c in chs],
                'o', ms=4, color='purple')
        ax.axhline(1, color='k', lw=1)
        ax.axhspan(0.9, 1.1, color='0.9', zorder=0)
        ax.set_xticks(range(len(chs)))
        ax.set_xticklabels(chs, rotation=90, fontsize=5)
        ax.set_ylabel('gain 07-28 / 07-17')
        ax.set_title('Transport to the 07-17 Y-88 campaign (band = +-10%)')
    else:
        ax.text(0.5, 0.5, 'no 07-17 calibration available', ha='center')
        ax.axis('off')
    ax.grid(alpha=0.25)

    fig.suptitle('2026-07-28 two-source plastic calibration (Y-88 + Cs-137, '
                 'opposite arms) — absolute energy scale', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / 'energy_calib.png', dpi=140)
    plt.close(fig)


def report(cal, cmp_, ctl):
    L = ['# Two-source plastic calibration — results (runs 224588-224596, 2026-07-28)',
         '',
         'Y-88 (699 + 1612 keVee) and Cs-137 (477 keVee) on opposite arms, one bar',
         'per source. Generated by `35_srccal_calib.py`.', '',
         '## Plastic bars — RECOMMENDED energy scale', '',
         'Fitted on the two clean single-gamma-quality edges only '
         f'({" and ".join(f"{e:.0f}" for e in CALIBRATION_ENERGIES["PSS"])} keVee); '
         'the 1612 keVee', 'edge is measured and reported as a check, not '
         'fitted on (see the note in 35_srccal_calib.py).', '',
         '| bar | ch | 477 mV | 699 mV | mV/MeVee | offset b [mV] | '
         '699/477 (exp 1.464) | 1612 meas/pred |',
         '|---|---|---|---|---|---|---|---|']
    for ch, r in sorted(cal.items()):
        if r['kind'] != 'PSS' or not r['n_points']:
            continue
        bar = next((b for b in (a + s for a in S.ARMS for s in 'LR')
                    if S.bar_key(b) == ch), '?')
        p = r['points']
        chk = r.get('check_vs_fitted_line', {}).get(f'{S.E_Y2:.2f}', {})
        L.append(f"| {bar} | {ch} | "
                 f"{p[f'{S.E_CS:.2f}']['mv']:.2f} | {p[f'{S.E_Y1:.2f}']['mv']:.2f} | "
                 f"{r.get('mv_per_mevee', float('nan')):.2f} | "
                 f"{r.get('offset_mv', float('nan')):+.2f} +- "
                 f"{r.get('offset_mv_err', float('nan')):.2f} | "
                 f"{r.get('ratio_699_477', '--')} | "
                 f"{chk.get('ratio', '--')} |")
    L += ['',
          'The line passes exactly through both fitted points, so its residuals',
          'carry no information — the linearity statement is the measured',
          '699/477 ratio against 1.4636, and the 1612 keVee check column.',
          '', '## Walls and liquids (bump model)', '',
          '| ch | mV/MeVee (origin) | n points | note |', '|---|---|---|---|']
    for ch, r in sorted(cal.items()):
        if r['kind'] == 'PSS':
            continue
        if not r['n_points']:
            L.append(f"| {ch} | -- | 0 | {r['status']} |")
        else:
            L.append(f"| {ch} | {r['mv_per_mevee_origin']:.2f} | "
                     f"{r['n_points']} | |")
    v = ctl['detn_LR_map']
    leak = ctl['far_source_leakage']
    nb = ctl['dark_template_is_neighbour_lit']['per_tree']['PSSC']
    fl = ctl['flagged_channels']
    L += ['', '## Controls', '',
          f"- **detn <-> L/R map**: **{v['verdict']}** "
          f"({v['n_agree']}/{v['n_total']} illuminated bars are the hot channel "
          f"under detn 1 = left, 2 = right). The README carried this as "
          f"\"per Dylan\", unverified; the campaign lights one bar at a time, so "
          f"it is now measured.",
          f"- **AR repeatability** (Y-88 on AR in 224588 and again in 224596): "
          f"{ctl['AR_repeatability']['diff_pct']} % per edge.",
          f"- **Far-source leakage**: rate ratio "
          f"{leak['rate_ratio_588_over_596']}, shape ratio flat to "
          f"{leak['shape_ratio_spread_pct']} % — {leak['verdict']}.",
          f"- **Background templates are not source-free**: the ring order is "
          f"A-D-C-B and the sources always sit on opposite arms, so each unlit "
          f"arm is adjacent to BOTH lit ones. Arm C's dark runs read "
          f"{nb['excess_of_dark_over_control_pct']} % above the Y-only control "
          f"on its two plastics. On a source bar the background is ~3 % of the "
          f"spectrum, so this moves an edge by <~1 %; it matters for the weak "
          f"same-arm and wall channels.",
          f"- **Saturation / pileup**: {fl['n_flagged']} channels flagged "
          f"({fl['criterion']}); max saturated fraction "
          f"{fl['max_sat_frac']:.2e} — nothing is clipping. Worst pileup: "
          + ', '.join(f'{k} {100 * p:.0f}%' for k, p in fl['worst_pileup'].items())
          + '.',
          '']
    if isinstance(cmp_, dict) and 'note' not in cmp_ and cmp_:
        L += ['## Transport to the 07-17 Y-88 campaign', '',
              '| ch | 07-17 mV/keVee | 07-28 mV/keVee | ratio | flag |',
              '|---|---|---|---|---|']
        for ch, c in sorted(cmp_.items()):
            L.append(f"| {ch} | {c['mv_per_kevee_0717']} | "
                     f"{c['mv_per_kevee_0728']} | "
                     f"{c['ratio_0728_over_0717']} | {c['flag']} |")
    (BASE / 'SRCCAL_RESULTS_2026-07-28.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L[:40]))
    print('...\n-> SRCCAL_RESULTS_2026-07-28.md')


def main():
    per_ch, edges_json, bump = collect()
    cal = calibrate(per_ch)
    cmp_ = compare_0717(cal, bump)
    ctl = controls(edges_json)
    (CALIB / 'srccal_energy_calib.json').write_text(json.dumps(
        {'note': ('Absolute energy scale from the 2026-07-28 two-source campaign. '
                  'mv_per_kevee = slope of the fitted line mV = a*E + b over the '
                  'measured Compton edges; mv_per_kevee_origin = the through-origin '
                  'slope, directly comparable with calib/y88_energy_calib.json '
                  '(07-17). PSS scales use only runs where the bar itself carried '
                  'the source.'),
         'energies_kevee': {'Cs137': S.E_CS, 'Y88_898': S.E_Y1, 'Y88_1836': S.E_Y2},
         'channels': cal, 'transport_vs_0717': cmp_, 'controls': ctl},
        indent=2, default=float))
    figure(cal, cmp_)
    report(cal, cmp_, ctl)
    print('-> calib/srccal_energy_calib.json, figures/33_srccal/energy_calib.png')


if __name__ == '__main__':
    main()
