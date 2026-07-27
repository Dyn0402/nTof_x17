#!/usr/bin/env python3
"""
A3 — run_64 singles 2-D scan: best drift/resist operating point per detector.

Tradeoff the scan exposes (user's framing): too much HV -> longer post-flash
saturation (less usable live window); too little HV -> inefficient. Operating
figure of merit:

  FOM(drift, resist, det) = eff_late  x  live_frac

  eff_late   = P(3D x/y pair | late probe) per trigger — recovered-window
               track-finding efficiency (the noise-robust metric; single-plane
               track-seg yield is inflated by B/C/D M1 noise). Det A is clean.
  live_frac  = post-flash live fraction of the 4-27 ms window, from the mid/late
               recovery ladder (pooled over drift per (det,resist) for stats):
                 mid ~ late (ratio>0.7) -> recovered by ~13.5 ms -> live ~ 0.85
                 mid partially back      -> recovery 13.5-27 ms  -> live ~ 0.55
                 mid ~ 0                  -> recovery >27 ms      -> live ~ 0.30

Grid: run_64 is a COMPLETE 6 drift (700..200 V) x 10 resist (560..515 V) scan,
60/60 sub-runs, so the 2-D surface is fully populated and the kernel smoother
runs as designed. The resist window sits 20 V below run_58's (580->540),
overlapping on 560..540 V: above 560 V run_58 remains the only measurement,
below 540 V run_64 is.

Per-cell late-probe yield is only ~15 pairs (+-25%), so a single-cell argmax is
statistically fragile. We therefore KERNEL-SMOOTH the efficiency surface: each
cell is pooled with its physical neighbours by a Gaussian kernel in
(drift V, resist V), the pair counts k and trials n smoothed separately, and
p = K/N with an effective-N binomial error (n_eff = (Sum w n)^2 / Sum w^2 n).
The optimum is reported with its 1-sigma PLATEAU region (all cells whose
smoothed efficiency is within 1 sigma of the max) rather than one cell.

Absolute efficiency is out of reach (single-arm events dominate -> no
inter-chamber coincidence, confirmed on the bench); all yields are RELATIVE
(per recorded trigger).

Run: .venv/bin/python ntof_july_analysis/run64_scan/optimize.py
Output -> <ANALYSIS_DIR>/July_HV_Scan/run64_scan/optimize/
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scan_lib as L  # noqa: E402
import feu_presence as FP  # noqa: E402
from analyze_tracks import per_cell_stats, binom_err  # noqa: E402

OUT = os.path.join(L.OUT_BASE, 'optimize')
DET_COL = {'A': 'crimson', 'B': 'royalblue', 'C': 'seagreen', 'D': 'darkorange'}

# kernel bandwidths for the efficiency smoother (physical HV units)
H_DRIFT, H_RESIST = 85.0, 7.0     # ~= one grid step -> pools each cell w/ neighbours
# drift velocity saturates above ~500-600 V (analyze_drift): below this the
# drift column is long (diffusion/pileup) and v is field-sensitive.
V_SAT_DRIFT = 500.0


# --------------------------------------------------------------------------- #
# recovery / live fraction (pooled over drift per (det, resist) for statistics)
# --------------------------------------------------------------------------- #
def live_frac_from_ratio(ratio):
    if not np.isfinite(ratio):
        return np.nan
    if ratio > 0.7:
        return 0.85
    if ratio > 0.2:
        return 0.55
    return 0.30


def recovery_by_resist(st):
    """{(det, resist): (mid/late track ratio, live_frac)}, drift-pooled."""
    out = {}
    for (det, r), g in st.groupby(['det', 'resist']):
        mid = g[g.probe_class == 'mid']; late = g[g.probe_class == 'late']
        k_mid = np.round(mid.p_trk * mid.n).sum(); n_mid = mid.n.sum()
        k_late = np.round(late.p_trk * late.n).sum(); n_late = late.n.sum()
        p_mid = k_mid / n_mid if n_mid else np.nan
        p_late = k_late / n_late if n_late else np.nan
        ratio = p_mid / p_late if p_late and p_late > 0 else np.nan
        out[(det, r)] = (ratio, live_frac_from_ratio(ratio))
    return out


# --------------------------------------------------------------------------- #
# kernel-smoothed efficiency surface
# --------------------------------------------------------------------------- #
def kn_grid(st, det, metric='pair', cls='late'):
    """(drifts, resists, K, N) count grids for one probe class, one detector."""
    s = st[(st.det == det) & (st.probe_class == cls)]
    pcol = 'p_pair' if metric == 'pair' else 'p_trk'
    drifts = np.array(sorted(s.drift.unique()), float)
    resists = np.array(sorted(s.resist.unique()), float)
    K = np.full((len(resists), len(drifts)), np.nan)
    N = np.full_like(K, np.nan)
    for i, r in enumerate(resists):
        for j, d in enumerate(drifts):
            cell = s[(s.drift == d) & (s.resist == r)]
            # After the run_64 FEU-dropout correction a cell can be genuinely
            # EMPTY (that detector was never read out at this HV point, e.g.
            # drift 700 / resist 515 for det A). Leave it NaN — smooth_surface
            # masks it out and the neighbours carry the estimate.
            if cell.empty:
                continue
            n_cell = float(cell.n.iloc[0])
            p_cell = float(cell[pcol].iloc[0])
            if n_cell <= 0 or not np.isfinite(p_cell):
                continue
            K[i, j] = round(p_cell * n_cell)
            N[i, j] = n_cell
    return drifts, resists, K, N


def smooth_surface(drifts, resists, K, N, hd=H_DRIFT, hr=H_RESIST):
    """Gaussian-kernel-smoothed efficiency P=K/N with effective-N binomial err."""
    nr, nd = K.shape
    P = np.full_like(K, np.nan); E = np.full_like(K, np.nan)
    Neff = np.full_like(K, np.nan)
    valid = np.isfinite(K) & np.isfinite(N) & (N > 0)
    Kv = np.where(valid, K, 0.0); Nv = np.where(valid, N, 0.0)
    for i in range(nr):
        for j in range(nd):
            wr = np.exp(-0.5 * ((resists[:, None] - resists[i]) / hr) ** 2)
            wd = np.exp(-0.5 * ((drifts[None, :] - drifts[j]) / hd) ** 2)
            w = np.where(valid, wr * wd, 0.0)
            Nw = np.sum(w * Nv)
            if Nw <= 0:
                continue
            p = np.sum(w * Kv) / Nw
            neff = Nw ** 2 / max(np.sum(w ** 2 * Nv), 1e-9)
            P[i, j] = p
            Neff[i, j] = neff
            E[i, j] = np.sqrt(max(p * (1 - p), 1e-9) / neff)
    return P, E, Neff


def optimum(drifts, resists, P, E):
    """Argmax cell + 1-sigma plateau (cells within 1 sigma of the max)."""
    if not np.isfinite(P).any():
        return None
    ij = np.unravel_index(np.nanargmax(P), P.shape)
    pmax, emax = P[ij], E[ij]
    plateau = np.isfinite(P) & (P >= pmax - emax)
    ri, di = np.where(plateau)
    return dict(
        drift=float(drifts[ij[1]]), resist=float(resists[ij[0]]),
        p=float(pmax), e=float(emax),
        drift_lo=float(drifts[di].min()), drift_hi=float(drifts[di].max()),
        resist_lo=float(resists[ri].min()), resist_hi=float(resists[ri].max()),
        plateau=plateau)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _heat(ax, drifts, resists, Z, opt, title, cmap='viridis'):
    im = ax.imshow(Z[::-1], aspect='auto', cmap=cmap,
                   extent=[-0.5, len(drifts) - 0.5, -0.5, len(resists) - 0.5])
    ax.set_xticks(range(len(drifts)))
    ax.set_xticklabels([f'{int(c)}' for c in drifts], fontsize=8)
    ax.set_yticks(range(len(resists)))
    ax.set_yticklabels([f'{int(c)}' for c in resists], fontsize=8)
    ax.set_xlabel('drift HV [V]', fontsize=9)
    ax.set_ylabel('resist HV (A/B/C) [V]', fontsize=9)
    ax.set_title(title, fontsize=10)
    if opt:
        # plateau outline + optimum star (y flipped: row 0 at bottom)
        pl = opt['plateau']
        for (i, j) in zip(*np.where(pl)):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor='white', lw=1.2, alpha=0.7))
        jo = list(drifts).index(opt['drift'])
        io = list(resists).index(opt['resist'])
        ax.plot(jo, io, marker='*', ms=20, color='gold',
                markeredgecolor='black')
    return im


def fig_heatmaps(surf, valcol, label, fname, cmap='viridis'):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    for ax, det in zip(axes.flat, 'ABCD'):
        d = surf[det]
        im = _heat(ax, d['drifts'], d['resists'], d[valcol], d.get('opt'),
                   f'Det {det}' + (' (clean M1)' if det == 'A' else ''), cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(ax.get_title(), color=DET_COL[det])
    fig.suptitle(f'run_64 — {label}  (kernel-smoothed; box = 1-sigma plateau, '
                 'star = optimum)', fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_profiles(surf, cls='late'):
    """Line profiles through each detector's optimum (raw + smoothed)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, det in zip(axes.flat, 'ABCD'):
        d = surf[det]
        opt = d.get('opt')
        if not opt:
            ax.set_title(f'Det {det}: no optimum', fontsize=10)
            continue
        drifts, resists = d['drifts'], d['resists']
        io = list(resists).index(opt['resist'])
        jo = list(drifts).index(opt['drift'])
        # vs drift at optimum resist
        ax.errorbar(drifts, d['Praw'][io], d['Eraw'][io], color=DET_COL[det],
                    marker='o', ms=4, capsize=2, alpha=0.4, ls='',
                    label='raw @ opt resist')
        ax.plot(drifts, d['P'][io], color=DET_COL[det], lw=2,
                label='smoothed @ opt resist')
        ax.axvspan(opt['drift_lo'], opt['drift_hi'], color=DET_COL[det],
                   alpha=0.08, lw=0)
        ax.axvline(V_SAT_DRIFT, color='gray', ls=':', lw=1)
        ax.set_title(f"Det {det}: opt drift {opt['drift']:.0f} V, "
                     f"resist {L.resist_for_det(opt['resist'], det):.0f} V",
                     fontsize=10, color=DET_COL[det])
        ax.set_xlabel('drift HV [V]')
        ax.set_ylabel(f'P(3D pair | {cls})')
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f'run_64 ({cls}) — efficiency vs drift at each detector\'s '
                 'optimum resist '
                 '(shaded = 1-sigma plateau; dotted = v_drift saturation ~500 V)',
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, f'profiles_{cls}.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def build_surfaces(st, cls='late'):
    rec = recovery_by_resist(st)
    surf = {}
    for det in 'ABCD':
        drifts, resists, K, N = kn_grid(st, det, 'pair', cls)
        P, E, Neff = smooth_surface(drifts, resists, K, N)
        Praw, Eraw = binom_err(K, N)
        # live_frac broadcast over drift from the per-resist recovery ladder
        lf = np.array([[live_frac_or_nan(rec, det, r) for _ in drifts]
                       for r in resists])
        fom = P * lf
        opt_eff = optimum(drifts, resists, P, E)
        opt_fom = optimum(drifts, resists, np.nan_to_num(fom, nan=-1), E)
        surf[det] = dict(drifts=drifts, resists=resists, K=K, N=N,
                         P=P, E=E, Praw=Praw, Eraw=Eraw, fom=fom, lf=lf,
                         opt=opt_eff, opt_fom=opt_fom)
    return surf, rec


def live_frac_or_nan(rec, det, r):
    return rec.get((det, r), (np.nan, np.nan))[1]


def suggest_setpoint(oe, of, det):
    """Physics-informed single setpoint from the efficiency plateau + v-sat +
    FOM. Drift: v-saturated end of the plateau (>=500 V if the plateau reaches
    it, else the plateau max). Resist: efficiency-optimum, pulled toward the FOM
    resist only if the FOM prefers a materially lower gain (live-window win)."""
    if oe['drift_hi'] >= V_SAT_DRIFT:
        drift = max(min(oe['drift'], oe['drift_hi']), V_SAT_DRIFT)
        drift = min(drift, oe['drift_hi'])
        if oe['drift'] < V_SAT_DRIFT <= oe['drift_hi']:
            drift = V_SAT_DRIFT            # push a low-drift argmax up to v-sat
    else:
        drift = oe['drift_hi']
    resist_abc = oe['resist']
    if of and of['resist'] < oe['resist']:
        resist_abc = 0.5 * (oe['resist'] + of['resist'])   # split eff vs FOM
    return drift, L.resist_for_det(round(resist_abc / 5) * 5, det)


CLS_DESC = {
    'early': ('the **4 ms group** — the 4 triggers at ~4.1 ms after the gamma '
              'flash, i.e. the EARLIEST the front end can be read at all'),
    'mid': 'the **13 ms group** — the 2 triggers at ~13.5 ms',
    'late': ('**all later triggers integrated** (27 / 41 / 55 / 69 ms), i.e. the '
             'fully-recovered window'),
}


def recommendation(surf, cls='late'):
    """Per-detector written operating-point recommendation for one time group."""
    lines = [f'# run_64 ({cls}) — recommended drift / resist operating point '
             'per detector',
             '',
             f'Time group: {CLS_DESC[cls]}.',
             '',
             '**The optimum is NOT the same for the three groups** — the resist '
             'dependence reverses with time since the flash (4 ms falls with '
             'HV, 13 ms peaks mid-range, 27+ ms rises). Use the file matching '
             'the window you actually care about; do not read one and assume '
             'the others.',
             '',
             f'Primary metric = **P(3D x/y pair | {cls})** per '
             'trigger: the noise-robust track-finding efficiency (single-arm '
             'events dominate, so this is RELATIVE, not absolute). Optima are '
             'kernel-smoothed (each cell pooled with its HV neighbours); the '
             '**plateau** is every cell within 1 sigma of the peak — inside it '
             'the setting does not matter within errors. Det D resist = the '
             'A/B/C setpoint - 10 V.',
             '',
             'Drift guidance: v_drift saturates above ~500-600 V (see '
             '`analyze_drift` for THIS run — do not assume run_58\'s knee), and '
             '`suggest_setpoint` pulls the drift up to that knee when the '
             'plateau reaches it, trading a little efficiency for a saturated '
             '(and therefore stable) drift velocity. **For the 4 ms group that '
             'trade is NOT obviously right** — efficiency there falls with '
             'drift as well as with resist, so if the 4 ms window is what you '
             'care about, read the efficiency optimum line rather than the '
             'suggested setpoint. '
             'run_64 scans 700..200 V in 6 steps and 560..515 V in resist; the '
             'resist window is 20 V below run_58\'s, so run_64 is the authority '
             'below 540 V and run_58 above 560 V.',
             '']
    rows = []
    for det in 'ABCD':
        d = surf[det]
        oe, of = d['opt'], d['opt_fom']
        if not oe:
            lines.append(f'## Det {det}: insufficient statistics\n')
            continue
        s_drift, s_resist = suggest_setpoint(oe, of, det)
        note = {'A': 'clean M1 — reference detector, HIGH confidence.',
                'B': 'bad M1, noise-dominated: LOW confidence — tracking '
                     'marginal, treat as "run at max available gain".',
                'C': 'bad M1 but usable.',
                'D': 'bad M1; resist held 10 V below A/B/C.'}[det]
        lines += [
            f'## Det {det}',
            f'- **Suggested setpoint: drift {s_drift:.0f} V, resist '
            f'{s_resist:.0f} V.**',
            f'- Efficiency optimum: drift {oe["drift"]:.0f} V, resist '
            f'{L.resist_for_det(oe["resist"], det):.0f} V -> '
            f'P(pair|{cls}) = {oe["p"]:.3f} +- {oe["e"]:.3f}.',
            f'- 1-sigma plateau (equivalent within errors): drift '
            f'[{oe["drift_lo"]:.0f}, {oe["drift_hi"]:.0f}] V, resist '
            f'[{L.resist_for_det(oe["resist_lo"], det):.0f}, '
            f'{L.resist_for_det(oe["resist_hi"], det):.0f}] V.',
            (f'- Live-window refinement (FOM = eff x live-frac): drift '
             f'{of["drift"]:.0f} V, resist {L.resist_for_det(of["resist"], det):.0f} V'
             f' — a lower resist here trades a little efficiency for a longer '
             f'post-flash live window.' if of and of['resist'] < oe['resist']
             else '- Live-window refinement: FOM optimum coincides with the '
             'efficiency optimum (no live-window penalty at this gain).'),
            f'- Note: {note}', '']
        rows.append(dict(
            det=det, sugg_drift=int(s_drift), sugg_resist=int(s_resist),
            eff_opt_drift=int(oe['drift']),
            eff_opt_resist=int(L.resist_for_det(oe['resist'], det)),
            eff=round(oe['p'], 4), eff_err=round(oe['e'], 4),
            drift_lo=int(oe['drift_lo']), drift_hi=int(oe['drift_hi']),
            resist_lo=int(L.resist_for_det(oe['resist_lo'], det)),
            resist_hi=int(L.resist_for_det(oe['resist_hi'], det)),
            fom_drift=int(of['drift']) if of else -1,
            fom_resist=int(L.resist_for_det(of['resist'], det)) if of else -1))
    return '\n'.join(lines), pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    ev, _, _ = L.load_all()
    ev = L.add_probe_class(ev)
    ev = ev[ev.flash_ok].copy()
    # run_64 FEU dropouts: per_cell_stats evaluates each detector only on its
    # live events, so the surface must be built from a table carrying the flags.
    ev = FP.attach(ev)
    st = per_cell_stats(ev)

    for cls in ('early', 'mid', 'late'):
        surf, rec = build_surfaces(st, cls)
        print('  ->', fig_heatmaps(
            surf, 'P', f'P(3D pair | {cls}) — {CLS_DESC[cls]}',
            f'eff_smooth_heatmap_{cls}.png'))
        print('  ->', fig_profiles(surf, cls))
        text, bp = recommendation(surf, cls)
        with open(os.path.join(OUT, f'recommendation_{cls}.md'), 'w') as f:
            f.write(text)
        bp.to_csv(os.path.join(OUT, f'best_points_{cls}.csv'), index=False)
        print('  -> ' + os.path.join(OUT, f'recommendation_{cls}.md'))
        if cls == 'early':
            print('\n' + text)


if __name__ == '__main__':
    main()
