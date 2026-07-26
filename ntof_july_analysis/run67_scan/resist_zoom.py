#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-detector resist-HV zoom for the FOLLOW-UP fine scan.

Question (operator, 2026-07-23): fix drift to 600-700 V, and for each detector
find the ~20 V resist range worth scanning finely. Target being ALIVE in the
2-8 ms post-flash window; where the early window is problematic for a detector,
push to a later window.

Method: pool drift {600, 700} and every COMPLETE plastic-threshold block (the HV
response is essentially threshold-independent, and pooling is needed because a
single (mip, drift, resist) cell has only ~5 pairs in a 6 ms window). Per
detector we read two things off the resist ladder in the target window:
  * P(3D x/y pair) per trigger — the tracking efficiency (noise-robust);
  * blind fraction — the post-flash aliveness (read out, produced no hits).
The optimum resist trades these off: gain (hence P) rises with resist, but so
does post-flash saturation (hence blindness) — the early-window optimum sits at
the highest resist that is still ALIVE, which is BELOW the late-window optimum.

Per detector the target window is chosen automatically as the earliest window in
which it is alive (blind < BLIND_OK); detectors that are blind at 2-8 ms fall
back to a later window and are flagged.

Output -> <ANALYSIS_DIR>/July_HV_Scan/run67_scan/resist_zoom/
  resist_zoom.png            P(pair) & blind vs resist, per det, target + late
  resist_zoom_recommend.md   the per-detector 20 V scan ranges + reasoning
  resist_zoom.csv            the pooled numbers

Run: .venv/bin/python ntof_july_analysis/run67_scan/resist_zoom.py
"""
import json
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
import stats as S  # noqa: E402

OUT = os.path.join(L.OUT_BASE, 'resist_zoom')
DRIFTS = [600, 700]          # the operator's chosen fine-scan drift band
BLIND_OK = 0.20              # "alive" threshold on the blind fraction
# candidate windows, earliest first; a detector is placed in the earliest one it
# is alive in. NON-OVERLAPPING so per_cell_stats bins cleanly.
TARGET = (2.0, 8.0)
LATE = (20.0, 30.0)          # reference / fallback for slow detectors
HV_FLOOR = 495               # do not suggest scanning below this (gas ceiling margin)


def pooled(st, det, wl):
    """P(pair)+err, blind, n vs resist for one det+window (drift+thr pooled)."""
    s = st[(st.det == det) & (st.window == wl)]
    out = []
    for r in sorted(s.resist.unique()):
        d = s[s.resist == r]
        k = np.round(d.p_pair * d.n).sum()
        n = d.n.sum()
        p, e = S.binom_err(k, n)
        out.append(dict(resist=int(r), p=float(p), e=float(e),
                        blind=float(d.blind_frac.mean()), n=int(n), k=int(k)))
    return pd.DataFrame(out)


def choose_window(df_t, df_l):
    """Target window if the detector is alive there at any resist, else late."""
    if (df_t.blind < BLIND_OK).any():
        return TARGET, df_t, 'target'
    return LATE, df_l, 'late'


def recommend_range(df, win_is_target):
    """Optimum resist + a 20 V scan range.

    Optimum = resist maximising P among the ALIVE (blind < BLIND_OK) resists.
    Range = 20 V centred on the optimum, but if a saturation edge (first resist
    at/above which blind >= BLIND_OK) sits inside the upper half, cap the top at
    that edge and push the window down (early-window optima are saturation-edge
    limited, so we scan below the edge, not across it). Rounded to 5 V, floored.
    """
    alive = df[df.blind < BLIND_OK]
    if alive.empty:
        alive = df                       # nothing alive: fall back to all
    ro = int(alive.loc[alive.p.idxmax()].resist)
    # 1-sigma plateau among alive resists
    pmax = float(alive.p.max()); emax = float(alive.loc[alive.p.idxmax()].e)
    plat = alive[alive.p >= pmax - emax].resist
    # saturation edge: lowest resist with blind >= BLIND_OK
    sat = df[df.blind >= BLIND_OK].resist
    sat_edge = int(sat.min()) if len(sat) else None
    lo, hi = ro - 10, ro + 10
    if sat_edge is not None and hi >= sat_edge:
        hi = sat_edge                    # scan up TO the edge, not past it
        lo = hi - 20
    lo = max(HV_FLOOR, int(round(lo / 5) * 5))
    hi = int(round(hi / 5) * 5)
    return dict(r_opt=ro, plateau=(int(plat.min()), int(plat.max())),
                sat_edge=sat_edge, scan_lo=lo, scan_hi=hi)


# DAQ repo calibrations dir — the file a downstream scan-builder reads. The
# calibrations tree already anticipates an `mm/` (Micromegas) folder.
CALIB_DIR = '/home/mx17/PycharmProjects/nTof_x17_DAQ/calibrations/mm'
CALIB_JSON = os.path.join(CALIB_DIR, 'resist_hv_run67.json')

DET_NOTE = {
    'A': 'clean M1 — reference, high confidence.',
    'B': 'noise-dominated M1 — low P everywhere and does NOT saturate; optimum '
         'is weak, driven by gain not aliveness.',
    'C': 'best early tracker but saturates sharply above the edge.',
    'D': 'slowest recovery — see the flag if it is blind at 2-8 ms.',
}


def _profile_block(df):
    return {
        'resist_V': [int(x) for x in df.resist],
        'p_pair_x1000': [round(x * 1000, 1) for x in df.p],
        'p_err_x1000': [round(x * 1000, 1) for x in df.e],
        'blind_frac': [round(x, 3) for x in df.blind],
        'n_trig': [int(x) for x in df.n],
    }


def export_calibration(rows, data, pooled_mips):
    """Write the per-detector resist recommendation to the DAQ calibrations tree
    (calibrations/mm/) for a downstream scan-builder to read. Regenerated from
    the analysis, never hand-edited — matches the calibrations/README convention.
    """
    os.makedirs(CALIB_DIR, exist_ok=True)
    role = {'A': 'early', 'B': 'early', 'C': 'early', 'D': 'late'}
    thr_txt = mip_join(pooled_mips)
    n_thr = len(pooled_mips)
    method = ('P(3D x/y pair) per recorded trigger (noise-robust track '
              'efficiency) and blind fraction (post-flash aliveness = read out '
              'but 0 hits) vs resist, pooled over drift {600,700} V and '
              + ('the ' if n_thr > 1 else '')
              + (f'{["one","two","all three"][n_thr-1]} ' if n_thr <= 3
                 else f'{n_thr} ')
              + f'complete plastic threshold{"s" if n_thr > 1 else ""} '
              f'({thr_txt}; HV response is threshold-independent). '
              'Early-window optimum = highest resist still ALIVE (blind < 0.20): '
              'gain rises with resist but so does post-flash saturation, and '
              'they cross at the knee.')
    caveats = [
        'RELATIVE efficiency (single-arm events dominate; no absolute '
        'normalization). Det A is the clean-M1 reference; B/C/D M1 are noisy so '
        'single-plane track yield is inflated — the 3D-pair metric used here is '
        'the noise-robust one.',
        'Per-cell statistics are thin in a 6 ms window (~5 pairs/cell), hence '
        'the drift+threshold pooling; the resist SHAPE is robust, individual '
        'points are ~15-25% error.',
        'Late windows (>=20 ms) prefer HIGHER resist (still rising at the 550 V '
        'ladder top), so these EARLY-window ranges are deliberately lower than a '
        'late-optimised setting would be.',
    ]
    if 141 not in pooled_mips:
        caveats.insert(1, '1.41 MIP threshold block incomplete at export time '
                       'and NOT included; it tightens errors but does not move '
                       'the ranges (the optimum is a gain/saturation property, '
                       'not a threshold one). Regenerate to fold it in.')
    calib = {
        'provenance': {
            'run': 'run_67 (2026-07-22/23)',
            'exported': '2026-07-23',
            'source': 'nTof_x17/ntof_july_analysis/run67_scan/resist_zoom.py',
            'purpose': 'Inform a FOLLOW-UP per-detector fine resist-HV scan. '
                       'These are recommended SCAN RANGES, not final operating '
                       'points.',
            'method': method,
            'thresholds_pooled': [L.MIP_LABEL[m] for m in
                                  sorted(pooled_mips,
                                         key=lambda m: [90, 113, 141].index(m))],
            'caveats': caveats,
        },
        'drift_hv_V': {
            'scan_range': list(DRIFTS),
            'chosen_by': 'operator 2026-07-23',
            'note': 'Fine scan fixes drift to this band; scan resist within it. '
                    'Results here are pooled over both drift points.',
        },
        'metric': 'P(3D x/y pair) per recorded trigger; blind_frac = fraction '
                  'read out with 0 hits.',
        'alive_threshold_blind_frac': BLIND_OK,
        'detectors': {det: _det_block(det, rows[det], data, role[det])
                      for det in 'ABCD'},
    }
    with open(CALIB_JSON, 'w') as f:
        json.dump(calib, f, indent=2)
    return CALIB_JSON


def _det_block(det, r, data, role):
    blk = {
        'role': role,
        'target_window_ms': list(TARGET if r['window'] == 'target' else LATE),
        'resist_optimum_V': r['r_opt'],
        'resist_1sigma_plateau_V': list(r['plateau']),
        'saturation_edge_V': r['sat_edge'],
        'fine_scan_resist_V': {'range': [r['scan_lo'], r['scan_hi']],
                               'suggested_step_V': 2},
        'note': DET_NOTE[det],
        'measured_profile': {
            L.win_label(*TARGET): _profile_block(data[(det, 'target')]),
            L.win_label(*LATE): _profile_block(data[(det, 'late')]),
        },
    }
    if det == 'D':
        # D's range is the computed 20-30 ms optimum (its earliest live window).
        blk['fine_scan_resist_V']['alt_for_early_aliveness'] = {
            'range': [500, 520],
            'why': 'D is blind at 2-8 ms for every resist 520-550 (blind 0.47 '
                   'at 520 rising to 0.95 at 550); to chase earlier aliveness '
                   'scan BELOW 520, trading gain for faster post-flash '
                   'recovery. Otherwise treat D as a >=20 ms detector.',
        }
        blk['note'] = ('DEAD in the 2-8 ms window at all resist 520-550 — treat '
                       'as a late (>=20 ms) detector. Optimum below is for the '
                       '20-30 ms window; it keeps improving toward 540-545 V in '
                       '30-50 ms, so raise the range to 535-555 V if the later '
                       'tail matters.')
    return blk


def fig(rows, data):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for j, det in enumerate('ABCD'):
        r = rows[det]
        dft = data[(det, 'target')]
        dfl = data[(det, 'late')]
        ax = axes[0, j]
        ax.errorbar(dft.resist, dft.p * 1000, dft.e * 1000, color=S.DET_COL[det],
                    marker='o', ms=5, capsize=2, label=f'{L.win_label(*TARGET)}')
        ax.errorbar(dfl.resist, dfl.p * 1000, dfl.e * 1000, color='0.55',
                    marker='s', ms=4, capsize=2, ls='--',
                    label=f'{L.win_label(*LATE)} (late ref)')
        ax.axvspan(r['scan_lo'], r['scan_hi'], color=S.DET_COL[det], alpha=0.12,
                   lw=0)
        ax.axvline(r['r_opt'], color=S.DET_COL[det], ls=':', lw=1.2)
        tag = '' if r['window'] == 'target' else '  [2-8 ms DEAD -> late]'
        ax.set_title(f'Det {det}{tag}', color=S.DET_COL[det], fontsize=11)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel('P(3D x/y pair) x1000')
            ax.legend(fontsize=8)
        ax = axes[1, j]
        ax.plot(dft.resist, dft.blind, color=S.DET_COL[det], marker='o', ms=5,
                label=f'{L.win_label(*TARGET)}')
        ax.plot(dfl.resist, dfl.blind, color='0.55', marker='s', ms=4, ls='--',
                label=f'{L.win_label(*LATE)}')
        ax.axhline(BLIND_OK, color='crimson', ls=':', lw=1,
                   label=f'alive < {BLIND_OK:g}')
        ax.axvspan(r['scan_lo'], r['scan_hi'], color=S.DET_COL[det], alpha=0.12,
                   lw=0)
        ax.set_xlabel('resist HV [V]')
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel('blind fraction\n(read out, 0 hits)')
            ax.legend(fontsize=7)
    fig.suptitle('run_67 — per-detector resist zoom for the fine scan '
                 f'(drift {DRIFTS[0]}-{DRIFTS[1]} V, m090+m113 pooled).  '
                 'Shaded = suggested 20 V range;  dotted = optimum.\n'
                 'top: tracking efficiency;  bottom: post-flash aliveness.  '
                 'Early-window optimum = highest resist still alive.',
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(OUT, 'resist_zoom.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def mip_join(mips):
    """'0.90 + 1.13 + 1.41 MIP' from a list of mip tags."""
    return ' + '.join(L.MIP_LABEL[m].replace(' MIP', '') for m in
                      sorted(mips, key=lambda m: [90, 113, 141].index(m))) + ' MIP'


def main():
    os.makedirs(OUT, exist_ok=True)
    ev, _ = S.load()
    st = S.per_cell_stats(ev, [TARGET, LATE])
    # pool over ALL thresholds with a complete HV grid (HV response is
    # threshold-independent, so this only adds statistics). Falls back
    # gracefully while a block is still processing.
    pooled_mips = S.complete_mips(st, verbose=False)
    st = st[(st.drift.isin(DRIFTS)) & (st.mip.isin(pooled_mips))].copy()

    rows, data, csv = {}, {}, []
    for det in 'ABCD':
        dft = pooled(st, det, L.win_label(*TARGET))
        dfl = pooled(st, det, L.win_label(*LATE))
        data[(det, 'target')] = dft
        data[(det, 'late')] = dfl
        (lo, hi), use_df, which = choose_window(dft, dfl)
        rec = recommend_range(use_df, which == 'target')
        rec['window'] = which
        rec['win_ms'] = L.win_label(lo, hi)
        rows[det] = rec
        for _, rr in dft.assign(window=L.win_label(*TARGET)).iterrows():
            csv.append({**rr.to_dict(), 'det': det})
        for _, rr in dfl.assign(window=L.win_label(*LATE)).iterrows():
            csv.append({**rr.to_dict(), 'det': det})
    pd.DataFrame(csv).to_csv(os.path.join(OUT, 'resist_zoom.csv'), index=False)
    print('  ->', fig(rows, data))
    print('  ->', export_calibration(rows, data, pooled_mips))

    # recommendation
    n_thr = len(pooled_mips)
    thr_word = {1: 'the single complete threshold', 2: 'the two complete '
                'thresholds', 3: 'all three thresholds'}.get(n_thr, f'{n_thr} '
                'thresholds')
    lines = ['# run_67 — per-detector resist ranges for the fine scan', '',
             f'Drift band: **{DRIFTS[0]}-{DRIFTS[1]} V** (operator choice). '
             f'Pooled over both drifts and {thr_word} '
             f'({mip_join(pooled_mips)}). Target window **{L.win_label(*TARGET)}**; a '
             f'detector blind there is pushed to **{L.win_label(*LATE)}** and '
             f'flagged. "Alive" = blind fraction < {BLIND_OK:g}.', '',
             '| Det | window used | optimum resist | 1σ plateau | sat. edge | '
             '**suggested 20 V scan** | note |', '|---|---|---|---|---|---|---|']
    for det in 'ABCD':
        r = rows[det]
        se = f"{r['sat_edge']} V" if r['sat_edge'] else '— (none)'
        flag = '' if r['window'] == 'target' else ' **(2-8 ms dead)**'
        lines.append(
            f"| {det} | {r['win_ms']}{flag} | {r['r_opt']} V | "
            f"{r['plateau'][0]}-{r['plateau'][1]} V | {se} | "
            f"**{r['scan_lo']}-{r['scan_hi']} V** | {DET_NOTE[det]} |")
    lines += ['', '5 V steps in this scan; a fine scan at ~2 V steps over each '
              'range gives ~10 points. Det D note: at 2-8 ms it is blind at '
              'every resist in 520-550 (0.47 blind even at 520, rising to 0.95 '
              'at 550), so it cannot be made an early detector in this band — '
              'either accept it as a 20-30 ms detector at the range above, or '
              'scan it BELOW 520 to chase earlier aliveness at the cost of gain.']
    text = '\n'.join(lines)
    with open(os.path.join(OUT, 'resist_zoom_recommend.md'), 'w') as f:
        f.write(text)
    print('\n' + text)


if __name__ == '__main__':
    main()
