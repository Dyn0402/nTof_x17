#!/usr/bin/env python3
"""hv_currents.py [--hv DIR] [--out DIR] -- Paper I §2/§4: resistive-electrode currents on the cosmic bench.

Stephan's design argument (Dylan, 2026-11-02): a uniform resistive layer came out with too low a resistance and
showed "hot spots" where current flowed; resistive strips plus passivation allowed a higher resistance and much less
current. This puts numbers on the strip chambers' currents, per chamber and against the resist HV, from the CAEN
monitor written every few seconds during each sub-run (hv_monitor.csv: '<board>:<ch> vmon/imon', imon in uA).

Per detector and sub-run (channels from that run's run_config.json, 'resist' and 'drift'):
  V            median resist vmon [V]
  I_med        median resist imon [uA] (the steady current: leakage + signal)
  I_p99        99th percentile (spikes: discharges recharging the electrode)
  spikes/h     samples with imon > I_med + max(0.05 uA, 5 x robust sigma), merged into episodes, per hour
  R_eff        V / I_med [GOhm] (an effective DC resistance to ground; not a model of the leakage path)
Inputs: <hv>/<run>/run_config.json and <hv>/<run>/<subrun>/hv_monitor.csv (copied from EOS cosmic_bench; the
fetch is in the 11-02 session notes). Writes <out>/currents.csv, currents.json and figures/currents_vs_hv.png.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

ORDER = ['mx17_2', 'mx17_3', 'mx17_4', 'mx17_6', 'mx17_7']
LABEL = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7', 'mx17_1': 'det1'}


# Physical board from the run-config description: the config NAMES were renumbered between nights (6-17: mx17_1 is
# the Arnaud board, later mx17_2), the descriptions follow the board. det2 (Arnaud, 12 June) is the first board of the
# n_TOF batch; Dylan 2026-11-02: no current data from before it is used (the Jan-May prototypes stay in the CSV only).
BOARDS = (('Bulked by Arnaud June 12', 'det2'), ('Bulked by Stephan June 15', 'det3'), ('Was board C', 'det4'),
          ('Was board D', 'det6'), ('Was board B', 'det7'))
FIRST_DAY = pd.Timestamp('2026-06-17')
# 6-19/6-20 weekend: the 'mx17_3' (Stephan 15 June) curve is the 6-22 det2 curve point for point (same 1.9 uA floor),
# and that weekend carries the known slot confusion (MEMORY: weekend det2/det3) -> board assignment not trusted
AMBIGUOUS_RUNS = {'mx17_det2_det3_weekend_6-20-26'}


def board_of(desc):
    for k, b in BOARDS:
        if k in (desc or ''):
            return b
    return None


def run_day(run):
    import re
    m = re.search(r'(\d{1,2})-(\d{1,2})-(\d{2})$', run)
    return pd.Timestamp(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else pd.NaT


def subrun_rows(run_dir):
    cfg = json.load(open(os.path.join(run_dir, 'run_config.json')))
    dets = [d for d in cfg['detectors'] if d.get('det_type') == 'mx17' or d['name'].startswith('mx17')]
    rows = []
    for f in sorted(glob.glob(os.path.join(run_dir, '*', 'hv_monitor.csv'))):
        sub = os.path.basename(os.path.dirname(f))
        try:
            H = pd.read_csv(f)
        except Exception:
            continue
        if len(H) < 20:
            continue
        t = pd.to_datetime(H.timestamp, errors='coerce'); hours = (t.max() - t.min()).total_seconds() / 3600
        dt = float(np.nanmedian(np.diff(t.astype('int64'))) / 1e9) if len(t) > 2 else np.nan
        for d in dets:
            ch = d.get('hv_channels', {})
            if 'resist' not in ch:
                continue
            b, c = ch['resist']; vcol, icol = f'{b}:{c} vmon', f'{b}:{c} imon'
            if vcol not in H or icol not in H:
                continue
            v = pd.to_numeric(H[vcol], errors='coerce').to_numpy(); i = pd.to_numeric(H[icol], errors='coerce').to_numpy()
            on = np.isfinite(v) & np.isfinite(i) & (v > 0.9 * np.nanmax(v)) & (np.nanmax(v) > 100)
            if on.sum() < 20:
                continue
            vv, ii = v[on], i[on]
            med = float(np.median(ii)); rs = float(1.4826 * np.median(np.abs(ii - med)))
            thr = med + max(0.05, 5 * rs)
            hi = ii > thr
            episodes = int(np.sum(hi[1:] & ~hi[:-1]) + (hi[0] if len(hi) else 0))
            dv, di = ch.get('drift', [None, None]), None
            if dv[0] is not None and f'{dv[0]}:{dv[1]} imon' in H:
                di = float(np.nanmedian(pd.to_numeric(H[f'{dv[0]}:{dv[1]} imon'], errors='coerce')[on]))
            rows.append(dict(run=cfg['run_name'], subrun=sub, det=d['name'], label=LABEL.get(d['name'], d['name']),
                             resist_type=d.get('resist_type'), description=d.get('description', ''), gas=cfg.get('gas'),
                             V=float(np.median(vv)), I_med=med, I_rsig=rs, I_p99=float(np.percentile(ii, 99)),
                             I_max=float(ii.max()), spikes_per_h=episodes / max(hours, 1e-3), hours=hours, dt_s=dt,
                             R_eff_GOhm=float(np.median(vv)) / med / 1e3 if med > 0 else np.nan, I_drift=di))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hv', default=os.path.join(os.environ.get('WORK_DATA', '/tmp'), 'claude-jobs/17c92c46/hv'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, 'figures'), exist_ok=True)
    rows = []
    for rd in sorted(glob.glob(os.path.join(a.hv, '*'))):
        if os.path.exists(os.path.join(rd, 'run_config.json')):
            rows += subrun_rows(rd)
    T = pd.DataFrame(rows).sort_values(['det', 'run', 'V'])
    T['board'] = T.description.map(board_of)
    T['day'] = T.run.map(run_day)
    T['use'] = T.board.notna() & (T.day >= FIRST_DAY) & ~T.run.isin(AMBIGUOUS_RUNS)
    T.to_csv(os.path.join(a.out, 'currents.csv'), index=False)
    summ = {}
    U = T[T.use]
    # per board and night, the current at the HV nearest 490 V
    summ['per_board_night_near_490V'] = {
        f'{b} {r}': dict(V=float(g.iloc[(g.V - 490).abs().argsort()[:1]].V.iloc[0]),
                         I_uA=float(g.iloc[(g.V - 490).abs().argsort()[:1]].I_med.iloc[0]), n_subruns=int(len(g)))
        for (b, r), g in U.groupby(['board', 'run'])}
    for (det, rt), g in U.groupby(['board', 'resist_type']):
        # linear fit I = G V + I0 over the chamber's sub-runs (the leakage part is ohmic if I0 ~ 0)
        G, I0 = np.polyfit(g.V, g.I_med, 1) if g.V.nunique() > 2 else (np.nan, np.nan)
        near = g.iloc[(g.V - 490).abs().argsort()[:3]]
        summ[det] = dict(
            n_subruns=int(len(g)), runs=sorted(g.run.unique().tolist()), V_range=[float(g.V.min()), float(g.V.max())],
            I_med_range_uA=[float(g.I_med.min()), float(g.I_med.max())],
            I_med_near_490V_uA=float(near.I_med.median()), V_near=float(near.V.median()),
            R_eff_near_490V_GOhm=float(near.R_eff_GOhm.median()), slope_uA_per_100V=float(100 * G),
            intercept_uA=float(I0), spikes_per_h_median=float(g.spikes_per_h.median()))
    # det3 on its own (6-27 Saturday scan: one chamber, no slot ambiguity): base + gain-driven excess
    from scipy.optimize import curve_fit
    g = T[(T.run == 'mx17_det3_saturday_scan_6-27-26') & T.subrun.str.contains('hv_scan')].sort_values('V')
    if len(g) > 5:
        f = lambda V, b, A, V2: b + A * 2 ** ((V - 490) / V2)
        (b, A, V2), cov = curve_fit(f, g.V, g.I_med, p0=(0.17, 1.0, 12.0))
        e = np.sqrt(np.diag(cov))
        D = T[(T.run == 'mx17_det3_saturday_scan_6-27-26') & T.subrun.str.startswith('drift_scan')]
        summ['det3_6-27_fit'] = dict(model='I = b + A * 2**((V-490)/V2)', b_uA=float(b), b_err=float(e[0]),
                                     A_uA_at_490=float(A), A_err=float(e[1]), V2_doubling_V=float(V2), V2_err=float(e[2]),
                                     gain_doubling_V=16.7, excess_gain_power=float(16.7 / V2),
                                     drift_scan_I_range_uA=[float(D.I_med.min()), float(D.I_med.max())] if len(D) else None)
        print('det3 6-27 fit:', summ['det3_6-27_fit'])
    json.dump(summ, open(os.path.join(a.out, 'currents.json'), 'w'), indent=1)
    for k, v in summ.items():
        if not isinstance(v, dict) or 'n_subruns' not in v:
            continue
        print(f"{k:40s} n={v['n_subruns']:3d} V {v['V_range'][0]:.0f}-{v['V_range'][1]:.0f}  I(~{v['V_near']:.0f} V) "
              f"{v['I_med_near_490V_uA']:.3f} uA  R_eff {v['R_eff_near_490V_GOhm']:.2f} GOhm  "
              f"slope {v['slope_uA_per_100V']:.3f} uA/100V  I0 {v['intercept_uA']:.3f}  spikes/h {v['spikes_per_h_median']:.1f}")
    try:
        make_figs(U, summ, a.out)
    except Exception as e:
        print('figure skipped:', e)


def make_figs(U, summ, out):
    """Two figures (Dylan 2026-11-03: the old all-boards plot was too busy for the text, y linear wanted):
    currents_vs_hv      main text, chamber A (det3) nights only: full range + a zoom of the leakage floor before the
                        exponential takes off (the same points, the same det3 6-27 fit);
    currents_vs_hv_all  appendix, every board, y linear, boards by colour above the axes, nights by marker."""
    import sys
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'papers'))
    import paper_style as ps
    ps.use()
    fit = summ.get('det3_6-27_fit')
    scan = U[U.subrun.str.contains('resist_') & ~U.subrun.str.startswith('drift_scan')]
    nights = sorted({r for (b, r), g in scan.groupby(['board', 'run']) if len(g) >= 4}, key=lambda r: (run_day(r), r))
    mk = dict(zip(nights, ['o', 's', '^', 'D', 'v', 'P', 'X']))      # one marker per night (deterministic)
    night_lab = {r: r.split('_')[-1][:-3] + (' scan' if 'scan' in r else '') for r in nights}
    order = ['det3', 'det2', 'det6', 'det7', 'det4']

    def curves(ax, boards, ms_a=3.6, ms=2.8):
        for (b, r), g in sorted(scan.groupby(['board', 'run']), key=lambda x: order.index(x[0][0])):
            g = g.sort_values('V')
            if len(g) < 4 or b not in boards:
                continue
            kw = ps.det(b, label=None, marker=mk[r], ms=ms_a if b == 'det3' else ms, alpha=1.0 if b == 'det3' else 0.85)
            kw['lw'] = 0.9 if b == 'det3' else 0.6
            ax.plot(g.V, g.I_med, mec='white', mew=0.3, **kw)

    def fitline(ax, lo=440):
        V = np.linspace(lo, 528, 200)
        ax.plot(V, fit['b_uA'] + fit['A_uA_at_490'] * 2 ** ((V - 490) / fit['V2_doubling_V']), '--',
                color=ps.MODEL, lw=1.0, zorder=6)

    fit_lab = r'det3 fit, $\times 2$ per %.1f V' % fit['V2_doubling_V']
    # ---- main text: chamber A only, full range + floor zoom
    fig, (ax, zx) = ps.fig_for(6.1, height=2.5, ncols=2, gridspec_kw=dict(width_ratios=[1.6, 1]))
    curves(ax, ('det3',)); curves(zx, ('det3',))
    fitline(ax, 425); fitline(zx, 420)
    ax.set_xlabel('resistive-strip HV [V]'); zx.set_xlabel('resistive-strip HV [V]')
    ax.set_ylabel(r'resistive-strip current [$\mu$A]')
    ax.set_ylim(0, 1.05 * float(scan[scan.board == 'det3'].I_med.max())); ax.set_xlim(415, 530)
    zx.set_xlim(415, 485); zx.set_ylim(0, 0.6)
    hn = [Line2D([], [], ls='none', marker=mk[r], color=ps.DET_COLOR['det3'], ms=3.5, label=night_lab[r])
          for r in nights if r in set(scan[scan.board == 'det3'].run)]
    hn.append(Line2D([], [], ls='--', color=ps.MODEL, lw=1.0, label=fit_lab))
    ax.legend(handles=hn, loc='upper left')
    zx.text(0.04, 0.95, 'leakage floor\nbefore the gain takes off', transform=zx.transAxes, va='top',
            fontsize=ps.BASE_PT * 0.8, color=ps.INK)
    zx.axhline(0.167, color=ps.MUTED, lw=0.9, ls=':', zorder=0)      # 425 V point of the 6-27 scan, the lowest HV
    zx.text(0.04, 0.167 / 0.6 - 0.03, r'0.17 $\mu$A at 425 V', transform=zx.transAxes, ha='left', va='top',
            fontsize=ps.BASE_PT * 0.75, color=ps.MUTED)
    ps.panel_labels([ax, zx])
    ps.save(fig, os.path.join(out, 'figures', 'currents_vs_hv'), data=U[U.board == 'det3'])
    # ---- appendix: every board
    fig, ax = ps.fig_for(6.1, height=3.0)
    curves(ax, order)
    fitline(ax, 440)
    # gain slope reference: the gas gain doubles every 16.7 V; drawn through the det3 fit's excess at 500 V x 2
    Vr = np.linspace(470, 525, 50)
    I500 = 2 * fit['A_uA_at_490'] * 2 ** (10 / fit['V2_doubling_V'])
    ax.plot(Vr, I500 * 2 ** ((Vr - 500) / 16.7), ':', color=ps.MUTED, lw=1.4, zorder=6)
    ax.set_xlabel('resistive-strip HV [V]'); ax.set_ylabel(r'median resistive-strip current [$\mu$A]')
    ax.set_ylim(0, 1.05 * float(scan.I_med.max())); ax.set_xlim(395, 530)
    hb = [Line2D([], [], **{k: v for k, v in ps.det(b).items() if k not in ('zorder', 'label', 'marker', 'lw')},
                 lw=1.6, label=b) for b in order if b in set(scan.board)]
    hm = [Line2D([], [], ls='--', color=ps.MODEL, lw=1.0, label=fit_lab),
          Line2D([], [], ls=':', color=ps.MUTED, lw=1.4, label=r'gain $\times 2$ per 16.7 V')]
    ax.add_artist(ax.legend(handles=hm, loc='upper left'))
    hn = [Line2D([], [], ls='none', marker=mk[r], color=ps.MUTED, ms=3.5, label=night_lab[r]) for r in nights]
    blank = Line2D([], [], ls='none', marker=None, label=' ')
    ncol = max(len(hb), len(hn))
    hb += [blank] * (ncol - len(hb)); hn += [blank] * (ncol - len(hn))
    ps.legend_above(ax, handles=[h for pair in zip(hb, hn) for h in pair], ncol=ncol, columnspacing=0.9,
                    handletextpad=0.35, handlelength=1.5)
    ps.save(fig, os.path.join(out, 'figures', 'currents_vs_hv_all'), data=U)


if __name__ == '__main__':
    main()
