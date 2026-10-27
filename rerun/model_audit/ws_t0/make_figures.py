#!/usr/bin/env python3
"""make_figures.py -- slide-ready PNGs for workstream D (t0) into ws_t0/figures/.

Inputs (all regenerable): lead_edge.json / lead_edge.npz (lead_edge.py), valley_det3.npz (valley.py),
scan_summary.json (scan_summary.py), the finalize tables under model_audit/<det>/final_physmask.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
M = '/home/dylan/x17/cosmic_bench/model_audit'
FIG = os.path.join(HERE, 'figures')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
CX, CY = '#2a6fb0', '#d0742a'
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})


def fig_explainer():
    """What t0 is: the charge column as a line in (arrival time, position); the fit's t0 only says where the mesh is."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    v, tan = 36.0, 0.35                       # um/ns, track slope
    w = tan * v * 1e-3                        # mm/ns
    t_mesh, T = 300.0, 30000 / v
    t = np.linspace(t_mesh, t_mesh + T, 50)
    x = 10 + w * (t - t_mesh)
    a = ax[0]
    a.plot(t, x, color='k', lw=4, alpha=0.25, label='the data: charge arriving on the strips')
    for dt, col, lab in ((0, CX, 'true t0 (mesh charge arrives)'), (-60, CY, 't0 one bin early (equally good line)')):
        a.plot([t_mesh + dt], [10 + w * dt], 'o', ms=12, color=col, label=lab)
        a.annotate('', xy=(t_mesh + dt, 9.2), xytext=(t_mesh + dt, 8.2), arrowprops=dict(arrowstyle='->', color=col, lw=2))
    a.plot(t_mesh - 60 + np.array([0, 60]), 10 + w * np.array([-60, 0]), ls=':', color=CY, lw=2)
    a.set_xlabel('arrival time on the strips [ns]'); a.set_ylabel('position across the strips [mm]')
    a.set_title('Move t0 earlier and p0 along the track:\nthe same line through the data', fontsize=13)
    a.legend(loc='upper left', fontsize=10, frameon=False); a.set_ylim(7.5, 22)
    # right: what changes = the depth each charge is assigned to -> the modelled diffusion width
    b = ax[1]
    u = np.linspace(0, T, 100)
    Dp, s0 = 0.023, 0.3
    b.plot(u, np.sqrt(s0 ** 2 + Dp ** 2 * u), color=CX, lw=3, label='t0 right: width the model expects')
    b.plot(u, np.sqrt(s0 ** 2 + Dp ** 2 * (u + 60)), color=CY, lw=3, ls='--', label='t0 one bin early: every charge "60 ns deeper"')
    b.set_xlabel('time since the mesh charge [ns]  (= depth / v)'); b.set_ylabel('modelled cloud width on the strips [mm]')
    b.set_title('What does change: the depth each charge is given\n(the only handle the fit has on t0)', fontsize=13)
    b.legend(fontsize=10, frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(f'{FIG}/t0_explainer.png'); plt.close(fig)


def fig_valley():
    f = f'{M}/ws_t0/valley_det3.npz'
    if not os.path.exists(f):
        return
    z = np.load(f, allow_pickle=True); d = z['d']; rows = list(z['rows'])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for vi, a in enumerate(ax):
        for r in [r for r in rows if r['view'] == vi][:5]:
            cv = (r['chi_valley'] - r['chi_valley'].min()) / r['dof']
            cf = (r['chi_fixed'] - r['chi_fixed'].min()) / r['dof']
            l, = a.plot(d, cv, lw=2)
            a.plot(d, cf, lw=1, ls=':', color=l.get_color())
        a.set_xlabel('t0 - fitted t0 [ns]'); a.set_title(f'det3 {"XY"[vi]}: solid = t0 and p0 slid together; dotted = p0 held', fontsize=12)
        for k in (-120, -60, 60, 120):
            a.axvline(k, color='0.85', lw=1, zorder=0)
        a.set_ylim(0, 10)
    ax[0].set_ylabel('chi2 increase per degree of freedom')
    fig.suptitle('Per event, t0 sits in a soft, one-sided valley with side minima one depth bin (60 ns) apart', fontsize=13)
    fig.tight_layout(); fig.savefig(f'{FIG}/t0_valley.png'); plt.close(fig)


def fig_latency():
    L = json.load(open(os.path.join(HERE, 'lead_edge.json')))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, det in enumerate(DETS):
        for j, p in enumerate('xy'):
            o = L[det][p]
            ax.errorbar(i + (j - 0.5) * 0.25, o['t_lead'], yerr=1.2533 * o['t_lead_rsig'] / np.sqrt(o['n']), fmt='o', ms=10,
                        color=(CX, CY)[j], label=f'{"XY"[j]} view' if i == 0 else None)
            ax.text(i + (j - 0.5) * 0.25, o['t_lead'] + 6, f'FEU{o["feu"]}', ha='center', fontsize=9, color='0.4')
    ax.set_xticks(range(5)); ax.set_xticklabels([f'det{d[-1]}' for d in DETS])
    ax.set_ylabel('earliest pulse, phase-corrected [ns]')
    ax.set_title('Model-free mesh-charge arrival: one latency for every chamber and view (±20 ns)', fontsize=13)
    ax.set_ylim(270, 370); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f'{FIG}/t0_latency_all_chambers.png'); plt.close(fig)


def fig_teeth():
    z = np.load(f'{M}/ws_t0/lead_edge.npz')
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for a, (det, p) in zip(ax, (('det3', 'x'), ('det7', 'y'))):
        d = pd.read_parquet(f'{M}/{det}/final_physmask/events.parquet')
        ok = d[p + '_ok'].astype(bool) & (d[p + '_ftst'] >= 0)
        t0c = (d[p + '_t0'] + 10 * d[p + '_ftst'])[ok]
        R = z[f'{det}_{p}']; tl = R[np.isfinite(R[:, 1]) & (R[:, 3] >= 3), 1]
        bins = np.arange(0, 640, 5)
        a.hist(t0c, bins=bins, color=CX, alpha=0.75, label='fitted t0 (free)')
        a.hist(tl, bins=bins, histtype='step', lw=2.5, color='k', weights=np.full(len(tl), len(t0c) / len(tl)),
               label='earliest pulse (no fit)')
        a.set_xlabel('phase-corrected time [ns]'); a.set_title(f'{det} {p.upper()}', fontsize=13)
        a.legend(frameon=False, fontsize=10)
    ax[0].set_ylabel('planes / 5 ns')
    fig.suptitle('The free t0 wanders early by up to 300 ns; the raw leading edge (no fit) sits at 300 ns on every chamber', fontsize=13)
    fig.tight_layout(); fig.savefig(f'{FIG}/t0_teeth.png'); plt.close(fig)


def fig_scan():
    f = os.path.join(HERE, 'scan_summary.json')
    if not os.path.exists(f):
        return
    S = json.load(open(f))
    L = json.load(open(os.path.join(HERE, 'lead_edge.json')))
    lo = min(L[d][p]['t_lead'] for d in DETS for p in 'xy'); hi = max(L[d][p]['t_lead'] for d in DETS for p in 'xy')
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.3))
    cols = dict(zip(DETS, ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3']))
    for det in DETS:
        if det not in S:
            continue
        v = json.load(open(f'{M}/{det}/calib_bundle/bundle.json'))['v_drift']
        for p, ls in (('x', '-'), ('y', '--')):
            rows = S[det][p]
            Cs = sorted(float(k) for k in rows if k != 'free')
            R = [rows[f'{c:g}'] for c in Cs]
            fr = rows['free']
            ax[0].plot(Cs, [100 * r['dchi2_rel'] for r in R], ls, color=cols[det], lw=2, label=f'{det} {p.upper()}')
            ax[1].plot(Cs, [100 * (r['ladder'] / v - 1) for r in R], ls, color=cols[det], lw=2)
            ax[2].plot(Cs, [100 * (r['pos_rsig_z'] / fr['pos_rsig_z'] - 1) for r in R], ls, color=cols[det], lw=2)
    ax[0].set_ylabel('total chi2 vs free t0 [%]')
    ax[1].set_ylabel('angle scale vs the drift velocity [%]')
    ax[2].set_ylabel('position resolution vs free t0 [%]')
    ax[0].set_title('Fit quality: free t0 always wins\n(it absorbs model error)', fontsize=12)
    ax[1].set_title('Angle scale = drift velocity\nat C ~ 300-330 ns', fontsize=12)
    ax[2].set_title('Position at the mesh: 10-30 % better\nwith t0 pinned (M3 reference included)', fontsize=12)
    for a in ax:
        a.set_xlabel('pinned latency C [ns]  (t0 = C - 10 ns x ftst)'); a.axhline(0, color='0.6', lw=1)
        a.axvspan(lo, hi, color='0.85', zorder=0, label=None)
    ax[2].set_ylim(-40, 70)
    ax[0].legend(fontsize=9, frameon=False, ncol=2, loc='upper left')
    ax[1].text(lo + 2, ax[1].get_ylim()[0] + 0.5, 'raw leading edge,\nall chambers', fontsize=9, color='0.35')
    fig.tight_layout(); fig.savefig(f'{FIG}/t0_scan.png'); plt.close(fig)


def fig_xy():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for a, det in zip(ax, ('det3', 'det7')):
        d = pd.read_parquet(f'{M}/{det}/final_physmask/events.parquet')
        ok = (d.x_ok.astype(bool) & d.y_ok.astype(bool) & d.x_quality_ok.astype(bool) & d.y_quality_ok.astype(bool)
              & (d.x_tan_theta.abs() < 0.45) & (d.y_tan_theta.abs() < 0.45) & (d.x_ftst >= 0) & (d.y_ftst >= 0))
        s = d[ok]
        raw = (s.x_t0 + 10 * s.x_ftst) - (s.y_t0 + 10 * s.y_ftst)
        ph = (s.x_t0 + s.x_q_u50 + 10 * s.x_ftst) - (s.y_t0 + s.y_q_u50 + 10 * s.y_ftst)
        bins = np.arange(-400, 401, 10)
        a.hist(raw, bins=bins, histtype='step', lw=2, color='0.5', label=f'free t0: x - y (median {np.median(raw):+.0f} ns)')
        a.hist(ph, bins=bins, histtype='stepfilled', alpha=0.6, color=CX, label=f'column middle t0+u50: x - y (median {np.median(ph):+.0f} ns)')
        a.axvline(0, color='k', lw=1)
        a.set_title(det, fontsize=13); a.set_xlim(-400, 400); a.set_xlabel('x - y time difference, phase-corrected [ns]'); a.legend(frameon=False, fontsize=10)
    ax[0].set_ylabel('events / 10 ns')
    fig.suptitle('Same electrons in both views: the column times agree at 0 ns (no FEU offset); the free t0s do not', fontsize=13)
    fig.tight_layout(); fig.savefig(f'{FIG}/t0_xy_coincidence.png'); plt.close(fig)


def fig_pin():
    """Held-out validation of the pinned latency (1320 events per chamber, production path from the bundle)."""
    fp, fa = os.path.join(HERE, 'pin_compare_lat.json'), os.path.join(HERE, 'heldout_compare_lat.json')
    if not (os.path.exists(fp) and os.path.exists(fa)):
        return
    P, A = json.load(open(fp)), json.load(open(fa))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    xs = np.arange(5)
    for j, p in enumerate('xy'):
        off = (j - 0.5) * 0.3
        pos = [100 * P[d][p]['delta'] / P[d][p]['base'] for d in DETS]
        pe = [100 * P[d][p]['err'] / P[d][p]['base'] for d in DETS]
        ax[0].bar(xs + off, pos, 0.28, yerr=pe, color=(CX, CY)[j], label=f'{p.upper()} view')
        s68 = [A[f'{d}|{p}|lat']['delta']['s68'] for d in DETS]
        ax[1].bar(xs + off, s68, 0.28, color=(CX, CY)[j])
        far = [A[f'{d}|{p}|lat']['delta']['far%'] for d in DETS]
        ax[2].bar(xs + off, far, 0.28, color=(CX, CY)[j])
    ax[0].set_ylabel('position resolution change [%]'); ax[0].set_title('Position at the mesh vs M3', fontsize=13)
    ax[1].set_ylabel('angle resolution change [deg]'); ax[1].set_title('Angle resolution (68 %)', fontsize=13)
    ax[1].set_ylim(-0.3, 0.3)
    ax[2].set_ylabel('tracks > 5 deg off [percentage points]'); ax[2].set_title('Far tail', fontsize=13)
    for a in ax:
        a.set_xticks(xs); a.set_xticklabels([f'det{d[-1]}' for d in DETS]); a.axhline(0, color='k', lw=1)
    ax[0].legend(frameon=False)
    fig.suptitle('t0 pinned at the measured trigger latency vs fitted freely (1320 held-out tracks per chamber)', fontsize=13)
    fig.tight_layout(); fig.savefig(f'{FIG}/t0_pin_heldout.png'); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_explainer(); fig_valley(); fig_latency(); fig_teeth(); fig_scan(); fig_xy(); fig_pin()
    print(sorted(os.listdir(FIG)))


if __name__ == '__main__':
    main()
