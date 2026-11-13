#!/usr/bin/env python3
"""make_figures.py -- every figure of the review, from the result files of papers/paper2/hits_daq and $WORK_DATA/paper2_hits.

    python scripts/make_figures.py            # writes figures/*.pdf (+ .png)
Figures whose inputs are missing are skipped with a note.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PAPERS)
import paper_style as ps  # noqa: E402

HD = os.path.join(PAPERS, 'paper2', 'hits_daq')
WD = os.environ.get('WORK_DATA', '/media/ucla/work') + '/paper2_hits'
FIG = os.path.join(HERE, '..', 'figures')
L_GAP, PITCH, V = 28.08, 0.78, 0.0358
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=200)
    plt.close(fig)
    print('  figures/' + name)


# ------------------------------------------------------------------ 1. schematic
def schematic():
    fig, ax = plt.subplots(1, 2, figsize=(6.1, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = ax[0]
    L, p, tan = 28.0, 0.78 * 3, 0.3            # exaggerated pitch for the drawing
    x0 = 0.35 * p
    z = np.linspace(0, L, 200)
    x = x0 + tan * z
    sig = np.sqrt(0.9 ** 2 + 0.12 ** 2 * z * 28)   # exaggerated spread
    a.fill_betweenx(z, x - sig, x + sig, color='#c7d4ea', lw=0, alpha=0.9, label='lateral spread')
    a.plot(x, z, color=ps.INK, lw=1.4, label='track')
    a.axhline(0, color='#333', lw=1.2)
    a.axhline(L, color='#333', lw=0.8, ls='--')
    a.text(-1.5 * p, L + 0.6, 'cathode', fontsize=7, color='#333')
    a.text(-1.5 * p, 0.6, 'mesh', fontsize=7, color='#333')
    xa, xb = x0, x0 + tan * L
    strips = np.arange(-3, 8) * p
    for s in strips:
        lo, hi = s - p / 2, s + p / 2
        if hi <= xa or lo >= xb:
            col, lab = '#d9534f', 'ghost'
        elif lo < xa or hi > xb:
            col, lab = '#e8a33d', 'partial'
        else:
            col, lab = '#3c8d5a', 'interior'
        a.add_patch(Rectangle((lo + 0.08, -2.6), p - 0.16, 1.6, color=col, lw=0))
    a.set(xlim=(-2.2 * p, 8.2 * p), ylim=(-3.2, L + 2.5), xlabel='strip coordinate', ylabel='drift depth z')
    a.set_xticks([]), a.set_yticks([0, L]), a.set_yticklabels(['0', 'L'])
    a.grid(False)
    a.text(0.02, 0.97, 'interior   partial   ghost', transform=a.transAxes, fontsize=7, va='top')
    for i, (c, xx) in enumerate(zip(('#3c8d5a', '#e8a33d', '#d9534f'), (0.02, 0.17, 0.30))):
        a.add_patch(Rectangle((xx, 0.86), 0.025, 0.04, transform=a.transAxes, color=c, lw=0))
    a.set_title('(a) one strip, several depths', loc='left')
    # (b) the ladder
    b = ax[1]
    N = 8
    i = np.arange(N)
    t = i * 1.0
    b.plot(i, t, 'o', color='#3c8d5a', ms=4, label='interior strips: offset only')
    b.plot([-1, N], [-1, N], color=ps.INK, lw=0.9, ls=':', label='true ladder')
    b.plot([0], [0.45], 'o', color='#e8a33d', ms=4)
    b.plot([N - 1], [N - 1 - 0.3], 'o', color='#e8a33d', ms=4, label='partial end strips')
    b.plot([-1], [0.15], 's', color='#d9534f', ms=4)
    b.plot([N], [N - 1 + 0.35], 's', color='#d9534f', ms=4, label='ghost strips')
    xs = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    ys = np.array([0.15, 0.45, 1, 2, 3, 4, 5, 6, 6.7, 7.35])
    c = np.polyfit(xs, ys, 1)
    b.plot([-1.3, 8.3], np.polyval(c, [-1.3, 8.3]), color='#b2182b', lw=1.2, label='fitted ladder (flatter)')
    b.set(xlabel='strip', ylabel='hit time', xticks=[], yticks=[])
    b.grid(False)
    b.legend(fontsize=6.3, loc='upper left')
    b.set_title('(b) the ends rotate the fit', loc='left')
    fig.tight_layout()
    save(fig, 'schematic')


# ------------------------------------------------------------------ 2. universal curve
def universal():
    U = load(os.path.join(HD, 'results', 'universal.json'))
    T = load(os.path.join(HD, 'results', 'toy_arms.json'))
    if not U:
        return print('  universal: missing')
    fig, ax = plt.subplots(1, 2, figsize=(6.1, 3.0))
    series = [('data|run1|x|u_dream', 'chamber A, X view (data vs M3)', dict(color=ps.DET_COLOR['det3'], marker='o', ls='none', ms=3.5, zorder=6)),
              ('data|run1|y|u_dream', 'chamber A, Y view', dict(color=ps.DET_COLOR['det3'], marker='o', mfc='none', ls='none', ms=3.5, zorder=6)),
              ('resim|x|base|u_dream', 'model, X', dict(color='#0b3d91', lw=1.0, alpha=0.6)),
              ('resim|x|metal|u_dream', 'model, metallic footprint', dict(color='#b2182b', lw=1.0)),
              ('resim|x|dp0|u_dream', 'model, no diffusion', dict(color='#E69F00', lw=1.0)),
              ('resim|x|point|u_dream', 'geometry only (strip width)', dict(color='#3c8d5a', lw=1.0)),
              ('resim|y|base|u_dream', 'model, Y (with RC)', dict(color='#0b3d91', lw=1.0, ls='--', alpha=0.6))]
    for key, lab, st in series:
        rows = [r for r in U['curves'].get(key, []) if r['n'] >= 60 and r['tan'] < 0.45]
        if not rows:
            continue
        t = np.array([r['tan'] for r in rows])
        y = np.array([r['ratio'] for r in rows])
        e = np.array([r['ratio_err'] for r in rows])
        if 'marker' in st:
            ax[0].errorbar(t, y, e, label=lab, **st)
            ax[1].errorbar(L_GAP * t / PITCH, y - 1, e, label=lab, **st)
        else:
            ax[0].plot(t, y, label=lab, **st)
            ax[1].plot(L_GAP * t / PITCH, y - 1, label=lab, **st)
    Nn = np.logspace(0.2, 1.3, 50)
    for C, lab in ((2.3, None), (7.0, None), (11.7, None)):
        ax[1].plot(Nn, C / Nn ** 2, color='#999', lw=0.6, ls=':')
    ax[1].text(10.5, 2.3 / 110, r'$C=2.3$', fontsize=6.5, color='#666')
    ax[1].text(10.5, 7.0 / 110, r'$C=7$', fontsize=6.5, color='#666')
    ax[1].text(10.5, 11.7 / 110, r'$C=12$', fontsize=6.5, color='#666')
    for l in U['lit']:
        meas = 'LTspice' not in l['label'] and 'MC' not in l['label']
        ax[1].plot(l['N'], l['ratio'] - 1, marker='*' if meas else 'D', color='#D55E00', ms=7 if meas else 4.5, ls='none', mec='k', mew=0.4,
                   label='published strip µTPCs (measured)' if l['label'].startswith('Bortfeldt 2014 FSM 10° (meas.)') else ('published (simulated)' if l['label'] == 'Bortfeldt 2014 FSM 10° (LTspice)' else None))
    # Scholer 2021 NSW test beam at 28 deg: peak at the true angle (ratio - 1 = 0, off the log axis): drawn as an upper
    # limit at the bottom of the axis, with the toy's 1.10 at the same N (NSW-like, VMM peak time) for comparison.
    n_sch = 5.0 * np.tan(np.radians(28.0)) / 0.4
    ax[1].annotate('', xy=(n_sch, 4.6e-3), xytext=(n_sch, 1.2e-2), arrowprops=dict(arrowstyle='->', color='#D55E00', lw=0.9))
    ax[1].plot(n_sch, 1.2e-2, marker='*', mfc='white', mec='#D55E00', mew=0.9, ms=7, ls='none',
               label='Scholer 2021 test beam 28° (no shift)')
    ax[1].plot(n_sch, 0.10, marker='x', color='#D55E00', ms=5, ls='none', label='toy at the same point (1.10)')
    ax[0].set(xlabel=r'$|\tan\theta|$ (reference)', ylabel=r'$\tan\theta_{\rm hits}\,/\,\tan\theta_{\rm true}$', ylim=(0.95, 2.6), xlim=(0.03, 0.42))
    ax[1].set(xscale='log', yscale='log', xlabel=r'$N = L\tan\theta/p$ (strips in the column)', ylabel='ratio $-$ 1', ylim=(4e-3, 120), xlim=(1.2, 22))
    ax[0].legend(fontsize=6.2, loc='upper right')
    h1, l1 = ax[1].get_legend_handles_labels()
    keep = [i for i, t in enumerate(l1) if not t.startswith(('model', 'geometry'))]   # lines: see panel (a)
    ax[1].legend([h1[i] for i in keep], [l1[i] for i in keep], fontsize=5.8, loc='upper right', framealpha=0.9)
    ax[0].set_title('(a) compression against angle', loc='left')
    ax[1].set_title(r'(b) against column length: ratio $-$ 1 $\simeq C/N^2$', loc='left')
    fig.tight_layout()
    save(fig, 'universal')


# ------------------------------------------------------------------ 3. depth-resolved sharing
def depth():
    D = load(os.path.join(WD, 'depth', 'full', 'summary.json'))
    if not D:
        return print('  depth: missing')
    g = np.array(D['grid'])
    fig, ax = plt.subplots(2, 3, figsize=(6.1, 4.0), sharex=True)
    arms = [('data', 'data', dict(color='k', lw=1.8, zorder=5)), ('pass', 'model (σ0 0.30 mm, D 0.023)', dict(color='#0b3d91', lw=1.0)),
            ('Dp20', 'model, D 0.020', dict(color='#56B4E9', lw=1.0)), ('nodiff', 'no diffusion', dict(color='#E69F00', lw=1.0)),
            ('metal', 'metallic footprint (0.07 mm)', dict(color='#b2182b', lw=1.0)), ('pureK', 'coupling k1 0.13, σ0 0.03', dict(color='#3c8d5a', lw=1.0, ls='--')),
            ('free', 'σ0 0.25 + k1 0.06', dict(color='#CC79A7', lw=1.0, ls='--'))]
    for iv, view in enumerate('xy'):
        C = D['curves'].get(f'{view}|tight|phall')
        if not C:
            continue
        for name, lab, st in arms:
            if name not in C:
                continue
            o = C[name]
            ax[iv, 0].plot(g, o['r1'], label=lab if iv == 0 else None, **st)
            ax[iv, 1].plot(g, o['r2'], **st)
            ax[iv, 2].plot(g, o['sigma'], **st)
        d = C['data']
        for j, (k, ek) in enumerate((('r1', 'r1_err'), ('r2', 'r2_err'))):
            y, e = np.array(d[k]), np.array(d[ek])
            ax[iv, j].fill_between(g, y - e, y + e, color='k', alpha=0.15, lw=0)
        ax[iv, 0].set(ylim=(0, 1.3 if view == 'x' else 2.2), ylabel=f'{view.upper()}: ' + r'$(W_{-1}+W_{+1})/W_0$')
        ax[iv, 1].set(ylim=(0, 0.3 if view == 'x' else 0.9), ylabel=r'$(W_{-2}+W_{+2})/W_0$')
        ax[iv, 2].set(ylim=(0, 0.9 if view == 'x' else 1.3), ylabel=r'$\sigma_{\rm eff}$ of 5 strips [mm]')
        ax[iv, 0].text(0.03, 0.9, f'n = {C["n"]} planes', transform=ax[iv, 0].transAxes, fontsize=6.5)
    for a in ax[1]:
        a.set_xlabel(r'$t - t_0$ [ns]')
    ax[0, 0].legend(fontsize=5.6, loc='upper left', bbox_to_anchor=(0.0, 0.88))
    ax[0, 0].set_title('(a) first neighbours', loc='left')
    ax[0, 1].set_title('(b) second neighbours', loc='left')
    ax[0, 2].set_title('(c) lateral width', loc='left')
    fig.tight_layout()
    save(fig, 'depth_sharing')


# ------------------------------------------------------------------ 4. phase: data vs models, and the closure
def phase():
    D = load(os.path.join(WD, 'depth', 'full', 'summary.json'))
    P = load(os.path.join(WD, 'phasecl', 'full', 'summary.json'))
    if not D:
        return print('  phase: missing')
    g = np.array(D['grid'])
    i0 = int(np.argmin(np.abs(g - 0)))
    fig, ax = plt.subplots(1, 2 if P else 1, figsize=(6.1, 2.8), squeeze=False)
    a = ax[0, 0]
    pb = D['phase_bins']
    xc = [0.5 * (pb[i] + pb[i + 1]) for i in range(3)]
    for name, lab, st in (('data', 'data', dict(color='k', marker='o', ms=4, lw=1.6)), ('pass', 'Gaussian footprint 0.30 mm', dict(color='#0b3d91', marker='s', ms=3)),
                          ('free', 'σ0 0.25 mm + k1 0.06', dict(color='#CC79A7', marker='^', ms=3)), ('pureK', 'coupling k1 0.13, σ0 0.03 mm', dict(color='#3c8d5a', marker='v', ms=3)),
                          ('metal', 'footprint 0.07 mm', dict(color='#b2182b', marker='d', ms=3))):
        for ls, col, view in (('-', 'near', 'x'), ('--', 'far', 'x')):
            y = [D['curves'][f'{view}|all|ph{i}'][name][f'r{col}'][i0] for i in range(3)]
            a.plot(xc, y, ls=ls, label=f'{lab}' if col == 'near' else None, **st)
    a.set(xlabel='|position in strip| (pitch)', ylabel='neighbour / centre at $t_0$', ylim=(0, 1.0), title='(a) X: near (solid) and far (dashed) neighbour')
    a.legend(fontsize=5.8, loc='upper left')
    if P:
        b = ax[0, 1]
        pb2 = np.array(P['phase_bins'])
        xc2 = 0.5 * (pb2[:-1] + pb2[1:])
        for key, lab, st in (('x|data|pass', 'data (pass p0)', dict(color='k', marker='o', ms=4, lw=1.6)),
                             ('x|noK_white|refit', 'truth: Gaussian 0.32 mm, refit p0', dict(color='#0b3d91', marker='s', ms=3)),
                             ('x|pureK_white|refit', 'truth: pure coupling 0.13, refit p0', dict(color='#3c8d5a', marker='v', ms=3)),
                             ('x|pureK_white|true', 'pure coupling, true p0', dict(color='#3c8d5a', marker='v', ms=3, ls=':', alpha=0.7))):
            rec = P['groups'].get(key)
            if not rec:
                continue
            for col, ls in (('near', st.get('ls', '-')), ('far', '--')):
                y = [bb[col] if bb else np.nan for bb in rec['bins']]
                e = [bb[col + '_err'] if bb else np.nan for bb in rec['bins']]
                st2 = {k: v for k, v in st.items() if k != 'ls'}
                b.errorbar(xc2, y, e, ls=ls, label=lab if col == 'near' else None, **st2)
        b.set(xlabel='|position in strip| (pitch)', ylabel='neighbour / centre, early window', ylim=(0, 1.0), title='(b) closure with known truth and refitted $p_0$')
        b.legend(fontsize=5.8, loc='upper left')
    fig.tight_layout()
    save(fig, 'phase')


# ------------------------------------------------------------------ 5. remedies
def remedies():
    R1, R2 = load(os.path.join(WD, 'remedies', 'run1.json')), load(os.path.join(WD, 'remedies', 'run2.json'))
    if not R1:
        return print('  remedies: missing')
    tiers = [('dream', 'leading edge (V0)'), ('dream_rob', '+ robust line'), ('dream_e17', '+ edge cut 1/6'), ('dream_shift', '+ edge shift'),
             ('dream_sub15_rob', '+ 15 % neighbour subtraction'), ('dream_sub30', '30 % subtraction'), ('cw_rob', 'charge-weighted time, robust'),
             ('cw_sub15_rob', 'charge-weighted + 15 %'), ('cw_sub30', 'charge-weighted + 30 %'), ('ideal_sub15_shift', 'template time + 15 % + shift')]
    bins = ('0.05-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.5')
    fig, ax = plt.subplots(1, 2, figsize=(6.1, 3.0), sharey=False)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(tiers)))
    for iv, view in enumerate('xy'):
        V = R1['views'][view]
        xs = np.arange(len(bins))
        for it, (k, lab) in enumerate(tiers):
            t = V['tiers'].get(k)
            if not t:
                continue
            y = [t['naive'][b]['steep'] if t['naive'][b]['steep'] is not None else np.nan for b in bins]
            ax[iv].plot(xs, y, marker='o', ms=3, color=cols[it], label=lab if iv == 0 else None, lw=1.0)
        W = [V['W'][b]['steep'] for b in bins]
        ax[iv].plot(xs, W, marker='s', ms=3.5, color='k', lw=1.6, label='waveform forward fit' if iv == 0 else None)
        ax[iv].axhline(0, color='#888', lw=0.6)
        ax[iv].set(xticks=xs, xticklabels=['0.05–0.1', '0.1–0.2', '0.2–0.3', '0.3–0.5'], xlabel=r'$|\tan\theta|$ bin', ylabel='median steepness bias [°]' if iv == 0 else '',
                   title=f'({"ab"[iv]}) {view.upper()} view, Run 1 ({V["n"]} planes)')
    ax[0].legend(fontsize=5.6, loc='upper right')
    fig.tight_layout()
    save(fig, 'remedies_bias')
    # s68 table-like bar chart
    fig, ax = plt.subplots(1, 1, figsize=(6.1, 2.8))
    names = [lab for _, lab in tiers] + ['forward fit on hit lists (H2c)', 'waveform forward fit']
    for iv, (view, off, col) in enumerate((('x', -0.2, ps.DET_COLOR['det3']), ('y', 0.2, '#56B4E9'))):
        V = R1['views'][view]
        vals = [V['tiers'][k]['H0']['all']['s68'] if k in V['tiers'] else np.nan for k, _ in tiers]
        vals += [1.20 if view == 'x' else 1.28, V['W']['all']['s68']]
        ax.bar(np.arange(len(names)) + off, vals, width=0.4, color=col, label=f'{view.upper()} view')
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=6.3)
    ax.set_ylabel('angle resolution s68 [°]\n(after a global scale calibration)')
    ax.legend()
    fig.tight_layout()
    save(fig, 'remedies_s68')


# ------------------------------------------------------------------ 6. pitch scan
def pitch():
    U = load(os.path.join(HD, 'results', 'universal.json'))
    T = load(os.path.join(HD, 'results', 'toy_pitch.json'))
    if not U:
        return print('  pitch: missing')
    fig, ax = plt.subplots(1, 2, figsize=(6.1, 2.8))
    ks = (1, 2, 3, 4, 6)
    for arm, lab, st in (('data', 'data (vs M3)', dict(color='k', marker='o', ms=4)), ('base', 'model', dict(color='#0b3d91', marker='s', ms=3)),
                         ('metal', 'metallic footprint', dict(color='#b2182b', marker='d', ms=3)), ('point', 'geometry only', dict(color='#3c8d5a', marker='^', ms=3))):
        ps_ = [PITCH * k for k in ks]
        C = [U['fits'][f'gang|x|{arm}|k{k}']['a'] * (L_GAP / (PITCH * k)) ** 2 if f'gang|x|{arm}|k{k}' in U['fits'] else np.nan for k in ks]
        a_ = [U['fits'][f'gang|x|{arm}|k{k}']['a'] if f'gang|x|{arm}|k{k}' in U['fits'] else np.nan for k in ks]
        ax[0].plot(ps_, a_, label=lab, **st)
        ax[1].plot(ps_, C, label=lab, **st)
    if T:
        for pref, lab, st in (('p', 'toy, σ0 0.30 + D 0.021', dict(color='#0b3d91', ls=':', lw=1.2)), ('p', 'toy, geometry only', dict(color='#3c8d5a', ls=':', lw=1.2))):
            xs, ys, cs = [], [], []
            for name, cfg in T['configs'].items():
                if (lab.endswith('only')) != name.endswith('_point'):
                    continue
                f = cfg['fits'].get('dream_plain')
                if f:
                    xs.append(cfg['params']['pitch']), ys.append(f['a']), cs.append(f['C'])
            o = np.argsort(xs)
            ax[0].plot(np.array(xs)[o], np.array(ys)[o], label=lab, **st)
            ax[1].plot(np.array(xs)[o], np.array(cs)[o], label=lab, **st)
    ax[0].set(xlabel='readout pitch [mm]', ylabel=r'end term $a$ in $\Delta\tan = a/\tan\theta$', title='(a) end term against pitch')
    ax[1].set(xlabel='readout pitch [mm]', ylabel=r'$C = a\,(L/p)^2$', title=r'(b) in units of $1/N^2$')
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, 'pitch_scan')


# ------------------------------------------------------------------ 7. toy: C against sigma/p and detectors
def toy():
    A = load(os.path.join(HD, 'results', 'toy_arms.json'))
    Dt = load(os.path.join(HD, 'results', 'toy_detectors.json'))
    Es = load(os.path.join(HD, 'results', 'toy_estimators.json'))
    if not A:
        return print('  toy: missing')
    fig, ax = plt.subplots(1, 2, figsize=(6.1, 2.9))
    # (a) C for the arms, three estimators x corrections
    names = list(A['configs'].keys())
    keys = [('dream_plain', 'leading edge'), ('cw_plain', 'charge-weighted'), ('ideal_plain', 'template time'), ('dream_shift', 'leading edge + edge shift')]
    xs = np.arange(len(names))
    for j, (k, lab) in enumerate(keys):
        y = [A['configs'][n]['fits'][k]['C'] if A['configs'][n]['fits'].get(k) else np.nan for n in names]
        ax[0].bar(xs + (j - 1.5) * 0.2, y, width=0.2, label=lab)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(names, rotation=25, ha='right', fontsize=6.5)
    ax[0].set(ylabel=r'$C$ (ratio $-$ 1 $= C/N^2$)', title='(a) toy: strip-end coefficient')
    ax[0].legend(fontsize=6)
    if Dt:
        for name, cfg in Dt['configs'].items():
            pr = cfg['params']
            sT = pr['D_T'] * np.sqrt(pr['L'] / 10) / pr['pitch']
            for k, mk in (('fd_plain', 'o'), ('dream_plain', 'o'), ('peak_plain', 's'), ('cw_plain', '^')):
                f = cfg['fits'].get(k)
                if f:
                    ax[1].plot(sT, f['C'], mk, ms=4, label=f'{name}: {k.split("_")[0]}')
    ax[1].set(xlabel=r'$\sigma_T(L)/p$ (diffusion at full drift over pitch)', ylabel='$C$', title='(b) toy: other geometries')
    ax[1].legend(fontsize=5)
    fig.tight_layout()
    save(fig, 'toy_C')


# ------------------------------------------------------------------ 8. ghosts
def ghosts():
    G = load(os.path.join(HD, 'results', 'ghost_census_sat_det3.json'))
    if not G:
        return print('  ghosts: missing')
    fig, ax = plt.subplots(1, 2, figsize=(6.1, 2.6), sharey=True)
    for iv, view in enumerate('xy'):
        for arm, lab, st in (('data', 'data', dict(color='k', marker='o', ms=4)), ('model+noise', 'model + noise', dict(color='#0b3d91', marker='s', ms=3))):
            rows = [o for k, o in G.items() if o['view'] == view and o['arm'] == arm and o['tan_hi'] <= 0.4]
            xc = [0.5 * (o['tan_lo'] + o['tan_hi']) for o in rows]
            ax[iv].plot(xc, [o['ghost_mesh'] for o in rows], ls='-', label=f'{lab}, mesh end' if iv == 0 else None, **st)
            ax[iv].plot(xc, [o['ghost_cathode'] for o in rows], ls='--', label=f'{lab}, cathode end' if iv == 0 else None, **st)
        ax[iv].set(xlabel=r'$|\tan\theta|$', title=f'({"ab"[iv]}) {view.upper()} view')
    ax[0].set_ylabel('ghost strips per plane')
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, 'ghosts')


# ------------------------------------------------------------------ 9. the four chambers
def chambers():
    J = load(os.path.join(HD, 'results', 'chambers_C.json'))
    if not J:
        return print('  chambers: missing')
    dets = [d for d in ('det4', 'det3', 'det2', 'det7') if d in J['chambers']]
    fig, ax = plt.subplots(1, 3, figsize=(6.1, 2.7))
    xs = np.arange(len(dets))
    names = [J['chambers'][d]['name'] for d in dets]
    for iv, view in enumerate('xy'):
        a = ax[iv]
        for arm, lab, st in (('data', 'data', dict(color='k', marker='o', ms=4.5, ls='none', zorder=5)),
                             ('model', 'model', dict(color='#0b3d91', marker='s', ms=3.5, ls='none')),
                             ('metallic', 'metallic footprint', dict(color='#b2182b', marker='d', ms=3.5, ls='none')),
                             ('rc0', 'model, RC off', dict(color='#CC79A7', marker='^', ms=3.5, ls='none')),
                             ('point', 'geometry only', dict(color='#3c8d5a', marker='v', ms=3.5, ls='none'))):
            y, e = [], []
            for d in dets:
                f = J['chambers'][d]['views'].get(view, {}).get('arms', {}).get(arm)
                y.append(f['C'] if f else np.nan), e.append(f.get('C_err', 0) if f else 0)
            if np.all(np.isnan(y)):
                continue
            a.errorbar(xs, y, e, label=lab if iv == 0 else None, **st)
        if view == 'x':
            a.plot(xs, [J['chambers'][d]['toy_law_C_X'] for d in dets], ls=':', color='#0b3d91', label='toy law, chamber $D_p$')
        a.set(xticks=xs, xticklabels=names, xlabel='chamber', ylabel='$C$' if iv == 0 else '', title=f'({"ab"[iv]}) {view.upper()} view', ylim=(0, 13))
    ax[0].legend(fontsize=5.6, loc='upper left', ncol=2)
    # (c) RC increment against D_RC
    c = ax[2]
    for arm, lab, st in (('data', 'data', dict(color='k', marker='o', ms=4.5, ls='none', zorder=5)), ('model', 'model', dict(color='#0b3d91', marker='s', ms=3.5, ls='none'))):
        xr, yr, er = [], [], []
        for d in dets:
            ch = J['chambers'][d]
            fy = ch['views'].get('y', {}).get('arms', {}).get(arm)
            f0 = ch['views'].get('y', {}).get('arms', {}).get('rc0')
            if fy and f0:
                xr.append(ch['hyper']['rc_D_y'] * 1e4), yr.append(fy['C'] - f0['C']), er.append(fy.get('C_err', 0))
        c.errorbar(xr, yr, er, label=lab, **st)
    c.set(xlabel=r'$D_{RC}$ [$10^{-4}$ mm$^2$/ns]', ylabel=r'$C_Y - C_Y$(RC off)', title='(c) the RC increment', xlim=(0, 8))
    c.legend(fontsize=6)
    fig.tight_layout()
    save(fig, 'chambers')


if __name__ == '__main__':
    ps.use()
    for f in (schematic, universal, depth, phase, remedies, pitch, toy, ghosts, chambers):
        try:
            f()
        except Exception as e:  # keep going; report
            print(f'  {f.__name__}: FAILED {e!r}')
