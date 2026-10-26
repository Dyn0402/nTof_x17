#!/usr/bin/env python3
"""make_report.py -- report.html for the det3 multi-particle study, generated from the outputs of legs.py, m3match.py,
choose.py and bench.py (and the overlay arms). Figures in report/figures/, referenced by relative links.

    ../../.venv/bin/python make_report.py     -> ~/x17/cosmic_bench/satstudy/multiparticle/report/report.html
"""
import html, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mplib as M  # noqa: E402

RD = os.path.join(M.OUT, 'report'); FD = os.path.join(RD, 'figures')
os.makedirs(FD, exist_ok=True)
DESC = ("What the second particles in det3's multi-particle cosmic events are, a reference-free rule to pick the "
        "triggering track, and the bench two-track benchmark for the X17 pair reconstruction.")
C1, C2, C3, C4 ='#2a78d6', '#d65f2a', '#6b6b6b', '#2aa36b'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})


def J(n):
    p = os.path.join(M.OUT, n)
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x, d=0):
    return f'{100 * x:.{d}f} %'


# ---------------------------------------------------------------- estimates (labelled as such in the report)
def estimates(n_trig, t_run):
    K2 = 0.1535                       # K/2 [MeV cm^2 / g] (PDG)
    ZA_AR, RHO_AR, GAP_CM = 18 / 39.95, 1.66e-3, 3.0
    flux = 1.0 / 60                   # cm^-2 s^-1, PDG rule of thumb for a horizontal detector at sea level
    area = 39.9 * 36.0                # det3 active area, cm^2 (common/mx17_active_area.py)
    r_det = flux * area
    window = 2.8e-6                   # a particle is visible if it arrives -840 ns .. +1920 ns around the trigger
    p_acc = r_det * window
    gas = K2 * ZA_AR * RHO_AR * GAP_CM            # N(>T) * T in the gas, MeV
    return dict(r_det=r_det, p_acc=p_acc, n_acc=p_acc * 6451, trig_rate=n_trig / t_run, n_trig=n_trig, t_run=t_run,
                gas_coef=gas, gas_50keV=gas / 0.05, gas_30keV=gas / 0.03,
                mat_per_gcm2_1MeV=K2 * 0.5 / 1.0, mat_per_gcm2_05MeV=K2 * 0.5 / 0.5,
                acc_window_frac=300 / 2760)


# ---------------------------------------------------------------- figures
def figs(E, Lg, Mx, ref, choose, bench, m3m):
    F = E[E.full]
    out = {}
    # 1. timing
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    b = np.arange(-600, 601, 30)
    ax.hist(np.clip(F.dt0, -600, 600), bins=b, color=C1, alpha=.85, label=f'second − M3 particle, mean of x and y ({len(F)})')
    w = (b[1] - b[0]) / 2760 * len(F)
    ax.axhline(w, color=C2, ls='--', label='independent particle: flat over the ~2.8 µs visible window')
    ax.set_xlabel('Δt0 [ns]'); ax.set_ylabel('events'); ax.legend(fontsize=8, frameon=False)
    ax.set_title('The second particle is in time with the triggering muon', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FD, 'dt0.png')); plt.close(fig)
    # 2. vertex height
    sel = (Mx.sepx.abs() > 15) & (Mx.sepy.abs() > 15) & (Mx.sepx.abs() < 250) & (Mx.sepy.abs() < 250)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.5))
    b = np.linspace(0, 1800, 37)
    d = F[F.dca < 20]
    axs[0].hist(np.clip(d.zv, 0, 1799), bins=b, color=C1, alpha=.85, label=f'data, DCA < 20 mm ({len(d)})')
    mz = Mx[sel & (Mx.dca < 20)].zv
    axs[0].hist(np.clip(mz, 0, 1799), bins=b, weights=np.full(len(mz), len(d) / len(mz)), histtype='step', color=C2, lw=1.6,
                label='two unrelated muons (mixed), same norm.')
    for z, nm in ((1302, 'M3'), (1185, 'M3'), (715, 'det3'), (144, 'M3'), (24, 'M3')):
        axs[0].axvline(z, color=C3, lw=.8, ls=':')
        axs[0].text(z, axs[0].get_ylim()[1] * .95, nm, rotation=90, fontsize=7, ha='right', va='top', color=C3)
    axs[0].axvspan(1350, 1500, color=C3, alpha=.08); axs[0].text(1425, 0.5, 'top scint.', fontsize=7, ha='center', rotation=90)
    axs[0].set_xlabel('height of closest approach Z_v [mm]'); axs[0].set_ylabel('events'); axs[0].legend(fontsize=7.5, frameon=False)
    b2 = np.linspace(0, 200, 41)
    axs[1].hist(np.clip(F.dca, 0, 199), bins=b2, color=C1, alpha=.85, density=True, label='data (2x2 events)')
    axs[1].hist(np.clip(Mx[sel].dca, 0, 199), bins=b2, density=True, histtype='step', color=C2, lw=1.6, label='two unrelated muons')
    axs[1].set_xlabel('closest approach to the M3 ray [mm]'); axs[1].set_ylabel('density'); axs[1].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FD, 'vertex.png')); plt.close(fig)
    # 3. angles and charge
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    b = np.linspace(0, 1.2, 25)
    axs[0].hist(np.clip(np.hypot(F.o_tx, F.o_ty), 0, 1.19), bins=b, density=True, color=C1, alpha=.85, label='second particle')
    axs[0].hist(np.hypot(F.tan_x, F.tan_y), bins=b, density=True, histtype='step', color=C2, lw=1.6, label='triggering muon (M3)')
    axs[0].set_xlabel('|tan| (3-D, from vertical)'); axs[0].legend(fontsize=8, frameon=False)
    b = np.linspace(0, 60, 31)
    axs[1].hist(np.clip(F.open, 0, 59.9), bins=b, density=True, color=C1, alpha=.85, label='data')
    axs[1].hist(np.clip(Mx[sel].open, 0, 59.9), bins=b, density=True, histtype='step', color=C2, lw=1.6, label='two unrelated muons')
    axs[1].set_xlabel('opening angle to the M3 ray [deg]'); axs[1].legend(fontsize=8, frameon=False)
    b = np.linspace(0, 5, 41)
    axs[2].hist(np.clip(F.o_qmip, 0, 4.99), bins=b, density=True, color=C1, alpha=.85, label='second particle')
    axs[2].hist(np.clip(F.m_qmip, 0, 4.99), bins=b, density=True, histtype='step', color=C2, lw=1.6, label='M3 particle, same events')
    axs[2].set_xlabel('charge per unit path [clean-single median]'); axs[2].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FD, 'angle_charge.png')); plt.close(fig)
    # 4. M3 planes
    if m3m:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ks = list(m3m['planes']); x = np.arange(len(ks))
        for i, (nm, c, lab) in enumerate((('sec', C1, 'on the second particle\'s line'), ('mir', C2, 'mirror null'),
                                          ('swap', C3, 'swap null'))):
            ax.bar(x + (i - 1) * .27, [100 * m3m['planes'][k][nm] for k in ks], .27, color=c, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([f'{k}\nZ={int(m3m["planes"][k]["z"])}' for k in ks], fontsize=8)
        ax.set_ylabel(f'% with an extra M3 cluster ≤ {m3m["match_mm"]:.0f} mm'); ax.legend(fontsize=8, frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(FD, 'm3planes.png')); plt.close(fig)
    # 5. choose rules
    if choose:
        rs = choose['rules']; ks = list(rs)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        y = np.arange(len(ks))
        acc = np.array([rs[k]['acc'] for k in ks]); lo = np.array([rs[k]['ci68'][0] for k in ks]); hi = np.array([rs[k]['ci68'][1] for k in ks])
        cols = [C2 if k == 'production_rank' else (C1 if k.startswith('logit') else C3) for k in ks]
        ax.barh(y, 100 * acc, color=cols, xerr=[100 * (acc - lo), 100 * (hi - acc)], ecolor='k', capsize=2)
        ax.axvline(100 * choose['chance'], color='k', ls=':', lw=1); ax.text(100 * choose['chance'], len(ks) - .4, ' chance', fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels([k.replace('_', ' ') + (' (held-out)' if k.startswith('logit') else '') for k in ks], fontsize=8)
        ax.set_xlabel('% of views where the rule picks the M3 particle'); ax.invert_yaxis()
        fig.tight_layout(); fig.savefig(os.path.join(FD, 'choose.png')); plt.close(fig)
    # 6. benchmark
    if bench:
        fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))
        sty = {'prod (10-26)': (C3, 'o', 'production, 10-26 run'), 'off': (C2, 's', 'production (re-run)'),
               'tt': (C1, 'D', 'joint two-track fit (WFT_TWO_TRACK_FIT=1)')}
        for arm, v in bench['arms'].items():
            if not v:
                continue
            c, mk, lab = sty[arm]
            mid = [0.5 * (b['lo'] + min(b['hi'], 120)) for b in v['bins']]
            axs[0].plot(mid, [100 * b['both_found'] for b in v['bins']], marker=mk, color=c, label=lab, ms=4)
            ok = [b for b in v['bins'] if 'sep_res' in b]
            mid2 = [0.5 * (b['lo'] + min(b['hi'], 120)) for b in ok]
            axs[1].plot(mid2, [b['sep_res'] for b in ok], marker=mk, color=c, ms=4, label=lab)
            axs[2].plot(mid2, [b['open_res'] for b in ok], marker=mk, color=c, ms=4, label=lab)
        r = bench['real']
        mid = [0.5 * (b['lo'] + min(b['hi'], 160)) for b in r['bins']]
        axs[1].plot(mid, [b['res'] * np.sqrt(2) for b in r['bins']], marker='^', color=C4, ms=5, ls='--',
                    label='real events: √2 × M3-leg residual')
        for a_, yl in zip(axs, ('both tracks found [%]', 'separation resolution per view [mm]', 'opening-angle resolution [deg]')):
            a_.set_xlabel('separation at mid-gap, min of x / y [mm]'); a_.set_ylabel(yl); a_.set_xscale('symlog', linthresh=20)
        axs[0].legend(fontsize=7, frameon=False); axs[1].legend(fontsize=7, frameon=False)
        axs[1].set_ylim(0, None); axs[2].set_ylim(0, None)
        fig.tight_layout(); fig.savefig(os.path.join(FD, 'bench.png')); plt.close(fig)
        # real separation spectrum
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        sb = bench['sep_bins']; h = r['sep_hist']
        ax.bar(range(len(h)), h, color=C1)
        ax.set_xticks(range(len(h))); ax.set_xticklabels([f'{sb[i]}–{sb[i + 1]}' for i in range(len(h))], fontsize=8)
        ax.set_xlabel('min(|sep x|, |sep y|) [mm]'); ax.set_ylabel('real 2x2 events')
        fig.tight_layout(); fig.savefig(os.path.join(FD, 'realsep.png')); plt.close(fig)


def table(rows, head):
    h = ''.join(f'<th>{html.escape(str(c))}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def fig(name, cap):
    return f'<figure><img src="figures/{name}" alt="{html.escape(cap)}"><figcaption>{cap}</figcaption></figure>'


def main():
    E = pd.read_parquet(os.path.join(M.OUT, 'events.parquet')); Lg = pd.read_parquet(os.path.join(M.OUT, 'legs.parquet'))
    Mx = pd.read_parquet(os.path.join(M.OUT, 'mixed.parquet'))
    ref, choose, bench, m3m = J('singles_ref.json'), J('choose.json'), J('bench.json'), J('m3match.json')
    ms = J('../multi_sub.json')
    figs(E, Lg, Mx, ref, choose, bench, m3m)
    F = E[E.full]
    sel = (Mx.sepx.abs() > 15) & (Mx.sepy.abs() > 15) & (Mx.sepx.abs() < 250) & (Mx.sepy.abs() < 250)
    est = estimates(47452, 8443.0)
    q = lambda s, p: float(np.nanpercentile(s, p))
    zv_up = ((F.dca < 20) & (F.zv >= 1050) & (F.zv < 1302)).sum(); n20 = int((F.dca < 20).sum())
    mz = Mx[sel & (Mx.dca < 20)].zv; mz_up = float(((mz >= 1050) & (mz < 1302)).mean())
    s_ = 'singles'
    pr = choose['rules'] if choose else {}
    ev = choose['event_level'] if choose else {}
    arms = bench['arms'] if bench else {}
    tt = arms.get('tt'); off = arms.get('off') or arms.get('prod (10-26)')
    ge = off.get('ge24', {}) if off else {}

    def bins_row(v, k):
        return [f'{100 * b["both_found"]:.0f}' if v else '–' for b in (v['bins'] if v else [])]
    # ---------------- text
    H = []
    H.append(f'''<h1>det3 multi-particle events: what the second particles are, choosing the trigger track, and a real
two-track benchmark</h1>
<p class="meta">det3 golden row sat_det3 (6-27 Saturday scan, long run, 490 V), production final_phys forward fits (wft) and
the phys bundle; M3 v2 rays and the C22 M3 cluster trees. Geometry is from the waveform fits only, never from hit
times. Generated by <code>rerun/multiparticle/make_report.py</code>, 2026-10-26.</p>

<div class="verdict"><h2>Verdict</h2><ol>
<li><b>The second particles are secondary electrons knocked out of the material above det3 by the triggering muon,
mostly in or just under the upper M3 doublet.</b> They are in time with the muon ({pct((F.dt0.abs() < 150).mean())}
within ±150 ns; an unrelated particle would give ~{pct(est["acc_window_frac"])}), they point back to the muon
({pct((F.dca < 20).mean())} pass within 20 mm of it, against {pct((Mx[sel].dca < 20).mean())} for two unrelated muons),
and where they do, they meet it at Z = 1050–1302 mm in {zv_up} of {n20} events (unrelated muons {pct(mz_up)}).
They are far steeper than the muons (median |tan| {q(np.hypot(F.o_tx, F.o_ty), 50):.2f} against
{q(np.hypot(F.tan_x, F.tan_y), 50):.2f}), mostly MIP-like in charge with a heavy tail ({pct((F.o_qmip > 2).mean())} above
2× the single-track median, singles 9 %), and M3 does not see them along their line (any upper-plane match
{pct(m3m["upper_any_sec"])} vs {pct(m3m["upper_any_mir"])}–{pct(m3m["upper_any_swap"])} for the nulls; none on all four
upper planes). Independent coincident muons are excluded (expected {est["n_acc"]:.3f} events); knock-ons made in det3's
own gas do not form a separate cluster (0 vertices in the gap) — they are the δ-blob class.</li>
<li><b>Yes, there is a reference-free rule that beats production on these events.</b> Among the particle candidates of a
view, production's choice is the M3 particle {pct(pr["production_rank"]["acc"])} of the time (chance
{pct(choose["chance"])}). "Most vertical" alone gives {pct(pr["most_vertical"]["acc"])}; a 5-feature conditional logit
(|tan|, prompt t0, time extent, χ²/dof, telescope acceptance), trained on even events and tested on odd and vice versa,
gives {pct(pr["logit_5"]["acc"])}, and {pct(pr["logit_noacc"]["acc"])} with det3-only features. Events with every
view right: {ev["production_rank"]["both_views_right"]} → {ev["logit_5"]["both_views_right"]} of {ev["logit_5"]["n_events"]}
(+{100 * (ev["logit_5"]["both_views_right"] - ev["production_rank"]["both_views_right"]) / 6451:.2f} points of interior
efficiency, with the multi-track credit). C22's rules (charge, column, χ², t0) indeed do not beat production; the gain
comes from the muon being steeper than its secondaries — a property of the cosmic trigger, not of X17 pairs.</li>
<li><b>Two-track benchmark (det3, bench overlay with M3 truth on both tracks):</b> both tracks found 0 % below 12 mm,
{pct(off["bins"][2]["both_found"]) if off else "–"} at 12–18 mm, ≥ 97.5 % from 18 mm (production seeder). Above
24 mm: separation σ {ge.get("sep_res", float("nan")):.2f} mm per view, opening angle σ {ge.get("open_res", float("nan")):.2f}°,
closest-approach σ {ge.get("dca_res", float("nan")):.1f} mm, vertex height σ {ge.get("zv_res_open_gt10", float("nan")):.0f} mm
(opening > 10°). The second track does not move the first at ≥ 18 mm (shift vs its single fit < 0.01 mm); real events
agree (M3 leg at single-track resolution beyond 30 mm, {bench["real"]["bins"][0]["res"]:.2f} mm / {bench["real"]["bins"][0]["dth"]:.2f}° below).
{("Joint two-track fit: " + ", ".join(f'{b["lo"]}–{b["hi"]} mm {100 * b["both_found"]:.0f} %' for b in tt["bins"][:4]) + f'; singles split {pct(tt["single_split"])} of {tt["n_single"]}.') if tt else "Joint two-track-fit arm: pending."}</li>
</ol></div>
''')
    # headline table
    H.append('<h2>Headline numbers</h2>')
    H.append(table([
        ['interior M3-triggered det3 tracks', '6 451', 'multitrack.py'],
        ['… with ≥ 2 particle-like clusters in a view', f'{ms["n"] if ms else 187} (2.9 %)', 'multitrack.py'],
        ['… 2×2 with the M3 particle found in both views ("full")', f'{len(F)} (1.8 %)', 'legs.py'],
        ['Δt0 second − muon: median / robust σ', f'{F.dt0.median():.0f} / {M.rsig(F.dt0):.0f} ns', 'singles x−y: σ ' + f'{ref["dt0xy_rsig"]:.0f} ns'],
        ['closest approach < 20 mm', f'{pct((F.dca < 20).mean())}', f'unrelated muons {pct((Mx[sel].dca < 20).mean())}'],
        ['… vertex at Z 1050–1302 mm', f'{zv_up} / {n20}', f'unrelated muons {pct(mz_up)}'],
        ['opening angle to the muon, median (10–90 %)', f'{q(F.open, 50):.0f}° ({q(F.open, 10):.0f}–{q(F.open, 90):.0f}°)',
         f'unrelated muons {q(Mx[sel].open, 50):.0f}°'],
        ['charge > 2× single median', pct((F.o_qmip > 2).mean()), 'singles 9 %'],
        ['M3 sees it on its line (any upper plane / all four)', f'{pct(m3m["upper_any_sec"])} / {pct(m3m["upper_all4_sec"])}',
         f'nulls {pct(m3m["upper_any_mir"])}, {pct(m3m["upper_any_swap"])}'],
        ['rule picks the M3 particle: production / best held-out', f'{pct(pr["production_rank"]["acc"])} / {pct(pr["logit_5"]["acc"])}',
         f'{choose["n_groups"]} views'],
    ], ['quantity', 'value', 'reference']))
    # Q1
    H.append(f'''<h2>1. What the second particles are</h2>
<p><b>Sample.</b> The 187 multi-particle events of <code>multitrack.py</code>: an interior M3 recipe ray through det3 and
≥ 2 particle-like production candidates (summed signal ≤ 24 samples, so a drift column, not a noise band) in a view.
In each view the M3 particle is the candidate nearest the ray at the mesh within 5 mm. In {len(F)} events the M3
particle and one other are found in both views; the other x and other y then make a 3-D line (2×2 is unambiguous
once the M3 legs are known). The rest are 1×2 / 2×1 (an overlap in one projection) or have the M3 particle missing.</p>
<p><b>Frames.</b> Raw strip frame with the M3 ray carried into it; Z is M3's (up; det3 mesh at 715, M3 at
1302/1185 and 144/24 mm). The M3 tangent is d pos / d Z and the wft ladder tangent is calibrated on it, so the lines compose.</p>''')
    H.append(fig('dt0.png', 'Δt0 between the second particle and the M3 particle (both views averaged). The same-view '
                 'difference cancels the FEU clock; its width is the single-leg t0 resolution (the x−y spread of one '
                 f'clean particle is σ {ref["dt0xy_rsig"]:.0f} ns). An unrelated particle would be flat (dashed).'))
    H.append(fig('vertex.png', 'Left: height of closest approach to the M3 ray, for second particles that pass within '
                 '20 mm of it, against two unrelated muons (M3 rays of two different clean events, same separation cut). '
                 'Right: closest-approach distance. Multiple scattering of an MeV electron over ~0.4 m of air (θ0 ≈ 15°) '
                 'smears both; the apparent vertex is biased low.'))
    H.append(fig('angle_charge.png', 'Second particle vs the triggering muon: inclination, opening angle (against unrelated '
                 'muons), and charge per unit path in units of the clean-single median.'))
    if m3m:
        H.append(fig('m3planes.png', 'Fraction of events with an extra M3 cluster (not the ray\'s) within 20 mm of the second '
                     'particle\'s extrapolation, per plane, against two nulls on the same events: the mirror point across '
                     'the ray and another event\'s offset. The upper planes are 470–590 mm above det3.'))
    H.append(f'''<h3>Against the hypotheses</h3>
{table([
    ['independent coincident cosmic', f'flux 1 cm⁻² min⁻¹ × {39.9 * 36.0:.0f} cm² = {est["r_det"]:.2f} Hz; × ~2.8 µs visible window = {est["p_acc"]:.1e} per trigger → <b>{est["n_acc"]:.3f}</b> of 6 451 (estimate)',
     f'flat Δt0; random direction and DCA', f'<b>excluded</b>: Δt0 prompt ({pct((F.dt0.abs() < 150).mean())} in ±150 ns), DCA small'],
    ['air-shower partner (same EAS)', 'not estimated from first principles; parallel to the muon, vertex far above',
     'opening ≲ few °, DCA large, Z_v ≫ 1.5 m', f'at most {int((F.open < 5).sum())} of {len(F)} have opening < 5° (≤ {100 * (F.open < 5).sum() / 6451:.2f} % of triggers)'],
    ['δ (knock-on) made in det3\'s gas', f'N(T&gt;T₀) = (K/2)(Z/A)ρx / T₀ = {est["gas_coef"]:.1e} MeV / T₀ → {pct(est["gas_50keV"], 1)} above 50 keV (estimate, pure Ar)',
     'vertex in the gap (Z 715–745), DCA ≈ 0', '<b>not these</b>: 0 events with DCA &lt; 10 mm and Z 690–770. A gas δ stays attached to the muon '
     '→ the δ-blob loss class (0.93 % of the interior), the same order as the estimate'],
    ['δ / small shower from material above', f'(K/2)(Z/A) t / T₀ ≈ {est["mat_per_gcm2_1MeV"]:.3f} per g/cm² above 1 MeV ({est["mat_per_gcm2_05MeV"]:.2f} above 0.5 MeV); emitted at cos θ ≈ √(T/(T+2mₑ)), 45° at 1 MeV (estimate)',
     'prompt, steep, DCA small at the material, MIP-like charge, not on M3\'s line once scattered',
     f'<b>the bulk</b>: vertices at Z 1050–1302 ({zv_up}/{n20}), steep, MIP-like median ({q(F.o_qmip, 50):.2f}), prompt'],
], ['hypothesis', 'expected rate', 'signature', 'data'])}
<p>The material estimate: 0.5–1 g/cm² of detector and frame in the lower plane of the upper M3 doublet (an assumption:
the plane's build is not documented here) gives 4–8 % of muons a knock-on above 1 MeV; with the geometric chance of
landing in det3's interior ≥ 20 mm from the muon in both views of order a third, that is 1–3 % per trigger, the
order of the observed 1.8 % (2×2) and 2.9 % (all multi-particle). This is an order-of-magnitude estimate, not a
prediction; a Geant4 run of the bench stack would make it one.</p>
<p><b>Charge.</b> Median {q(F.o_qmip, 50):.2f} of the single-track median per unit path (the muon in the same events
{q(F.m_qmip, 50):.2f}): above ~0.5 MeV an electron is near minimum ionising. The tail (27 % above 2×, singles 9 %) is
consistent with lower-energy electrons (dE/dx doubles near ~50–80 keV) or two overlapping secondaries — the histogram
shows a second bump near 2.3–2.6×, which is what two minimum-ionising particles sharing one cluster (a photon
conversion pair, or two knock-ons) would give. It cannot be split further from these fits.</p>
<p><b>M3.</b> The M events have extra M3 clusters about twice as often as clean singles on every plane (e.g. upper x
{pct(E.m3_ncl_x1.ge(2).mean())} vs 15 %), but not on the second particle's line beyond the nulls. Two readings, not
separable here: the secondaries are made below the upper doublet's planes, or above and scattered out of the 20 mm
window (an MeV electron scatters ~15° in 0.5 m of air plus det3's window). M3 never reconstructs them as a 4-layer
track: {int(((E.m3_nalt_x + E.m3_nalt_y) > 0).sum())} of the {len(E)} events have an alternative χ² &lt; 1 combination
in some view, and an alternative χ² &lt; 5 combination lies on the second particle (≤ 5 mm, Δtan &lt; 0.1) in one view of
{int(((F.alt_d_x < 5) | (F.alt_d_y < 5)).sum())} event and in both views of {int(((F.alt_d_x < 5) & (F.alt_d_y < 5)).sum())}.
So there is no real-data sample with telescope truth on both tracks.</p>''')
    # Q2
    rows = []
    for k, v in pr.items():
        extra = ''
        if 'vs_production_rank' in v:
            a_ = v['vs_production_rank']; b_ = v.get('vs_in_acceptance_then_rank', {})
            extra = f'+{a_["wins"]}/−{a_["losses"]} vs production (p={a_["p"]:.1g}); +{b_.get("wins")}/−{b_.get("losses")} vs acceptance-then-rank (p={b_.get("p", 1):.2g})'
        rows.append([k.replace('_', ' ') + (' (2-fold held-out)' if k.startswith('logit') else ''), pct(v['acc']),
                     f'{100 * v["ci68"][0]:.0f}–{100 * v["ci68"][1]:.0f} %', extra])
    H.append(f'''<h2>2. A reference-free rule to choose the triggering track</h2>
<p>{choose["n_groups"]} views (of the multi-particle events) hold the M3 particle among ≥ 2 particle candidates
({choose["n_cands"]} candidates, chance {pct(choose["chance"])}). A rule sees only det3 and the fixed telescope geometry
("acc": the candidate's projection stays inside the region recipe rays cross on all four M3 planes — a triggering
particle must cross them). The logits are conditional (softmax over a view's candidates), L2-regularised, trained on
even event ids and tested on odd and vice versa; weights per fold are in <code>choose.json</code>.</p>''')
    H.append(table(rows, ['rule', 'picks the M3 particle', '68 % CI', 'paired, held-out']))
    H.append(fig('choose.png', 'Per-view accuracy of each rule. Single-feature rules have no free parameter; the logits are '
                 'held-out.'))
    H.append(f'''<p>Medians, M3 particle vs other: |tan| {choose["medians"]["atan"]["m3"]:.2f} vs {choose["medians"]["atan"]["other"]:.2f},
χ²/dof {choose["medians"]["chi2"]["m3"]:.0f} vs {choose["medians"]["chi2"]["other"]:.0f}, strips {choose["medians"]["nstr"]["m3"]:.1f}
vs {choose["medians"]["nstr"]["other"]:.0f}, log charge {choose["medians"]["lq"]["m3"]:.2f} vs {choose["medians"]["lq"]["other"]:.2f}.
Event level (all {ev["logit_5"]["n_events"]} multi-particle events, every view must end on the M3 particle; views where
it was not found count as failures): {", ".join(f'{k.replace("_", " ")} {v["both_views_right"]}' for k, v in ev.items())}.
The ceiling is the share of events in which the M3 particle is a candidate in both views: 80 % (150 of 187,
multitrack.json "any").</p>
<p><b>Why this differs from C22.</b> C22's <code>twotrack.py</code> tried production rank, largest charge, longest column,
lowest χ² and closest t0 on all outliers; here too none of those beats production (largest charge is worse:
{pct(pr["largest_q"]["acc"])}). The working feature is inclination: the trigger telescope accepts near-vertical muons,
while their secondaries are steep. It is a property of the cosmic trigger. On n_TOF, where every track counts and no
track is "the trigger", there is nothing to choose; for the bench efficiency it moves the multi-particle class.</p>''')
    # Q3
    if bench:
        rows = []
        for arm, v in arms.items():
            if not v:
                rows.append([arm, 'pending'] + [''] * 7); continue
            rows.append([arm] + [f'{100 * b["both_found"]:.0f} %' for b in v['bins']] + [f'{pct(v["single_split"])} of {v["n_single"]}'])
        H.append(f'''<h2>3. The two-track benchmark</h2>
<p><b>Overlay with truth on both tracks</b> (<code>rerun/satstudy/overlay.py</code>; arms re-run by
<code>overlay_arm.py</code> with WFT_DEAD_SEED=0, as the 10-26 run): two clean det3 single tracks whose t0 agree
within 30 ns, summed in the waveforms (the second track's footprint ± 5 mm, its noise added there), 40 pairs per
separation bin. A track is found when a distinct candidate lies within 2 mm of its own single fit. Truth for the
resolutions is each donor's M3 ray.</p>''')
        H.append(table(rows, ['arm'] + [f'{bench["sep_bins"][i]}–{bench["sep_bins"][i + 1]} mm' for i in range(len(bench['sep_bins']) - 1)]
                       + ['single tracks split']))
        rr = []
        for b in (off['bins'] if off else []):
            if 'sep_res' in b:
                rr.append([f'{b["lo"]}–{b["hi"]}', b['n_found'], f'{b["sep_res"]:.2f}', f'{b["sep_bias"]:+.2f}', f'{b["pos_res"]:.2f}',
                           f'{b["ang_res_deg"]:.2f}', f'{b["open_res"]:.2f}', f'{b["dca_res"]:.1f}',
                           f'{b.get("zv_res_open_gt10", float("nan")):.0f} ({b.get("n_open_gt10", 0)})', f'{b["shift_vs_single"]:.3f}'])
        H.append(table(rr, ['sep [mm]', 'both found', 'σ sep / view [mm]', 'bias |sep| [mm]', 'σ pos [mm]', 'σ angle [°]',
                            'σ opening [°]', 'σ DCA [mm]', 'σ Z_v, open&gt;10° [mm] (n)', 'shift vs own single fit [mm]']))
        H.append(fig('bench.png', 'Both-found efficiency, separation resolution and opening-angle resolution against '
                     'separation, overlay arms; green: real multi-particle events (√2 × the M3 leg residual against M3, '
                     'the second leg having no truth).'))
        rb = bench['real']
        H.append(table([[f'{b["lo"]}–{b["hi"]}', b['n'], f'{b["res"]:.2f}', f'{b["dth"]:.2f}', pct(b['agree'])] for b in rb['bins']],
                       ['separation to the other particle [mm]', 'M3 legs', 'σ residual [mm]', 'σ angle [°]', 'agrees with M3 (3 mm, 5°)']))
        H.append(fig('realsep.png', f'Separation of the real 2×2 events (min over views). Below 18 mm the production seeder '
                     f'cannot return both, so the real spectrum there is the {ms["sub"]["m3_not_found"]["n"] if ms else 37} '
                     '"M3 not found" events (30 are projection overlaps), not these.'))
        H.append(f'''<p>Clean singles: σ 0.59 / 0.54 mm, 1.02 / 1.00° (x / y). Real opening-angle error from the M3 leg alone:
σ {rb["open_err_from_m3_leg"]:.2f}° ({rb["n_full"]} events); the overlay's full opening-angle σ
{ge.get("open_res", float("nan")):.2f}° is that from both legs, ≈ √2 × the single-track angle.</p>
<h3>How it relates to T1 / T2 (SAME_CHAMBER_PAIRS.md)</h3>
<ul>
<li><b>T1 (two-track separation)</b> measured on n_TOF run_145 with the beam seeder: physical limit ~1 strip pitch,
real-track limit 2–3 mm with the fixed chain + joint fit; coincident overlay pairs below 12 mm found 19 → 58 % (A),
36 → 49 % (C). The det3 bench with the production (bench) seeder is cut at its 12 mm seed gap: 0 % below 12 mm,
~22 % at 12–18, then fully efficient. The bench is therefore limited by the seeder, not by physics: with the joint
two-track fit (production settings, F = 300) both are found in {", ".join(f'{100 * b["both_found"]:.0f} %' for b in tt["bins"][:3]) if tt else "–"}
of the 0–6 / 6–12 / 12–18 mm pairs with 0 of {tt["n_single"] if tt else "–"} clean singles split, and the recovered pairs keep
~1 mm separation resolution. That is the same scale as T1's beam overlay (A &lt; 12 mm 58 %, C 49 % with its tuned F). The re-pass on n_TOF lost pairs below ~20 mm in either view — the same scale.</li>
<li><b>T2 (single-track angle truth)</b>: at ≥ 18 mm the second track does not perturb the first (fits identical to
the singles to &lt; 0.01 mm, 0.05°), so pair angles inherit T2's single-track angle scale and resolution directly; the
pair opening-angle resolution is √2 × the single-track angle (≈ 1.4–1.5° on det3).</li>
<li><b>Vertex.</b> σ(Z_v) ≈ σ(sep) / tan(opening): {ge.get("zv_res_scaled", float("nan")):.0f} mm × 1/tan(open) here.
T1's pair_angle.py: a common-vertex pair diverges by only 0.13 × its separation across the 30 mm gap, so close pairs
are parallel inside one chamber; the in-chamber vertex is weak and a vertex needs the second chamber's lever arm.</li>
</ul>''')
    H.append(f'''<h2>What this does not rule out</h2><ul>
<li>The vertex heights are straight-line extrapolations of particles that are mostly MeV electrons; multiple scattering
in air and in det3's window smears them by ~0.1 m and biases them low. "Made in or just under the upper M3 doublet" is
the best reading, not a measurement of the production point. The material between det3 and the doublet, and the
doublet's own build, are not documented here.</li>
<li>The δ and air-shower rates are labelled estimates (PDG knock-on formula, rule-of-thumb flux, assumed material
thickness); they bound orders of magnitude only. No simulation of the bench stack was run.</li>
<li>Charge cannot tell an electron above ~0.5 MeV from a muon; the high-charge tail is not resolved into low-energy
electrons vs two overlapping secondaries.</li>
<li>The choosing rule is trained on {choose["n_groups"]} views (2-fold by event parity; paired held-out comparison
p ≈ 1e-11 against production, ≈ 0.01 against the parameter-free acceptance rule). Its main lever, inclination,
belongs to the cosmic trigger and does not transfer to beam data.</li>
<li>The overlay sums two single tracks of different triggers: their ionisation is real, but the pair is unrelated
(no common vertex, t0 within 30 ns by selection), and only the second track's footprint ± 5 mm is added. Real
two-track events with truth on both legs do not exist here: M3 does not reconstruct the second particle.</li>
<li>All of this is det3 at one operating point (490 V, bench gas); the 12 mm seeder gap and the noise are bench
settings, and the spark-vetoed events (126, 1.95 %; 45 of them showers) are not in the 187.</li>
</ul>
<h2>Files</h2>
<p>Code <code>rerun/multiparticle/</code> (mplib, legs, m3match, choose, overlay_arm, bench, make_report); outputs
<code>~/x17/cosmic_bench/satstudy/multiparticle/</code>: legs.parquet, events.parquet, mixed.parquet, m3match.*, choose.json,
bench.json, bench_pairs.parquet, overlay_off/, overlay_tt/.</p>''')
    css = '''<style>
:root{--fg:#1d1d1f;--bg:#fff;--mut:#666;--acc:#2a78d6;--card:#f5f7fa;--bd:#dde3ea}
@media (prefers-color-scheme: dark){:root{--fg:#e8e8ea;--bg:#16181b;--mut:#9aa;--acc:#3987e5;--card:#1f2328;--bd:#333a42}
img{background:#fff}}
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:var(--fg);background:var(--bg);max-width:1050px;margin:0 auto;padding:16px;line-height:1.45}
h1{font-size:1.45em}h2{margin-top:1.6em;border-bottom:1px solid var(--bd);padding-bottom:.2em}.meta{color:var(--mut);font-size:.9em}
.verdict{background:var(--card);border-left:4px solid var(--acc);padding:.4em 1em;border-radius:4px}
table{border-collapse:collapse;margin:.8em 0;font-size:.86em;display:block;overflow-x:auto}td,th{border:1px solid var(--bd);padding:4px 7px;vertical-align:top}
th{background:var(--card)}figure{margin:1em 0}figure img{max-width:100%;border:1px solid var(--bd)}figcaption{font-size:.85em;color:var(--mut)}
code{font-size:.9em}</style>'''
    doc = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>det3 multi-particle events</title><meta name="description" content="{DESC}">{css}</head><body>{"".join(H)}</body></html>'
    open(os.path.join(RD, 'report.html'), 'w').write(doc)
    # self-contained copy for the notes site (images as data: URIs, nothing external)
    import base64, re
    inl = re.sub(r'src="figures/([^"]+)"', lambda m: 'src="data:image/png;base64,' + base64.b64encode(
        open(os.path.join(FD, m.group(1)), 'rb').read()).decode() + '"', doc)
    open(os.path.join(RD, 'mx17-det3-multiparticle.html'), 'w').write(inl)
    print('wrote', os.path.join(RD, 'report.html'), 'and the inline copy')


if __name__ == '__main__':
    main()
