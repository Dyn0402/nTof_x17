#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_chi2_plots.py -- figures + measured numbers for the det3 position-miss
follow-up (owner ask 2026-07-13). Reads edge_chi2_data_<KEY>.npz (from
edge_chi2_extract.py) and the run's alignment.json (for the aligned->detector-
local inverse transform), writes PNGs here and edge_chi2_meta_<KEY>.json.

Three questions:
  1. Is the > ~2 mm miss the REFERENCE TRACKER's fault?  (M3 chi2 / NClus / angle)
  2. What is the detector's INTRINSIC accuracy once reference/chamber pathologies
     are removed?
  3. Where are the physical active-area edges / the ~2 cm passivated Y strips, and
     is the efficiency denominator handling them?

Run:  ../../.venv/bin/python edge_chi2_plots.py [KEY]        (default g_det3_wknd)
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEY = next((a for a in sys.argv[1:] if not a.startswith('-')), 'g_det3_wknd')
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, f'edge_chi2_data_{KEY}.npz'), allow_pickle=True)

# ---- alignment inverse (aligned frame -> detector-local strip frame 0..398 mm) ----
# Resolved generically from qa_config (works for any registered run key, not just det3's
# two headline runs -- originally a hardcoded 2-entry dict, generalised 2026-07-14 to
# measure the passivation on det2/4/6/7 too).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_config import get_config, setup_paths
setup_paths()
_align_path = os.path.join(get_config(KEY).OUT_BASE, 'alignment_tpc_veto50', 'alignment.json')
al = json.load(open(_align_path))
th = np.deg2rad(al['theta_deg']); cx, cy = al['centre_x'], al['centre_y']
xoff, yoff = al['x_offset'], al['y_offset']
c, s = np.cos(th), np.sin(th)
STRIP_MAX = 398.58


def inv(xp, yp):
    """aligned (x',y') -> detector-local (x,y)."""
    u = np.asarray(xp) - cx - xoff; v = np.asarray(yp) - cy - yoff
    return c * u + s * v + cx, -s * u + c * v + cy


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.tight_layout(); fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print('wrote', name)


meta = {'key': KEY}

# ============================ reco-level arrays ============================
r = d['r']
chi2m = np.maximum(d['m3_chi2x'], d['m3_chi2y'])          # M3 reference track chi2 (worse plane)
mult = d['mult'].astype(float)                            # detector strips fired (discharge proxy)
far2, far5, near2 = r > 2.0, r > 5.0, r <= 2.0
badM3, badCh = chi2m >= 1.0, mult >= 25

meta['counts'] = dict(n_reco=int(len(r)),
                      pct_gt2=100 * float(far2.mean()), pct_gt5=100 * float(far5.mean()),
                      median_mm=float(np.median(r)), core_within5=100 * float((r <= 5).mean()))

# ---------------------------------------------------------------------------
# FIGURE 1 -- Is it the reference tracker's fault?
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

# (a) |r| vs M3 chi2max : profile + 2D
xb = np.linspace(0, 5, 26); xc = 0.5 * (xb[:-1] + xb[1:])
med = np.array([np.median(r[(chi2m >= xb[i]) & (chi2m < xb[i + 1])]) if ((chi2m >= xb[i]) & (chi2m < xb[i + 1])).sum() > 30
                else np.nan for i in range(len(xc))])
p2 = np.array([100 * far2[(chi2m >= xb[i]) & (chi2m < xb[i + 1])].mean() if ((chi2m >= xb[i]) & (chi2m < xb[i + 1])).sum() > 30
               else np.nan for i in range(len(xc))])
p5 = np.array([100 * far5[(chi2m >= xb[i]) & (chi2m < xb[i + 1])].mean() if ((chi2m >= xb[i]) & (chi2m < xb[i + 1])).sum() > 30
               else np.nan for i in range(len(xc))])
h = ax[0].hist2d(np.clip(chi2m, 0, 5), np.clip(r, 0, 20), bins=[40, 40],
                 range=[[0, 5], [0, 20]], cmap='inferno', norm=matplotlib.colors.LogNorm())
ax[0].plot(xc, med, color='cyan', lw=2, label='median |r|')
ax[0].axhline(2, color='w', ls=':', lw=0.8); ax[0].axhline(5, color='w', ls='--', lw=0.8)
ax[0].set_xlabel('M3 reference track  max(Chi2X,Chi2Y)'); ax[0].set_ylabel('|r| = |reco - ray| [mm]')
ax[0].legend(loc='upper left'); ax[0].set_title('(a) position residual vs M3 track chi2\n(recipe already cuts chi2<5)')

# (b) far-rate vs M3 chi2
ax[1].plot(xc, p2, 'o-', color='darkorange', label='|r|>2 mm')
ax[1].plot(xc, p5, 's-', color='firebrick', label='|r|>5 mm')
ax[1].set_xlabel('M3 reference track  max(Chi2X,Chi2Y)'); ax[1].set_ylabel('miss rate [%]')
ax[1].legend(); ax[1].grid(alpha=0.3)
ax[1].set_title('(b) miss rate rises with reference-track chi2\ntop-decile chi2: %.0f%% miss >2mm vs %.0f%% for rest'
                % (100 * far2[chi2m >= np.percentile(chi2m, 90)].mean(),
                   100 * far2[chi2m < np.percentile(chi2m, 90)].mean()))

# (c) cause attribution stacked bars for >2mm and >5mm
def attrib(mask):
    f = mask
    return [100 * (badM3 & ~badCh)[f].mean(), 100 * (badCh & ~badM3)[f].mean(),
            100 * (badM3 & badCh)[f].mean(), 100 * (~badM3 & ~badCh)[f].mean()]
lab = ['bad M3 only\n(chi2>=1)', 'discharge only\n(mult>=25)', 'both', 'neither\n(resolution)']
cols = ['#3b6fb0', '#c0392b', '#8e44ad', '#7f8c8d']
A2, A5 = attrib(far2), attrib(far5)
bot2 = bot5 = 0
for i in range(4):
    ax[2].bar(0, A2[i], bottom=bot2, color=cols[i], width=0.6)
    ax[2].bar(1, A5[i], bottom=bot5, color=cols[i], width=0.6, label=lab[i])
    ax[2].text(0, bot2 + A2[i] / 2, f'{A2[i]:.0f}', ha='center', va='center', fontsize=9, color='w')
    ax[2].text(1, bot5 + A5[i] / 2, f'{A5[i]:.0f}', ha='center', va='center', fontsize=9, color='w')
    bot2 += A2[i]; bot5 += A5[i]
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(['|r|>2 mm', '|r|>5 mm'])
ax[2].set_ylabel('% of miss events'); ax[2].legend(fontsize=8, loc='lower right')
ax[2].set_title('(c) cause of the miss\n~half of >2mm sit on a bad reference track')
save(fig, f'fig1_reference_fault_{KEY}.png')
meta['reference_fault'] = dict(
    m3chi2_med_near=float(np.median(chi2m[near2])), m3chi2_med_far2=float(np.median(chi2m[far2])),
    m3chi2_med_far5=float(np.median(chi2m[far5])),
    topdecile_far2=float(100 * far2[chi2m >= np.percentile(chi2m, 90)].mean()),
    rest_far2=float(100 * far2[chi2m < np.percentile(chi2m, 90)].mean()),
    attrib_gt2=dict(zip(['badM3_only', 'discharge_only', 'both', 'neither'], A2)),
    attrib_gt5=dict(zip(['badM3_only', 'discharge_only', 'both', 'neither'], A5)))

# ---------------------------------------------------------------------------
# FIGURE 2 -- intrinsic detector accuracy once reference/chamber cleaned
# ---------------------------------------------------------------------------
cleanM3 = chi2m < 1.0; cleanCh = mult < 25; both = cleanM3 & cleanCh
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
# (a) within-5mm efficiency bars
subs = [('all reco', np.ones(len(r), bool)), ('clean M3\nchi2<1', cleanM3),
        ('clean chamber\nmult<25', cleanCh), ('clean both', both)]
vals = [100 * (r[m] <= 5).mean() for _, m in subs]
fr = [100 * m.mean() for _, m in subs]
bars = ax[0].bar([x[0] for x in subs], vals, color=['#7f8c8d', '#2980b9', '#27ae60', '#16a085'])
for b, v, f in zip(bars, vals, fr):
    ax[0].text(b.get_x() + b.get_width() / 2, v + 0.2, f'{v:.1f}%', ha='center', fontsize=9)
ax[0].set_ylim(90, 100); ax[0].set_ylabel('reconstructed hits within 5 mm [%]')
ax[0].set_title('(a) intrinsic accuracy of the CHAMBER\nrises to %.1f%% with a clean reference + no discharge' % vals[3])
# (b) |r| CDF
gg = np.linspace(0, 15, 300)
for name, m, col in [('all reco', np.ones(len(r), bool), '#7f8c8d'),
                     ('clean M3', cleanM3, '#2980b9'), ('clean both', both, '#16a085')]:
    cdf = np.searchsorted(np.sort(r[m]), gg) / m.sum()
    ax[1].plot(gg, 100 * cdf, color=col, lw=2, label=name)
ax[1].axvline(2, color='k', ls=':', lw=0.8); ax[1].axvline(5, color='k', ls='--', lw=0.8)
ax[1].set_xlabel('|r| [mm]'); ax[1].set_ylabel('cumulative [%]'); ax[1].legend(loc='lower right')
ax[1].set_ylim(60, 100); ax[1].grid(alpha=0.3); ax[1].set_title('(b) |r| CDF')
# (c) chamber-side mechanism: |r| vs multiplicity
mb = np.arange(5, 51, 3); mc = 0.5 * (mb[:-1] + mb[1:])
mp = np.array([100 * far5[(mult >= mb[i]) & (mult < mb[i + 1])].mean() if ((mult >= mb[i]) & (mult < mb[i + 1])).sum() > 30
               else np.nan for i in range(len(mc))])
ax[2].plot(mc, mp, 'o-', color='#c0392b')
ax[2].axvline(50, color='purple', ls='--', label='spark cut (50)')
ax[2].set_xlabel('event multiplicity (strips fired)'); ax[2].set_ylabel('|r|>5 mm rate [%]')
ax[2].legend(); ax[2].grid(alpha=0.3)
ax[2].set_title('(c) chamber-side mechanism\nsub-spark high-multiplicity (discharge) also misses')
save(fig, f'fig2_intrinsic_accuracy_{KEY}.png')
meta['intrinsic'] = dict(within5_all=vals[0], within5_cleanM3=vals[1],
                         within5_cleanCh=vals[2], within5_both=vals[3],
                         frac_cleanM3=fr[1], frac_cleanCh=fr[2], frac_both=fr[3])

# ---------------------------------------------------------------------------
# FIGURE 3 -- edges & passivation (detector-local frame)
# ---------------------------------------------------------------------------
cat = d['cat']; locx, locy = inv(d['cx'], d['cy'])
recon = cat == 0; notspark = cat != 2


def eff_profile(coord, sel, lo, hi, bw=4.0, minN=15):
    e = np.arange(lo, hi + bw, bw); ce = 0.5 * (e[:-1] + e[1:])
    eff = np.full(len(ce), np.nan)
    for i in range(len(ce)):
        m = (coord >= e[i]) & (coord < e[i + 1]) & sel
        if m.sum() >= minN:
            eff[i] = 100 * recon[m].mean()
    return ce, eff


def cross50(ce, eff, plat, rising):
    half = plat / 2; idx = np.where(np.isfinite(eff))[0]
    for k in range(len(idx) - 1):
        i, j = idx[k], idx[k + 1]
        if rising and eff[i] < half <= eff[j]:
            return ce[i] + (ce[j] - ce[i]) * (half - eff[i]) / (eff[j] - eff[i])
        if not rising and eff[i] >= half > eff[j]:
            return ce[i] + (ce[j] - ce[i]) * (half - eff[i]) / (eff[j] - eff[i])
    return np.nan


inXpl = (locx > 40) & (locx < 360); inYpl = (locy > 40) & (locy < 360)
ceY, effY = eff_profile(locy, notspark & inXpl, -20, 420)
ceX, effX = eff_profile(locx, notspark & inYpl, -20, 420)
platY = np.nanmedian(effY[(ceY > 60) & (ceY < 340)]); platX = np.nanmedian(effX[(ceX > 60) & (ceX < 340)])
yLo, yHi = cross50(ceY, effY, platY, True), cross50(ceY, effY, platY, False)
xLo, xHi = cross50(ceX, effX, platX, True), cross50(ceX, effX, platX, False)

fig, ax = plt.subplots(2, 2, figsize=(14, 11))
# (a) 2D efficiency map in detector-local frame
nb = 40
Hn, xe, ye = np.histogram2d(locx[recon], locy[recon], bins=nb, range=[[-20, 420], [-20, 420]])
Ht, _, _ = np.histogram2d(locx[notspark], locy[notspark], bins=[xe, ye])
effm = np.full_like(Hn, np.nan); ok = Ht >= 12; effm[ok] = 100 * Hn[ok] / Ht[ok]
im = ax[0, 0].imshow(effm.T, origin='lower', extent=[-20, 420, -20, 420], aspect='equal',
                     cmap='viridis', vmin=0, vmax=100)
plt.colorbar(im, ax=ax[0, 0], label='reco_near efficiency [%]')
for v in (0, STRIP_MAX):
    ax[0, 0].axvline(v, color='w', ls=':', lw=0.7); ax[0, 0].axhline(v, color='w', ls=':', lw=0.7)
ax[0, 0].axhspan(-20, yLo, color='red', alpha=0.12); ax[0, 0].axhspan(yHi, 420, color='red', alpha=0.12)
ax[0, 0].set_xlabel('detector-local X [mm]'); ax[0, 0].set_ylabel('detector-local Y [mm]')
ax[0, 0].set_title('(a) efficiency map (detector frame)\nred = passivated Y bands')

# (b) efficiency vs local Y
ax[0, 1].plot(ceY, effY, 'o-', color='#2c3e50', ms=3)
ax[0, 1].axvspan(-20, yLo, color='red', alpha=0.15, label='passivated')
ax[0, 1].axvspan(yHi, 420, color='red', alpha=0.15)
ax[0, 1].axhline(platY / 2, color='grey', ls=':', lw=0.8)
for v in (yLo, yHi):
    ax[0, 1].axvline(v, color='red', ls='--', lw=1)
ax[0, 1].axvline(0, color='k', ls=':', lw=0.7); ax[0, 1].axvline(STRIP_MAX, color='k', ls=':', lw=0.7)
ax[0, 1].set_xlabel('detector-local Y [mm]'); ax[0, 1].set_ylabel('reco_near efficiency [%]')
ax[0, 1].legend(loc='lower center')
ax[0, 1].set_title('(b) Y edges: active [%.0f, %.0f] mm\npassivated %.0f mm (low) + %.0f mm (high)'
                   % (yLo, yHi, yLo - 0, STRIP_MAX - yHi))

# (c) efficiency vs local X (sharp geometric edges, no passivation)
ax[1, 0].plot(ceX, effX, 'o-', color='#2c3e50', ms=3)
ax[1, 0].axhline(platX / 2, color='grey', ls=':', lw=0.8)
for v in (xLo, xHi):
    ax[1, 0].axvline(v, color='green', ls='--', lw=1)
ax[1, 0].axvline(0, color='k', ls=':', lw=0.7); ax[1, 0].axvline(STRIP_MAX, color='k', ls=':', lw=0.7)
ax[1, 0].set_xlabel('detector-local X [mm]'); ax[1, 0].set_ylabel('reco_near efficiency [%]')
ax[1, 0].set_title('(c) X edges: active [%.0f, %.0f] mm\nsharp geometric edges at 0 / %.0f mm, no passivation'
                   % (xLo, xHi, STRIP_MAX))

# (d) reco_far RATE vs distance to nearest active edge (is the tail edge-driven?)
lxr, lyr = inv(d['ray_x'], d['ray_y'])
dxe = np.minimum(lxr - 0, STRIP_MAX - lxr); dye = np.minimum(lyr - yLo, yHi - lyr)
dedge = np.minimum(dxe, dye)
eb = np.array([0, 10, 20, 40, 80, 200]); ec = 0.5 * (eb[:-1] + eb[1:])
rate5 = [100 * (r[(dedge >= eb[i]) & (dedge < eb[i + 1])] > 5).mean() for i in range(len(ec))]
rate2 = [100 * (r[(dedge >= eb[i]) & (dedge < eb[i + 1])] > 2).mean() for i in range(len(ec))]
ax[1, 1].plot(ec, rate2, 'o-', color='darkorange', label='|r|>2 mm')
ax[1, 1].plot(ec, rate5, 's-', color='firebrick', label='|r|>5 mm')
ax[1, 1].axhline(100 * (r > 5).mean(), color='firebrick', ls=':', lw=0.8)
ax[1, 1].axhline(100 * (r > 2).mean(), color='darkorange', ls=':', lw=0.8)
ax[1, 1].set_xlabel('distance to nearest active edge [mm]'); ax[1, 1].set_ylabel('miss rate [%]')
ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)
ax[1, 1].set_title('(d) miss rate is FLAT vs edge distance\nv2 tail is NOT edge-enhanced (dotted = bulk)')
save(fig, f'fig3_edges_passivation_{KEY}.png')

meta['edges'] = dict(
    plateau_eff_Y=float(platY), plateau_eff_X=float(platX),
    active_Y=[float(yLo), float(yHi)], active_X=[float(xLo), float(xHi)],
    passivation_Y_low_mm=float(yLo - 0), passivation_Y_high_mm=float(STRIP_MAX - yHi),
    active_Y_span_mm=float(yHi - yLo), active_X_span_mm=float(xHi - xLo),
    nominal_strip_mm=STRIP_MAX,
    far5_rate_by_edgedist=dict(zip([f'{eb[i]}-{eb[i+1]}' for i in range(len(ec))], [float(v) for v in rate5])),
    far2_rate_by_edgedist=dict(zip([f'{eb[i]}-{eb[i+1]}' for i in range(len(ec))], [float(v) for v in rate2])))
# active-area efficiency inside the measured rectangle, 09-style (spark IN denominator)
inrect = (locx >= xLo) & (locx <= xHi) & (locy >= yLo) & (locy <= yHi)
nrect = int(inrect.sum())
meta['active_rectangle'] = dict(N=nrect, **{
    k: float(100 * ((cat == v) & inrect).sum() / nrect)
    for k, v in {'reco_near': 0, 'reco_far': 1, 'spark': 2, 'hit_no_reco': 3, 'no_hit': 4}.items()})
meta['active_rectangle']['nominal_area_efficiency_if_full_40x40'] = float(
    meta['active_rectangle']['reco_near'] * (yHi - yLo) / STRIP_MAX)

json.dump(meta, open(os.path.join(HERE, f'edge_chi2_meta_{KEY}.json'), 'w'), indent=2)
print('\n' + json.dumps(meta, indent=2))
