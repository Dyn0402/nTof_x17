#!/usr/bin/env python3
"""make_report.py -- figures + report.html for the noise-band study (HANDOFF satstudy follow-up 1).

    python make_report.py       # reads $BANDS_OUT (default ~/x17/cosmic_bench/satstudy/bands): {patches,fdfcheck,hvscan}.json, patches_*.npz, ntof_*.npz
                                # writes $BANDS_OUT/summary.json, figures/*.png, report.html

Inputs are produced by patches.py, fdfcheck.py, hvscan.py (local) and ntof_scan.py (lxplus; the npz files are copied
from /eos/user/d/dneff/bands/ to $BANDS_OUT).  Re-run after any of them changes.
"""
import glob, html, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('BANDS_OUT', os.path.expanduser('~/x17/cosmic_bench/satstudy/bands')); FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, '..', 'satstudy'))
import bands as B  # noqa: E402

P = json.load(open(os.path.join(OUT, 'patches.json')))
FD = json.load(open(os.path.join(OUT, 'fdfcheck.json')))
HV = json.load(open(os.path.join(OUT, 'hvscan.json'))) if os.path.exists(os.path.join(OUT, 'hvscan.json')) else {}
ORDER = ['sat_det3', 'g_det3_wknd', 'g_det3', 'g_det2', 'g_det4', 'g_det6_long', 'g_det7_long']
LAB = {'sat_det3': 'det3 (A) 6-27 FEU 7/8', 'g_det3_wknd': 'det3 (A) 6-28 FEU 7/8', 'g_det3': 'det3 (A) 6-22 FEU 3/4',
       'g_det2': 'det2 (B) 6-22 FEU 6/8', 'g_det4': 'det4 (E) 6-24 FEU 6/8', 'g_det6_long': 'det6 (C) 6-26 FEU 3/4',
       'g_det7_long': 'det7 (D) 6-26 FEU 6/8'}
COL = {'sat_det3': '#1f5fbf', 'g_det3_wknd': '#5b8fd9', 'g_det3': '#98b8e8', 'g_det2': '#2a9d8f', 'g_det4': '#8a8a8a',
       'g_det6_long': '#e8890c', 'g_det7_long': '#c0392b'}
LOCAL = {'sat_det3', 'g_det3_wknd', 'g_det3', 'g_det7_long'}           # fixed-patch family
S = {}                                                                   # numbers for summary.json

# ---------------------------------------------------------------- 1. where: occupancy profile per chamber
fig, axs = plt.subplots(len(ORDER), 2, figsize=(11, 1.55 * len(ORDER)), sharex=True)
for i, k in enumerate(ORDER):
    Z = np.load(os.path.join(OUT, f'patches_{k}.npz')); pb = P[k]['p_band']
    for j, v in enumerate('xy'):
        ax = axs[i, j]; p = Z[f'pos_{v}']; o = Z[f'occ_{v}'] * pb * 100
        ok = np.isfinite(p); s = np.argsort(p[ok])
        ax.fill_between(p[ok][s], o[ok][s], step='mid', color=COL[k], lw=0)
        for c in range(1, 8):                                  # connector boundaries in position
            ch = np.arange(c * 64 - 1, c * 64 + 1); q = p[ch]
            if np.all(np.isfinite(q)) and abs(q[1] - q[0]) < 2:
                ax.axvline(q.mean(), color='0.8', lw=0.6, zorder=0)
        ax.set_ylim(0, max(4, o[ok].max() * 1.1)); ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_title(f'{v} view: % of triggers with this strip in a band', fontsize=9)
        if j == 0:
            ax.set_ylabel(LAB[k].replace(' 6-', '\n6-'), fontsize=7)
axs[-1, 0].set_xlabel('x strip position [mm]'); axs[-1, 1].set_xlabel('y strip position [mm]')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'where.png'), dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 2. persistence into the next trigger
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for k in ORDER:
    pe = P[k]['persistence']; x, y, e = [], [], []
    for r in pe:
        if r['n_band_chips'] >= 10 and r['hi']:
            m = np.sqrt(r['lo'] * r['hi']) if r['lo'] > 0 else r['hi'] / 2
            pr = r['p_same_chip_next'] / r['p_chip_base']; n = r['n_band_chips']
            x.append(m * 1e3); y.append(pr); e.append(np.sqrt(r['p_same_chip_next'] * (1 - r['p_same_chip_next']) / n + 1 / n ** 2) / r['p_chip_base'])
    ax.errorbar(x, y, e, color=COL[k], marker='o', ms=4, lw=1.5, capsize=0, label=LAB[k])
ax.axhline(1, color='0.5', lw=0.8, ls=':')
ax.set_xscale('log'); ax.set_xlabel('time to the next trigger [ms]')
ax.set_ylabel('P(same chip banded in next trigger) / base rate')
ax.set_title('A band survives into a trigger < 1 ms later, not one > 3 ms later: lifetime ~ 1 ms', fontsize=9)
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'persistence.png'), dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 3. cross-chamber independence (same triggers)
pairs = []
for a, b in (('g_det3', 'g_det2'), ('g_det6_long', 'g_det7_long')):
    A = np.load(os.path.join(OUT, f'patches_{a}.npz')); Bb = np.load(os.path.join(OUT, f'patches_{b}.npz'))
    com, ia, ib = np.intersect1d(A['eid'], Bb['eid'], return_indices=True)
    xa, xb = A['anyb'][ia], Bb['anyb'][ib]
    pairs.append(dict(a=a, b=b, n=int(len(com)), p_a=float(xa.mean()), p_b=float(xb.mean()),
                      p_b_given_a=float(xb[xa].mean()), p_a_given_b=float(xa[xb].mean()),
                      both=int((xa & xb).sum()), both_expected=float(xa.mean() * xb.mean() * len(com))))
S['cross_chamber'] = pairs

# ---------------------------------------------------------------- 4. HV scans
if HV:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
    for name, col, lab in (('det3_resist', '#1f5fbf', 'det3 (6-27, drift 1000 V)'), ('det6', '#e8890c', 'det6 (6-26, drift 700 V)'),
                           ('det7', '#c0392b', 'det7 (6-26, drift 700 V)')):
        if name not in HV:
            continue
        R = sorted(HV[name], key=lambda r: r['resist'])
        axs[0].errorbar([r['resist'] for r in R], [100 * r['p_band'] for r in R], [100 * r['err'] for r in R],
                        color=col, marker='o', ms=4, lw=1.5, label=lab + ': band')
        axs[0].plot([r['resist'] for r in R], [100 * r['p_spark'] for r in R], color=col, ls=':', marker='s', ms=3, lw=1,
                    label=lab + ': spark (> 150 strips)')
    axs[0].set_xlabel('resist (amplification) HV [V]'); axs[0].set_ylabel('% of triggers'); axs[0].set_yscale('log')
    axs[0].set_title('Band rate against the amplification field', fontsize=9); axs[0].legend(fontsize=6.5)
    if 'det3_drift' in HV:
        R = sorted(HV['det3_drift'], key=lambda r: r['drift'])
        axs[1].errorbar([r['drift'] for r in R], [100 * r['p_band'] for r in R], [100 * r['err'] for r in R], color='#1f5fbf',
                        marker='o', ms=4, lw=1.5, label='det3 band (resist 490 V)')
        axs[1].plot([r['drift'] for r in R], [100 * r['p_spark'] for r in R], color='#1f5fbf', ls=':', marker='s', ms=3,
                    lw=1, label='det3 spark')
        axs[1].set_xlabel('drift HV [V] (30 mm gap)'); axs[1].set_ylabel('% of triggers')
        axs[1].set_title('Band rate against the drift field', fontsize=9); axs[1].legend(fontsize=7); axs[1].set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'hvscan.png'), dpi=130); plt.close(fig)
    # log-slope of the band rate with the resist HV, against the bench gain slope (x2 per 16.5 V, mx17-hv-slope-test)
    fits = {}
    for name in ('det3_resist', 'det6', 'det7'):
        if name not in HV:
            continue
        R = [r for r in HV[name] if r['p_band'] > 0.002 and r['p_band'] < 0.3 and r['n'] > 1000]
        if len(R) >= 3:
            v = np.array([r['resist'] for r in R]); y = np.log(np.array([r['p_band'] for r in R]))
            a = np.polyfit(v, y, 1)[0]
            fits[name] = dict(n_points=len(R), v_range=[int(v.min()), int(v.max())], factor_per_10V=float(np.exp(10 * a)),
                              doubling_V=float(np.log(2) / a), power_of_gain=float(a / (np.log(2) / 16.5)))
    S['hv_fits'] = fits
    S['hvscan'] = {n: [{k: r[k] for k in ('resist', 'drift', 'n', 'p_band', 'err', 'p_spark', 'p_band_x', 'p_band_y')} for r in HV[n]]
                   for n in HV}

# ---------------------------------------------------------------- 5. n_TOF
NT = {}
CH = {'A (det3)': (3, 4), 'B (det2)': (5, 6), 'C (det6)': (7, 8), 'D (det7)': (1, 2)}
for f in sorted(glob.glob(os.path.join(OUT, 'ntof_run*.npz'))):
    run = os.path.basename(f)[5:-4]; Z = np.load(f); tags = sorted({k.split('|')[0] + '|' + k.split('|')[1] for k in Z.files})
    r = {}
    for ch, (fx, fy) in CH.items():
        n = nb = nbx = nby = nsp = 0; noise = []; cnt = {'x': np.zeros(512), 'y': np.zeros(512)}
        for t in tags:
            kx, ky = f'{t}|{fx}|', f'{t}|{fy}|'
            if kx + 'eid' not in Z.files or ky + 'eid' not in Z.files:
                continue
            com, ix, iy = np.intersect1d(Z[kx + 'eid'], Z[ky + 'eid'], return_indices=True)
            if len(com) == 0:
                continue
            bx = Z[kx + 'band'][ix].any(1); by = Z[ky + 'band'][iy].any(1)
            n += len(com); nb += int((bx | by).sum()); nbx += int(bx.sum()); nby += int(by.sum())
            nsp += int(((Z[kx + 'nsig'][ix] + Z[ky + 'nsig'][iy]) > 150).sum())
            noise.append([float(np.median(Z[kx + 'noise'])), float(np.median(Z[ky + 'noise']))])
            cnt['x'] += Z[kx + 'cnt']; cnt['y'] += Z[ky + 'cnt']
        if n:
            r[ch] = dict(n=n, p_band=nb / n, p_band_x=nbx / n, p_band_y=nby / n, p_spark=nsp / n, n_band=nb,
                         noise_med=np.median(noise, 0).round(1).tolist(),
                         chips_x=(cnt['x'].reshape(8, 64).sum(1) / max(cnt['x'].sum(), 1)).round(3).tolist(),
                         chips_y=(cnt['y'].reshape(8, 64).sum(1) / max(cnt['y'].sum(), 1)).round(3).tolist())
    NT[run] = r
S['ntof'] = NT

# bench at the n_TOF absolute threshold: the n_TOF noise is ~2x the bench's (RdClk_Div 6->4 on 23 July); recount the
# bench bands with the offset threshold scaled by the measured noise ratio (flatness cut unchanged)
BENCH_NOISE = {}
EQUIV = {}
if NT:
    for key, ch in (('sat_det3', 'A (det3)'), ('g_det7_long', 'D (det7)'), ('g_det2', 'B (det2)'), ('g_det6_long', 'C (det6)')):
        Z = np.load(B.scan_path(key)); nn = {}
        rows = [NT[r][ch]['noise_med'] for r in NT if ch in NT[r]]
        if not rows:
            continue
        nt = np.median(rows, 0)
        res = {}
        for j, v in enumerate('xy'):
            nb_ = float(np.median(Z[f'noise_{v}'])); nn[v] = nb_
            scale = nt[j] / nb_
            mc = Z[f'mc_{v}'].astype(np.float32); sd = Z[f'sd_{v}'].astype(np.float32); n = Z[f'noise_{v}']
            off = (np.abs(mc) > 5 * scale * n) & (sd < 2 * scale * n)
            res[v] = dict(noise_bench=nb_, noise_ntof=float(nt[j]), scale=float(scale),
                          p_band_scaled=float((off.reshape(len(off), 8, 64).sum(2) >= 8).any(1).mean()),
                          p_band_nominal=float(B.band_events(Z, v).any(1).mean()))
            del mc, sd, off
        EQUIV[key] = res
S['bench_at_ntof_threshold'] = EQUIV

# ---------------------------------------------------------------- summary numbers
for k in ORDER:
    r = P[k]
    S.setdefault('chambers', {})[k] = dict(
        label=LAB[k], n=r['n'], p_band=r['p_band'], p_band_x=r['p_band_x'], p_band_y=r['p_band_y'], p_spark=r['p_spark'],
        p_disch_any=r['p_disch_any'], p_band_given_spark=r['p_band_given_spark'], p_band_given_nospark=r['p_band_given_nospark'],
        xy_same_sign=r['xy']['same_sign'], xy_n=r['xy']['n_both'], patch_sign_coherence_med=r['patch_sign_coherence'],
        patch_width_mm=r['patch_width_mm'], edge_on_connector=r['edge_on_connector_boundary'],
        persist_lt1ms=[r['persistence'][0]['p_same_chip_next'], r['persistence'][0]['n_band_chips'], r['persistence'][0]['p_chip_base']],
        persist_1_3ms=[r['persistence'][1]['p_same_chip_next'], r['persistence'][1]['n_band_chips']],
        persist_pairs=dict(n=r['persist_pairs']['n'], same_sign=r['persist_pairs']['same_sign'], ratio_q=r['persist_pairs']['ratio_q']),
        after_disch=r['after_discharge_next_trigger'],
        since_discharge=dict(med_band=r['since_discharge']['median_dt_band'], med_ctrl=r['since_discharge']['median_dt_ctrl'],
                             frac_band=r['since_discharge']['frac_band'], frac_ctrl=r['since_discharge']['frac_ctrl'],
                             edges=r['since_discharge']['edges_s'], med_amp=r['since_discharge']['median_amp_band'],
                             corr=r['since_discharge']['corr_logdt_logamp']))
S['fdfcheck'] = {v: {k: FD[v][k] for k in ('n_events', 'n_band', 'words_compared', 'mismatches', 'block_layout_band', 'block_layout_quiet')}
                 for v in 'xy'}
S['fdfcheck_trailer45_corr_chip_mean'] = {v: [w['corr_with_chip_mean'] for w in FD[v]['header_trailer_words'] if w['part'] == 'trl'][0] for v in 'xy'}
S['fdfcheck_header_ks_max'] = {v: max(max(w['ks_same_chip'] or [0]) for w in FD[v]['header_trailer_words'] if w['part'] == 'hdr') for v in 'xy'}
json.dump(S, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)

# ---------------------------------------------------------------- HTML
def pct(x, d=1):
    return '–' if x is None else f'{100 * x:.{d}f} %'


def fmt2(x):
    return '–' if x is None else f'{x:.2f}'


C = S['chambers']; sd = C['sat_det3']
rows = ''.join(
    f"<tr><td>{html.escape(c['label'])}</td><td>{c['n']}</td><td><b>{pct(c['p_band'])}</b></td><td>{pct(c['p_spark'])}</td>"
    f"<td>{pct(c['p_band_given_spark'])} / {pct(c['p_band_given_nospark'])}</td>"
    f"<td>{'fixed patch' if k in LOCAL else 'whole plane'}</td><td>{pct(c['xy_same_sign'], 0)}</td>"
    f"<td>{c['patch_sign_coherence_med']['x']['50']:.2f} / {c['patch_sign_coherence_med']['y']['50']:.2f}</td>"
    f"<td>{c['patch_width_mm']['x']['50']:.0f} / {c['patch_width_mm']['y']['50']:.0f}</td>"
    f"<td>{fmt2(c['persist_lt1ms'][0])} ({c['persist_lt1ms'][1]}) vs {c['persist_lt1ms'][2]:.3f}</td></tr>"
    for k, c in C.items())
xc = ''.join(f"<tr><td>{LAB[p['a']]} + {LAB[p['b']]}</td><td>{p['n']}</td><td>{pct(p['p_b'])}</td><td>{pct(p['p_b_given_a'])}</td>"
             f"<td>{pct(p['p_a'])}</td><td>{pct(p['p_a_given_b'])}</td><td>{p['both']} (expected {p['both_expected']:.0f})</td></tr>" for p in pairs)
hvrows = ''
for n, R in sorted(HV.items()):
    for r in sorted(R, key=lambda r: (r['resist'], r['drift'])):
        hvrows += (f"<tr><td>{n}</td><td>{r['resist']}</td><td>{r['drift']}</td><td>{r['n']}</td><td>{pct(r['p_band'], 2)} ± {pct(r['err'], 2)}</td>"
                   f"<td>{pct(r['p_spark'], 2)}</td><td>{pct(r['p_band_x'], 2)} / {pct(r['p_band_y'], 2)}</td></tr>")
ntrows = ''.join(f"<tr><td>run_{run}</td><td>{ch}</td><td>{d['n']}</td><td><b>{pct(d['p_band'], 2)}</b></td><td>{pct(d['p_band_x'], 2)} / {pct(d['p_band_y'], 2)}</td>"
                 f"<td>{pct(d['p_spark'], 2)}</td><td>{d['noise_med'][0]} / {d['noise_med'][1]}</td></tr>"
                 for run, r in NT.items() for ch, d in r.items())
eqrows = ''.join(f"<tr><td>{LAB[k]}</td><td>{v}</td><td>{d['noise_bench']:.1f}</td><td>{d['noise_ntof']:.1f}</td>"
                 f"<td>{pct(d['p_band_nominal'], 2)}</td><td><b>{pct(d['p_band_scaled'], 2)}</b></td></tr>"
                 for k, r in EQUIV.items() for v, d in r.items())
fdx, fdy = S['fdfcheck']['x'], S['fdfcheck']['y']
sincerows = ''.join(f"<tr><td>{LAB[k]}</td><td>{c['since_discharge']['med_band']:.1f} s</td><td>{c['since_discharge']['med_ctrl']:.1f} s</td>"
                    f"<td>{c['since_discharge']['corr']:.2f}</td></tr>" for k, c in C.items() if c['since_discharge']['corr'] is not None)
VERDICT = open(os.path.join(HERE, 'verdict.html')).read() if os.path.exists(os.path.join(HERE, 'verdict.html')) else ''

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MX17 noise bands</title>
<style>
:root{{--bg:#fff;--ink:#1d1d1f;--mut:#5f6368;--line:#e2e2e2;--acc:#1f5fbf;--box:#f5f7fb}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#16181c;--ink:#e8e8ea;--mut:#a0a4aa;--line:#33363c;--acc:#7aa7ef;--box:#1f2329}}}}
:root[data-theme="dark"]{{--bg:#16181c;--ink:#e8e8ea;--mut:#a0a4aa;--line:#33363c;--acc:#7aa7ef;--box:#1f2329}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;max-width:1050px;margin:0 auto;padding:16px}}
h1{{font-size:1.5em}} h2{{font-size:1.15em;margin-top:2em;border-bottom:1px solid var(--line)}}
table{{border-collapse:collapse;font-size:13px;margin:.6em 0;display:block;overflow-x:auto}}
td,th{{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;white-space:nowrap}}
.box{{background:var(--box);border-left:4px solid var(--acc);padding:10px 14px;margin:1em 0}}
img{{max-width:100%;background:#fff}} .cap{{color:var(--mut);font-size:13px}} code{{font-size:13px}}
</style></head><body>
<h1>The noise bands on the MX17 chambers: what they are and what they are not</h1>
<p class="cap">rerun/bands (generated by make_report.py, 2026). Follow-up 1 of the det3 loss-categories study (rerun/satstudy). Existing data only: the June bench runs and the n_TOF runs.</p>
{VERDICT}
<h2>1. Per chamber (bench golden rows, every trigger read)</h2>
<table><tr><th>run</th><th>triggers</th><th>band</th><th>spark (&gt;150 strips)</th><th>band | spark / no spark</th><th>where</th>
<th>x,y same sign</th><th>sign coherence in patch (x/y, median)</th><th>patch width x/y [mm]</th><th>P(band again, next trigger &lt; 1 ms) (n) vs base</th></tr>{rows}</table>
<p class="cap">A band = ≥ 8 channels of one 64-channel chip whose window mean after common mode is &gt; 5σ off and flat (RMS &lt; 2σ) — the definition of
rerun/satstudy/bands.py. Sign coherence 1 = every channel of a patch shifted the same way; 0.5 = half up, half down.</p>

<h2>2. Not data corruption (a)</h2>
<p>The raw <code>.fdf</code> of golden-row file 003 (FEU 7 = x, FEU 8 = y) was walked word by word with an independent parser
(<code>fdfcheck.py</code>, word classes of rerun/decoder_audit) for {fdx['n_band']} band events and as many quiet ones per view:
<b>{fdx['words_compared'] + fdy['words_compared']:,} amplitude words compared, {fdx['mismatches'] + fdy['mismatches']} mismatches</b>.
Every chip block of a band event has the normal layout (4 header words, 64 data, 6 trailer words; band {fdx['block_layout_band']} /
{fdy['block_layout_band']}). The header words (trigger id, SCA cell) of a banded chip are distributed as in quiet events of the same chip
(largest KS distance {max(S['fdfcheck_header_ks_max'].values()):.2f}). The one trailer word that differs tracks the chip's own mean
amplitude (r = {S['fdfcheck_trailer45_corr_chip_mean']['x']:.2f} x, {S['fdfcheck_trailer45_corr_chip_mean']['y']:.2f} y): the FEU itself
digitised the shifted baseline. The whole-file decoder audit (rerun/decoder_audit, 2026-10-23) already had every word of every June file
identical between the two decoders with no incomplete events in this file. So the offsets are in the signal that reached the ADC.</p>

<h2>3. Where the bands sit (b)</h2>
<img src="figures/where.png" alt="band occupancy per strip">
<p class="cap">Per strip, the fraction of all triggers in which that strip is part of a band. Grey lines: 64-channel connector
(= DREAM chip, = M1 connector) boundaries. det3 is the same corner (x &lt; 50 mm, y &gt; 250 mm) in three runs on two FEU pairs; det7 is
x 275–375 mm, y 25–250 mm; det2, det4, det6 are spread over the whole plane.</p>
<p>Patch edges fall on a connector boundary in {pct(sd['edge_on_connector']['x'])} (x) / {pct(sd['edge_on_connector']['y'])} (y) of edges on
the golden row, against 3.1 % for a random edge (plane ends excluded); the patches are not cut by the readout connectors.
The drawings: the readout board gerbers (MX17_Geant/design/gerbers/readout_pcb) give the connector fields (pogo pads at ±100 mm along two
edges) and unlabelled top-copper areas at those two edges; the resistive strips run along y and are fed / grounded only by copper buses at
both y-ends (detector spec, Dylan 2026-08-07). No drawing in the repo or MX17_Documentation marks where the HV is fed onto the bus or where
the mesh is connected, so the patches cannot be tied to either. det7's patch is in the middle of the plane. The only defect information in hand is the board log in the n_TOF run_config:
det7 (board B) "had one or two bubbles, but the pillars underneath were still there, so just no caps"; det6 "a column of waves in the mesh
and maybe a spot with no pillar caps"; det3 nothing noted. The bubble positions are not recorded.</p>

<h2>4. Lifetime: about a millisecond</h2>
<img src="figures/persistence.png" alt="persistence">
<p>Triggers are random (Poisson, ~1.4–5.6 Hz), so some arrive within a millisecond of the previous one. On the golden row a band chip is
banded again in {sd['persist_lt1ms'][0]:.0%} of the next triggers that come &lt; 1 ms later ({sd['persist_lt1ms'][1]} chips), in
{sd['persist_1_3ms'][0]:.0%} at 1–3 ms ({sd['persist_1_3ms'][1]}), and at the base rate ({sd['persist_lt1ms'][2]:.1%}) beyond.
The other det3 runs give 0.27 (6-22, 22 chips) and 0.17 (6-28, 12 chips; 0.18 at 1–3 ms): pooled 28 of 64 = 0.44 against a base of 0.03. det2/det4/det6 show 0.4–0.5 at &lt; 1 ms. So a band lasts of order 1 ms: flat across the 1.9 µs window (as measured in
bands.py, τ ≫ 100 µs) but gone 3 ms later. In the persisting pairs the sign is kept in only {sd['persist_pairs']['same_sign']:.0%}
(det3; det4 {C['g_det4']['persist_pairs']['same_sign']:.0%}) and the amplitude ratio spreads from ×{sd['persist_pairs']['ratio_q']['25']:.1f}
to ×{sd['persist_pairs']['ratio_q']['75']:.1f} (quartiles): on det3 the baseline wanders on a ~ms scale rather than decaying from one
step. det7 does not persist ({C['g_det7_long']['persist_lt1ms'][0]:.2f} vs base {C['g_det7_long']['persist_lt1ms'][2]:.2f}).</p>
<p>With a ~1 ms lifetime, a 4 % occupancy means ~40 band onsets per second on det3's corner alone, untriggered: far more than the
discharges seen in triggered windows anywhere on det3 (4.3 % of 5.6 Hz).</p>

<h2>5. Bands and discharges (c, d)</h2>
<p><b>Same trigger:</b> on det3, P(band | spark trigger) = P(band | no spark) within errors (table 1). On det2 and det4 a spark trigger
carries a band in 35–38 % of cases (whole-plane family; these look like the aftermath of the discharge in the same window). On det6/det7 the
comparison is biased — a spark makes the channels non-flat, so the band flag cannot fire in the same trigger.</p>
<p><b>Sparkiest chamber:</b> det7 has both the most sparks ({pct(C['g_det7_long']['p_spark'])} of triggers, ≥ 8 strips of a chip &gt;
3000 ADC in {pct(C['g_det7_long']['p_disch_any'])}) and the most bands ({pct(C['g_det7_long']['p_band'])}); det6 is nearly as sparky
({pct(C['g_det6_long']['p_spark'])}) with a band rate like det2's ({pct(C['g_det6_long']['p_band'])}), and det3 has sparks
3–6 % with bands 4 %. Across five chambers the ranking is not monotonic: spark rate alone does not set the band rate.</p>
<p><b>Time since the last discharge on the same chip</b> (every trigger, spark-vetoed included; a discharge = ≥ 8 strips of the chip &gt; 3000 ADC):</p>
<table><tr><th>run</th><th>median, band chips</th><th>median, quiet chips</th><th>corr(log Δt, log |offset|)</th></tr>{sincerows}</table>
<p>Band chips are not closer in time to a (triggered) discharge on that chip than quiet chips are, and the band amplitude does not fall
with the time since one. Only within 3 ms is there a link: a chip that discharged in a trigger is banded in the next trigger if it comes
&lt; 3 ms later in {sd['after_disch'][0]['p_band_next']:.0%} of {sd['after_disch'][0]['n_disch_chips']} cases on det3
(det7 {C['g_det7_long']['after_disch'][0]['p_band_next']:.0%} of {C['g_det7_long']['after_disch'][0]['n_disch_chips']}), against
{sd['after_disch'][2]['p_band_next']:.1%} beyond 30 ms. A discharge can start a band, but triggered discharges are far too rare to make
most of them.</p>

<h2>6. Not pickup: two chambers in the same triggers are independent</h2>
<table><tr><th>chambers (same run, same triggers)</th><th>triggers</th><th>P(B)</th><th>P(B | band in A)</th><th>P(A)</th><th>P(A | band in B)</th><th>both</th></tr>{xc}</table>
<p>Pickup on the bench (mains, a noisy supply, the trigger) would put bands in both chambers of a run at once. It does not: the
coincidence is what independent chambers give (det6+det7 slightly above, 1.2×, consistent with shared showers).</p>

<h2>7. Field dependence: the June HV scans (f)</h2>
{'<img src="figures/hvscan.png" alt="HV scans">' if HV else '<p>(not run)</p>'}
<p>{' '.join(f"<b>{n}</b>: ×{f['factor_per_10V']:.2f} per 10 V ({f['v_range'][0]}–{f['v_range'][1]} V, {f['n_points']} points), doubling every {f['doubling_V']:.0f} V = gain^{f['power_of_gain']:.1f}." for n, f in S.get('hv_fits', {}).items())}
(gain: ×2 per 16.5 V on the bench, mx17-hv-slope-test.) The two det3 resist scans were taken at different times and interleave
on one curve, so this is not a drift in time.</p>
<table><tr><th>scan</th><th>resist [V]</th><th>drift [V]</th><th>triggers</th><th>band</th><th>spark</th><th>band x / y</th></tr>{hvrows}</table>

<h2>8. n_TOF (e)</h2>
<table><tr><th>run</th><th>chamber</th><th>triggers</th><th>band</th><th>x / y</th><th>spark</th><th>median noise x / y [ADC]</th></tr>{ntrows}</table>
<p class="cap">ntof_scan.py on lxplus, first 4 000 triggers of one or two files of several sub-runs per run, all eight FEUs, the bench
definition. run_149 and run_89: beam-off cosmics (scintillator trigger, ~25 Hz); run_145: beam. 20 samples × 60 ns window (bench: 32).
Chamber map: A = det3 (FEU 3/4), B = det2 (5/6), C = det6 (7/8), D = det7 (1/2).</p>
<p>The n_TOF noise is about twice the bench's (DREAM RdClk_Div 6 → 4 on 23 July, ntof_pedestal_qa), so 5σ there is a higher absolute
threshold. The bench counted again at the n_TOF threshold (offset and flatness cuts scaled by the noise ratio):</p>
<table><tr><th>bench run</th><th>view</th><th>bench noise</th><th>n_TOF noise</th><th>bench band rate, nominal</th><th>at the n_TOF threshold</th></tr>{eqrows}</table>

<p><b>Reading.</b> The n_TOF gain is lower than the bench's at the nominal voltages: by the HV ledger (ntof_july_analysis/hv_tradeoff:
gas +72.6 V, site −4.7 V, electronics +12.8/+34.8 V) chamber A at 540 V ≡ bench 437–459 V and D at 520 V ≡ bench 417–439 V. On the bench
det3 has no band at ≤ 455 V and det7 0.1–0.9 % at 400–440 V (section 7), so the n_TOF beam-off rates (A 0 of 77 k, D 0.14–0.23 %) are what a
gain-driven chamber effect predicts; the higher 5σ threshold alone would keep ~80 % of the bench rate (table above). det7's n_TOF bands sit on the
same connectors as on the bench (x connectors 7–8, y connectors 2–6). Under beam (run_145) A and D carry 0.6 % and 1.0 %, spread over many
chips: a beam-related baseline effect (flash recovery, ntof-mm-gamma-flash-response) on top, not the bench patch. run_89's B (det2) 1.3 % sits on
x connector 5 and the two y-plane ends — the whole-plane family's chamber, not characterised further.</p>

<h2>9. What this does not rule out (stated limitations)</h2>
<ul>
<li>The source of the ~ms disturbances is not seen directly: the DREAM records 1.9 µs per trigger, so an untriggered discharge,
a burst of ion feedback, or a micro-discharge at a defect between triggers is invisible. The HV monitor samples too slowly to resolve
ms current pulses. With no detector access, this cannot be closed by measurement.</li>
<li>The drawings in hand do not show the HV feed point on the resistive bus, the mesh connection or the protection network, so the
patch positions cannot be compared with them.</li>
<li>The persistence numbers rest on small samples at Δt &lt; 1 ms (12–96 band chips per run).</li>
<li>The n_TOF comparison changes several things at once (gas Ar/iso 90/10 vs 95/5, HV 520–540 V vs 490 V, noise ×2, 20 vs 32
samples, cabling and grounding), so it says the bands are a bench-condition phenomenon of these chambers, not which condition.</li>
<li>Only the golden row was checked word by word against the raw file (2 FEUs, 240 events); the other runs rest on the whole-file
decoder audit.</li>
</ul>
</body></html>"""
open(os.path.join(HERE, 'report.html'), 'w').write(page)
print('report.html written;', json.dumps({k: S[k] for k in ('cross_chamber', 'bench_at_ntof_threshold')}, indent=1)[:3000])
