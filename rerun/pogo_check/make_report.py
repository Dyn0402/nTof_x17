#!/usr/bin/env python3
"""make_report.py: figures + report.html for the pogo check, from out/*.npz (accumulate) and results/*.json
(pogo_check).  Re-run after either changes:  ../../.venv/bin/python make_report.py"""
import glob, html, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layout, pogo_check as PC

FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
RES = sorted(glob.glob(os.path.join(HERE, 'results', '*.json')))
A = layout.pogo_neighbours(); P = layout.plane_positions()


def load(key):
    return np.load(os.path.join(HERE, 'out', key + '.npz'), allow_pickle=True)


# 1. the naive spatial correlation: C against strip distance, every plane
fig, ax = plt.subplots(figsize=(8, 4.6))
dd = np.arange(1, 31)
for f in RES:
    key = os.path.basename(f)[:-5]; T = PC.pair_tables(load(key)); up = np.triu(T['ok'], 1)
    med = [np.median(T['C'][up & (T['d'] == k)]) for k in dd]
    ax.plot(dd, med, color='0.6', lw=1)
T = PC.pair_tables(load('g_det7_long_x')); up = np.triu(T['ok'], 1)
on1 = (P[:, None, 0] == 0) & (P[None, :, 0] == 0)
m = up & on1 & (A > 0) & (T['d'] <= 30)
ax.scatter(T['d'][m] + np.random.uniform(-.2, .2, m.sum()), T['C'][m], s=7, color='#1f5fbf', alpha=.6,
           label='det7 X card 1: pogo-neighbour pairs')
ax.axhline(PC.C_MIN, color='#d0263a', ls=':', lw=1)
ax.set_xlabel('strip distance d'); ax.set_ylabel('amplitude correlation C')
ax.set_title('Neighbouring strips share charge: C falls with d (grey = median, each plane)\n'
             'a pad short sits far above its own distance (C ~ 1 at d = 5, 10, 15)', fontsize=9)
ax.plot([], [], color='0.6', label='median C at d, 10 bench planes'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'corr_vs_distance.png'), dpi=140); plt.close(fig)

# 2. jump test: excess vs ratio spread, null (cross-card) against pogo neighbours
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, key, title in [(axs[0], 'g_det7_long_x', 'det7 X (g_det7_long)'), (axs[1], 'sat_det3_x', 'det3 X (sat_det3), clean')]:
    T = PC.pair_tables(load(key)); up = np.triu(T['ok'], 1) & (T['J'] >= PC.N_MIN) & (T['d'] >= PC.D_MIN)
    for msk, col, lab in [(up & T['cross'], '0.6', 'different cards (null: cannot be a pad short)'),
                          (up & ~T['cross'] & (A == 0), '#e8890c', 'same card, not pogo neighbours'),
                          (up & ~T['cross'] & (A > 0), '#1f5fbf', 'pogo neighbours')]:
        ax.scatter(T['sdj'][msk], T['es'][msk], s=6, color=col, alpha=.55, label=f'{lab} ({msk.sum()})')
    ax.axhline(PC.E_MIN, color='#d0263a', ls=':', lw=1); ax.axvline(PC.SD_TWIN, color='#d0263a', ls=':', lw=1)
    ax.set_xlim(0, 1.6); ax.set_xlabel('sd ln(A_a / A_b) over the jump events'); ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
axs[0].set_ylabel('jump excess e_sym')
fig.suptitle('Jump test: a and b fire with quiet strips between them (top) and read the same charge (left)', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'jump_test.png'), dpi=140); plt.close(fig)

# 3. det7 X card 1 on the pad field: track-confirmed links
R = json.load(open(os.path.join(HERE, 'results', 'g_det7_long_x.json')))
pos, gnd, nc = layout.card_layout()
fig, ax = plt.subplots(figsize=(11, 4.4))
for (c, r) in gnd: ax.add_patch(plt.Rectangle((r - .3, c - .3), .6, .6, color='0.45'))
pogo = set(R['cards'][0]['pogo_channels']); other = set(R['cards'][0]['other_channels'])
dense = {3, 5, 8, 9, 10, 13, 14, 15, 18, 19, 20, 23, 24, 25, 28, 29, 30, 33, 34, 35, 38, 39, 40, 43, 44, 45, 47, 48, 49,
         51, 52, 53, 57, 58}
for ch, (c, r) in pos.items():
    fc = '#1f5fbf' if ch in pogo else ('#e8890c' if ch in other else ('#f3c4c9' if ch in dense else '#e3e8ee'))
    ax.add_patch(plt.Circle((r, c), .36, color=fc, ec='0.5', lw=.4))
    ax.text(r, c, ch, ha='center', va='center', fontsize=5, color='w' if ch in pogo else 'k')
for L in R['links']:
    if L['cross'] or L['a'] >= 256 or L['b'] >= 256 or not (L['twin'] or L['excess']) or not A[L['a'], L['b']]:
        continue
    (c0, r0), (c1, r1) = pos[L['a']], pos[L['b']]
    ax.plot([r0, r1], [c0, c1], color='#1f5fbf', lw=2.2, alpha=.8, zorder=0)
ax.set_xlim(-.8, 29.8); ax.set_ylim(9.8, -.8); ax.set_aspect('equal')
ax.set_xlabel('pogo row (1 mm, along the board edge)'); ax.set_ylabel('pogo col (radial)')
ax.set_title('det7 X card 1, track-confirmed links between pogo neighbours. blue = pogo-shaped net, orange = other net, '
             'pink = census "dense block" (fires only in saturated big events)', fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'det7x_pad_field.png'), dpi=150); plt.close(fig)

# report
rows = []
for f in RES:
    R = json.load(open(f)); nl = R['null']
    for c in R['cards']:
        enr = '—' if c['enrichment'] is None else f"{c['enrichment']:.1f}"
        rows.append(f"<tr><td>{html.escape(R['detector'])}</td><td>{R['run_key']}</td><td>{R['plane'].upper()}</td>"
                    f"<td>{c['card']} (conn {c['connectors'][0]}–{c['connectors'][1]})</td><td><b>{c['verdict']}</b></td>"
                    f"<td>{c['links']}</td><td>{len(c['pogo_channels'])}</td><td>{len(c['other_channels'])}</td>"
                    f"<td>{enr}</td>"
                    f"<td>{len(c['adjacent_twins'])}</td><td>{nl['cross_links']} / {nl['cross_pairs_tested']}</td></tr>")
method = open(os.path.join(HERE, 'METHOD.md')).read()
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pogo-field short check</title>
<meta name="description" content="A general check for M1 pogo-field pad shorts that is not fooled by charge sharing between neighbouring strips; applied to the June bench fleet.">
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:16px;color:#1d1f22;background:#fbfbfa}}
table{{border-collapse:collapse;font-size:.85rem}}td,th{{border-bottom:1px solid #ddd;padding:3px 8px;text-align:left}}
img{{max-width:100%}}.box{{border-left:4px solid #1f5fbf;background:#fff;padding:8px 14px;border:1px solid #ddd}}
pre{{white-space:pre-wrap;font-size:.8rem;background:#fff;border:1px solid #ddd;padding:10px}}</style></head><body>
<h1>Pogo-field short check</h1>
<div class="box"><b>Verdict.</b> Of the 10 bench planes (20 M1 cards) only <b>det7 X card 1</b> (g_det7_long, connectors 1–4)
shows a pogo-field fault: 10 nets of 34 channels, each a chain of pogo-grid neighbours whose strips fire together across quiet
strips with identical amplitudes. No link crosses between cards in that plane. det7 Y and det6 Y have many strongly correlated pairs
<i>between</i> the two cards (261 and 361 links). No pad short can do that, so something else is acting on their edge connectors.
That is reported as OTHER, not as pogo. det3 is clean.<br>
<b>Correction to the det7 study.</b> The census "dense 2-D block" (cols 7–9, rows 0–14: channels 3–58) is not confirmed by tracks.
Those channels fired (a hit &gt; 50 ADC) in 5 of ~35 k clean events, where their neighbours fired ~1600 times. They co-move only in
saturated big events (median ~430 hits). Treat them as not reading tracks, not as shorted nets.<br>
<b>Not resolved:</b> on det7 X connector 4 (ch 192–232) a lattice of <i>diagonal</i> pogo neighbours reads the same charge. Half of
those channels are noisy, and the diagonal pad gap is the widest (0.61 mm), so pin bridging is the less likely cause. It is left as OTHER.
<br><b>Consequence:</b> g_det7_long carries an asterisk. Quote det7 X results from card 2 (connectors 5–8) only.</div>
<h2>Figures</h2>
<p><img src="figures/corr_vs_distance.png"><br>The distance-matched null for plain correlation: every plane's median C at each strip distance.</p>
<p><img src="figures/jump_test.png"><br>The track-proof test. Grey pairs sit on different cards, so they can never be a pad short. Links are the pairs in the top-left box.</p>
<p><img src="figures/det7x_pad_field.png"></p>
<h2>Per card</h2>
<table><tr><th>detector</th><th>run</th><th>plane</th><th>card</th><th>verdict</th><th>links</th><th>pogo-net ch</th><th>other ch</th><th>enrichment</th><th>adjacent twins</th><th>cross-card links / tested</th></tr>
{''.join(rows)}</table>
<h2>Method</h2><pre>{html.escape(method)}</pre></body></html>"""
open(os.path.join(HERE, 'report.html'), 'w').write(page)
print('report.html written')
