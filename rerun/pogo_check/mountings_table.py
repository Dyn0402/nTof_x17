#!/usr/bin/env python3
"""mountings_table.py -- one line per (mounting, plane, card) from results/m_*.json (run_mountings.py + pogo_check.py):
verdict, links, enrichment, the largest pogo-shaped net. Writes results/mountings_summary.{json,md}."""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, 'results', 'm_*.json'))):
    d = json.load(open(f)); run, det = d['run_key'][2:].split('__')
    for c in d['cards']:
        pn = [n for n in c['nets'] if n['pogo_shaped']]
        rows.append(dict(run=run, det=det, plane=d['plane'], card=c['card'], nev=d['nev'], verdict=c['verdict'],
                         links=c['links'], enrichment=c['enrichment'], n_pogo_ch=len(c['pogo_channels']),
                         n_other_ch=len(c['other_channels']), biggest_pogo_net=max((n['n'] for n in pn), default=0),
                         n_pogo_nets_ge3=sum(n['n'] >= 3 for n in pn)))
json.dump(rows, open(os.path.join(HERE, 'results', 'mountings_summary.json'), 'w'), indent=1)
L = ['| night | det | plane | card | events | verdict | pogo-net ch (nets ≥ 3, largest) | other ch | enrichment |', '|' + '---|' * 9]
for r in rows:
    e = '—' if r['enrichment'] is None else f"{r['enrichment']:.1f}"
    L.append(f"| {r['run'].replace('mx17_', '')} | {r['det']} | {r['plane'].upper()} | {r['card']} | {r['nev']} | {r['verdict']} | "
             f"{r['n_pogo_ch']} ({r['n_pogo_nets_ge3']}, {r['biggest_pogo_net']}) | {r['n_other_ch']} | {e} |")
open(os.path.join(HERE, 'results', 'mountings_summary.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
