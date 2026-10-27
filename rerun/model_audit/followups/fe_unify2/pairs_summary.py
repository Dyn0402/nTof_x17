#!/usr/bin/env python3
"""pairs_summary.py -- FE: n_TOF close-pair efficiency of wft_beam under the one reconstruction (two-track fit, 4-pitch
minimum separation) against the same chain with the two-track fit off, on the overlay bench of
sept26_prelim_analysis/intra_bench.py (run_145 stat090_0000, A and C, is2_v2 bundles; condor, fe_pairs.sub).
'found' = fraction of donor tracks recovered (eff_tracks); '>= 2 tracks' = overlays reconstructed as two gated tracks.  -> results/pairs.json/.md"""
import json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser('~/x17/cosmic_bench/model_audit/fe_unify2/pairs/out')


def main():
    a = pd.read_csv(os.path.join(D, 'summary_fe_unified.csv')); b = pd.read_csv(os.path.join(D, 'summary_fe_onetrack.csv'))
    a, b = a[a['mode'] == 'overlay'], b[b['mode'] == 'overlay']
    m = a.merge(b, on=['arm', 'cls', 'sep_lo'], suffixes=('_tt', '_one'))
    out, L = {}, ['| arm | class | separation | n overlays | donor tracks found: two-track | one-track | events with >= 2 tracks: two-track | one-track |',
                  '|---|---|---|---|---|---|---|---|']
    for arm in 'AC':
        for cls in ('coincident', 'between', 'offset', 'all'):
            for lo, hi, lab in ((0, 12, '< 12 mm'), (12, 24, '12-24 mm'), (24, 1e9, '>= 24 mm')):
                s = m[(m.arm == arm) & (m.sep_lo >= lo) & (m.sep_lo < hi) & ((m.cls == cls) if cls != 'all' else True)]
                if not len(s):
                    continue
                f = lambda c, n: float((s[c] * s[n]).sum() / s[n].sum())
                r = dict(n=int(s.n_events_tt.sum()), found_tt=f('eff_tracks_tt', 'n_events_tt'), found_one=f('eff_tracks_one', 'n_events_one'),
                         merged_tt=f('n_tracks_ge2_tt', 'n_events_tt'), merged_one=f('n_tracks_ge2_one', 'n_events_one'))
                out[f'{arm}|{cls}|{lab}'] = r
                L.append(f'| {arm} | {cls} | {lab} | {r["n"]} | {100 * r["found_tt"]:.1f} % | {100 * r["found_one"]:.1f} % | '
                         f'{100 * r["merged_tt"]:.1f} % | {100 * r["merged_one"]:.1f} % |')
    md = '\n'.join(L) + '\n'; print(md)
    json.dump(out, open(os.path.join(HERE, 'results', 'pairs.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'results', 'pairs.md'), 'w').write(md)


if __name__ == '__main__':
    main()
