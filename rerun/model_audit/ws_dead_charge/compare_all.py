#!/usr/bin/env python3
"""compare_all.py -- every dead/short/silent channel list of every dataset, in STRIP POSITION, per physical chamber (F12).

Datasets: June bench rows (W3 census channel_layer/ + ws census/, and flank_census on the hits), SPS H4 det4 (uRWELL
census + flank), n_TOF chambers A-D = det3/det2/det6/det7 (flank on run_55 / run_79 / run_145 combined hits).
Labels per position: C census dead, S census short, D flank dead, Z silent (occupancy < 5 % of its neighbourhood),
B bundle mask (the promoted golden bundle), M H4 mask (channel_layer/det4_sps_h4.json).
Each flagged position is classified:
  detector   flagged in >= 2 datasets with DIFFERENT electronics (other FEU or channel number) -> transfers by position
  electronics flagged only where one FEU+channel sits behind it, and that FEU+channel is flagged on another chamber too
  one-off    flagged in one dataset only (time change, or the statistics of that dataset)
  other      the rest (same electronics in every dataset that flags it; cannot be separated from the chamber)
-> f12_table.json, f12_table.md, figures/f12_dead_map.png
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
from det4_sps_map import POSITION_MM, VIEW   # noqa: E402

C = np.arange(512)
BPOS = np.round(49.92 * (C // 64) + (63 - C % 64) * 0.78, 2)
GOLD = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}
NTOF = {'A': 'det3', 'B': 'det2', 'C': 'det6', 'D': 'det7'}
ROWS_ORDER = {'det2': ['ovn_det2', 'o22_det2', 'g_det2', 'o22_long_det2'],
              'det3': ['g_det3', 'o22_det3', 'o23_long_det3', 'g_det3_wknd', 'sat_det3'],
              'det4': ['o23_long_det4', 'g_det4'], 'det6': ['g_det6', 'g_det6_longer', 'g_det6_long'],
              'det7': ['g_det7', 'g_det7_longer', 'g_det7_long']}
MIN_JUDGED = 100            # a dataset/view judging fewer channels is listed but not used to classify


def load_datasets():
    D = []                  # dict(det, view, name, kind, feu, pos(512), flags{pos: set}, judged set(pos), elec{pos: (feu, ch)})
    for det, keys in ROWS_ORDER.items():
        for key in keys:
            fl = os.path.join(HERE, 'census', f'flank_{det}_{key}.json')
            if not os.path.exists(fl):
                continue
            F = json.load(open(fl))
            cen = None
            for cp in (os.path.join(REPO, 'rerun', 'channel_layer', f'{det}_{key}.json'), os.path.join(HERE, 'census', f'{det}_{key}.json')):
                if os.path.exists(cp):
                    cen = json.load(open(cp)); break
            for vi, v in enumerate('xy'):
                if not F.get(v):
                    continue
                feu = F['feus'][vi]
                flags = {}
                for c in F[v]['dead']:
                    flags.setdefault(BPOS[c], set()).add('D')
                for c in F[v].get('silent', []):
                    flags.setdefault(BPOS[c], set()).add('Z')
                judged = {BPOS[c] for c in F[v]['judged']}
                if cen is not None and v in cen:
                    jc = set(range(512)) - set(cen[v].get('unjudged', []))
                    if len(jc) >= MIN_JUDGED:
                        for c in cen[v]['dead']:
                            flags.setdefault(BPOS[c], set()).add('C')
                        judged |= {BPOS[c] for c in jc}
                    for c in cen[v]['shorts']:
                        flags.setdefault(BPOS[c], set()).add('S')
                if key == GOLD[det]:
                    b = json.load(open(os.path.expanduser(f'~/x17/cosmic_bench/model_audit/{det}/calib_bundle/bundle.json')))
                    for c in b['dead'][v]:
                        flags.setdefault(BPOS[c], set()).add('B')
                D.append(dict(det=det, view=v, name=f'bench {key} (FEU {feu})', kind='bench', feu=feu, flags=flags,
                              judged=judged, elec={BPOS[c]: (feu, int(c)) for c in C}, n_judged=len(F[v]['judged'])))
    for tag in ('run56_m10V', 'run63_flat', 'run71_raw'):
        F = json.load(open(os.path.join(HERE, 'census', f'flank_sps_{tag}.json')))
        U = json.load(open(os.path.join(HERE, 'census', f'sps_{tag}.json')))
        M = json.load(open(os.path.join(REPO, 'rerun', 'channel_layer', 'det4_sps_h4.json')))
        P = np.round(POSITION_MM, 2)
        for v in 'xy':
            flags = {}
            for c in F[v]['dead']:
                flags.setdefault(P[c], set()).add('D')
            for c in U[v]['dead']:
                flags.setdefault(P[c], set()).add('C')
            for c in M[v]['dead']:
                flags.setdefault(P[c], set()).add('M')
            judged = {P[c] for c in F[v]['judged']} | {P[c] for c in U[v]['judged']}
            D.append(dict(det='det4', view=v, name=f'SPS {tag} (FEU 03)', kind='sps', feu=103, flags=flags, judged=judged,
                          elec={P[c]: (103, int(c)) for c in C if VIEW[c] == v}, n_judged=len(judged)))
    for run in ('run_55', 'run_79', 'run_145'):
        fp = os.path.join(HERE, 'census', f'flank_ntof_{run}.json')
        if not os.path.exists(fp):
            continue
        F = json.load(open(fp))
        for arm, det in NTOF.items():
            for v in 'xy':
                r = F[arm][v]; pos = np.round(np.array(r['pos'], float), 2)
                flags = {}
                for c in r['dead']:
                    flags.setdefault(pos[c], set()).add('D')
                for c in r.get('silent', []):
                    flags.setdefault(pos[c], set()).add('Z')
                D.append(dict(det=det, view=v, name=f'n_TOF {arm} {run} (FEU {r["feu"]})', kind='ntof', feu=r['feu'] + 200,
                              flags=flags, judged={pos[c] for c in r['judged']},
                              elec={pos[c]: (r['feu'] + 200, int(c)) for c in C if np.isfinite(pos[c])}, n_judged=len(r['judged'])))
    return D


def classify(D):
    out = {}
    # electronics flags: (feu, ch) -> set of chambers that flag it (bench FEUs are shared between chambers over June)
    by_elec = {}
    for d in D:
        for p, lab in d['flags'].items():
            if lab & {'D', 'C', 'Z'} and p in d['elec']:
                by_elec.setdefault(d['elec'][p], set()).add(d['det'])
    for det in GOLD:
        for v in 'xy':
            ds = [d for d in D if d['det'] == det and d['view'] == v]
            use = [d for d in ds if d['n_judged'] >= MIN_JUDGED]
            allp = sorted({p for d in use for p, lab in d['flags'].items() if lab & {'D', 'C', 'S', 'Z'}})
            rows = []
            for p in allp:
                where = [d for d in use if p in d['flags'] and d['flags'][p] & {'D', 'C', 'S', 'Z'}]
                judged_in = [d for d in use if p in d['judged']]
                elecs = {d['elec'].get(p) for d in where}
                other_chambers = set().union(*[by_elec.get(e, set()) for e in elecs]) - {det} if elecs else set()
                if len(where) >= 2 and len(elecs) >= 2:
                    cls = 'detector'
                elif where and other_chambers and len(elecs) == 1:
                    cls = 'electronics'
                elif len(where) <= 1:
                    cls = 'one-off'
                else:
                    cls = 'other'
                rows.append(dict(pos=float(p), cls=cls, flagged_in=[d['name'] for d in where], judged_in=len(judged_in),
                                 labels={d['name']: ''.join(sorted(d['flags'][p])) for d in ds if p in d['flags']},
                                 electronics=sorted([list(e) for e in elecs if e]), same_electronics_other_chambers=sorted(other_chambers)))
            out[f'{det}_{v}'] = dict(datasets=[dict(name=d['name'], n_judged=d['n_judged'],
                                                    n_flagged=sum(1 for lab in d['flags'].values() if lab & {'D', 'C', 'S', 'Z'}))
                                               for d in ds], positions=rows,
                                     counts={k: sum(r['cls'] == k for r in rows) for k in ('detector', 'electronics', 'one-off', 'other')})
    return out


def figure(D, T):
    fig, axs = plt.subplots(5, 2, figsize=(18, 22))
    col = {'C': '#d62728', 'S': '#9467bd', 'D': '#1f77b4', 'Z': '#7f7f7f', 'B': '#2ca02c', 'M': '#2ca02c'}
    for i, det in enumerate(GOLD):
        for j, v in enumerate('xy'):
            ax = axs[i, j]
            ds = [d for d in D if d['det'] == det and d['view'] == v]
            for r, d in enumerate(ds):
                jp = np.array(sorted(d['judged']))
                if len(jp):
                    ax.scatter(jp, np.full(len(jp), r), s=2, color='#dddddd', marker='|')
                for p, lab in d['flags'].items():
                    for k, l in enumerate(sorted(lab)):
                        ax.scatter([p], [r + 0.18 * (k - 1)], s=14 if l not in 'BM' else 30, color=col[l],
                                   marker='o' if l not in 'BM' else 's', facecolors='none' if l in 'BM' else col[l])
            ax.set_yticks(range(len(ds))); ax.set_yticklabels([d['name'] for d in ds], fontsize=7)
            ax.set_xlim(-2, 402); ax.set_ylim(-0.7, len(ds) - 0.3)
            for k in range(1, 8):
                ax.axvline(k * 49.92, color='k', lw=.3, alpha=.4)
            c = T[f'{det}_{v}']['counts']
            ax.set_title(f'{det} {v.upper()} view — detector {c["detector"]}, electronics {c["electronics"]}, one-off {c["one-off"]}, other {c["other"]}',
                         fontsize=9)
            if i == 4:
                ax.set_xlabel('strip position in the chamber frame [mm] (lines = 64-strip connectors)')
    hs = [plt.Line2D([], [], color=col[l], marker='o' if l not in 'BM' else 's', ls='', mfc='none' if l in 'BM' else col[l]) for l in 'CSDZB']
    fig.legend(hs, ['census dead (M3/uRWELL crossing)', 'census short', 'flank dead (neighbours fire, it does not)',
                    'silent (occupancy < 5 % of neighbours)', 'bench bundle mask / H4 mask'], loc='upper center', ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(HERE, 'figures', 'f12_dead_map.png'), dpi=110)


def main():
    D = load_datasets()
    T = classify(D)
    json.dump(T, open(os.path.join(HERE, 'f12_table.json'), 'w'), indent=1)
    L = ['| chamber view | datasets (judged ch / flagged) | detector | electronics | one-off | other |', '|---|---|---|---|---|---|']
    for k, t in T.items():
        ds = '; '.join(f"{d['name']}: {d['n_judged']}/{d['n_flagged']}" for d in t['datasets'])
        c = t['counts']
        L.append(f"| {k} | {ds} | {c['detector']} | {c['electronics']} | {c['one-off']} | {c['other']} |")
    open(os.path.join(HERE, 'f12_table.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    for k, t in T.items():
        for cls in ('detector', 'electronics'):
            ps = [r for r in t['positions'] if r['cls'] == cls]
            print(f"{k} {cls}: " + ', '.join(f"{r['pos']:.1f}[{len(r['flagged_in'])}]" for r in ps))
    if '--no-fig' not in sys.argv:
        figure(D, T)


if __name__ == '__main__':
    main()
