import sys, json
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from c_lib import *
D = {k: load(rel) for k, rel in ROWS.items()}
V = {(k, p): view(D[k], p) for k in D for p in 'xy'}
def pool(keys, p):
    return {q: np.concatenate([V[(k, p)][q] for k in keys]) for q in ('U', 'W', 'w', 'tr')}
out = {'k': {}, 'ctrl': {}, 'coverage': {}}
for p in 'xy':
    for k in D:
        v = V[(k, p)]
        out['coverage'][f'{k}_{p}'] = dict(U=[float(np.percentile(v['U'], 1)), float(np.percentile(v['U'], 99))],
                                         W=[float(np.percentile(v['W'], 1)), float(np.percentile(v['W'], 99))], n=len(v['U']))
NB = 150
for n in (4, 6, 8):
    for p in 'xy':
        G = pool(['run1', 'run2'], p)
        mp_g = Maps(G['U'], G['W'], G['w'], G['tr'], n)
        mp = {'run1': Maps(*[V[('run1', p)][q] for q in ('U', 'W', 'w', 'tr')], n),
              'run2': Maps(*[V[('run2', p)][q] for q in ('U', 'W', 'w', 'tr')], n)}
        # 6-25 target with the pooled golden map
        r = kfit(V[('r625', p)], mp_g, boot=NB)
        r['vg'] = mp_g.vg; r['ratio_vg_vp'] = mp_g.vg / r['vp']
        r['cells_valid'] = int(np.isfinite(mp_g.A).sum())
        r['sd_A_fine_cells'] = float(np.nanstd(mp_g.Afine)); r['sd_A_plane_cells'] = float(np.nanstd(mp_g.Aplane))
        r['mean_err_A'] = float(np.nanmean(mp_g.Ae)); r['sd_S_cells'] = float(np.nanstd(mp_g.S)); r['mean_err_S'] = float(np.nanmean(mp_g.Se))
        out['k'][f'{n}_{p}'] = r
        # controls
        for src, dst in (('run1', 'run2'), ('run2', 'run1')):
            rc = kfit(V[(dst, p)], mp[src], boot=NB)
            rc['n_map'] = len(V[(src, p)]['U']); rc['vp_ratio'] = mp[src].vg / rc['vp']
            rc['sd_A_fine_cells'] = float(np.nanstd(mp[src].Afine)); rc['mean_err_A'] = float(np.nanmean(mp[src].Ae))
            out['ctrl'][f'{n}_{p}_{src}->{dst}'] = rc
        # golden pooled map on run1 / run2 directly (in-sample, k expected ~1)
        for dst in ('run1', 'run2'):
            out['ctrl'][f'{n}_{p}_pooled->{dst}(in-sample)'] = kfit(V[(dst, p)], mp_g, boot=0)
        print(n, p, {k: round(v, 3) for k, v in out['k'][f'{n}_{p}'].items() if isinstance(v, float)}, flush=True)
        for kk in ('run1->run2', 'run2->run1'):
            c = out['ctrl'][f'{n}_{p}_{kk}']; print('   ctrl', kk, {a: round(c[a], 3) for a in ('kp', 'kp_e', 'kf', 'kf_e', 'ks', 'ks_e')}, flush=True)
json.dump(out, open(f'{OUT}/kfit.json', 'w'), indent=1, default=float)
