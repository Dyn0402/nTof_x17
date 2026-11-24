"""attribute.py -- which switch made the late (>= 16 ms) differences of `all` vs stage3_r2_final (ad hoc, T2)."""
import sys; sys.path.insert(0, '/media/ucla/work/claude-jobs/eb8200b3/T2_ref_compare')
import numpy as np, pandas as pd, compare as C
C.COLS = sorted(set(C.COLS + ['x_n_seed', 'y_n_seed', 'x_n_strips', 'y_n_strips', 'x_rank', 'y_rank', 'n_cand_x', 'n_cand_y']))
cfg = C.read_view(sys.argv[1]); cfg = cfg[cfg.arm.isin(['A', 'C'])]
ref = C.read_view(sys.argv[2], keep=C.acquisitions(cfg))
for arm in 'AC':
    for lo, hi in ((16, 75), (1, 12)):
        c = cfg[(cfg.arm == arm) & (cfg.ms >= lo) & (cfg.ms < hi)]; r = ref[(ref.arm == arm) & (ref.ms >= lo) & (ref.ms < hi)]
        p, oa, ob = C.pair(c, r)
        evr = r[r.gated].groupby(C.KEY).size()
        oa_k = oa.set_index(C.KEY).index
        in_ref_event = oa_k.isin(evr.index)
        print(f'--- {arm} {lo}-{hi} ms: paired {len(p)} onlyC {len(oa)} onlyR {len(ob)}')
        for nm, t in (('paired', p), ('onlyC', oa), ('onlyR', ob)):
            if not len(t): continue
            print(f'  {nm:7s} |tanx|<0.08 {np.mean(t.tanx.abs() < 0.08):.2f}  x_n_strips<=2 {np.mean(t.x_n_strips <= 2):.2f} '
                  f'y_n_strips<=2 {np.mean(t.y_n_strips <= 2):.2f} track_id>0 {np.mean(t.track_id > 0):.2f} '
                  f'wall {t.wall.mean():.2f} median x_n_seed {np.nanmedian(t.x_n_seed):.0f}')
        print(f'  onlyC tracks in an event the ref also has a gated track in: {in_ref_event.mean():.2f}')
        # only-ref tracks: present in cfg ungated? (lost to the gate vs not reconstructed)
        allc = c.set_index(C.KEY).index
        print(f'  onlyR tracks whose event has any (ungated) cfg track: {ob.set_index(C.KEY).index.isin(allc).mean():.2f}')
