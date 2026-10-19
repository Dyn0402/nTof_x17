"""Replay swapped overlays and log what _repair_pairs sees."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(__file__))
import headline as h
from sept26_prelim_analysis import intra_bench as ib
from ntof_tracking import wft_beam as wb
from wft import io as wio, reco as wr
from wft.calib import CalibrationBundle

ARM = sys.argv[1] if len(sys.argv) > 1 else 'C'
V = 'final_replace_profc'
S = h.scored(V)
M = pd.read_parquet(os.path.join(h.BASE, V, 'overlays.parquet')).set_index('oid')
o = S[(S['mode'] == 'overlay') & (S.sbin >= 4) & S.in_swapped_track & (S.arm == ARM)
      & (S.n_tracks >= 2)]
oids = sorted(set(o.oid))[:12]
bundle = str(ib.reco_dir(ARM) / 'calib_bundle_prelim')
cal = CalibrationBundle.load(bundle)
opts = dict(TWO_TRACK=True, TWO_TRACK_F=300.0, TWO_TRACK_F_CORROB=120.0,
            TWO_TRACK_T0='tied', TWO_TRACK_RESID_Z=8.0)
wr._worker_init(bundle, str(ib.out_dir() / f'xy_pairing_{ARM}_profc.json'), opts)
cfg = wb.beam_config(ARM, run=ib.RUN, sub_run=ib.SUBRUN)
pos = wio.strip_position_map(cfg)
D = ib.donors(ARM).set_index(['tag', 'event_id'])

LOG = []
_orig = wr._repair_pairs


def spy(out, cand_fits, pairing, dt):
    xs, ys = cand_fits['x'], cand_fits['y']
    for a in range(len(out)):
        for b in range(a + 1, len(out)):
            ia, ja, ga = out[a]
            ib_, jb, gb = out[b]
            rec = dict(ga=ga, gb=gb, dt=dt,
                       gate_swap=bool(wr._gate(xs[ia], ys[jb], dt) and wr._gate(xs[ib_], ys[ja], dt)))
            p0 = dict(pairing); p0.pop('prof', None)
            for nm, pp in (('lq', p0), ('all', pairing)):
                rec[f'{nm}_keep'] = wr.xy_pair_cost(xs[ia], ys[ja], pp, dt) + wr.xy_pair_cost(xs[ib_], ys[jb], pp, dt)
                rec[f'{nm}_swap'] = wr.xy_pair_cost(xs[ia], ys[jb], pp, dt) + wr.xy_pair_cost(xs[ib_], ys[ja], pp, dt)
            rec['qc'] = tuple(round(wr.constrained_charge(f)) for f in (xs[ia], ys[ja], xs[ib_], ys[jb]))
            rec['pd_keep'] = (round(wr.profile_distance(xs[ia], ys[ja], dt, 1.0), 3), round(wr.profile_distance(xs[ib_], ys[jb], dt, 1.0), 3))
            rec['pd_swap'] = (round(wr.profile_distance(xs[ia], ys[jb], dt, 1.0), 3), round(wr.profile_distance(xs[ib_], ys[ja], dt, 1.0), 3))
            rec['t0s'] = tuple(round(f.t0) for f in (xs[ia], ys[ja], xs[ib_], ys[jb]))
            rec['has_q'] = all(getattr(f, '_q', None) is not None for f in (xs[ia], xs[ib_], ys[ja], ys[jb]))
            rec['px'] = (xs[ia].p0, xs[ib_].p0)
            rec['py'] = (ys[ja].p0, ys[jb].p0)
            LOG.append(rec)
    return _orig(out, cand_fits, pairing, dt)


wr._repair_pairs = spy
rng = np.random.default_rng(0)
for tag in sorted(set(M.loc[oids].tag)):
    ids = [i for i in oids if M.loc[i].tag == tag]
    eids = set(M.loc[ids].a_eid) | set(M.loc[ids].b_eid)
    td = ib.TagData(ARM, tag, cfg, cal, pos, eids, rng, local_mm=16.0, local_mode='rescue',
                    overlay='replace')
    for i in ids:
        m = M.loc[i]
        pl, _ = td.payload(int(i), int(m.a_eid), int(m.b_eid), 'overlay')
        n0 = len(LOG)
        r = wr._worker_fit(pl)
        ta, tb = D.loc[(tag, m.a_eid)], D.loc[(tag, m.b_eid)]
        print(f'oid {i}: truth x {ta.x_p0:.1f}/{tb.x_p0:.1f}  y {ta.y_p0:.1f}/{tb.y_p0:.1f}'
              f'  q x {ta.x_q_sum:.0f}/{tb.x_q_sum:.0f} y {ta.y_q_sum:.0f}/{tb.y_q_sum:.0f}  ntr {r["n_tracks"]}')
        for rec in LOG[n0:]:
            print('    ', {k: (round(v, 2) if isinstance(v, float) else
                           tuple(round(z, 1) for z in v) if isinstance(v, tuple) else v)
                        for k, v in rec.items()})
        if len(LOG) == n0:
            print('     repair not called')
