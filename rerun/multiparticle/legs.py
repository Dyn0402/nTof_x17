#!/usr/bin/env python3
"""legs.py -- what the second particles in det3's multi-particle events are (HANDOFF satstudy follow-up 2, question 1).

    ../../.venv/bin/python legs.py        -> ~/x17/cosmic_bench/satstudy/multiparticle/{legs,events,mixed}.parquet

Sample: the 187 'M' events of multitrack.py (interior M3 recipe ray through det3; >= 2 particle-like production
candidates, time extent <= 24 samples, in a view). Per view the M3 leg is the particle candidate nearest the ray at the
mesh within 5 mm (as multitrack.sub); every other particle candidate is an 'other' leg. An event is 'full' (2x2 with
the M3 leg found in both views) when each view holds the M3 leg plus >= 1 other; its second particle is then a 3-D
line (the other x leg with the other y leg, the brightest if there are two). Geometry is the forward fit's (p0, ladder
tan), never hit times.

Per leg: separation from the M3 leg at mid-gap, angle difference, t0 difference, charge per unit 3-D path relative to
the clean-single median ('MIP units'), time extent, depth coverage. Per full event: the 3-D opening angle to the M3
ray (truth) and to det3's own M3 leg (what an X17 analysis measures), closest approach and its height Z, the second
particle's extrapolation to the four M3 planes (inside the region recipe rays cross?), the M3 cluster counts there and
whether an alternative 4-layer M3 combination (m3feat m3_alt_*) lines up with it.

Nulls: 'mixed' = the M3 ray of one clean single event against the M3 ray of another, with the same det3 separation
range: what two unrelated cosmic muons would give for opening angle, closest approach and Z.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mplib as M  # noqa: E402

R_MM = 5.0


def singles_ref(C, D, R):
    """Clean singles: the detected rank-0 leg per view -> t0, q per path (for MIP units), the x/y t0 difference."""
    ids = D.index[(D.cls == 'S') & D['prod']]
    s = C[C.event_id.isin(ids) & (C['rank'] == 0)].pivot(index='event_id', columns='plane',
                                                         values=['t0', 'q_sum', 'tl', 'ext', 'q_uend', 'n_strips'])
    s = s.dropna()
    path = np.sqrt(1 + s.tl.x ** 2 + s.tl.y ** 2)
    return s, dict(qx_med=float(np.median(s.q_sum.x / path)), qy_med=float(np.median(s.q_sum.y / path)),
                   t0_med=float(np.median(0.5 * (s.t0.x + s.t0.y))), t0_rsig=M.rsig(0.5 * (s.t0.x + s.t0.y)),
                   dt0xy_med=float(np.median(s.t0.x - s.t0.y)), dt0xy_rsig=M.rsig(s.t0.x - s.t0.y),
                   n=int(len(s)))


def main():
    L = M.ladder(); C = M.candidates(L); R = M.rays(); a = M.align()
    D = pd.read_parquet(os.path.join(M.S.OUT, 'multitrack_det3.parquet'))
    T = pd.read_parquet(os.path.join(M.S.OUT, 'multi_sub.parquet'))
    F = M.m3feat()
    s, ref = singles_ref(C, D, R)
    Mi = D.index[D.cls == 'M']
    P = C[C.particle & C.event_id.isin(Mi)].copy()
    for v in 'xy':
        P.loc[P.plane == v, 'mesh'] = R[f'mesh_{v}'].reindex(P.event_id[P.plane == v]).to_numpy()
        P.loc[P.plane == v, 'tref'] = R[f'tan_{v}'].reindex(P.event_id[P.plane == v]).to_numpy()
        P.loc[P.plane == v, 'qmip'] = (P.q_sum / ref[f'q{v}_med'])[P.plane == v]
    P['res'] = P.p0 - P.mesh
    rows = []
    for (e, v), g in P.groupby(['event_id', 'plane']):
        near = g[g.res.abs() <= R_MM]
        m3 = near.loc[near.res.abs().idxmin()] if len(near) else None
        for i, c in g.iterrows():
            r = dict(event_id=int(e), plane=v, rank=int(c['rank']), p0=c.p0, tl=c.tl, mid=c.mid, t0=c.t0, q=c.q_sum,
                     qmip_raw=c.qmip, ext=c.ext, q_uend=c.q_uend, q_u50=c.q_u50, q_u90=c.q_u90, n_strips=c.n_strips,
                     chi2dof=c.chi2 / max(c.dof, 1), n_dropped=c.n_dropped, res=c.res, tref=c.tref,
                     is_m3=m3 is not None and i == m3.name, m3_found=m3 is not None, n_in_view=len(g))
            if m3 is not None and i != m3.name:
                r.update(sep=c.mid - m3.mid, dtan=c.tl - m3.tl, dt0=c.t0 - m3.t0, qratio=c.q_sum / m3.q_sum)
            rows.append(r)
    Lg = pd.DataFrame(rows)
    Lg['sub'] = T['sub'].reindex(Lg.event_id).to_numpy(); Lg['topo'] = T['topo'].reindex(Lg.event_id).to_numpy()
    # ---- full events: 3-D second particle
    boxes = M.m3_boxes(M.rays_all(), a)
    ev = []
    for e, g in Lg.groupby('event_id'):
        r = dict(event_id=int(e), sub=T.loc[e, 'sub'], topo=T.loc[e, 'topo'])
        legs = {}
        for v in 'xy':
            gv = g[g.plane == v]
            m = gv[gv.is_m3]; o = gv[~gv.is_m3]
            legs[v] = (m.iloc[0] if len(m) else None, o.loc[o.q.idxmax()] if len(o) else None)
            r[f'n{v}'] = len(gv)
        r['full'] = all(legs[v][0] is not None and legs[v][1] is not None for v in 'xy')
        rr = R.loc[e]
        r.update(mesh_x=rr.mesh_x, mesh_y=rr.mesh_y, tan_x=rr.tan_x, tan_y=rr.tan_y)
        f = F.loc[e] if e in F.index else None
        if f is not None:
            for k in ('m3_extra', 'm3_nalt_x', 'm3_nalt_y', 'm3_spark') + tuple(f'm3_ncl_{v}{l}' for v in 'xy' for l in range(4)):
                r[k] = f[k]
        if r['full']:
            (mx, ox), (my, oy) = legs['x'], legs['y']
            r.update(o_p0x=ox.p0, o_p0y=oy.p0, o_tx=ox.tl, o_ty=oy.tl, o_t0x=ox.t0, o_t0y=oy.t0,
                     m_p0x=mx.p0, m_p0y=my.p0, m_tx=mx.tl, m_ty=my.tl, m_t0x=mx.t0, m_t0y=my.t0,
                     o_qx=ox.q, o_qy=oy.q, m_qx=mx.q, m_qy=my.q, o_extx=ox.ext, o_exty=oy.ext,
                     o_uendx=ox.q_uend, o_uendy=oy.q_uend, m_uendx=mx.q_uend, m_uendy=my.q_uend,
                     sepx=ox.mid - mx.mid, sepy=oy.mid - my.mid)
            p2, d2 = M.line(ox.p0, oy.p0, ox.tl, oy.tl)
            p1, d1 = M.line(rr.mesh_x, rr.mesh_y, rr.tan_x, rr.tan_y)          # M3 truth
            pm, dm = M.line(mx.p0, my.p0, mx.tl, my.tl)                         # det3's own M3 leg
            r['dca'], r['zv'], r['open'] = M.dca(p1, d1, p2, d2)
            r['dca_reco'], r['zv_reco'], r['open_reco'] = M.dca(pm, dm, p2, d2)
            path_o = np.sqrt(1 + ox.tl ** 2 + oy.tl ** 2); path_m = np.sqrt(1 + mx.tl ** 2 + my.tl ** 2)
            r['o_qmip'] = 0.5 * (ox.q / ref['qx_med'] + oy.q / ref['qy_med']) / path_o
            r['m_qmip'] = 0.5 * (mx.q / ref['qx_med'] + my.q / ref['qy_med']) / path_m
            r['o_lqxy'] = np.log(ox.q / oy.q)
            r['dt0'] = 0.5 * ((ox.t0 - mx.t0) + (oy.t0 - my.t0))
            r['o_dt0xy'] = ox.t0 - oy.t0
            # the second particle at the M3 planes (M3 frame) and inside the recipe region?
            xm, ym = M.raw_to_m3(ox.p0, oy.p0, a); txm, tym = M.tan_raw_to_m3(ox.tl, oy.tl, a)
            for z in M.ZM3:
                X, Y = xm + txm * (z - M.Z0), ym + tym * (z - M.Z0); (bx, by) = boxes[z]
                r[f'in_{int(z)}'] = bool(bx[0] <= X <= bx[1] and by[0] <= Y <= by[1])
            # an alternative M3 4-layer combination on the second particle (M3 x view feeds raw y at theta ~ 90)
            for mv, (rv, pv, tv) in (('x', ('y', oy.p0, oy.tl)), ('y', ('x', ox.p0, ox.tl))):
                best = np.inf
                if f is not None:
                    for alt in json.loads(f[f'm3_alt_{mv}']):
                        c2, apx, apy, atx, aty = alt
                        ap, at = (apy, aty) if rv == 'y' else (apx, atx)
                        if abs(at - tv) < 0.1:
                            best = min(best, abs(ap - pv))
                r[f'alt_d_{rv}'] = best
        ev.append(r)
    E = pd.DataFrame(ev).set_index('event_id')
    # ---- null: two unrelated cosmic muons (M3 rays of two different clean singles), det3 separation as the data
    rng = np.random.default_rng(3)
    Rs = R.loc[s.index]
    i = rng.integers(0, len(Rs), 200_000); j = rng.integers(0, len(Rs), 200_000); k = i != j
    A, B = Rs.iloc[i[k]], Rs.iloc[j[k]]
    sx = (B.mesh_x.to_numpy() + 15 * B.tan_x.to_numpy()) - (A.mesh_x.to_numpy() + 15 * A.tan_x.to_numpy())
    sy = (B.mesh_y.to_numpy() + 15 * B.tan_y.to_numpy()) - (A.mesh_y.to_numpy() + 15 * A.tan_y.to_numpy())
    p1, d1 = M.line(A.mesh_x, A.mesh_y, A.tan_x, A.tan_y); p2, d2 = M.line(B.mesh_x, B.mesh_y, B.tan_x, B.tan_y)
    dd, zz, oo = M.dca(p1, d1, p2, d2)
    Mx = pd.DataFrame(dict(sepx=sx, sepy=sy, dca=dd, zv=zz, open=oo))
    Lg.to_parquet(os.path.join(M.OUT, 'legs.parquet')); E.to_parquet(os.path.join(M.OUT, 'events.parquet'))
    Mx.to_parquet(os.path.join(M.OUT, 'mixed.parquet'))
    json.dump(ref, open(os.path.join(M.OUT, 'singles_ref.json'), 'w'), indent=1)
    print(ref); print(E['full'].value_counts()); print(E.topo.value_counts())


if __name__ == '__main__':
    main()
