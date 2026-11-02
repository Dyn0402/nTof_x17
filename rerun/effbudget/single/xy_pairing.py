#!/usr/bin/env python3
"""xy_pairing.py [--root R] [--out DIR] -- Paper I: pairing X and Y segments of two particles, on real cosmics.

With two particles in a chamber, each view sees two segments and two X/Y pairings are possible; the wrong one makes
two ghost tracks. The views share nothing geometric (x and y are independent coordinates), but they read the same
avalanche charge (pads: X/Y charge share 0.49/0.51) at the same drift times, so the pass pairs them by the charge
balance and the charge-arrival times (wft.reco.xy_pair_cost, calibrated per row on clean single tracks).

Truth on the bench: events with >= 2 candidates in BOTH views where the M3 ray picks out exactly one X candidate
(|p0 - M3| < 5 mm, the next > 10 mm) and exactly one Y candidate the same way. That candidate pair is one particle;
the brightest remaining candidate of each view is the other. The question is a 2x2 one: keep or swap.

Pairing rules compared:
  production   the pass's own track_id (select_tracks + the pairing calibration of the row);
  charge       |lq - med| summed over the two pairs, keep vs swap (lq = log q_x/q_y, med/width from single tracks);
  charge+time  the same plus the u50 (median charge-arrival time) difference, each in units of its single-track width;
  brightness   strongest X with strongest Y (the naive rule select_tracks used before 2026-09-14);
  random       50 %.
Writes <out>/xy_pairing.json, <out>/xy_pairing_events.parquet.
"""
import argparse, json, os
import numpy as np
import pandas as pd

ROWS = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}
R_IN, R_OUT = 5.0, 10.0


def rsig(x):
    q = np.nanpercentile(x, [16, 84]); return 0.5 * (q[1] - q[0])


def load(root, rel):
    C = pd.read_parquet(os.path.join(root, 'Analysis_pass1', rel, 'wft/events.candidates.parquet'))
    C = C[C['rank'] >= 0]
    B = pd.read_parquet(os.path.join(root, 'x17rerun/paper1_eff/budget', rel, 'rays_det3.parquet'))
    B = B.drop_duplicates('event_id').set_index('event_id')
    B = B[(~B.not_measured) & (B.region == 'interior') & B.good_x]
    C = C[C.event_id.isin(B.index)].copy()
    C['mesh'] = np.where(C.plane == 'x', B.mesh_x.reindex(C.event_id).to_numpy(), B.mesh_y.reindex(C.event_id).to_numpy())
    C['d'] = (C.p0 - C.mesh).abs()
    return C, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('COSMIC_ROOT', '/eos/experiment/ntof/data/x17/cosmic_bench'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res, evs = {}, []
    for key, rel in ROWS.items():
        C, B = load(a.root, rel)
        n = C.groupby(['event_id', 'plane']).size().unstack(fill_value=0)
        # single-track reference: exactly one candidate per view, both near M3
        one = n.index[(n.x == 1) & (n.y == 1)]
        S = C[C.event_id.isin(one)].pivot(index='event_id', columns='plane')
        S = S[(S['d']['x'] < R_IN) & (S['d']['y'] < R_IN)]
        lq1 = np.log(S['q_sum']['x'].clip(lower=1) / S['q_sum']['y'].clip(lower=1))
        du1 = S['q_u50']['x'] - S['q_u50']['y']
        ref = dict(n=int(len(S)), lq_med=float(lq1.median()), lq_rsig=float(rsig(lq1)),
                   du_med=float(du1.median()), du_rsig=float(rsig(du1)))
        rows = []
        for e, g in C[C.event_id.isin(n.index[(n.x >= 2) & (n.y >= 2)])].groupby('event_id'):
            gx = g[g.plane == 'x'].sort_values('d'); gy = g[g.plane == 'y'].sort_values('d')
            if not (gx.d.iloc[0] < R_IN and gx.d.iloc[1] > R_OUT and gy.d.iloc[0] < R_IN and gy.d.iloc[1] > R_OUT):
                continue
            xm, ym = gx.iloc[0], gy.iloc[0]
            xo = gx.iloc[1:].sort_values('q_sum').iloc[-1]; yo = gy.iloc[1:].sort_values('q_sum').iloc[-1]
            lq = lambda u, v: np.log(max(u.q_sum, 1) / max(v.q_sum, 1))
            zq = lambda u, v: abs(lq(u, v) - ref['lq_med']) / ref['lq_rsig']
            zt = lambda u, v: abs((u.q_u50 - v.q_u50) - ref['du_med']) / ref['du_rsig']
            keep_q, swap_q = zq(xm, ym) ** 2 + zq(xo, yo) ** 2, zq(xm, yo) ** 2 + zq(xo, ym) ** 2
            keep_t = keep_q + zt(xm, ym) ** 2 + zt(xo, yo) ** 2
            swap_t = swap_q + zt(xm, yo) ** 2 + zt(xo, ym) ** 2
            tid = (int(xm.track_id), int(ym.track_id)); to = (int(xo.track_id), int(yo.track_id))
            if tid[0] >= 0 and tid[0] == tid[1]:
                prod = 'correct'
            elif (tid[0] >= 0 and tid[0] == to[1]) or (tid[1] >= 0 and tid[1] == to[0]):
                prod = 'wrong'                       # an M3 segment joined to the other particle's segment
            else:
                prod = 'unpaired'                    # the M3 particle in no selected track (plausibility gate)
            n_impl = int((~g.plausible.astype(bool)).sum())
            # charge asymmetry of the two particles (how hard the problem is)
            qa = abs(np.log((xm.q_sum + ym.q_sum) / max(xo.q_sum + yo.q_sum, 1)))
            rows.append(dict(row=key, event_id=int(e), prod=prod, charge=keep_q < swap_q, charge_time=keep_t < swap_t,
                             bright=(xm.q_sum > xo.q_sum) == (ym.q_sum > yo.q_sum), lq_true_m=lq(xm, ym),
                             lq_true_o=lq(xo, yo), lq_swap_1=lq(xm, yo), lq_swap_2=lq(xo, ym), q_asym=qa,
                             sep_x=abs(xm.p0 - xo.p0), sep_y=abs(ym.p0 - yo.p0),
                             dt0=abs(xm.t0 - xo.t0), nx=int(len(gx)), ny=int(len(gy)), n_implausible=n_impl,
                             xm_rank=int(xm['rank']), ym_rank=int(ym['rank']), xo_rank=int(xo['rank']),
                             yo_rank=int(yo['rank'])))
        E = pd.DataFrame(rows); evs.append(E)
        res[key] = dict(reference=ref, n=int(len(E)))
    E = pd.concat(evs, ignore_index=True)
    out = {}
    for nm, m in [('all', np.ones(len(E), bool)), ('two particles of similar charge (|ln q1/q2| < 0.5)', E.q_asym < 0.5),
                  ('different charge (|ln q1/q2| >= 0.5)', E.q_asym >= 0.5)]:
        U = E[m]; k = len(U)
        f = lambda s: dict(frac=float(s.mean()), err=float(np.sqrt(s.mean() * (1 - s.mean()) / k)) if k else None)
        out[nm] = dict(n=int(k), production=U['prod'].value_counts(normalize=True).to_dict(),
                       production_correct=f(U['prod'].eq('correct')),
                       production_correct_of_paired=f(U[U['prod'] != 'unpaired']['prod'].eq('correct')),
                       production_correct_all_plausible=f(U[U.n_implausible == 0]['prod'].eq('correct')),
                       n_all_plausible=int((U.n_implausible == 0).sum()),
                       charge=f(U.charge), charge_time=f(U.charge_time), brightness=f(U.bright))
    res['pooled'] = out
    E.to_parquet(os.path.join(a.out, 'xy_pairing_events.parquet'))
    json.dump(res, open(os.path.join(a.out, 'xy_pairing.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
