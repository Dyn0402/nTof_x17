#!/usr/bin/env python3
"""pogo_check.py out/<run>_<plane>.npz [...] --out-dir DIR

Turns the pair statistics of accumulate.py into links, nets and a per-card verdict.  METHOD.md has the argument;
in short:

  1. jump excess  e(a->b) = P(b fires in a separate cluster | a fires) - median of the same rate from the other
                  card's channels (b's own accidental / pickup level).  e_sym = min of the two directions.
                  Separate cluster = quiet strips between, so neither charge sharing nor the track's own extent
                  can make it: the naive spatial correlation of neighbouring strips is removed by construction.
  2. follow       rho = correlation of ln A_a and ln A_b over the jump events.  A pad short puts one charge on
                  both, so the amplitudes follow each other over the whole dynamic range.
  3. link         J >= N_MIN, strip distance >= D_MIN, e_sym >= E_MIN, rho >= RHO_MIN, and a near-identical
                  ratio in every jump event: sd ln(A_a/A_b) < SD_TWIN and |mean| < DM_TWIN ("twin").  Without the
                  ratio cut, two pieces of one inclined track split by under-threshold strips also pass (det3 X:
                  42 such pairs at 6-29 strips, sd 0.3-0.7); a shared node reads the same charge, sd ~0.1.
                  Pairs that pass everything but the ratio are kept as 'coupled' (reported, not used for nets):
                  this includes real net members on noisy channels, whose pickup spreads the ratio.
  3b. excess     the census amplitude correlation C (cosine over events) against the SAME plane's correlation at the
     correlation  SAME strip distance: Z = (C - median_d) / (1.4826 MAD_d), over all judged pairs at distance d.  This
                  is the distance-matched null for the naive worry: strips close on the detector share charge, so C
                  falls smoothly with distance (fleet median 0.6-0.8 at d=1, 0.14-0.23 at d=5, 0.03 at d=10).  A
                  pad short is far out in the tail at its distance (det7 X: 1.00 at d=5).  Link if d >= D_MIN,
                  C >= C_MIN and Z >= Z_MIN.  This catches nets whose strips are contiguous (the dense 2-D block on
                  det7 X), where no quiet gap ever forms and the jump test is blind.
  4. null         pairs on different M1 cards: no pad short can join them, everything else that correlates two
                  strips (pickup, crosstalk, accidentals, a second track, an edge effect) can.  Their link rate is
                  the plane's false-link rate.
  5. layout       nets = connected components of the within-card links.  A net is pogo-shaped when its members are
                  connected through links between pogo-grid neighbours alone, and it has none of the cross-card
                  links of step 4.  Enrichment = link rate on pogo-neighbour pairs / link rate on non-neighbour
                  pairs more than 12 strips apart (beyond two pogo steps).  Card verdict: POGO (a pogo-shaped net of
                  >= 3 channels or two of any size, enrichment >= ENRICH_MIN), POGO? (a single pogo-shaped pair),
                  OTHER (only links the pogo layout does not explain), CLEAN (no links).  '+ OTHER' = both.
  6. adjacent     strip distance <= 2 cannot jump.  Pairs read with a ratio spread sd ln(A_a/A_b) < SD_ADJ over
     twins        >= 30 co-fires are listed as adjacent twins; on the M1 they are either pogo-horizontal or Mec8 /
                  cable neighbours, and the data cannot tell which.
"""
import argparse, json, os, sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layout

N_MIN = 10
D_MIN = 3
E_MIN = 0.10
RHO_MIN = 0.5
SD_TWIN, DM_TWIN = 0.30, 0.20
SD_ADJ, K_ADJ = 0.25, 30
ENRICH_MIN = 10.0
C_MIN, Z_MIN = 0.8, 8.0
F_MIN = 30                 # a channel needs this many firing events to be judged

P = layout.plane_positions()
A = layout.pogo_neighbours()
CARD = P[:, 0]


def pair_tables(z):
    F, J, K = z['F'], z['J'].astype(float), z['K'].astype(float)
    si = z['strip_index']
    d = np.abs(si[:, None] - si[None, :])
    ok = (si[:, None] >= 0) & (si[None, :] >= 0) & (F[:, None] >= F_MIN) & (F[None, :] >= F_MIN)
    np.fill_diagonal(ok, False)
    cross = CARD[:, None] != CARD[None, :]
    r = J / np.maximum(F[:, None], 1)
    base = np.nanmedian(np.where(ok & cross & (d >= D_MIN), r, np.nan), axis=0)
    e = r - np.nan_to_num(base)[None, :]
    es = np.minimum(e, e.T)
    n = np.maximum(J, 1)
    X, Y = z['X'] / n, z['X'].T / n
    va, vb, cv = z['XX'] / n - X ** 2, z['XX'].T / n - Y ** 2, z['XY'] / n - X * Y
    rho = cv / np.sqrt(np.maximum(va * vb, 1e-12))
    sdj = np.sqrt(np.maximum(va + vb - 2 * cv, 0))
    Kn = np.maximum(K, 1)
    sdr = np.sqrt(np.maximum(z['R2'] / Kn - (z['R1'] / Kn) ** 2, 0))
    C = z['S'] / np.sqrt(np.maximum(np.outer(z['Nq'], z['Nq']), 1e-9))
    db = np.where(d <= 40, d, 40 + (d - 40) // 10)            # distance bins: exact to 40, then 10 wide
    Zc = np.zeros_like(C); upp = np.triu(ok, 1)
    for k in np.unique(db[upp]):
        m = upp & (db == k); v = C[m]
        med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-3
        Zc[m] = (v - med) / mad
    Zc = Zc + Zc.T
    return dict(C=C, Zc=Zc, d=d, ok=ok, cross=cross, es=es, rho=rho, sdj=sdj, dm=X - Y, sdr=sdr, J=J, K=K)


def components(nodes, edges):
    if not edges:
        return []
    idx = {c: i for i, c in enumerate(nodes)}
    a = [idx[x] for x, _ in edges]; b = [idx[y] for _, y in edges]
    M = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(nodes), len(nodes)))
    _, lab = connected_components(M, directed=False)
    out = {}
    for c, l in zip(nodes, lab):
        out.setdefault(l, []).append(int(c))
    return sorted((sorted(v) for v in out.values() if len(v) > 1), key=len, reverse=True)


def analyse(z):
    T = pair_tables(z)
    d, ok, cross = T['d'], T['ok'], T['cross']
    tested = ok & (T['J'] >= 0) & (d >= D_MIN)
    coupled = tested & (T['J'] >= N_MIN) & (T['es'] >= E_MIN) & (T['rho'] >= RHO_MIN)
    twin = coupled & (T['sdj'] < SD_TWIN) & (np.abs(T['dm']) < DM_TWIN)
    excess = ok & (d >= D_MIN) & (T['C'] >= C_MIN) & (T['Zc'] >= Z_MIN)
    link = twin | excess
    coupled = coupled & ~link
    adj_twin = ok & (d <= 2) & (d >= 1) & (T['K'] >= K_ADJ) & (T['sdr'] < SD_ADJ)
    up = np.triu(np.ones((512, 512), bool), 1)

    n_cross_tested = int((tested & cross & up).sum()); n_cross_link = int((link & cross & up).sum())
    cross_ch = sorted(set(np.flatnonzero((link & cross).any(1)).tolist()))
    out = dict(run_key=str(z['run_key']), detector=str(z['detector']), plane=str(z['plane']), feu=int(z['feu']),
               nev=int(z['nev']),
               cuts=dict(N_MIN=N_MIN, D_MIN=D_MIN, E_MIN=E_MIN, RHO_MIN=RHO_MIN, SD_TWIN=SD_TWIN, DM_TWIN=DM_TWIN,
                         SD_ADJ=SD_ADJ, K_ADJ=K_ADJ, ENRICH_MIN=ENRICH_MIN, F_MIN=F_MIN,
                         **{k: (v.item() if hasattr(v, 'item') else v) for k, v in z['cuts'].item().items()}),
               null=dict(cross_pairs_tested=n_cross_tested, cross_links=n_cross_link,
                         false_link_rate=n_cross_link / max(n_cross_tested, 1), channels_with_cross_links=cross_ch),
               cards=[])
    noisy = z['rms'] > 3 * np.median(z['rms'])
    for c in (0, 1):
        on = (CARD[:, None] == c) & (CARD[None, :] == c) & up
        lk = link & on; tw = twin & on; cp = coupled & on
        adjm = A > 0
        n_adj_t = int((tested & on & adjm).sum()); n_adj_l = int((lk & adjm).sum())
        far = tested & on & ~adjm & (d > 12)
        n_far_t = int(far.sum()); n_far_l = int((lk & far).sum())
        rate_adj = n_adj_l / max(n_adj_t, 1); rate_far = n_far_l / max(n_far_t, 1)
        enrich = rate_adj / rate_far if rate_far > 0 else (np.inf if n_adj_l else np.nan)
        a, b = np.nonzero(lk)
        nets = []
        for comp in components(sorted(set(a) | set(b)), list(zip(a, b))):
            s = set(comp)
            ae = [(x, y) for x, y in zip(a, b) if x in s and A[x, y] > 0]
            spans = len(components(comp, ae)) == 1 and len(components(comp, ae)[0]) == len(comp)
            touches_cross = sorted(s & set(cross_ch))
            nets.append(dict(channels=comp, n=len(comp), pogo_shaped=bool(spans and not touches_cross),
                             spanned_by_pogo_links=bool(spans), cross_card_members=touches_cross,
                             noisy=int(noisy[comp].sum())))
        aa, bb = np.nonzero(adj_twin & on)
        pogo_ch = sorted(x for n in nets if n['pogo_shaped'] for x in n['channels'])
        other_ch = sorted(x for n in nets if not n['pogo_shaped'] for x in n['channels'])
        pogo_nets = sum(1 for n in nets if n['pogo_shaped'] and n['n'] >= 3) + (sum(1 for n in nets if n['pogo_shaped']) >= 2)
        if not nets:
            verdict = 'CLEAN'
        elif pogo_nets >= 2 and (not np.isfinite(enrich) or enrich >= ENRICH_MIN):
            verdict = 'POGO' + (' + OTHER' if other_ch else '')
        elif pogo_ch:
            verdict = 'POGO? (one pair / weak)' + (' + OTHER' if other_ch else '')
        else:
            verdict = 'OTHER'
        out['cards'].append(dict(
            card=c + 1, connectors=[4 * c + 1, 4 * c + 4], verdict=verdict,
            links=int(lk.sum()), coupled_links=int(cp.sum()),
            pogo_adjacent=dict(tested=n_adj_t, linked=n_adj_l, rate=rate_adj),
            far_nonadjacent=dict(tested=n_far_t, linked=n_far_l, rate=rate_far),
            enrichment=None if not np.isfinite(enrich) else float(enrich),
            pogo_channels=pogo_ch, other_channels=other_ch, nets=nets,
            adjacent_twins=[[int(x), int(y)] for x, y in zip(aa, bb)],
            d_spectrum={int(k): int(v) for k, v in zip(*np.unique(d[lk], return_counts=True))}))
    links = [dict(a=int(x), b=int(y), d=int(d[x, y]), adj=int(A[x, y]), e=round(float(T['es'][x, y]), 3),
                  rho=round(float(T['rho'][x, y]), 3), sd=round(float(T['sdj'][x, y]), 3), J=int(T['J'][x, y]),
                  C=round(float(T['C'][x, y]), 3), Z=round(float(T['Zc'][x, y]), 1),
                  twin=bool(twin[x, y]), excess=bool(excess[x, y]), cross=bool(cross[x, y]))
             for x, y in zip(*np.nonzero((link | coupled) & up))]
    return out, links, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', nargs='+'); ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for f in a.npz:
        z = np.load(f, allow_pickle=True)
        out, links, _ = analyse(z)
        key = f"{out['run_key']}_{out['plane']}"
        json.dump(dict(out, links=links), open(os.path.join(a.out_dir, key + '.json'), 'w'), indent=1)
        nl = out['null']
        print(f"{key:20s} cross-card links {nl['cross_links']}/{nl['cross_pairs_tested']} | " + ' | '.join(
            f"card {c['card']}: {c['verdict']:22s} links {c['links']:4d} pogo-net ch {len(c['pogo_channels']):3d} "
            f"other ch {len(c['other_channels']):3d} enrich {c['enrichment'] if c['enrichment'] is None else round(c['enrichment'], 1)} "
            f"adj-twins {len(c['adjacent_twins'])}" for c in out['cards']))


if __name__ == '__main__':
    main()
