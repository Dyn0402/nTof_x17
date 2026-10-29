#!/usr/bin/env python3
"""satgate_pairs.py --root <gate tree> [--a sat_base] [--b sat_cand ...] -- the full-reconstruction pieces of the
error-model gate that gate_compare.py does not show (ERROR_MODEL.md, 10-29).

Per chamber and arm, from <root>/<row>/wft/<arm>/events{,.candidates}.parquet (the gate RECO tables):
  events, both views ok, quality_ok per view (the relative chi2/dof flag), n_tracks >= 1 ("paired": an x-y track was
  formed) and >= 2, events with an accepted split, rescued tracks (head-on rescue candidates that ended in a track);
  x-y charge agreement: robust sigma of log(q_x / q_y) on single-track events with both views quality_ok, and the
  pairing power P(|log qx_i/qy_i| < |log qx_i/qy_j|) (j a random other such event; 0.5 = charge useless);
  the charge level: median q_sum of arm B / arm A per view on the same events.
Per event, A vs B on events fitted in both: same w (1 um/ns), |dtheta| > 1 deg, quality flag agreement, same
n_tracks, and for events with >= 2 tracks in both arms the fraction whose track list is the same (every x-y pair
matched within 0.5 mm in both p0).
Writes results/satgate_pairs.json and prints markdown tables.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'model_audit'))
from gate_compare import ROWS  # noqa: E402


def load(root, row, arm):
    d = os.path.join(root, row, 'wft', arm)
    p = os.path.join(d, 'events.parquet')
    if not os.path.exists(p):
        return None, None
    c = os.path.join(d, 'events.candidates.parquet')
    return pd.read_parquet(p), (pd.read_parquet(c) if os.path.exists(c) else None)


def tracks(C):
    """event_id -> sorted list of (x p0, y p0) of the gated tracks."""
    if C is None:
        return {}
    g = C[(C['track_id'] >= 0) & C['track_gated'] & (C['rank'] >= 0)]
    out = {}
    for (e, t), s in g.groupby(['event_id', 'track_id']):
        px = s.loc[s['plane'] == 'x', 'p0']; py = s.loc[s['plane'] == 'y', 'p0']
        if len(px) and len(py):
            out.setdefault(int(e), []).append((float(px.iloc[0]), float(py.iloc[0])))
    return {e: sorted(v) for e, v in out.items()}


def same_tracks(a, b, tol=0.5):
    if len(a) != len(b):
        return False
    used = set()
    for xa, ya in a:
        k = next((j for j, (xb, yb) in enumerate(b) if j not in used and abs(xa - xb) < tol and abs(ya - yb) < tol), None)
        if k is None:
            return False
        used.add(k)
    return True


def charge(E, rng):
    m = (E['x_ok'] & E['y_ok'] & E['x_quality_ok'] & E['y_quality_ok'] & (E['n_tracks'] == 1)
         & (E['x_q_sum'] > 0) & (E['y_q_sum'] > 0))
    qx, qy = E.loc[m, 'x_q_sum'].to_numpy(float), E.loc[m, 'y_q_sum'].to_numpy(float)
    lq = np.log(qx / qy)
    pp = []
    for _ in range(20):
        j = rng.permutation(len(qx)); j = np.where(j == np.arange(len(qx)), (j + 1) % len(qx), j)
        pp.append(np.mean(np.abs(lq) < np.abs(np.log(qx / qy[j]))))
    return int(m.sum()), float(1.4826 * np.median(np.abs(lq - np.median(lq)))), float(np.mean(pp))


def arm_stats(E, C, rng):
    n = len(E)
    r = dict(n=n, both_ok=float((E['x_ok'] & E['y_ok']).mean()),
             qok_x=float(E['x_quality_ok'].mean()), qok_y=float(E['y_quality_ok'].mean()),
             paired=float((E['n_tracks'] >= 1).mean()), multi=int((E['n_tracks'] >= 2).sum()),
             splits=int((E.get('n_splits', pd.Series(0, index=E.index)) > 0).sum()))
    r['rescued'] = (int(C[(C['rescued']) & (C['track_id'] >= 0) & C['track_gated']]['event_id'].nunique())
                    if C is not None else None)
    r['n_q'], r['lq_rsig'], r['pairing'] = charge(E, rng)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--a', default='sat_base')
    ap.add_argument('--b', nargs='*', default=['sat_cand', 'sat_cap']); ap.add_argument('--tag', default='satgate_pairs')
    a = ap.parse_args()
    rng = np.random.default_rng(1); out = {}
    T1 = ['| chamber arm | events | both ok | quality_ok X / Y | paired (n_tracks ≥ 1) | n_tracks ≥ 2 | splits | rescued | '
          'x-y log-charge σ (n) | pairing power |', '|' + '---|' * 10]
    T2 = ['| chamber arm vs base | view | same w | dθ > 1° | quality flag agrees | q_sum B/A p50 | χ² B/A p50 | '
          'same n_tracks | same track list (≥ 2 tracks) |', '|' + '---|' * 9]
    for det, row in ROWS.items():
        A, CA = load(a.root, row, a.a)
        if A is None:
            continue
        sa = arm_stats(A, CA, rng); out[f'{det}|{a.a}'] = sa
        tA = tracks(CA)
        for arm, s, in [(a.a, sa)] + [(b, None) for b in a.b]:
            if s is None:
                B, CB = load(a.root, row, arm)
                if B is None:
                    continue
                s = arm_stats(B, CB, rng); out[f'{det}|{arm}'] = s
            T1.append(f"| {det} {arm} | {s['n']} | {s['both_ok']:.1%} | {s['qok_x']:.1%} / {s['qok_y']:.1%} | "
                      f"{s['paired']:.1%} | {s['multi']} | {s['splits']} | {s['rescued']} | {s['lq_rsig']:.3f} ({s['n_q']}) | "
                      f"{s['pairing']:.3f} |")
            if arm == a.a:
                continue
            m = A.merge(B, on='event_id', suffixes=('_a', '_b'))
            tB = tracks(CB)
            nt = float(np.mean(m['n_tracks_a'] == m['n_tracks_b']))
            ev2 = [e for e in m['event_id'] if len(tA.get(e, [])) >= 2 and len(tB.get(e, [])) >= 2]
            st = float(np.mean([same_tracks(tA[e], tB[e]) for e in ev2])) if ev2 else np.nan
            for v in 'xy':
                k = m[f'{v}_ok_a'].astype(bool) & m[f'{v}_ok_b'].astype(bool); s2 = m[k]
                g = lambda c, x: s2[f'{v}_{c}_{x}'].to_numpy(float)
                pr = dict(n=int(k.sum()), same_w=float(np.mean(np.abs(g('w', 'b') - g('w', 'a')) * 1e3 < 1.0)),
                          dth_gt1=float(np.nanmean(np.abs(g('theta_deg', 'b') - g('theta_deg', 'a')) > 1.0)),
                          qflag=float(np.mean(s2[f'{v}_quality_ok_a'].to_numpy() == s2[f'{v}_quality_ok_b'].to_numpy())),
                          q_ratio=float(np.nanmedian(g('q_sum', 'b') / g('q_sum', 'a'))),
                          chi2_ratio=float(np.nanmedian(g('chi2', 'b') / g('chi2', 'a'))),
                          same_ntracks=nt, same_tracklist=st, n_multi_both=len(ev2))
                out[f'{det}|{arm}|{v}'] = pr
                T2.append(f"| {det} {arm} | {v.upper()} | {pr['same_w']:.1%} | {pr['dth_gt1']:.1%} | {pr['qflag']:.1%} | "
                          f"{pr['q_ratio']:.4f} | {pr['chi2_ratio']:.3f} | {nt:.1%} | {st:.1%} ({len(ev2)}) |")
    md = '\n'.join(T1) + '\n\n' + '\n'.join(T2) + '\n'
    print(md)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', f'{a.tag}.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'results', f'{a.tag}.md'), 'w').write(md)


if __name__ == '__main__':
    main()
