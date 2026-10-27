#!/usr/bin/env python3
"""analyse.py --q DIR --stack zsstack.json --lat lateral.json --out results.json

F16 table from the baseline-FIXED caches (sps_build_cache >= 7505338).  Per cache and arm (z_ = ZS zeros fitted as
measured zeros, c_ = censored, the F16 setting):
  w      median fitted lateral speed in the inclined view (Y; the mount tilts Y), |w|/tan(mount)
  T_q    column end from the STACKED NNLS depth profile (the 50 % point of its trailing edge, plateau = median of the
         profile between 300 and 900 ns), and the old per-event q_uend median for reference
  T_box  column duration from the stacked strip sum (zsstack.json, model-free: box (x) electronics template)
  L      = |w| * T: the lateral extent of the column; gap_wT = L / tan(mount) needs NO drift velocity
Writes JSON and prints a markdown table.
"""
import argparse, glob, json, os, re
import numpy as np

TAN = {'rot25': np.tan(np.radians(25.64)), 'op25': np.tan(np.radians(25.64)), 'rot15': np.tan(np.radians(15.465))}
FIELD = {'run63_rot25_operating_00_d675': 234.5, 'run63_rot25_operating_00_d625': 217.1, 'run63_rot25_operating_00_d575': 199.8,
         'run63_rot25_operating_00_d525': 182.4, 'run63_rot25_operating_00_d475': 165.0, 'run63_rot25_operating_00_d425': 147.6,
         'run62_rot25_ladder_driftscan_gap150V_d700': 243.2, 'run62_rot25_ladder_driftscan_gap150V_d600': 208.5,
         'run62_rot25_ladder_driftscan_gap150V_d500': 173.8, 'run63_flat_operating_03_flat700': 243.2,
         'run61_op25_meshscan_m70V_op25': 243.1, 'run61_rot15_ladder_meshscan_m20V_d700': 243.2}


def tan_of(key):
    for k, v in TAN.items():
        if k in key:
            return float(v)
    return 0.0


def trailing_edge(st, u):
    """50 % crossing of the stacked depth profile's trailing edge (plateau = median over 300-900 ns)."""
    pl = np.median(st[(u > 300) & (u < 900)])
    j = np.where((u > 900) & (st < 0.5 * pl))[0]
    if not len(j):
        return np.nan
    k = j[0]
    return float(u[k - 1] + (st[k - 1] - 0.5 * pl) / (st[k - 1] - st[k]) * (u[k] - u[k - 1]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--q', required=True); ap.add_argument('--stack', required=True)
    ap.add_argument('--lat', default=None); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ST = json.load(open(a.stack)); LAT = json.load(open(a.lat)) if a.lat else {}
    res = {}
    for f in sorted(glob.glob(os.path.join(a.q, '*.npz'))):
        arm, key = re.match(r'([zc])_(.*)\.npz', os.path.basename(f)).groups()
        Z = np.load(f); P, Q = Z['per'], Z['q']; K = Q.shape[1]; u = (np.arange(K) + 0.5) * 60.0
        tn = tan_of(key); r = dict(arm=arm, key=key, field=FIELD.get(key), tan=tn)
        for vi, view in enumerate('xy'):
            m = P[:, 1] == vi
            if m.sum() < 30:
                continue
            q = Q[m] / np.maximum(Q[m].sum(1, keepdims=True), 1e-9); st = q.mean(0)
            w = P[m, 3]; wm = float(np.median(w)); we = float(1.2533 * 1.4826 * np.median(np.abs(w - wm)) / np.sqrt(m.sum()))
            Tq = trailing_edge(st, u)
            r[view] = dict(n=int(m.sum()), w=wm, w_err=we, chi2dof=float(np.median(P[m, 4] / np.maximum(P[m, 5], 1))),
                           uend_p50=float(np.nanmedian(P[m, 8])), T_q=Tq, profile=st.tolist(), K=K,
                           T_box=ST.get(key, {}).get(view, {}).get('T_box'), T_flat=ST.get(key, {}).get(view, {}).get('T_flat'))
            if tn > 0 and view == 'y':
                for tag in ('T_q', 'T_box'):
                    T = r[view][tag]
                    if T:
                        r[view][f'gap_wT_{tag}'] = abs(wm) * T / 1e3 / tn
                        r[view][f'v_slope'] = abs(wm) / tn
        if key in LAT:
            r['lat'] = {k: LAT[key][k] for k in ('L', 'L_err', 'gap', 'gap_err', 'sigma')}
        res[f'{arm}_{key}'] = r
    json.dump(res, open(a.out, 'w'), indent=1)
    print('| arm | cache | E [V/cm] | |w_y|/tan [µm/ns] | T_box [ns] | T_q [ns] | uend p50 | 28.8/T_box | 30/T_box | gap = |w|·T_box/tan [mm] | gap = |w|·T_q/tan | χ²/dof y |')
    print('|' + '---|' * 12)
    for k, r in sorted(res.items(), key=lambda kv: (kv[1]['arm'], -(kv[1]['field'] or 0))):
        y = r.get('y', {})
        if not y:
            continue
        Tb = y.get('T_box') or np.nan; vs = abs(y['w']) / r['tan'] if r['tan'] else np.nan
        print(f"| {r['arm']} | {r['key'].replace('_operating_00', '').replace('_ladder_driftscan_gap150V', '').replace('_meshscan', '')} | {r['field']} | "
              f"{vs:.2f} | {Tb:.0f} | {y['T_q']:.0f} | {y['uend_p50']:.0f} | {28800 / Tb:.2f} | {30000 / Tb:.2f} | "
              f"{y.get('gap_wT_T_box', np.nan):.1f} | {y.get('gap_wT_T_q', np.nan):.1f} | {y['chi2dof']:.0f} |")


if __name__ == '__main__':
    main()
