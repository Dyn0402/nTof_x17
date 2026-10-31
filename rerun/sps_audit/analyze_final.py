#!/usr/bin/env python3
"""analyze_final.py --tel DIR --fp DIR --hits DIR [--out results/final.json]

The final SPS det4 numbers on the final telescope tables (tel_job.py F* tags: final seed rule + the pass's hot probe, NO
pass cut, every head-on estimator per event) and the footprint tables (footprint.py):

  resolution  per dataset x estimator (p0, p_mid, centroid, hits where combined_hits carry det4) x view, against the
              uRWELL telescope, by the run-53 method (analyze.py: seeded 2-D affine, fixed +-1.2 mm Gaussian core,
              back-plane zones, pointing with the run-53 f = 0.301; the 3-zone free fit where the zones allow)
  cut flow    the selection, each cut with its count and the number that justifies it; and the cuts NOT applied,
              each with the measurement that shows it is not needed
  timing      det4 t0: X-Y difference, and against the trigger phase (ftst)
  footprint   cluster strips, lead fraction, neighbour ratio, charge, vs resist voltage and drift field (flat rows)
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
from analyze import (seeded_fit, decompose, gauss_sigma, Z_BACK, Z_DET4, ZGRID, FID_MM, det4_clusters,  # noqa: E402
                     POSITION_MM, VIEW)

DATASETS = {   # name -> (tag prefixes, registry keys or None, hits time gate for the hits estimator)
    'run53': (['G53'], None, (600, 1850)),
    'run55 d700': (['G5500'], ['run55_flatdrift__d700'], (600, 2600)),
    'run55 d600': (['G5500'], ['run55_flatdrift__d600'], (600, 2600)),
    'run55 d500': (['G5510'], ['run55_flatdrift__d500'], (600, 3200)),
    'run55 d400': (['G5520'], ['run55_flatdrift__d400'], (600, 3800)),
    'run63 flat700': (['G6302', 'G6303'], ['run63_flat__flat700'], None),
    'run66 r755+r780': (['G66'], ['run66_flat_resist__r755', 'run66_flat_resist__r780'], None),
    'run66 r580-r730': (['G66'], [f'run66_flat_resist__r{r}' for r in (580, 605, 630, 655, 680, 705, 730)], None),
    'run71 raw700': (['G71'], ['run71_raw__raw700'], None),
    'run71 raw450+275': (['G71'], ['run71_raw__raw450', 'run71_raw__raw275'], None),
}
EST = ('p0', 'p_mid', 'c_pos', 'hits')
CORE = 1.2


def tsig(r):
    """(sigma, err, n) of a time distribution [ns]: 1.4826 x MAD (robust to the pile-up and mis-timed tail); the
    run-53 fixed-window Gaussian (gauss_sigma) assumes a mm scale and does not apply."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 100:
        return np.nan, np.nan, len(r)
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    return float(s), float(s / np.sqrt(2 * len(r))), len(r)


def load(tel, prefixes, keys):
    fs = sorted(f for p in prefixes for f in glob.glob(os.path.join(tel, p + '_[0-9][0-9][0-9].parquet')))
    if not fs:
        return None, {}
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    if keys:
        d = d[d.key.isin(keys)]
    hits = {}
    for f in fs:
        h = f.replace('.parquet', '_hits.npz')
        if os.path.exists(h):
            hits[(os.path.basename(f).split('_')[0], f[-11:-8])] = dict(np.load(h))
    return d.reset_index(drop=True), hits


def hits_xy(d, hits, gate):
    """run-53 estimator from the combined_hits det4 hits: lead cluster per view, one cluster per view, gated in time."""
    out = {v: np.full(len(d), np.nan) for v in 'xy'}
    for (tag, g), h in hits.items():
        rows = np.flatnonzero((d.group == g).values)
        if not len(rows) or not len(h['ev']):
            continue
        idx = {int(e): i for i, e in enumerate(d.eid.values[rows])}
        m = np.isin(h['ev'], list(idx)) & (h['time'] > gate[0]) & (h['time'] < gate[1])
        ev = np.array([idx[int(e)] for e in h['ev'][m]], int); ch = h['ch'][m].astype(int); amp = h['amp'][m].astype(float)
        isx = VIEW[ch] == 'x'
        for v, k in (('x', isx), ('y', ~isx)):
            if k.any():
                lead, nc, _ = det4_clusters(ev[k], POSITION_MM[ch[k]], amp[k], len(rows))
                lead[nc != 1] = np.nan
                out[v][rows] = lead
    return out


def chi2ok(d, v):
    r = d[f'd4_{v}_chi2'] / d[f'd4_{v}_dof'].clip(lower=1)
    good = d[f'd4_{v}_ok'].astype(float).fillna(0).astype(bool) & np.isfinite(r)
    return (good & (r <= 5.0 * np.nanmedian(r[good]))).values if good.any() else good.values


def estimators(d, hits, gate):
    E = {}
    for e in ('p0', 'p_mid', 'c_pos'):
        E[e] = {v: d[f'd4_{v}_{e}'].values.astype(float) for v in 'xy'}
    if hits and gate:
        h = hits_xy(d, hits, gate)
        if np.mean(np.isfinite(h['x']) & np.isfinite(h['y'])) > 0.2:
            E['hits'] = h
    return E


def residuals(D, ex, ey, z=Z_DET4):
    tx = (D['fx_p'] + (D['bx_p'] - D['fx_p']) * z / Z_BACK).values
    ty = (D['fy_p'] + (D['by_p'] - D['fy_p']) * z / Z_BACK).values
    M = np.column_stack([ex, ey, np.ones(len(ex))])
    cx, rx = seeded_fit(M, tx, ey); cy, ry = seeded_fit(M, ty, ex)
    return rx, ry, np.array([[cx[0], cx[1]], [cy[0], cy[1]]])


def one(d, hits, gate):
    tel = np.all([(d[f'{p}_n'] == 1).values for p in ('fx', 'fy', 'bx', 'by')], axis=0)
    E = estimators(d, hits, gate)
    valid = {e: np.isfinite(E[e]['x']) & np.isfinite(E[e]['y']) for e in E}
    common = tel & np.all(list(valid.values()), axis=0)
    D = d[common].reset_index(drop=True)
    res = dict(n_rows=int(len(d)), n_tel=int(tel.sum()), n_common=int(common.sum()), est={}, tests={})
    R = {}
    for e in E:
        ex, ey = E[e]['x'][common], E[e]['y'][common]
        if len(D) < 60:
            continue
        rx, ry, A = residuals(D, ex, ey)
        fY, fX = np.abs(ry) < FID_MM, np.abs(rx) < FID_MM
        zs = []
        for z in ZGRID:
            a, b, _ = residuals(D, ex, ey, z)
            zs.append((float(z), float(gauss_sigma(b[np.abs(a) < FID_MM])[0]), float(gauss_sigma(a[np.abs(b) < FID_MM])[0])))
        res['est'][e] = dict(det_A=float(np.linalg.det(A)), roll=float(np.degrees(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))),
                             fid_frac=float(np.mean(fY & fX)), zscan=zs,
                             X=dict(decompose(ry[fX], D.by_p.values[fX], 'y', Z_DET4), n_fid=int(fX.sum())),
                             Y=dict(decompose(rx[fY], D.bx_p.values[fY], 'x', Z_DET4), n_fid=int(fY.sum())))
        R[e] = (rx, ry, fX, fY)
    # cuts NOT applied: does each change the core width of the main estimator (p0) on the fiducial sample?
    if 'p0' in R:
        rx, ry, fX, fY = R['p0']
        tests = {'chi2/dof <= 5 x median (pass quality)': (chi2ok(D, 'x') & chi2ok(D, 'y')),
                 'pass seed cut (peak >= 300 ADC, >= 4 strips, both views)':
                     ((D.seed_amp_x >= 300) & (D.seed_amp_y >= 300) & (D.win_strips_x >= 4) & (D.win_strips_y >= 4)).values,
                 'centroid cluster <= 6 strips both views': ((D.d4_x_c_n_strips <= 6) & (D.d4_y_c_n_strips <= 6)).values}
        for name, m in tests.items():
            row = {}
            for v, r, f in (('X', ry, fX), ('Y', rx, fY)):
                a, ea, na = gauss_sigma(r[f & m]); b, eb, nb = gauss_sigma(r[f & ~m]); c, ec, nc = gauss_sigma(r[f])
                row[v] = dict(pass_=[float(a), float(ea), int((f & m).sum())], fail=[float(b), float(eb), int((f & ~m).sum())],
                              all=[float(c), float(ec), int(f.sum())])
            res['tests'][name] = dict(frac_pass=float(np.mean(m)), **row)
    # timing on the same events (p0 fits): t0 X - Y, and t0 against the trigger phase
    if 'd4_x_t0' in D:
        ok = chi2ok(D, 'x') & chi2ok(D, 'y')
        dt = (D.d4_x_t0 - D.d4_y_t0).values[ok]
        s, e, n = tsig(dt)
        tim = dict(n=int(ok.sum()), sigma_dxy_ns=float(s), err=float(e), s68_dxy_ns=float(np.nanpercentile(np.abs(dt - np.nanmedian(dt)), 68.27)))
        if 'd4_ftst' in D:
            for v in 'xy':
                t = D[f'd4_{v}_t0'].values[ok]; f = D.d4_ftst.values[ok]
                r = np.concatenate([t[f == c] - np.nanmedian(t[f == c]) for c in np.unique(f) if (f == c).sum() > 20]) if ok.any() else np.array([])
                s2, e2, _ = tsig(r)
                tim[f'sigma_t0_{v}_in_ftst_class_ns'] = float(s2)
                tim['ftst_classes'] = sorted(int(c) for c in np.unique(f))
                tim[f'median_t0_{v}_per_ftst'] = {int(c): float(np.nanmedian(t[f == c])) for c in np.unique(f) if (f == c).sum() > 20}
        res['timing'] = tim
    return res, R, D


def cut_flow(logs_dir, tag_re):
    """Telescope tracks with one cluster per plane-view and those with a det4 record (seed >= 50 ADC both views), from the
    tel_job logs ('det4 records: N of M telescope tracks')."""
    n_rec = n_tel = 0
    for f in glob.glob(os.path.join(logs_dir, '*.out')):
        if not re.match(tag_re, os.path.basename(f)):
            continue
        for ln in open(f):
            m = re.search(r'det4 records: (\d+) of (\d+)', ln)
            if m:
                n_rec += int(m.group(1)); n_tel += int(m.group(2))
    return n_tel, n_rec


def footprint(fp_dir):
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}
    out = []
    for f in sorted(glob.glob(os.path.join(fp_dir, '*.parquet'))):
        t = pd.read_parquet(f); k = os.path.basename(f)[:-8]; r = reg[k]
        row = dict(key=k, run=r['run'], gas=r['gas'], resist_V=float(r['resist_V'] or 'nan'), drift_V=float(r['drift_V'] or 'nan'),
                   field=float(r['field_Vcm'] or 'nan'), raw=int(r['raw']), n=len(t))
        for v in 'xy':
            ok = np.isfinite(t[f'{v}_c_pos'])
            row[f'{v}_strips_mean'] = float(t[f'{v}_c_n_strips'][ok].mean())
            row[f'{v}_strips_median'] = float(t[f'{v}_c_n_strips'][ok].median())
            row[f'{v}_lead_frac'] = float(t[f'{v}_c_lead_frac'][ok].median())
            row[f'{v}_nb'] = float(np.nanmedian(np.r_[t[f'{v}_c_nb_lo'][ok], t[f'{v}_c_nb_hi'][ok]]))
            row[f'{v}_q_sum'] = float(t[f'{v}_c_q_sum'][ok].median())
            row[f'{v}_q_lead'] = float(t[f'{v}_c_q_lead'][ok].median())
        # time: the cluster's leading edge (footprint.leading_edge), X against Y, and against the trigger phase ftst
        if 'x_t_le_ns' in t:
            for tag in ('t_le', 't_le20', 't_abs'):
                dxy = (t[f'x_{tag}_ns'] - t[f'y_{tag}_ns']).dropna().values
                s, e, _ = tsig(dxy)
                row[f'{tag}_sigma_xy_ns'], row[f'{tag}_sigma_xy_err'] = float(s), float(e)
                for v in 'xy':
                    tt = t[f'{v}_{tag}_ns'].values; f = t.ftst.values; ok = np.isfinite(tt)
                    best = (np.inf, np.nan, None)
                    for sgn in (-1, 0, 1):              # the trigger phase in 10 ns steps of the 60 ns clock
                        r = tt[ok] + sgn * 10.0 * f[ok]
                        s2, e2, _ = tsig(r)
                        if np.isfinite(s2) and s2 < best[0]:
                            best = (s2, e2, sgn)
                    row[f'{v}_{tag}_sigma_ns'], row[f'{v}_{tag}_sigma_err'], row[f'{v}_{tag}_ftst_sign'] = float(best[0]), float(best[1]), best[2]
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tel', required=True); ap.add_argument('--fp', required=True); ap.add_argument('--logs', default=None)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'final.json'))
    a = ap.parse_args()
    res = {}; dump = {}
    for name, (pre, keys, gate) in DATASETS.items():
        d, hits = load(a.tel, pre, keys)
        if d is None or not len(d):
            print(name, 'no data'); continue
        R, RR, D = one(d, hits, gate)
        if a.logs:
            R['flow_logs'] = dict(zip(('n_tel_single', 'n_det4_record'), cut_flow(a.logs, '|'.join(p + '_' for p in pre))))
        res[name] = R
        print(f'\n== {name}: rows {R["n_rows"]}, single-cluster telescope {R["n_tel"]}, all estimators valid {R["n_common"]}')
        for e, S in R['est'].items():
            line = f'  {e:6s} det(A) {S["det_A"]:+.3f} roll {S["roll"]:+.1f} fid {S["fid_frac"]:.2f}'
            for v in 'XY':
                V = S[v]
                zz = ' '.join(f'{q["pitch"]}:{q["sigma_res"] * 1e3:.0f}+-{q["err"] * 1e3:.0f}->{q["sigma_det4"] * 1e3:.0f}' for q in V['zones'])
                fit = f' FIT f={V["fit"]["f_back"]:.3f} det4={V["fit"]["sigma_det4"] * 1e3:.0f}+-{V["fit"]["sigma_det4_err"] * 1e3:.0f}' if V['fit'] else ''
                line += f' | {v} n{V["n_fid"]} res {V["sigma_res"] * 1e3:.0f} [{zz}]{fit}'
            print(line)
            dump[f'{name}|{e}|x'] = RR[e][1][RR[e][2]]; dump[f'{name}|{e}|y'] = RR[e][0][RR[e][3]]
            dump[f'{name}|{e}|bx'] = D.bx_p.values[RR[e][3]]; dump[f'{name}|{e}|by'] = D.by_p.values[RR[e][2]]
            dump[f'{name}|{e}|rx_all'] = RR[e][0]; dump[f'{name}|{e}|ry_all'] = RR[e][1]
        dump[f'{name}|ty'] = (D.fy_p + (D.by_p - D.fy_p) * Z_DET4 / Z_BACK).values
        dump[f'{name}|tx'] = (D.fx_p + (D.bx_p - D.fx_p) * Z_DET4 / Z_BACK).values
        dump[f'{name}|d4x'] = D.d4_x_p0.values; dump[f'{name}|d4y'] = D.d4_y_p0.values
        for t, T in R['tests'].items():
            print(f'  test "{t}": passes {T["frac_pass"]:.2f}; X core pass/fail/all ' +
                  '/'.join(f'{x[0] * 1e3:.0f}' for x in (T['X']['pass_'], T['X']['fail'], T['X']['all'])) +
                  ' ; Y ' + '/'.join(f'{x[0] * 1e3:.0f}' for x in (T['Y']['pass_'], T['Y']['fail'], T['Y']['all'])))
        if 'timing' in R:
            print('  timing', {k: v for k, v in R['timing'].items() if not k.startswith('median')})
    F = footprint(a.fp)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    F.to_csv(os.path.join(HERE, 'results', 'footprint.csv'), index=False, float_format='%.4f')
    print(F[['key', 'resist_V', 'field', 'n', 'x_strips_mean', 'y_strips_mean', 'x_lead_frac', 'y_lead_frac', 'x_nb', 'y_nb', 'x_q_sum']].to_string(index=False))
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    np.savez_compressed(a.out.replace('.json', '_resid.npz'), **dump)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
