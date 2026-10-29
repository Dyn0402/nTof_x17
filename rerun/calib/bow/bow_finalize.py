"""Finalize the chamber bow: ONE block per detector from its golden row, applied to every other dataset of that detector
through the production code, with no bow feedback from the target (Dylan 10-29: no slot-dependent corrections; extract
on the golden set, apply to the others).

Per chamber:
  block   = set_bow.block_from_table on the golden table (all events), tan form (a_x, a_y) + mesh height h
  bundle  = the chamber's promoted calib_bundle with the block installed and w0/kw re-measured on the golden table by
            rerun/w12/set_w0kw.py --bow-json (exactly the pass's W0KW node)
Per target dataset of the chamber, three arms, all through set_w0kw.py and wft.reco.bow_UW / tan_bow (= apply_bow):
  nobow      no block; w0/kw re-measured on the target (what every row gets today)
  strict     the golden bundle as is: golden block AND golden w0/kw, nothing measured on the target
  pass       golden block; only the rigid tilt w0 and scale kw re-measured on the target (set_w0kw, as vrefit does)
w0/kw are 2 numbers per view fitted on thousands of target events (no held-out split; the bow shape itself never sees
the target).  The golden row's own numbers are in-sample for the shape (see bow_transfer.py for its 2-fold value).
Position: residual p0 - ref_mesh with the per-row alignment terms refit, with and without the golden own-view h_<view>.
Stability: bootstrap of the golden shape (30 resamples) -> spread of the tan offset it predicts across the chamber.
Usage: python -I bow_finalize.py [--manifest M.json] [--out-dir D]
  -> D/bow_finalize.json, D/blocks/bow_block_<det>.json, a table on stdout (default D = results/).
  Default datasets = FB's tables (fbcommon.DATASETS).  The pass's TRANSFER GATE gives a manifest instead:
  {"bundles": {det: golden calib_bundle dir}, "golden": {det: name},
   "datasets": {name: [det, joined table parquet (ws_v_w0/build_table layout), note]}}
"""
import json, os, shutil, subprocess, sys, tempfile
from types import SimpleNamespace
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '../../..'))
FB = os.path.join(REPO, 'rerun/model_audit/followups/fb_bow_maps')
sys.path[:0] = [REPO, FB]
from fbcommon import DATASETS as FB_DATASETS, DETS, ROOT, load_path, bundle as fb_bundle, surf_terms, huber              # noqa: E402
from gate_bow import prep, fit_bow2d                                      # noqa: E402
from set_bow import block_from_table                                      # noqa: E402
from wft.calib import CalibrationBundle                                   # noqa: E402
from wft import reco as wr                                                # noqa: E402

PY = os.path.join(REPO, '.venv/bin/python')
EXCLUDE = {'det3_o23', 'det2_o17'}
TMP = os.environ.get('BOW_TMP') or tempfile.mkdtemp(prefix='bowfin_')


def s68(d):
    d = d[np.isfinite(d)]
    return float(0.5 * np.diff(np.percentile(d, [16, 84]))[0]) if len(d) > 20 else np.nan


def angles_npz(name, T, d):
    """03_angles-style per_event.npz for set_w0kw from a table (fbcommon layout), on the gate's event selection."""
    os.makedirs(d, exist_ok=True); z = {}
    for p in 'xy':
        D = prep(T, p); k = D['m']
        z[f'{p}_eid'] = D['eid'][k]; z[f'{p}_w'] = D['wv'][k] / 1e3; z[f'{p}_tan_ref'] = D['t'][k]
    np.savez(f'{d}/per_event.npz', **z)
    return d


def w0kw(src_bundle, name, T, raw_table, out, bow_json=None):
    ang = angles_npz(name, T, f'{TMP}/ang_{name}')
    norm = f'{TMP}/table_{name}.parquet'
    if not os.path.exists(norm):                 # set_w0kw wants event_id as a column (the golden tables index on it)
        import pandas as pd
        d = pd.read_parquet(raw_table)
        (d.reset_index() if 'event_id' not in d.columns else d).to_parquet(norm)
    raw_table = norm
    cmd = [PY, '-I', f'{REPO}/rerun/w12/set_w0kw.py', name, '--bundle', src_bundle, '--angles', ang, '--out', out,
           '--table', raw_table]
    if bow_json:
        cmd += ['--bow-json', bow_json]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return CalibrationBundle.load(out)


def score(cal, T):
    """angle error [deg] per view through the production mapping (both views' coordinates, as apply_bow)."""
    D = {p: prep(T, p) for p in 'xy'}
    raw = T  # fbcommon table: x_wv (um/ns), x_p0
    out = {}
    for p in 'xy':
        k = D[p]['m']; eids = D[p]['eid'][k]
        sub = raw.loc[eids]
        tans = []
        for _, r in sub.iterrows():
            fx = SimpleNamespace(p0=r.x_p0, w=r.x_wv / 1e3) if np.isfinite(r.x_p0) and np.isfinite(r.x_wv) else None
            fy = SimpleNamespace(p0=r.y_p0, w=r.y_wv / 1e3) if np.isfinite(r.y_p0) and np.isfinite(r.y_wv) else None
            f = fx if p == 'x' else fy
            if getattr(cal, 'bow', None):
                U, W = wr.bow_UW(cal, fx, fy)
                tans.append(wr.tan_bow(cal, p, f.w, U, W))
            else:
                tans.append(wr.tan_const(cal, p, f.w))
        t = D[p]['t'][k]; tf = np.array(tans)
        out[p] = np.degrees(np.arctan(tf)) - np.degrees(np.arctan(t))
    return out


def pos_score(block, T):
    """position residual s68 [mm] without / with the golden block's own-view height h_<view>; the per-row alignment
    terms (offset, two in-plane slopes, the plane z offset t*h0) are refit on the row in both arms, as its own
    waveform alignment does."""
    out = {}
    for p in 'xy':
        D = prep(T, p); k = D['m']
        r, t, U, W = D['r'][k], D['t'][k], D['U'][k], D['W'][k]
        h = surf_terms(U, W)[:, 1:] @ np.array([block[f'h_{p}'][c] for c in ('gu', 'gw', 'kuu', 'kww', 'kuw')])
        X = np.c_[np.ones_like(U), U, W, t]
        res = []
        for y in (r, r - t * h):
            res.append(s68(y - X @ huber(X, y, 0.5)))
        out[p] = tuple(res)
    return out


def stability(Tg):
    rng = np.random.default_rng(11); res = {}
    for p in 'xy':
        D = prep(Tg, p); k = np.where(D['m'])[0]
        G = np.stack(np.meshgrid(np.linspace(-1.6, 1.6, 9), np.linspace(-1.6, 1.6, 9)), -1).reshape(-1, 2)
        S = surf_terms(G[:, 0], G[:, 1])[:, 1:]
        preds = []
        for _ in range(30):
            m = np.zeros_like(D['m']); m[rng.choice(k, len(k))] = True
            b = fit_bow2d(D, m); preds.append(S @ (b[1:6] / b[6]))
        P = np.degrees(np.array(preds))
        res[p] = dict(range_deg=float(np.ptp(P.mean(0))), boot_rms_deg=float(np.sqrt(np.mean(P.std(0) ** 2))))
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest'); ap.add_argument('--out-dir', default=f'{HERE}/results')
    a = ap.parse_args()
    if a.manifest:
        man = json.load(open(a.manifest))
        DATASETS = {n: (v[0], v[1], '', v[2]) for n, v in man['datasets'].items()}
        GOLD = man['golden']; BUNDLES = man['bundles']
    else:
        DATASETS = {n: v for n, v in FB_DATASETS.items() if n not in EXCLUDE}
        GOLD = {d: d for d in DETS}; BUNDLES = {d: f'{ROOT}/{d}/calib_bundle' for d in DETS}
    bdict = {d: json.load(open(f'{b}/bundle.json')) for d, b in BUNDLES.items()}
    load = lambda n: load_path(DATASETS[n][1], bdict[DATASETS[n][0]], DATASETS[n][0], DATASETS[n][2], DATASETS[n][3], n)
    OUT = a.out_dir
    os.makedirs(f'{OUT}/blocks', exist_ok=True)
    rng = np.random.default_rng(5); out = {}
    for det, gname in GOLD.items():
        Tg = load(gname)
        blk = block_from_table(Tg, det, f'{gname}: ' + DATASETS[gname][2] + ' ' + DATASETS[gname][3])
        bj = f'{OUT}/blocks/bow_block_{det}.json'; json.dump(blk, open(bj, 'w'), indent=1)
        src = BUNDLES[det]
        gold = w0kw(src, gname, Tg, DATASETS[gname][1], f'{TMP}/{det}_gold', bj)
        stab = stability(Tg)
        out[det] = dict(stability=stab, rows={})
        for name in [n for n, v in DATASETS.items() if v[0] == det]:
            T = load(name)
            arms = {'nobow': w0kw(src, name, T, DATASETS[name][1], f'{TMP}/{name}_nobow'),
                    'strict': gold,
                    'pass': w0kw(f'{TMP}/{det}_gold', name, T, DATASETS[name][1], f'{TMP}/{name}_pass')}
            E = {a: score(c, T) for a, c in arms.items()}
            row = dict(note=DATASETS[name][3], golden=(name == gname), pos=pos_score(blk, T))
            for p in 'xy':
                base = E['nobow'][p]; row[p] = {}
                for a in E:
                    d = E[a][p]
                    bs = [s68(d[j]) - s68(base[j]) for j in (rng.integers(0, len(d), len(d)) for _ in range(200))]
                    row[p][a] = dict(s68=s68(d), d=float(s68(d) - s68(base)), err=float(np.std(bs)),
                                     far=float(100 * np.mean(np.abs(d) > 5)), n=int(len(d)),
                                     w0=float(arms[a].w0.get(p, 0)), kw=float(arms[a].kw.get(p, 1)))
            out[det]['rows'][name] = row
            print(name, {p: {a: round(row[p][a]['d'], 3) for a in E} for p in 'xy'}, flush=True)
    json.dump(out, open(f'{OUT}/bow_finalize.json', 'w'), indent=1)
    print('\n| chamber | dataset | view | n | no bow s68 | strict Δ (far %) | pass Δ (far %) | position s68 [mm] no h → golden h |')
    print('|---|---|---|---|---|---|---|---|')
    for det, r in out.items():
        for name, row in r['rows'].items():
            for p in 'xy':
                q = row[p]; ps = row['pos'][p]
                print(f"| {det} | {name}{' (golden, in-sample)' if row['golden'] else ''} — {row['note']} | {p.upper()} | {q['nobow']['n']} | "
                      f"{q['nobow']['s68']:.3f} | {q['strict']['d']:+.3f} ± {q['strict']['err']:.3f} ({q['strict']['far']:.1f}) | "
                      f"{q['pass']['d']:+.3f} ± {q['pass']['err']:.3f} ({q['pass']['far']:.1f}) | {ps[0]:.3f} → {ps[1]:.3f} |")
    print('\nstability of the golden shape (bootstrap): ' + '; '.join(
        f"{d} X range {r['stability']['x']['range_deg']:.2f}° ± {r['stability']['x']['boot_rms_deg']:.3f}, Y {r['stability']['y']['range_deg']:.2f}° ± {r['stability']['y']['boot_rms_deg']:.3f}"
        for d, r in out.items()))


if __name__ == '__main__':
    main()
