#!/usr/bin/env python3
"""headon2.py -- head-on recovery pushed to the extreme cases (1 strip, low SNR, n_TOF's doubled noise) by signal
injection, with the fake rate measured on the same noise (Dylan 10-26: "keep working on an adaptive/elegant head-on
recovery ... see if you can even extend to the more extreme scenarios without skyrocketing the fake count").

    ../../.venv/bin/python headon2.py extract     # waveforms: head-on donors' footprints + every 'outside' plane
    ../../.venv/bin/python headon2.py scan        # rules x gain x noise -> headon2.json

Everything works on waveforms (the DAQ hits are emulated from them: significance = SIG_SCALE x max / noise, a hit
at >= 5, as mm_processor; measured ratio 1.26-1.39, correlation 0.98-1.00 on real events).

Signal: interior M3 rays with |tan| < 0.03 in the view under study (x: 868, y: 1 131 of the golden row). The donor's
strips within +-REG mm of its M3 footprint (mesh + 15 tan) are scaled by k (gain relative to 490 V) and added onto a
real noise plane: the same view of an 'outside' event (M3 misses det3; all 1 339, hits or not, so bands and stray
particles are in). noise x2 (n_TOF after 23 July) adds a second outside plane x sqrt(3), and the hit significance is
then taken against 2 sigma, as a DAQ pedestal run would set it. Truth: the M3 footprint; found = a seed whose
charge centroid is within FOUND mm.
Fakes: the same noise planes with nothing added, each with the t0 of a random real track of the other view.

Rules (each adds to the one before, as a rescue: it only acts when the earlier rules give the plane no seed):
  prod     plane-relative floor 10 %, clusters (12 mm gap) of >= 3 strips  (wft.seed)
  ho20     + 2-strip clusters whose weaker strip has significance >= 20     (reco_variant.py --headon-sig)
  s1_T     + a single strip with significance >= T (plain 1-strip rule, no other information)
  col_T    + the column finder: needs the OTHER view's track t0. Per strip, the baseline is the mean of the samples
           outside the column window (before t0 - PRE and after t0 + T_DRIFT + POST: a median, so a drifting
           pedestal is removed and not integrated); the column statistic Z = sum over [t0 - PRE, t0 + T_DRIFT + POST] of (W - baseline)
           / (sigma sqrt(n)). The best strip with Z >= T, if (a) the charge starts inside the window (pre-window
           samples flat: a flat offset / band has no edge) and (b) it lasts >= MIN_DUR samples above 30 % of its
           peak (a drift column, not a noise blip); the cluster is it and its neighbours with Z >= T/3.
"""
import json, os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402

CACHE = os.path.join(S.OUT, 'headon2_wf.pkl')
EV = os.path.expanduser('~/x17/cosmic_bench/outliers/final/mx17_det3_saturday_scan_6-27-26/'
                        'long_run_resist_490V_drift_1000V/mx17_3/wft/final_phys/events.parquet')
SIG_SCALE, HIT_SIG, REG, FOUND = 1.3, 5.0, 4.0, 2.0
TS, T_DRIFT, PRE, POST, MIN_DUR = 60.0, 840.0, 120.0, 180.0, 6
GAP, REL = 12.0, 0.10
KS = (1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05)


def rays():
    R = pd.read_parquet(os.path.expanduser('~/x17/cosmic_bench/effbudget/rays_det3.parquet'))
    return R[~R.not_measured].set_index('event_id')


def extract():
    from qa_config import get_config
    from wft import io as wio
    R = rays(); I = R[R.region == 'interior']; O = R[R.region == 'outside']
    E = pd.read_parquet(EV).drop_duplicates('event_id').set_index('event_id')
    cfg = get_config(S.KEY); pm = wio.strip_position_map(cfg)
    out = dict(pos={}, noise={}, donors={}, empty={}, t0={})
    for v, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        o = 'y' if v == 'x' else 'x'
        D = I[I[f'tan_{v}'].abs() < 0.03]
        # the other view must have a fit (its t0 is the column finder's input)
        D = D[D.index.isin(E.index[E[f'{o}_ok'].fillna(False).astype(bool)])]
        p = pm[feu]; out['pos'][v] = p
        want = set(D.index) | set(O.index); dn, em = {}, {}
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            r = wio.FeuReader(f); out['noise'][v] = r.noise
            for eid, ft, W in r.iter_events(want & set(r.event_ids.tolist())):
                if eid in D.index:
                    c = D.loc[eid, f'mesh_{v}'] + 15 * D.loc[eid, f'tan_{v}']
                    reg = np.flatnonzero(np.isfinite(p) & (np.abs(p - c) <= REG))
                    dn[eid] = dict(reg=reg, W=W[reg].astype(np.float32), truth=float(c),
                                   t0=float(E.loc[eid, f'{o}_t0']), cat=D.loc[eid, f'cat_{v}_5'])
                if eid in O.index:
                    em[eid] = W.astype(np.float16)
        out['donors'][v], out['empty'][v] = dn, em
        # t0 pool for fakes: real tracks of the other view
        out['t0'][v] = E[f'{o}_t0'][E[f'{o}_ok'].fillna(False).astype(bool)].to_numpy()
        print(v, len(dn), 'donors', len(em), 'noise planes', flush=True)
    pickle.dump(out, open(CACHE, 'wb'))


# --------------------------------------------------------------------------- #
def clusters(pos, idx, gap=GAP):
    o = idx[np.argsort(pos[idx])]
    if not len(o):
        return []
    cut = np.flatnonzero(np.diff(pos[o]) > gap) + 1
    return np.split(o, cut)


def seed_hits(W, sigma, pos, rule_extra):
    """W: (512, ns) waveform plane; sigma: per-channel noise used for significance. Returns list of (rule, strips)."""
    ok = np.isfinite(pos)
    sig = np.where(ok, SIG_SCALE * W.max(1) / np.maximum(sigma, 1e-3), 0.0)
    hit = np.flatnonzero(sig >= HIT_SIG)
    if not len(hit):
        return [], sig
    kept = hit[sig[hit] >= REL * sig[hit].max()]
    cl = clusters(pos, kept)
    out = [('prod', c) for c in cl if len(c) >= 3]
    if out:
        return out, sig
    for name, thr in rule_extra:
        if name == 'ho20':
            out += [('ho20', c) for c in cl if len(c) == 2 and sig[c].min() >= thr]
        elif name == 's1':
            out += [('s1', c) for c in cl if len(c) >= 1 and sig[c].max() >= thr]
        if out:
            return out, sig
    return out, sig


def column(W, sigma, pos, t0, thr):
    ns = W.shape[1]; t = np.arange(ns) * TS
    pre = t < t0 - PRE
    win = (t >= t0 - PRE) & (t <= t0 + T_DRIFT + POST)
    if pre.sum() < 2 or win.sum() < 4 or (~win).sum() < 6:
        return None
    base = np.median(W[:, ~win], axis=1)
    X = W - base[:, None]
    Z = X[:, win].sum(1) / (np.maximum(sigma, 1e-3) * np.sqrt(win.sum()))
    Z = np.where(np.isfinite(pos), Z, -np.inf)
    # pre-window flatness: a band/offset is flat everywhere and removed by the baseline; a genuinely early signal
    # (rising before the window) is not a column of THIS track
    prez = np.abs(W[:, pre] - base[:, None]).max(1) / np.maximum(sigma, 1e-3)
    best = int(np.argmax(Z))
    if Z[best] < thr or prez[best] > 5:
        return None
    pr = X[best, win]; dur = int((pr > 0.3 * pr.max()).sum())
    if dur < MIN_DUR:
        return None
    nb = np.flatnonzero(np.isfinite(pos) & (np.abs(pos - pos[best]) <= 1.6) & (Z >= thr / 3))
    return nb, Z


def centroid(W, pos, c):
    q = np.clip(W[c].max(1), 0, None)
    return float((pos[c] * q).sum() / q.sum()) if q.sum() > 0 else float(pos[c].mean())


RESCUE = [('ho20', 20.0), ('s1', 20.0), ('s1', 10.0), ('s1', 7.0), ('col', 10.0), ('col', 15.0), ('col', 20.0)]
#: cross-view charge consistency: |ln(Q_this / (R_Q x q_other))| <= Q_TOL; R_Q and its spread from detected head-ons.
#: '+q': the tolerance is Q_TOL_SIG x the measured spread; '+qa' (adaptive): Q_TOL_SIG x hypot(spread, sigma_Q / Q), with
#: sigma_Q the cluster's own noise over the window, and Q > 3 sigma_Q required
Q_TOL_SIG = 3.0


def win_mask(ns, t0):
    t = np.arange(ns) * TS
    return (t >= t0 - PRE) & (t <= t0 + T_DRIFT + POST)


def qraw(W, c, t0):
    """Charge of cluster c in the column window, baseline = median outside it [ADC x samples]."""
    w = win_mask(W.shape[1], t0)
    base = np.median(W[c][:, ~w], axis=1) if (~w).sum() >= 4 else 0.0
    return float((W[c][:, w] - np.atleast_1d(base)[:, None]).sum())


def rescue(W, sigma, pos, t0):
    """Production seeds, or (when there are none) every rescue candidate: [(rule, cluster)]."""
    sd, sig = seed_hits(W, sigma, pos, [])
    if sd:
        return sd
    out = []
    hit = np.flatnonzero(sig >= HIT_SIG)
    cl = clusters(pos, hit[sig[hit] >= REL * sig[hit].max()]) if len(hit) else []
    for name, thr in RESCUE:
        if name == 'ho20':
            out += [(f'ho20', c) for c in cl if len(c) == 2 and sig[c].min() >= thr]
        elif name == 's1':
            out += [(f's1_{thr:g}', c) for c in cl if sig[c].max() >= thr]
        else:
            c = column(W, sigma, pos, t0, thr)
            if c is not None:
                out.append((f'col_{thr:g}', c[0]))
    return out


RULE_NAMES = ['ho20', 's1_20', 's1_10', 's1_7', 'col_10', 'col_15', 'col_20']


SIGMA_PLANE = None   # per-channel noise of the plane under study (set by scan)


def evaluate(cands, W, pos, t0, qo, rq, qsig, truth=None):
    """-> {rule[+q]: fired / found}. A rule = ho20 OR itself (the rules stack on the 2-strip rule)."""
    res = {}
    prod = [c for r, c in cands if r == 'prod']
    hitf = (lambda c: True) if truth is None else (lambda c: abs(centroid(W, pos, c) - truth) <= FOUND)
    res['prod'] = any(hitf(c) for c in prod)
    if prod:
        for r in RULE_NAMES:
            res[r] = res[r + '+q'] = res[r + '+qa'] = res['prod']
        return res
    qok = lambda c: qo > 0 and abs(np.log(max(qraw(W, c, t0), 1e-3) / (rq * qo))) <= Q_TOL_SIG * qsig

    def qok_a(c):
        # adaptive: the cluster's own charge noise widens the tolerance, sigma^2 = qsig^2 + (sigma_Q / Q)^2
        if qo <= 0:
            return False
        q = qraw(W, c, t0); nwin = int(win_mask(W.shape[1], t0).sum())
        sq = float(np.sqrt((SIGMA_PLANE[c] ** 2).sum() * nwin))
        if q <= 3 * sq:
            return False
        return abs(np.log(q / (rq * qo))) <= Q_TOL_SIG * np.hypot(qsig, sq / q)
    for r in RULE_NAMES:
        cs = [c for n, c in cands if n == r or n == 'ho20']
        res[r] = any(hitf(c) for c in cs)
        res[r + '+q'] = any(hitf(c) and qok(c) for c in cs)
        res[r + '+qa'] = any(hitf(c) and qok_a(c) for c in cs)
    return res


def scan():
    C = pickle.load(open(CACHE, 'rb')); rng = np.random.default_rng(3)
    Ev = pd.read_parquet(EV).drop_duplicates('event_id').set_index('event_id')
    out = dict(ks=KS, rules=['prod'] + RULE_NAMES + [r + '+q' for r in RULE_NAMES] + [r + '+qa' for r in RULE_NAMES], signal={}, fake={}, recover={},
               calib={}, n={})
    for v in 'xy':
        o = 'y' if v == 'x' else 'x'
        pos, noise = C['pos'][v], C['noise'][v].astype(np.float32)
        E = [w.astype(np.float32) for w in C['empty'][v].values()]
        dn = C['donors'][v]
        ok = Ev[f'{o}_ok'].fillna(False).astype(bool)
        pool = Ev.loc[ok, [f'{o}_t0', f'{o}_q_sum']].to_numpy()
        qo_d = {e: float(Ev.loc[e, f'{o}_q_sum']) for e in dn}
        for nf in (1, 2):
            sigma = noise * nf
            global SIGMA_PLANE
            SIGMA_PLANE = sigma
            planes = [E[i] + (np.sqrt(3) * E[(i + 1) % len(E)] if nf == 2 else 0) for i in range(len(E))]
            clean = [W for W in planes if not seed_hits(W, sigma, pos, [])[0]]
            # charge calibration: detected head-ons at k = 1 (production seed on the footprint)
            rat = []
            for j, (eid, d) in enumerate(dn.items()):
                W = clean[j % len(clean)].copy(); W[d['reg']] += d['W']
                sd, _ = seed_hits(W, sigma, pos, [])
                for _, c in sd:
                    if abs(centroid(W, pos, c) - d['truth']) <= FOUND and qo_d[eid] > 0:
                        rat.append(qraw(W, c, d['t0']) / qo_d[eid])
            rat = np.log(np.clip(np.array(rat), 1e-6, None))
            rq = float(np.exp(np.median(rat))); qsig = float(1.4826 * np.median(np.abs(rat - np.median(rat))))
            out['calib'][f'{v}|{nf}'] = dict(r_q=rq, sigma_ln=qsig, n=int(len(rat)), n_clean=len(clean), n_planes=len(planes))
            print(v, nf, 'clean planes', len(clean), '/', len(planes), 'R_Q', round(rq, 3), 'sigma ln', round(qsig, 3), flush=True)
            for k in KS:
                F = []
                for W in clean:
                    for t0, qo in pool[rng.integers(0, len(pool), 3)]:
                        F.append(evaluate(rescue(W, sigma, pos, t0), W, pos, t0, k * qo, rq, qsig))
                F = pd.DataFrame(F); out['fake'][f'{v}|{nf}|{k:g}'] = {c: float(F[c].mean()) for c in F}
                Sg = []
                for j, (eid, d) in enumerate(dn.items()):
                    W = clean[j % len(clean)].copy(); W[d['reg']] += k * d['W']
                    Sg.append(evaluate(rescue(W, sigma, pos, d['t0']), W, pos, d['t0'], k * qo_d[eid], rq, qsig, d['truth']))
                Sg = pd.DataFrame(Sg)
                out['signal'][f'{v}|{nf}|{k:g}'] = {c: float(Sg[c].mean()) for c in Sg}
                miss = ~Sg['prod']
                out['recover'][f'{v}|{nf}|{k:g}'] = dict(n=int(miss.sum()), **{c: float(Sg[c][miss].mean()) for c in Sg if c != 'prod'})
                print(f'  k {k:<5g} eff', ' '.join(f'{c}={x:.3f}' for c, x in out['signal'][f'{v}|{nf}|{k:g}'].items()), flush=True)
                print(f'          fake', ' '.join(f'{c}={x:.4f}' for c, x in out['fake'][f'{v}|{nf}|{k:g}'].items()), flush=True)
            out['n'][f'{v}'] = len(dn)
    json.dump(out, open(os.path.join(S.OUT, 'headon2.json'), 'w'), indent=1)


if __name__ == '__main__':
    {'extract': extract, 'scan': scan}[sys.argv[1]]()
