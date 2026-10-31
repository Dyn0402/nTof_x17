#!/usr/bin/env python3
"""kernel_job.py <registry key> --out O.npz [--n 3000] [--qlo 400] [--qhi 3000]

The det4 charge-sharing kernel of one SPS row on the FINAL cache builder (sps_build_cache: the oscillating ch 372/510,
the per-plateau hot channels and, in their ON events, the channels coupled to an oscillator all censored; the same
events and windows as the pass, peak >= 300 ADC and >= 4 strips).  For the Paper I/II kernel figures (kernel_forms:
run71 RAW head-on; angled_kernel: run63 flat700 against run63_rot25 d425/d325, one TAX access apart).

Per event and view:
  - lead = the brightest strip that is not censored (H4 dead list + RAW_CM_BAD + hot + coupled), its amplitude inside
    [qlo, qhi] ADC (unsaturated, above the 5-sigma seed);
  - profile a(d) / a(0) for d = -3..+3 strips in position, each a(d) the strip's maximum within +-4 samples (240 ns)
    of the lead's peak sample (as sps_cluster); NaN where the strip is censored, missing or not the adjacent pitch;
  - traces W(d, t) / a(0), aligned on the lead's peak sample, t = -5..+20 samples, for d = 0, +-1, +-2.
Writes the per-event profiles and traces (so the plot can bootstrap) and the view's mean peak-time offsets.
"""
import argparse, csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
import sps_build_cache as sbc                                     # noqa: E402

PITCH = 0.78
OFFS = np.arange(-3, 4)
T_LO, T_HI = -5, 20
T_HALF = 4


def one_view(P, bad, qlo, qhi):
    W = np.asarray(P['W'], float); pos = np.asarray(P['pos'], float); ch = np.asarray(P['ch'], int)
    good = ~np.isin(ch, np.asarray(sorted(bad), int))
    afull = np.where(good, W.max(1), -np.inf)
    i0 = int(np.argmax(afull)); a0 = afull[i0]
    if not (qlo <= a0 <= qhi):
        return None
    tpk = int(np.argmax(W[i0]))
    a = W[:, max(0, tpk - T_HALF):tpk + T_HALF + 1].max(1)
    prof = np.full(len(OFFS), np.nan)
    tr = np.full((5, T_HI - T_LO), np.nan)
    for k, d in enumerate(OFFS):
        j = i0 + d
        if 0 <= j < len(a) and good[j] and abs(abs(pos[j] - pos[i0]) - abs(d) * PITCH) < 0.3 * PITCH:
            prof[k] = a[j] / a0
            if abs(d) <= 2:
                lo, hi = tpk + T_LO, tpk + T_HI
                seg = np.full(T_HI - T_LO, np.nan)
                s0, s1 = max(lo, 0), min(hi, W.shape[1])
                seg[s0 - lo:s1 - lo] = W[j, s0:s1] / a0
                tr[[0, 1, -1, 2, -2].index(d)] = seg
    return prof, tr, a0, pos[i0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--qlo', type=float, default=400.0); ap.add_argument('--qhi', type=float, default=3000.0)
    a = ap.parse_args()
    row = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}[a.key]
    dead = json.load(open(os.path.join(REPO, 'rerun', 'channel_layer', 'det4_sps_h4.json')))
    evs = {}
    for sub in row['subruns'].split():
        got = sbc.build_cache(row['run'], sub, row['dataset'], a.n - len(evs), only=[row['plateau']], verbose=False)
        evs.update(got.get(row['plateau'], {}))
        if len(evs) >= a.n:
            break
    out = {}
    for v in 'xy':
        P, T, A, X = [], [], [], []
        for k, e in evs.items():
            hot = set()
            for (r_, s_, L), chs in sbc.HOT_LAST.items():
                if L == row['plateau']:
                    hot |= set(chs)
            bad = set(dead[v]['dead']) | set(sbc.RAW_CM_BAD) | hot | set(e.get(f'censor_{v}', []))
            r = one_view(e[v], bad, a.qlo, a.qhi)
            if r is None:
                continue
            P.append(r[0]); T.append(r[1]); A.append(r[2]); X.append(r[3])
        out[f'{v}_prof'] = np.array(P); out[f'{v}_tr'] = np.array(T); out[f'{v}_amp'] = np.array(A); out[f'{v}_pos'] = np.array(X)
        print(f'{a.key} {v}: {len(P)} events, mean a(+-1)/a(0) '
              f'{np.nanmean(np.array(P)[:, [2, 4]]) if P else float("nan"):.3f}', flush=True)
    out['offs'] = OFFS; out['t_rel_ns'] = (np.arange(T_LO, T_HI)) * 60.0; out['tr_order'] = np.array([0, 1, -1, 2, -2])
    out['meta'] = np.array(json.dumps(dict(key=a.key, n_events=len(evs), qlo=a.qlo, qhi=a.qhi,
                                           hot={f'{k[1]} {k[2]}': v for k, v in sbc.HOT_LAST.items()},
                                           coupled={f'{k[1]} {k[2]}': v for k, v in sbc.COUPLE_LAST.items()})))
    np.savez_compressed(a.out, **out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
