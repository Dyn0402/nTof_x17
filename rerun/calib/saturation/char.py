#!/usr/bin/env python3
"""char.py -- what a clipped DREAM pulse looks like in the raw ADC (no fit), on the held-out events 180-1500 of each
chamber's cache + rails sidecar (rails.py).

    char.py [DET ...]        -> results/char_<det>.json, printed table

Per chamber:
  * clipping census: planes / strips with a raw-4095 sample, plateau length L (samples at 4095)
  * the top rail in W units (4095 - ped - CM): where it sits against the fit's SAT = 3550; clipped samples the 3550
    rule misses (W < 3550) and unclipped samples it censors (W >= 3550, raw < 4095)
  * after the clip: does the strip fall to the ADC floor (raw 0), how fast, for how long, vs L; the same for strips that
    never clipped (control: the raw minimum after the peak of a bright unclipped strip)
  * coherent chip events: |CM| > COH = 1000 ADC in any sample of the window (a whole 64-channel block moved; the CM-subtracted W
    of every strip in that block is then not a measurement)
  * raw maxima near the rail (pile-up at 4095 vs a soft roll-off below it)
"""
import json, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(os.environ.get('CLAUDE_JOB_DIR', os.path.expanduser('~/.claude/jobs/c68d6c6a')), 'tmp')
COH = 1000.0     # |CM| above this: a whole 64-channel block moved (chip-wide saturation / underflow)
DETS = ['det3', 'det2', 'det4', 'det6', 'det7']


def load(det):
    C = pickle.load(open(f'{TMP}/t15/caches/{det}.pkl', 'rb'))
    R = pickle.load(open(f'{TMP}/satur/rails_{det}.pkl', 'rb'))
    return C, R


def q(a, ps=(10, 50, 90)):
    a = np.asarray(a, float)
    return [round(float(x), 1) for x in np.percentile(a, ps)] if len(a) else None


def char(det):
    C, R = load(det)
    ids = sorted(C)[180:1500]
    S = dict(planes=0, planes_clip=0, planes_sat3550=0, planes_coh=0, strips_clip=0)
    railW, miss, extra, nclip_s = [], 0, 0, 0
    byL = {}                                      # L -> list of (floor reached, n floor samples, fall samples, end raw)
    ctrl_min = []                                 # unclipped strips with max 3000-4094: min after peak - baseline
    rawmax = []
    for e in ids:
        for p in 'xy':
            if p not in R.get(e, {}):
                continue
            r = R[e][p]; raw = r['raw'].astype(int); W = np.asarray(C[e][p]['W'], float)
            S['planes'] += 1
            cl = raw >= 4095
            S['planes_clip'] += int(cl.any()); S['planes_sat3550'] += int((W >= 3550).any())
            coh = np.abs(r['cm']) > COH
            S['planes_coh'] += int(coh.any())
            railW += list(W[cl & ~coh]); miss += int((cl & (W < 3550)).sum()); extra += int(((W >= 3550) & ~cl).sum())
            nclip_s += int(cl.sum())
            m = raw.max(1); rawmax += list(m[m > 3800])
            base = np.median(raw[:, :4], axis=1)
            for i in range(len(raw)):
                if coh[i].any():
                    continue
                if cl[i].any():
                    S['strips_clip'] += 1
                    k = np.flatnonzero(cl[i]); a, b = k.min(), k.max(); L = int(b - a + 1)
                    post = raw[i, b + 1:]
                    fl = np.flatnonzero(post <= 0)
                    byL.setdefault(min(L, 10), []).append((int(len(fl) > 0), int(len(fl)), int(fl[0] + 1) if len(fl) else -1,
                                                           int(post[-1]) if len(post) else -1, int(len(post))))
                elif 3000 <= m[i] < 4095:
                    j = int(np.argmax(raw[i]))
                    if j + 1 < raw.shape[1]:
                        ctrl_min.append(float(raw[i, j + 1:].min()))
    out = dict(det=det, **S, frac_planes_clip=S['planes_clip'] / S['planes'],
               clipped_samples=nclip_s, top_rail_W_q=q(railW, (1, 10, 50, 90, 99)),
               clipped_but_W_below_3550=miss, W_ge_3550_not_clipped=extra,
               rawmax_at_4095=int(np.sum(np.array(rawmax) == 4095)),
               rawmax_4000_4094=int(np.sum((np.array(rawmax) >= 4000) & (np.array(rawmax) < 4095))),
               rawmax_3900_3999=int(np.sum((np.array(rawmax) >= 3900) & (np.array(rawmax) < 4000))),
               ctrl_unclipped_bright_rawmin_q=q(ctrl_min), ctrl_n=len(ctrl_min),
               ctrl_frac_reach_floor=float(np.mean(np.array(ctrl_min) <= 5)) if ctrl_min else None)
    L = {}
    for k in sorted(byL):
        a = np.array(byL[k])
        L[str(k)] = dict(n=len(a), p_floor=round(float(a[:, 0].mean()), 3),
                         floor_samples_med=float(np.median(a[a[:, 0] == 1, 1])) if a[:, 0].any() else None,
                         fall_to_floor_med=float(np.median(a[a[:, 0] == 1, 2])) if a[:, 0].any() else None,
                         post_samples_med=float(np.median(a[:, 4])),
                         end_raw_q=q(a[:, 3]))
    out['by_plateau_len'] = L
    allp = np.concatenate([np.array(v) for v in byL.values()]) if byL else np.zeros((0, 5))
    out['p_floor_all'] = float(allp[:, 0].mean()) if len(allp) else None
    json.dump(out, open(f'{HERE}/results/char_{det}.json', 'w'), indent=1)
    return out


def main(dets):
    rows = [char(d) for d in dets]
    print('| chamber | planes w/ clip | planes coherent CM | clipped strips | top rail W p10/p50/p90 | clipped, W<3550 | '
          'W>=3550, not clipped | P(floor after clip) | ctrl bright unclipped reach floor |')
    print('|---|---|---|---|---|---|---|---|---|')
    for o in rows:
        t = o['top_rail_W_q']
        print(f"| {o['det']} | {100*o['frac_planes_clip']:.1f} % | {100*o['planes_coh']/o['planes']:.1f} % | {o['strips_clip']} | "
              f"{t[1]:.0f}/{t[2]:.0f}/{t[3]:.0f} | {o['clipped_but_W_below_3550']} of {o['clipped_samples']} | "
              f"{o['W_ge_3550_not_clipped']} | {100*o['p_floor_all']:.0f} % | {100*o['ctrl_frac_reach_floor']:.1f} % (n {o['ctrl_n']}) |")
    print('\nP(strip falls to raw 0 after the clip) and the median fall time [samples] by plateau length L:')
    print('| chamber | ' + ' | '.join(f'L={k}' for k in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']) + ' |')
    print('|---|' + '---|' * 10)
    for o in rows:
        cells = []
        for k in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
            b = o['by_plateau_len'].get(k)
            cells.append('—' if b is None else f"{100*b['p_floor']:.0f} % ({b['n']}; fall {b['fall_to_floor_med']})")
        print(f"| {o['det']} | " + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main(sys.argv[1:] or DETS)
