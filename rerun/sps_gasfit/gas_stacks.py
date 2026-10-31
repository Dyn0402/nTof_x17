#!/usr/bin/env python3
"""gas_stacks.py --out-dir D [--n 4000] [--blocks 20] <registry key> [...]

Trigger-placed strip-offset stacks of det4's SPS events for the gas fit (rerun/sps_gasfit/README.md), built with the
pass's own cache builder (sps_build_cache.build_cache: seed fix, per-plateau hot probe, RAW common mode; the same
events the final SPS pass reconstructs).  Per row and view, per block of consecutive events (time order), for every
strip offset o = -16..16 from the seed strip (in position) and DREAM sample s:

  S[b, o, s] = sum of the sample over events,  N[b, o, s] = number of events that HAVE it.

The stacked waveform of a strip set is sum_o S/N.  Rules from the stack-artefact traps (memory
stack-artefacts-dream-waveforms): time = sample index (the beam trigger opens the window; no alignment on the pulse);
a RAW sample the FEU dropped (exact 0 after the builder's baseline) is missing, not zero; a ZS zero is a measured
zero; an oscillating/hot strip the builder blanked is missing.  Events whose window holds a sample >= SAT (the pass's
3550 rule) are kept apart (``sat`` accumulators), because a clipped lead strip flattens the column.

RAW rows also get the zero-suppression emulation (``zs4``, ``zs5``): each channel's noise sigma from its pre-trigger
samples (the first PRE), a sample at or below k sigma set to 0 as the FEU drops it.  The fit corrects ZS stacks
additively with (emulated - RAW) at the nearest field (cloud_basics FINDINGS §25; a ratio fails past the drift end).

Writes <out-dir>/<key>.npz and <key>.json (counts, blocks' wall-clock, hot channels).
"""
import argparse, csv, json, os, sys
import numpy as np

REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis'),
                os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
HW = 16
SAT = 3550.0
PRE = 8
KS = (4.0, 5.0)


def offsets(rec, v, pos_seed, pitch):
    return np.rint((rec[v]['pos'] - pos_seed) / pitch).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+'); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n', type=int, default=4000); ap.add_argument('--blocks', type=int, default=20)
    a = ap.parse_args()
    import sps_build_cache as sbc
    from det4_sps_map import POSITION_MM, PITCH_MM
    os.makedirs(a.out_dir, exist_ok=True)
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}
    for key in a.keys:
        row = reg[key]; raw = row['raw'] == '1'; nsmp = int(row['n_samples']); evs = {}; hot = {}
        for sub in row['subruns'].split():
            need = a.n - len(evs)
            if need <= 0:
                break
            got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=need, only=[row['plateau']], verbose=False)
            hot[sub] = sbc.HOT_LAST.get((row['run'], sub, row['plateau']), [])
            for eid, ev in got.get(row['plateau'], {}).items():
                if len(evs) < a.n:
                    evs[(sub, eid)] = ev
        E = sorted(evs.values(), key=lambda e: e['t_wall'])
        n = len(E); nb = max(1, min(a.blocks, n // 50)); blk = np.minimum(np.arange(n) * nb // max(n, 1), nb - 1)
        print(f'{key}: {n} events, {nb} blocks, raw={raw}', flush=True)
        # per-channel pre-trigger noise (RAW only), over the present samples of every event
        sig = None
        if raw:
            s2 = np.zeros(512); c2 = np.zeros(512)
            for e in E:
                for v in 'xy':
                    W = e[v]['W'][:, :PRE]; ok = W != 0
                    np.add.at(s2, e[v]['ch'], (W ** 2 * ok).sum(1)); np.add.at(c2, e[v]['ch'], ok.sum(1))
            sig = np.sqrt(s2 / np.maximum(c2, 1)); sig[c2 < 20] = np.nan
        arms = ['all'] + ([f'zs{int(k)}' for k in KS] if raw else [])
        acc = {(v, arm, sat): (np.zeros((nb, 2 * HW + 1, nsmp)), np.zeros((nb, 2 * HW + 1, nsmp)))
               for v in 'xy' for arm in arms for sat in (0, 1)}
        prof = {v: np.zeros(2 * HW + 1) for v in 'xy'}; nev = np.zeros((nb, 2), int); tw = np.zeros(nb)
        for i, e in enumerate(E):
            b = blk[i]; tw[b] += e['t_wall']
            for v in 'xy':
                r = e[v]; W = np.asarray(r['W'], float); o = offsets(e, v, POSITION_MM[e[f'seed_ch_{v}']], PITCH_MM)
                k = np.abs(o) <= HW
                W, o, ch = W[k], o[k] + HW, r['ch'][k]
                sat = int(W.max() >= SAT); nev[b, sat] += (v == 'x')
                present = (W != 0) if raw else np.ones_like(W, bool)
                prof[v][o] += np.clip(W, 0, None).sum(1)
                for arm in arms:
                    if arm == 'all':
                        Wa, pa = W, present
                    else:
                        thr = float(arm[2:]) * sig[ch][:, None]
                        Wa = np.where(W > np.nan_to_num(thr, nan=1e9), W, 0.0); pa = present   # ZS keeps the zeros, not the drops
                    S, N = acc[(v, arm, sat)]
                    np.add.at(S[b], o, Wa * pa)
                    if raw:
                        np.add.at(N[b], o, pa)
                    else:
                        N[b] += 1          # ZS: a strip of the window with no kept sample is a measured zero, all samples

        nb_ev = np.bincount(blk, minlength=nb)
        out = {f'{v}_{arm}_{"sat" if sat else "ok"}_{q}': arr for (v, arm, sat), (S, N) in acc.items()
               for q, arr in (('S', S.astype(np.float32)), ('N', N.astype(np.int32)))}
        out.update(t_wall=tw / np.maximum(nb_ev, 1), n_ev=nev, prof_x=prof['x'], prof_y=prof['y'])
        if sig is not None:
            out['sigma'] = sig
        np.savez_compressed(os.path.join(a.out_dir, f'{key}.npz'), **out)
        json.dump(dict(key=key, n=n, blocks=int(nb), raw=raw, n_samples=nsmp, field_Vcm=float(row['field_Vcm']),
                       gas=row['gas'], mount=row['mount'], zs_sigma=row['zs_sigma'], hot=hot,
                       n_sat=int(nev[:, 1].sum()), t_wall=[float(x) for x in tw / np.maximum(nb_ev, 1)]),
                  open(os.path.join(a.out_dir, f'{key}.json'), 'w'), indent=1)
        print(f'{key}: saturated {nev[:, 1].sum()}/{n}', flush=True)


if __name__ == '__main__':
    main()
