#!/usr/bin/env python3
"""d_osc.py -- chamber D's permanently oscillating channels (open items, 2026-11-23).

    python -m rerun.ntof.ringing.d_osc find     # the channel list, per run, from the char_all.py caches
    python -m rerun.ntof.ringing.d_osc tracks   # how many D tracks have one in their fit window, and what it does

About 50 channels per D view carry a 2.5-2.8 MHz oscillation at every time after the flash (also >= 30 ms, where A
and C are quiet), at 5-13 times their pedestal noise.  A channel is called oscillating when its plain-CM waveform goes
below -5 sigma (pedestal noise) in more than 20 % of the late (>= 30 ms) events.  `find` writes the list (channels
common to the three runs) and says which ones the D bundle masks; `tracks` takes every gated D track of the window
test (window_own.py) and compares tracks whose own fit window holds an unmasked oscillating channel with the others:
late pointing excess, wall confirmation, x-y charge balance, chi2/dof.
Out: $WORK_DATA/ntof_pass1/ana/ringing/d_osc_channels.json, d_osc_tracks.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from rerun.ntof.ringing.char_all import load, OUT, cache_path          # noqa: E402

RUNS = (('run_104', 'stat090_0003'), ('run_139', 'stat090_0003'), ('run_145', 'stat090_0000'))
BUNDLE = Path('/media/ucla/work/ntof_pass1/bundles/pf/D_E2_beam_L1/bundle.json')


def find():
    b = json.loads(BUNDLE.read_text())
    out = {'rule': 'plain-CM waveform below -5 sigma (pedestal noise) in > 20 % of the >= 30 ms events, '
                   'run_145 stat090_0000 / run_104 / run_139 stat090_0003 tag 000', 'runs': {}, 'common': {}}
    for pl, feu in (('x', 1), ('y', 2)):
        sets = []
        for run, sub in RUNS:
            A, ms, ev, nz = load(run, sub, '000', feu)
            N, ns, _ = A.shape
            B = A.reshape(N, ns, 8, 64)
            R = (B - np.median(B, axis=3)[..., None]).reshape(N, ns, 512)
            lt = ms > 30
            fneg = ((R[lt] / nz).min(1) < -5).mean(0)
            rms = R[lt].std(axis=(0, 1))
            s = np.flatnonzero(fneg > 0.2)
            sets.append(set(s.tolist()))
            out['runs'].setdefault(run, {})[pl] = dict(channels=s.tolist(), rms_adc=np.round(rms[s], 1).tolist(),
                                                       rms_over_noise=np.round(rms[s] / nz[s], 2).tolist())
        com = sorted(set.intersection(*sets))
        masked = set(b['dead'][pl]) | set(b['hot'][pl])
        out['common'][pl] = dict(channels=com, masked=[c for c in com if c in masked],
                                 unmasked=[c for c in com if c not in masked])
        print(pl, len(com), 'common;', len(out['common'][pl]['unmasked']), 'unmasked')
    (OUT / 'd_osc_channels.json').write_text(json.dumps(out, indent=1))
    return out


def tracks():
    from sept26_prelim_analysis.capsule_tof import crossing
    from rerun.ntof.ringing.window_own import _pos_wrap, windows
    ch = json.loads((OUT / 'd_osc_channels.json').read_text())['common']
    rows = []
    for f in sorted(OUT.glob('wo_run_*.parquet')):
        d = pd.read_parquet(f)
        d = d[d.arm == 'D'].copy()
        run, sub = d.run.iloc[0], d.subrun.iloc[0]
        for pl, feu in (('x', 1), ('y', 2)):
            pos, _ = _pos_wrap(run, sub, feu)
            pitch = float(np.nanmedian(np.abs(np.diff(np.sort(pos[np.isfinite(pos)])))))
            un = np.zeros(512, bool)
            un[ch[pl]['unmasked']] = True
            n = []
            for p0, t, ns in zip(d[f'{pl}_p0'], d[f'{pl}_tan_theta'], d[f'{pl}_n_strips']):
                if not (np.isfinite(p0) and np.isfinite(t)):
                    n.append(0)
                    continue
                own, _ = windows(pos, p0, t, ns, pitch)
                n.append(int((own & un).sum()))
            d[f'{pl}_n_osc'] = n
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d = d[d.t_since_flash_ns.notna()].copy()
    d['ms'] = d.t_since_flash_ns / 1e6
    d['pos'], d['disp'] = d.p0_z, -d.p0_x / d.d_x * d.d_z          # D measures z (capsule_tof.VIEW)
    d['wall'] = d.wall_D.astype('boolean').fillna(False).astype(bool)
    d['osc'] = (d.x_n_osc + d.y_n_osc) > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        d['lb'] = np.log(d.x_q_sum / d.y_q_sum)
    late = d[(d.ms >= 20) & (np.minimum(d.x_q_sum, d.y_q_sum) >= 600)]
    u0, _, _ = crossing(late.pos.to_numpy(), late.disp.to_numpy())
    rng = np.random.default_rng(0)

    def exc(s):
        if len(s) < 30:
            return np.nan
        m = np.abs(s.pos + s.disp - u0)
        sh = np.abs(s.pos.to_numpy() + rng.permutation(s.disp.to_numpy()) - u0)
        return float((m < 15).mean() - (sh < 15).mean())
    out = []
    for lo, hi in ((1, 8), (8, 20), (20, 30), (30, 80)):
        for k, s in d[(d.ms >= lo) & (d.ms < hi)].groupby('osc'):
            out.append(dict(lo=lo, hi=hi, osc_in_window=bool(k), n=len(s), frac=len(s) / max(((d.ms >= lo) & (d.ms < hi)).sum(), 1),
                            pointing_excess=exc(s), wall=float(s.wall.mean()),
                            balance_mad=float(1.4826 * np.nanmedian(np.abs(s.lb - np.nanmedian(s.lb)))),
                            x_n_strips=float(s.x_n_strips.median())))
    T = pd.DataFrame(out)
    T.to_csv(OUT / 'd_osc_tracks.csv', index=False)
    print(T.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('find', 'tracks'))
    a = ap.parse_args()
    find() if a.step == 'find' else tracks()


if __name__ == '__main__':
    main()
