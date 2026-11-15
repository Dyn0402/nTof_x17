#!/usr/bin/env python3
"""d104_latency.py -- which D latency (L1 or L2) each acquisition of run_104 was taken with.

D's trigger latency stepped by ~35-40 ns between run_103 (07-29) and run_105 (07-30) (plan §6 11-06; registry
`lat_epoch`).  run_104 (07-29 19:22 -> 07-30 11:29, 198 acquisitions) straddles it, so pass 1 reconstructed its D arm
twice: Analysis_pass1/side/run104_D_L1 (t0 pinned on the L1 latency, x -17.8 / y -11.2 ns at ftst 0) and run104_D_L2
(the r149 pin, x 18.3 / y 20.0 ns).  Both pins are soft (t0_prior_sigma 15 ns), so the same events were fitted under
two hypotheses that differ only in the latency.  Per acquisition, three independent readings:

  dchi2   median over planes fitted ok under both pins of (chi2_L1 - chi2_L2)/dof, with each pin's own prior term
          ((t0 - pin)/15)^2 removed so only the waveform chi2 is compared.  < 0 means the waveforms prefer L1.
  pull    the latency the data pull toward: median of (t0 + 10 ftst) under each pin, single quality tracks.  With a
          15 ns soft pin a wrong pin is pulled toward the truth, so pull_L1 - pin_L1 > 0 and pull_L2 - pin_L2 < 0
          both say "the truth lies between, nearer the other".
  point   capsule pointing (capsule_point.score: gated tracks with DCA to the capsule axis < 15 mm) per pin.

    d104_latency.py [--data DIR]     # DIR holds run104_D_L1/ and run104_D_L2/ (default $WORK_DATA/ntof_pass1/run104)
    d104_latency.py --l1 control_L1 --l2 control_L2 --name control   # any pair of trees run_<r>/<sub>/D/*_D.parquet

Writes DIR/results/d104_by_acq.csv (or <name>_by_acq.csv) and prints a summary with the change point.  run configs
are read from DIR/rc/run_<r>.json.

The control (condor 4673558, 2026-11-15): 24 L2-era beam acquisitions (runs 106, 108) reconstructed under the L1 pin
(EOS Analysis_pass1/tests/d104_control_L1), against the main pass's L2-pin outputs of the same acquisitions.  If L1
also wins there, its edge on run_104 is a property of the pin on beam data, not evidence of run_104's epoch.
"""
import argparse, glob, os, sys, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [NT, os.path.abspath(os.path.join(NT, '..', '..'))]
warnings.filterwarnings('ignore')

PIN = {'L1': {'x': -17.8, 'y': -11.2}, 'L2': {'x': 18.3, 'y': 20.0}}   # t0_abs at ftst 0 of D_E2_beam(_L1)
SIG = 15.0


def per_acq(dfs, pts):
    a, b = dfs['L1'], dfs['L2']
    out = {}
    m = a.merge(b, on='event_id', suffixes=('_1', '_2'))
    for p in 'xy':
        for L, s in (('L1', '_1'), ('L2', '_2')):
            t0, ft = m[f'{p}_t0{s}'], m[f'{p}_ftst{s}']
            m[f'{p}_cd{s}'] = m[f'{p}_chi2{s}'] - ((t0 - (PIN[L][p] - 10 * ft)) / SIG) ** 2
        ok = (m[f'{p}_ok_1'] == True) & (m[f'{p}_ok_2'] == True) & (m[f'{p}_ftst_1'] >= 0)
        mm = m[ok]
        d = (mm[f'{p}_cd_1'] - mm[f'{p}_cd_2']) / mm[f'{p}_dof_1'].clip(lower=1)
        out[f'dchi2_{p}'] = float(np.median(d)) if len(d) else np.nan
        out[f'frac_L1_better_{p}'] = float((d < 0).mean()) if len(d) else np.nan
        out[f'n_{p}'] = int(len(d))
        for L, df in dfs.items():
            s = df[(df[f'{p}_ok'] == True) & (df[f'{p}_quality_ok'] == True) & (df.n_tracks == 1) & (df[f'{p}_ftst'] >= 0)]
            v = (s[f'{p}_t0'] + 10 * s[f'{p}_ftst']).to_numpy(float)
            v = v[np.abs(v - np.median(v)) < 200] if len(v) else v
            out[f'pull_{L}_{p}'] = float(np.median(v) - PIN[L][p]) if len(v) >= 10 else np.nan
    for L, s in pts.items():
        out[f'n_point15_{L}'] = s['n_point_15']
        out[f'frac_point15_{L}'] = s['frac_point_15']
        out[f'n_tracks_{L}'] = s['n_tracks']
    out['n_events'] = int(len(a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.environ.get('WORK_DATA', os.path.expanduser('~/work_data')),
                                                   'ntof_pass1', 'run104'))
    ap.add_argument('--l1', default='run104_D_L1')
    ap.add_argument('--l2', default='run104_D_L2')
    ap.add_argument('--name', default='d104')
    a = ap.parse_args()
    import capsule_point
    rows = []
    for f1 in sorted(glob.glob(f'{a.data}/{a.l1}/run_*/*/D/*_D.parquet')):
        f2 = f1.replace(f'{a.data}/{a.l1}/', f'{a.data}/{a.l2}/')
        sub, name = f1.split('/')[-3], os.path.basename(f1)
        rc = f'{a.data}/rc/{f1.split("/")[-4]}.json'
        tag = name.split(f'{sub}_')[1][:-len('_D.parquet')]
        if not os.path.exists(f2):
            print('no L2 for', f1); continue
        dfs = {'L1': pd.read_parquet(f1), 'L2': pd.read_parquet(f2)}
        pts = {}
        for L, f in (('L1', f1), ('L2', f2)):
            try:
                pts[L], _ = capsule_point.score(f, 'D', rc, L)
            except Exception as e:   # an acquisition with no gated track
                pts[L] = dict(n_point_15=0, frac_point_15=np.nan, n_tracks=0)
        r = dict(run=int(f1.split('/')[-4][4:]), subrun=sub, tag=tag, time=pd.to_datetime('20' + tag[:6] + tag[7:9] + tag[10:12], format='%Y%m%d%H%M'),
                 **per_acq(dfs, pts))
        rows.append(r)
    T = pd.DataFrame(rows).sort_values(['time', 'subrun', 'tag']).reset_index(drop=True)
    T['dchi2'] = T[['dchi2_x', 'dchi2_y']].mean(axis=1)
    T['dpoint'] = T.n_point15_L1 - T.n_point15_L2
    os.makedirs(f'{a.data}/results', exist_ok=True)
    T.to_csv(f'{a.data}/results/{a.name}_by_acq.csv', index=False)
    # single change point: L1 up to k, L2 after, maximising the total chi2 preference
    s = np.nan_to_num(T.dchi2.to_numpy() * T[['n_x', 'n_y']].sum(axis=1).to_numpy())
    gain = [(-s[:k]).sum() + s[k:].sum() for k in range(len(T) + 1)]
    k = int(np.argmax(gain))
    T['vote_chi2'] = np.where(T.dchi2 < 0, 'L1', 'L2')
    pd.set_option('display.width', 250, 'display.max_rows', 400)
    cols = ['time', 'subrun', 'tag', 'n_events', 'dchi2_x', 'dchi2_y', 'frac_L1_better_x', 'pull_L1_x', 'pull_L2_x',
            'pull_L1_y', 'pull_L2_y', 'n_point15_L1', 'n_point15_L2', 'vote_chi2']
    print(T[cols].to_string(float_format=lambda x: f'{x:.2f}'))
    print(f'\nchange point (chi2, one step): first {k} acquisitions L1, last {len(T) - k} L2; '
          f'between {T.time.iloc[k-1] if k else "-"} {T.tag.iloc[k-1] if k else ""} and '
          f'{T.time.iloc[k] if k < len(T) else "-"} {T.tag.iloc[k] if k < len(T) else ""}')
    for name, sl in (('before', T.iloc[:k]), ('after', T.iloc[k:])):
        if len(sl):
            print(f'{name}: {len(sl)} acq, chi2 votes L1 {int((sl.dchi2 < 0).sum())} / L2 {int((sl.dchi2 >= 0).sum())}; '
                  f'point<15 L1 {sl.n_point15_L1.sum()} L2 {sl.n_point15_L2.sum()}; '
                  f'pull_L1 x {sl.pull_L1_x.median():+.1f} y {sl.pull_L1_y.median():+.1f}, '
                  f'pull_L2 x {sl.pull_L2_x.median():+.1f} y {sl.pull_L2_y.median():+.1f} ns')


if __name__ == '__main__':
    main()
