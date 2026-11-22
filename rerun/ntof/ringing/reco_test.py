#!/usr/bin/env python3
"""reco_test.py -- re-run the pass-1 waveform reconstruction on run_145 with the ringing correction (README.md).

    python -m rerun.ntof.ringing.reco_test --arm A --mode deflect_hits --tags 260805_14H06_000,260805_14H06_001
                                            [--subrun stat090_0000] [--late-frac 0.3] [--jobs 8]

Modes:
  pass1         the pass-1 configuration, unchanged (checks that this driver reproduces the stage-2 tables)
  deflect_hits  waveforms corrected (deflect.py, gated on ringing planes); the analyzer's hits still seed, but in a
                corrected plane a hit whose corrected waveform stays below 5 sigma (peak, or 3-sample running sum) is
                dropped, so a ringing lobe no longer seeds a window
  deflect_wave  waveforms corrected; seeds from the corrected waveforms (wft_beam.hits_from_waveforms)

Events: every trigger with 1 <= ToF < 20 ms, plus a random LATE_FRAC of the triggers at >= 30 ms (the unchanged
check).  Bundles: $WORK_DATA/ntof_pass1/bundles/pf/<arm>_E2_beam (the ledger's choice for run_145 A and C).
Output: $WORK_DATA/ntof_pass1/ana/ringing/reco/<mode>/run_145_<sub>_<arm>.parquet (+ .candidates.parquet, .meta.json).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'rerun' / 'ntof'))
WD = os.environ.get('WORK_DATA', '/media/ucla/work')
FEUS = dict(A=(3, 4), C=(7, 8), D=(1, 2))
WF_ROOTS = [f'{WD}/ntof_pass1/runs/', f'{WD}/claude-jobs/bcad1a92/wf/']
TRIGTIME = '/media/dylan/data/x17/sept26_prelim/trigtime'


def events(run, sub, late_frac, seed=1, lo=1.0, hi=20.0):
    import pandas as pd
    t = pd.read_parquet(f'{TRIGTIME}/trigtime_{run}_{sub}.parquet')
    ms = t.t_pred_ns / 1e6
    early = t[(ms >= lo) & (ms < hi) & (t.is_flash == 0)].eventId
    late = t[(ms >= 30) & (t.is_flash == 0)].eventId
    late = late.sample(frac=late_frac, random_state=seed)
    return set(early.astype(int)) | set(late.astype(int))


def hits_filter(wb, cfg, nsig=5.0):
    """Wrap wft_beam.read_hits_tag: in a plane the corrected reader changed, drop hits whose corrected waveform
    stays below nsig (peak or 3-sample running sum, as hits_from_waveforms)."""
    import pandas as pd
    from wft import io as wio
    from rerun.ntof.ringing import deflect as DF
    orig = wb.read_hits_tag

    def read(path, feus):
        h = orig(path, feus)
        tag = wio.file_tag(path.replace('_feu-combined_hits', '_00'))
        keep = np.ones(len(h), bool)
        for feu in feus:
            fs = [f for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu) if wio.file_tag(f) == tag]
            if not fs:
                continue
            rdr = wio.FeuReader(fs[0])
            nz = np.maximum(rdr.noise.astype(np.float64), 1e-3)
            hf = h[h.feu == feu]
            want = set(hf.eventId.astype(int)) & set(cfg._wanted)
            drop = []
            for eid, _f, W in rdr.iter_events(want):
                if not getattr(rdr, '_last_corrected', False):
                    continue
                c = np.cumsum(np.pad(W, ((0, 0), (1, 0))), axis=1)
                rs = (c[:, 3:] - c[:, :-3]).max(axis=1) / (nz * np.sqrt(3) * 1.3)
                sig = np.maximum(W.max(axis=1) / nz, rs)
                drop.append(pd.DataFrame(dict(eventId=eid, channel=np.flatnonzero(sig < nsig))))
            if drop:
                d = pd.concat(drop)
                key = set(zip(d.eventId.astype(int), d.channel.astype(int)))
                m = (h.feu == feu).to_numpy()
                kk = np.array([(int(e), int(c)) in key for e, c in zip(h.eventId[m], h.channel[m])])
                keep[np.flatnonzero(m)[kk]] = False
        cfg._hits_dropped = getattr(cfg, '_hits_dropped', 0) + int((~keep).sum())
        return h[keep].reset_index(drop=True)
    wb.read_hits_tag = read


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--arm', required=True, choices='ACD')
    ap.add_argument('--mode', required=True, choices=('pass1', 'deflect_hits', 'deflect_wave'))
    ap.add_argument('--tags', default=None)
    ap.add_argument('--late-frac', type=float, default=0.3)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--tof', default='1,20', help='early ToF window [ms]')
    ap.add_argument('--label', default='', help='appended to the output directory name')
    a = ap.parse_args()
    import csv
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    reg = [r for r in csv.DictReader(open(REPO / 'rerun' / 'ntof' / 'ntof_registry.csv'))
           if f"run_{r['run']}" == a.run and r['subrun'] == a.subrun and r['excluded'] != 'True']
    base = next(b for b in WF_ROOTS if os.path.isdir(f'{b}{a.run}/{a.subrun}/decoded_root'))
    tags = sorted(r['tag'] for r in reg)
    if a.tags:
        tags = [t for t in tags if t in a.tags.split(',')]
    r0 = reg[0]
    fx, fy = FEUS[a.arm]
    out = Path(WD) / 'ntof_pass1' / 'ana' / 'ringing' / 'reco' / (a.mode + a.label) / f'{a.run}_{a.subrun}_{a.arm}.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    hits_base = f'{WD}/ntof_pass1/runs/'          # the staged pass-1 hits (combined_hits_root)
    cfg = wb.BeamConfig(KEY=f'{a.run}_{a.subrun}_mx17_{a.arm}', RUN=a.run, SUB_RUN=a.subrun, DET_NAME=f'mx17_{a.arm}',
                        MX17_FEU_X=fx, MX17_FEU_Y=fy, BASE_PATH=base, OUT_BASE=str(out.parent),
                        HITS_DIR='combined_hits_root',
                        conditions=dict(gas=r0['gas'], sample_ns=float(r0['sample_period_ns']),
                                        n_samples=int(float(r0['n_samples']))))
    cfg.file_tags = tags
    if base != hits_base:                           # hits live with the staged copy, waveforms maybe elsewhere
        hd = f'{hits_base}{a.run}/{a.subrun}/combined_hits_root/'
        if os.path.isdir(hd):
            type(cfg).combined_hits_dir = property(lambda self: hd)
    lo, hi = (float(v) for v in a.tof.split(','))
    wanted = events(a.run, a.subrun, a.late_frac, lo=lo, hi=hi)
    cfg._wanted = wanted
    stats = {}
    if a.mode != 'pass1':
        from rerun.ntof.ringing import deflect as DF
        stats = DF.install('gated')
        # mark the reader when the last event was corrected (for the hits filter)
        it = wio.FeuReader.iter_events

        def it2(self, wanted=None, with_raw=False):
            for x in it(self, wanted, with_raw):
                n0 = stats.get('corrected', 0)
                self._last_corrected = n0 > getattr(self, '_n_corr_seen', 0)
                self._n_corr_seen = n0
                yield x
        wio.FeuReader.iter_events = it2
        if a.mode == 'deflect_hits':
            hits_filter(wb, cfg)
    bundle = f'{WD}/ntof_pass1/bundles/pf/{a.arm}_E2_beam'
    allow = {t: wanted for t in tags}
    df = wb.reconstruct_subrun(cfg, bundle, str(out), jobs=a.jobs, allow_events=allow,
                               allow_meta=dict(source='reco_test ToF window'), allow_bow=True, min_strips=3,
                               extra_meta=dict(ringing_mode=a.mode, ringing_stats=stats, tags=tags),
                               seed_source='waveform' if a.mode == 'deflect_wave' else 'hits')
    print(f'{a.mode} {a.arm}: {len(df)} events; corrected planes {stats}; hits dropped {getattr(cfg, "_hits_dropped", 0)}')


if __name__ == '__main__':
    main()
