#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wft_beam.py -- run the waveform-first reconstruction (`wft/`) on a BEAM run.

PRELIMINARY implementation of PLAN_08 phases 1-2, deliberately minimal: enough
to produce a first run_79 track table that can be joined to the n_TOF
scintillators and checked end to end. It is NOT the calibrated production
chain -- see the caveats in `RUN79_PRELIM_*.md` and PLAN_08 §6 for what a real
in-situ calibration still owes.

Everything beam-specific lives here rather than in `wft/` so that the bench
package keeps its validated behaviour bit-for-bit (and so that concurrent work
on `wft/` does not collide with this).

Three pieces:

  BeamConfig          duck-types the `qa_config` config object that `wft.io`
                      and `wft.reco` consume (paths, FEUs, detector name).
  seeds_from_hits_beam  beam-appropriate seeding: the bench spark veto (50
                      hits/event) would throw away most of run_79, where the
                      median event has tens of hits per plane.
  reconstruct_subrun  per-file-tag streaming driver. The bench driver loads the
                      whole sub-run's hits in one DataFrame; here that is ~32 M
                      hits, so hits and waveforms are handled tag by tag.

Usage:
    python -m ntof_tracking.wft_beam bundle --det A
    python -m ntof_tracking.wft_beam reco   --det A --tags 4 --jobs 6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Pin BLAS to one thread per process, BEFORE numpy is imported (the thread pool
# is sized at import and inherited by the forked workers). The fit is already
# parallel over events, so a threaded BLAS inside each worker only oversubscribes:
# measured here, 6 workers x 8 OpenBLAS threads put 48 runnable threads on 8
# cores, drove the load average to ~39, and made the run SLOWER, not faster.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Overridable so the same driver runs on other hosts (e.g. the desktop, where
# the data lives under /media/ucla/x17match/) without editing this table.
BEAM_BASE = os.environ.get('WFT_BEAM_BASE',
                           '/media/dylan/data/x17/beam_july/runs/')
ANALYSIS_BASE = os.environ.get('WFT_BEAM_ANALYSIS',
                               '/media/dylan/data/x17/beam_july/analysis/wft/')
BENCH_ANALYSIS = os.environ.get('WFT_BENCH_ANALYSIS',
                                '/media/dylan/data/x17/cosmic_bench/Analysis/')

# beam arm -> (bench detector, FEU x, FEU y, bench bundle to seed from).
# Same table as mx_june_wft/bench/framing_compare.py, plus the bundle chosen in
# mx_june_wft/HANDOFF_2026-07-30.md's fleet status table.
BEAM_DETS = {
    'A': dict(bench='mx17_3', feu_x=3, feu_y=4,
              # r06 (2026-08-19): c2 slaved to 0.6*c1. calib_bundle_lp2 carried
              # c2/c1 = 1.14 -- an inverted ladder, physically impossible, since
              # the +-2 strip is reached only through the +-1.
              bundle=BENCH_ANALYSIS + 'mx17_det3_saturday_scan_6-27-26/'
                     'long_run_resist_490V_drift_1000V/mx17_3/wft/calib_bundle_r06',
              gap_mm=27.9),
    'B': dict(bench='mx17_2', feu_x=5, feu_y=6,
              # NB the bench key here is o22_long_det2 (`longer_run`), NOT
              # g_det2 (`long_run`) -- that collision is what hid det2's
              # inversion for two days. lp carried c2/c1 = 1.53.
              bundle=BENCH_ANALYSIS + 'mx17_det2_det3_overnight_6-22-26/'
                     'longer_run/mx17_2/wft/calib_bundle_r06',
              gap_mm=30.5),
    'C': dict(bench='mx17_6', feu_x=7, feu_y=8,
              # NOT moved to r06: det6's lp bundle is already physical
              # (c2/c1 = 0.82), so there is nothing to correct here.
              bundle=BENCH_ANALYSIS + 'mx17_det6_det7_overnight_6-26-26/'
                     'long_run/mx17_6/wft/calib_bundle_lp',
              gap_mm=30.0),
    'D': dict(bench='mx17_7', feu_x=1, feu_y=2,
              # calib_bundle_lp superseded det7's v36 in the 7-31 fleet rollout;
              # r06 supersedes lp in turn -- lp carried c2/c1 = 1.75, the worst
              # inversion in the fleet.
              bundle=BENCH_ANALYSIS + 'mx17_det6_det7_overnight_6-26-26/'
                     'long_run/mx17_7/wft/calib_bundle_r06',
              gap_mm=30.0),
}

# Drift velocity for run_79's Ar/iC4H10 90/10 at 233 V/cm, CERN 720.8 Torr.
# Magboltz clean-gas value from mx_july_beam_qa/DRIFT_WINDOW_ANALYSIS.md §1.
# THIS IS A PREDICTION, NOT A MEASUREMENT: the angle scale is proportional to
# 1/v, and B/C/D breathed ~0.8 % H2O on 07-19 which would cost them ~20 %.
# PLAN_08 §6.3 (column-endpoint + Magboltz) is how it gets measured.
V_DRIFT_MAGBOLTZ = 42.6
# per-arm prior: A was the dry one on 07-19, B/C/D were wet
V_DRIFT_PRIOR = {'A': 42.6, 'B': 34.0, 'C': 34.0, 'D': 36.0}

# run_79 DREAM: 12-bit ADC (hard rail 4095) on a ~330 ADC pedestal.
SAT_ADC_BEAM = 3700.0


# --------------------------------------------------------------------- config
@dataclass
class BeamConfig:
    """The subset of the qa_config interface that wft actually consumes."""
    KEY: str
    RUN: str
    SUB_RUN: str
    DET_NAME: str
    MX17_FEU_X: int
    MX17_FEU_Y: int
    BASE_PATH: str = BEAM_BASE
    MAP_CSV_PATH: str = os.path.join(REPO, 'mx17_m1_map.csv')
    OUT_BASE: str = ''
    file_tags: Optional[List[str]] = None      # None = all tags in the sub-run
    conditions: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.MX17_FEUS = [self.MX17_FEU_X, self.MX17_FEU_Y]
        if not self.OUT_BASE:
            self.OUT_BASE = os.path.join(ANALYSIS_BASE, self.RUN, self.SUB_RUN,
                                         self.DET_NAME)

    @property
    def run_config_path(self):
        return f'{self.BASE_PATH}{self.RUN}/run_config.json'

    @property
    def combined_hits_dir(self):
        return f'{self.BASE_PATH}{self.RUN}/{self.SUB_RUN}/combined_hits_root/'

    def out_dir(self, *parts):
        d = os.path.join(self.OUT_BASE, *parts)
        os.makedirs(d, exist_ok=True)
        return d


def beam_config(arm: str, run: str = 'run_79', sub_run: str = 'stat090_0000',
                n_tags: Optional[int] = None) -> BeamConfig:
    d = BEAM_DETS[arm]
    with open(f'{BEAM_BASE}{run}/run_config.json') as f:
        rc = json.load(f)
    daq = rc['dream_daq_info']
    cfg = BeamConfig(KEY=f'{run}_{sub_run}_mx17_{arm}', RUN=run, SUB_RUN=sub_run,
                     DET_NAME=f'mx17_{arm}',
                     MX17_FEU_X=d['feu_x'], MX17_FEU_Y=d['feu_y'],
                     conditions=dict(run=run, sub_run=sub_run,
                                     gas=rc.get('gas'),
                                     sample_ns=float(daq['sample_period']),
                                     n_samples=int(daq['n_samples_per_waveform']),
                                     latency=int(daq['latency'])))
    if n_tags:
        cfg.file_tags = subrun_tags(cfg)[:n_tags]
    return cfg


def subrun_tags(cfg: BeamConfig) -> List[str]:
    """The '<date>_<time>_<idx>' tags of the sub-run, in order."""
    from wft import io as wio
    fx = wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, cfg.MX17_FEU_X)
    return sorted({wio.file_tag(f) for f in fx})


def hits_file_for_tag(cfg: BeamConfig, tag: str) -> Optional[str]:
    hits = glob.glob(os.path.join(cfg.combined_hits_dir, f'*{tag}*.root'))
    return hits[0] if hits else None


# --------------------------------------------------------------------- bundle
def make_bundle(arm: str, out: Optional[str] = None, v_drift: Optional[float] = None,
                run: str = 'run_79', sub_run: str = 'stat090_0000') -> str:
    """Seed a run_79 bundle from the bench bundle.

    What transfers (PLAN_08 §2): the impulse template and the resistive-strip
    sharing kernel -- hardware properties of the same chambers. What does not:
    v_drift, the diffusion pair, and the DAQ constants. This routine transfers
    the first group verbatim, replaces the DAQ constants with run_79's, and
    puts a PRIOR (not a measurement) in v_drift; sigma_p0/Dp are left at the
    bench values, which is the largest un-validated assumption in this chain.
    """
    from wft.calib import CalibrationBundle
    d = BEAM_DETS[arm]
    cal = CalibrationBundle.load(d['bundle'])
    with open(f'{BEAM_BASE}{run}/run_config.json') as f:
        rc = json.load(f)
    daq = rc['dream_daq_info']
    cal.v_drift = float(v_drift if v_drift is not None
                        else V_DRIFT_PRIOR.get(arm, V_DRIFT_MAGBOLTZ))
    cal.sample_ns = float(daq['sample_period'])
    cal.sat_adc = SAT_ADC_BEAM
    # K x 60 ns must still span the full drift column: 18 x 60 = 1080 ns covers
    # the 20-sample (1200 ns) window. The bench ablation found shortening K to
    # match a shorter window makes things WORSE, so it is left alone.
    cal.n_depth_bins = 18
    cal.run_key = f'{run}/{sub_run}/mx17_{arm}'
    cal.conditions = dict(run=run, sub_run=sub_run, gas=rc.get('gas'),
                          n_samples=int(daq['n_samples_per_waveform']),
                          sample_ns=float(daq['sample_period']))
    cal.provenance = dict(cal.provenance)
    cal.provenance.update(
        seeded_from=d['bundle'],
        transferred='template + sharing kernel + w0/kw (bench)',
        replaced=f'v_drift -> {cal.v_drift} um/ns (PRIOR: Magboltz 90/10 '
                 f'@233 V/cm / 07-19 water estimate, NOT measured in situ), '
                 f'sat_adc -> {SAT_ADC_BEAM}, sample_ns/conditions -> run_79',
        not_revalidated='sigma_p0, Dp (gas-dependent), template (assumed same '
                        'DREAM shaping), dt_xy',
        status='PRELIMINARY -- see ntof_tracking/TRACK_PLAN_08 phase 2')
    out = out or os.path.join(ANALYSIS_BASE, run, sub_run, f'mx17_{arm}',
                              'calib_bundle_prelim')
    cal.save(out, note=f'preliminary run_79 bundle for arm {arm}')
    print(cal.summary())
    print('wrote', out)
    return out


# --------------------------------------------------------------------- seeding
# Beam seeding constants. The bench numbers (SPARK_VETO_HITS = 50, 3
# candidates) were tuned on cosmics with ~6 strips in one cluster; run_79's
# median event has 19-44 hits per plane and its p99 is the full plane.
BUSY_PLANE_HITS = 150      # after the significance floor: flash/discharge veto
MIN_STRIPS_BEAM = 5        # a column, not a single-strip deposit
N_CANDIDATES_BEAM = 5
WIDE_STRIPS = 10           # >= this many strips AND
ISOCHRONOUS_SPAN = 3       #   < this many samples of time span = not a column
                           #   (det D's X plane is full of these -- see
                           #    mx_june_wft/WINDOW_ABLATION_2026-07-30.md §1)


def read_hits_tag(path: str, feus) -> 'pd.DataFrame':
    """One hits file, the two FEUs of one detector.

    Keeps the LARGEST pulse per (event, channel): at beam a channel can carry
    several hits per event (pileup, and ringing after a saturated event), and a
    later secondary would otherwise be read as the column's deep edge.
    """
    import pandas as pd
    import uproot
    df = uproot.open(path)['hits'].arrays(
        ['eventId', 'feu', 'channel', 'amplitude', 'significance', 'max_sample'],
        library='pd')
    df = df[df['feu'].isin(list(feus))]
    df = df.sort_values('amplitude').drop_duplicates(
        ['eventId', 'feu', 'channel'], keep='last')
    return df


def seeds_from_hits_beam(df, pos_maps, feu_x, feu_y,
                         busy=BUSY_PLANE_HITS,
                         n_candidates=N_CANDIDATES_BEAM,
                         min_strips=MIN_STRIPS_BEAM) -> Dict[int, dict]:
    """Beam seeds: {eventId: {'x': [Seed], 'y': [Seed], 'n_hits', 'spark'}}.

    Same contract and the same clustering as `wft.seed.seeds_from_hits` -- only
    channels ever cross the boundary, never hit times as geometry. What differs
    is the event/cluster admission policy:

      * the veto is PER PLANE and much looser (a busy plane is dropped, the
        event's other plane is kept);
      * more candidate clusters are offered to the fit, because multi-cluster
        events are the norm here rather than the exception;
      * clusters that are wide but isochronous are dropped. Using the hit-time
        SPAN to decide that a deposit is not a drift column is a detection
        question (which clusters to fit), not a geometry one -- the fit still
        gets only channels.
    """
    from wft import seed as wseed
    df = wseed.apply_significance_floor(df, wseed.SIG_REL_FLOOR)
    out: Dict[int, dict] = {}
    if len(df) == 0:
        return out
    for eid, g in df.groupby('eventId', sort=False):
        rec = {'x': [], 'y': [], 'n_hits': int(len(g)), 'spark': False}
        for plane, feu in (('x', feu_x), ('y', feu_y)):
            gp = g[g['feu'] == feu]
            if len(gp) == 0 or len(gp) > busy:
                continue
            ch = gp['channel'].to_numpy().astype(int)
            cands = wseed.seed_candidates(pos_maps[feu][ch], ch,
                                          gp['amplitude'].to_numpy(),
                                          min_strips=min_strips,
                                          n_candidates=n_candidates)
            smp = dict(zip(ch, gp['max_sample'].to_numpy()))
            keep = []
            for s in cands:
                m = np.array([smp.get(int(c), np.nan) for c in s.channels],
                             dtype=float)
                m = m[np.isfinite(m)]
                span = (m.max() - m.min()) if len(m) else 0.0
                if s.n_strips >= WIDE_STRIPS and span < ISOCHRONOUS_SPAN:
                    continue
                keep.append(s)
            rec[plane] = keep
        if rec['x'] or rec['y']:
            out[int(eid)] = rec
    return out


# --------------------------------------------------------------------- driver
def reconstruct_subrun(cfg: BeamConfig, bundle_path: str, out_path: str,
                       jobs: int = 6, limit_per_tag: Optional[int] = None,
                       pad_strips: int = 3, verbose: bool = True):
    """Reconstruct a beam sub-run tag by tag into one parquet table.

    Mirrors `wft.reco.reconstruct_run` (and reuses its worker), but streams the
    hits per file tag instead of concatenating the whole sub-run.
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    from wft import io as wio
    from wft import reco as wreco
    from wft.calib import CalibrationBundle

    cal = CalibrationBundle.load(bundle_path)
    bad = cal.check_conditions(cfg.conditions)
    if bad and verbose:
        print('[wft-beam] bundle/run condition mismatches: ' + '; '.join(bad))
    pos_maps = wio.strip_position_map(cfg)
    feu_x, feu_y = cfg.MX17_FEU_X, cfg.MX17_FEU_Y
    tags = cfg.file_tags or subrun_tags(cfg)
    if verbose:
        print(f'[wft-beam] {cal.summary()}')
        print(f'[wft-beam] {cfg.DET_NAME} {cfg.RUN}/{cfg.SUB_RUN}: '
              f'{len(tags)} file tag(s)', flush=True)

    def _write(rows, path, tags_done=()):
        d = pd.DataFrame(rows)
        if len(d):
            d = d.sort_values('event_id').reset_index(drop=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d.to_parquet(path, index=False)
        if cand_rows:
            pd.DataFrame(cand_rows).sort_values(
                ['event_id', 'plane', 'rank']).reset_index(drop=True).to_parquet(
                path.replace('.parquet', '.candidates.parquet'), index=False)
        # the sidecar goes with EVERY checkpoint: a table without its bundle
        # metadata is unusable downstream (v_drift lives there, and it is the
        # angle scale), and a half-finished run is exactly when that bites
        _write_meta(d, path, cal, cfg, bundle_path, feu_x, feu_y, tags_done,
                    n_seeded, pad_strips, partial=len(tags_done) < len(tags))
        return d

    rows, cand_rows, n_seeded, done = [], [], 0, []
    with ProcessPoolExecutor(max_workers=jobs, initializer=wreco._worker_init,
                             initargs=(bundle_path,)) as pool:
        for tag in tags:
            hp = hits_file_for_tag(cfg, tag)
            if hp is None:
                print(f'[wft-beam]   {tag}: no hits file, skipped')
                continue
            hits = read_hits_tag(hp, (feu_x, feu_y))
            seeds = seeds_from_hits_beam(hits, pos_maps, feu_x, feu_y)
            del hits
            wanted = set(seeds)
            if limit_per_tag:
                wanted = set(sorted(wanted)[:limit_per_tag])
            n_seeded += len(wanted)
            if verbose:
                print(f'[wft-beam]   {tag}: {len(wanted):,} seeded events',
                      flush=True)
            payloads = _windows_for_tag(cfg, tag, pos_maps, seeds, wanted,
                                        pad_strips)
            if not payloads:
                continue
            for r in pool.map(wreco._worker_fit, payloads, chunksize=8):
                cand_rows.extend(r.pop('_cand', []))
                rows.append(r)
            done.append(tag)
            if verbose:
                print(f'[wft-beam]   {tag}: {len(payloads):,} windowed, '
                      f'{len(rows):,} events reconstructed total', flush=True)
            # checkpoint after every tag: this machine is shared and a beam
            # sub-run is hours of CPU, so a table that only exists at the end
            # is a table that gets thrown away
            _write(rows, out_path, done)

    df = _write(rows, out_path, done)
    if verbose:
        print(f'[wft-beam] wrote {out_path} ({len(df):,} events)')
    return df


def _write_meta(df, out_path, cal, cfg, bundle_path, feu_x, feu_y, tags_done,
                n_seeded, pad_strips, partial=False):
    meta = dict(n_events=int(len(df)), n_seeded=int(n_seeded),
                status='PRELIMINARY' + (' (PARTIAL: run still going)'
                                        if partial else ''),
                partial=bool(partial), tags_done=list(tags_done),
                calibration=bundle_path,
                bundle=dict(detector=cal.detector, run_key=cal.run_key,
                            v_drift=cal.v_drift, hyper=cal.hyper,
                            sat_adc=cal.sat_adc, n_depth_bins=cal.n_depth_bins,
                            conditions=cal.conditions,
                            provenance=cal.provenance),
                # Same stamp as wft.reco.reconstruct_run: whether the fit
                # applied the per-plane angle constants in-reco. Without it a
                # post-hoc corrector cannot tell this table from a frozen-gen
                # one and would double-apply. applied=False -> the angles are
                # on the uncorrected mapping and need the post-hoc pass.
                angle_constants=dict(applied=bool(cal.w0 or cal.kw),
                                     w0=dict(cal.w0), kw=dict(cal.kw)),
                run=dict(key=cfg.KEY, run=cfg.RUN, sub_run=cfg.SUB_RUN,
                         detector=cfg.DET_NAME, feu_x=feu_x, feu_y=feu_y,
                         file_tags=list(cfg.file_tags or [])),
                selection=dict(seeder='ntof_tracking.wft_beam',
                               busy_plane_hits=BUSY_PLANE_HITS,
                               min_strips=MIN_STRIPS_BEAM,
                               n_candidates=N_CANDIDATES_BEAM,
                               wide_strips=WIDE_STRIPS,
                               isochronous_span=ISOCHRONOUS_SPAN,
                               pad_strips=pad_strips),
                reco_config=dict(model_frac=float(os.environ.get('WFT_MODEL_FRAC', 0)),
                                 prescan=os.environ.get('WFT_PRESCAN', '0'),
                                 pair_select=os.environ.get('WFT_PAIR_SELECT', '0'),
                                 chi2dof_bad=os.environ.get('WFT_CHI2DOF_BAD', '300')))
    with open(out_path.replace('.parquet', '.meta.json'), 'w') as f:
        json.dump(meta, f, indent=1, default=str)


def _windows_for_tag(cfg, tag, pos_maps, seeds, wanted, pad_strips):
    """Waveform windows for one file tag -> the fit worker's payloads."""
    from wft import io as wio
    fx = [f for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN,
                                      cfg.MX17_FEU_X) if wio.file_tag(f) == tag]
    fy = [f for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN,
                                      cfg.MX17_FEU_Y) if wio.file_tag(f) == tag]
    if not fx or not fy:
        return []
    buf = {}
    for plane, path, feu in (('x', fx[0], cfg.MX17_FEU_X),
                             ('y', fy[0], cfg.MX17_FEU_Y)):
        rdr = wio.FeuReader(path)
        want = wanted & set(rdr.event_ids.tolist())
        for eid, ftst, wfm in rdr.iter_events(want):
            cl = seeds[eid][plane]
            if not cl:
                continue
            wins, used = [], []
            for s in cl:
                win = wio.extract_window(wfm, rdr.noise, pos_maps[feu],
                                         s.channels, pad_strips)
                if win is None:
                    continue
                wins.append(dict(W=win.W, pos=win.pos, noise=win.noise,
                                 ch=win.ch))
                used.append(s)
            if not wins:
                continue
            rec = buf.setdefault(eid, {'w': {}, 's': {}})
            rec['w'][plane] = wins
            rec['s'][plane] = used
            rec['ftst_' + plane] = ftst
    payloads = []
    for eid, rec in buf.items():
        # per-plane ftst dict — wft.reco._worker_fit's contract since the
        # 2026-08-11 t0-prior work (it derives ftst_diff itself; the old
        # scalar here made every event with ftst_x != ftst_y fit as None)
        ftst = {p: rec.get('ftst_' + p) for p in ('x', 'y')}
        payloads.append((eid, rec['w'], rec['s'], seeds[eid]['n_hits'],
                         seeds[eid]['spark'], ftst))
    return payloads


# ------------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('bundle', help='seed a run_79 bundle from the bench one')
    b.add_argument('--det', default='A')
    b.add_argument('--run', default='run_79')
    b.add_argument('--subrun', default='stat090_0000')
    b.add_argument('--v-drift', type=float, default=None)
    b.add_argument('--out', default=None)

    r = sub.add_parser('reco', help='reconstruct a beam sub-run')
    r.add_argument('--det', default='A')
    r.add_argument('--run', default='run_79')
    r.add_argument('--subrun', default='stat090_0000')
    r.add_argument('--tags', type=int, default=None,
                   help='use only the first N file tags (13 per sub-run)')
    r.add_argument('--jobs', type=int, default=6)
    r.add_argument('--limit-per-tag', type=int, default=None)
    r.add_argument('--bundle', default=None)
    r.add_argument('--out', default=None)

    a = ap.parse_args()
    if a.cmd == 'bundle':
        make_bundle(a.det, out=a.out, v_drift=a.v_drift, run=a.run,
                    sub_run=a.subrun)
        return 0
    cfg = beam_config(a.det, a.run, a.subrun, n_tags=a.tags)
    bundle = a.bundle or os.path.join(ANALYSIS_BASE, a.run, a.subrun,
                                      f'mx17_{a.det}', 'calib_bundle_prelim')
    if not os.path.isdir(bundle):
        bundle = make_bundle(a.det, run=a.run, sub_run=a.subrun)
    out = a.out or cfg.out_dir() + '/events_prelim.parquet'
    reconstruct_subrun(cfg, bundle, out, jobs=a.jobs,
                       limit_per_tag=a.limit_per_tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
