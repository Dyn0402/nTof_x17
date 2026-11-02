#!/usr/bin/env python3
"""make_ntof_registry.py -- P0 of NTOF_PASS_PLAN_2026-11-02: one row per n_TOF acquisition in the pass.

Scope (Dylan 11-02, Q1 "minimum for now"): the 36 production runs (stage-0 sample, in_sample) + every beam-off
cosmic sub-run at the production point (ntof_cosmics inventory, era != pre_production).  Pre-production goes in a
last pass later; the builder takes --scope all for it.

Inputs
  stage-0 sample      /media/dylan/data/x17/sept26_prelim/stage0/sample.csv   (freeze_sample.py; runs by TITLE)
  cosmic inventory    ntof_cosmics/results/cosmic_subruns.csv                   (inventory.py)
  run configs         $NTOF_RUNS/run_<n>/run_config.json  (default /media/dylan/data/x17/beam_july/runs)
  EOS manifest        rerun/ntof/eos_manifest.csv.gz  (run, subrun, kind dec|fdf|hits, file, bytes), listed on
                      lxplus 2026-11-02 from /eos/experiment/ntof/data/x17/july_beam/runs

Output  rerun/ntof/ntof_registry.csv  -- one row per (run, subrun, tag); tag = <yymmdd>_<HHhMM>_<idx> of the
        datrun files.  Every per-row number the n_TOF pass needs comes from here (no env switches).

Condition columns (CLAUDE.md "Two n_TOF conditions", ntof_pedestal_qa/README.md, sept26 DATASET.md):
  noise_era     quiet (run <= 67) / unplaced (run 68) / noisy (run >= 69): RdClk_Div 6 -> 4 on 23 July
  access_epoch  pre_access_27jul / post_access_27jul (access 27 July 11:23-14:11; boundary taken at 14:11)
  a_cm_x3       chamber A common mode x3.3, 21 July -> access
  a_x_conn8_dead  FEU 3 ch 448-511 disconnected, 22 July -> access (every sub-run of run_79)
"""
import argparse
import gzip
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SAMPLE = Path('/media/dylan/data/x17/sept26_prelim/stage0/sample.csv')
COSMIC = REPO / 'ntof_cosmics/results/cosmic_subruns.csv'
RUNS = Path(os.environ.get('NTOF_RUNS', '/media/dylan/data/x17/beam_july/runs'))
MANIFEST = HERE / 'eos_manifest.csv.gz'
OUT = HERE / 'ntof_registry.csv'
EOS_RUNS = '/eos/experiment/ntof/data/x17/july_beam/runs'

ACCESS_END = datetime(2026, 7, 27, 14, 11)     # local (CEST) wall time, as the file tags
A_CM_START = datetime(2026, 7, 21, 0, 0)       # day resolution (pedestal history)
A_CONN8_START = datetime(2026, 7, 22, 0, 0)    # day resolution (pedestal history)
CHAMBERS = 'ABCD'
BENCH = dict(A='det3', B='det2', C='det6', D='det7')
TAG_RE = re.compile(r'datrun_(\d{6})_(\d{2})H(\d{2})_(\d{3})_(\d{2})')


def tag_time(tag):
    d, hh, mm, _ = tag.split('_')[0], *tag.split('_')[1].split('H'), None
    return datetime.strptime(tag.split('_')[0] + tag.split('_')[1], '%y%m%d%HH%M')


def load_manifest():
    rows = []
    with gzip.open(MANIFEST, 'rt') as f:
        for line in f:
            p = line.split()
            if p[0] == 'run':
                continue
            rows.append(dict(run=int(p[0]), subrun=p[1], kind=p[2], file=p[3],
                             bytes=int(p[4]) if len(p) > 4 else 0))
    m = pd.DataFrame(rows)
    t = m.file.str.extract(TAG_RE)
    m['tag'] = t[0] + '_' + t[1] + 'H' + t[2] + '_' + t[3]
    m['feu'] = pd.to_numeric(t[4], errors='coerce')
    m.loc[m.kind == 'hits', 'tag'] = m.loc[m.kind == 'hits', 'file'].str.extract(
        r'datrun_(\d{6}_\d{2}H\d{2}_\d{3})_feu')[0]
    return m[m.tag.notna()]


def run_config(run):
    p = RUNS / f'run_{run}' / 'run_config.json'
    return json.load(open(p)) if p.exists() else None


def subrun_conf(cfg, subrun):
    """Per-sub-run readout + HV: sub_runs entry overrides the dream_daq_info defaults."""
    daq = cfg['dream_daq_info']
    sr = next((s for s in cfg.get('sub_runs', []) if s.get('sub_run_name') == subrun), {})
    out = dict(latency=sr.get('latency', daq.get('latency')),
               n_samples=sr.get('n_samples_per_waveform', daq.get('n_samples_per_waveform')),
               sample_period_ns=daq.get('sample_period'), zero_suppress=bool(daq.get('zero_suppress')),
               inter_packet_delay=sr.get('inter_packet_delay', daq.get('inter_packet_delay')),
               gas=cfg.get('gas'), title=(cfg.get('trigger') or '').split('.')[0][:160])
    hvs = sr.get('hvs', {})
    dets = {d['name']: d for d in cfg.get('detectors', [])}
    for ch in CHAMBERS:
        d = dets.get(f'mx17_{ch}', {})
        hc = d.get('hv_channels', {})
        for kind in ('resist', 'drift'):
            slot = hc.get(kind)
            v = hvs.get(str(slot[0]), {}).get(str(slot[1])) if slot else None
            out[f'{kind}_{ch}'] = v
        f = d.get('dream_feus', {})
        out[f'feu_x_{ch}'] = f.get('x_1', [None])[0]
        out[f'feu_y_{ch}'] = f.get('y_1', [None])[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', choices=['pass1', 'all'], default='pass1')
    a = ap.parse_args()

    s = pd.read_csv(SAMPLE)
    prod = s[s.in_sample].assign(rclass='production')
    prod = prod[['run', 'subrun', 'rclass', 'ntof_run', 'match_status', 'cut']]
    c = pd.read_csv(COSMIC)
    if a.scope == 'pass1':
        c = c[c.era != 'pre_production']
    cos = c.assign(rclass='cosmic')[['run', 'subrun', 'rclass', 'clean_beam_off', 'beam_state', 'ntof_runs']]
    scope = pd.concat([prod, cos], ignore_index=True)
    if a.scope == 'all':
        rest = s[~s.in_sample & (s['mode'] == 'beam') & s.subrun.notna()].assign(rclass='pre_production')
        scope = pd.concat([scope, rest[['run', 'subrun', 'rclass', 'ntof_run', 'match_status', 'cut']]],
                          ignore_index=True)

    m = load_manifest()
    rows = []
    cfgs = {}
    for r in scope.itertuples(index=False):
        if r.run not in cfgs:
            cfgs[r.run] = run_config(r.run)
        cfg = cfgs[r.run]
        conf = subrun_conf(cfg, r.subrun) if cfg else {}
        mm = m[(m.run == r.run) & (m.subrun == r.subrun)]
        mm = mm[~mm.file.str.contains('pedestal')]
        tags = sorted(mm.tag.unique())
        if not tags:
            rows.append(dict(r._asdict(), tag=None, excluded=True, exclude_reason='no datrun files on EOS', **conf))
            continue
        for tag in tags:
            t = mm[mm.tag == tag]
            dec, fdf, hits = t[t.kind == 'dec'], t[t.kind == 'fdf'], t[t.kind == 'hits']
            tt = tag_time(tag)
            row = dict(r._asdict(), tag=tag, t_local=tt.strftime('%Y-%m-%d %H:%M'), **conf,
                       n_dec=len(dec), n_fdf=len(fdf), n_hits=len(hits),
                       dec_gb=round(dec.bytes.sum() / 1e9, 3), fdf_gb=round(fdf.bytes.sum() / 1e9, 3),
                       dec_min_bytes=int(dec.bytes.min()) if len(dec) else 0,
                       eos_dir=f'{EOS_RUNS}/run_{r.run}/{r.subrun}')
            row['noise_era'] = 'quiet' if r.run <= 67 else ('unplaced' if r.run == 68 else 'noisy')
            row['access_epoch'] = 'pre_access_27jul' if tt < ACCESS_END else 'post_access_27jul'
            row['a_cm_x3'] = A_CM_START <= tt < ACCESS_END
            row['a_x_conn8_dead'] = A_CONN8_START <= tt < ACCESS_END
            reasons = []
            if len(fdf) and fdf.bytes.max() == 0:
                reasons.append('0-byte fdf files (acquisition never recorded data)')
            elif len(dec) < 8:
                reasons.append(f'decoded_root has {len(dec)}/8 FEUs')
            if len(dec) and dec.bytes.min() < 10_000:
                reasons.append('empty decoded_root file(s)')
            if conf.get('zero_suppress'):
                reasons.append('zero-suppressed (not the production readout)')
            row['excluded'] = bool(reasons)
            row['exclude_reason'] = '; '.join(reasons)
            rows.append(row)
    reg = pd.DataFrame(rows)
    for ch in CHAMBERS:
        reg[f'bench_{ch}'] = BENCH[ch]
    reg = reg.sort_values(['rclass', 'run', 'subrun', 'tag'], na_position='last')
    reg.to_csv(OUT, index=False)
    ok = reg[~reg.excluded]
    print(f'{len(reg)} acquisitions, {len(ok)} kept, {reg.excluded.sum()} excluded -> {OUT}')
    print(ok.groupby('rclass').agg(runs=('run', 'nunique'), subruns=('subrun', 'nunique'),
                                    acq=('tag', 'size'), dec_TB=('dec_gb', lambda x: round(x.sum() / 1e3, 2))))
    print(reg[reg.excluded][['run', 'subrun', 'tag', 'exclude_reason']].to_string(index=False))
    for col in ('latency', 'n_samples', 'gas', 'noise_era', 'access_epoch', 'a_x_conn8_dead'):
        print(col, ok[col].value_counts(dropna=False).to_dict())
    print('HV sets:', ok.groupby(['resist_A', 'resist_B', 'resist_C', 'resist_D',
                                   'drift_A', 'drift_B', 'drift_C', 'drift_D'], dropna=False).size().to_dict())


if __name__ == '__main__':
    main()
