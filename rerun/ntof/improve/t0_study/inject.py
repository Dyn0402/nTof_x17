#!/usr/bin/env python3
"""inject.py -- injection test of the chamber's response to a particle's arrival time (README.md, step 1).

    python -m rerun.ntof.improve.t0_study.inject --arm A --subrun stat090_0000 [--n-donors 300] [--jobs 14]

What it does.  A donor is a real late single track: ToF >= 25 ms, the only gated track of its chamber, one seed
cluster per view, not railed, not a split child, and with an in-time hit in its own wall group (so it arrived with
the trigger).  Its waveforms are cut out on the seed cluster's strips plus PAD strips on each side, shifted by k whole
samples (60 ns each, so the shift is exact), and ADDED to the waveforms of a host trigger.  The host is then
reconstructed with the pass-1 configuration (the run's E2_beam bundle, min 3 strips, as rerun/ntof/ringing/reco_test.py).

Hosts, one injection each:
  empty    a late (>= 25 ms) trigger with no seed at all in this chamber.  Every donor goes into one empty host at
           every k of the grid: the response (t0, tc, tan, acceptance) to a known arrival offset.
  (the y view is shifted by k_y = k + 0 or +-1 so that both views see the same true offset; see the plan step)
  pair     a late trigger holding another clean in-time single track.  Each gets one donor at a random k: the
           accidental same-chamber pair, with the true offset known.
  control  empty hosts reconstructed with nothing injected: how often an empty host makes a track by itself.

Seeding.  The host keeps its analyzer hits (as pass 1).  On the injected channels the analyzer hits are replaced by
hits from the injected waveform (5 sigma on the sample or the 3-sample running sum, as wft_beam.hits_from_waveforms),
since the analyzer never saw the injected charge.  The k = 0 injection measures what this seeding change, and the
extra noise of adding a second event's waveform on the donor strips (sqrt 2 there), do by themselves.

True offset.  The pin is t0_abs[view][ftst] = c - 10 ns x ftst (one ftst tick = 10 ns of trigger phase).  The donor's
charge sits at the donor's trigger phase, so in the host the injected track arrives at
    delta = 60 k + pin(ftst_donor) - pin(ftst_host)     per view,
relative to the host's pin, plus the donor's own in-time residual (its pass-1 t0 - pin, stored).

A negative k (early arrival) moves samples out of the start of the window; the samples it opens at the end are filled
with the donor's last sample (the slow tail), not with zero, since a zero would draw a false column end.  So k < 0 is
an approximation at the window end; k >= 0 is exact (the opened start samples hold no charge from a late particle).

Output: $WORK_DATA/ntof_pass1/ana/t0_study/inject/run_145_<sub>_<arm>.{parquet,candidates.parquet,meta.json}
and .plan.parquet (one row per host: kind, k, donor, ftst, delta, the donor's pass-1 values).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO), str(REPO / 'rerun' / 'ntof')]
WD = os.environ.get('WORK_DATA', '/media/ucla/work')
FEUS = dict(A=(3, 4), C=(7, 8), D=(1, 2))
WF_ROOTS = [f'{WD}/ntof_pass1/runs/', f'{WD}/claude-jobs/bcad1a92/wf/']
TRIGTIME = '/media/dylan/data/x17/sept26_prelim/trigtime'
OUT = Path(WD) / 'ntof_pass1' / 'ana' / 't0_study' / 'inject'
KGRID = list(range(-6, 13))          # -360 ... +720 ns
PAD = 2                              # strips each side of the seed cluster carried with the donor (the kernel's reach)
LATE_MS = 25.0
SAMPLE_NS = 60.0


def shift(sig, k):
    """sig (n_ch, ns) moved k samples later (k > 0) or earlier (k < 0); see the module docstring for the fill."""
    if k == 0:
        return sig.copy()
    out = np.zeros_like(sig)
    if k > 0:
        out[:, k:] = sig[:, :-k]
    else:
        out[:, :k] = sig[:, -k:]
        out[:, k:] = sig[:, -1:]
    return out


def feu_of(path):
    return int(re.search(r'_(\d\d)\.root$', path).group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--arm', required=True, choices='AC')
    ap.add_argument('--n-donors', type=int, default=300)
    ap.add_argument('--n-control', type=int, default=300)
    ap.add_argument('--n-pair', type=int, default=1500)
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--label', default='')
    ap.add_argument('--pairs-only', action='store_true', help='only the pair hosts (same donors and plan as the full run)')
    ap.add_argument('--floor-local-mm', type=float, default=None,
                    help='wft.seed.SIG_FLOOR_LOCAL_MM for the reconstruction (default: the production plane-wide floor)')
    a = ap.parse_args()
    import pandas as pd
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from wft.calib import CalibrationBundle

    rng = np.random.default_rng(a.seed)
    reg = [r for r in csv.DictReader(open(REPO / 'rerun' / 'ntof' / 'ntof_registry.csv'))
           if f"run_{r['run']}" == a.run and r['subrun'] == a.subrun and r['excluded'] != 'True']
    tags = sorted(r['tag'] for r in reg)
    base = next(b for b in WF_ROOTS if os.path.isdir(f'{b}{a.run}/{a.subrun}/decoded_root'))
    r0 = reg[0]
    fx, fy = FEUS[a.arm]
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f'{a.run}_{a.subrun}_{a.arm}{a.label}'
    cfg = wb.BeamConfig(KEY=f'{a.run}_{a.subrun}_mx17_{a.arm}', RUN=a.run, SUB_RUN=a.subrun, DET_NAME=f'mx17_{a.arm}',
                        MX17_FEU_X=fx, MX17_FEU_Y=fy, BASE_PATH=base, OUT_BASE=str(OUT),
                        HITS_DIR='combined_hits_root',
                        conditions=dict(gas=r0['gas'], sample_ns=float(r0['sample_period_ns']),
                                        n_samples=int(float(r0['n_samples']))))
    assert float(r0['sample_period_ns']) == SAMPLE_NS
    cfg.file_tags = tags
    hd = f'{WD}/ntof_pass1/runs/{a.run}/{a.subrun}/combined_hits_root/'
    if base != f'{WD}/ntof_pass1/runs/' and os.path.isdir(hd):
        type(cfg).combined_hits_dir = property(lambda self: hd)
    bundle = f'{WD}/ntof_pass1/bundles/pf/{a.arm}_E2_beam'
    cal = CalibrationBundle.load(bundle)
    pin = {p: {int(k): float(v) for k, v in cal.t0_abs[p].items()} for p in 'xy'}
    pos_maps = wio.strip_position_map(cfg)

    # ---- the late triggers, and the donors (wall-timed clean in-time singles) -----------------------------------
    t = pd.read_parquet(f'{TRIGTIME}/trigtime_{a.run}_{a.subrun}.parquet')
    late = set(t.eventId[(t.t_pred_ns / 1e6 >= LATE_MS) & (t.is_flash == 0)].astype(int))
    W = pd.read_parquet(f'{WD}/ntof_pass1/ana/t0_study/wall_hits.parquet',
                        filters=[('run', '==', a.run), ('subrun', '==', a.subrun), ('arm', '==', a.arm)])
    child = W.x_split_child.fillna(False).astype(bool) | W.y_split_child.fillna(False).astype(bool)
    W = W[(W.t_since_flash_ns >= LATE_MS * 1e6) & (W.n_gated_arm == 1) & ~child & ~W.t0_railed.astype(bool)
          & (W.n_cand_x == 1) & (W.n_cand_y == 1)]
    tw = W.t_wall
    pk = tw[(tw > -60) & (tw < 40)].median()
    clean = W[np.abs(tw - pk) < 30].drop_duplicates(['tag', 'event_id'])
    # stage-2 pass-1 values of the clean events (the donor's own reco: the reference)
    s2 = pd.concat([pd.read_parquet(f'{WD}/ntof_pass1/stage2/{a.run}/{a.subrun}/{a.arm}/'
                                    f'{a.run}_{a.subrun}_{tg}_{a.arm}.parquet').assign(tag=tg) for tg in tags])
    keep = ['tag', 'event_id'] + [f'{p}_{c}' for p in 'xy' for c in ('p0', 't0', 'q_u50', 'q_u90', 'tan_theta', 'ftst',
                                                                    'q_sum', 'n_strips')]
    clean = clean[['tag', 'event_id']].merge(s2[keep], on=['tag', 'event_id'])

    # ---- seeds from the analyzer hits: which late triggers are empty; the donors' clusters ---------------------
    entries = {}                       # tag -> {eid: entry index} (both FEUs must hold the event)
    seeded, seeds_all = {}, {}
    for tg in tags:
        idx = None
        for feu in (fx, fy):
            f = [p for p in wio.subrun_files(base, a.run, a.subrun, feu) if wio.file_tag(p) == tg][0]
            import uproot
            ev = uproot.open(f)['nt'].arrays(['eventId'], library='np')['eventId'].astype(int)
            d = {int(e): i for i, e in enumerate(ev)}
            idx = d if idx is None else {e: i for e, i in d.items() if e in idx and idx[e] >= wio.N_PED_EVENTS}
        entries[tg] = {e: i for e, i in idx.items() if i >= wio.N_PED_EVENTS}
        h = wb.read_hits_tag(wb.hits_file_for_tag(cfg, tg), (fx, fy))
        sd = wb.seeds_from_hits_beam(h, pos_maps, fx, fy, hot=cal.hot, min_strips=3, dead=cal.dead)
        seeds_all[tg] = sd
        seeded[tg] = {e for e, s in sd.items() if s['x'] or s['y']}
        # an empty host has no analyzer hit cluster at all; also no hit-bearing seed entry
        seeded[tg] |= set(h.eventId.astype(int)[h.groupby('eventId').channel.transform('size') >= 3])
    clean = clean[[e in entries[tg] and len(seeds_all[tg].get(e, {}).get('x', [])) == 1
                   and len(seeds_all[tg].get(e, {}).get('y', [])) == 1
                   for tg, e in zip(clean.tag, clean.event_id)]].reset_index(drop=True)
    empty = [(tg, e) for tg in tags for e in entries[tg] if e in late and e not in seeded[tg]]
    rng.shuffle(empty)
    order = rng.permutation(len(clean))
    nd = min(a.n_donors, len(clean) // 2)
    donors = clean.iloc[order[:nd]].reset_index(drop=True)
    pair_hosts = clean.iloc[order[nd:nd + a.n_pair]].reset_index(drop=True)
    need = nd * len(KGRID) + a.n_control
    if len(empty) < need:
        raise SystemExit(f'only {len(empty)} empty late hosts for {need} injections; lower --n-donors')
    print(f'{a.arm} {a.subrun}: {len(clean)} clean in-time singles -> {nd} donors, {len(pair_hosts)} pair hosts; '
          f'{len(empty)} empty late hosts', flush=True)

    # ---- the donors' waveforms on their cluster +- PAD strips --------------------------------------------------
    sig = {}                            # (donor index) -> {feu: (ch, W[ch])}
    for tg, grp in donors.groupby('tag'):
        want = set(grp.event_id.astype(int))
        di = dict(zip(grp.event_id.astype(int), grp.index))
        for plane, feu in (('x', fx), ('y', fy)):
            f = [p for p in wio.subrun_files(base, a.run, a.subrun, feu) if wio.file_tag(p) == tg][0]
            rdr = wio.FeuReader(f)
            for eid, _ft, Wf in rdr.iter_events(want):
                s = seeds_all[tg][eid][plane][0]
                win = wio.extract_window(Wf, rdr.noise, pos_maps[feu], s.channels, PAD)
                sig.setdefault(di[eid], {})[feu] = (win.ch.astype(int), Wf[win.ch].astype(np.float32).copy())
    donors = donors[[i in sig and len(sig[i]) == 2 for i in donors.index]]

    # ---- the plan --------------------------------------------------------------------------------------------
    plan, it = [], iter(empty)
    for i in donors.index:
        for k in KGRID:
            tg, e = next(it)
            plan.append(dict(kind='empty', host_tag=tg, host_eid=e, donor=i, k=k))
    for _ in range(a.n_control):
        tg, e = next(it)
        plan.append(dict(kind='control', host_tag=tg, host_eid=e, donor=-1, k=0))
    for j in pair_hosts.index:
        plan.append(dict(kind='pair', host_tag=pair_hosts.tag[j], host_eid=int(pair_hosts.event_id[j]),
                         donor=int(rng.choice(donors.index)), k=int(rng.choice(KGRID)), host_row=j))
    plan = pd.DataFrame(plan)
    if a.pairs_only:
        plan = plan[plan.kind == 'pair'].reset_index(drop=True)
    hft = {}
    for tg in tags:
        for plane, feu in (('x', fx), ('y', fy)):
            f = [p for p in wio.subrun_files(base, a.run, a.subrun, feu) if wio.file_tag(p) == tg][0]
            import uproot
            arr = uproot.open(f)['nt'].arrays(['eventId', 'ftst'], library='np')
            hft.update({(tg, plane, int(e)): int(v) for e, v in zip(arr['eventId'], arr['ftst'])})
    for p in 'xy':
        plan[f'host_ftst_{p}'] = [hft.get((tg, p, int(e)), -1) for tg, e in zip(plan.host_tag, plan.host_eid)]
    # the y view's own shift: the donor's charge keeps the donor's x-y trigger phase, so with the same k in both views
    # the injected track's x-y column times would differ by e = 10 ns x ((ftst_x - ftst_y)_host - (ftst_x - ftst_y)_donor),
    # which is 0 or +-60 ns, one sample.  k_y = k + e / 60 makes both views see the same true offset (delta_x = delta_y).
    dfx = plan.donor.map(donors.x_ftst).fillna(0).astype(int)
    dfy = plan.donor.map(donors.y_ftst).fillna(0).astype(int)
    e = 10 * ((plan.host_ftst_x - plan.host_ftst_y) - (dfx - dfy))
    e = e.where(plan.kind != 'control', 0)
    if not np.all(np.mod(e, SAMPLE_NS) == 0):
        raise SystemExit('x-y phase difference not a whole sample; the y shift cannot be exact')
    plan['k_x'] = plan.k
    plan['k_y'] = (plan.k + e // SAMPLE_NS).astype(int)
    inj = {}                            # (feu, tag, eid) -> (ch, shifted signal)
    for r in plan[plan.kind != 'control'].itertuples():
        for feu, kk in ((fx, r.k_x), (fy, r.k_y)):
            ch, s = sig[r.donor][feu]
            inj[(feu, r.host_tag, int(r.host_eid))] = (ch, shift(s, int(kk)))

    # ---- patch the reader (adds the donor) and the hits (injected channels from the injected waveform) ---------
    orig_iter = wio.FeuReader.iter_events

    def iter_events(self, wanted=None, with_raw=False):
        feu, tg = feu_of(self.path), wio.file_tag(self.path)
        for x in orig_iter(self, wanted, with_raw):
            key = (feu, tg, x[0])
            if key in inj:
                ch, s = inj[key]
                Wx = x[2].copy()
                Wx[ch] += s
                x = (x[0], x[1], Wx) + tuple(x[3:])
            yield x
    wio.FeuReader.iter_events = iter_events

    orig_hits = wb.read_hits_tag

    def read_hits(path, feus):
        h = orig_hits(path, feus)
        tg = wio.file_tag(path.replace('_feu-combined_hits', '_00'))
        mine = {k: v for k, v in inj.items() if k[1] == tg}
        if not mine:
            return h
        add, drop = [], set()
        for feu in feus:
            want = {k[2] for k in mine if k[0] == feu}
            if not want:
                continue
            f = [p for p in wio.subrun_files(base, a.run, a.subrun, feu) if wio.file_tag(p) == tg][0]
            rdr = wio.FeuReader(f)
            nz = np.maximum(rdr.noise.astype(np.float64), 1e-3)
            for eid, _ft, Wf in rdr.iter_events(want):
                ch, _ = inj[(feu, tg, eid)]
                Wc = np.asarray(Wf[ch], np.float64)
                c = np.cumsum(np.pad(Wc, ((0, 0), (1, 0))), axis=1)
                rs = (c[:, 3:] - c[:, :-3]).max(axis=1) / (nz[ch] * np.sqrt(3) * 1.3)
                amp = Wc.max(axis=1)
                sg = np.maximum(amp / nz[ch], rs)
                ok = sg >= wb.WAVE_SIG
                drop |= {(eid, feu, int(c_)) for c_ in ch}
                add.append(pd.DataFrame(dict(eventId=eid, feu=feu, channel=ch[ok], amplitude=amp[ok],
                                             significance=sg[ok], max_sample=Wc[ok].argmax(axis=1))))
        m = np.array([(int(e), int(f), int(c_)) in drop for e, f, c_ in zip(h.eventId, h.feu, h.channel)], bool)
        h = pd.concat([h[~m]] + add, ignore_index=True)
        return h.astype(dict(eventId='int64', feu='int64', channel='int64'))
    wb.read_hits_tag = read_hits

    if a.floor_local_mm is not None:
        from wft import seed as wseed
        wseed.SIG_FLOOR_LOCAL_MM = float(a.floor_local_mm)
    # ---- reconstruct ----------------------------------------------------------------------------------------
    allow = {tg: set(plan.host_eid[plan.host_tag == tg].astype(int)) for tg in tags}
    wb.reconstruct_subrun(cfg, bundle, str(stem) + '.parquet', jobs=a.jobs, allow_events=allow,
                          allow_meta=dict(source='t0_study inject plan'), allow_bow=True, min_strips=3,
                          extra_meta=dict(inject=dict(kgrid=KGRID, pad=PAD, late_ms=LATE_MS, wall_peak=float(pk),
                                                      n_donors=int(len(donors)), seed=a.seed,
                                                      pairs_only=a.pairs_only, floor_local_mm=a.floor_local_mm)))

    # ---- the plan table, with the true offsets and the donor's own values --------------------------------------
    D = donors.add_prefix('d_')
    plan = plan.merge(D, left_on='donor', right_index=True, how='left')
    H = pair_hosts.add_prefix('h_')
    plan = plan.merge(H, left_on='host_row', right_index=True, how='left') if 'host_row' in plan else plan
    for p in 'xy':
        dft = plan[f'd_{p}_ftst'].fillna(-1).astype(int)
        pd_ = np.array([pin[p].get(v, np.nan) for v in dft])
        ph = np.array([pin[p].get(v, np.nan) for v in plan[f'host_ftst_{p}']])
        plan[f'delta_{p}'] = SAMPLE_NS * plan[f'k_{p}'] + pd_ - ph
        plan[f'd_{p}_dt0'] = plan[f'd_{p}_t0'] - pd_
        plan[f'host_pin_{p}'] = ph
    plan.to_parquet(str(stem) + '.plan.parquet', index=False)
    print('wrote', str(stem) + '.plan.parquet', len(plan))


if __name__ == '__main__':
    main()
