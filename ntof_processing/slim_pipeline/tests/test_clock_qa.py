#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_clock_qa.py -- prove every clock-QA check actually fires.

    python test_clock_qa.py            # 14 cases, ~5 s, no data needed

A monitor that has only ever seen good data is not a monitor: it is a program
that prints "PASS". Each case below synthesises a slim with ONE defect injected
and asserts that the named check reaches the expected level and that the
healthy control does not. If a threshold is ever loosened past the point of
uselessness, a case here goes red.

The synthetic segment mimics the real thing closely enough for the checks --
same tree names and columns, a realistic 6 ns residual core, 80 ms of
time-since-flash, ~1000 bunches -- but it is NOT physics. It exists to exercise
the judgement, not to validate the fit.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ntof_processing.slim_pipeline import clock_qa as Q       # noqa: E402
from ntof_processing.slim_pipeline import config as C         # noqa: E402

RNG = np.random.default_rng(20260809)
N_EV, N_BUNCH = 40_000, 800


def make_segment(d: Path, *, efficiency=0.958, accidental=0.00046,
                 cv_gap=0.0004, resid_mean=0.0, resid_rms=6.3, drift_ns=0.0,
                 K=1.1063e-4, T0=-252.6, arm_shift=None, arm_resid_shift=None,
                 da_rms=6.5, dk_rms_ppm=0.6, frac_fitted=0.99, flash_hits=0,
                 late_clip=False, primary_out_frac=0.02, ringing_loud_frac=0.0,
                 bootstrap_snr=180.0, with_bunches=True, with_beam=True,
                 empty_frac=0.0, empty_leak=0, empty_full_bursts=False,
                 dream_run='run_00', dream_subrun='synthetic', ntof_run=0,
                 n_ev=None, join=('count', 40, None), delta_hint_s=None,
                 wall_leg_frac=1.0):
    """Write a synthetic ntof_hits directory with the requested properties."""
    d.mkdir(parents=True, exist_ok=True)
    global N_EV, RNG
    # Reseed per case, so a case's data depends only on its own arguments.
    # Sharing one stream across CASES makes every case depend on how many draws
    # the ones before it happened to make: adding a case in the middle moved a
    # borderline one (resid_mean = 10.0 against a 10 ns per-arm threshold)
    # across its line, which is noise in the harness rather than a finding.
    RNG = np.random.default_rng(20260809)
    N_EV = n_ev or 40_000
    n_flash = 500
    n_phys = N_EV - n_flash
    ev_id = np.arange(N_EV, dtype=np.uint64)
    is_flash = np.zeros(N_EV, np.uint8); is_flash[:n_flash] = 1
    phys = is_flash == 0
    bunch = RNG.integers(100, 100 + N_BUNCH, N_EV).astype(np.int32)
    t = RNG.uniform(0, 80e6, N_EV)                     # ns since flash

    matched = np.zeros(N_EV, np.uint8)
    idx = np.flatnonzero(phys)
    take = RNG.choice(idx, int(efficiency * n_phys), replace=False)
    matched[take] = 1
    m = matched == 1

    resid = np.full(N_EV, np.nan, np.float32)
    r = RNG.normal(resid_mean, resid_rms, m.sum())
    if drift_ns:                                       # linear in t
        r = r + drift_ns * (t[m] / 80e6 - 0.5)
    arm = np.full(N_EV, -1, np.int8)
    arm[m] = RNG.integers(0, 4, m.sum())
    if arm_resid_shift:                                # one arm's fit is wrong
        for k, v in arm_resid_shift.items():
            r = r + np.where(arm[m] == 'ABCD'.index(k), v, 0.0)
    r = np.clip(r, -C.ACCEPT_NS, C.ACCEPT_NS)
    resid[m] = r

    events = dict(
        eventId=ev_id, bunch=bunch, t_dream_ns=t, is_flash=is_flash,
        t_pred_ns=t * (1 + K) + T0, matched=matched, residual_ns=resid,
        arm=arm, da_ns=np.zeros(N_EV, np.float32),
        dk=np.zeros(N_EV, np.float32),
        corr_ns=np.zeros(N_EV, np.float32),
        corr_cv_ns=np.zeros(N_EV, np.float32))

    ub = np.unique(bunch)
    # Empty pulses: PS pulses that delivered no protons. A real slim drops
    # their triggers and keeps the bunch row with has_beam = 0, so by
    # construction they are never fitted -- 'bunches fitted' must not count
    # them, and any of their triggers found in the events tree is a filter
    # regression. `empty_leak` injects exactly that.
    n_empty = int(empty_frac * ub.size)
    has_beam = np.ones(ub.size, np.uint8)
    inten = RNG.choice([413.0, 851.0], ub.size)
    n_trig = np.full(ub.size, N_EV // ub.size, np.int32)
    if n_empty:
        has_beam[-n_empty:] = 0
        inten[-n_empty:] = RNG.uniform(0.0, 1.0, n_empty)
        # A no-beam pulse holds background only: ~2 triggers against ~90.
        # `empty_full_bursts` is the mis-assigned join, where the dropped
        # bunches hold full bursts and the beam record is not what is wrong.
        n_trig[-n_empty:] = (n_trig[-n_empty:] if empty_full_bursts
                             else RNG.integers(0, 5, n_empty))
    beam_ub = ub[has_beam == 1]
    nfit = int(frac_fitted * beam_ub.size)
    fitted = np.zeros(ub.size, np.uint8)
    fitted[np.isin(ub, beam_ub[:nfit])] = 1
    bunches = dict(
        bunch=ub.astype(np.int32),
        n_triggers=n_trig,
        has_beam=has_beam,
        intensity_e10=inten.astype(np.float32),
        fitted=fitted,
        da_ns=RNG.normal(0, da_rms, ub.size).astype(np.float32),
        dk=(RNG.normal(0, dk_rms_ppm, ub.size) * 1e-6).astype(np.float32),
        n_core=np.full(ub.size, 40, np.int32))
    if not with_beam:
        del bunches['has_beam'], bunches['intensity_e10']
    if n_empty and not empty_leak:      # the filter did its job
        keep = np.isin(bunch, beam_ub)
        events = {k: v[keep] for k, v in events.items()}
        ev_id, is_flash, bunch, t = (events['eventId'], events['is_flash'],
                                     events['bunch'], events['t_dream_ns'])
        matched, resid, arm = (events['matched'], events['residual_ns'],
                               events['arm'])
        phys = is_flash == 0
        m = matched == 1
        n_flash = int((~phys).sum()); n_phys = int(phys.sum()); N_EV = ev_id.size

    # hits, in four populations mirroring what a real slim carries:
    #   background   6 per matched trigger, any tree, small amplitude
    #   primary      one MIP-sized PSS hit on the trigger's own arm at
    #                dt ~ N(-5, 7) -- what "largest on the trigger arm" finds
    #   ringing      after-pulse followers behind the primaries: same channel,
    #                a few % of the parent amplitude, dt decaying to ~1 us
    #   clip         (late_clip only) a one-sided uniform LIQ population in
    #                the outer late decile -- what real truncation looks like
    W = C.SLIM_NS
    per = 6
    n_bg = int(m.sum()) * per
    hid = [np.repeat(ev_id[m], per)]
    dt = [np.clip(RNG.normal(0, 35, n_bg), -W, W)]
    det = [RNG.integers(0, len(C.SCINT_TREES), n_bg).astype(np.int16)]
    detn = [RNG.integers(0, 4, n_bg).astype(np.int32)]
    amp0 = [RNG.uniform(60, 900, n_bg)]
    ctl = [(RNG.random(n_bg) < 0.02).astype(np.uint8)]

    n_m = int(m.sum())
    hid.append(ev_id[m])
    p_dt = RNG.normal(-5, 7, n_m)
    out = RNG.random(n_m) < primary_out_frac           # displaced primaries
    p_dt[out] = RNG.uniform(50, 900, out.sum()) * RNG.choice([-1, 1], out.sum())
    dt.append(np.clip(p_dt, -W, W))
    det.append((4 + arm[m]).astype(np.int16))          # PSS tree of own arm
    p_detn = RNG.integers(0, 2, n_m).astype(np.int32)
    detn.append(p_detn)
    p_amp = RNG.uniform(3000, 9000, n_m)
    amp0.append(p_amp)
    ctl.append(np.zeros(n_m, np.uint8))

    lead = RNG.random(n_m) < 0.5                       # parents that ring
    nf = RNG.poisson(3.0, lead.sum())
    f_ev = np.repeat(ev_id[m][lead], nf)
    f_parent = np.repeat(np.flatnonzero(lead), nf)
    f_dt = (np.repeat(dt[1][lead], nf)
            + RNG.exponential(250.0, f_ev.size) + 18.0)
    keepf = f_dt <= W
    f_ratio = np.where(RNG.random(f_ev.size) < ringing_loud_frac,
                       RNG.uniform(0.30, 0.90, f_ev.size),
                       RNG.uniform(0.005, 0.045, f_ev.size))
    hid.append(f_ev[keepf])
    dt.append(f_dt[keepf])
    det.append(np.repeat(det[1][lead], nf)[keepf])     # parent's channel,
    detn.append(np.repeat(p_detn[lead], nf)[keepf])    # so the flag can see it
    amp0.append((p_amp[f_parent] * f_ratio)[keepf])
    ctl.append(np.zeros(int(keepf.sum()), np.uint8))

    if late_clip:                                      # one-sided truncation
        n_c = 2 * n_m
        hid.append(RNG.choice(ev_id[m], n_c))
        dt.append(RNG.uniform(0.9 * W, W, n_c))
        det.append(RNG.integers(8, 12, n_c).astype(np.int16))   # LIQ
        detn.append(RNG.integers(0, 4, n_c).astype(np.int32))
        amp0.append(RNG.uniform(60, 900, n_c))
        ctl.append(np.zeros(n_c, np.uint8))

    if flash_hits:                                     # hits on flash triggers
        hid.append(ev_id[:flash_hits])
        dt.append(RNG.uniform(-W, W, flash_hits))
        det.append(RNG.integers(0, 12, flash_hits).astype(np.int16))
        detn.append(RNG.integers(0, 4, flash_hits).astype(np.int32))
        amp0.append(RNG.uniform(60, 900, flash_hits))
        ctl.append(np.zeros(flash_hits, np.uint8))

    # ------------------------------------------------------- the WALL leg
    # DREAM triggered on a wall AND plastic coincidence on ONE arm, so every
    # physics trigger has both legs. The fixture carried only the plastic until
    # 2026-08-13, which left `pulses fully matched` nothing but background to
    # find and made the HEALTHY case fail its own verdict.
    #
    # The legs go on ALL physics triggers, not just `matched` ones. `matched`
    # means the offline N1081B EMULATION rebuilt the coincidence; the hits are
    # there either way, and clock_qa deliberately asks the physical question
    # instead -- measured on run_79/stat090_0000, 99.5 % of the triggers the
    # emulator calls unmatched do have both legs inside +-25 ns. Tying the
    # coincidence to `matched` capped the fixture's per-pulse fraction at the
    # match efficiency and put its median at 94.6 % against the campaign's
    # measured 96.2 %.
    #
    # APPENDED LAST ON PURPOSE: every draw above keeps the RNG stream it had,
    # so the other cases see bit-identical data (see the reseeding note above).
    # wall_leg_frac < 1 is how a segment with unmatched PULSES is built.
    def _leg(sel_ids, sel_arm, tree_base):
        """One MIP-sized primary per trigger, on that trigger's own arm."""
        n_ = sel_ids.size
        if not n_:
            return
        hid.append(sel_ids)
        dt.append(np.clip(RNG.normal(-5, 7, n_), -W, W))
        det.append((tree_base + sel_arm).astype(np.int16))
        detn.append(RNG.integers(1, 9, n_).astype(np.int32))
        amp0.append(RNG.uniform(3000, 9000, n_))
        ctl.append(np.zeros(n_, np.uint8))

    w_keep = RNG.random(n_m) < wall_leg_frac
    _leg(ev_id[m][w_keep], arm[m][w_keep], 0)          # WAL tree of own arm

    # the unmatched physics triggers: both legs, at the measured 99.5 %
    um = phys & (matched == 0)
    if um.any():
        u_keep = RNG.random(int(um.sum())) < 0.995 * wall_leg_frac
        u_id = ev_id[um][u_keep]
        # `events.arm` is -1 for an unmatched trigger -- the emulation never
        # assigned one, so there is nothing to reuse. The trigger still fired on
        # a real arm; draw it here rather than letting -1 address tree -1.
        u_arm = RNG.integers(0, 4, u_id.size).astype(np.int8)
        _leg(u_id, u_arm, 0)                           # WAL
        _leg(u_id, u_arm, 4)                           # PSS

    hid = np.concatenate(hid)
    dt = np.concatenate(dt)
    det = np.concatenate(det)
    detn = np.concatenate(detn)
    amp0 = np.concatenate(amp0)
    ctl = np.concatenate(ctl)
    hits = dict(eventId=hid.astype(np.uint64), det=det.astype(np.int16),
                detn=detn, dt_ns=dt.astype(np.float32), is_control=ctl,
                amp=amp0.astype(np.float32), amp_0=amp0.astype(np.float32),
                satuflag=np.zeros(dt.size, np.uint8))

    fname = f'ntof_hits_{dream_run}_{dream_subrun}_{ntof_run:06d}.root'
    with uproot.recreate(d / fname) as f:
        f['events'] = events
        f['hits'] = hits
        if with_bunches:
            f['bunches'] = bunches

    arm_off = dict(Q.REF_ARM)
    if arm_shift:
        for k, v in arm_shift.items():
            arm_off[k] = arm_off[k] + v
    boot = None
    if bootstrap_snr:
        boot = dict(peak_ns=0.0, counts=int(200 * bootstrap_snr / 10),
                    floor=max(1.0, 200 * bootstrap_snr / 10 / bootstrap_snr),
                    snr=bootstrap_snr, n_candidates=50000, search_ns=50000.0,
                    hist=dict(lo_ns=-50000.0, bin_ns=200.0,
                              counts=[1] * 500))
    # join provenance: (chosen_by, margin, r_sig), or None for a file written
    # before the block existed
    jn = None
    if join is not None:
        by, margin, r_sig = join
        jn = dict(pulse_match_offset_s=12.34, pulse_match_margin=margin,
                  pulse_match_chosen_by=by, pulse_match_r_sig=r_sig,
                  delta_s=0.829, delta_margin=507,
                  delta_hint_s=delta_hint_s)
    (d / 'calibration.json').write_text(json.dumps(dict(
        K=K, T0_ns=T0, arm_offset_ns=arm_off, accept_ns=C.ACCEPT_NS,
        slim_ns=W, control_shift_ns=C.CONTROL_SHIFT_NS,
        n_bunches_fitted=nfit, fit=dict(iters=[], bootstrap=boot),
        join=jn)))
    (d / 'qa.json').write_text(json.dumps(dict(
        efficiency=efficiency, efficiency_cv=efficiency - cv_gap,
        accidental=accidental, purity=1 - accidental,
        n_events=N_EV, n_physics=n_phys, n_flash=n_flash,
        n_hits=int(dt.size), hits_per_trigger=float(dt.size / n_phys),
        seconds=1.0)))
    (d / 'provenance.json').write_text(json.dumps(dict(
        dream_run=dream_run, dream_subrun=dream_subrun, ntof_run=ntof_run,
        ntof_processing='synthetic')))
    return d


CASES = [
    # (name, kwargs, check name, expected level)
    ('healthy',            {},                              None,   'PASS'),
    # --- join lock provenance ---------------------------------------------
    # A thin count margin is not proof the segment is wrong -- it is a
    # statement that little evidence chose its lock. 14 of the first
    # campaign's 170 ACCEPTED segments sat at margin <= 2.
    ('join margin thin',   dict(join=('count', 5, None)),
     'pulse_match margin adequate', 'WARN'),
    ('join margin tie',    dict(join=('count', 0, None)),
     'pulse_match margin adequate', 'FAIL'),
    # THE GUARD AGAINST OVER-TIGHTENING. Two segments in the margin study sat
    # at count margin 0 and were CORRECT -- the intensity fluctuation, not the
    # count, is what chose them. If a future tightening makes this FAIL, the
    # gate has started rejecting good data, which is the failure mode the
    # whole fix exists to avoid.
    ('margin 0 but arbitrated', dict(join=('intensity', 0, 4.5)),
     'pulse_match margin adequate', 'WARN'),
    # a scan-verified override carries no margin of its own and must not be
    # punished for it
    ('scan verified',      dict(join=('verified', None, None)),
     'pulse_match margin adequate', 'PASS'),
    # coincidence_arbiter: the lock was CHOSEN by the wall+plastic coincidence
    # rather than confirmed by it afterwards. Strongest evidence in the
    # pipeline, so it adjudicates like a hand-run shift scan -- and it must not
    # be punished for carrying no count margin, which is the whole point of it.
    ('arbiter chose the lock', dict(join=('coincidence', None, None)),
     'pulse_match margin adequate', 'PASS'),
    # arbitration that did not actually separate is not arbitration
    ('intensity too weak', dict(join=('intensity', 1, 1.2)),
     'pulse_match margin adequate', 'FAIL'),
    ('no join provenance', dict(join=None),
     'pulse_match margin adequate', 'NA'),
    ('efficiency low',     dict(efficiency=0.80),           'match efficiency', 'WARN'),
    ('efficiency dead',    dict(efficiency=0.40),           'match efficiency', 'FAIL'),
    ('accidental high',    dict(accidental=0.004),          'accidental rate', 'WARN'),
    ('accidental huge',    dict(accidental=0.05),           'accidental rate', 'FAIL'),
    ('overfit per-bunch',  dict(cv_gap=0.03),               'cross-validation gap', 'FAIL'),
    ('fit off-centre',     dict(resid_mean=5.0),            'residual centred', 'WARN'),
    ('fit badly off',      dict(resid_mean=10.0),           'residual centred', 'FAIL'),
    ('residuals wide',     dict(resid_rms=14.0),            'residual width', 'WARN'),
    ('clock drifts',       dict(drift_ns=12.0),             'no residual drift in time-since-flash', 'WARN'),
    ('K unphysical',       dict(K=3.0e-4),                  'K in physical range', 'FAIL'),
    ('arm offset moved',   dict(arm_shift={'C': 20.0}),     'arm offsets vs reference', 'WARN'),
    # the lat051 signature: fitted offsets displaced, so one arm's matched
    # residuals centre off zero while everything global stays green
    ('arm residual off',   dict(arm_resid_shift={'C': 11.0}),
     'per-arm residuals centred', 'FAIL'),
    ('arm residual drifted', dict(arm_resid_shift={'B': 4.0}),
     'per-arm residuals centred', 'WARN'),
    ('per-bunch scatter',  dict(da_rms=22.0),               'per-bunch offset scatter', 'WARN'),
    ('bunches unfitted',   dict(frac_fitted=0.60),          'bunches fitted', 'FAIL'),
    # An empty-pulse-heavy segment is the PS having a bad night, not a bad
    # clock: 20 % of its pulses delivered no protons, every bunch that DID get
    # beam was fitted, and the check must stay green. Judged over all bunches
    # this was 'bunches fitted' 0.80 = WARN, which is what the first campaign's
    # four WARNs actually were.
    ('empty pulses, beam fine', dict(empty_frac=0.20),      'bunches fitted', 'PASS'),
    ('empty pulses + real deficit', dict(empty_frac=0.20, frac_fitted=0.60),
     'bunches fitted', 'FAIL'),
    ('no-beam triggers left in', dict(empty_frac=0.10, empty_leak=1),
     'no-beam pulses filtered out', 'FAIL'),
    # the run_116/stat090_0013 signature: the join fitted a -1,324 s offset and
    # paired unrelated bursts to unrelated pulses, so the "empty" bunches held
    # 66-108 triggers each. The beam record is not what is wrong there.
    ('dropped pulses hold full bursts', dict(empty_frac=0.10,
                                             empty_full_bursts=True),
     'dropped pulses look like no beam', 'FAIL'),
    ('old file, no beam record', dict(with_beam=False),
     'no-beam pulses filtered out', 'NA'),
    ('hits on flash',      dict(flash_hits=250),            'flash triggers carry no hits', 'FAIL'),
    # truncation is ONE-SIDED late excess at the edge; the symmetric pedestal
    # a biased control leaves must NOT fire it (that is the 'healthy' case)
    ('window clipping',    dict(late_clip=True),            'coincidence contained in slim window', 'FAIL'),
    # ringing followers with amplitudes too big for the shadow flag = a late
    # tail that is NOT explained by ringing, which a human should look at
    ('late tail not ringing', dict(ringing_loud_frac=0.55),
     'PSS late tail is ringing', 'WARN'),
    # the largest plastic pulse per trigger wanders off the accept window
    ('plastic primary displaced', dict(primary_out_frac=0.30),
     'plastic primary within accept', 'WARN'),
    ('plastic primary lost', dict(primary_out_frac=0.55),
     'plastic primary within accept', 'FAIL'),
    # --- unmatched pulses, THE follow-up quantity --------------------------
    # A pulse is matched when >= PULSE_MIN_FRAC (0.60 since 2026-08-15, was
    # 0.80) of its physics triggers show the wall+plastic coincidence. The
    # two populations are three orders of magnitude apart (96.2 % at the
    # right lock against ~0.05 % at a wrong one), so a segment sits at one end
    # or the other and the level is really asking "how many pulses fell off
    # the good end". wall_leg_frac scales the per-pulse coincidence: 0.68
    # puts the per-pulse spread across the bar (a few below), 0.45 puts most
    # of it below.
    ('a few pulses unmatched', dict(wall_leg_frac=0.68),
     'pulses fully matched', 'WARN'),
    ('most pulses unmatched', dict(wall_leg_frac=0.45),
     'pulses fully matched', 'FAIL'),
    ('weak coarse peak',   dict(bootstrap_snr=9.0),         'coarse peak above floor', 'WARN'),
    ('no bootstrap record', dict(bootstrap_snr=0.0),        'coarse peak above floor', 'NA'),
    ('old file, no bunches', dict(with_bunches=False),      'bunches fitted', 'NA'),
]


# Collateral that is CORRECT, not a threshold accident: a defect that two
# checks can both see should light both up. A global +10 ns residual offset
# displaces every arm by +10 ns, which is exactly what the per-arm check is
# built to notice (its FAIL threshold is 10).
ALSO_FAILS = {
    'fit badly off': ('per-arm residuals centred',),
    # A plastic primary displaced outside the accept window IS a lost
    # wall+plastic coincidence -- the same defect, seen once per trigger by
    # 'plastic primary within accept' and once per pulse by 'pulses fully
    # matched'. Suppressing the second would be hiding a true positive.
    'plastic primary displaced': ('pulses fully matched',),
    'plastic primary lost': ('pulses fully matched',),
}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix='clockqa_'))
    fails = []
    print(f'{"case":<24} {"check":<38} {"got":<6} {"want":<6}')
    print('-' * 78)
    try:
        for name, kw, check, want in CASES:
            d = make_segment(tmp / name.replace(' ', '_'), **kw)
            q = Q.analyse(d)
            if check is None:
                got = q.verdict
            else:
                hit = [c for c in q.checks if c['name'] == check]
                if not hit:
                    fails.append(f'{name}: no check named {check!r}')
                    continue
                got = hit[0]['level']
            ok = got == want
            print(f'{name:<24} {(check or "VERDICT"):<38} {got:<6} {want:<6}'
                  f'{"" if ok else "   <-- MISMATCH"}')
            if not ok:
                fails.append(f'{name}: {check or "verdict"} was {got}, '
                             f'expected {want}')
            # the healthy control must stay clean in every non-verdict case
            if check is not None and want in ('WARN', 'FAIL'):
                others = [c for c in q.checks
                          if c['name'] != check and c['level'] == 'FAIL'
                          and c['name'] not in ALSO_FAILS.get(name, ())]
                if others and name not in ('K unphysical',):
                    fails.append(f'{name}: collateral FAIL in '
                                 f'{[c["name"] for c in others]}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print('-' * 78)
    if fails:
        print(f'{len(fails)} PROBLEM(S):')
        for f in fails:
            print(f'  {f}')
        return 1
    print(f'all {len(CASES)} cases behaved as specified')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
