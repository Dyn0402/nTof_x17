#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clock_qa.py -- judge one segment's DREAM -> n_TOF clock fit, and say why.

    python clock_qa.py <ntof_hits_dir>            # one segment, prints verdict
    python clock_qa.py <dir> --json clock_qa.json # and writes the record

The clock fit is the load-bearing step of the whole slim: get it wrong and
every hit in the file is attached to the wrong trigger, while the file itself
looks perfectly healthy. A silent 360 ns error is what the ntof_hits format
CANNOT show you, so this exists to make it loud.

Design notes
------------
* Reads only the WRITTEN products (slim .root + calibration.json), never the
  30 GB source, so it runs anywhere in a second and can be re-run on files
  produced by an older version of the pipeline.
* Every check has a threshold with a reason attached, and each returns
  PASS / WARN / FAIL. WARN means "a human should look"; FAIL means "do not use
  this segment". A check that cannot be evaluated returns NA and says so --
  it never silently passes.
* The thresholds are deliberately absolute (per segment). Outlier-against-the-
  fleet checks live in `make_clock_dashboard.py`, because they need every
  segment at once and a single segment cannot know it is odd.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ntof_processing.slim_pipeline import config as C          # noqa: E402

ARMS = ('A', 'B', 'C', 'D')
FAMILIES = ('WAL', 'PSS', 'LIQ')

# ---------------------------------------------------------------- thresholds
# Every number here is measured, not chosen. The reference pair is
# run_79/stat090_0000 x 224572 (DREAM_NTOF_CALIBRATION.md); run_77 x 224571 is
# the second independent pair and brackets most of these.
TH = dict(
    # efficiency 95.84 % reference, 95.70 % on run_77. A pair that has genuinely
    # less overlap can sit lower without being wrong, so WARN is generous.
    efficiency_warn=0.90, efficiency_fail=0.70,
    # accidental 0.049 % reference, 0.061 % on run_77.
    accidental_warn=0.002, accidental_fail=0.010,
    # in-sample minus cross-validated. 0.03 pts reference, 0.075 pts run_77.
    # A large gap means the per-bunch fit is reading back its own input.
    cv_gap_warn=0.005, cv_gap_fail=0.020,
    # coarse-search peak over the accidental floor beside it. The fit refuses
    # below 6; anything under 20 is close enough to the floor to distrust.
    boot_snr_warn=20.0, boot_snr_fail=6.0,
    # per-bunch scatter: 6.67 ns / 0.62 ppm reference.
    da_rms_warn=15.0, da_rms_fail=30.0,
    dk_rms_ppm_warn=2.0, dk_rms_ppm_fail=5.0,
    # Of the bunches that HAD BEAM: 100.0000 % over the first campaign
    # (45,225 parasitic and 49,473 dedicated, two exceptions, both the first
    # bunch of a sub-run). Empty pulses are excluded from this fraction since
    # 2026-08-10 -- judged over all bunches it measures the PS, not the fit.
    bunch_fit_frac_warn=0.90, bunch_fit_frac_fail=0.70,
    # residual core after the per-bunch correction: 6 ns RMS by construction.
    resid_rms_warn=12.0, resid_rms_fail=20.0,
    # the fit is centred by construction; a systematic offset means it settled
    # somewhere other than the peak.
    resid_mean_warn=3.0, resid_mean_fail=8.0,
    # residual must not drift across the segment -- that is what per-bunch is
    # for. Slope over the whole sub-run, in ns.
    drift_warn=8.0, drift_fail=20.0,
    # K is a clock rate ratio and physically ~1.1e-4 for this DAQ pair.
    k_lo=0.9e-4, k_hi=1.3e-4,
    # arm offsets reproduce between sub-runs to <= 2.6 ns; the four differ from
    # each other by ~25 ns, so a 15 ns move is well outside normal.
    arm_dev_warn=15.0, arm_dev_fail=40.0,
    # matched residuals must centre PER ARM, not just overall: the offsets are
    # fitted quantities and a wrong one leaves its arm displaced while the
    # arms average to zero. run_78/stat090_lat051_c0_0005 sat at C +11.3 ns /
    # D -8.1 ns from exactly that and passed every global check.
    arm_resid_warn=3.0, arm_resid_fail=10.0,
    # per (matched trigger, its own arm): the LARGEST-amplitude plastic hit in
    # the slim window lands within +-ACCEPT_NS. Measured 92.0 % on the
    # reference segment at the production +-1 us window, 91.5-92.6 % on the
    # run_78 short segments (pss_ringing/report_veto.html measured 89.5 % on
    # a +-3 us slim, where more unrelated singles compete). "Earliest" is the
    # wrong estimator at 720 kHz singles -- it gives 31 %.
    pss_primary_warn=0.80, pss_primary_fail=0.60,
    # fraction of the PSS 150-1000 ns late excess removed by the in-window
    # shadow flag (amp_0 < 0.05 x a bigger earlier hit, same channel, <=1 us).
    # Measured 100.6 % on the reference segment, 99.4/104.4 % on the run_78
    # shorts (>100 % = the residual goes slightly negative, i.e. the whole
    # excess is explained to within subtraction noise). The full-stream flag
    # removes 99.5 %; for the LATE tail in-window lookback is complete
    # because an after-pulse at +dt always has its parent inside the window.
    ringing_removed_warn=0.90,
    # counts below which the late excess is too small to classify at all.
    ringing_min_excess=100,
    # flash triggers are written with no hits by construction.
    flash_hits_fail=0,
)
# The shadow flag's operating point, shared with the slim's stored branches.
SHADOW_RATIO = C.SHADOW_RATIO
SHADOW_HOLD_NS = C.SHADOW_HOLD_NS
# Reference arm offsets, run_79 x 224572. Used only as a shape comparison.
REF_ARM = dict(A=-17.06, B=7.79, C=1.86, D=-1.01)


@dataclass
class Check:
    name: str
    level: str            # PASS | WARN | FAIL | NA
    value: float | None
    detail: str
    threshold: str = ''


@dataclass
class SegmentQA:
    segment: dict = field(default_factory=dict)
    verdict: str = 'NA'
    checks: list = field(default_factory=list)
    clock: dict = field(default_factory=dict)
    match: dict = field(default_factory=dict)
    perbunch: dict = field(default_factory=dict)
    hits: dict = field(default_factory=dict)
    bootstrap: dict = field(default_factory=dict)


def _arrays(f, name):
    """{branch: array} for a tree, whatever uproot hands back.

    `arrays(library='np')` returns a dict for some files and a structured
    ndarray for others (it depends on how the tree was written), and `'x' in
    arr` raises on the latter. Normalise once here rather than guessing.
    """
    if name not in f:
        return {}
    a = f[name].arrays(library='np')
    if isinstance(a, np.ndarray) and a.dtype.names:
        return {n: a[n] for n in a.dtype.names}
    return dict(a)


def _hist(x, lo, hi, n):
    """A compact histogram, JSON-ready."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    h, e = np.histogram(x, bins=np.linspace(lo, hi, n + 1))
    return dict(lo=float(lo), hi=float(hi), bin=float((hi - lo) / n),
                counts=[int(c) for c in h])


def _profile(x, y, lo, hi, n):
    """Median of y in n bins of x -- for spotting drift, not for fitting."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return dict(centres=[], median=[], n=[])
    e = np.linspace(lo, hi, n + 1)
    idx = np.clip(np.digitize(x, e) - 1, 0, n - 1)
    c, md, cnt = [], [], []
    for i in range(n):
        s = idx == i
        if s.sum() < 20:
            continue
        c.append(float(0.5 * (e[i] + e[i + 1])))
        md.append(float(np.median(y[s])))
        cnt.append(int(s.sum()))
    return dict(centres=c, median=md, n=cnt)


def _level(value, warn, fail, worse='high'):
    if value is None or not np.isfinite(value):
        return 'NA'
    bad = (value >= fail) if worse == 'high' else (value <= fail)
    if bad:
        return 'FAIL'
    warn_ = (value >= warn) if worse == 'high' else (value <= warn)
    return 'WARN' if warn_ else 'PASS'


def _shadow_flag(gid, t, amp, ratio=SHADOW_RATIO, t_hold=SHADOW_HOLD_NS):
    """True where a hit sits in the shadow of a bigger earlier hit on its channel.

    The plastics ring: every large pulse is followed by a train of real
    secondary pulses out to ~1 us (pss_ringing/report.html), and this is the
    adopted cut for them -- amp < ratio x any amp on the same channel within
    t_hold BEFORE the hit. Same walk-back as pss_ringing/afterpulse_flag.py,
    restricted to what the slim window can see: for the LATE tail the parent
    is always in the window (an after-pulse at +dt has its parent within
    t_hold before it, i.e. above dt - t_hold >= -1 us), so in-window lookback
    is complete there; only early-side hits with pre-window parents are
    missed, and the tail is one-sided late.

    `gid` is any int64 key separating (trigger, detector, channel, window).
    Arrays may come in any order; the flag returns in that order.
    """
    gid = np.asarray(gid)
    t = np.asarray(t, np.float64)
    amp = np.asarray(amp, np.float64)
    order = np.lexsort((t, gid))
    g, tt, aa = gid[order], t[order], amp[order]
    out = np.zeros(t.size, bool)
    active = np.arange(1, t.size)
    for k in range(1, t.size):
        j = active - k
        keep = j >= 0
        active, j = active[keep], j[keep]
        if active.size == 0:
            break
        inwin = (g[j] == g[active]) & (tt[active] - tt[j] <= t_hold) \
            & (tt[active] - tt[j] > 0)
        active, j = active[inwin], j[inwin]
        if active.size == 0:
            break
        out[active[aa[active] < ratio * aa[j]]] = True
    res = np.empty_like(out)
    res[order] = out
    return res


def analyse(d: Path) -> SegmentQA:
    """Everything, for one ntof_hits directory."""
    root = sorted(d.glob('ntof_hits_*.root'))
    if not root:
        raise FileNotFoundError(f'no ntof_hits_*.root in {d}')
    cal = json.loads((d / 'calibration.json').read_text())
    qa = json.loads((d / 'qa.json').read_text())
    prov = json.loads((d / 'provenance.json').read_text())

    with uproot.open(root[0]) as f:
        ev = _arrays(f, 'events')
        hits = _arrays(f, 'hits')
        bunches = _arrays(f, 'bunches')

    out = SegmentQA()
    out.segment = dict(
        dream_run=prov.get('dream_run'), dream_subrun=prov.get('dream_subrun'),
        ntof_run=prov.get('ntof_run'), processing=prov.get('ntof_processing'),
        file=root[0].name, size_mb=round(root[0].stat().st_size / 1e6, 1),
        dir=str(d))

    phys = ev['is_flash'] == 0
    matched = (ev['matched'] == 1) & phys
    resid = ev['residual_ns'][matched]
    checks: list[Check] = []

    # ---------------------------------------------------------------- clock
    K, T0 = cal['K'], cal['T0_ns']
    arm_off = cal.get('arm_offset_ns', {})
    out.clock = dict(K=K, T0_ns=T0, arm_offset_ns=arm_off,
                     iters=cal.get('fit', {}).get('iters', []),
                     n_bunches_fitted=cal.get('n_bunches_fitted'),
                     slim_ns=cal.get('slim_ns'), accept_ns=cal.get('accept_ns'))
    checks.append(Check(
        'K in physical range', 'PASS' if TH['k_lo'] <= K <= TH['k_hi'] else 'FAIL',
        K, f'clock rate ratio {K:.6e}',
        f'{TH["k_lo"]:.2e}..{TH["k_hi"]:.2e}'))

    # ------------------------------------------------------------ bootstrap
    boot = cal.get('fit', {}).get('bootstrap')
    if boot:
        out.bootstrap = boot
        lvl = _level(-boot['snr'], -TH['boot_snr_warn'], -TH['boot_snr_fail'])
        checks.append(Check(
            'coarse peak above floor', lvl, boot['snr'],
            f'tallest bin {boot["counts"]:,} at {boot["peak_ns"]:+.0f} ns over '
            f'a floor of {boot["floor"]:.0f}/bin',
            f'S/N >= {TH["boot_snr_warn"]:g}'))
    else:
        checks.append(Check(
            'coarse peak above floor', 'NA', None,
            'no bootstrap record -- produced before the coarse search existed, '
            'so the fit relied on a seed and may have converged by luck', ''))

    # ---------------------------------------------------------------- match
    eff, effcv = qa['efficiency'], qa['efficiency_cv']
    gap = eff - effcv
    out.match = dict(
        efficiency=eff, efficiency_cv=effcv, cv_gap=gap,
        accidental=qa['accidental'], purity=qa['purity'],
        n_events=qa['n_events'], n_physics=qa['n_physics'],
        n_flash=qa['n_flash'],
        residual_hist=_hist(resid, -C.ACCEPT_NS, C.ACCEPT_NS, 50),
        residual_rms=float(np.std(resid)) if resid.size else float('nan'),
        residual_mean=float(np.mean(resid)) if resid.size else float('nan'),
        residual_mad=(float(np.median(np.abs(resid - np.median(resid))))
                      if resid.size else float('nan')))

    checks.append(Check('match efficiency',
                        _level(eff, TH['efficiency_warn'], TH['efficiency_fail'],
                               worse='low'),
                        eff, f'{eff:.2%} of physics triggers matched',
                        f'>= {TH["efficiency_warn"]:.0%}'))
    checks.append(Check('accidental rate',
                        _level(qa['accidental'], TH['accidental_warn'],
                               TH['accidental_fail']),
                        qa['accidental'],
                        f'{qa["accidental"]:.4%} from the +100 us control',
                        f'<= {TH["accidental_warn"]:.1%}'))
    checks.append(Check('cross-validation gap',
                        _level(gap, TH['cv_gap_warn'], TH['cv_gap_fail']),
                        gap,
                        f'in-sample {eff:.2%} vs held-out {effcv:.2%} '
                        f'({gap*100:+.3f} pts)',
                        f'<= {TH["cv_gap_warn"]*100:.1f} pts'))
    checks.append(Check('residual centred',
                        _level(abs(out.match['residual_mean']),
                               TH['resid_mean_warn'], TH['resid_mean_fail']),
                        out.match['residual_mean'],
                        f'mean matched residual '
                        f'{out.match["residual_mean"]:+.2f} ns',
                        f'|mean| <= {TH["resid_mean_warn"]:g} ns'))
    checks.append(Check('residual width',
                        _level(out.match['residual_rms'], TH['resid_rms_warn'],
                               TH['resid_rms_fail']),
                        out.match['residual_rms'],
                        f'RMS {out.match["residual_rms"]:.2f} ns inside the '
                        f'+-{C.ACCEPT_NS:g} ns accept window',
                        f'<= {TH["resid_rms_warn"]:g} ns'))

    # per-arm: efficiency and offset balance
    per_arm = {}
    for i, a in enumerate(ARMS):
        s = matched & (ev['arm'] == i)
        r = ev['residual_ns'][s]
        per_arm[a] = dict(
            n=int(s.sum()),
            frac=float(s.sum() / max(matched.sum(), 1)),
            residual_mean=float(np.mean(r)) if r.size else float('nan'),
            offset_ns=float(arm_off.get(a, float('nan'))),
            offset_vs_ref=float(arm_off.get(a, float('nan')) - REF_ARM[a]))
    out.match['per_arm'] = per_arm
    dev = max((abs(v['offset_vs_ref']) for v in per_arm.values()
               if np.isfinite(v['offset_vs_ref'])), default=float('nan'))
    checks.append(Check(
        'arm offsets vs reference',
        _level(dev, TH['arm_dev_warn'], TH['arm_dev_fail']), dev,
        '  '.join(f'{a} {per_arm[a]["offset_ns"]:+.1f}'
                  f'({per_arm[a]["offset_vs_ref"]:+.1f})' for a in ARMS)
        + ' ns vs run_79',
        f'max |dev| <= {TH["arm_dev_warn"]:g} ns'))
    # A wrong per-arm offset hides from every global check: the arms average
    # to zero, the efficiency barely moves inside +-25 ns, and the offsets
    # themselves can sit inside the vs-reference tolerance. What cannot hide
    # is the matched residuals of that arm centring somewhere other than zero.
    arm_resid = max((abs(v['residual_mean']) for v in per_arm.values()
                     if np.isfinite(v['residual_mean']) and v['n'] >= 100),
                    default=float('nan'))
    checks.append(Check(
        'per-arm residuals centred',
        _level(arm_resid, TH['arm_resid_warn'], TH['arm_resid_fail']),
        arm_resid,
        '  '.join(f'{a} {per_arm[a]["residual_mean"]:+.1f}' for a in ARMS)
        + ' ns mean matched residual per arm',
        f'max |mean| <= {TH["arm_resid_warn"]:g} ns'))

    # ------------------------------------------------- drift across the segment
    t = ev['t_dream_ns'][matched]
    out.match['residual_vs_t'] = _profile(t / 1e6, resid, 0, 85, 34)
    b = ev['bunch'][matched].astype(float)
    if b.size:
        out.match['residual_vs_bunch'] = _profile(
            b, resid, float(b.min()), float(b.max()) + 1, 40)
        eb = ev['bunch'][phys].astype(float)
        mb = matched[phys].astype(float)
        out.match['efficiency_vs_bunch'] = _profile(
            eb, mb, float(eb.min()), float(eb.max()) + 1, 40)
    prof = out.match.get('residual_vs_t', {})
    drift = (max(prof['median']) - min(prof['median'])
             if prof.get('median') else float('nan'))
    checks.append(Check(
        'no residual drift in time-since-flash',
        _level(drift, TH['drift_warn'], TH['drift_fail']), drift,
        f'median residual spans {drift:.1f} ns across 0-80 ms '
        f'(the per-bunch fit should have flattened this)',
        f'span <= {TH["drift_warn"]:g} ns'))

    # -------------------------------------------------------------- per-bunch
    if len(bunches) and 'da_ns' in bunches:
        fit = bunches['fitted'].astype(bool)
        da, dk = bunches['da_ns'][fit], bunches['dk'][fit]
        # 'bunches fitted' is a question about the CLOCK FIT, so it must be
        # asked of the bunches that had beam. Measured 2026-08-10 over the whole
        # first campaign: 1,658 of the 1,660 bunches that never got their own
        # (da_b, dk_b) were PS pulses that delivered no protons, and in 114 of
        # 116 segments the unfitted count equalled the empty-pulse count
        # exactly. Judged over all bunches, this check reports beam
        # availability -- an accelerator fact the QA cannot act on -- and its
        # four campaign WARNs were exactly that. Availability is reported
        # separately below, and never as a PASS/FAIL.
        if 'has_beam' in bunches:
            beam = bunches['has_beam'].astype(bool)
            fit_scope, scope_txt = beam, 'bunches that had beam'
        else:
            beam = np.ones(fit.size, bool)
            fit_scope, scope_txt = beam, 'bunches (pre-2026-08-10 file: no ' \
                                         'beam record, empty pulses included)'
        frac = float(fit[fit_scope].sum() / max(int(fit_scope.sum()), 1))
        inten = bunches.get('intensity_e10')
        out.perbunch = dict(
            n_bunches=int(fit.size), n_fitted=int(fit.sum()), frac_fitted=frac,
            n_bunches_beam=int(beam.sum()), n_bunches_empty=int((~beam).sum()),
            beam_availability=float(beam.mean()) if fit.size else float('nan'),
            has_beam_column='has_beam' in bunches,
            parasitic_fraction=(float(np.mean(np.asarray(inten)[beam]
                                              < C.PARASITIC_E10))
                                if inten is not None and beam.any()
                                else float('nan')),
            intensity_median_e10=(float(np.median(np.asarray(inten)[beam]))
                                  if inten is not None and beam.any()
                                  else float('nan')),
            da_rms=float(np.std(da)) if da.size else float('nan'),
            dk_rms_ppm=float(np.std(dk) * 1e6) if dk.size else float('nan'),
            da_hist=_hist(da, -60, 60, 48),
            dk_hist=_hist(dk * 1e6, -5, 5, 40),
            bunch=[int(x) for x in bunches['bunch'][fit]],
            da=[round(float(x), 3) for x in da],
            dk_ppm=[round(float(x) * 1e6, 4) for x in dk],
            n_core=[int(x) for x in bunches['n_core'][fit]])
        checks.append(Check(
            'bunches fitted',
            _level(frac, TH['bunch_fit_frac_warn'], TH['bunch_fit_frac_fail'],
                   worse='low'), frac,
            f'{int(fit[fit_scope].sum()):,} of {int(fit_scope.sum()):,} '
            f'{scope_txt} got their own correction'
            + (f'; {int((~beam).sum()):,} empty pulses excluded '
               f'(beam availability {beam.mean():.1%})'
               if (~beam).any() else ''),
            f'>= {TH["bunch_fit_frac_warn"]:.0%}'))
        # The filter is load-bearing, so assert it ran: no event in the file may
        # belong to a bunch the beam record says was empty. A no-beam trigger is
        # detector background whose t_since_flash is referenced to another
        # background trigger, and it can never match -- 0 of 2,764 did over the
        # first campaign.
        if 'has_beam' in bunches:
            empty_b = bunches['bunch'][~beam]
            n_leak = int(np.isin(ev['bunch'], empty_b).sum()) if empty_b.size \
                else 0
            checks.append(Check(
                'no-beam pulses filtered out',
                'PASS' if n_leak == 0 else 'FAIL', n_leak,
                f'{n_leak:,} trigger(s) from the {empty_b.size:,} empty pulses '
                f'survived into the events tree', '== 0'))
            # ...and that what was dropped was really no-beam. DREAM's gate
            # opens on the PS timing regardless of protons, but only background
            # walks through it: 1-2 triggers against ~92 in a beam bunch. A
            # dropped pulse holding a FULL burst means bursts are landing on
            # the wrong bunches -- the beam record is not what is wrong. This is
            # the only check that can see a mis-assigned join from the written
            # file alone, and it costs one ratio.
            nt = bunches['n_triggers']
            if (~beam).any() and beam.any():
                ratio = float(np.median(nt[~beam])
                              / max(np.median(nt[beam]), 1))
                out.perbunch['empty_trigger_ratio'] = ratio
                checks.append(Check(
                    'dropped pulses look like no beam',
                    _level(ratio, C.EMPTY_TRIGGER_RATIO_WARN,
                           C.EMPTY_TRIGGER_RATIO_FAIL), ratio,
                    f'the {int((~beam).sum()):,} dropped pulses held '
                    f'{np.median(nt[~beam]):.0f} triggers each against '
                    f'{np.median(nt[beam]):.0f} in a beam bunch'
                    + ('' if ratio < C.EMPTY_TRIGGER_RATIO_WARN else
                       ' -- a no-beam pulse cannot produce a full DREAM burst, '
                       'so suspect the burst-to-bunch assignment'),
                    f'<= {C.EMPTY_TRIGGER_RATIO_WARN:g} of the beam median'))
            else:
                checks.append(Check(
                    'dropped pulses look like no beam', 'NA', None,
                    'no empty pulses in this segment' if beam.all()
                    else 'no beam bunches to compare against', ''))
        else:
            checks.append(Check(
                'no-beam pulses filtered out', 'NA', None,
                'no beam record in the bunches tree -- file predates the '
                'empty-pulse filter, so it may carry no-beam triggers '
                '(campaign measurement: 0.05 % of triggers, all unmatched)',
                ''))
        checks.append(Check(
            'per-bunch offset scatter',
            _level(out.perbunch['da_rms'], TH['da_rms_warn'], TH['da_rms_fail']),
            out.perbunch['da_rms'],
            f'da RMS {out.perbunch["da_rms"]:.2f} ns between bunches',
            f'<= {TH["da_rms_warn"]:g} ns'))
        checks.append(Check(
            'per-bunch rate scatter',
            _level(out.perbunch['dk_rms_ppm'], TH['dk_rms_ppm_warn'],
                   TH['dk_rms_ppm_fail']), out.perbunch['dk_rms_ppm'],
            f'dk RMS {out.perbunch["dk_rms_ppm"]:.2f} ppm between bunches',
            f'<= {TH["dk_rms_ppm_warn"]:g} ppm'))
    else:
        checks.append(Check('bunches fitted', 'NA', None,
                            'no per-bunch columns -- file predates them', ''))
        checks.append(Check('no-beam pulses filtered out', 'NA', None,
                            'no per-bunch columns -- file predates them', ''))
        checks.append(Check('dropped pulses look like no beam', 'NA', None,
                            'no per-bunch columns -- file predates them', ''))

    # ------------------------------------------------------------------ hits
    det = {t_: i for i, t_ in enumerate(C.SCINT_TREES)}
    sig = hits['is_control'] == 0
    fam_stats = {}
    for fam in FAMILIES:
        ids = [det[f'{fam}{a}'] for a in ARMS]
        m = np.isin(hits['det'], ids)
        ms, mc = m & sig, m & ~sig
        fam_stats[fam] = dict(
            n_signal=int(ms.sum()), n_control=int(mc.sum()),
            per_trigger=float(ms.sum() / max(qa['n_physics'], 1)),
            dt_signal=_hist(hits['dt_ns'][ms], -cal['slim_ns'],
                            cal['slim_ns'], 60),
            dt_control=_hist(hits['dt_ns'][mc], -cal['slim_ns'],
                             cal['slim_ns'], 60))
    out.hits = dict(
        families=fam_stats, n_total=int(hits['eventId'].size),
        n_signal=int(sig.sum()), n_control=int((~sig).sum()),
        per_trigger=qa.get('hits_per_trigger'),
        dt_p50_p90_p99=[float(v) for v in np.percentile(
            np.abs(hits['dt_ns'][sig]), [50, 90, 99])] if sig.sum() else [])

    # ------------------------------------------------- the plastics ring
    # Every large plastic pulse is followed by real secondary pulses out to
    # ~1 us (pss_ringing/), and they are most of the PSS content of the slim
    # window. Classify them with the in-window shadow flag, then ask the two
    # questions the +-25 ns analysis slice depends on:
    #   1. is the late tail RINGING (explained by the flag), rather than
    #      coincidence yield the window is mis-handling;
    #   2. per matched trigger, does the LARGEST plastic pulse on the
    #      trigger's own arm land inside +-25 ns.
    pss_ids = np.array([det[f'PSS{a}'] for a in ARMS])
    pm = np.isin(hits['det'], pss_ids)
    fl = np.zeros(int(pm.sum()), bool)
    ring = dict(late_excess=None, late_removed=None, core_excess=None,
                core_cost=None, flagged_frac=None)
    if pm.sum():
        if 'shadow_amp' in hits:
            # The slim stored the full-stream shadow (parents seen even when
            # they fall outside the window) -- use it.
            fl = (hits['amp_0'][pm]
                  < SHADOW_RATIO * hits['shadow_amp'][pm])
        else:
            # Older file: recompute in-window, which is complete for the late
            # tail (an after-pulse at +dt always has its parent inside).
            # channel key: (trigger, tree, channel, signal-vs-control window)
            gid = (((hits['eventId'][pm].astype(np.int64) * 12
                     + hits['det'][pm]) * 64
                    + hits['detn'][pm].astype(np.int64)) * 2
                   + hits['is_control'][pm])
            fl = _shadow_flag(gid, hits['dt_ns'][pm].astype(np.float64),
                              hits['amp_0'][pm].astype(np.float64))
        d_, c_ = hits['dt_ns'][pm], hits['is_control'][pm].astype(bool)

        def _ex(keep, lo_, hi_):
            s_ = (d_ >= lo_) & (d_ < hi_) & keep
            return int((s_ & ~c_).sum()) - int((s_ & c_).sum())
        allh = np.ones(d_.size, bool)
        late0 = _ex(allh, 150.0, cal['slim_ns'])
        late1 = _ex(~fl, 150.0, cal['slim_ns'])
        core0 = _ex(allh, -25.0, 25.0)
        core1 = _ex(~fl, -25.0, 25.0)
        ring = dict(
            late_excess=late0,
            late_removed=(1.0 - late1 / late0) if late0 > 0 else None,
            core_excess=core0,
            core_cost=(1.0 - core1 / core0) if core0 > 0 else None,
            flagged_frac=float(fl.mean()))
    out.hits['pss_ringing'] = ring
    if ring['late_excess'] is not None \
            and ring['late_excess'] >= TH['ringing_min_excess']:
        rv = ring['late_removed']
        checks.append(Check(
            'PSS late tail is ringing',
            _level(rv, TH['ringing_removed_warn'], 0.0, worse='low'), rv,
            f'shadow flag removes {rv:.1%} of the {ring["late_excess"]:,} '
            f'excess hits at 150-{cal["slim_ns"]:g} ns, for '
            f'{ring["core_cost"]:.1%} of the +-25 ns core',
            f'>= {TH["ringing_removed_warn"]:.0%} removed'))
    else:
        checks.append(Check(
            'PSS late tail is ringing', 'NA',
            ring['late_excess'],
            f'late excess {ring["late_excess"]} below the '
            f'{TH["ringing_min_excess"]} counts needed to classify', ''))

    # Per matched trigger: the largest-amplitude plastic hit on the trigger's
    # own arm. NOT the earliest -- in a +-1 us window at 720 kHz singles the
    # earliest hit is almost always an unrelated single (31 % inside +-25 ns
    # against 90 % for largest; pss_ringing/report_veto.html).
    primary = dict(n=None, within_core=None, median_ns=None)
    if pm.sum() and matched.any():
        ev_sorted = np.argsort(ev['eventId'])
        eid_s = ev['eventId'][ev_sorted]
        pos = np.searchsorted(eid_s, hits['eventId'][pm])
        pos = np.clip(pos, 0, eid_s.size - 1)
        okj = eid_s[pos] == hits['eventId'][pm]
        h_arm = np.full(pm.sum(), -1, np.int64)
        h_arm[okj] = ev['arm'][ev_sorted][pos[okj]]
        h_matched = np.zeros(pm.sum(), bool)
        h_matched[okj] = matched[ev_sorted][pos[okj]]
        own = (h_matched & (hits['is_control'][pm] == 0)
               & (hits['det'][pm] - pss_ids.min() == h_arm))
        if own.sum():
            eids = hits['eventId'][pm][own].astype(np.int64)
            amps = hits['amp_0'][pm][own].astype(np.float64)
            dts = hits['dt_ns'][pm][own].astype(np.float64)
            o = np.lexsort((-amps, eids))
            first = np.ones(o.size, bool)
            first[1:] = eids[o][1:] != eids[o][:-1]
            fd = dts[o][first]
            primary = dict(n=int(fd.size),
                           within_core=float(np.mean(np.abs(fd) <= C.ACCEPT_NS)),
                           median_ns=float(np.median(fd)))
    out.match['pss_primary'] = primary
    if primary['n']:
        checks.append(Check(
            'plastic primary within accept',
            _level(primary['within_core'], TH['pss_primary_warn'],
                   TH['pss_primary_fail'], worse='low'),
            primary['within_core'],
            f'largest plastic pulse on the trigger arm: '
            f'{primary["within_core"]:.1%} of {primary["n"]:,} triggers within '
            f'+-{C.ACCEPT_NS:g} ns (median {primary["median_ns"]:+.1f} ns)',
            f'>= {TH["pss_primary_warn"]:.0%}'))
    else:
        checks.append(Check('plastic primary within accept', 'NA', None,
                            'no matched triggers with plastic hits on their '
                            'own arm', ''))

    # flash triggers must carry no hits at all
    flash_ids = set(ev['eventId'][~phys].tolist())
    n_flash_hits = (int(np.isin(hits['eventId'], list(flash_ids)).sum())
                    if flash_ids else 0)
    checks.append(Check(
        'flash triggers carry no hits',
        'PASS' if n_flash_hits <= TH['flash_hits_fail'] else 'FAIL',
        n_flash_hits,
        f'{n_flash_hits:,} hits attached to flash triggers '
        f'({len(flash_ids):,} flash events)', '== 0'))

    # Window containment, per family, against the control -- SIDED.
    #
    # The obvious test -- "is the kept dt still RISING at the edge?" -- was
    # nearly useless (a peak wider than the window passes it; measured 0.94
    # while 23 % of the plastic was being cut). Its replacement, |dt| edge
    # excess over the core with a 3 sigma requirement, had its own false
    # alarm: the +100 us control under- or over-states the local accidental
    # floor by a little (the singles rate varies across the 80 ms), which
    # leaves a small FLAT pedestal across the whole window. On big-statistics
    # segments that pedestal passes 3 sigma easily and lit up 45 of 116
    # campaign segments -- all on LIQ, and the fleet total was early +14,491
    # against late +15,623: SYMMETRIC, which no truncated coincidence is.
    #
    # Truncation of a real tail is one-sided (the PSS ringing tail is late by
    # 500x). So flag on the ASYMMETRY of the edge excess: (late - early)
    # against the core density. A rate-bias pedestal is common mode between
    # the two edges and cancels; a truncated tail is not. PSS is judged on
    # ringing-CLEANED hits -- the ringing past +-1 us is known, cut by the
    # shadow flag, and not coincidence yield.
    #
    # The null is EMPIRICAL, not Poisson. Measured fleet-wide (2026-08-09),
    # the per-decile asymmetries in the mid band (0.3-0.7 W, no coincidence
    # content for any family after the flag) scatter 2-3x wider than Poisson
    # -- real pedestal structure spread across the whole window. Against
    # sqrt(Poisson) the LIQ edge lit up 18 of 119 segments with RANDOM sign;
    # against max(Poisson, mid-band RMS) the fleet |z| tops out at 3.2, still
    # sign-random. So the null is the mid band, and the gate is z >= 4: above
    # everything the healthy fleet produces, far below what truncation gives
    # (the +-150 ns mistake was a coherent 22x one-sided tail).
    W = cal['slim_ns']
    contain = {}
    worst_fam, worst = None, 0.0
    keep_all = np.ones(hits['det'].size, bool)
    keep_all[np.flatnonzero(pm)[fl]] = False
    for fam in FAMILIES:
        ids = [det[f'{fam}{a}'] for a in ARMS]
        m = np.isin(hits['det'], ids) & keep_all
        d_ = hits['dt_ns']

        def _ex2(a_, b_):
            s_ = int(((d_ >= a_) & (d_ < b_) & m & sig).sum())
            c_ = int(((d_ >= a_) & (d_ < b_) & m & ~sig).sum())
            return s_ - c_, s_ + c_
        core, _ = _ex2(-0.1 * W, 0.1 * W)
        eg_e, n_e = _ex2(-W, -0.9 * W)
        eg_l, n_l = _ex2(0.9 * W, W)
        mids = []
        for i in range(3, 7):
            late_i, _ = _ex2(0.1 * W * i, 0.1 * W * (i + 1))
            early_i, _ = _ex2(-0.1 * W * (i + 1), -0.1 * W * i)
            mids.append(late_i - early_i)
        pk = core / 2.0                   # core is 0.2 W, edges 0.1 W each
        asym = eg_l - eg_e
        sig_pois = np.sqrt(max(n_e + n_l, 1))
        sig_mid = float(np.sqrt(np.mean(np.square(mids)))) if mids else 0.0
        sigma = max(sig_pois, sig_mid)
        signif = asym / sigma
        rat = abs(asym) / pk if pk > 0 else float('nan')
        contain[fam] = dict(
            peak_excess_per_decile=pk, early_excess=eg_e, late_excess=eg_l,
            edge_asymmetry=asym, mid_asym_rms=sig_mid,
            poisson_sigma=float(sig_pois), asym_significance=float(signif),
            ratio=rat)
        # The same dt histograms as above, with the ringing removed -- what is
        # left of a family once the after-pulses and the 81-82 ns cable echo
        # are flagged out (`amp_0 < 0.05 x shadow_amp` on the same channel
        # within 1 us). Stored beside the raw ones so the two can be drawn
        # together: on PSS the late tail collapses onto the accidental floor,
        # and on WAL/LIQ nothing moves, which is the control that says the flag
        # is not just eating hits.
        fam_stats[fam]['dt_signal_ring_cut'] = _hist(
            hits['dt_ns'][m & sig], -W, W, 60)
        fam_stats[fam]['dt_control_ring_cut'] = _hist(
            hits['dt_ns'][m & ~sig], -W, W, 60)
        fam_stats[fam]['n_signal_ring_cut'] = int((m & sig).sum())
        fam_stats[fam]['n_control_ring_cut'] = int((m & ~sig).sum())
        if np.isfinite(rat) and rat > worst and abs(signif) >= 4.0:
            worst_fam, worst = fam, rat
    out.hits['containment'] = contain
    checks.append(Check(
        'coincidence contained in slim window',
        _level(worst, 0.05, 0.50), worst,
        '  '.join(f'{f} {contain[f]["ratio"]:.3f}'
                  f'({contain[f]["asym_significance"]:+.0f}s)'
                  for f in FAMILIES)
        + f' = one-sided edge/core excess density at +-{W:g} ns '
          f'(PSS after the ringing flag; z against the mid-band null)'
        + (f'; {worst_fam} has a significant one-sided edge, so real '
           f'coincidences are being cut' if worst > 0.05 else
           '; nothing one-sided and significant at the edge'),
        '|late-early|/core <= 0.05 at z >= 4'))

    out.checks = [asdict(c) for c in checks]
    lv = [c.level for c in checks]
    out.verdict = ('FAIL' if 'FAIL' in lv
                   else 'WARN' if 'WARN' in lv else 'PASS')
    return out


COLOUR = dict(PASS='\033[32m', WARN='\033[33m', FAIL='\033[31m',
              NA='\033[90m', reset='\033[0m')


def report(q: SegmentQA, colour=True) -> str:
    def c(level, s):
        return f'{COLOUR[level]}{s}{COLOUR["reset"]}' if colour else s
    s = q.segment
    L = [f'{s["dream_run"]}/{s["dream_subrun"]} x n_TOF {s["ntof_run"]}'
         f'   [{s["processing"]}]',
         f'{s["file"]}  {s["size_mb"]} MB',
         '',
         f'  K  {q.clock["K"]:.6e}      T0 {q.clock["T0_ns"]:+9.2f} ns',
         f'  efficiency {q.match["efficiency"]:.3%} '
         f'(held-out {q.match["efficiency_cv"]:.3%})   '
         f'accidental {q.match["accidental"]:.4%}']
    # The beam, reported and never judged: availability is the PS's business,
    # and the parasitic mix is what most of the fleet's efficiency spread is
    # (r = -0.82 over the first campaign).
    pb = q.perbunch
    if pb.get('has_beam_column'):
        L.append(f'  beam       {pb["beam_availability"]:.1%} availability '
                 f'({pb["n_bunches_empty"]:,} empty of {pb["n_bunches"]:,} '
                 f'pulses)   {pb["parasitic_fraction"]:.0%} parasitic, '
                 f'median {pb["intensity_median_e10"]:.0f}e10')
    L.append('')
    for ch in q.checks:
        mark = dict(PASS='ok  ', WARN='WARN', FAIL='FAIL', NA='--  ')[ch['level']]
        L.append(f'  {c(ch["level"], mark)}  {ch["name"]:<34} {ch["detail"]}')
    L += ['', f'  VERDICT: {c(q.verdict, q.verdict)}']
    return '\n'.join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+', type=Path,
                    help='ntof_hits directories (or parents to search)')
    ap.add_argument('--json', action='store_true',
                    help='write clock_qa.json next to each slim')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    targets = []
    for d in a.dirs:
        targets += ([d] if list(d.glob('ntof_hits_*.root'))
                    else sorted(p.parent for p in
                                d.rglob('ntof_hits_*.root')))
    if not targets:
        print('no slim files found')
        return 2

    worst, n = 'PASS', {'PASS': 0, 'WARN': 0, 'FAIL': 0}
    broken = []
    for d in targets:
        # One unreadable directory must not cost the other 118 their verdicts.
        # It is still an error -- reported at the end and in the exit code --
        # but a sweep that aborts on the first bad file does not sweep.
        try:
            q = analyse(d)
        except Exception as e:                                  # noqa: BLE001
            broken.append(f'{d}: {type(e).__name__}: {e}')
            print(f'  !! {broken[-1]}')
            continue
        n[q.verdict] = n.get(q.verdict, 0) + 1
        if q.verdict == 'FAIL' or (q.verdict == 'WARN' and worst != 'FAIL'):
            worst = q.verdict
        if not a.quiet:
            print(report(q)); print('-' * 72)
        if a.json:
            (d / 'clock_qa.json').write_text(json.dumps(asdict(q), indent=1))
    print(f'{len(targets)} segment(s): '
          f'{n["PASS"]} pass, {n["WARN"]} warn, {n["FAIL"]} fail'
          + (f', {len(broken)} UNREADABLE' if broken else ''))
    return 0 if (n['FAIL'] == 0 and not broken) else 1


if __name__ == '__main__':
    raise SystemExit(main())
