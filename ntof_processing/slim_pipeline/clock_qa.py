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
    # 953 of 961 bunches fitted on run_77 = 99.2 %.
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
    # flash triggers are written with no hits by construction.
    flash_hits_fail=0,
)
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
        frac = float(fit.sum() / max(fit.size, 1))
        out.perbunch = dict(
            n_bunches=int(fit.size), n_fitted=int(fit.sum()), frac_fitted=frac,
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
            f'{fit.sum():,} of {fit.size:,} bunches got their own correction',
            f'>= {TH["bunch_fit_frac_warn"]:.0%}'))
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

    # Window containment, per family, against the control.
    #
    # The obvious test -- "is the kept dt still RISING at the edge?" -- is the
    # one this replaces, and it is nearly useless: a coincidence peak WIDER
    # than the window falls away slowly and passes it comfortably. Measured on
    # the reference pair it returned 0.94 ("flat, window is wide enough") while
    # 23 % of the plastic yield was in fact being cut off.
    #
    # What works is comparing like with like: background-subtracted excess in
    # the outer decile of the window against the same width at the centre. Both
    # are 0.1 W wide, so the ratio is a density ratio and needs no scaling, and
    # subtracting the +100 us control removes the accidental floor that made
    # the naive version look flat.
    W = cal['slim_ns']
    contain = {}
    worst_fam, worst = None, 0.0
    for fam in FAMILIES:
        ids = [det[f'{fam}{a}'] for a in ARMS]
        m = np.isin(hits['det'], ids)
        d_ = np.abs(hits['dt_ns'])
        core = d_ <= 0.1 * W
        edge = (d_ > 0.9 * W) & (d_ <= W)
        pk = int((m & sig & core).sum()) - int((m & ~sig & core).sum())
        eg = int((m & sig & edge).sum()) - int((m & ~sig & edge).sum())
        rat = eg / pk if pk > 0 else float('nan')
        contain[fam] = dict(peak_excess=pk, edge_excess=eg, ratio=rat)
        if np.isfinite(rat) and rat > worst:
            worst_fam, worst = fam, rat
    out.hits['containment'] = contain
    checks.append(Check(
        'coincidence contained in slim window',
        _level(worst, 0.05, 0.50), worst,
        '  '.join(f'{f} {contain[f]["ratio"]:.3f}' for f in FAMILIES)
        + f' = edge/peak excess density at +-{W:g} ns'
        + (f'; {worst_fam} is still substantial at the edge, so real '
           f'coincidences are being cut' if worst > 0.05 else ''),
        'edge/peak <= 0.05'))

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
         f'accidental {q.match["accidental"]:.4%}',
         '']
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
    for d in targets:
        q = analyse(d)
        n[q.verdict] = n.get(q.verdict, 0) + 1
        if q.verdict == 'FAIL' or (q.verdict == 'WARN' and worst != 'FAIL'):
            worst = q.verdict
        if not a.quiet:
            print(report(q)); print('-' * 72)
        if a.json:
            (d / 'clock_qa.json').write_text(json.dumps(asdict(q), indent=1))
    print(f'{len(targets)} segment(s): '
          f'{n["PASS"]} pass, {n["WARN"]} warn, {n["FAIL"]} fail')
    return 0 if n['FAIL'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
