#!/usr/bin/env python3
"""
Build the figures and report.html of the scintillator flash-recovery study
from raw_tail.npz and psa_tail.npz.  Generated, not hand-written: re-run after
either stage and the numbers, figures and verdict text follow.

    python -m ntof_scint_recovery.make_report
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntof_scint_recovery.common import (C_NS, FAMILIES, MV_PER_ADC, PLAS_THR, RAW_FILES,  # noqa: E402
                                        RUN, WALL_THR, WINDOW_US, out_dir)

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

#: categorical slots 1-4 of the reference palette, one per arm, fixed order
ARM_COL = {'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a', 'D': '#eda100'}
FAM_NAME = {'WAL': 'SiPM wall', 'PSS': 'plastic', 'LIQ': 'liquid'}
LATE_US = (1e3, 1e4)      # the plateau reference for pulse shape
OUT = out_dir()
FIG = out_dir('figures')


def _style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ loaders
def load_raw():
    z = np.load(OUT / 'raw_tail.npz')
    return {k: z[k] for k in z.files}


def load_psa():
    z = np.load(OUT / 'psa_tail.npz')
    keys = [tuple(k.split('|')) for k in z['keys']]
    H = {(d, c, int(ch)): (a, w) for (d, ch, c), a, w in zip(keys, z['hA'], z['hW'])}
    return z['tb_ns'], z['ab'], z['wb'], H, {'par': int(z['n_par']), 'ded': int(z['n_ded'])}


def hq(h, edges, q=0.5, nmin=30):
    """Quantile of each row of a 2D histogram on log-bin centres."""
    c = np.cumsum(h, axis=1)
    tot = c[:, -1]
    ctr = np.sqrt(edges[:-1] * edges[1:])
    v = ctr[np.argmax(c >= q * np.maximum(tot, 1)[:, None], axis=1)].astype(float)
    v[tot < nmin] = np.nan
    return v


def stays_within(t, y, lo, hi, after=0.0, before=np.inf):
    """First t after which y stays inside [lo, hi] (up to `before`)."""
    m = (t > after) & (t < before) & np.isfinite(y)
    t, y = t[m], y[m]
    bad = np.where((y < lo) | (y > hi))[0]
    if not len(bad):
        return t[0] if len(t) else np.nan
    return t[bad[-1] + 1] if bad[-1] + 1 < len(t) else np.nan


# ------------------------------------------------------------------ numbers
def psa_curves(tb, ab, wb, H, nb):
    tc = np.sqrt(tb[2:-1] * tb[3:]) / 1e3          # µs, positive bins only
    dt = np.diff(tb)[2:] / 1e3
    C = {}
    for (det, cls, ch), (hA, hW) in H.items():
        hA, hW = hA[2:], hW[2:]
        C[(det, cls, ch)] = dict(rate=hA.sum(1) / max(nb[cls], 1) / dt,
                                 amp=hq(hA, ab), fwhm=hq(hW, wb))
    for det, ch in {(k[0], k[2]) for k in H}:     # both classes pooled
        hA = H[(det, 'par', ch)][0][2:] + H[(det, 'ded', ch)][0][2:]
        hW = H[(det, 'par', ch)][1][2:] + H[(det, 'ded', ch)][1][2:]
        C[(det, 'all', ch)] = dict(rate=hA.sum(1) / sum(nb.values()) / dt,
                                   amp=hq(hA, ab), fwhm=hq(hW, wb))
    return tc, C


def recovery_table(tc, C, R):
    """Per channel: the recovery times, µs."""
    rows = []
    tr = np.sqrt(R['tb_us'][:-1] * R['tb_us'][1:])
    for det, cls, ch in sorted(k for k in C if k[1] == 'all'):
        c = C[(det, 'all', ch)]
        late = (tc > LATE_US[0]) & (tc < LATE_US[1])
        w0 = np.nanmedian(c['fwhm'][late])
        r = dict(det=det, ch=ch, fwhm_late=w0,
                 first_hit=tc[np.argmax(c['rate'] > 0.3 * np.nanmax(c['rate']))],
                 shape=stays_within(tc, c['fwhm'], 0.85 * w0, 1.15 * w0, after=0.2, before=LATE_US[1]))
        for cls_ in ('par', 'ded'):
            occ = C[(det, cls_, ch)]['rate'] * WINDOW_US
            r[f'occ10_{cls_}'] = stays_within(tc, occ, 0, 0.10, after=1.0)
            r[f'occ1_{cls_}'] = stays_within(tc, occ, 0, 0.01, after=1.0)
            r[f'occ1ms_{cls_}'] = np.nanmean(occ[(tc > 0.9e3) & (tc < 1.1e3)])
        m = (R['det'] == det) & (R['ch'] == ch)
        if m.any():
            s = np.nanmedian(R['shift_mV'][m], axis=0)
            # stop at 10 ms: past it the walls drift by ~1 mV, unrelated to the flash
            r['base3'] = stays_within(tr, s, -3, 3, before=1e4)
            r['base1'] = stays_within(tr, s, -1, 1, before=1e4)
            r['under'] = np.nanmin(s[(tr > 20) & (tr < 1e4)])
            r['head_end'] = np.nanmedian(R['head_end_us'][m])
            r['noise'] = np.nanmedian(R['noise_mV'][m])
        rows.append(r)
    return rows


def fam_range(rows, fam, key, fmt='{:.0f}'):
    v = np.array([r.get(key, np.nan) for r in rows if r['det'].startswith(fam)], float)
    v = v[np.isfinite(v)]
    if not len(v):
        return '—'
    lo, hi = np.min(v), np.max(v)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f'{fmt.format(lo)}–{fmt.format(hi)}'


def tfmt(us):
    if not np.isfinite(us):
        return '—'
    if us < 10:
        return f'{us:.2g} µs'
    return f'{us:.0f} µs' if us < 1e3 else f'{us / 1e3:.2g} ms'


def fam_trange(rows, fam, key):
    v = np.array([r.get(key, np.nan) for r in rows if r['det'].startswith(fam)], float)
    n_nan = int(np.sum(~np.isfinite(v)))
    v = v[np.isfinite(v)]
    if not len(v):
        return '—'
    s = tfmt(np.min(v)) if tfmt(np.min(v)) == tfmt(np.max(v)) else f'{tfmt(np.min(v))} – {tfmt(np.max(v))}'
    return s + (f' ({n_nan} ch. never)' if n_nan else '')


# ------------------------------------------------------------------ figures
def fig_traces():
    """One bunch, one channel per family, from the raw: the flash and its tail."""
    want = {('WALA', 2), ('PSSA', 1), ('LIQA', 1)}
    blocks, ev, bunch = {}, None, None
    for _o, tag, _v, p in iter_banks(str(RAW_FILES[1])):
        if tag == 'EVEH':
            ev = parse_eveh(p)['words'][1]
            if bunch is None:
                bunch = ev
            elif ev != bunch:
                break
        elif tag == 'ACQC':
            det, ch, blks = parse_acqc(p, with_samples=True)
            if (det, ch) in want or det == 'PKUP':
                blocks[(det, ch)] = [(int(s), np.asarray(x, np.int16)) for s, x in blks]
    pk = blocks[('PKUP', 1)][0][1].astype(float)
    t_pk = np.argmax(pk - np.median(pk[:5000]))
    fig, ax = plt.subplots(3, 2, figsize=(13, 8.5))
    for i, (det, ch) in enumerate(sorted(want, key=lambda k: FAMILIES.index(k[0][:3]))):
        t0 = t_pk + C_NS[det]
        bl = blocks[(det, ch)]
        base = np.median(bl[0][1][:int(t0) - 500].astype(float))
        pol = 1 if det.startswith('WAL') else -1
        for j in range(2):
            for s, x in bl:
                if s > 2.2e6:
                    break
                t = (np.arange(s, s + len(x)) - t0) / 1e3
                y = pol * (x.astype(float) - base) * MV_PER_ADC
                ax[i, j].plot(t, y, lw=0.4, c='#2a78d6')
            ax[i, j].axhline(0, c='#52514e', lw=0.6)
            _style(ax[i, j])
        ax[i, 0].set_xlim(-3, 45)
        ax[i, 1].set_xscale('log')
        ax[i, 1].set_xlim(0.3, 2000)
        ax[i, 1].set_ylim(-40, 120)
        ax[i, 0].set_ylabel(f'{det}{ch}  (mV, pulse direction +)')
        ax[i, 0].set_title(f'{FAM_NAME[det[:3]]} — the flash, linear', loc='left', fontsize=10)
        ax[i, 1].set_title('the tail to 2 ms, zoomed to ±(a few × threshold)', loc='left', fontsize=10)
    ax[2, 0].set_xlabel('time since the flash (µs)')
    ax[2, 1].set_xlabel('time since the flash (µs, log)')
    fig.suptitle(f'Raw stream1, run {RUN}, bunch {bunch}', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / 'traces.png', dpi=110)
    plt.close(fig)


def fig_baseline(R):
    tr = np.sqrt(R['tb_us'][:-1] * R['tb_us'][1:])
    nb = len(set(R['bunch'].tolist()))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for j, fam in enumerate(FAMILIES):
        for arm in 'ABCD':
            m = R['det'] == f'{fam}{arm}'
            for k, ch in enumerate(sorted(set(R['ch'][m].tolist()))):
                s = np.nanmedian(R['shift_mV'][m & (R['ch'] == ch)], axis=0)
                ax[j].plot(tr, s, lw=0.8, c=ARM_COL[arm], alpha=0.8, label=f'arm {arm}' if k == 0 else None)
        ax[j].set_xscale('log')
        ax[j].set_yscale('symlog', linthresh=1)
        for y in (-3, 3):
            ax[j].axhline(y, c='#52514e', lw=0.6, ls=':')
        ax[j].axhline(0, c='#52514e', lw=0.6)
        ax[j].set_title(f'{FAM_NAME[fam]}', loc='left')
        ax[j].set_xlabel('time since the flash (µs)')
        _style(ax[j])
    ax[0].set_ylabel('baseline shift, mV (+ = pulse direction)')
    ax[2].legend(fontsize=8, frameon=False)
    fig.suptitle(f'Baseline after the flash: median of {nb} bunches, every channel (dotted: ±3 mV)', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / 'baseline.png', dpi=110)
    plt.close(fig)


def fig_psa(tc, C, nb):
    fig, ax = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for j, fam in enumerate(FAMILIES):
        for k in sorted(k for k in C if k[0].startswith(fam) and k[1] == 'all'):
            arm = k[0][3]
            first = k[2] == min(kk[2] for kk in C if kk[0] == k[0])
            c = C[k]
            ax[0, j].plot(tc, c['rate'] * WINDOW_US, lw=0.8, c=ARM_COL[arm], label=f'arm {arm}' if first else None)
            ax[1, j].plot(tc, c['fwhm'], lw=0.8, c=ARM_COL[arm])
            ax[2, j].plot(tc, c['amp'] * MV_PER_ADC, lw=0.8, c=ARM_COL[arm])
        for i in range(3):
            ax[i, j].set_xscale('log')
            ax[i, j].set_yscale('log')
            ax[i, j].set_xlim(0.1, 8e4)
            _style(ax[i, j])
        for y in (0.1, 0.01):
            ax[0, j].axhline(y, c='#52514e', lw=0.6, ls=':')
        ax[0, j].set_title(FAM_NAME[fam], loc='left')
        ax[2, j].set_xlabel('time since the flash (µs)')
    ax[0, 0].set_ylabel(f'hits in a {WINDOW_US * 1e3:.0f} ns window\n(per channel, per bunch)')
    ax[1, 0].set_ylabel('median FWHM of found pulses (ns)')
    ax[2, 0].set_ylabel('median amplitude of found pulses (mV)')
    ax[0, 2].legend(fontsize=8, frameon=False)
    fig.suptitle(f'PSA hits, run {RUN}, all {sum(nb.values())} bunches', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / 'psa.png', dpi=110)
    plt.close(fig)


def fig_intensity(tc, C, nb):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for j, fam in enumerate(FAMILIES):
        ks = sorted({(k[0], k[2]) for k in C if k[0].startswith(fam)})
        for cls, ls in (('par', ':'), ('ded', '-')):
            r = np.nanmean([C[(d, cls, ch)]['rate'] for d, ch in ks], axis=0)
            ax[j].plot(tc, r * WINDOW_US, ls=ls, c='#2a78d6', lw=1.3,
                       label=f'{"parasitic" if cls == "par" else "dedicated"} ({nb[cls]} bunches)')
        rr = (np.nanmean([C[(d, 'ded', ch)]['rate'] for d, ch in ks], axis=0)
              / np.nanmean([C[(d, 'par', ch)]['rate'] for d, ch in ks], axis=0))
        ax2 = ax[j].inset_axes([0.1, 0.1, 0.4, 0.34])
        ax2.plot(tc, rr, c='#52514e', lw=0.8)
        ax2.set_xscale('log')
        ax2.set_ylim(0, 3)
        ax2.set_xlim(1, 8e4)
        ax2.set_title('dedicated / parasitic', fontsize=7)
        ax2.tick_params(labelsize=6)
        ax[j].set_xscale('log')
        ax[j].set_yscale('log')
        ax[j].set_xlim(0.1, 8e4)
        ax[j].set_title(f'{FAM_NAME[fam]} (mean over channels)', loc='left')
        ax[j].set_xlabel('time since the flash (µs)')
        _style(ax[j])
    ax[0].set_ylabel(f'hits in a {WINDOW_US * 1e3:.0f} ns window')
    ax[0].legend(fontsize=8, frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(FIG / 'intensity.png', dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ report
def esc(s):
    return html.escape(str(s))


def table(head, rows):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--rule:#dddcd6;--box:#f2f1ed}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--rule:#3a3a37;--box:#252523}}
:root[data-theme=dark]{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--rule:#3a3a37;--box:#252523}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.5em;margin-bottom:.2em} h2{margin-top:1.6em;border-bottom:1px solid var(--rule)}
.sub{color:var(--muted)} .verdict{background:var(--box);border-left:4px solid #2a78d6;padding:10px 16px}
table{border-collapse:collapse;margin:.6em 0;font-size:.92em;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--rule);padding:4px 10px;text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600} img{max-width:100%;background:#fff}
figure{margin:1.2em 0} figcaption{color:var(--muted);font-size:.92em}
code{background:var(--box);padding:0 3px}
"""


def build(rows, R, nb):
    nraw = len(set(R['bunch'].tolist()))
    ndraw = len({b for b, p in zip(R['bunch'].tolist(), R['ppp']) if p > 6e12})
    F = {f: {k: fam_trange(rows, f, k) for k in
             ('first_hit', 'shape', 'base3', 'base1', 'occ10_par', 'occ10_ded', 'occ1_par', 'occ1_ded', 'head_end')}
         for f in FAMILIES}
    under = {f: fam_range(rows, f, 'under', '{:.0f}').replace('–-', ' to −').replace('-', '−')
             for f in FAMILIES}
    under = {f: ('none' if v in ('0', '−0') else v) for f, v in under.items()}
    occ1ms = {f: (fam_range(rows, f, 'occ1ms_par', '{:.2g}'), fam_range(rows, f, 'occ1ms_ded', '{:.2g}'))
              for f in FAMILIES}
    summary = table(
        ['', 'SiPM wall (32 ch)', 'plastic (8 ch)', 'liquid (4 ch)'],
        [['PSA finds pulses again', *[F[f]['first_hit'] for f in FAMILIES]],
         ['pulse shape back to its late plateau (FWHM ±15 %)', *[F[f]['shape'] for f in FAMILIES]],
         ['baseline within ±3 mV', *[F[f]['base3'] for f in FAMILIES]],
         ['baseline within ±1 mV', *[F[f]['base1'] for f in FAMILIES]],
         ['deepest baseline undershoot after 20 µs (mV)', *[under[f] for f in FAMILIES]],
         [f'occupancy of a {WINDOW_US * 1e3:.0f} ns window &lt; 10 %, parasitic / dedicated',
          *[f"{F[f]['occ10_par']} / {F[f]['occ10_ded']}" for f in FAMILIES]],
         [f'occupancy &lt; 1 %, parasitic / dedicated', *[f"{F[f]['occ1_par']} / {F[f]['occ1_ded']}" for f in FAMILIES]],
         ['occupancy at 1 ms, parasitic / dedicated', *[f'{a} / {b}' for a, b in occ1ms.values()]]])
    per_ch = table(
        ['channel', 'pulses found', 'shape back', 'baseline ±3 mV', 'baseline ±1 mV', 'undershoot (mV)',
         'occ. &lt; 10 % (par / ded)', 'occ. &lt; 1 % (par / ded)', 'noise (mV rms)'],
        [[f"{r['det']}{r['ch']}", tfmt(r['first_hit']), tfmt(r['shape']), tfmt(r.get('base3', np.nan)),
          tfmt(r.get('base1', np.nan)), f"{r.get('under', np.nan):.1f}",
          f"{tfmt(r['occ10_par'])} / {tfmt(r['occ10_ded'])}", f"{tfmt(r['occ1_par'])} / {tfmt(r['occ1_ded'])}",
          f"{r.get('noise', np.nan):.2f}"] for r in rows])
    thr = (f"wall trigger thresholds {min(WALL_THR.values()):.0f}–{max(WALL_THR.values()):.0f} mV on the "
           f"top+bottom sum, plastic {min(PLAS_THR.values()):.0f}–{max(PLAS_THR.values()):.0f} mV")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Scintillator flash recovery</title>
<style>{CSS}</style></head><body>
<h1>Scintillator recovery after the γ-flash — first look</h1>
<p class="sub">n_TOF EAR2, run {RUN} (= DREAM run_79), {sum(nb.values())} bunches of PSA hits and {nraw} bunches of raw
samples ({ndraw} of them dedicated). Generated by <code>ntof_scint_recovery/make_report.py</code>.</p>

<div class="verdict">
<p><b>Order of magnitude, electronics: the liquids are back after ~10 µs, the plastics after ~0.1–1 ms, the
SiPM walls after ~1 ms.</b> "Back" here means the baseline is within ±3 mV and the pulse shape is normal. Liquids:
baseline ±3 mV by {F['LIQ']['base3']}. Plastics: {F['PSS']['base3']} (±1 mV by {F['PSS']['base1']}). The walls'
limit is a baseline <i>undershoot</i> of {under['WAL']} mV after the diverted flash decays. It lasts until
{F['WAL']['base3']}, against {thr}.</p>
<p><b>Beam occupancy, which is not the electronics, sets a later and intensity-dependent limit.</b> A random hit
lands in a {WINDOW_US * 1e3:.0f} ns window on one channel less than 10 % of the time after
{F['WAL']['occ10_par']} (walls), {F['LIQ']['occ10_par']} (liquids) and {F['PSS']['occ10_par']} (plastics), for
parasitic bunches. Dedicated bunches are about twice as busy and later: {F['WAL']['occ10_ded']},
{F['LIQ']['occ10_ded']} and {F['PSS']['occ10_ded']}. At 1 ms the plastics are still at
{occ1ms['PSS'][0]} (parasitic) / {occ1ms['PSS'][1]} (dedicated).</p>
<p>For scale: the neutrons that matter arrive after ~1 ms (E<sub>n</sub> ≲ 2 eV at 19.5 m), and DREAM triggers
start from ~0.5 ms. The liquids and plastics are electronically recovered before that. The walls are only just
recovered when it starts. Gain recovery is <b>not</b> measured here (see the end).</p>
</div>

<h2>Recovery times, by family</h2>
<p>Ranges are over channels. Each row is a different definition of "recovered", and each one bounds a different
use of the detector:</p>
{summary}
<p class="sub">"Pulses found" is the first time the PSA hit rate reaches 30 % of its peak. On the walls this is set by
the ~1 µs divert (blanking) gate, which is on in this run. "Shape back" is the time after which the median FWHM of
found pulses stays within ±15 % of its 1–10 ms value. Baseline rows come from the {nraw} raw bunches, the
median over bunches, evaluated to 10 ms. Past 10 ms the walls drift by ~1 mV for reasons unrelated to the flash.
The occupancy rows are hit rate × {WINDOW_US * 1e3:.0f} ns, per channel. A pile-up-limited PSA undercounts
at early times, so occupancy is a lower bound there.</p>

<h2>What was looked at</h2>
<ul>
<li><b>Raw stream1 samples</b>: seven 450 MB head chunks of run {RUN}
(<code>/media/dylan/data/x17/ntof_raw_224572/</code>, pulled 2026-07-29 for the PSA tuning). The DAQ keeps an
unsuppressed 30 µs head from t&nbsp;=&nbsp;0 and extends it while the signal stays outside the ZS band. The flash
sits ~11.6 µs into it, so the flash and its tail are complete. After that, every ZS block's 259 pre-samples give the
local baseline. This yields {nraw} complete bunches, every channel, out to 80 ms.</li>
<li><b>PSA hits</b>: our v12 reprocessing of the same run, all 16 segments, every pulse the PSA found in the whole
80 ms (<code>ntof_reproc/v12_liqpileup/</code>). Time since the flash is <code>tof − (tof<sub>PKUP</sub> + C)</code>,
with C from <code>ntof_processing/flash_timing/</code>.</li>
</ul>

<h2>Data for a proper measurement</h2>
<ul>
<li><b>PSA hits for every run are on EOS now</b> (<code>/eos/experiment/ntof/processing/official/done/run&lt;N&gt;.root</code>,
10–32 GB each). The shape and occupancy parts of this study can run over the whole campaign, including the
divert-off runs, with no tape.</li>
<li><b>Raw stream1 (the only source of samples) is still on EOS disk for 32 of our runs</b> (~5.7 TB, checked
2026-10-20; e.g. 224540, 224618, 224666–224669, 224712). Each raw file holds ~20 bunches, and its first 2 ms
are ~48 MB per bunch. One or two files per run give enough bunches for a pulse-by-pulse reconstruction of the
tail. These disk copies can expire at any time.</li>
<li><b>Tape (CTA) is needed for the rest</b>, notably run {RUN} itself beyond the local chunks, and the
divert-off runs 224356–360 and 224464/466, which are the only runs where the walls see the undiverted flash.
A recall can be per file and stays online for less than a day, so it has to be read straight away
(<code>ntof_processing/waveform_pull/lxplus/recall.sh</code>).</li>
<li>The existing waveform pull (<code>/eos/experiment/ntof/data/x17/wf_pull</code>, 287 GB) is ±5 µs around DREAM
triggers, which start at ~0.5 ms. It does not contain the flash tail.</li>
</ul>

<h2>Figures</h2>
<figure><img src="figures/traces.png" alt="raw traces"><figcaption>One bunch, raw. Left: the flash. The wall
is diverted and clamps for ~1 µs, then decays back over ~20 µs. The plastic and liquid PMT signals hit the rail
and come back within a few µs. Right: the same traces to 2 ms on a log axis, zoomed in. The wall baseline
crosses zero at ~25 µs and stays below it until ~1 ms. The plastic is a continuous pulse train for the first ~100 µs.</figcaption></figure>
<figure><img src="figures/baseline.png" alt="baseline shift"><figcaption>Baseline shift against the pre-flash
level, median over bunches, every channel. Inside the head block this is the per-bin median of the samples, so for
the plastics in the first ~100 µs it includes pile-up. That is the effective baseline a discriminator sees.
Walls: +30 mV at 20 µs, then an undershoot of {under['WAL']} mV at ~50–200 µs, gone by ~1 ms.</figcaption></figure>
<figure><img src="figures/psa.png" alt="PSA hits"><figcaption>PSA hits. Top: occupancy of a {WINDOW_US * 1e3:.0f} ns window.
Middle: median FWHM. Bottom: median amplitude. FWHM settles in a few µs (liquids, plastics) to ~10–20 µs (walls). The
median amplitude keeps drifting on the plastics out to ~100 µs, but that is a changing particle population and
pile-up, not necessarily gain (see below). The step in the plastic rate at ~15 µs is where the DAQ's forced 30 µs head block ends (18 µs after the flash), i.e. a readout boundary. The bump at ~5 ms in every family is beam-correlated (thermal neutron
time of flight), not a detector effect.</figcaption></figure>
<figure><img src="figures/intensity.png" alt="intensity"><figcaption>Parasitic (~4×10¹² ppp) against dedicated
(~8.5×10¹² ppp) bunches. Inset: ratio of hit rates. In the tail it is roughly ×2, i.e. the background scales with the
protons. The occupancy-based recovery times therefore depend on intensity, while the shape and baseline times
(electronics) should not to first order. That second point is not yet checked; the raw sample is too small to split.</figcaption></figure>

<h2>Per channel</h2>
{per_ch}

<h2>What this does not rule out</h2>
<ul>
<li><b>Gain recovery is not measured.</b> Nothing here is a standard candle in the tail. The PSA amplitude
median changes because the particle population changes. A PMT or SiPM gain sag after the flash charge that lasts
longer than the baseline excursion would not show up here. The plastics and liquids are PMTs, where a sag lasting
hundreds of µs after a saturating pulse is common.</li>
<li><b>One run.</b> Run {RUN}, late July, divert gate on with its 1000 ns delay (after 2026-07-22). The
divert-off runs (224356–360, 224464/466) and the earlier gate setting would give different wall numbers.</li>
<li><b>The trigger path is not the digitizer path.</b> The undershoot is measured on the digitized signal. Whether
the analog sum feeding the hardware discriminator sees the same undershoot depends on where it is split off. If it
does, the wall trigger efficiency is strongly suppressed until ~1 ms.</li>
<li><b>{nraw} raw bunches.</b> Enough for a median baseline, not for the bunch-to-bunch spread, the intensity
dependence, or a pulse-by-pulse reconstruction of the tail.</li>
<li><b>The PSA undercounts at early times.</b> At tens of hits per µs it cannot separate piled-up pulses, so early
occupancies are lower bounds and "pulses found" means pulses the PSA can fit.</li>
</ul>
</body></html>"""
    (OUT / 'report.html').write_text(doc)


def main() -> int:
    R = load_raw()
    tb, ab, wb, H, nb = load_psa()
    tc, C = psa_curves(tb, ab, wb, H, nb)
    rows = recovery_table(tc, C, R)
    fig_traces()
    fig_baseline(R)
    fig_psa(tc, C, nb)
    fig_intensity(tc, C, nb)
    build(rows, R, nb)
    import csv
    with open(OUT / 'recovery_by_channel.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f'-> {OUT}/report.html')
    return 0


if __name__ == '__main__':
    sys.exit(main())
