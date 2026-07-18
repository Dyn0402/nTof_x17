"""
12_plastic_hv_scan.py — Plastic-PMT HV plateau scan analysis (run224466 vs the
CAEN HV log in ~/beam_july/scint_hv_scan/2026-07-16_13-32-34_plastic_scan_1/).

The scan held all 8 plastic PMTs at a common voltage per ~10-min step (pass 1:
1600->1200 V in 100 V steps, SiPM flash-gating OFF; pass 2: 1550->1250 V,
gating ON), ~equal protons per step. Bunches are assigned to steps by wall
clock (PKUP psTime, ns UTC; scan windows are local = UTC+2). Two pseudo-steps
are added: the pre-scan stretch (all plastics at 1500 V, gating OFF) and the
post-scan end_state (per-channel nominal HV 1275-1325 V).

Per step this script accumulates
  - plastic inclusive-late amplitude spectra per PMT  (gain/plateau curve),
  - plastic late hit counts + delivered protons       (rate per proton),
  - wall-tagged plastic and plastic-tagged WALL amplitude spectra per channel,
    offsets applied, +-8 ns signal / +20..120 ns sideband (07 machinery)
    -> does the plastic HV move the wall MIP peak?
  - wall gamma-flash tof zoom per arm                 (confirms gating state).

Cache: cache/12_hvscan_<run>.npz. Table output: per-PMT median amp vs V with
per-pass power-law index, and wall MIP peak per arm vs step.

Usage: python 12_plastic_hv_scan.py [run_file]
"""

import importlib.util
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import uproot

import hitcache

BASE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('coinc', BASE / '02_coincidence_scan.py')
coinc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coinc)

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224466.root'
CACHE = BASE / 'cache'

LOCAL_UTC_OFFSET = timedelta(hours=2)          # CEST
LATE_TOF = 1e5                                 # ns after tflash, as in 07/09
W_SIG = 8.0
SB_LO, SB_HI = 20.0, 120.0
SB_SCALE = (2 * W_SIG) / (SB_HI - SB_LO)
DT_MAX = SB_HI + 25
AMP_EDGES = np.geomspace(40, 8e4, 301)
FLASH_EDGES = np.linspace(9e3, 16e3, 701)      # same zoom as 01

# Per-run scan config: (label, V or None for per-channel nominal, pass,
# gating ON?, start, end) in LOCAL time. Step windows from the CAEN log CSV
# (first/last row per step_label), start trimmed for the HV ramp when needed.
SCANS = {
    # ~/beam_july/scint_hv_scan/2026-07-16_13-32-34_plastic_scan_1/
    'run224466': ('2026-07-16', [
        ('pre_1500V',   1500, 0, False, '13:04:33', '13:32:30'),
        ('s1_1600V',    1600, 1, False, '13:32:34', '13:42:00'),
        ('s2_1500V',    1500, 1, False, '13:42:05', '13:51:41'),
        ('s3_1400V',    1400, 1, False, '13:51:46', '14:01:47'),
        ('s4_1300V',    1300, 1, False, '14:01:52', '14:11:28'),
        ('s5_1200V',    1200, 1, False, '14:11:33', '14:21:34'),
        ('s6_1550V',    1550, 2, True,  '14:21:39', '14:31:45'),
        ('s7_1450V',    1450, 2, True,  '14:31:50', '14:40:20'),
        ('s8_1350V',    1350, 2, True,  '14:40:25', '14:49:56'),
        ('s9_1250V',    1250, 2, True,  '14:50:01', '14:59:37'),
        ('end_nominal', None, 3, True,  '15:01:00', '15:19:30'),   # AL1325 AR1275 BL1325 rest 1300
    ]),
    # ~/beam_july/scint_hv_scan/2026-07-17_17-41-11_plastic_scan_2/
    # FIFO signal path, SiPM gating ON throughout; no usable pre-scan window
    # (first raw file closed 17:40, scan started 17:41). Step 1 start trimmed
    # +20 s for the ~1300->1600 V ramp (~36 V/s); ~3 s inter-step ramps are
    # covered by the 5 s CSV cadence gaps.
    'run224489': ('2026-07-17', [
        ('s1_1600V',    1600, 1, True, '17:41:30', '17:50:27'),
        ('s2_1500V',    1500, 1, True, '17:50:32', '18:00:03'),
        ('s3_1400V',    1400, 1, True, '18:00:08', '18:09:08'),
        ('s4_1300V',    1300, 1, True, '18:09:13', '18:18:49'),
        ('s5_1200V',    1200, 1, True, '18:18:54', '18:27:55'),
        ('s6_1550V',    1550, 2, True, '18:28:00', '18:37:31'),
        ('s7_1450V',    1450, 2, True, '18:37:36', '18:46:37'),
        ('s8_1350V',    1350, 2, True, '18:46:42', '18:55:43'),
        ('s9_1250V',    1250, 2, True, '18:55:48', '19:05:24'),
        # ramp back to nominal done in seconds; last PKUP psTime 19:17:04
        ('end_nominal', None, 3, True, '19:05:45', '19:17:30'),
    ]),
}
_stem = (Path(sys.argv[1]).stem if len(sys.argv) > 1 else 'run224466')
if _stem not in SCANS:
    sys.exit(f'no scan config for {_stem} — add it to SCANS in 12_plastic_hv_scan.py')
DAY, STEPS = SCANS[_stem]
NOMINAL_V = {('A', 0): 1325, ('A', 1): 1275, ('B', 0): 1325, ('B', 1): 1300,
             ('C', 0): 1300, ('C', 1): 1300, ('D', 0): 1300, ('D', 1): 1300}


def local_to_epoch_ns(hms):
    dt = datetime.strptime(f'{DAY} {hms}', '%Y-%m-%d %H:%M:%S') - LOCAL_UTC_OFFSET
    return dt.replace(tzinfo=timezone.utc).timestamp() * 1e9


def bunch_step_map(run_file, n_bunches):
    """step index per bunch number (index 0..n_bunches, -1 = outside any step)."""
    f = uproot.open(run_file)
    pk = f['PKUP'].arrays(['BunchNumber', 'psTime'], library='np')
    o = np.argsort(pk['BunchNumber'])
    bn, t = pk['BunchNumber'][o], pk['psTime'][o]
    step_of = np.full(n_bunches + 1, -1, dtype=np.int64)
    for i, (label, v, pas, gate, t0, t1) in enumerate(STEPS):
        m = (t >= local_to_epoch_ns(t0)) & (t < local_to_epoch_ns(t1))
        step_of[bn[m]] = i
    return step_of


def main():
    intensity = coinc.bunch_intensity(RUN_FILE)
    good_bunches = coinc.select_bunches(RUN_FILE.stem, intensity)
    n_bunches = len(intensity)
    step_of = bunch_step_map(RUN_FILE, n_bunches)
    offs_json = json.loads(
        (BASE / 'calib' / f'time_offsets_{RUN_FILE.stem}.json').read_text())

    n_step = len(STEPS)
    nb = len(AMP_EDGES) - 1
    # per-step delivered protons (sum of PulseIntensity of assigned good bunches)
    protons = np.zeros(n_step)
    sel = step_of[good_bunches] >= 0
    np.add.at(protons, step_of[good_bunches][sel], intensity[good_bunches - 1][sel])
    n_good_per_step = np.bincount(step_of[good_bunches][sel], minlength=n_step)
    print(f'{len(good_bunches)} good bunches; per step: '
          + ', '.join(f'{l}:{n}' for (l, *_), n in zip(STEPS, n_good_per_step)))

    results = {
        'pss_amp': np.zeros((n_step, 4, 2, nb)),        # inclusive late
        'pss_n_late': np.zeros((n_step, 4, 2)),
        'wal_mip': np.zeros((n_step, 4, 2, 8, nb)),     # (step, arm, sig/side, ch, amp)
        'pss_mip': np.zeros((n_step, 4, 2, 2, nb)),     # (step, arm, sig/side, bar, amp)
        'wal_flash': np.zeros((n_step, 4, len(FLASH_EDGES) - 1)),
        'protons': protons,
        'n_good_bunches': n_good_per_step.astype(float),
    }

    for ai, st in enumerate('ABCD'):
        off = np.zeros((8, 2))
        for wc in range(8):
            for pc in range(2):
                off[wc, pc] = offs_json['stations'][st][
                    f'WAL{st}{wc + 1}_PSS{st}{pc + 1}']['offset_ns']

        wal = hitcache.load(RUN_FILE, f'WAL{st}',
                            ['BunchNumber', 'tof', 'detn', 'amp'],
                            good_bunches=good_bunches)
        pss = hitcache.load(RUN_FILE, f'PSS{st}',
                            ['BunchNumber', 'tof', 'detn', 'amp', 'tflash'],
                            good_bunches=good_bunches)

        # wall flash zoom per step (all wall hits, any channel)
        s_w = step_of[wal['BunchNumber']]
        m = s_w >= 0
        fb = np.clip(np.digitize(wal['tof'][m], FLASH_EDGES) - 1,
                     0, len(FLASH_EDGES) - 2)
        inside = (wal['tof'][m] >= FLASH_EDGES[0]) & (wal['tof'][m] < FLASH_EDGES[-1])
        results['wal_flash'][:, ai] += np.bincount(
            s_w[m][inside] * (len(FLASH_EDGES) - 1) + fb[inside],
            minlength=n_step * (len(FLASH_EDGES) - 1)
        ).reshape(n_step, -1)

        # late plastic hits
        late = (pss['tof'] - pss['tflash']) > LATE_TOF
        pss = {k: v[late] for k, v in pss.items()}
        s_p = step_of[pss['BunchNumber']]
        bar = pss['detn'].astype(np.int64) - 1
        ok = s_p >= 0
        ab = np.clip(np.digitize(pss['amp'][ok], AMP_EDGES) - 1, 0, nb - 1)
        results['pss_amp'][:, ai] += np.bincount(
            (s_p[ok] * 2 + bar[ok]) * nb + ab, minlength=n_step * 2 * nb
        ).reshape(n_step, 2, nb)
        results['pss_n_late'][:, ai] += np.bincount(
            s_p[ok] * 2 + bar[ok], minlength=n_step * 2).reshape(n_step, 2)

        # wall x plastic coincidences (late pss as in 07), tagged by step
        t_p, t_w = pss['tof'], wal['tof']
        det_w = wal['detn'].astype(np.int64)
        key_p = hitcache.bunch_key(pss['BunchNumber'], t_p)
        key_w = hitcache.bunch_key(wal['BunchNumber'], t_w)
        for ri, oi in hitcache.iter_pairs(key_p, key_w, -DT_MAX, DT_MAX, t_p, t_w):
            ch_p = bar[ri]
            ch_w = det_w[oi] - 1
            s = s_p[ri]
            dt_cal = (t_w[oi] - t_p[ri]) - off[ch_w, ch_p]
            inwin = s >= 0
            for j, mask in enumerate((np.abs(dt_cal) <= W_SIG,
                                      (dt_cal >= SB_LO) & (dt_cal <= SB_HI))):
                mm = mask & inwin
                if not mm.any():
                    continue
                aw = np.clip(np.digitize(wal['amp'][oi][mm], AMP_EDGES) - 1, 0, nb - 1)
                ap = np.clip(np.digitize(pss['amp'][ri][mm], AMP_EDGES) - 1, 0, nb - 1)
                results['wal_mip'][:, ai, j] += np.bincount(
                    (s[mm] * 8 + ch_w[mm]) * nb + aw, minlength=n_step * 8 * nb
                ).reshape(n_step, 8, nb)
                results['pss_mip'][:, ai, j] += np.bincount(
                    (s[mm] * 2 + ch_p[mm]) * nb + ap, minlength=n_step * 2 * nb
                ).reshape(n_step, 2, nb)
        print(f'arm {st}: {int(results["wal_mip"][:, ai, 0].sum()):,} sig pairs '
              f'across steps', flush=True)

    np.savez_compressed(
        CACHE / f'12_hvscan_{RUN_FILE.stem}.npz',
        amp_edges=AMP_EDGES, flash_edges=FLASH_EDGES, sb_scale=SB_SCALE,
        step_labels=np.array([s[0] for s in STEPS]),
        step_volts=np.array([s[1] if s[1] else 0 for s in STEPS], dtype=float),
        step_pass=np.array([s[2] for s in STEPS]),
        step_gating=np.array([s[3] for s in STEPS]),
        **results)
    print(f'Cached -> {CACHE / f"12_hvscan_{RUN_FILE.stem}.npz"}')

    # ---------------- summary tables ----------------
    cen = np.sqrt(AMP_EDGES[:-1] * AMP_EDGES[1:])

    def median_amp(h):
        c = np.cumsum(h)
        return cen[np.searchsorted(c, c[-1] / 2)] if c[-1] > 50 else np.nan

    print('\nPlastic inclusive-late spectra: median amp [ADC] vs step')
    hdr = ' '.join(f'{s[0][:9]:>9s}' for s in STEPS)
    print(f'{"PMT":8s} {hdr}')
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            med = [median_amp(results['pss_amp'][i, ai, b]) for i in range(n_step)]
            print(f'PSS{st}{b + 1}    ' + ' '.join(f'{m:9.0f}' if np.isfinite(m)
                                                   else f'{"-":>9s}' for m in med))

    # per-pass power-law index from medians (G ~ V^n)
    print('\nGain power-law index n (median amp ~ V^n), per PMT per pass:')
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            row = []
            for pas in (1, 2):
                ii = [i for i, s in enumerate(STEPS) if s[2] == pas]
                v = np.array([STEPS[i][1] for i in ii], float)
                m = np.array([median_amp(results['pss_amp'][i, ai, b]) for i in ii])
                okm = np.isfinite(m) & (m > 0)
                if okm.sum() >= 3:
                    n_fit = np.polyfit(np.log(v[okm]), np.log(m[okm]), 1)[0]
                    row.append(f'pass{pas}: n={n_fit:5.2f}')
            print(f'  PSS{st}{b + 1}: ' + '   '.join(row))

    # wall MIP peak (07-style smoothed mode) per arm per step, ch-summed
    kern = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    kern /= kern.sum()
    print('\nWall MIP peak [ADC] (ch-summed, sideband-subtracted) vs step:')
    print(f'{"arm":4s} {hdr}')
    for ai, st in enumerate('ABCD'):
        row = []
        for i in range(n_step):
            sub = (results['wal_mip'][i, ai, 0].sum(axis=0)
                   - SB_SCALE * results['wal_mip'][i, ai, 1].sum(axis=0))
            sm = np.convolve(sub, kern, mode='same')
            msk = cen > 150
            row.append(cen[msk][np.argmax(sm[msk])] if sub.sum() > 500 else np.nan)
        print(f'{st:4s} ' + ' '.join(f'{p:9.0f}' if np.isfinite(p)
                                     else f'{"-":>9s}' for p in row))
    print('\n(gating OFF: pre + s1-s5; ON: s6-s9 + end_nominal. '
          'end_nominal V: AL1325 AR1275 BL1325 rest 1300)')


if __name__ == '__main__':
    main()
