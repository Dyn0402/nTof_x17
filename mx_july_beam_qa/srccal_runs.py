"""srccal_runs.py — run map and physics constants for the 2026-07-28 two-source
plastic calibration (Y-88 + Cs-137 simultaneously, on opposite arms).

Campaign (DAQ titles, all 2026-07-28, X17_measurement, beam off):

    224588  Y on AR, Cs on CR     11:31:06-11:37:41   607 triggers
    224589  Y on AL, Cs on CL     11:38:49-11:44:22   603
    224590  Y on BR, Cs on DR     11:46:58-11:53:32   612
    224591  Y on BL, Cs on DL     11:55:24-12:02:04   621
    224592  Y on CR, Cs on AR     12:02:54-12:09:25   605
    224593  Y on CL, Cs on AL     12:10:31-12:17:07   606
    224594  Y on DR, Cs on BR     12:17:20-12:22:56   607
    224595  Y on DL, Cs on BL     12:23:49-12:29:25   609
    224596  Y on AR (no Cs)       12:30:40-12:37:13   609     <- CONTROL

Differences from the 2026-07-17 Y-88 scan (runs 224476-79, scripts 21/22/23):

  * the source is centred on ONE bar, not placed between the two bars of an arm,
    so the illuminated channel is unambiguous (and the assumed detn<->L/R map is
    testable — see `assumed_detn`);
  * two sources run at once on OPPOSITE arms (ring order A-D-C-B, so A<->C and
    B<->D are the opposite pairs), which doubles the yield and puts both sources
    in the same DAQ state;
  * every bar is measured with BOTH sources across the campaign, giving three
    Compton edges per plastic (477 / 699 / 1612 keVee) instead of two — so the
    energy scale is a fitted LINE with an intercept, not a through-origin slope;
  * 224596 repeats 224588's Y position with the Cs REMOVED: it is both a
    repeatability point on AR and a direct measurement of how much the Cs source
    leaks into the far arm.

Compton edges (E_edge = 2E^2/(m_e c^2 + 2E); organic scintillator, no photopeak):

    Cs-137   661.66 keV gamma  ->  477.34 keVee
    Y-88     898.04 keV gamma  ->  698.63 keVee   (dominant)
    Y-88    1836.06 keV gamma  -> 1612.06 keVee   (weaker, cleaner separation)

The two Y-88 values are kept bit-identical to `22_y88_edges.py` / the
`calib/y88_energy_calib.json` keys so the two campaigns can be compared without
a unit or convention shim.
"""

# --- physics ---------------------------------------------------------------
E_CS = 477.34      # keVee, Cs-137 661.66 keV Compton edge
E_Y1 = 698.63      # keVee, Y-88 898.04 keV Compton edge  (as in 22_y88_edges)
E_Y2 = 1612.06     # keVee, Y-88 1836.06 keV Compton edge (as in 22_y88_edges)
# NB the two Y-88 values are carried over verbatim from 22_y88_edges.py so the
# campaigns share keys and conventions. Recomputing 2E^2/(m_e c^2 + 2E) with
# m_e c^2 = 510.999 keV gives 699.13 and 1611.75 — 0.07 % and 0.02 % away. That
# is far below the ~5 % convention systematic on an edge position, so it is not
# worth breaking continuity over; do not "fix" it in one campaign only.

EDGES_OF = {'Y88': (E_Y1, E_Y2), 'Cs137': (E_CS,)}

# --- geometry / channel map ------------------------------------------------
ARMS = 'ABCD'
NCH = {'PSS': 2, 'WAL': 8, 'LIQ': 1}
# README: "PSS: detn 1 = left bar, 2 = right bar (seen from back; per Dylan)".
# ASSUMED, not yet confirmed in data — 35 tests it (the illuminated bar must be
# the high-rate one, and this campaign lights exactly one bar at a time).
assumed_detn = {'L': 1, 'R': 2}

# Ring order A-D-C-B => opposite pairs are A<->C and B<->D. The run map below
# should satisfy that for every run; `check_map()` asserts it.
OPPOSITE = {'A': 'C', 'C': 'A', 'B': 'D', 'D': 'B'}

# --- run map ---------------------------------------------------------------
# bar label = arm letter + L|R  (e.g. 'AR');  None = source not present
SOURCE_MAP = {
    'run224588': {'Y88': 'AR', 'Cs137': 'CR'},
    'run224589': {'Y88': 'AL', 'Cs137': 'CL'},
    'run224590': {'Y88': 'BR', 'Cs137': 'DR'},
    'run224591': {'Y88': 'BL', 'Cs137': 'DL'},
    'run224592': {'Y88': 'CR', 'Cs137': 'AR'},
    'run224593': {'Y88': 'CL', 'Cs137': 'AL'},
    'run224594': {'Y88': 'DR', 'Cs137': 'BR'},
    'run224595': {'Y88': 'DL', 'Cs137': 'BL'},
    'run224596': {'Y88': 'AR', 'Cs137': None},      # control: Y only
}
RUNS = list(SOURCE_MAP)
CONTROL_RUN = 'run224596'

# --- 2026-07-17 legacy Y-88 scan (runs 224476-79), for the cross-check -------
# One Y-88 source per run, placed BETWEEN the two bars of an arm, so BOTH bars
# are illuminated; the bar named here only decides which one is tagged
# `source_bar`. These runs were reprocessed by the same 07-30 official pass as
# the 07-28 campaign, so pushing them through the SAME chain isolates how much
# of the 07-28-vs-equalization discrepancy is analysis (new PSA + new fitter)
# and how much is hardware. See 36_srccal_legacy_check.py.
LEGACY_SOURCE_MAP = {
    'run224476': {'Y88': 'AL'},
    'run224477': {'Y88': 'BL'},
    'run224478': {'Y88': 'CL'},
    'run224479': {'Y88': 'DL'},
}
LEGACY_RUNS = list(LEGACY_SOURCE_MAP)
ALL_MAP = {**SOURCE_MAP, **LEGACY_SOURCE_MAP}

# The 07-19 HV equalization (nTof_x17_DAQ calibrations/pss/hv_equalization_y88_fifo.json)
# moved every plastic PMT so that its Y-88 699 keVee edge would land on a common
# target, and those voltages are still the standing set — so this number is a
# hard prediction for the 07-28 data.
EQUALIZED_TARGET_699_MV = 31.2
# Standing plastic PMT bias, CAEN card 07, applied 2026-07-19 and unchanged
# since (confirmed on the DAQ machine, run_config_beam.py + scint_hv_config.py).
# 'n' = measured gain power-law index, gain ~ V^n, valid 1200-1600 V.
PLASTIC_HV_V = {'AL': 1237, 'AR': 1177, 'BL': 1440, 'BR': 1248,
                'CL': 1214, 'CR': 1312, 'DL': 1331, 'DR': 1448}
PLASTIC_HV_INDEX_N = {'AL': 6.94, 'AR': 5.08, 'BL': 5.19, 'BR': 7.10,
                      'CL': 3.80, 'CR': 6.64, 'DL': 5.77, 'DR': 6.39}
# HV in force for the 07-17 legacy scan (flat operational set, pre-equalization,
# and pre-FIFO readout — both change the amplitude scale).
LEGACY_HV_V = {'AL': 1325, 'AR': 1275, 'BL': 1325, 'BR': 1300,
               'CL': 1300, 'CR': 1300, 'DL': 1300, 'DR': 1300}

# Acquisition window per trigger (n_TOF EAR2 X17 setting, 1 GS/s x 20 ms).
# Used only to quote absolute rates; all subtractions normalise by TRIGGER
# COUNT, which is exact regardless of this number.
WINDOW_MS = 20.0

EOS_DONE = '/eos/experiment/ntof/processing/official/done'


# --- helpers ---------------------------------------------------------------
def bar_channel(bar):
    """'AR' -> ('PSSA', 2): tree name and 1-based detn of a plastic bar."""
    arm, side = bar[0], bar[1]
    return f'PSS{arm}', assumed_detn[side]


def bar_key(bar):
    """'AR' -> 'PSSA2', the per-channel key used in the caches and calib JSON."""
    tree, detn = bar_channel(bar)
    return f'{tree}{detn}'


def sources_in(run):
    """{'Y88': 'AR', 'Cs137': 'CR'} with absent sources dropped. Works for both
    the 07-28 campaign and the 07-17 legacy scan."""
    return {k: v for k, v in ALL_MAP[run].items() if v}


def lit_arms(run):
    """Arms with a source on them in this run (these arms' WAL/LIQ see it too)."""
    return {b[0] for b in sources_in(run).values()}


def dark_runs_for(arm, runs=None):
    """Runs in which `arm` carries NO source — the background template pool for
    every channel of that arm (ambient + electronics, same DAQ state, same day).
    Note the neighbour bar of a lit arm is NOT dark: it sees the source through
    the light guide and through scattering, which is a signal, not background."""
    return [r for r in (runs or RUNS) if arm not in lit_arms(r)]


def source_on_arm(run, arm):
    """'Y88' | 'Cs137' | None — which source (if any) illuminates `arm`."""
    for src, bar in sources_in(run).items():
        if bar[0] == arm:
            return src
    return None


def check_map():
    """Sanity: opposite arms, every bar measured with both sources exactly once
    (bar AR twice with Y: 224588 + the 224596 control)."""
    for r, s in SOURCE_MAP.items():
        if s['Cs137']:
            assert OPPOSITE[s['Y88'][0]] == s['Cs137'][0], f'{r}: not opposite'
    seen = {}
    for r in RUNS:
        for src, bar in sources_in(r).items():
            seen.setdefault((bar, src), []).append(r)
    bars = [a + s for a in ARMS for s in 'LR']
    for b in bars:
        for src in ('Y88', 'Cs137'):
            assert (b, src) in seen, f'{b} never saw {src}'
    assert seen[('AR', 'Y88')] == ['run224588', 'run224596']
    return seen


if __name__ == '__main__':
    seen = check_map()
    print(f'{len(RUNS)} runs, map consistent; per bar/source runs:')
    for k in sorted(seen):
        print(f'  {k[0]:3s} {k[1]:6s} {seen[k]}')
    for a in ARMS:
        print(f'  arm {a}: dark in {dark_runs_for(a)}')
