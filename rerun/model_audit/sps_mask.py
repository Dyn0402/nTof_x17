"""The det4 channel layer at H4 (model audit F12, 2026-10-27).

A bench bundle's ``dead``/``hot``/``gain`` are bench FEU 6/8 channel numbers; at H4 det4 sits on FEU 03 with connectors
3-6 only and every plug inverted, so the same number is a different strip (bench X ch 300 is a Y strip at H4).  Modes:
  h4     (default) the mask MEASURED at H4 with the uRWELL by the bench census rule (rerun/channel_layer/det4_sps_h4.json,
         FEU 03 numbering: ws_dead_charge/sps_census.py + make_sps_mask.py)
  h4_stripes  also the strips under det4's dead amplification stripes (they carry spread charge: a model choice, ungated)
  bench  the bench bundle's lists carried by strip position (CalibrationBundle.remap_channels)
  none   no mask (what the audit's SPS arms ran with before 10-27)
"""
import json, os, sys
import numpy as np

REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
H4_MASK = os.path.join(REPO, 'rerun', 'channel_layer', 'det4_sps_h4.json')
MODES = ('h4', 'h4_stripes', 'bench', 'none')


def _sps_pos():
    sys.path.insert(0, os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment'))
    from det4_sps_map import POSITION_MM, VIEW
    return {v: np.where(VIEW == v, POSITION_MM, np.nan) for v in ('x', 'y')}


def apply(cal, mode='h4'):
    """Put the H4 channel layer into a (bench det4) bundle in place; returns a short record for the table meta."""
    if mode not in MODES:
        raise ValueError(f'sps mask mode {mode!r} not in {MODES}')
    if mode == 'none':
        cal.dead, cal.hot = {'x': [], 'y': []}, {}
        cal.gain = {p: np.ones(512) for p in ('x', 'y')}
        return dict(mode=mode)
    if mode == 'bench':
        c = np.arange(512)
        bp = 49.92 * (c // 64) + (63 - c % 64) * 0.78      # every June bench row (strip_position_map, checked)
        rec = cal.remap_channels({'x': bp, 'y': bp}, _sps_pos(), target='SPS H4 FEU 03')
        return dict(mode=mode, remap=rec['planes'])
    m = json.load(open(H4_MASK))
    key = 'dead_stripes' if mode == 'h4_stripes' else 'dead'
    cal.dead = {p: [int(c) for c in m[p][key]] for p in ('x', 'y')}
    cal.hot = {}
    cal.gain = {p: np.ones(512) for p in ('x', 'y')}
    return dict(mode=mode, file=os.path.relpath(H4_MASK, REPO), n_dead={p: len(cal.dead[p]) for p in ('x', 'y')})
