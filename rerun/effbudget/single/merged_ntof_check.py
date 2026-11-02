#!/usr/bin/env python3
"""merged_ntof_check.py TABLE.candidates.parquet BUNDLE_DIR [...] [--json OUT]
-- how often the 'charge' split corroboration would fire on n_TOF candidate tables (MERGED_PROJECTION_2026-11-02.md §5).

A table-level replay: per event, the pre-split candidate lists (one-track fits; a split parent restored in place of its
children), the candidates the production selector reconsiders (select_tracks 'gate' without pairing, as
resolve_two_tracks does), and for each of those whether the 'count' rule (_cross_plane_mismatch) and the 'charge' rule
(_charge_corroboration, with the table's bundle xy_pairing) would corroborate a split.  An event where only 'charge'
fires is one the mode could change (an attempt the 'count' rule would not have made at F_corrob, or a refused split
that one plausible child could now let through); whether the fit then accepts it needs the waveforms.  No truth on
beam: this sizes the change, it does not say whether it is right.
"""
import argparse, json, os, sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from wft import reco  # noqa: E402
from wft.calib import CalibrationBundle  # noqa: E402
from rerun.effbudget.single.pairing_replay import fits_of, ftst_diff_of  # noqa: E402


def check(path, bdir):
    cal = CalibrationBundle.load(bdir)
    reco.wm.use_calibration(cal)
    pairing = cal.xy_pairing
    reco.TC_TOL_NS = reco.tc_tol(pairing)
    C = pd.read_parquet(path)
    # pre-split lists: drop split children, put the parent back at the children's best rank
    kids = C[(C['rank'] >= 0) & C.split_child]
    par = C[C.split_replaced].copy()
    if len(par):
        best = kids.groupby(['event_id', 'plane'])['rank'].min()
        par['rank'] = [best.get((e, p), 99) for e, p in zip(par.event_id, par.plane)]
    C = pd.concat([C[(C['rank'] >= 0) & ~C.split_child], par])
    key = ['acq_tag', 'event_id'] if 'acq_tag' in C else ['event_id']
    n = dict(events=0, count_fires=0, charge_fires=0, charge_only=0, charge_only_implausible_other=0)
    for _k, g in C.groupby(key, sort=False):
        n['events'] += 1
        fits = fits_of(g)
        fd = ftst_diff_of(g)
        try:
            pre = reco.select_tracks(fits, fd, cal, max_tracks=reco.MAX_TRACKS)
        except Exception:
            pre = []
        sel = {'x': {i for i, _j, _g in pre}, 'y': {j for _i, j, _g in pre}}
        cf = qf = False
        imp = False
        for p, o in (('x', 'y'), ('y', 'x')):
            cnt = reco._cross_plane_mismatch(fits[p], fits[o])
            for r, f in enumerate(fits[p][:reco.TWO_TRACK_MAX_TRY + 2]):
                if r not in sel[p]:
                    continue
                cf |= cnt
                c = reco._charge_corroboration(f, p, fits[p], fits[o], pairing, fd)
                if c is not None:
                    qf = True
                    imp |= not (c['a']._plausible and c['b']._plausible)
        n['count_fires'] += cf
        n['charge_fires'] += qf
        n['charge_only'] += qf and not cf
        n['charge_only_implausible_other'] += qf and not cf and imp
    n['charge_only_pct'] = 100.0 * n['charge_only'] / max(n['events'], 1)
    return dict(table=path, bundle=bdir, split_corrob=reco.split_corrob_mode(pairing), **n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='+', help='TABLE BUNDLE_DIR ...')
    ap.add_argument('--json')
    a = ap.parse_args()
    res = [check(t, b) for t, b in zip(a.pairs[::2], a.pairs[1::2])]
    s = json.dumps(res, indent=1)
    print(s)
    if a.json:
        open(a.json, 'w').write(s)


if __name__ == '__main__':
    main()
