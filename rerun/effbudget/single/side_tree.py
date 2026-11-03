"""side_tree.py -- read the chamber A two-track re-pass (EOS Analysis_pass1_twotrack, bundles select=cost +
split_corrob=charge) in place of Analysis_pass1, for the det3 single-track scripts (--analysis NAME).

The ray table (budget/rays_det3.parquet) and the A1 classes (recat.parquet) come from pass 1 and are not regenerated
(budget.py / a1_job.py need the M3 and hit trees, i.e. a condor stage job).  What the re-pass changes is the candidate
table, so the two columns of the ray table that are functions of the candidates alone are recomputed with
budget.cand_arms on BOTH trees, and a ray's stored value is replaced only where the two trees disagree:
  det_any_5   some sane ranked candidate within 5 mm in both views (the multi-track credit)
  detected    rank 0 within 5 mm in both views, n_dropped <= MAX_DROPPED (cat_xy_5 == 'detected'; closure checked)
A ray that stops being 'detected' is labelled 'far' (the budget ladder's next free label); only single_track_area's
credit uses it.  Default tree (Analysis_pass1): returns the table unchanged.
"""
import os, sys
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PASS1 = 'Analysis_pass1'


def cand_path(root, analysis, rel):
    return os.path.join(root, analysis, rel, 'wft/events.candidates.parquet')


def overlay(root, analysis, rel, B, verbose=True):
    """B: rays_det3 table indexed by event_id (as read by the scripts). Returns a copy with the two columns updated."""
    if analysis == PASS1:
        return B
    sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'effbudget')]
    import budget  # noqa: E402  (cand_arms)
    arms = {}
    for nm, an in (('p1', PASS1), ('side', analysis)):
        R = B.reset_index()[['event_id', 'mesh_x', 'mesh_y']].copy()
        budget.cand_arms(R, cand_path(root, an, rel), [5.0])
        arms[nm] = R.set_index('event_id')
    B = B.copy()
    st_any = B.det_any_5.astype(bool); st_det = (B.cat_xy_5 == 'detected')
    p_any, s_any = arms['p1'].det_any_5.reindex(B.index).astype(bool), arms['side'].det_any_5.reindex(B.index).astype(bool)
    p_det, s_det = arms['p1'].det_prod_5.reindex(B.index).astype(bool), arms['side'].det_prod_5.reindex(B.index).astype(bool)
    if verbose:
        print(f'  [side tree {analysis}] closure of pass-1 candidates vs stored ray table: det_any_5 mismatches '
              f'{int((p_any != st_any).sum())}, detected mismatches {int((p_det != st_det).sum())} of {len(B)} rays')
        print(f'  [side tree] changed: det_any {int((p_any != s_any).sum())} (gained {int((~p_any & s_any).sum())}, lost '
              f'{int((p_any & ~s_any).sum())}), detected {int((p_det != s_det).sum())} (gained {int((~p_det & s_det).sum())}, '
              f'lost {int((p_det & ~s_det).sum())})')
    B['det_any_5'] = np.where(p_any != s_any, s_any, st_any)
    cat = B.cat_xy_5.astype(object).copy()
    cat[(~p_det & s_det).to_numpy()] = 'detected'
    cat[(p_det & ~s_det).to_numpy()] = 'far'
    B['cat_xy_5'] = cat
    return B
