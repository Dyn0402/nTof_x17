"""The pass-1 bench rows of the M3 per-track pointing study, and where their inputs are kept."""
import os

WORK = os.path.join(os.environ.get('WORK_DATA', os.path.expanduser('~/x17_local')), 'nTof_x17', 'm3_pointing')
INPUTS = os.path.join(WORK, 'inputs')
#: effbudget per-ray tables of pass 1 (rerun/effbudget/pass1; a local copy of EOS x17rerun/paper1_eff/budget)
RAYS = os.environ.get('M3P_RAYS', '/media/ucla/work/paper1_figs/budget')
JT = '/eos/experiment/ntof/data/x17/cosmic_bench/june_tests'
SHORT = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
#: public names (Paper I): A = det3, B = det2, C = det6, D = det7, E = det4
PUBLIC = {'mx17_3': 'A', 'mx17_2': 'B', 'mx17_6': 'C', 'mx17_7': 'D', 'mx17_4': 'E'}


def _row(tag, run, subrun, det, pair=None):
    return dict(tag=tag, run=run, subrun=subrun, det=det, eos_run_dir=f'{JT}/{run}', pair=pair)


ROWS = [
    # chamber A golden runs (top slot): Run 1 and Run 2
    _row('A_run1', 'mx17_det3_saturday_scan_6-27-26', 'long_run_resist_490V_drift_1000V', 'mx17_3'),
    _row('A_run2', 'mx17_det3_p2_det1_overnight_6-27-26', 'long_run_p2_det1_sanity_check', 'mx17_3'),
    # two chambers in the telescope at once: one in each slot (the pair tag joins them)
    _row('B_622', 'mx17_det2_det3_overnight_6-22-26', 'long_run', 'mx17_2', pair='p622'),
    _row('A_622', 'mx17_det2_det3_overnight_6-22-26', 'long_run', 'mx17_3', pair='p622'),
    _row('A_623', 'mx17_det3_det4_overnight_6-23-26', 'long_run', 'mx17_3', pair='p623'),
    _row('E_623', 'mx17_det3_det4_overnight_6-23-26', 'long_run', 'mx17_4', pair='p623'),
    _row('C_626', 'mx17_det6_det7_overnight_6-26-26', 'long_run', 'mx17_6', pair='p626'),
    _row('D_626', 'mx17_det6_det7_overnight_6-26-26', 'long_run', 'mx17_7', pair='p626'),
    _row('E_624', 'mx17_det4_day_6-24-26', 'long_run', 'mx17_4'),
]
