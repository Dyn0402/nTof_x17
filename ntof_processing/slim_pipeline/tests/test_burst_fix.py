#!/usr/bin/env python3
"""slim.apply_burst_fix on a synthetic frame: bunch move, flash re-reference
in the right time base, flash re-tag, provenance in attrs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ntof_processing.slim_pipeline import clockfit as cf     # noqa: E402
from ntof_processing.slim_pipeline import slim               # noqa: E402
import ntof_dream_merge.ntof_io as io                        # noqa: E402

io.pkup_bunches = lambda r: dict(BunchNumber=np.array([201, 202, 300]),
                                 intensity_e10=np.array([500., 500., 777.]))
S = 4_383_062.6
seg = slim.Segment('run_x', 'sub', 224572, burst_fix={
    5: dict(bunch=300, flash_shift_ns=S),          # first burst mid-gate
    6: dict(flash_shift_ns=-1500.0),               # orphan ahead of the flash
    9: dict(bunch=300)})                           # not in the sub-run
ev = pd.DataFrame(dict(
    eventId=np.arange(9), burst_id=[5, 5, 5, 6, 6, 6, 7, 7, 7],
    is_flash=[True, False, False, True, False, False, True, False, False],
    t_since_flash_ns=np.array([0, 1_100_000, 2_000_000, 0, 1500, 3_000_000,
                               0, 1_000_000, 2_000_000], np.int64),
    BunchNumber=[-1, -1, -1, 201, 201, 201, 202, 202, 202],
    bunch_intensity_e10=[np.nan] * 3 + [500.] * 6, join_resid_s=[np.nan] * 9))
attrs = dict(burst_map=dict(burst_id=[5, 6, 7], bunch=[-1, 201, 202],
                            resid_ms=[None, 1.0, 2.0]))
out, at = slim.apply_burst_fix(seg, ev, attrs, log=lambda *a: None)
b5 = out[out.burst_id == 5]
assert (b5.BunchNumber == 300).all() and (b5.bunch_intensity_e10 == 777.).all()
assert b5.t_since_flash_ns.iloc[0] == round(S / (1 + cf.K_SEED)), \
    'flash shift must be converted from n_TOF ns to DREAM ns'
assert not b5.is_flash.any(), 'the stand-in flash is freed after a + shift'
b6 = out[out.burst_id == 6]
assert list(b6.is_flash) == [True, True, False], \
    'orphan stays tagged, true flash (t~0) tagged, physics untouched'
assert (out[out.burst_id == 7].t_since_flash_ns ==
        [0, 1_000_000, 2_000_000]).all()
assert set(at['burst_fix']) == {'5', '6'} and at['burst_fix']['5']['was_bunch'] == -1
assert at['burst_map']['bunch'] == [300, 201, 202] and '5' in at['burst_map']['fix']
print('all burst_fix cases behaved as specified')
