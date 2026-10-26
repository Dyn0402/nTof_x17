#!/usr/bin/env python3
"""overlay_arm.py ARM -- re-run the satstudy bench overlay (rerun/satstudy/overlay.py: two clean det3 single tracks
summed in the waveforms, both with M3 truth) under a reconstruction arm, into multiparticle/overlay_<ARM>/.

    flock <heavy.lock> ../../.venv/bin/python overlay_arm.py off   # as the 10-26 overlay (WFT_DEAD_SEED=0)
    flock <heavy.lock> ../../.venv/bin/python overlay_arm.py tt    # + WFT_TWO_TRACK_FIT=1 (production settings, as reco_tt)

Same donors and the same pairs as satstudy/overlay (seed 7, 40 per separation bin + 30 singles). The wft switches are
read at import, so the environment is set here before anything imports wft.
"""
import os, sys

ARMS = {'off': {}, 'tt': {'WFT_TWO_TRACK_FIT': '1'}}
arm = sys.argv[1]
os.environ.update(WFT_PIVOT_START='1', WFT_COL_GUARD='0.01', WFT_DEAD_SEED='0',
                  X17_BENCH_ROOT=os.path.expanduser('~/x17/cosmic_bench/outliers/stage'), **ARMS[arm])
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mplib as M  # noqa: E402
import overlay as OV  # noqa: E402  (rerun/satstudy)

OV.OD = os.path.join(M.OUT, f'overlay_{arm}')
OV.build(40, int(os.environ.get('JOBS', '2')))
