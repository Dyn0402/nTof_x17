#!/usr/bin/env python3
"""ga_heldout.py -- run rerun/model_audit/heldout_arm.py unchanged on a numpy-free cache copy (condor's LCG_105 has
numpy 1.x and cannot unpickle numpy-2 arrays): pickle.load is wrapped so the window lists become arrays again."""
import os, pickle, runpy, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
_orig = pickle.load


def _load(f, *a, **k):
    E = _orig(f, *a, **k)
    if isinstance(E, dict):
        for e in E.values():
            if isinstance(e, dict):
                for p in 'xy':
                    if isinstance(e.get(p), dict):
                        for key in ('ch', 'pos', 'W', 'noise', 'missing'):
                            if key in e[p] and not isinstance(e[p][key], np.ndarray):
                                e[p][key] = np.asarray(e[p][key], dtype=int if key == 'ch' else (bool if key == 'missing' else float))
    return E


pickle.load = _load
sys.argv = [os.path.join(REPO, 'rerun', 'model_audit', 'heldout_arm.py')] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
