#!/usr/bin/env python3
"""rule_on_lobes.py <dev dir> <npz> ...: the builder's oscillator rules (near_oscillators, coupled_channels; seed-hot is
not evaluated here) on the full per-event peaks of lobes.py, starting from SEED_BAD."""
import os, sys
import numpy as np

sys.path.insert(0, sys.argv[1])
import sps_build_cache as sbc   # noqa: E402

for f in sys.argv[2:]:
    pk = np.load(f)['peak']
    near = sbc.near_oscillators(pk, sbc.SEED_BAD.copy())
    osc = sbc.SEED_BAD | near
    print(os.path.basename(f), 'near oscillators', list(np.flatnonzero(near)), 'coupled', sbc.coupled_channels(pk, osc))
