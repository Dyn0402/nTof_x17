#!/usr/bin/env python3
"""look_lobes.py <npz> ...: per channel, the rate of events where it peaks above 300 ADC (ON) and the fraction of those
ON events in which it shows >= 3 lobes (periodic), for the channels whose ON rate is highest relative to their live
neighbours.  Tells oscillators (periodic) from beam strips (one pulse)."""
import os, sys
import numpy as np

CODE = os.environ['WFT_REPO']
sys.path[:0] = [os.path.join(CODE, 'rerun', 'model_audit')]
import sps_build_cache as sbc   # noqa: E402
from det4_sps_map import POSITION_MM, VIEW   # noqa: E402

for f in sys.argv[1:]:
    d = np.load(f); pk, lb = d['peak'], d['lobes']
    on = pk > 300; r = on.mean(0); per = ((lb >= 3) & on).mean(0)
    fr = np.where(on.sum(0) > 0, ((lb >= 3) & on).sum(0) / np.maximum(on.sum(0), 1), 0)
    rows = []
    for c in range(512):
        if VIEW[c] not in ('x', 'y') or r[c] < 0.005:
            continue
        nb = sbc.live_neighbours(c)
        rows.append((r[c] / max(np.median(r[nb]) if nb else 0, 1 / len(pk)), c))
    rows.sort(reverse=True)
    print(f'== {os.path.basename(f)} n={len(pk)}; whole view: ON events with >= 3 lobes = {((lb >= 3) & on).sum() / max(on.sum(), 1):.4f}')
    print('   ch view   pos     ON    ratio_nb  frac>=3lobes  per-event periodic')
    for ratio, c in rows[:14]:
        print(f'  {c:4d}  {VIEW[c]}  {POSITION_MM[c]:7.2f}  {r[c]:.3f}  {ratio:7.1f}    {fr[c]:.3f}       {per[c]:.4f}')
