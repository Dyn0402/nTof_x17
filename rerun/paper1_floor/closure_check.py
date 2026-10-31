"""closure_check.py -- the frames against budget_pass1.json (tilt_2d40, unexplained) + the finer-cell tilt x-validation (README)."""
import numpy as np

import floorlib as C

B = C.budget()['new']
for det, key in C.CALIB.items():
    F = C.load(key)
    for p in ('x', 'y'):
        g = C.sample(F, p)
        b = B.get(f'{det}|calib|vt|{p}|ig') or {}
        un = C.q(C.q(C.s68(g.dt), b.get('toy_all') or 0.0), 0.05)
        half = (g.event_id % 2).to_numpy()
        fine = {}
        for c in (20, 10, 5):
            fine[c] = C.s68(C.xval(g.ds.to_numpy(), (np.floor(g.p0 / c).astype(int), np.floor(g.o_p0 / 40).astype(int)),
                                   half))
        print(f"{det} {p} n={len(g)} s68 {C.s68(g.d_vt):.3f} slope {C.s68(g.ds):.3f} 2d40 {C.s68(g.dt):.3f} "
              f"(budget {b.get('tilt_2d40', float('nan')):.3f}) unexpl {un:.3f} (budget {b.get('unexplained', float('nan')):.3f}) "
              f"finer-own-cell {fine}")
