#!/usr/bin/env python3
"""Closure check of a flash product against the official PSA hits of its run.

    python -m ntof_processing.flash_pull.verify flash_224666_0054.root [...]

Every official hit of a bunch in the file, on a kept channel, with tof inside
config.WINDOWS_NS must sit inside a kept block of the same (bunch, det, detn).
Shares nothing with the pull but the file: the hits come from the processed
ROOT, the blocks from the product.  `edge` is a hit within EDGE_NS past a block
end, the waveform pull's measured PRE_SAMPLES convention (2 ns on PSSD); it is
reported and does not fail.  Writes <stem>_verify.json beside the product.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import uproot

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'ntof_processing.flash_pull'

from . import config as F                            # noqa: E402

EDGE_NS = 3


def verify(path: Path) -> dict:
    meta = json.loads(path.with_name(path.stem + '_provenance.json').read_text())
    run = meta['run']
    f = uproot.open(path)
    b = f['blocks'].arrays(['bunch', 'det', 'detn', 'tof0', 'n'], library='np')
    # a bunch the pull flagged as cut short (truncated file) is reported, not checked
    skipped = set(meta.get('bunches_incomplete', []))
    bunches = set(f['bunches']['bunch'].array(library='np').tolist()) - skipped
    off = uproot.open(F.OFFICIAL / f'run{run}.root')
    res = {'run': run, 'file': path.name, 'bunches_checked': len(bunches),
           'bunches_skipped_incomplete': sorted(skipped), 'per_det': {}}
    tot = dict(hits=0, inside=0, edge=0, missing=0)
    for det, code in F.DET_CODE.items():
        if det not in [k.split(';')[0] for k in off.keys()]:
            continue
        h = off[det].arrays(['BunchNumber', 'detn', 'tof'], library='np')
        m = np.isin(h['BunchNumber'], list(bunches))
        inwin = np.zeros(m.sum(), bool)
        tof = h['tof'][m]
        for lo, hi in F.WINDOWS_NS:
            inwin |= (tof >= lo) & (tof < hi)
        hb, hn, ht = h['BunchNumber'][m][inwin], h['detn'][m][inwin], tof[inwin]
        sel = b['det'] == code
        # lexicographic key (bunch, detn, tof0): blocks of one channel are disjoint
        key_b = b['bunch'][sel].astype(np.int64) * 1000 + b['detn'][sel]
        o = np.lexsort((b['tof0'][sel], key_b))
        kb, t0, t1 = key_b[o], b['tof0'][sel][o], b['tof0'][sel][o] + b['n'][sel][o]
        kh = hb.astype(np.int64) * 1000 + hn
        # last block of the same channel starting at or before the hit
        pos = np.searchsorted(kb * 10**9 + t0, kh * 10**9 + np.floor(ht).astype(np.int64), side='right') - 1
        ok = pos >= 0
        pos = np.clip(pos, 0, max(len(kb) - 1, 0))
        same = ok & (kb[pos] == kh) if len(kb) else np.zeros(len(kh), bool)
        inside = same & (ht < t1[pos]) if len(kb) else same
        edge = same & ~inside & (ht < t1[pos] + EDGE_NS) if len(kb) else same
        r = dict(hits=int(len(kh)), inside=int(inside.sum()), edge=int(edge.sum()),
                 missing=int(len(kh) - inside.sum() - edge.sum()))
        res['per_det'][det] = r
        for k in tot:
            tot[k] += r[k]
    res.update(tot, passed=tot['missing'] == 0)
    path.with_name(path.stem + '_verify.json').write_text(json.dumps(res, indent=2))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('products', nargs='+', type=Path)
    a = ap.parse_args(argv)
    bad = 0
    for p in a.products:
        r = verify(p)
        print(f"{p.name}: {r['hits']} hits in window, inside {r['inside']}, edge {r['edge']}, "
              f"missing {r['missing']} -> {'PASS' if r['passed'] else 'FAIL'}")
        bad += not r['passed']
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
