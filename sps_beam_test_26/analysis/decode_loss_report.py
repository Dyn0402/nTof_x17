#!/usr/bin/env python3
"""Report the data loss recorded in decoded ROOT files, and hand back the
per-sample acceptance any waveform average has to be divided by.

The decoder writes a `decode_stats` tree and a `sample_acceptance` histogram
into every file it produces.  This reads them back, because the loss is
otherwise invisible: a DREAM FEU under RAW bandwidth pressure drops
sample-group packets silently, exits 0, and the missing samples are
indistinguishable from genuinely quiet channels.

  python decode_loss_report.py <dataset|glob> [--csv out.csv]

  from decode_loss_report import acceptance_for
  acc = acceptance_for(glob.glob(".../dec_*.root"))   # 64-bin array, or None
"""
from __future__ import annotations

import argparse
import glob as globmod
import os
import sys

import numpy as np
import uproot


FIELDS = ("events", "events_missing", "closed_eoe", "closed_eventid",
          "closed_eof", "samples_expected", "raw_mode",
          "sample_acceptance_mean")


def read_stats(path):
    """(dict, acceptance array) for one decoded file; (None, None) if absent."""
    try:
        F = uproot.open(path)
    except Exception as e:                                    # noqa: BLE001
        return {"error": str(e)}, None
    keys = {k.split(";")[0] for k in F.keys()}
    if "decode_stats" not in keys:
        return None, None
    st = F["decode_stats"].arrays(library="np")
    d = {k: st[k][0] for k in FIELDS if k in st}
    acc = None
    if "sample_acceptance" in keys:
        acc = F["sample_acceptance"].to_numpy()[0]
    return d, acc


def acceptance_for(paths):
    """Event-weighted mean per-sample acceptance over several decoded files.

    Returns None when the files carry no stats (decoded before the decoder
    recorded them) -- in which case the caller must NOT silently assume 1.0.
    """
    num = None
    den = 0.0
    for p in paths:
        d, acc = read_stats(p)
        if d is None or acc is None or "events" not in d:
            continue
        n = float(d["events"])
        num = acc * n if num is None else num + acc * n
        den += n
    return None if den == 0 else num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="a datasets.py name, or a glob of ROOT files")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    paths = sorted(globmod.glob(args.target))
    if not paths:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import datasets
            D = datasets.get(args.target)
            paths = sorted(globmod.glob(D["stage"] + "**/dec_*.root",
                                        recursive=True))
        except Exception:                                     # noqa: BLE001
            pass
    if not paths:
        raise SystemExit(f"no decoded files matched {args.target!r}")

    print(f"{len(paths)} decoded files\n")
    hdr = (f"{'file':<44}{'events':>10}{'missing':>9}{'by EoE':>10}"
           f"{'by evtID':>10}{'accept':>8}")
    print(hdr)
    print("-" * len(hdr))
    tot = dict.fromkeys(("events", "events_missing", "closed_eoe",
                         "closed_eventid"), 0)
    nostats = 0
    for p in paths:
        d, acc = read_stats(p)
        name = os.path.basename(p)[:43]
        if d is None:
            print(f"{name:<44}{'-- no decode_stats (decoded before it existed)':>47}")
            nostats += 1
            continue
        if "error" in d:
            print(f"{name:<44}  unreadable: {d['error'][:40]}")
            continue
        for k in tot:
            tot[k] += int(d.get(k, 0))
        a = float(d.get("sample_acceptance_mean", 1.0))
        raw = int(d.get("raw_mode", 0))
        print(f"{name:<44}{int(d['events']):>10}{int(d['events_missing']):>9}"
              f"{int(d['closed_eoe']):>10}{int(d['closed_eventid']):>10}"
              f"{(f'{a:.3f}' if raw else '   n/a'):>8}")

    print("-" * len(hdr))
    print(f"{'TOTAL':<44}{tot['events']:>10}{tot['events_missing']:>9}"
          f"{tot['closed_eoe']:>10}{tot['closed_eventid']:>10}")

    acc = acceptance_for(paths)
    if acc is not None:
        loss = 1.0 - acc.mean()
        print(f"\nevent-weighted per-sample acceptance: mean {acc.mean():.4f} "
              f"(min {acc.min():.3f}, max {acc.max():.3f})")
        if loss > 0.001:
            print(f"\n  !! {100*loss:.1f} % of sample-groups were never shipped by the FEU.")
            print("  !! Divide every mean waveform by this acceptance array, or")
            print(f"  !! every dispersed-copy amplitude comes out ~{100*loss:.0f} % low.")
        flat = acc.max() - acc.min()
        print(f"  acceptance is {'FLAT' if flat < 0.10 else 'NOT flat'} "
              f"across the window (spread {flat:.3f})"
              + ("" if flat < 0.10 else "  <-- a shape bias, not just a rate loss"))
    if nostats:
        print(f"\n  {nostats} file(s) predate the loss accounting -- re-decode "
              f"them rather than assuming they are clean.")

    if args.csv and acc is not None:
        np.savetxt(args.csv, np.c_[np.arange(len(acc)), acc],
                   delimiter=",", header="sample,acceptance", comments="")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
