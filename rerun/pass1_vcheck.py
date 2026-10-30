#!/usr/bin/env python3
"""pass1_vcheck.py <dets e.g. mx17_2,mx17_3> -- sanity of the vrefit bundles (pass 1, lxplus): per chamber the kw range, the
folded v by drift field, and the rows with kw outside 0.85-1.15 or v < 30 um/ns at 1000 V.  Reads the archive registry."""
import csv, json, os, sys
A = "/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1"
dets = sys.argv[1].split(',')
rows = [r for r in csv.DictReader(open(os.path.expanduser("~/x17rerun/code_6a0bb1f/rerun/pass_registry.csv")))
        if r["refit"] == "v" and r["det"] in dets]
bad, vals = [], []
for r in rows:
    p = os.path.join(A, r["run"], r["subrun"], r["det"], "wft", "calib_bundle", "bundle.json")
    if not os.path.exists(p):
        bad.append(r["key"]); continue
    b = json.load(open(p))
    vals.append((r["det"], r["drift_V"], round(b["v_drift"], 2), b["kw"]["x"], b["kw"]["y"], bool(b["bow"])))
print("rows", len(rows), "missing bundles", len(bad), bad[:3])
for d in dets:
    v = [x for x in vals if x[0] == d]
    if not v: continue
    print(d, "n", len(v), "bow carried", sum(x[5] for x in v),
          "kw_x %.3f-%.3f kw_y %.3f-%.3f" % (min(x[3] for x in v), max(x[3] for x in v), min(x[4] for x in v), max(x[4] for x in v)))
    for dv in sorted(set(x[1] for x in v), key=float):
        vs = [x[2] for x in v if x[1] == dv]
        print("   drift %6s V: n=%2d folded v %.2f-%.2f um/ns" % (dv, len(vs), min(vs), max(vs)))
print("outliers (kw outside 0.85-1.15 or 1000 V with v < 30):")
for r in rows:
    p = os.path.join(A, r["run"], r["subrun"], r["det"], "wft", "calib_bundle", "bundle.json")
    if not os.path.exists(p): continue
    b = json.load(open(p))
    kx, ky = b["kw"]["x"], b["kw"]["y"]
    if not (0.85 < kx < 1.15 and 0.85 < ky < 1.15) or (r["drift_V"] == "1000" and b["v_drift"] < 30):
        n = json.load(open(p.replace("calib_bundle/bundle.json", "vrefit/events0.meta.json"))).get("n_events") if os.path.exists(p.replace("calib_bundle/bundle.json", "vrefit/events0.meta.json")) else "?"
        print("  %-70s cls=%-11s v=%.2f kw %.3f/%.3f n=%s" % (r["key"][5:75], r["cls"], b["v_drift"], kx, ky, n))
