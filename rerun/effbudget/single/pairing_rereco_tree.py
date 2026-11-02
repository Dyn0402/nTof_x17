#!/usr/bin/env python3
"""pairing_rereco_tree.py --rereco DIR --pass1 R --out T KEY=REL ... -- lay pairing_rereco.py 'gate' outputs out as an
Analysis_pass1 tree (events{,.candidates}.parquet + events.meta.json), with the row's budget rays, so pairing_replay.py
(--all-rows --root T --bundles R) and pairing_m3_all.py (--root T) score a waveform re-reconstruction in both modes.

Used for det6/det7 (PAIRING_FIX_2026-11-02.md §7): once _n_gated counts in 'gate' (wft.reco, 2026-11-02), a 'cost'
re-reconstruction differs from the 'gate' one only in select_tracks, which the replay applies exactly, so the waveform
pass is needed once (gate) to check the inputs reproduce pass 1.
"""
import argparse, json, os, shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rereco', required=True); ap.add_argument('--pass1', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('rows', nargs='+')
    a = ap.parse_args()
    for kr in a.rows:
        key, rel = kr.split('=', 1)
        src = os.path.join(a.rereco, key, 'gate')
        dst = os.path.join(a.out, 'Analysis_pass1', rel, 'wft')
        os.makedirs(dst, exist_ok=True)
        for f in ('events.parquet', 'events.candidates.parquet'):
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
        m = json.load(open(os.path.join(src, 'events.meta.json')))
        m['calibration'] = f'/Analysis_pass1/{rel}/wft/calib_bundle'      # resolved under --bundles by pairing_replay
        json.dump(m, open(os.path.join(dst, 'events.meta.json'), 'w'), indent=1, default=str)
        det = rel.split('/')[-1].replace('mx17_', 'det')
        rp = os.path.join('x17rerun/paper1_eff/budget', rel, f'rays_{det}.parquet')
        os.makedirs(os.path.dirname(os.path.join(a.out, rp)), exist_ok=True)
        shutil.copyfile(os.path.join(a.pass1, rp), os.path.join(a.out, rp))
        print(key, '->', dst)


if __name__ == '__main__':
    main()
