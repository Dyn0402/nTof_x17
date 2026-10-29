#!/usr/bin/env python3
"""bow_node.py -- the pass's BOW node (PASS_PLAN E2): the chamber bow block from the golden row, on EOS inputs.

One block per detector, measured on its golden calibration row and inherited by every row of that detector (Dylan
10-29: no slot- or run-dependent bow; extract on the golden set, apply to the others).  Rows re-measure only w0/kw
(set_w0kw --bow-json), as every row does anyway.  Evidence: rerun/calib/bow/BOW_TRANSFER.md (bow_finalize.py).

Inputs (the CAL3 products of the golden row): the reconstruction table (events.parquet), its waveform alignment
(alignment.json), the row's v2 M3 rays directory, and the bundle the table was reconstructed with.  The table is joined
with M3 exactly as ws_v_w0/build_table does; the block is set_bow.block_from_table: tan-form drift tilt a_x/a_y
(wft.reco.tan_bow), per-view mesh height h_x/h_y (wft.reco.bow_h), scale gradients s_x/s_y.

  bow_node.py <qa_config key> <det> --events E --alignment A --m3dir M --bundle B --out bow_block_<det>.json
"""
import argparse, json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
FB = os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fb_bow_maps')
sys.path[:0] = [REPO, FB, os.path.join(REPO, 'rerun', 'model_audit', 'ws_v_w0')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('det')
    ap.add_argument('--events', required=True); ap.add_argument('--alignment', required=True)
    ap.add_argument('--m3dir', required=True); ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--table-out', default=None, help='keep the joined per-event table here (default: a temp file)')
    a = ap.parse_args()
    from build_table import build
    from fbcommon import load_path
    from set_bow import block_from_table
    tab = a.table_out or os.path.join(tempfile.mkdtemp(prefix='bow_node_'), 'table.parquet')
    build(a.det, key=a.key, table=a.events, alignment=a.alignment, out=tab, m3dir=a.m3dir)
    b = json.load(open(os.path.join(a.bundle, 'bundle.json')))
    T = load_path(tab, b, a.det, name=a.key)
    blk = block_from_table(T, a.det, a.key)
    blk['provenance'] = dict(node='rerun/calib/bow_node.py', key=a.key, events=a.events, alignment=a.alignment,
                             m3dir=a.m3dir, bundle=a.bundle)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(blk, open(a.out, 'w'), indent=1)
    print(f'{a.det}: bow block -> {a.out}  (n {blk["n_events"]}; h_x kuu {blk["h_x"]["kuu"]:.2f}, h_y kww {blk["h_y"]["kww"]:.2f} mm)')


if __name__ == '__main__':
    main()
