#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmic_tracks.py -- beam-off cosmic sub-runs through the campaign stage 3, and
the first through-going numbers.  HANDOFF_TRACKING_2026-10-02.md §2.

THE RECO is the campaign's full pass, unchanged: `make_stage2_campaign.py
--full-pass --tags-json` (no stage 1 -- cosmic runs have no n_TOF slim), same
bundles, v_drift pinned at 42.6 um/ns, written to its OWN EOS directory
(`cosmics_fullpass`) so nothing can land in the campaign's.

THE ANGLE SCALE IS BORROWED.  `k_arm` assumes tracks come from the capsule,
which through-goers do not, so a cosmic run cannot measure its own k (handoff
§1b) -- and `k_arm` has no --out and would overwrite kcal/ silently, so it is
never run here.  Each `--k-from` run gives one track table; quote everything
under both neighbours and read any shift as the systematic.

NOTHING HERE WRITES OUTSIDE THIS PACKAGE.  The campaign defaults resolve to the
data disk (`paths.out`); every product below goes to `results/tracking/` (the
parquet is gitignored) and the tarballs to `--tarballs`.

    python ntof_cosmics/cosmic_tracks.py fetch   --run run_149 --subrun cosbounce_cos_0000 --tarballs <dir>
    python ntof_cosmics/cosmic_tracks.py build   --run run_149 --subrun cosbounce_cos_0000 --k-from run_147,run_150
    python ntof_cosmics/cosmic_tracks.py analyse --run run_149 --subrun cosbounce_cos_0000 --k-from run_147,run_150
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.acceptance import topology  # noqa: E402

OUT = HERE / 'results' / 'tracking'
EOS_OUT = '/eos/user/d/dneff/x17/cosmics_fullpass'
ARMS = ('A', 'B', 'C', 'D')
#: the beam-on per-track pointing preselection (`source_imaging._track_table`)
DCA_MAX = 30.0
#: `tight_coincidence.BACK_TO_BACK_DEG`, the cut these numbers are meant to set
BACK_TO_BACK_DEG = 170.0
MATCH_NS = 50.0


def _guard(p: Path) -> Path:
    """Output-path hook (the data disk is allowed since 2026-10-07)."""
    return p


def reco_dir(run, sub):
    return _guard(OUT / 'reco' / run / sub)


def tracks_path(run, sub, k_from):
    return _guard(OUT / f'k_{k_from}' / f'tracks_{run}_{sub}.parquet')


# --------------------------------------------------------------------------- #
def fetch(run, sub, tarballs: Path):
    tarballs = _guard(tarballs)
    tarballs.mkdir(parents=True, exist_ok=True)
    pat = f'{run}_{sub}_beam_*.tar.gz'
    subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes',
                    '--include', pat, '--exclude', '*',
                    f'lxplus:{EOS_OUT}/', f'{tarballs}/'], check=True)
    got = sorted(tarballs.glob(pat))
    d = reco_dir(run, sub)
    d.mkdir(parents=True, exist_ok=True)
    for t in got:
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                # strip the leading out/ so mx17_<arm>/ sits under the sub-run
                parts = Path(m.name).parts
                if len(parts) < 2 or parts[0] != 'out':
                    continue
                m.name = str(Path(*parts[1:]))
                tf.extract(m, d)
    print(f'{len(got)} tarball(s) -> {d}')
    return len(got)


def _k(src_run):
    kf = paths.out('kcal') / f'k_arm_{src_run}.json'
    cal = json.loads(kf.read_text())
    return {a: float(v) for a, v in (cal.get('apply') or {}).items()}


def build(run, sub, k_from):
    from sept26_prelim_analysis import build_tracks as BT
    dest = tracks_path(run, sub, k_from).parent
    dest.mkdir(parents=True, exist_ok=True)
    k = _k(k_from)
    print(f'--- k from {k_from}: {k}')
    tracks, meta = BT.build(run, sub, reco_dir(run, sub), stage1=None,
                            allow=None, out_dir=dest, k_arm=k)
    tracks['k_source'] = k_from
    tracks.to_parquet(tracks_path(run, sub, k_from), index=False)
    return tracks


# --------------------------------------------------------------------------- #
def n_triggers(run, sub) -> int:
    inv = pd.read_csv(HERE / 'results' / 'cosmic_subruns.csv')
    r = inv[(inv.run == int(run.split('_')[1])) & (inv.subrun == sub)]
    return int(r.events.iloc[0])


def _axis_dca(P, D):
    """Closest approach of lines (P, D) to the beam axis (global y): the
    `build_tracks.pointing` estimator, so the cosmic and beam-on numbers are
    the same quantity."""
    p, d = P[:, [0, 2]], D[:, [0, 2]]
    dd = np.einsum('ij,ij->i', d, d)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -np.einsum('ij,ij->i', p, d) / np.where(dd > 1e-12, dd, np.nan)
    c = P + s[:, None] * D
    return np.hypot(c[:, 0], c[:, 2]), c[:, 1]


def pairs(t: pd.DataFrame) -> pd.DataFrame:
    """Every pair of gated tracks in DIFFERENT arms of one trigger.

    open_deg is `source_imaging._vertex_frame`'s (arccos of d1.d2), so the
    170 deg cut means the same here as on the beam-on sample.  The JOINED line
    is the one through the two chambers' track points: a straight through-goer
    must lie on it, so its axis DCA is the capsule-vertex handle, and each
    chamber's angle to it is the capsule-free angle-scale check (handoff §1b).
    """
    from sept26_prelim_analysis.source_imaging import _dca_two_lines
    g = t[t.gated & t.angle_calibrated].reset_index(drop=True)
    L, R = [], []
    for _, e in g.groupby('event_id'):
        ix = e.index.to_numpy()
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                if e.arm.iat[a] != e.arm.iat[b]:
                    L.append(ix[a])
                    R.append(ix[b])
    a, b = g.loc[L].reset_index(drop=True), g.loc[R].reset_index(drop=True)
    sw = (a.arm > b.arm).to_numpy()
    a.loc[sw], b.loc[sw] = b.loc[sw].to_numpy(), a.loc[sw].to_numpy()
    P1, D1 = a[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), a[['d_x', 'd_y', 'd_z']].to_numpy(float)
    P2, D2 = b[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), b[['d_x', 'd_y', 'd_z']].to_numpy(float)
    V, sep = _dca_two_lines(P1, D1, P2, D2)
    J = P2 - P1
    J /= np.linalg.norm(J, axis=1)[:, None]
    jdca, jy = _axis_dca(P1, J)

    def to_joined(D):
        return np.degrees(np.arccos(np.abs(np.einsum('ij,ij->i', D, J)).clip(0, 1)))

    out = pd.DataFrame(dict(
        event_id=a.event_id.to_numpy(), arm1=a.arm.to_numpy(), arm2=b.arm.to_numpy(),
        track1=a.track_id.to_numpy(), track2=b.track_id.to_numpy(),
        open_deg=np.degrees(np.arccos(np.einsum('ij,ij->i', D1, D2).clip(-1, 1))),
        sep_mm=sep, v_r=np.hypot(V[:, 0], V[:, 2]), vy=V[:, 1],
        joined_dca_mm=jdca, joined_y_mm=jy,
        dev1_deg=to_joined(D1), dev2_deg=to_joined(D2),
        joined_vertical_deg=np.degrees(np.arccos(np.abs(J[:, 1]))),
        dca1_mm=a.dca_axis_mm.to_numpy(), dca2_mm=b.dca_axis_mm.to_numpy()))
    out['topo'] = [topology(x, y) for x, y in zip(out.arm1, out.arm2)]
    out['pair'] = out.arm1 + out.arm2
    out['beam_presel'] = (out.dca1_mm < DCA_MAX) & (out.dca2_mm < DCA_MAX)
    out['back_to_back'] = (out.topo == 'opposing') & (out.open_deg > BACK_TO_BACK_DEG)
    return out


#: "one straight line": the two chambers' lines pass within this of each other
CLEAN_SEP_MM = 20.0


def slope_check(t: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """Each chamber's track slope against the line through both chambers.

    Opposing chambers only (A-C, B-D), on CLEAN pairs.  Slopes are taken
    against the chamber normal (global z for A/C, x for B/D), in the two
    in-plane directions.  A ratio of 1 means the borrowed k is right for that
    chamber with NO capsule assumption; k_true = k_borrowed * ratio.

    Selecting on sep_mm favours pairs whose lines agree, which pulls the
    ratio toward 1 rather than away from it -- a departure is conservative.
    """
    ti = t.set_index(['event_id', 'arm', 'track_id'])
    rows = []
    for pair, normal, inplane in (('AC', 'z', 'xy'), ('BD', 'x', 'zy')):
        c = P[(P.pair == pair) & (P.sep_mm < CLEAN_SEP_MM)]
        if c.empty:
            continue
        a = ti.loc[list(zip(c.event_id, c.arm1, c.track1))]
        b = ti.loc[list(zip(c.event_id, c.arm2, c.track2))]
        J = {ax: b[f'p0_{ax}'].to_numpy() - a[f'p0_{ax}'].to_numpy() for ax in 'xyz'}
        for arm, tr in ((pair[0], a), (pair[1], b)):
            for ax in inplane:
                s_ = tr[f'd_{ax}'].to_numpy() / tr[f'd_{normal}'].to_numpy()
                j = J[ax] / J[normal]
                m = np.abs(j) > 0.1
                rows.append(dict(arm=arm, axis=ax, n=int(m.sum()),
                                 median_ratio=float(np.median(s_[m] / j[m])) if m.any() else np.nan,
                                 lsq_ratio=float(np.sum(s_ * j) / np.sum(j * j)),
                                 corr=float(np.corrcoef(s_, j)[0, 1])))
    return pd.DataFrame(rows)


def clock(run, sub) -> pd.DataFrame | None:
    cm = sorted((HERE / 'results' / 'clock_match').glob(f'pairs_{run}_{sub}_*.csv'))
    if not cm:
        return None
    c = pd.concat([pd.read_csv(p) for p in cm], ignore_index=True)
    c['ntof_arm'] = np.array(ARMS)[c.arm.to_numpy()]
    c['matched'] = c.res.abs() <= MATCH_NS
    return c.rename(columns={'eventId': 'event_id'})[
        ['event_id', 'ntof_arm', 'res', 'matched', 'bunch']]


def analyse(run, sub, k_from) -> dict:
    t = pd.read_parquet(tracks_path(run, sub, k_from))
    ntrig = n_triggers(run, sub)
    g = t[t.gated]
    arms_per_trig = g.groupby('event_id').arm.nunique()
    per_arm = {a: int(g[g.arm == a].event_id.nunique()) for a in ARMS}
    cal = {a: bool(t[t.arm == a].angle_calibrated.any()) for a in ARMS}

    P = pairs(t)
    P['k_source'] = k_from
    c = clock(run, sub)
    if c is not None:
        P = P.merge(c, on='event_id', how='left')
        P['ntof_in_pair'] = [(m is True) and (n in (x, y)) for m, n, x, y in
                             zip(P.matched, P.ntof_arm, P.arm1, P.arm2)]
    P.to_parquet(tracks_path(run, sub, k_from).with_name(f'pairs_{run}_{sub}.parquet'),
                 index=False)

    # one pair per trigger for the trigger-level split: the most collinear
    best = P.sort_values('open_deg', ascending=False).drop_duplicates('event_id')
    s = dict(
        run=run, subrun=sub, k_from=k_from, n_triggers=ntrig,
        arms_calibrated={a: v for a, v in cal.items()},
        n_trig_ge1_track=int((arms_per_trig >= 1).sum()),
        n_trig_ge2_arms=int((arms_per_trig >= 2).sum()),
        n_trig_per_arm=per_arm,
        n_pairs=int(len(P)),
        trig_by_pair=best.pair.value_counts().to_dict(),
        open_deg_quantiles_opposing={str(q): float(best[best.topo == 'opposing']
                                                   .open_deg.quantile(q))
                                     for q in (0.05, 0.16, 0.5, 0.84, 0.95)}
        if (best.topo == 'opposing').any() else {},
        frac_opposing_above_170=float((best[best.topo == 'opposing'].open_deg
                                       > BACK_TO_BACK_DEG).mean())
        if (best.topo == 'opposing').any() else None,
        n_beam_presel=int(best.beam_presel.sum()),
        n_beam_presel_b2b=int((best.beam_presel & best.back_to_back).sum()),
    )
    if c is not None:
        tm = c[c.matched].event_id
        s['n_trig_matched'] = int(tm.nunique())
        s['n_ge2arm_matched'] = int(best.event_id.isin(tm).sum())
        s['frac_matched_ntof_arm_in_pair'] = float(
            best[best.matched == True].ntof_in_pair.mean())  # noqa: E712
    clean = best[(best.topo == 'opposing') & (best.sep_mm < CLEAN_SEP_MM)]
    s['n_opposing_clean'] = int(len(clean))
    s['frac_clean_above_170'] = (float((clean.open_deg > BACK_TO_BACK_DEG).mean())
                                 if len(clean) else None)
    s['open_deg_quantiles_clean'] = {str(q): float(clean.open_deg.quantile(q))
                                     for q in (0.05, 0.16, 0.5, 0.84, 0.95)} if len(clean) else {}
    sc = slope_check(t, P)
    sc.to_csv(tracks_path(run, sub, k_from).with_name(f'slope_check_{run}_{sub}.csv'),
              index=False)
    s['slope_check'] = sc.to_dict('records')
    (tracks_path(run, sub, k_from).with_name(f'summary_{run}_{sub}.json')
     .write_text(json.dumps(s, indent=1, default=str)))
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('step', choices=('fetch', 'build', 'analyse'))
    ap.add_argument('--run', default='run_149')
    ap.add_argument('--subrun', default='cosbounce_cos_0000')
    ap.add_argument('--k-from', default='run_147,run_150',
                    help='comma list of beam runs to borrow k from; one '
                         'track table each')
    ap.add_argument('--tarballs', type=Path, default=None)
    a = ap.parse_args()
    if a.step == 'fetch':
        if a.tarballs is None:
            sys.exit('--tarballs is required')
        return 0 if fetch(a.run, a.subrun, a.tarballs) else 1
    for k in a.k_from.split(','):
        if a.step == 'build':
            build(a.run, a.subrun, k)
        else:
            print(json.dumps(analyse(a.run, a.subrun, k), indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
