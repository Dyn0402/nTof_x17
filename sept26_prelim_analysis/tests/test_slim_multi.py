"""Sub-runs with several n_TOF slims (tracking audit 2026-11-22, finding 12).

Synthetic, data-free checks of the shared reader (paths.slim_files / concat_slim / read_slim_tree), the readers built
on it (trigger_time, candidate_filter, freeze_sample), the stage-1 table patch and ntof_stage3's --subruns parser.
The ROOT files are tiny ones written into tmp_path.

    .venv/bin/python -m pytest sept26_prelim_analysis/tests/test_slim_multi.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                                  # noqa: E402
from sept26_prelim_analysis import candidate_filter as CF                 # noqa: E402
from sept26_prelim_analysis import trigger_time as TT                     # noqa: E402
from sept26_prelim_analysis import freeze_sample as FS                    # noqa: E402
from sept26_prelim_analysis import stage1_slim_patch as SP                # noqa: E402

RUN, SUB = 'run_116', 'stat090_0027'


# --------------------------------------------------------------------------- tiny slims
def _write_slim(d: Path, ntof_run: int, ev_ids, bunch0: int, hits):
    """One slim with an events tree (ev_ids) and a hits tree [(eventId, det, dt_ns)]."""
    import uproot
    d.mkdir(parents=True, exist_ok=True)
    n = len(ev_ids)
    ev = dict(eventId=np.asarray(ev_ids, np.uint64), bunch=(bunch0 + np.arange(n) // 2).astype(np.int32),
              t_dream_ns=np.linspace(1e6, 7e7, n), is_flash=np.zeros(n, np.uint8),
              t_pred_ns=np.linspace(1e6, 7e7, n) + ntof_run, matched=np.ones(n, np.uint8),
              arm=np.zeros(n, np.int8))
    h = np.array(hits, dtype=float).reshape(-1, 3)
    hi = dict(eventId=h[:, 0].astype(np.uint64), det=h[:, 1].astype(np.uint8), dt_ns=h[:, 2].astype(np.float32),
              detn=np.ones(len(h), np.int32), tof=np.ones(len(h)), amp=np.ones(len(h), np.float32),
              area_0=np.ones(len(h), np.float32), satuflag=np.zeros(len(h), np.int32),
              pileup1=np.zeros(len(h), np.int32), is_control=np.zeros(len(h), np.uint8))
    p = d / f'ntof_hits_{RUN}_{SUB}_{ntof_run}.root'
    with uproot.recreate(p) as f:                # TTrees, as slim.py writes (a bare dict is an RNTuple in uproot 5.7)
        for name, arrs in (('events', ev), ('hits', hi)):
            f.mktree(name, {k: v.dtype for k, v in arrs.items()})
            f[name].extend(arrs)
    return p


@pytest.fixture
def two_slims(tmp_path):
    """run_116/stat090_0027 as on EOS: 224639 = events 1-6, 224640 = events 10-15 (bunch restarts at 1)."""
    d = tmp_path / 'runs' / RUN / SUB / 'ntof_hits'
    # event 2: WAL A + PSS A (coinc A); event 11: coinc A and coinc C (2 arms -> IMPLIED material)
    _write_slim(d, 224640, range(10, 16), 1, [(11, CF.DET_CODE['WALA'], 0), (11, CF.DET_CODE['PSSA'], 3),
                                             (11, CF.DET_CODE['WALC'], 1), (11, CF.DET_CODE['PSSC'], -2)])
    _write_slim(d, 224639, range(1, 7), 1, [(2, CF.DET_CODE['WALA'], 0), (2, CF.DET_CODE['PSSA'], 5)])
    (d / 'calibration.json').write_text('{}')            # sidecars are not slims
    return tmp_path / 'runs'


# --------------------------------------------------------------------------- paths
def test_slim_ntof_run():
    assert paths.slim_ntof_run('/x/ntof_hits_run_116_stat090_0027_224640.root') == 224640
    assert paths.slim_ntof_run('ntof_hits_run_79_cosbounce_cos_0000_224574.root') == 224574
    with pytest.raises(ValueError):
        paths.slim_ntof_run('ntof_wf_run_116_stat090_0027_224640.root')


def test_slim_files_all_sorted(two_slims):
    f = paths.slim_files(RUN, SUB, two_slims)
    assert [paths.slim_ntof_run(p) for p in f] == [224639, 224640]
    assert paths.slim_files(RUN, 'stat090_9999', two_slims) == []


def test_concat_disjoint_adds_ntof_run():
    a = pd.DataFrame(dict(eventId=[1, 2, 2], bunch=[1, 1, 2]))
    b = pd.DataFrame(dict(eventId=[5, 6], bunch=[1, 2]))
    d = paths.concat_slim([(224639, a), (224640, b)])
    assert d.ntof_run.dtype == np.int32
    assert list(d.ntof_run) == [224639] * 3 + [224640] * 2
    # bunch 1 is in both runs: only (ntof_run, bunch) is a key
    assert d.groupby(['ntof_run', 'bunch']).ngroups == 4


def test_concat_overlap_raises():
    a = pd.DataFrame(dict(eventId=[1, 2]))
    b = pd.DataFrame(dict(eventId=[2, 3]))
    with pytest.raises(ValueError, match='two slim files'):
        paths.concat_slim([(1, a), (2, b)], where='x')


def test_concat_unique_within():
    a = pd.DataFrame(dict(eventId=[1, 1]))
    paths.concat_slim([(1, a)])                                   # hits: repeats are normal
    with pytest.raises(ValueError, match='repeats'):
        paths.concat_slim([(1, a)], unique_within=True)            # events: one row per trigger
    with pytest.raises(ValueError):
        paths.concat_slim([])


def test_read_slim_tree(two_slims):
    d = paths.read_slim_tree(RUN, SUB, 'events', ['eventId', 'bunch'], runs_root=two_slims, unique_within=True)
    assert len(d) == 12 and sorted(d.ntof_run.unique()) == [224639, 224640]
    a = paths.read_slim_tree(RUN, SUB, 'hits', ['eventId', 'det'], runs_root=two_slims, library='np')
    assert set(a) == {'eventId', 'det', 'ntof_run'} and len(a['eventId']) == 6
    with pytest.raises(FileNotFoundError):
        paths.read_slim_tree(RUN, 'stat090_9999', 'hits', ['eventId'], runs_root=two_slims)


# --------------------------------------------------------------------------- readers
def test_trigger_time_all_files(two_slims, tmp_path):
    r = TT.write_subrun(RUN, SUB, runs_root=two_slims, out_dir=tmp_path / 'tt')
    assert r['n'] == 12 and r['n_slim'] == 2 and r['ntof_runs'] == '224639;224640'
    d = pd.read_parquet(TT.product(RUN, SUB, tmp_path / 'tt'))
    assert d.eventId.is_unique and set(d.eventId) == set(range(1, 7)) | set(range(10, 16))
    tracks = pd.DataFrame(dict(event_id=[2, 11, 8]))               # 8 is in neither slim
    t, prov = TT.attach(tracks, RUN, SUB, tmp_path / 'tt')
    assert np.isfinite(t.t_since_flash_ns.iloc[:2]).all() and np.isnan(t.t_since_flash_ns.iloc[2])
    assert prov['n_no_trigger_in_slim'] == 1


def test_trigtime_env_dir(monkeypatch, tmp_path):
    monkeypatch.setenv('X17_TRIGTIME_DIR', str(tmp_path / 'alt'))
    assert TT.product(RUN, SUB) == tmp_path / 'alt' / f'trigtime_{RUN}_{SUB}.parquet'
    assert TT.product(RUN, SUB, tmp_path / 'x').parent == tmp_path / 'x'   # an explicit dir still wins


def test_arm_flags_subrun(two_slims):
    fl = CF.arm_flags_subrun(CF.slim_files_in(two_slims / RUN / SUB / 'ntof_hits'))
    assert fl.eventId.is_unique and len(fl) == 12
    f = fl.set_index('eventId')
    assert bool(f.loc[2, 'coinc_A']) and f.loc[2, 'n_coinc_arms'] == 1 and f.loc[2, 'ntof_run'] == 224639
    assert f.loc[11, 'n_coinc_arms'] == 2 and f.loc[11, 'ntof_run'] == 224640
    assert set(CF.NTOF_COLS) <= set(fl.columns)
    assert CF.arm_flags_subrun([]) is None


def test_freeze_sample_load_slim(tmp_path):
    inv = tmp_path / 'slim_inventory_x.tsv'
    inv.write_text('ntof_hits_run_116_stat090_0027_224640.root\t91\n'
                   'ntof_hits_run_116_stat090_0027_224639.root\t6\n'
                   'ntof_hits_run_116_stat090_0026_224639.root\t50\n')
    s = FS.load_slim(inv)
    assert s[(116, 'stat090_0027')] == ('224639;224640', 97, 2)
    assert s[(116, 'stat090_0026')] == (224639, 50, 1)


# --------------------------------------------------------------------------- stage-1 patch
def _cands():
    """Five triggers as stage 1 wrote them from the FIRST slim only (events 10+ have no n_TOF record)."""
    d = pd.DataFrame(dict(
        cls=['SINGLE', 'IMPLIED', 'SINGLE', 'INTER', 'NONE'],
        n_arms_strict=[1, 1, 1, 2, 0], n_arms_loose=[1, 1, 2, 2, 0], n_busy_arms=[0, 0, 0, 0, 0],
        n_sep_best=[0, 1, 0, 0, 0], eventId=np.array([1, 3, 11, 12, 13], np.int64),
        A_trk_x=[1, 1, 1, 1, 0], A_trk_y=[1, 1, 1, 1, 0], A_strips=[5, 5, 5, 5, 0], n_clean=[5, 5, 5, 9, 0]))
    for a in 'BCD':
        d[f'{a}_trk_x'] = d[f'{a}_trk_y'] = d[f'{a}_strips'] = 0
    first = dict(bunch=[1, 2, np.nan, np.nan, np.nan], is_flash=[0, 0, np.nan, np.nan, np.nan],
                 matched=[1, 1, np.nan, np.nan, np.nan], n_coinc_arms=[0, 2, np.nan, np.nan, np.nan],
                 n_wall_arms=[0, 2, np.nan, np.nan, np.nan])
    for k, v in first.items():
        d[k] = v
    for k in ('coinc', 'wall', 'plastic', 'liq'):
        for a in 'ABCD':
            d[f'{k}_{a}'] = [False, k in ('coinc', 'wall', 'plastic') and a in 'AC', None, None, None]
    d['run'], d['subrun'], d['tag'] = RUN, SUB, ['t0', 't0', 't1', 't1', 't1']
    return d


def _flags(n_coinc):
    f = pd.DataFrame(dict(eventId=np.array([1, 3, 11, 12, 13], np.uint64), ntof_run=np.int32(224639),
                          bunch=[1, 2, 1, 1, 2], is_flash=0, matched=1, n_coinc_arms=n_coinc, n_wall_arms=n_coinc))
    for k in ('coinc', 'wall', 'plastic', 'liq'):
        for a in 'ABCD':
            f[f'{k}_{a}'] = False
    f.loc[2:, 'ntof_run'] = 224640
    return f


def test_patch_reclassifies_single_implied_only():
    old = _cands()
    new = SP.patch(old, _flags([0, 2, 2, 2, 2]), strict=True, where='t')
    # event 11 (second slim, 2 coincident arms) SINGLE -> IMPLIED; INTER / NONE never change
    assert list(new.cls) == ['SINGLE', 'IMPLIED', 'IMPLIED', 'INTER', 'NONE']
    assert new.bunch.notna().all() and list(new.ntof_run) == [224639, 224639, 224640, 224640, 224640]
    cols = list(new.columns)
    assert cols.index('ntof_run') + 1 == cols.index('bunch')
    assert [c for c in cols if c != 'ntof_run'] == list(old.columns)
    assert (new.eventId.to_numpy() == old.eventId.to_numpy()).all()
    # an IMPLIED whose coincidence the new flags drop goes back to SINGLE
    assert SP.patch(old, _flags([0, 1, 0, 0, 0]), strict=True).cls.tolist()[:3] == ['SINGLE'] * 3
    # the derived tables are candidate_filter's own
    assert CF.census(new).set_index('cls').loc['IMPLIED', 'n'] == 2
    assert set(CF.census_per_tag(new).tag) == {'t0', 't1'}


def test_patch_refuses_inconsistent_table():
    bad = _cands()
    bad.loc[0, 'cls'] = 'IMPLIED'                    # 0 coincident arms stored, yet IMPLIED
    with pytest.raises(RuntimeError, match='reproduce'):
        SP.patch(bad, _flags([0, 2, 2, 2, 2]))
    bad = _cands()
    bad.loc[2, 'n_sep_best'] = 2                     # would be INTRA, not SINGLE
    with pytest.raises(RuntimeError, match='strip-side'):
        SP.patch(bad, _flags([0, 2, 2, 2, 2]))
    with pytest.raises(RuntimeError):                # strict decides which arm count is "lit"
        SP.patch(_cands(), _flags([0, 2, 2, 2, 2]), strict=False)


def test_patch_subrun_and_cli(two_slims, tmp_path):
    old = _cands()
    old['eventId'] = np.array([1, 2, 11, 12, 13], np.int64)
    old.loc[1, ['cls', 'n_coinc_arms']] = ['SINGLE', 1]
    ind, outd = tmp_path / 's1', tmp_path / 's1new'
    ind.mkdir()
    old.to_parquet(ind / f'candidates_{RUN}_{SUB}.parquet', index=False)
    lst = tmp_path / 'aff.txt'
    lst.write_text(f'{RUN} {SUB}\n')
    with pytest.raises(SystemExit):                                  # never in place
        SP.main(['--list', str(lst), '--in-dir', str(ind), '--out-dir', str(ind)])
    assert SP.main(['--list', str(lst), '--in-dir', str(ind), '--out-dir', str(outd),
                    '--runs-root', str(two_slims)]) == 0
    new = pd.read_parquet(outd / f'candidates_{RUN}_{SUB}.parquet')
    assert new.set_index('eventId').loc[11, 'cls'] == 'IMPLIED'
    for k in ('census', 'census_per_tag', 'arm_rates', 'slim_patch'):
        assert any(outd.glob(f'{k}_{RUN}_{SUB}.*'))
    assert pd.read_parquet(ind / f'candidates_{RUN}_{SUB}.parquet').equals(old)    # original untouched


def test_stage3_parse_subruns(tmp_path):
    sys.path.insert(0, str(REPO / 'rerun' / 'ntof' / 'stage3'))
    sys.path.insert(0, str(REPO / 'rerun' / 'ntof'))
    import ntof_stage3 as S3
    assert S3.parse_subruns('run_116/stat090_0027,run_79/stat090_0008') == {
        ('run_116', 'stat090_0027'), ('run_79', 'stat090_0008')}
    f = tmp_path / 'l.txt'
    f.write_text('run_116 stat090_0027\n# c\nrun_79/stat090_0008\n')
    assert len(S3.parse_subruns(str(f))) == 2
    with pytest.raises(SystemExit):
        S3.parse_subruns('116/stat090_0027')
