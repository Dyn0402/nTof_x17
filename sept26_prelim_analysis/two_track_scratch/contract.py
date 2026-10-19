"""The split-ab contract, fixed chain against current production, on the SAME
triggers (both merged 7-tag runs, both with --pairing).

    .venv/bin/python sept26_prelim_analysis/two_track_scratch/contract.py
Writes ~/x17/sept26_prelim/intra_bench/contract_fixed_vs_current.csv.
"""
import os
import pandas as pd

D = os.path.expanduser('~/x17/sept26_prelim/intra_bench/')
K = ['tag', 'event_id']


def stats(E, T):
    cs = E[E.clean_single]
    return dict(
        triggers=len(E), events_split=int((E.n_splits > 0).sum()),
        clean_singles=len(cs), clean_singles_split=int((cs.n_splits > 0).sum()),
        frac_clean_split=float((cs.n_splits > 0).mean()) if len(cs) else float('nan'),
        events_fewer_tracks=int((E.n_tracks_new < E.n_tracks_prod).sum()),
        events_more_tracks=int((E.n_tracks_new > E.n_tracks_prod).sum()),
        prod_tracks=len(T), not_recovered=int((~T.recovered).sum()),
        clean_tracks_lost=int((~T[T.clean_single].recovered).sum()))


rows = []
for arm in ('A', 'C'):
    fd, cd = D + f'split_ab_fixed_{arm}_7tags/', D + f'split_ab_current_{arm}_7tags/'
    if not (os.path.exists(fd + 'events.parquet') and os.path.exists(cd + 'events.parquet')):
        print(arm, 'missing a merged run')
        continue
    F, FT = pd.read_parquet(fd + 'events.parquet'), pd.read_parquet(fd + 'tracks.parquet')
    C, CT = pd.read_parquet(cd + 'events.parquet'), pd.read_parquet(cd + 'tracks.parquet')
    common = F[K].merge(C[K], on=K)
    for chain, E, T in (('current', C, CT), ('fixed', F, FT)):
        E, T = E.merge(common, on=K), T.merge(common.drop_duplicates(), on=K)
        rows.append(dict(arm=arm, tag='all', chain=chain, **stats(E, T)))
        for tag, e in E.groupby('tag'):
            rows.append(dict(arm=arm, tag=tag, chain=chain, **stats(e, T[T.tag == tag])))
R = pd.DataFrame(rows)
R.to_csv(D + 'contract_fixed_vs_current.csv', index=False)
pd.set_option('display.width', 250)
print(R[R.tag == 'all'].to_string(index=False))
