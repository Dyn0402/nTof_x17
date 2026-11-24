from r1_02_select import *
for K, lam in itertools.product([10, 12], [0.03, 0.1, 0.3]):
    tot = {n: [0, 0] for n in NS}
    for src, dst in (('run1', 'run2'), ('run2', 'run1')):
        F = fieldT(src, K, lam)
        for p in 'xy':
            for n in NS:
                M, A, Ae, vg = MEAS[(dst, p, n)]
                r_, t_, nn = rss(predmap(F, dst, p, n), A, Ae); tot[n][0] += r_; tot[n][1] += t_
    print(K, lam, {n: round(1 - tot[n][0] / tot[n][1], 3) for n in NS}, flush=True)
