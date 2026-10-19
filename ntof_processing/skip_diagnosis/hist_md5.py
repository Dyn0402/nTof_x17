import hashlib, sys, uproot
def text(p):
    o = uproot.open(p)['history']
    for attr in ('fString', 'fTitle', 'fName'):
        try:
            v = o.member(attr)
            if isinstance(v, (str, bytes)) and len(v) > 200:
                return v.decode() if isinstance(v, bytes) else v
        except Exception:
            pass
    return str(o)
for p in sys.argv[1:]:
    try:
        s = text(p)
        tag = 'v12_liqpileup' if 'v12_liqpileup' in s else ('??? ' + s[:40])
        print(f'{hashlib.md5(s.encode()).hexdigest()}  len={len(s):6d}  {tag:>14}  {p.split("/")[-1]}')
    except Exception as e:
        print(f'FAIL {type(e).__name__}  {p.split("/")[-1]}')
