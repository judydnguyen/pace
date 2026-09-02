#!/usr/bin/env python3
"""Validate a gpqa raw run for coco response-crossing: prints per-subset
combined-matcher accuracy at shift 0/+1/-1; exit 1 if any subset looks
shifted or scrambled (shift0 < 0.5 for an opus-class model)."""
import json, glob, re, sys
pat = re.compile(r"\(([A-D])\)")
def norm(s): return re.sub(r"[^a-z0-9^.\-]+","", s.lower())
def extract(resp, choices):
    m=pat.findall(resp)
    if m: return f"({m[-1]})"
    s=re.sub(r"[^A-Za-z0-9]","",resp)
    if s.upper() in "ABCD" and len(s)==1: return f"({s.upper()})"
    nr=norm(resp)
    hits=[i for i,c in enumerate(choices) if norm(c)==nr] or \
         [i for i,c in enumerate(choices) if norm(c) and norm(c) in nr]
    return f"({'ABCD'[hits[0]]})" if len(hits)==1 else None
model=sys.argv[1]; thresh=float(sys.argv[2]) if len(sys.argv)>2 else 0.5
bad=False
for task in ("diamond","extended","main"):
    fs=sorted(glob.glob(f"results/raw_results/gpqa/{model}/samples_gpqa_{task}_cot_zeroshot_*.jsonl"))
    if not fs: print(f"{task}: NO FILE"); bad=True; continue
    docs={}
    for l in open(fs[-1]):
        r=json.loads(l); docs.setdefault(r["doc_id"], r)
    ds=sorted(docs); n=len(ds)
    def acc(shift):
        ok=0
        for i,d in enumerate(ds):
            j=i+shift
            if j<0 or j>=n: continue
            r=docs[d]; r2=docs[ds[j]]
            resp=(r["resps"][0][0] if r.get("resps") else "") or ""
            if extract(resp,[r2["doc"][f"choice{k}"] for k in range(1,5)])==r2["target"]: ok+=1
        return ok/n
    a0,a1,am=acc(0),acc(1),acc(-1)
    verdict="OK" if (a0>=thresh and a0>max(a1,am)) else "CORRUPT"
    if verdict=="CORRUPT": bad=True
    print(f"{task}: shift0={a0:.3f} +1={a1:.3f} -1={am:.3f} -> {verdict}")
sys.exit(1 if bad else 0)
