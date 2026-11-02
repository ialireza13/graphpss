"""Break down PSS-net test errors: by boundary distance, segment length and protein size.

    python scripts/analyze_errors.py    # writes runs/test_analysis.json
"""
import sys,json,time
sys.path.insert(0,'src')
import numpy as np,torch
from pssnet.data import Collate,GraphStore,load_or_make_split
from pssnet.models import build_model
st=torch.load('runs/pssnet_gatv2_s0/best.pt',map_location='cpu',weights_only=False); cfg=st['config']
s=GraphStore.load(cfg['data']['store']); sp=load_or_make_split(cfg['data']['split'],len(s))
m=build_model(cfg).eval(); m.load_state_dict(st['state_dict']); col=Collate(s,1)
ids=list(sp['test']); t0=time.time()
res=[]
with torch.no_grad():
    for k in range(0,len(ids),64):
        b=col([int(i) for i in ids[k:k+64]]); p=m(b).argmax(1).numpy(); y=b.y.numpy(); g=b.batch.numpy()
        for j in range(g.max()+1):
            res.append((y[g==j],p[g==j]))
        if k%6400==0: print(k,time.time()-t0,flush=True)
Y=np.concatenate([r[0] for r in res]);P=np.concatenate([r[1] for r in res])
out={'acc':float((Y==P).mean()),'n':len(Y),'proteins':len(res),'per_protein':float(np.mean([(a==b).mean() for a,b in res]))}
D=[];SL=[];pa=[]
for y,p in res:
    n=len(y); ch=np.flatnonzero(np.diff(y))  # boundary between i and i+1
    if len(ch)==0: d=np.full(n,99)
    else:
        d=np.minimum.reduce([np.abs(np.arange(n)-c) for c in ch]+[np.abs(np.arange(n)-c-1) for c in ch]) if len(ch)<400 else None
    D.append(d)
    st_=np.r_[0,ch+1]; en=np.r_[ch+1,n]
    for a,b_ in zip(st_,en): SL.append((int(y[a]),int(b_-a),int((p[a:b_]==y[a]).sum()),int(((p[a:b_]==y[a]).mean()>=.5))))
D=np.concatenate(D)
out['by_boundary_distance']={}
for lab,mask in (('0',D==0),('1',D==1),('2',D==2),('3+',D>=3)):
    out['by_boundary_distance'][lab]={'frac_res':float(mask.mean()),'acc':float((Y[mask]==P[mask]).mean()),'share_of_errors':float((Y[mask]!=P[mask]).sum()/(Y!=P).sum())}
SL=np.array(SL)
out['segments']={}
for c,name in ((1,'helix'),(2,'sheet'),(0,'coil')):
    for lo,hi in ((1,3),(4,6),(7,10),(11,10**6)):
        x=SL[(SL[:,0]==c)&(SL[:,1]>=lo)&(SL[:,1]<=hi)]
        if len(x): out['segments'][f'{name}_{lo}-{hi}']={'n':len(x),'res_recall':float(x[:,2].sum()/x[:,1].sum()),'seg_found':float(x[:,3].mean())}
L=np.array([len(r[0]) for r in res]);A=np.array([(a==b).mean() for a,b in res])
out['by_len']={}
for lo,hi in ((1,49),(50,99),(100,199),(200,399),(400,799),(800,2000)):
    k=(L>=lo)&(L<=hi); out['by_len'][f'{lo}-{hi}']={'n':int(k.sum()),'pp_acc':float(A[k].mean()),'p10':float(np.percentile(A[k],10))}
out['pp_percentiles']={str(q):float(np.percentile(A,q)) for q in (5,25,50,75,95)}
json.dump(out,open('runs/test_analysis.json','w'),indent=1); print(json.dumps(out,indent=1))
