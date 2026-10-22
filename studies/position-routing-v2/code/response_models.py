"""Development-only projection and predictive gain/low-rank models."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import numpy as np


def fit(x,y,keys,cells,rank=0,ridge=1.,projection=None):
    gains={}
    for key in sorted(set(keys)):
        mask=np.array([k==key for k in keys]);gains[key]=float(np.sum(x[mask]*y[mask])/max(np.sum(x[mask]**2),1e-30))
    residual=y-np.array([gains[k] for k in keys])[:,None]*x
    if rank:
        if projection is None:
            _,_,vt=np.linalg.svd(np.concatenate([x,y]),full_matrices=False);p=vt[:rank].T
        else:p=projection[:,:rank]
        maps={};xp=x@p;yp=residual@p
        for cell in sorted(set(cells)):
            ix=np.array([k==cell for k in cells]);gram=xp[ix].T@xp[ix];penalty=max(float(np.trace(gram)/rank),1e-20)*ridge
            maps[cell]=np.linalg.solve(gram+penalty*np.eye(rank),xp[ix].T@yp[ix])
    else:p=np.zeros((x.shape[1],0));maps={}
    return dict(gains=gains,projection=p,maps=maps,rank=rank,ridge=ridge)


def predict(model,x,keys,cells):
    out=np.array([model['gains'][k] for k in keys])[:,None]*x
    if model['rank']:
        p=model['projection'];xp=x@p
        for cell,m in model['maps'].items():
            ix=np.array([c==cell for c in cells]);out[ix]+=(xp[ix]@m)@p.T
    return out


def errors(x,y,prediction):
    norm=np.sum(x*x,axis=1);error=y-prediction
    para=np.sum(error*x,axis=1)**2/np.maximum(norm,1e-30)
    total=np.sum(error*error,axis=1)
    return total,np.maximum(0,total-para)


def folds(names,contexts):
    for n in sorted(set(names)):
        for c in sorted(set(contexts)):
            train=(names!=n)&(contexts!=c);test=(names==n)&(contexts==c)
            yield n,c,train,test


def select(x,y,keys,cells,names,contexts):
    candidates=[(rank,ridge) for rank in [1,2,4,8] for ridge in [.001,.01,.1,1.,10.]]
    totals={str(q):0. for q in candidates};gain_total=0.;trace=[]
    for n,c,train,test in folds(names,contexts):
        tx,ty=x[train],y[train];tk=[k for k,m in zip(keys,train) if m];tc=[k for k,m in zip(cells,train) if m]
        ek=[k for k,m in zip(keys,test) if m];ec=[k for k,m in zip(cells,test) if m]
        _,_,vt=np.linalg.svd(np.concatenate([tx,ty]),full_matrices=False);projection=vt[:8].T
        base=fit(tx,ty,tk,tc);gain_loss=float(np.sum((y[test]-predict(base,x[test],ek,ec))**2));gain_total+=gain_loss
        foldloss={}
        for rank,ridge in candidates:
            model=fit(tx,ty,tk,tc,rank,ridge,projection);loss=float(np.sum((y[test]-predict(model,x[test],ek,ec))**2))
            totals[str((rank,ridge))]+=loss;foldloss[str((rank,ridge))]=loss
        trace.append(dict(name=int(n),context=int(c),train=int(train.sum()),test=int(test.sum()),gain_loss=gain_loss,losses=foldloss))
    chosen=min(candidates,key=lambda q:(totals[str(q)],q[0],-q[1]))
    return chosen,dict(chosen_rank=chosen[0],chosen_ridge=chosen[1],gain_cv_sse=gain_total,transform_cv_sse=totals[str(chosen)],candidates=totals,folds=trace)


def crossed_interval(numerator,denominator,names,contexts,seed=2026100809,draws=20000):
    ns=sorted(set(names));cs=sorted(set(contexts));ng=np.zeros((len(ns),len(cs)));dg=ng.copy()
    for i,n in enumerate(ns):
        for j,c in enumerate(cs):
            mask=(names==n)&(contexts==c);ng[i,j]=np.sum(numerator[mask]);dg[i,j]=np.sum(denominator[mask])
    rng=np.random.default_rng(seed);a=rng.integers(len(ns),size=(draws,len(ns)));b=rng.integers(len(cs),size=(draws,len(cs)))
    boot=ng[a[:,:,None],b[:,None,:]].sum((1,2))/np.maximum(dg[a[:,:,None],b[:,None,:]].sum((1,2)),1e-30)
    return dict(mean=float(np.sum(numerator)/np.sum(denominator)),ci95=np.quantile(boot,[.025,.975]).tolist(),
        names=len(ns),contexts=len(cs),draws=draws,numerator_grid=ng.tolist(),denominator_grid=dg.tolist(),
        name_effects=(ng.sum(1)/dg.sum(1)).tolist(),context_effects=(ng.sum(0)/dg.sum(0)).tolist())
