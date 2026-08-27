from __future__ import annotations
import hashlib
import numpy as np

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def linear_cka(x,y,eps=1e-12):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=y.shape[0]: raise ValueError('CKA inputs require same observations')
    x=x-x.mean(axis=0,keepdims=True); y=y-y.mean(axis=0,keepdims=True)
    xy=x.T@y; xx=x.T@x; yy=y.T@y
    num=float(np.sum(xy*xy)); den=float(np.sqrt(np.sum(xx*xx)*np.sum(yy*yy)))
    return 0.0 if den<=eps else float(np.clip(num/den,0.0,1.0))

def cosine_vector(a,b,eps=1e-12):
    a=np.asarray(a,float).reshape(-1); b=np.asarray(b,float).reshape(-1)
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return 0.0 if den<=eps else float(np.clip(np.dot(a,b)/den,-1.0,1.0))

def rankdata_average(x):
    x=np.asarray(x,float).reshape(-1); order=np.argsort(x,kind='mergesort'); ranks=np.empty(len(x),float)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        r=0.5*((i+1)+j); ranks[order[i:j]]=r; i=j
    return ranks

def pearson(a,b,eps=1e-12):
    a=np.asarray(a,float).reshape(-1); b=np.asarray(b,float).reshape(-1)
    a=a-a.mean(); b=b-b.mean(); den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return 0.0 if den<=eps else float(np.clip(np.dot(a,b)/den,-1.0,1.0))

def spearman(a,b): return pearson(rankdata_average(a),rankdata_average(b))

def relative_abs_disagreement(a,b,eps=1e-12):
    a=np.asarray(a,float).reshape(-1); b=np.asarray(b,float).reshape(-1)
    scale=float(np.mean(0.5*(np.abs(a)+np.abs(b))))
    return float(np.mean(np.abs(a-b))/max(scale,eps))

def latent_effect_strength(pred_full,pred_mean,eps=1e-12):
    p=np.asarray(pred_full,float); q=np.asarray(pred_mean,float)
    return float(np.mean(np.abs(p-q))/max(float(np.mean(np.abs(p))),eps))

def inverse_permutation(perm):
    p=np.asarray(perm,dtype=int); inv=np.empty_like(p)
    for i,j in enumerate(p): inv[j]=i
    return tuple(int(x) for x in inv)

def empirical_p_high(obs,null):
    z=np.asarray(null,float); return float((1+np.sum(z>=obs))/(len(z)+1))

def empirical_p_low(obs,null):
    z=np.asarray(null,float); return float((1+np.sum(z<=obs))/(len(z)+1))

def block_shuffle_indices(n,repeats,rng):
    base=np.arange(n); return [rng.permutation(base) for _ in range(repeats)]

def gate_kernel_signatures(wtheta,units=50):
    w=np.asarray(wtheta,float)
    if w.shape!=(5,4*units): raise ValueError(f'Unexpected latent LSTM kernel shape {w.shape}')
    return np.stack([np.linalg.norm(w[:,g*units:(g+1)*units],axis=1) for g in range(4)],axis=1)

def permuted_product_max_error(theta,wtheta,perm):
    theta=np.asarray(theta,float); w=np.asarray(wtheta,float); p=np.asarray(perm,dtype=int)
    lhs=theta@w; rhs=theta[:,p]@w[p,:]
    return float(np.max(np.abs(lhs-rhs)))
