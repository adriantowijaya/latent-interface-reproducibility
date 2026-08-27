from __future__ import annotations

import hashlib
import itertools
import math
from typing import Iterable, Sequence

import numpy as np


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def context_from_transition_intensity(x: np.ndarray):
    x = np.asarray(x, dtype=float).reshape(-1)
    q1, q2 = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    c = np.where(x <= q1, 0, np.where(x <= q2, 1, 2)).astype(int)
    return c, float(q1), float(q2)


def evenly_spaced_indices(indices: Sequence[int], n: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    if len(idx) < n:
        return idx.copy()
    if n == 1:
        return idx[[0]]
    pos = np.linspace(0, len(idx) - 1, n)
    take = np.rint(pos).astype(int)
    # np.rint can in principle duplicate indices; enforce deterministic uniqueness.
    out=[]; seen=set()
    for t in take:
        v=int(idx[t])
        if v not in seen:
            out.append(v); seen.add(v)
    if len(out) < n:
        for v in idx:
            v=int(v)
            if v not in seen:
                out.append(v); seen.add(v)
                if len(out)==n: break
    return np.asarray(out[:n], dtype=int)


def assignment_cost_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Aligned panels must have identical [n,K] shape")
    k=a.shape[1]
    c=np.empty((k,k),float)
    for i in range(k):
        for j in range(k):
            c[i,j]=float(np.mean((a[:,i]-b[:,j])**2))
    return c


def optimal_permutation(cost: np.ndarray):
    c=np.asarray(cost,float)
    if c.ndim!=2 or c.shape[0]!=c.shape[1]:
        raise ValueError("Cost must be square")
    k=c.shape[0]
    best=None; best_cost=float("inf")
    for p in itertools.permutations(range(k)):
        v=float(sum(c[i,p[i]] for i in range(k)))
        if v < best_cost - 1e-15 or (abs(v-best_cost)<=1e-15 and (best is None or p < best)):
            best_cost=v; best=p
    return tuple(best), best_cost


def apply_permutation(theta: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    x=np.asarray(theta,float)
    return x[:, np.asarray(perm,dtype=int)]


def mean_total_variation(a: np.ndarray,b: np.ndarray)->float:
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.mean(0.5*np.sum(np.abs(a-b),axis=1)))


def normalized_js(a: np.ndarray,b: np.ndarray,eps:float=1e-12)->float:
    a=np.asarray(a,float); b=np.asarray(b,float)
    aa=np.clip(a,eps,1.0); bb=np.clip(b,eps,1.0)
    aa=aa/aa.sum(axis=1,keepdims=True); bb=bb/bb.sum(axis=1,keepdims=True)
    m=0.5*(aa+bb)
    kl1=np.sum(aa*np.log(aa/m),axis=1); kl2=np.sum(bb*np.log(bb/m),axis=1)
    return float(np.mean(0.5*(kl1+kl2))/math.log(2.0))


def mean_cosine(a: np.ndarray,b: np.ndarray,eps:float=1e-12)->float:
    a=np.asarray(a,float); b=np.asarray(b,float)
    num=np.sum(a*b,axis=1); den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return float(np.mean(num/np.maximum(den,eps)))


def component_correlations(a: np.ndarray,b: np.ndarray):
    a=np.asarray(a,float); b=np.asarray(b,float)
    vals=[]
    for k in range(a.shape[1]):
        x=a[:,k]; y=b[:,k]
        sx=float(np.std(x)); sy=float(np.std(y))
        if sx < 1e-12 and sy < 1e-12:
            vals.append(1.0 if np.allclose(x,y,atol=1e-10,rtol=0) else 0.0)
        elif sx < 1e-12 or sy < 1e-12:
            vals.append(0.0)
        else:
            vals.append(float(np.corrcoef(x,y)[0,1]))
    return vals


def adjusted_rand_index(labels_a: Iterable[int], labels_b: Iterable[int]) -> float:
    a=np.asarray(list(labels_a),dtype=int); b=np.asarray(list(labels_b),dtype=int)
    if a.shape!=b.shape: raise ValueError("label shapes differ")
    n=len(a)
    if n<2: return 1.0
    ua,ia=np.unique(a,return_inverse=True); ub,ib=np.unique(b,return_inverse=True)
    cont=np.zeros((len(ua),len(ub)),dtype=np.int64)
    for i,j in zip(ia,ib): cont[i,j]+=1
    comb2=lambda x: x*(x-1)//2
    sum_nij=float(sum(comb2(int(x)) for x in cont.ravel()))
    ai=cont.sum(axis=1); bj=cont.sum(axis=0)
    sum_ai=float(sum(comb2(int(x)) for x in ai)); sum_bj=float(sum(comb2(int(x)) for x in bj))
    total=float(comb2(n))
    expected=sum_ai*sum_bj/total if total>0 else 0.0
    max_index=0.5*(sum_ai+sum_bj)
    denom=max_index-expected
    if abs(denom)<1e-15:
        return 1.0 if np.array_equal(a,b) else 0.0
    return float((sum_nij-expected)/denom)


def hard_agreement(a: np.ndarray,b: np.ndarray)->float:
    return float(np.mean(np.argmax(a,axis=1)==np.argmax(b,axis=1)))


def pair_metrics(a: np.ndarray,b: np.ndarray)->dict:
    a=np.asarray(a,float); b=np.asarray(b,float)
    cor=component_correlations(a,b)
    return {
        "tv":mean_total_variation(a,b),
        "js":normalized_js(a,b),
        "cosine":mean_cosine(a,b),
        "ari":adjusted_rand_index(np.argmax(a,axis=1),np.argmax(b,axis=1)),
        "hard_agreement":hard_agreement(a,b),
        "component_correlation_mean":float(np.mean(cor)),
        "component_correlation_min":float(np.min(cor)),
    }


def taxonomy(ari: float, aligned_tv: float, alignment_gain_tv: float, ari_min=.75, tv_max=.15, gain_min=.10)->str:
    hard=bool(ari>=ari_min); soft=bool(aligned_tv<=tv_max); gain=bool(alignment_gain_tv>=gain_min)
    if hard and soft:
        return "PERMUTATION-DOMINATED" if gain else "STABLE"
    if hard != soft:
        return "SOFT-REPARAMETERISED"
    return "STRUCTURALLY UNSTABLE"
