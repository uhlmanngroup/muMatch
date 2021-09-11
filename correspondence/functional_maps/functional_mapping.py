## Standard Libraries ##
import os
import sys
import math

## Numerical Libraries ##
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix, diags, vstack
from scipy.sparse.linalg import spsolve, lsqr

## Auto diff
import jax.numpy as jnp
import jax


## Local Imports ##
cur_dir = os.path.dirname(__file__)
sys.path.insert(1, cur_dir)


''' ======================================================================================================= '''
    ####                                    Functional Matching                                         ###
''' ======================================================================================================= '''


def spectral_array(mesh, x, k):
    x = mesh.pointwise_2_vector(x, k)
    x = mesh.pointwise_2_vector(x.T, k).T
    return x

def zero_diagonal(x):
    x = x.copy()
    np.fill_diagonal(x,0)
    return x

def norm(x):
    return x/np.linalg.norm(x)

def root(x,n):
    return np.sign(x)*( np.abs(x)**(1.0/float(n)) )


def coupling_array(mesh, k, sigma):
    x = np.exp(-0.5 * (mesh.g / mesh.g.mean() / sigma)**2)
    x = zero_diagonal( spectral_array(mesh, x, k=-1) )
    J0 = x[:k][:,:k]
    J1 = x[:k][:,k:] @ x[k:][:,:k]
    J2 = x[:k][:,k:] @ x[k:][:,k:] @ x[k:][:,:k]
    func = lambda a,n: norm( zero_diagonal(a) ) / math.factorial(n) / 1.67
    couplings = [func(c,n+1) for n,c in enumerate([J0,J1,J2])]
    return couplings


def spectral_extraction(mesh, k):
    ## Laplacian
    l = mesh.eigen[0][:k] + 1
    l = 1.0 / np.sqrt(l)
    l /= np.linalg.norm(l)
    ## Metrics
    J = []
    for sigma in [0.3, 0.45, 0.6]:
        J += coupling_array(mesh, k, sigma=sigma)
    return l,J


def commutator(x, y, operator = np.subtract):
    x,y = np.meshgrid(x,y)
    return operator(x,y)


def spectral_functional(src, dst, k):
    ## Extraction
    ls,J_s = spectral_extraction(src, k)
    ld,J_d = spectral_extraction(dst, k)
    dl = commutator(ls, ld)
    ## Metrics
    def functor(C):
        y = 3 * jnp.linalg.norm(dl * C, ord='fro')
        for a,b in zip(J_s, J_d):
            y += jnp.linalg.norm(C @ a @ C.T - b, ord='fro') #alpha *
        return y
    # Compilation
    f  = jax.jit(functor)
    df = jax.jit(jax.grad(functor, argnums = 0))
    return f,df


def superOperatorPromotion(A, n):
    r,c = A.shape
    B  = lil_matrix((r * n, c * n), dtype = np.float)
    for i in range(r * n):
        for k in range(c):
            j = k*n + i%n
            B[i,j] = A[i//n,k]
    return B.tocsr()


def initialisation(src, dst, k):
    s1,s2 = [m.pointwise_2_vector(m.scalars['signatures'], k) for m in (src,dst)]
    l1,l2 = np.meshgrid(src.eigen[0][:k], dst.eigen[0][:k])
    ## LHS
    A1 = superOperatorPromotion(s1.dot(s1.T), k)
    A2 = diags( np.abs(1/np.sqrt(1+l1) - 1/np.sqrt(1+l2)).flatten() )
    A = vstack([A1,A2])
    ## RHS
    b1 = (s1.dot(s2.T)).flatten()
    b2 = np.zeros((A2.shape[-1]))
    b = np.concatenate([b1,b2], axis=0)
    ## Result
    return lsqr(A,b)[0].reshape(k,k)


def correspondenceMatrixSolver(src, dst, k, optimise = True):
    ## Setting up
    f,df = spectral_functional(src, dst, k)
    fun = lambda x: np.asarray(f(x.reshape(k,k)), dtype = np.float64)
    jac = lambda x: np.asarray(df(x.reshape(k,k)), dtype = np.float64).flatten()
    ## Initial guess
    C = initialisation(src, dst, k)
    if optimise:
        res = minimize(fun = fun, jac = jac, x0 = C.flatten(), method = 'L-BFGS-B', options = {'maxiter': 1000})
        C = res.x.reshape(k,k)
    return C


def soft_correspondence(src, dst, C):
    kd,ks = C.shape
    Q = src.mass @ src.eigen[-1][:,:ks] @ C.T @ dst.eigen[-1][:,:kd].T
    P = Q**2
    P /= np.sum(P, axis = 1, keepdims = True)
    return P


''' ======================================================================================================= '''
    ###                                           End                                                   ###
''' ======================================================================================================= '''


if __name__ == '__main__':
    pass
