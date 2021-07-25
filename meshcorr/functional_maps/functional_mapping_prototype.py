## Standard Libraries ##
import os
import sys

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
    x = mesh.pointwise_2_vector(x)
    x = mesh.pointwise_2_vector(x.T).T
    return x

def metric_signatures(mesh):
    res = []
    for sigma in [0.05, 0.1, 0.2, 0.5]:
        x = np.exp(-0.5 * (mesh.g / sigma)**2) / sigma
        x = spectral_array(mesh, x)
        res.append(x)
    return res

def spectral_extraction(mesh):
    ## Signatures
    s = mesh.scalars['signatures']
    s = mesh.pointwise_2_vector(s)
    ## Laplacian
    l = np.abs(mesh.eigen[0]) + 1
    l = np.diag(1.0 / np.sqrt(l))
    ## Metrics
    g = metric_signatures(mesh)
    return s,l,g


class distortion_functional:
    def __init__(self, src, dst):
        pass
    def __call__(self, i, j, C):
        res = np.linalg.norm(C @ s1[i] - s2[j])
        res += np.linalg.norm(C @ l1[i][:,i] - l2[j][:,j] @ C)
        res += np.linalg.norm(C @ g1[i][:,i] @ C.T - g2[j][:,j] )



class multi_resolution_correspondence:
    def __init__(self, src, dst):
        pass
    def iteration(self,):
        pass
    def __call__(self,):
        pass





'''
def spectral_functional(src, dst, k, alpha, beta):
    ## Extraction
    s1,l1,g1 = spectral_extraction(src, k)
    s2,l2,g2 = spectral_extraction(dst, k)
    ## Metrics
    def functor(C):
        y = jnp.linalg.norm(C @ s1 - s2, ord='fro')
        y += alpha * jnp.linalg.norm(C @ l1 - l2 @ C, ord='fro')
        for a,b in zip(g1,g2):
            y += beta*jnp.linalg.norm(C @ a @ C.T - b, ord='fro')
        return y
    # Compilation
    f  = jax.jit(functor)
    df = jax.jit(jax.grad(functor, argnums = 0))
    return f,df


def correspondenceMatrixSolver(src, dst, k, alpha = 0, beta = 0, maxiter = 1000):
    ## Setting up
    f,df = spectral_functional(src, dst, k, alpha, beta)
    fun = lambda x: np.asarray(f(x.reshape(k,k)), dtype = np.float64)
    jac = lambda x: np.asarray(df(x.reshape(k,k)), dtype = np.float64).flatten()
    ## Initial guess
    C0 = initialisation(src, dst, k).flatten()
    res = minimize(fun = fun, jac = jac, x0 = C0, method = 'L-BFGS-B', options = {'maxiter': maxiter})
    C = res.x.reshape(k,k)
    return C
'''



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
    import matplotlib.pyplot as plt

    A = np.random.rand(17,21)
    x = np.random.rand(21,17)

    L = leftOperatorPromotion(A, n=x.shape[-1])
    R = rightOperatorPromotion(A, n=x.shape[0])

    l = (L @ x.flatten()).reshape(A.shape[0], x.shape[-1])
    r = (R @ x.flatten()).reshape(x.shape[0], A.shape[-1])

    print(np.isclose(l, A @ x).all())
    print(np.isclose(r, x @ A).all())
