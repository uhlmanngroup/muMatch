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
    x = mesh.pointwise_2_vector(x, k)
    x = mesh.pointwise_2_vector(x.T, k).T
    return x


def metric_signatures(mesh, k):
    res = []
    for sigma in [0.05, 0.1, 0.2, 0.5]:
        x = np.exp(- 0.5 * (mesh.g / sigma)**2)
        x = spectral_array(mesh, x, k)
        x /= np.linalg.norm(x)
        res.append(jnp.array(x))
    return res


def spectral_extraction(mesh, k):
    ## Signatures
    s = mesh.scalars['signatures']
    s = mesh.pointwise_2_vector(s,k)
    ## Laplacian
    l = mesh.eigen[0][:k] + 1
    l = np.diag(1.0 / np.sqrt(l))
    ## JAX conversion
    s,l = (jnp.array(x) for x in (s,l))
    ## Metrics
    g = metric_signatures(mesh, k)
    return s,l,g


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


def superOperatorPromotion(A, n):
    r,c = A.shape
    B  = lil_matrix((r * n, c * n), dtype = np.float)
    for i in range(r * n):
        for k in range(c):
            j = k*n + i%n
            B[i,j] = A[i//n,k]
    return B.tocsr()


def initialisation(src, dst, k):
    s1 = src.pointwise_2_vector(src.scalars['signatures'], k)
    s2 = dst.pointwise_2_vector(dst.scalars['signatures'], k)
    v1,v2 = np.meshgrid(src.eigen[0][:k], dst.eigen[0][:k])
    ## LHS
    A1 = superOperatorPromotion(s1.dot(s1.T), k)
    A2 = diags( np.abs(1/np.sqrt(1+v1) - 1/np.sqrt(1+v2)).flatten() )
    A = vstack([A1,A2])
    ## RHS
    b1 = (s1.dot(s2.T)).flatten()
    b2 = np.zeros((A2.shape[-1]))
    b = np.concatenate([b1,b2], axis=0)
    ## Result
    return lsqr(A,b)[0].reshape(k,k)


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
