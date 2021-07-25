import os
import sys
import math

# Numerical
import numpy as np
from scipy.optimize import minimize

## Auto diff
import jax.numpy as jnp
import jax

# Local Imports #
cur_dir = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(cur_dir))

import geometric_utilities as util



''' ======================================================================================================= '''
#                                        Sparse to Dense                                                  #
''' ======================================================================================================= '''

def dirac_deltas(mesh, idx):
    x = np.zeros((mesh.N(), idx.size))
    jdx = np.arange(idx.size)
    x[idx,jdx] = 1
    return mesh.pointwise_2_vector(x)


def l12_functors(X,Y):
    def l21_norm(X):
        return jnp.sum(jnp.linalg.norm(X, axis=0))
    def functor(C):
        return l21_norm(C @ X - Y)
    f = jax.jit(functor)
    df = jax.jit(jax.grad(functor, argnums=0))
    return f,df


def l12_solver(X,Y):
    shape = (Y.shape[0], X.shape[0])
    f,df = l12_functors(X,Y)
    fun = lambda c: np.asarray(f(c.reshape(shape)), dtype=np.double)
    jac = lambda c: np.asarray(df(c.reshape(shape)), dtype=np.double).flatten()
    x0 = np.linalg.lstsq(X.T, Y.T)[0].T.flatten()
    res = minimize(fun=fun, jac=jac, x0=x0, method='L-BFGS-B')
    return res.x.reshape(shape)


def l12_filtered_correspondence(src, dst, i, j):
    x,y = dirac_deltas(src,i), dirac_deltas(dst,j)
    return l12_solver(x,y)

def orthogonal_correspondence(src, dst, i, j):
    x,y = dirac_deltas(src,i), dirac_deltas(dst,j)
    c = util.orthogonalProcrustes(y.T, x.T).T
    return c



''' ======================================================================================================= '''
    ###                                           End                                                   ###
''' ======================================================================================================= '''
