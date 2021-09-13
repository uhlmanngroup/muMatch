import os
import sys
import math

# Numerical
import numpy as np
from scipy.optimize import minimize

## Auto diff
import jax.numpy as jnp
import jax


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
    x,y = src.dirac_deltas(i), dst.dirac_deltas(j)
    return l12_solver(x,y)
