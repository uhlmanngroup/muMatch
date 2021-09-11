## Python imports ##
import sys
import os
import json

## Numerical ##
import numpy as np
import math
from scipy.optimize import linear_sum_assignment

## Mesh and Graphics
import igl
import vedo as vp
import matplotlib.pyplot as plt

## Local Imports ##
sys.path.insert(1, "../tools")

import geometric_utilities as util
from mesh_class import Mesh, mesh_loader

from itertools import tee

from joblib import Memory, Parallel, delayed
# To have a cache for computations which are taking time to complete
memory = Memory(location=".joblib_cache", verbose=0)

''' ======================================================================================================= '''
    #### ---------------------------------------------------------------------------------------------- ###
''' ======================================================================================================= '''


def pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def rbf(x):
    i = (x < 2)
    y = np.empty(x.shape, dtype = x.dtype)
    y[i] = np.log(x[i]**(x[i]**2))
    y[~i] = (x[~i]**2)*np.log(x[~i])
    return y


def thin_plate_spline(x0, y0, x):
    r0 = util.differenceMatrix(x0)
    A = rbf(r0)
    w = np.linalg.lstsq(a=A, b=(y0-x0))[0]
    r = util.differenceMatrix(x0, x)
    return rbf(r) @ w


def deformation_transform(src, dst, *args):
    cs,cd = [m.centroid() for m in (src,dst)]
    R = util.orthogonalProcrustes(dst.v - cd, src.v - cs)
    for x in args:
        x.shift(-cs).rotate(R).shift(cd)
    return


def deformation_transform_with_tps(src, dst, x):
    src,dst = (m.copy() for m in (src,dst))
    deformation_transform(src, dst, src, x)
    j = util.metric_sampling(src.g, 50)
    dx = thin_plate_spline(src.v[j], dst.v[j], x.v)
    x.shift(dx)
    return


def nearest_neighbours(src, dst, quadratic = False, return_points = True):
    d = util.differenceMatrix(dst, src)
    idx = np.isnan(d)
    d /= d[~idx].mean()
    d[idx] = 2 * d[~idx].max()
    cost = d**2 if quadratic else np.abs(d)
    i,j = linear_sum_assignment(cost, maximize=False)
    res = dst[j] if return_points else np.stack([i,j],axis=0)
    return res


class ICP:
    def __init__(self, x):
        self.__template = vp.Mesh([x, None])
    def __call__(self, x, rigid = True):
        mesh = vp.Mesh([x, None])
        mesh.alignTo(self.__template, rigid=rigid)
        return mesh.points()


@memory.cache
def icp_alignement(point_clouds, iterations = 5, idx = None):
    if not idx:
        idx = np.argmin([x.shape[0] for x in point_clouds])
    average = point_clouds[idx].copy()
    for _ in range(iterations):
        icp = ICP(average)
        aligned = [icp(pts) for pts in point_clouds]
        func = lambda x: nearest_neighbours(average, x, quadratic = True)
        aligned = Parallel(n_jobs=-1)(delayed(func)(pts) for pts in aligned)
        average = np.mean(aligned, axis=0)
    return idx, np.stack(aligned, axis = 0)



if __name__ == "__main__":
    dir_in = "../example_data"

    loader = mesh_loader(os.path.join(dir_in, "processed_data"))

    src = loader("1")
    dst = loader("2")
    i,j = np.load(os.path.join(dir_in, "match_results", "1_2.npy"))

    deformation_transform(src[i], dst[j], src)
    _,aligned = icp_alignement([m.v for m in [src,dst]], iterations = 2, idx = 0)

    mu = np.mean(aligned, axis=0)
    dev = np.linalg.norm(np.std(aligned,axis=0), axis=-1)

    average = src.copy()
    average.v = mu
    dev = average.filter(dev, k=-1)
    average.display(scalar=dev)

    classes = np.asarray(2*["tooth"])

    fout = os.path.join(dir_in, "aligned_point_clouds", "example.npz")
    np.savez(fout, point_clouds = aligned, classes = classes )
