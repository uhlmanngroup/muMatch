## Python imports
import os
import sys
import math
import json
import itertools
from functools import reduce

## Numerical
import numpy as np
from scipy.optimize import linear_sum_assignment

## Mesh and Graphics
import igl
import vedo as vp
import matplotlib.pyplot as plt

# Local imports
import functional_maps.zoom_out as zo
import functional_maps.functional_mapping as fm
import product_manifold_filters.product_manifold_filter as pmf

sys.path.insert(1, "../tools")

import geometric_utilities as util
from mesh_class import Mesh, mesh_loader



## Global vars
''' ============================================================= '''
dir_in  = "../example_data/processed_data"
dir_out = "../example_data/match_results"

loader = mesh_loader(dir_in)
fig = vp.plotter.Plotter()
''' ============================================================= '''


def readJSON(fn):
    data: dict
    with open(fn) as file:
        data = json.load(file)
    return data


def compute_correspondence(src, dst):
    C = fm.correspondenceMatrixSolver(src, dst, k = 4, optimise = True)
    C = zo.zoomout_refinement(src, dst)(C)
    P = fm.soft_correspondence(src, dst, C)
    assign = lambda x: linear_sum_assignment(x, maximize=True)
    i,j = pmf.product_manifold_filter_assignment(assign, src.g, dst.g, P, sigma = .75, gamma = .5, iterations = 2)
    return i,j


def main(f1, f2, display_result = False, save_result = False):
    ## Loading ##
    src = loader(f1, normalise=True)
    dst = loader(f2, normalise=True)

    fn1,fn2 = f1,f2
    if (dst.N() < src.N()):
        fn1,fn2 = f2,f1
        src,dst = dst,src

    fout = os.path.join(dir_out, fn1 + "_" + fn2 + ".npy")

    i,j = np.load(fout) if os.path.exists(fout) else compute_correspondence(src, dst)
    dg = np.abs(src.g[i][:,i] - dst.g[j][:,j]).mean()
    print(fn1 + " -> " + fn2 + ": geodesic distortion = {0:.3f}".format(dg))

    if display_result:
        R = util.orthogonalProcrustes(src.v[i], dst.v[j])
        dst.rotate(R)
        fig.add([m.vedo() for m in (src,dst)])
        fig.show()
        fig.clear()

    if save_result:
        np.save(fout, np.stack([i,j],axis=0))

    return
    

if __name__ == '__main__':

    main("1", "2", display_result=True, save_result=True)
