#! /usr/bin/env python

import os
import igl
import argparse
import numpy as np
import vedo as vp


from   tools.mapping_viewer import renderCorrespondence_vtk
import tools.geometric_utilities as util

num_eigenfunctions = 100

__doc__ = "Visualise meshes put in correspondence interactively"


def readMesh(fn):
    v,f  =  igl.read_triangle_mesh(fn)
    v   -= v.mean(axis=0)
    v   /= np.sqrt(util.area(v,f))
    return v,f


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, prog="visualise_interactive")

    parser.add_argument("m1", help="Source mesh")
    parser.add_argument("m2", help="Target mesh")

    args = parser.parse_args()

    v1, f1 = igl.read_triangle_mesh(args.m1)
    v2, f2 = igl.read_triangle_mesh(args.m2)

    m1 = vp.Mesh([v1, f1])
    m2 = vp.Mesh([v2, f2])

    renderCorrespondence_vtk(m1, m2)
