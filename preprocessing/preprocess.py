## Standard Libraries ##
import sys
import os
import json

## Numerical Libraries ##
import numpy as np
import math

## Mesh related ##
import igl
import vedo as vp
import pymeshfix

## Other ##

from tqdm import tqdm

## Local Imports ##
sys.path.insert(1, "../tools")
sys.path.insert(1, "../meshcorr/functional_maps")

import geometric_utilities as util
import feature_descriptors as fd
from   mesh_class import Mesh, mesh_loader

clean = False

def clean_mesh(mesh):
    v,f = mesh.points(), np.asarray(mesh.faces())
    meshfix = pymeshfix.MeshFix(v, f)
    meshfix.repair()
    return vp.Mesh([meshfix.v, meshfix.f])


def batch_preprocess(dir_in, dir_out, type, target_size = 3000, num_eigenfunctions = 100, num_wks=100, num_hks=0):

    fconfig = os.path.join(dir_out, 'config' + '.json')
    config  =  {'N': target_size, 'num_eigenfunctions': num_eigenfunctions, 'type': type, 'num_hks': num_hks, 'num_wks': num_wks, 'num_gaussian': 18}
    with open(fconfig, 'w') as json_file:
        json.dump(config, json_file)

    loader = mesh_loader(dir_out)
    dirs = {fn: os.path.join(dir_out, fn) for fn in ["meshes", "geodesic_matrices", "eigen", "signatures"]}

    for k in dirs:
        if not os.path.exists(dirs[k]):
            os.mkdir(dirs[k])

    files = []

    print("\n" + 60*"-")
    print("Resampling meshes")
    print(60*"-" + "\n")

    raw_files = os.listdir(dir_in)

    for f in tqdm(raw_files):
        fin = os.path.join(dir_in, f)
        fn,_ = os.path.splitext(f)
        fout = os.path.join(dirs["meshes"], fn + ".ply")
        files.append(fn)
        if os.path.exists(fout):
            continue
        mesh = vp.load(fin)
        if clean:
            mesh = clean_mesh(mesh)
        while (mesh.N() < target_size):
            mesh.subdivide(N=1, method=0)
        mesh.decimate(N=target_size)
        mesh.write(fout)

    print("\n" + 60*"-")
    print("Calculating geodesic matrices")
    print(60*"-" + "\n")

    for fn in tqdm(files):
        fgeo = os.path.join(dirs["geodesic_matrices"], fn + ".npy")
        if os.path.exists(fgeo):
            continue
        try:
            mesh = loader(fn)
            np.save(fgeo, mesh.g)
        except:
            print('Geodesic matrix error with: {}'.format(fn))

    print("\n" + 60*"-")
    print("Calculating Laplacian eigendecomposition")
    print(60*"-" + "\n")

    sizes,minima,maxima = [[] for _ in range(3)]

    for fn in tqdm(files):
        feigen = os.path.join(dirs["eigen"], fn + ".npz")
        try:
            mesh = loader(fn)
            evals,evecs = mesh.eigen
            evecs_t = np.transpose(mesh.mass @ evecs)
            sizes.append( mesh.N() )
            minima.append( 1e-2 + evals[0 < evals].min() )
            maxima.append( evals.max() )
            if not os.path.exists(feigen):
                np.savez(feigen, evals = evals, evecs = evecs, evecs_t = evecs_t)
        except:
            print("Eigen-decomposition error with {}: skipping.".format(fn))


    emin = float(min(minima))
    emax = float(max(maxima))
    descriptor =  fd.descriptor_class(emin, emax, num_wks=num_wks, num_hks=num_hks)

    print("\n" + 60*"-")
    print("Calculating signature functions")
    print(60*"-" + "\n")

    for fn in tqdm(files):
        fsigs = os.path.join(dirs["signatures"], fn + ".npy")
        if os.path.exists(fsigs):
            continue
        try:
            mesh = loader(fn)
            signatures = descriptor(mesh)
            np.save(fsigs, signatures)
        except:
            print("Signature function error with {}: skipping.".format(fn))

    print("\n\n" + "Preprocessing complete." + "\n\n" )
    return

if __name__ == "__main__":
    dir_in  = "../example_data/raw"
    dir_out = "../example_data/processed_data"

    batch_preprocess(dir_in, dir_out, type="neus", target_size = 2000, num_wks=100, num_hks=0)
