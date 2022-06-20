import os

import numpy as np
import pymeshfix
import vedo as vp
from tqdm import tqdm
from vedo import Mesh as vMesh

from ..tools.mesh_class import mesh_loader
from . import feature_descriptors as fd


def clean_mesh(vedo_mesh: vMesh):
    v, f = vedo_mesh.points(), np.asarray(vedo_mesh.faces())
    meshfix = pymeshfix.MeshFix(v, f)
    meshfix.repair()
    return vp.Mesh([meshfix.v, meshfix.f])


def batch_preprocess(dir_in, dir_out, config):

    loader = mesh_loader(dir_out, k=config["functional_dimension"], type="")
    dirs = {
        fn: os.path.join(dir_out, fn)
        for fn in ["meshes", "geodesic_matrices", "eigen", "signatures"]
    }

    for k in dirs:
        if not os.path.exists(dirs[k]):
            os.mkdir(dirs[k])

    files = []

    print("\n" + 60 * "-")
    print("Resampling meshes")
    print(60 * "-" + "\n")

    raw_files = os.listdir(dir_in)
    # TODO: Should hidden files be excluded in general? I do not think any hidden files should be used.
    raw_files = [file for file in raw_files if ".DS_Store" not in file]
    target_size = config["number_vertices"]

    for f in tqdm(raw_files):
        fin = os.path.join(dir_in, f)
        fn, _ = os.path.splitext(f)
        fout = os.path.join(dirs["meshes"], fn + ".ply")
        files.append(fn)
        if os.path.exists(fout):
            continue
        mesh = vp.load(fin)
        if config["clean"]:
            mesh = clean_mesh(mesh)
        while mesh.N() < target_size:
            mesh.subdivide(N=1, method=0)
        mesh.decimate(N=target_size)
        mesh.write(fout)

    print("\n" + 60 * "-")
    print("Calculating geodesic matrices")
    print(60 * "-" + "\n")

    for fn in tqdm(files):
        fgeo = os.path.join(dirs["geodesic_matrices"], fn + ".npy")
        if os.path.exists(fgeo):
            continue
        try:
            mesh = loader(fn)
            np.save(fgeo, mesh.g)
        except Exception:
            print(f"Geodesic matrix error with: {fn}")

    print("\n" + 60 * "-")
    print("Calculating Laplacian eigendecomposition")
    print(60 * "-" + "\n")

    sizes, minima, maxima = [[] for _ in range(3)]

    for fn in tqdm(files):
        feigen = os.path.join(dirs["eigen"], fn + ".npz")
        try:
            mesh = loader(fn)
            evals, evecs = mesh.eigen
            evecs_t = np.transpose(mesh.mass @ evecs)
            sizes.append(mesh.num_vertices())
            minima.append(1e-2 + evals[0 < evals].min())
            maxima.append(evals.max())
            if not os.path.exists(feigen):
                np.savez(feigen, evals=evals, evecs=evecs, evecs_t=evecs_t)
        except Exception as err:
            print(f"{fn} Eigen-decomposition error : {err}")

    emin = float(min(minima))
    emax = float(max(maxima))
    num_hks, num_wks, num_gaussian = [
        config[key] for key in ["number_hks", "number_wks", "number_gaussian"]
    ]
    descriptor = fd.DescriptorClass(
        emin, emax, num_wks=num_wks, num_hks=num_hks, num_gaussian=num_gaussian
    )

    print("\n" + 60 * "-")
    print("Calculating signature functions")
    print(60 * "-" + "\n")

    for fn in tqdm(files):
        fsigs = os.path.join(dirs["signatures"], fn + ".npy")
        if os.path.exists(fsigs):
            continue
        try:
            mesh = loader(fn)
            signatures = descriptor(mesh)
            np.save(fsigs, signatures)
        except Exception:
            print(f"Signature function error with {fn}: skipping.")

    print("\n\n" + "Preprocessing complete." + "\n\n")
    return
