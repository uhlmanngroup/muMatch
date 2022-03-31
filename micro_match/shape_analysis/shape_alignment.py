import os
from itertools import tee

import networkx as nx
import numpy as np
import vedo as vp
from joblib import Memory, Parallel, delayed
from scipy.optimize import linear_sum_assignment

from ..tools import geometric_utilities as util
from ..tools.mesh_class import mesh_loader

# To have a cache for computations which are taking time to complete
memory = Memory(location=".joblib_cache", verbose=0)


def pairwise(iterable):
    # s -> (s0,s1), (s1,s2), (s2, s3), ...
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def split(fn):
    f, _ = os.path.splitext(fn)
    f1, f2 = f.split("_")
    return [f1, f2]


def readResults(input_dir, fn1, fn2):
    fin = os.path.join(input_dir, fn1 + "_" + fn2 + ".npy")
    if os.path.isfile(fin):
        x, y = np.load(fin)
    else:
        fin = os.path.join(input_dir, fn2 + "_" + fn1 + ".npy")
        y, x = np.load(fin)
    return x, y


def rbf(x):
    i = x < 2
    y = np.empty(x.shape, dtype=x.dtype)
    y[i] = np.log(x[i] ** (x[i] ** 2))
    y[~i] = (x[~i] ** 2) * np.log(x[~i])
    return y


def thin_plate_spline(x0, y0, x):
    r0 = util.differenceMatrix(x0)
    A = rbf(r0)
    w = np.linalg.lstsq(a=A, b=(y0 - x0))[0]
    r = util.differenceMatrix(x0, x)
    return rbf(r) @ w


def deformation_transform(src, dst, *args):
    cs, cd = [m.centroid() for m in (src, dst)]
    R = util.orthogonalProcrustes(dst.v - cd, src.v - cs)
    for x in args:
        x.shift(-cs).rotate(R).shift(cd)
    return


def nearest_neighbours(src, dst, quadratic=False, return_points=True):
    d = util.differenceMatrix(dst, src)
    idx = np.isnan(d)
    d /= d[~idx].mean()
    d[idx] = 2 * d[~idx].max()
    cost = d ** 2 if quadratic else np.abs(d)
    i, j = linear_sum_assignment(cost, maximize=False)
    res = dst[j] if return_points else np.stack([i, j], axis=0)
    return res


class ICP:
    def __init__(self, x):
        self.__template = vp.Mesh([x, None])

    def __call__(self, x, rigid=True):
        mesh = vp.Mesh([x, None])
        mesh.alignTo(self.__template, rigid=rigid)
        return mesh.points()


@memory.cache
def icp_alignement(point_clouds, iterations=5, idx=None):
    if not idx:
        idx = np.argmin([x.shape[0] for x in point_clouds])
    average = point_clouds[idx].copy()
    for _ in range(iterations):
        icp = ICP(average)
        aligned = [icp(pts) for pts in point_clouds]
        func = lambda x: nearest_neighbours(average, x, quadratic=True)
        aligned = Parallel(n_jobs=-1)(delayed(func)(pts) for pts in aligned)
        average = np.mean(aligned, axis=0)
    return idx, np.stack(aligned, axis=0)


def build_graph_from_directory(dir_in):
    files = []
    for f in os.listdir(dir_in):
        name, ext = os.path.splitext(f)
        if ext == ".npy":
            files.append(f)

    pairs = [split(fn) for fn in files]
    names = list({item for sublist in pairs for item in sublist})
    nodes = [n for n in range(len(names))]
    _map_ = {name: n for n, name in zip(nodes, names)}

    g = nx.Graph()
    g.add_nodes_from(nodes)

    for f1, f2 in pairs:
        g.add_edge(_map_[f1], _map_[f2])

    return names, g


def process_directory(data_dir, match_dir, display=False):
    loader = mesh_loader(data_dir)
    names, G = build_graph_from_directory(match_dir)
    connected = list(max(nx.connected_components(G), key=len))

    graph_centrality = nx.degree_centrality(G)
    connected_centrality = [graph_centrality[n] for n in connected]
    target_idx = max(
        range(len(connected_centrality)), key=connected_centrality.__getitem__
    )

    meshes = [loader(names[n], normalise=True) for n in connected]
    aligned = []

    for source_idx in connected:
        path = nx.shortest_path(G, source=source_idx, target=target_idx)
        x = meshes[source_idx].copy()
        for a, b in pairwise(path):
            i, j = readResults(match_dir, names[a], names[b])
            deformation_transform(meshes[a][i], meshes[b][j], x)
        aligned.append(x.v)

    template_idx, point_clouds = icp_alignement(aligned, iterations=2)

    if display:
        v = np.mean(point_clouds, axis=0)
        deviation = np.linalg.norm(np.std(point_clouds, axis=0), axis=-1)
        scalar = meshes[template_idx].filter(deviation, k=-1)

        average = meshes[template_idx].copy()
        average.v = v
        average.display(scalar=scalar)

    return names, point_clouds


if __name__ == "__main__":
    pass
