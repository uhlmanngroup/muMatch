import math
import os
import sys

import igl
import vedo as vp
import numpy as np
from joblib import Memory, Parallel, delayed
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, spsolve
from scipy.spatial import Delaunay, KDTree

sys.path.insert(1, os.path.join(os.path.dirname(__file__)))

import multiprocessing as mp
from typing import List

from joblib import Parallel, delayed

# To have a cache for computations which are taking time to complete
memory = Memory(location=".joblib_cache", verbose=0)


""" ================================================================================= """
##################             Point Cloud Utilities           ##################
""" ================================================================================= """


def readMesh(fn, normalise=False):
    mesh = vp.load(fn)
    v = mesh.points(copy=True).astype(np.double)
    f = np.asarray(mesh.faces())
    if normalise:
        v -= np.mean(v, axis=0)
        v /= math.sqrt(area(v,f))
    return v,f


def differenceMatrix(array1, array2=None, norm=True):

    """
    Pairwise distances between points in two arrays. Scales like N*M
    where N and M are the sizes of the point clouds in question.

    Args:
    array1: An array of shape [N1,D], where D is the dimension.
    array2: An array of shape [N2,D].

    Returns
    An array of shape [N2,N1].
    """

    if array2 is None:
        array2 = array1.copy()

    max_size = 1e9
    s1 = array1.size * array1.itemsize
    s2 = array2.size * array2.itemsize

    if len(array1.shape) == 1:
        array1 = np.expand_dims(array1, 1)

    if len(array2.shape) == 1:
        array2 = np.expand_dims(array2, 1)

    n1 = array1.shape[0]
    n2 = array2.shape[0]
    dim = list(array1.shape)[1:]
    result: np.ndarray

    if max_size < n2 * s1 + n1 * s2:
        func = lambda x: array1 - x
        res = Parallel(n_jobs=-1)(delayed(func)(y) for y in array2)
        result = np.stack(res, axis=0)
    else:
        expanded_array1 = np.tile(array1, [n2] + len(dim) * [1])
        expanded_array2 = np.tile(
            array2[:, np.newaxis, ...], [1, n1] + len(dim) * [1]
        ).reshape(*[n1 * n2] + dim)
        result = (expanded_array1 - expanded_array2).reshape(n2, n1, *dim)
    if norm:
        result = np.linalg.norm(result, axis=-1)
    return result


""" ================================================================================= """
##################             Alignment Operations            ##################
""" ================================================================================= """


def centre_and_norm(X):
    X -= np.mean(X, axis=0)
    X /= np.sqrt(np.mean(np.sum(X ** 2, axis=-1)))
    return


def orthogonalProcrustes(X, Y):
    """
    X ~ Y.dot(R)
    """
    U, _, VT = np.linalg.svd(X.T @ Y)
    R = (VT.T).dot(U.T)
    return R


""" ================================================================================= """
##################                Laplacian Ops                ##################
""" ================================================================================= """


def laplace_beltrami_operator(l, m):
    minv = diags(1 / (m.diagonal() + 1e-12))
    return minv.dot(l)


def laplace_eigen_decomposition(
    l: np.array, m: np.array, k: int
) -> List[np.array]:
    evals, evecs = eigsh(A=-l, k=k, M=m, which="SM")
    evecs /= np.sqrt(np.sum(m.dot(evecs ** 2), axis=0, keepdims=True))
    return [evals, evecs]


def laplacianSmoothing(l, m, s, mu=1e-3):
    m_inv_l = spsolve(m, l)
    ql = l.T @ m_inv_l
    return spsolve(mu * ql + (1 - mu) * m, m.dot(s))


def gaussianCurvature(v, f):
    s = igl.gaussian_curvature(v, f)
    b = boundaryVertices(f)
    s[b] = 0
    s[np.isnan(s)] = 0
    return s


def meanCurvature(v, f):
    print("this is using the wrong def of laplace_beltrami_operator")
    assert(False)
    lb = laplace_beltrami_operator(v, f)
    hn = -lb.dot(v)
    h = np.linalg.norm(hn, axis=-1)
    b = boundaryVertices(f)
    h[b] = 0
    h[np.isnan(h)] = 0
    return h


""" ================================================================================= """
##################               Mesh Operations               ##################
""" ================================================================================= """


def vedo_decimate(v, f, N=None, frac=None):
    import vedo as vp

    vedo_mesh = vp.Mesh([v, f])
    args = {"method": "quadric"}

    if N is not None:
        args["N"] = int(N)
    elif frac is not None:
        args["fraction"] = frac
    else:
        raise ValueError

    vedo_mesh.decimate(**args)
    u, f = vedo_mesh.points(), np.asarray(vedo_mesh.faces())
    d = differenceMatrix(v, u)
    idx = d.argmin(axis=-1)
    return True, u, f, None, idx


def vedo_subdivide(v, f, N=1, method=0):
    import vedo as vp
    vedo_mesh = vp.Mesh([v, f])
    vedo_mesh.subdivide(N=N, method=method)
    u,f = vedo_mesh.points(), np.asarray(vedo_mesh.faces())
    d = differenceMatrix(v, u)
    idx = d.argmin(axis=-1)
    return u,f,idx


def reorder_mesh(v, f, idx):
    nf = np.zeros(shape=f.shape, dtype=f.dtype)
    for i, j in enumerate(idx):
        nf[f == j] = i
    nv = v[idx]
    return nv, nf


def extractEdges(faces):

    if ~isinstance(faces, np.ndarray):
        faces = np.array(faces)

    edges = np.concatenate(
        [faces[:, i] for i in ([0, 1], [1, 2], [2, 0])], axis=0
    )
    edges = np.sort(edges, axis=1)
    _, ind = np.unique(edges, return_index=True, axis=0)
    return edges[ind]


def edge_lengths(v,f):
    i,j = extractEdges(f).T
    return np.linalg.norm(v[i]-v[j], axis=-1)


def boundaryVertices(faces):
    if ~isinstance(faces, np.ndarray):
        faces = np.array(faces)
    e = np.concatenate([faces[:, i] for i in ([0, 1], [1, 2], [2, 0])], axis=0)
    e = np.sort(e, axis=1)
    e, c = np.unique(e, return_counts=True, axis=0)
    return np.unique(e[c == 1])


def meshNeighbours(faces):

    nodes_list = [[] for _ in range(50)]
    neighbours_list = [[] for _ in range(50)]
    numbers_list = []

    p, q = extractEdges(faces).T
    nodes = np.unique(faces)

    for i in nodes:
        nhbrs = np.concatenate((q[p == i], p[q == i]))
        num = len(nhbrs)
        numbers_list.append(num)
        nodes_list[num].append(i)
        neighbours_list[num].append(nhbrs)

    numbers_list = np.unique(numbers_list).tolist()
    nodes_list = [np.array(i) for i in nodes_list if 0 < len(i)]
    neighbours_list = [np.stack(i) for i in neighbours_list if 0 < len(i)]

    return numbers_list, nodes_list, neighbours_list


@memory.cache
def geodesicMatrix(v, f, i1=np.array([]), i2=np.array([])):
    """
    Find the pairwise geodesic distances between a set of vertices
    within a mesh. Make sure that i1.size < i2.size
    """
    if i1.size == 0:
        i1 = np.arange(v.shape[0])

    if i2.size == 0:
        i2 = i1

    func = lambda i: igl.exact_geodesic(v, f, np.array([i]), i2)
    d = Parallel(n_jobs=-1)(delayed(func)(j) for j in i1)
    return np.stack(d)


def area(v, f):
    return face_areas(v,f).sum()


def face_areas(v,f):
    a, b, c = f.T
    areas = 0.5 * np.cross(v[b] - v[a], v[c] - v[a])
    return np.linalg.norm(areas, axis=-1)

def face_normals(v, f):
    a, b, c = f.T
    normals = np.cross(v[b] - v[a], v[c] - v[a])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9
    return normals


""" ================================================================================= """
##################               Propagation Ops               ##################
""" ================================================================================= """


def metric_sampling(g, target):
    idx = [np.argmax(g.mean(axis=0))]
    while len(idx) < target:
        idx.append(np.argmax(g[idx].min(axis=0)))
    return np.asarray(idx)


def propogate_points(
    src: np.array, dst: np.array, points: np.array, f: np.array
) -> np.array:
    n_src = face_normals(src, f)
    n_dst = face_normals(dst, f)
    _, i, q = igl.point_mesh_squared_distance(points, src, f)
    l = np.linalg.norm((points - q) * n_src[i], axis=-1, keepdims=True)
    j = f[i]
    u, v, w = np.asarray(src[j.T], order="C", dtype=q.dtype)
    bc = igl.barycentric_coordinates_tri(q, u, v, w)
    bc[np.isnan(bc).any(axis=-1)] = 1 / 3
    pts = np.sum(bc[..., np.newaxis] * dst[j], axis=1)
    pts += l * n_dst[i]
    return pts


def extrapolate_geodesic_matrix(
    src: np.array, dst: np.array, g: np.array, f: np.array
) -> np.array:
    _, i, q = igl.point_mesh_squared_distance(dst, src, f)
    j = f[i]
    l = np.linalg.norm(src[j] - np.expand_dims(q, 1), axis=-1)
    h = (g[j] + np.expand_dims(l, -1)).min(axis=1)
    h = (h[:, j] + np.expand_dims(l, 0)).min(axis=-1)
    np.fill_diagonal(h, 0)
    return 0.5 * (h + h.T)


def extrapolate_scalars(
    src: np.array, dst: np.array, scalars: np.array, f: np.array
) -> np.array:
    if len(scalars.shape) == 1:
        scalars = scalars[..., np.newaxis]
    _, i, q = igl.point_mesh_squared_distance(dst, src, f)
    j = f[i]
    u, v, w = np.asarray(src[j.T], order="C", dtype=np.double)
    bc = igl.barycentric_coordinates_tri(q, u, v, w)
    return np.sum(bc[..., np.newaxis] * scalars[j], axis=1)


""" ================================================================================= """
##################                   Misc                      ##################
""" ================================================================================= """


def sign(x):
    return 1 - 2 * np.asarray(x < 0).astype(np.int)


def safe_divide(x, y, eps=1e-9):
    return sign(y) * x / (np.abs(y) + eps)


def safe_inverse(x, eps=1e-9):
    return safe_divide(1, x, eps)


def safe_sqrt(x):
    return sign(x) * np.sqrt(np.abs(x))


def first_n(x, n, axis=0):
    i = np.argsort(x, axis=axis).take(indices=range(0, n), axis=axis)
    y = np.zeros(x.shape, dtype=np.bool)
    np.put_along_axis(y, indices=i, values=True, axis=axis)
    return y


""" ================================================================================= """
##################              Path Operations                ##################
""" ================================================================================= """


def arcLength(p):
    p = np.append(p, np.expand_dims(p[0], 0), axis=0)
    dp = np.diff(p, axis=0)
    delta = np.linalg.norm(dp, axis=-1)
    s = [0]
    for dl in delta:
        s.append(s[-1] + dl)
    s = np.array(s) / s[-1]
    return s[:-1]


""" ================================================================================= """
###################               End                       #####################
""" ================================================================================= """


if __name__ == "__main__":
    pass
