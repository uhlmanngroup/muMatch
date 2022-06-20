import numpy as np
from scipy import ndimage
from skimage.measure import block_reduce

""" ======================================================================================================= """
"""                                            SHOT                                                         """
""" ======================================================================================================= """


def signOrientate(x, dx, N):
    pos_x = np.sum(np.sum(x * dx, axis=-1) < N)
    if pos_x < N - pos_x:
        x *= -1
    return None


def getSHOTLocalRF(mesh, idx, neighbours, radius):
    N = neighbours.size
    if N < 3:
        raise Exception("Not enough points for computing SHOT descriptor")
    dv = mesh.v[neighbours] - mesh.v[idx]
    w = radius - mesh.g[idx, neighbours]
    M = np.einsum("ki,kj->kij", dv, dv) * w[:, np.newaxis, np.newaxis]
    M = M.sum(axis=0) / w.sum()
    evals, evecs = np.linalg.eigh(M)
    evecs = evecs[np.argsort(evals)]
    X, Z = evecs[[-1, 0]]
    signOrientate(X, dv, N)
    signOrientate(Z, dv, N)
    Y = np.cross(Z, X)
    return X, Y, Z


def neighbouringPoints(mesh, index, radius):
    neighbours = np.argwhere(mesh.g[index] < radius).flatten()
    return np.setdiff1d(neighbours, [index])


def filter_histogram(x):
    tri = np.array([1, 2, 1]) / 4.0
    x = ndimage.convolve(
        x, np.expand_dims(tri, axis=(1, 2, 3)), mode="nearest"
    )
    x = ndimage.convolve(x, np.expand_dims(tri, axis=(0, 2, 3)), mode="wrap")
    x = ndimage.convolve(
        x, np.expand_dims(tri, axis=(0, 1, 3)), mode="nearest"
    )
    x = ndimage.convolve(
        x, np.expand_dims(tri, axis=(0, 1, 2)), mode="nearest"
    )
    return block_reduce(x, block_size=(3, 3, 3, 3), func=np.mean)


def calculate_SHOT_descriptors(mesh, radius):
    """
    SHOT descriptors
    there will be 11 x 8 x 2 x 2 bins (normal angle, azimuthal angle, radial and elevation) in the end
    these have upsampled by x3 for filtering purposes
    """

    cosine_bins = np.linspace(-1, 1, num=33, endpoint=False)
    azimuthal_bins = np.linspace(-1, 1, num=24, endpoint=False)
    radial_bins = np.linspace(0, radius, num=6, endpoint=False)
    elevation_bins = np.linspace(-1, 1, num=6, endpoint=False)

    def describe_vertex(feat_index):

        centralPoint = mesh.v[feat_index]
        neighs = neighbouringPoints(mesh, feat_index, radius)
        ref_X, ref_Y, ref_Z = getSHOTLocalRF(mesh, feat_index, neighs, radius)
        dv = mesh.v[neighs] - centralPoint
        normals = mesh.normals[neighs]

        xInFeatRef = np.sum(dv * ref_X, axis=-1)
        yInFeatRef = np.sum(dv * ref_Y, axis=-1)
        zInFeatRef = np.sum(dv * ref_Z, axis=-1)

        radii = np.linalg.norm(dv, axis=-1)
        elevation = zInFeatRef / (radii + 1e-3)
        azimuth = np.angle(xInFeatRef + 1.0j * yInFeatRef) / np.pi
        cosine = np.sum(normals * ref_Z, axis=-1)

        c = np.digitize(cosine, cosine_bins, right=False) - 1
        a = np.digitize(azimuth, azimuthal_bins, right=False) - 1
        r = np.digitize(radii, radial_bins, right=False) - 1
        e = np.digitize(elevation, elevation_bins, right=False) - 1

        histogram = np.zeros((33, 24, 6, 6))
        histogram[c, a, r, e] += 1
        return filter_histogram(histogram).flatten()

    N = mesh.num_vertices()
    signatures = [describe_vertex(idx) for idx in range(N)]
    return np.stack(signatures, axis=0)
