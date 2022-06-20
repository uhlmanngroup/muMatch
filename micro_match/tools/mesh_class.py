import json
import os

import igl
import numpy as np

from . import geometric_utilities as util

""" ================================================================================= """
"""                         Mesh Class Definition                                     """
""" ================================================================================= """


class Mesh:
    def __init__(
        self,
        vertices=np.array([]),
        faces=np.array([]),
        geodesic_matrix=np.array([]),
        num_eigenvectors=100,
        type="",
    ):
        self.__vertices = vertices
        self.__faces = faces
        self.__geodesic_matrix = geodesic_matrix
        self.__num_eigenvectors = num_eigenvectors
        self.__laplacian = np.array([])
        self.__mass = np.array([])
        self.__normals = np.array([])
        self.__eigen = [np.array([]) for _ in range(2)]
        self.__scalars = {}
        self.__type = type
        self.__name = ""

    def load(self, fn):
        v, f = igl.read_triangle_mesh(fn)
        self.__vertices = v
        self.__faces = f
        return self

    def __getitem__(self, idx):
        if idx.size == self.num_vertices():
            v, f = util.reorder_mesh(self.vertices, self.faces, idx)
        else:
            v = self.vertices[idx]
            f = np.array([])

        g = (
            np.array([])
            if (self.__geodesic_matrix.size == 0)
            else self.geodesic_matrix[idx][:, idx]
        )
        res = Mesh(
            v,
            f,
            geodesic_matrix=g,
            num_eigenvectors=self.num_eigenvectors,
            type=self.type,
        )
        for fn in self.scalars:
            try:
                res.scalars[fn] = self.scalars[fn][idx].copy()
            except Exception:
                pass

        return res

    def copy(self):
        res = Mesh(
            vertices=self.__vertices.copy(),
            faces=self.__faces.copy(),
            geodesic_matrix=self.__geodesic_matrix.copy(),
            num_eigenvectors=self.__num_eigenvectors,
            type=self.__type,
        )
        res.mass = self.__mass.copy()
        res.normals = self.__normals.copy()
        res.eigen = [e.copy() for e in self.__eigen]
        res.name = self.name
        for fn in self.scalars:
            res.scalars[fn] = self.scalars[fn].copy()
        return res

    """ ================================================ """
    """                     Properties                   """
    """ ================================================ """

    def area(self):
        return util.area(self.vertices, self.faces)

    def num_vertices(self):
        """
        Returns the number of vertices of a mesh.

        Returns
        -------
            Number of vertices of mesh
        """
        return self.__vertices.shape[0]

    """ ================================================ """
    """                 Getters and Setters              """
    """ ================================================ """

    @property
    def vertices(self):
        return self.__vertices

    @vertices.setter
    def vertices(self, v):
        self.__vertices = v

    @property
    def faces(self):
        return self.__faces

    @faces.setter
    def faces(self, f):
        self.__faces = f

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def type(self):
        return self.__type

    @property
    def scalars(self):
        return self.__scalars

    @scalars.setter
    def scalars(self, scalars):
        self.__scalars = scalars

    @property
    def num_eigenvectors(self):
        return self.__num_eigenvectors

    @num_eigenvectors.setter
    def num_eigenvectors(self, k):
        self.__eigen = (
            [e[..., :k] for e in self.__eigen]
            if (k <= self.__num_eigenvectors)
            else [np.array([]) for _ in range(2)]
        )
        self.__num_eigenvectors = k

    @property
    def geodesic_matrix(self):
        if self.__geodesic_matrix.size == 0:
            # if (self.N() <= 500):
            self.__geodesic_matrix = util.geodesicMatrix(
                self.vertices, self.faces
            )
            # else:
            #    _,v,f,_,i = util.vedo_decimate(self.v, self.f, N = 500)
            #    g_sparse = util.geodesicMatrix(v,f)
            #    self.__g = util.extrapolate_geodesic_matrix(v, self.v, g_sparse, f)
        return self.__geodesic_matrix

    @geodesic_matrix.setter
    def geodesic_matrix(self, geodesicmatrix):
        self.__geodesic_matrix = geodesicmatrix

    @property
    def gaussian(self):
        if self.__gaussian.size == 0:
            self.__gaussian = util.gaussianCurvature(self.vertices, self.faces)
        return self.__gaussian

    @gaussian.setter
    def gaussian(self, gaussian):
        self.__gaussian = gaussian

    @property
    def normals(self):
        if self.__normals.size == 0:
            self.__normals = igl.per_vertex_normals(
                self.__vertices, self.__faces, weighting=0
            )
        return self.__normals

    @normals.setter
    def normals(self, normals):
        self.__normals = normals

    @property
    def eigen(self):
        if any([e.size == 0 for e in self.__eigen]):
            self.__eigen = util.laplace_eigen_decomposition(
                self.discrete_laplacian, self.mass, self.num_eigenvectors
            )
        return self.__eigen

    @eigen.setter
    def eigen(self, eigen):
        self.__num_eigenvectors = eigen[0].size
        self.__eigen = eigen

    @property
    def mass(self):
        if self.__mass.size == 0:
            self.__mass = igl.massmatrix(
                self.__vertices, self.__faces, igl.MASSMATRIX_TYPE_VORONOI
            )
        return self.__mass

    @mass.setter
    def mass(self, mass):
        self.__mass = mass

    @property
    def discrete_laplacian(self):
        if self.__laplacian.size == 0:
            self.__laplacian = igl.cotmatrix(self.vertices, self.faces)
        return self.__laplacian

    @discrete_laplacian.setter
    def discrete_laplacian(self, laplacian):
        self.__laplacian = laplacian

    """ ================================================ """
    """                Scalar Conversion                 """
    """ ================================================ """

    def pointwise_2_vector(self, scalar, k=-1):
        evecs = self.mass @ self.eigen[-1]
        if 0 < k:
            evecs = evecs[:, :k]
        return evecs.T @ scalar

    def vector_2_pointwise(self, vector):
        _, evecs = self.eigen
        k = vector.shape[0]
        return evecs[:, :k] @ vector

    def filter(self, scalar, k=-1):
        s = self.pointwise_2_vector(scalar, k)
        return self.vector_2_pointwise(s)

    def filter_array(self, array, k=-1):
        array = self.filter(array, k=k)
        return self.filter(array.T, k=k).T

    def dirac_deltas(self, i, k=-1):
        x = np.zeros((self.num_vertices(), i.size))
        j = np.arange(i.size)
        x[i, j] = 1
        return self.pointwise_2_vector(x, k)

    """ ================================================ """
    """                    Resampling                    """
    """ ================================================ """

    def decimate(self, N=None, frac=None):
        v, f = [x.copy() for x in (self.vertices, self.faces)]
        _, v, f, _, idx = util.vedo_decimate(
            v, f, N=N, frac=frac
        )  # igl.decimate(self.v, self.f, target)
        geodesic_matrix = (
            np.array([])
            if (self.__geodesic_matrix.size == 0)
            else self.geodesic_matrix[idx][:, idx].copy()
        )
        res = Mesh(
            v,
            f,
            geodesic_matrix=geodesic_matrix,
            num_eigenvectors=self.num_eigenvectors,
            type=self.type,
        )
        for fn in self.scalars:
            res.scalars[fn] = self.scalars[fn][idx].copy()
        return res, idx

    def upsample(self, target):
        v, f = [x.copy() for x in (self.vertices, self.faces)]
        idx: np.ndarray
        while v.shape[0] < target:
            v, f, idx = util.vedo_subdivide(v, f)
        geodesic_matrix = util.extrapolate_geodesic_matrix(
            self.vertices, v, self.geodesic_matrix, self.faces
        )
        res = Mesh(
            v,
            f,
            geodesic_matrix=geodesic_matrix,
            num_eigenvectors=self.num_eigenvectors,
            type=self.type,
        )
        return res.decimate(N=target)[0]

    """ ================================================ """
    """                Inplace Modifiers                 """
    """ ================================================ """

    def scale(self, factor):
        self.__vertices *= factor
        return self

    def rotate(self, R):
        self.__vertices = self.vertices @ R
        return self

    def shift(self, *args):
        dv: np.ndarray
        if 1 == len(args):
            (dv,) = args
        elif 3 == len(args):
            dx, dy, dz = args
            dv = np.array([dx, dy, dz], dtype=self.__vertices.dtype)
        else:
            raise Exception("wrong number of arguments. Must be 1 or 3.")
        self.__vertices += dv
        return self

    def centroid(self):
        return np.mean(self.vertices, axis=0)

    def centre(self):
        self.__vertices -= self.centroid()
        return self

    def centre_on_boundary(self):
        b = util.boundaryVertices(self.faces)
        self.__vertices -= self.vertices[b].mean()
        return

    def normalise(self):
        norm = np.sqrt(self.area())
        self.__vertices /= norm
        self.__geodesic_matrix /= norm
        return self

    """ ================================================ """
    """                     Display                      """
    """ ================================================ """

    def display(self, **kwargs):
        import vedo as vp

        msh = self.vedo()

        if "scalar" in kwargs:
            s = kwargs["scalar"]
            if isinstance(s, np.ndarray):
                msh.cmap("jet", s)
            elif s in self.scalars:
                msh.cmap("jet", self.scalars[s])
            else:
                raise ValueError("Argument not recognised.")
            msh.addScalarBar()

        if "alpha" in kwargs:
            msh.alpha(kwargs["alpha"])

        actors = [msh]

        if "i" in kwargs:
            i = kwargs["i"]
            # c   = ['r', 'g', 'b', 'k', 'y', 'cyan']
            sph = vp.shapes.Spheres(self.vertices[i], r=0.01)
            actors.append(sph)

        fig = None
        if "fig" not in kwargs:
            fig = vp.plotter.Plotter()
        else:
            fig = kwargs["fig"]

        fig.add(actors)
        fig.show()

    def vedo(self, c=None):
        import vedo as vp

        return vp.Mesh([self.vertices, self.faces], c=c)

    def from_vedo(self, vedo_mesh):
        self.__vertices = vedo_mesh.points().copy()
        self.__faces = np.asarray(vedo_mesh.faces())
        return self

    def write(self, fn, method="vedo"):
        ret = False
        if method == "igl":
            ret = igl.write_triangle_mesh(fn, self.vertices, self.faces)
        else:
            self.vedo().write(fn)
            ret = True
        return ret

    """ ================================================ """
    """                  End of Class                    """
    """ ================================================ """


def readConfig(dir):
    config: dict
    fn = os.path.join(dir, "config.json")
    with open(fn) as file:
        config = json.load(file)
    return config


class mesh_loader:
    def __init__(self, dir, k=-1, type=""):
        self.__dir = dir
        self.__k = k
        self.__type = type

    def __call__(self, fn, normalise=False):
        fmsh = os.path.join(self.__dir, "meshes", fn + ".ply")
        v, f = util.readMesh(fmsh, normalise=False)
        fgeo = os.path.join(self.__dir, "geodesic_matrices", fn + ".npy")
        geodesic_matrix = (
            np.load(fgeo) if os.path.exists(fgeo) else np.array([])
        )
        mesh = Mesh(
            vertices=v,
            faces=f,
            geodesic_matrix=geodesic_matrix,
            num_eigenvectors=self.__k,
            type=self.__type,
        )

        feig = os.path.join(self.__dir, "eigen", fn + ".npz")
        if os.path.exists(feig):
            eigen = np.load(feig)
            mesh.eigen = [eigen[key] for key in ["evals", "evecs"]]

        fsig = os.path.join(self.__dir, "signatures", fn + ".npy")
        if os.path.exists(fsig):
            mesh.scalars["signatures"] = np.load(fsig)

        mesh.name = fn
        if normalise:
            mesh.scalars["area"] = mesh.area()
            mesh.centre()
            mesh.normalise()

        return mesh
