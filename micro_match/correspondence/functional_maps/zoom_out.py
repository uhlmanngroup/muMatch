import numpy as np
from scipy.optimize import linear_sum_assignment

from ...tools import geometric_utilities as util
from . import functional_mapping as fm


class fast_assigment_functor:
    def __init__(self, src, dst, N_sampling=100):
        self.src = src
        self.dst = dst
        self.i_src = util.metric_sampling(src.g, N_sampling)
        self.i_dst = util.metric_sampling(dst.g, N_sampling)
        self.j_dst = dst.g[self.i_dst].argmin(axis=0)

    def reduce(self, scalar):
        shape = [self.i_dst.size] + list(scalar.shape)[1:]
        res = np.zeros(shape)
        np.add.at(res, self.j_dst, scalar)
        return res

    def __call__(self, C):
        P_dense = fm.soft_correspondence(self.src, self.dst, C)
        P_sparse = np.transpose(self.reduce(P_dense[self.i_src].T)).astype(
            np.float32
        )
        a, b = linear_sum_assignment(P_sparse, maximize=True)
        return self.i_src[a], self.i_dst[b]


class zoomout_refinement:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.kmax = np.array([src.k, dst.k])
        self.assignment = fast_assigment_functor(src, dst)

    def iteration(self, C, dk):
        ks, kd = C.shape
        ks += dk
        kd += dk
        i, j = self.assignment(C)
        x, y = self.src.dirac_deltas(i, ks), self.dst.dirac_deltas(j, kd)
        C = util.orthogonalProcrustes(y.T, x.T).T
        return C

    def __call__(self, C):
        while np.all(C.shape < self.kmax):
            C = self.iteration(C, dk=max(1, int(0.1 * C.shape[0])))
        return C
