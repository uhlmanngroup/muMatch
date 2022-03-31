## Standard Libraries ##
import sys
import os

## Numerical Libraries ##
import numpy as np
import scipy as sp

from ..tools import geometric_utilities as util
#from shot_descriptor import calculate_SHOT_descriptors

''' ======================================================================================================= '''
    ####                                      Mesh Feature Descriptors                                  ###
''' ======================================================================================================= '''


def generate_sequence(xmin, xmax, nsamples, sequence_type = 'log_linear'):
    x   = None
    if (sequence_type == 'log_linear'):
        x   =  np.linspace(np.log(xmin), np.log(xmax)/1.02, nsamples)
        s   =  7*(x[1]-x[0])
        res =  [x,s]
    elif (sequence_type == 'log_sampled'):
        tmin,tmax = (util.safe_inverse(_x_) for _x_ in (xmax,xmin))
        x   =  np.exp(np.linspace(np.log(tmin), np.log(tmax), nsamples))
        res = x
    else:
        raise Exception('generate sequence: invalid type')
    return res


def gaussian_descriptor(mesh, num):
    k = util.gaussianCurvature(mesh.v, mesh.f)
    mus = [10.0**(-t) for t in np.linspace(1,9, num=num, endpoint=True)]
    ks = [util.laplacianSmoothing(mesh.l, mesh.mass, k, mu=mu) for mu in mus]
    return np.stack(ks, axis=-1)



def wave_kernel_signature(evecs, evals, eps, sigma):
    '''
    The Wave Kernel Signature: A Quantum Mechanical Approach To Shape Analysis
    '''
    dE           =  util.differenceMatrix(eps,np.log(evals))
    gauss_kernel =  np.exp(-0.5*(dE/sigma)**2)
    signatures   =  (evecs**2).dot(gauss_kernel)
    signatures  /=  np.sum(gauss_kernel, axis = 0, keepdims  = True) + 1e-6
    return signatures



def heat_kernel_signature(evecs, evals, T):
    '''
    Computes the heat kernel signature according to the spectrum of a graph operator (e.g., Laplacian).
    '''
    omega           =  np.outer(evals,T)
    low_pass_filter =  np.exp(-omega)
    signatures      =  (evecs**2).dot(low_pass_filter)
    signatures     /=  np.sum(low_pass_filter, axis = 0, keepdims = True) + 1e-6
    return signatures


class descriptor_class:
    def __init__(self, emin, emax, num_wks, num_hks, num_gaussian):
        eps,sigma  =  generate_sequence(emin, emax, num_wks, sequence_type = 'log_linear')
        T  =  generate_sequence(emin, emax, num_hks, sequence_type = 'log_sampled')
        self.T  =  T
        self.eps  =  eps
        self.sigma =  sigma
        self.num_gaussian = num_gaussian

    def __call__(self, mesh):
        evals,evecs = mesh.eigen
        kap = gaussian_descriptor(mesh, self.num_gaussian)
        wks = wave_kernel_signature(evecs[:,1:], evals[1:], self.eps, self.sigma)
        hks = heat_kernel_signature(evecs[:,1:], evals[1:], self.T)
        sigs = np.concatenate([wks, hks, kap], axis = -1)
        sigs /= np.sqrt(np.sum(mesh.mass @ sigs**2, axis = 0, keepdims = True)) + 1e-6
        return sigs



''' ======================================================================================================= '''
    ####                                          End                                                   ###
''' ======================================================================================================= '''


if __name__ == '__main__':
    pass
