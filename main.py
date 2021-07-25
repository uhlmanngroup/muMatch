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
import meshcorr.functional_maps.zoom_out as zo
import meshcorr.tools.geometric_utilities as util
import meshcorr.deep_functional_maps.predictor as dfm
import meshcorr.functional_maps.functional_mapping as fm
import meshcorr.functional_maps.feature_descriptors as fd
import meshcorr.product_manifold_filters.product_manifold_filter as pmf

from meshcorr.tools.mesh_class import Mesh
from meshcorr.tools.mapping_viewer import renderCorrespondence_vtk
from meshcorr.product_manifold_filters.assignment_functors import elastic_assignment_functor

''' ======================================================================================================= '''
    ####                                           Main                                                 ###
''' ======================================================================================================= '''



def readConfig(dir, type):
    config: dict
    fn = os.path.join(dir, 'configs', type + '.json')
    with open(fn) as file:
        config = json.load(file)
    config['type'] = type
    return config


class loader_class:
    def __init__(self, dir, config):
        args =  [np.array(config[i]) for i in ['emin', 'emax','num_wks', 'radius']]
        self.config = config
        self.descriptor = fd.descriptor_class(*args)
        self.msh = os.path.join(dir, 'meshes')
        self.dsc = os.path.join(dir, 'descriptors')
        self.geo = os.path.join(dir, 'geodesic_matrices')
        self.eig = [os.path.join(dir,f) for f in ['eigenvalues', 'eigenvectors']]
        self.par = os.path.join(dir, 'partitions')

    def __call__(self, fn):
        v,f,_ = util.readMesh(self.msh + '/' + fn + '.ply', True)
        g = np.load(self.geo + '/' + fn + '.npy')
        g/= np.mean(g)
        mesh = Mesh(v,f, g=g, k=self.config['num_eigenfunctions'], type=self.config['type'])
        mesh.eigen = [np.load(dir + '/' + fn + '.npy') for dir in self.eig]
        #symm = self.descriptor(mesh)
        #mesh.scalars['partitions'] = np.load(self.par + '/' + fn + '.npy')[..., np.newaxis]
        mesh.scalars['signatures'] = np.load(self.dsc + '/' + fn + '.npy')
        return mesh



#def ideal_match(fn1, fn2):
#    a    =  np.load(os.path.join('/home/jamesklatzow/Documents/EBI/Preprocess/data/TOSCA/indices', fn1 + '.npy'))
#    b    =  np.load(os.path.join('/home/jamesklatzow/Documents/EBI/Preprocess/data/TOSCA/indices', fn2 + '.npy'))
#    v,_  =  igl.read_triangle_mesh(os.path.join('/home/jamesklatzow/Documents/EBI/Datasets/TOSCA_High_res/meshes', fn2 + '.obj'))
#    j    =  np.argmin(util.differenceMatrix(v[a],v[b]), axis = 0)
#    return j

def readLandmarks(fn):
    fin = os.path.join("/home/jamesklatzow/Documents/EBI/Code/analysis/teeth/landmarks", fn + ".pts")
    data = []
    with open(fin,'r') as file:
        for line in file:
            data.append(line.split())
    return np.array(data[2:-1])[:,1:].astype(float)


def check_orientation(src, dst, j):
    n1,n2 = [x.normals() for x in [src,dst]]
    pass


if __name__ == '__main__':

    import pyshot
    from scipy import linalg
    import copy

    dir_in = "/home/jamesklatzow/Documents/EBI/Preprocess/data/teeth"
    pairs = np.load(os.path.join(dir_in, "pairs.npy"))
    config =  readConfig(dir_in, 'teeth')
    loader = loader_class(dir_in, config)
    files = [fn[:-4] for fn in os.listdir(os.path.join(dir_in, "meshes"))]
    N = len(files)
    #args =  [np.array(config[i]) for i in ['emin', 'emax','num_wks', 'radius']]
    #descriptor = fd.descriptor_class(*args)
    #print(pairs.shape)
    res = []
    fig = vp.plotter.Plotter()
    for f1,f2 in pairs:
    #for m in range(N):
    #    for n in range(m+1,N):
        try:
            #f1 = files[m] #"cat" + str(m) #
            #f2 = files[n] #"cat" + str(n) #
            src = loader(f1)
            dst = loader(f2)

            fn1,fn2 = f1,f2
            if (dst.N() < src.N()):
                fn1,fn2 = f2,f1
                src,dst = dst,src

            C = fm.correspondenceMatrixSolver(src, dst, k = 18, alpha = 1, beta = 2)
            C = zo.zoomout_refinement(src, dst)(C)
            P = fm.soft_correspondence(src, dst, C)
            assign = lambda x: linear_sum_assignment(x, maximize=True)
            i,j = pmf.product_manifold_filter_assignment(assign, src.g, dst.g, P, sigma = .5, gamma = .65, iterations = 2)
            dg = np.linalg.norm(src.g[i][:,i] - dst.g[j][:,j], ord = 'fro')
            print(fn1 + "->" + fn2 + ": {0:.2f}".format(dg))
            res.append( dg )
            #res = dst[j].copy()
            #res.f = src.f
            #j = np.arange(src.N())
            #renderCorrespondence_vtk(src.vedo(), res.vedo(), j)
            fout = os.path.join("/home/jamesklatzow/Documents/EBI/Code/analysis/teeth/results", fn1 + "_" + fn2 + ".npy")
            np.save(fout, np.stack([i,j],axis=0))
        except:
            print("problem")

    print(np.mean(res))

    '''
    p1 = src.scalars['partitions'].squeeze()
    p2 = src.scalars['partitions'].squeeze()
    x = np.logical_and(p1[i] < 0, p2[j] < 0)
    y = np.logical_and(p1[i] < 0, p2[j] > 0)
    g1 = np.sum(src.g[y][:, p1[i] > 0].min(axis=-1))
    g2 = np.sum(src.g[x][:, p1[i] > 0].min(axis=-1))
    dg = min(g1,g2)

    dg = 100 * np.abs(src.g[i][:,i] - dst.g[j][:,j]) /src.g.max()
    #error.append(dg)
    print("{0:.3f}".format(dg.mean()))

    res = dst[j].copy()
    res.f = src.f
    j = np.arange(src.N())
    renderCorrespondence_vtk(src.vedo(), res.vedo(), j)

    print("\n" + 30*"=" + "\n")
    '''

    #assign = lambda x: linear_sum_assignment(x, maximize=True)
    #i,j = pmf.product_manifold_filter_assignment(assign, src.g, dst.g, P, sigma = .5, gamma = .65, iterations = 2)
    #res = dst[j].copy()
    #res.f = src.f
    #j = np.arange(src.N())
    #renderCorrespondence_vtk(src.vedo(), res.vedo(), j)

    '''
    l = readLandmarks(f1)
    src.subdivide(N=2, method=0)
    dv = util.differenceMatrix(l,src.points())
    idx = dv.argmin(axis=0)
    #dst = loader(f2)
    N = src.N()
    v = src.points()
    fig = vp.plotter.Plotter()
    i,j = 8084,8210
    x = src.geodesic(i,j).points()
    x = vp.shapes.Line(x, c='b', lw = 8)
    y = vp.shapes.Spheres(v[[i,j]], c='r', r = 0.008)
    fig = vp.plotter.Plotter()
    fig.add([src, x, y])
    fig.show()
    '''

    '''
    S1 = descriptor(src)
    S2 = descriptor(dst)
    src,dst = [t.vedo() for t in (src,dst)]
    n = 0
    for s1,s2 in zip(S1.T, S2.T):
        print(n)
        src.addPointArray(s1,'s1')
        dst.addPointArray(s2,'s2')
        fig.add([dst,src])
        fig.show()
        fig.clear()
        n += 1

    S1 = src.eigen[-1]
    S2 = dst.eigen[-1]
    print(S1[:,0].mean(), S2[:,0].mean())
    src = src.vedo()
    dst = dst.shift(1,0,0).vedo()
    n = 0
    for s1,s2 in zip(S1.T, S2.T):
        print(n)
        src.addPointArray(s1,'s1')
        dst.addPointArray(s2,'s2')
        fig.add([dst,src])
        src.addScalarBar()
        #dst.addScalarBar()
        fig.show()
        fig.clear()
        n += 1

    C = np.zeros((10,10))
    I = np.eye(10)
    gx = spectral_array(src, src.g, 10)
    gy = spectral_array(dst, dst.g, 10)
    np.fill_diagonal(gx,0)
    np.fill_diagonal(gy,0)

    plt.imshow(gx)
    plt.figure()
    plt.imshow(gy)
    plt.show()

    dA = np.sum(((C @ gx) - (gy @ C))**2) #(C @ p @ C.T) - s
    dB = np.sum(((I @ gx) - (gy @ I))**2) #(C @ q @ C.T) - t

    print(dB , dA)
    '''

    '''
    C = zo.zoomout_refinement(src, dst)(C)
    plt.imshow(C)
    plt.show()

    P = fm.soft_correspondence(src, dst, C)

    i,j = pmf.product_manifold_filter_assignment(assign, src.g, dst.g, P, sigma = .2, gamma = .6, iterations = 2)

    res = dst[j].copy()
    res.f = src.f
    j = np.arange(src.N())
    renderCorrespondence_vtk(src.vedo(), res.vedo(), j)


    dir_in = "/home/jamesklatzow/Documents/EBI/Preprocess/data/TOSCA"
    config =  readConfig(dir_in, 'tosca')
    loader = loader_class(dir_in, config)
    src = loader("cat0").vedo()

    N = src.N()
    v = src.points()

    fig = vp.plotter.Plotter()
    for i in range(0,N,50):
        print(i)
        sphere = vp.shapes.Sphere(v[i], c = 'r', r = 0.02)
        fig.add([src, sphere])
        fig.show()
        fig.clear()

    i,j = 2050,2750
    x = src.geodesic(i,j).points()
    x = vp.shapes.Line(x, c='b', lw = 5)
    y = vp.shapes.Spheres(v[[i,j]], c='r', r = 0.02)

    fig = vp.plotter.Plotter()
    fig.add([src, x, y])
    fig.show()

    t = 2*np.pi*np.linspace(0,1,5,endpoint=False)
    v = np.zeros((8,3), dtype = np.float32)
    v[1:6,0] = np.cos(t)
    v[1:6,1] = np.sin(t)
    v[0,2] = .2
    v[6:] = np.array([[1.8,0,0],[1.5,0.5,0]])

    f = np.array([  [0,1,2] , [0,2,3], [0,3,4] , [0,4,5] , [0,5,1] , [1,6,7]])
    i,j = util.extractEdges(f).T

    x = vp.Mesh([v,f])
    y = vp.shapes.Spheres(v, r = 0.05, c = 'r')
    z = vp.shapes.Lines(v[i], v[j], c = 'k', lw = 2)

    fig = vp.plotter.Plotter()
    fig.add([x,y,z])
    fig.show()

    dir_in = "/home/jamesklatzow/Documents/EBI/Preprocess/data/limbs"
    raw_dir = '/home/jamesklatzow/Documents/EBI/Datasets/limb_source_data/Limbs/'
    #target_dir = '/home/jamesklatzow/Documents/EBI/Datasets/limb_source_data/Limbs/Late/Forelimb/WT'
    #dir_out = "/home/jamesklatzow/Documents/EBI/Code/analysis/new_embryos"

    config =  readConfig(dir_in, 'limbs')
    #func = dfm.dfmPredictorwithOptimisation(600, 'early_fore_wt_limbs', 1e-4, 50)
    #func = dfm.dfmPredictor(600, 'limbs')
    loader = loader_class(dir_in, config)

    #files = [f.split('.npy')[0] for f in os.listdir(dir_in + '/partitions')]
    #targs = [f.split('.stl')[0] for f in os.listdir(target_dir)]
    #files = [x for x in files if x in targs]
    #files = [x for x in files if x.split('_')[-1][0] == 'L']

    stage, position = 'Late','Fore' #'Early', 'Mid',

    files = [f.split('.npy')[0] for f in os.listdir(dir_in + '/partitions')]

    targs = []
    for var in ['WT', 'MUT']:
        target_dir = os.path.join(raw_dir, stage, position + 'limb', var)
        targs += [f.split('.stl')[0] for f in os.listdir(target_dir)]

    files = [x for x in files if x in targs]
    '''

    '''
    f1 = files[np.argmin([loader(fn).N() for fn in files])]
    src = loader(f1)

    for f2 in files:
        dst = loader(f2)

        if (dst.N() < src.N()):
            print("swapping")
            src,dst = dst,src
            f1,f2 = f2,f1

        fout = os.path.join(dir_in,'correspondences', f1 + '_' + f2 + '.npy')
        if f1==f2: #os.path.exists(fout) or
            print("skipping")
            continue

        print(f1, f2, src.N(), dst.N())
        C = fm.estimate_correspondence_matrix(src, dst, alpha=1) #func(src,dst) # fm.estimate_correspondence_matrix(src, dst, alpha=1) #
        P = fm.soft_correspondence(src, dst, C)
        assign = lambda x: linear_sum_assignment(x, maximize=True)
        i,j = pmf.product_manifold_filter_assignment(assign, src.g, dst.g, P, sigma = .5, gamma = .65, iterations = 2)
        #np.save(fout, np.stack([i,j], axis=0))
        res = dst[j].copy()
        res.f = src.f
        j = np.arange(src.N())
        renderCorrespondence_vtk(src.vedo(), res.vedo(), j)

    files = os.listdir(os.path.join(dir_in,'correspondences'))
    for n,f in enumerate(files[11:]):
        fn = f.split('.npy')[0].split("_")
        f1 = "_".join(fn[:5])
        f2 = "_".join(fn[5:])
        print(n+11, f1, f2)
        src = loader(f1)
        dst = loader(f2)
        i,j = np.load(os.path.join(dir_in, 'correspondences', f))
        res = dst[j].copy()
        res.f = src.f
        j = np.arange(src.N())
        renderCorrespondence_vtk(src.vedo(), res.vedo(), j)

    '''
