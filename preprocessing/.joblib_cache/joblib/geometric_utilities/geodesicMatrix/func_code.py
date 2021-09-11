# first line: 244
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
