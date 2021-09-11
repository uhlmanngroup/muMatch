# first line: 93
@memory.cache
def icp_alignement(point_clouds, iterations = 5, idx = None):
    if not idx:
        idx = np.argmin([x.shape[0] for x in point_clouds])
    average = point_clouds[idx].copy()
    for _ in range(iterations):
        icp = ICP(average)
        aligned = [icp(pts) for pts in point_clouds]
        func = lambda x: nearest_neighbours(average, x, quadratic = True)
        aligned = Parallel(n_jobs=-1)(delayed(func)(pts) for pts in aligned)
        average = np.mean(aligned, axis=0)
    return idx, np.stack(aligned, axis = 0)
