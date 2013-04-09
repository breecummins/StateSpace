import StateSpaceReconstruction as SSR
import numpy as np
import bottleneck as bn

# All three methods for finding the closest N points appear to be mostly equivalent in performance for arrays of 10000 by 3. The method using argsort will suffer the most as the array increases in length.

# def findClosest(poi,pts,N):
#     '''
#     Find the closest N points in pts (numpy array) to poi.

#     '''
#     dists = np.sqrt(((pts - poi)**2).sum(1))
#     m = np.max(dists)
#     out = []
#     for _ in range(N):
#         i = dists.argmin()
#         out.append((i,dists[i]))
#         dists[i] = 2*m
#     return out

# def findClosest(poi,pts,N):
#     '''
#     Find the closest N points in pts (numpy array) to poi.

#     '''
#     dists = np.sqrt(((pts - poi)**2).sum(1))
#     inds = dists.argsort()[:N]
#     return [(i,dists[i]) for i in inds]

def findClosest(poi,pts,N):
    '''
    Find the closest N points in pts (numpy array) to poi.

    '''
    dists = np.sqrt(((pts - poi)**2).sum(1))
    inds = bn.argpartsort(dists,N)[:N]
    return [(i,dists[i]) for i in inds]