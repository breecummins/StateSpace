import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def makeShadowManifold(timeseries, numlags):
    '''
    timeseries is a sequence of observations.
    numlags is a integer indicating the dimension of the shadow manifold
    to be constructed. It is required that numlags < len(timeseries).

    '''
    for t in range(len(timeseries)-numlags+1):
        yield timeseries[t,t+numlags]

def plotShadowManifold(timeseries, numlags):
    '''
    timeseries is a sequence of observations.
    numlags is a integer indicating the dimension of the shadow manifold
    to be constructed. It is required that numlags < len(timeseries) and
    numlags < 4.

    '''
    if numlags > 3:
        raise(SystemExit,'The manifold cannot be plotted because it is greater than dimension 3.')
    elif numlags == 3:
        pts = makeShadowManifold(timeseries,numlags)
        xyz = np.array(list(pts))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Axes3D.plot(xyz[:,0],xyz[:,1],xyz[:,2])
    elif numlags == 2:
        pts = makeShadowManifold(timeseries,numlags)
        xy = np.array(list(pts))
        fig = plt.figure()
        plt.plot(xy[:,0],xy[:,1])
    elif numlags == 1:
        fig = plt.figure()
        plt.plot(timeseries)



