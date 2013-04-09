import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import StateSpaceReconstruction as SSR

def plotManifold(timeseries,show=1,hold=0,style='b-'):
    '''
    timeseries is a sequence of observations containing at most three columns

    '''
    s = timeseries.shape
    if not hold:
        fig = plt.figure()
        if len(s) == 2 and s[1] == 3:
            ax = fig.add_subplot(111, projection='3d')
    else:
        plt.hold('on')
        ax = plt.gca()
    if len(s) == 1:
        plt.plot(timeseries,style)
    elif len(s) == 2 and s[1] == 2:
        plt.plot(timeseries[:,0],timeseries[:,1],style)
    elif len(s) == 2 and s[1] == 3:
        ax.plot(timeseries[:,0],timeseries[:,1],timeseries[:,2],style)
    else:
        raise(SystemExit,'A timeseries of dimension ' + str(timeseries.shape) + ' cannot be plotted')
    if show:
        plt.show()



def plotShadowManifold(timeseries, numlags, lagsize, show=1,hold=0,style='b-'):
    '''
    timeseries is a sequence of observations.
    numlags is a integer indicating the dimension of the shadow manifold
    to be constructed. It is required that numlags < 4 for plotting.
    lagsize is the number of time points in each lag.

    '''
    if not hold:
        fig = plt.figure()
        if numlags == 3:
            ax = fig.add_subplot(111, projection='3d')
    else:
        plt.hold('on')
        ax = plt.gca()
    if numlags > 3:
        raise(SystemExit,'The manifold cannot be plotted because it is greater than dimension 3.')
    elif numlags == 3:
        pts = SSR.makeShadowManifold(timeseries,numlags,lagsize)
        xyz = np.array(list(pts))
        ax.plot(xyz[:,0],xyz[:,1],xyz[:,2],style)
    elif numlags == 2:
        pts = SSR.makeShadowManifold(timeseries,numlags,lagsize)
        xy = np.array(list(pts))
        plt.plot(xy[:,0],xy[:,1],style)
    elif numlags == 1:
        plt.plot(timeseries,style)
    if show:
        plt.show()



