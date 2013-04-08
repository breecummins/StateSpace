import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import StateSpaceReconstruction as SSR

def plotManifold(timeseries):
    '''
    timeseries is a sequence of observations containing at most three columns

    '''
    s = timeseries.shape
    if len(s) == 1:
        fig = plt.figure()
        plt.plot(timeseries)
    elif len(s) == 2 and s[1] == 2:
        fig = plt.figure()
        plt.plot(timeseries[:,0],timeseries[:,1])
    elif len(s) == 2 and s[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(timeseries[:,0],timeseries[:,1],timeseries[:,2])
        plt.show() 
    else:
        raise(SystemExit,'A timeseries of dimension ' + str(timeseries.shape) + ' cannot be plotted')




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
        pts = SSR.makeShadowManifold(timeseries,numlags)
        xyz = np.array(list(pts))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xyz[:,0],xyz[:,1],xyz[:,2])
    elif numlags == 2:
        pts = SSR.makeShadowManifold(timeseries,numlags)
        xy = np.array(list(pts))
        fig = plt.figure()
        plt.plot(xy[:,0],xy[:,1])
    elif numlags == 1:
        fig = plt.figure()
        plt.plot(timeseries)



