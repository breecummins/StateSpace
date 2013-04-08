import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import StateSpaceReconstruction as SSR

def plotManifold(timeseries,show=1,hold=0,style='b-'):
    '''
    timeseries is a sequence of observations containing at most three columns

    '''
    if not hold:
        fig = plt.figure()
    else:
        plt.hold('on')
    s = timeseries.shape
    if len(s) == 1:
        plt.plot(timeseries,style)
    elif len(s) == 2 and s[1] == 2:
        plt.plot(timeseries[:,0],timeseries[:,1],style)
    elif len(s) == 2 and s[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(timeseries[:,0],timeseries[:,1],timeseries[:,2],style)
    else:
        raise(SystemExit,'A timeseries of dimension ' + str(timeseries.shape) + ' cannot be plotted')
    if show:
        plt.show()



def plotShadowManifold(timeseries, numlags, show=1,hold=0,style='b-'):
    '''
    timeseries is a sequence of observations.
    numlags is a integer indicating the dimension of the shadow manifold
    to be constructed. It is required that numlags < len(timeseries) and
    numlags < 4.

    '''
    if not hold:
        fig = plt.figure()
    else:
        plt.hold('on')
    if numlags > 3:
        raise(SystemExit,'The manifold cannot be plotted because it is greater than dimension 3.')
    elif numlags == 3:
        pts = SSR.makeShadowManifold(timeseries,numlags,style)
        xyz = np.array(list(pts))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xyz[:,0],xyz[:,1],xyz[:,2])
    elif numlags == 2:
        pts = SSR.makeShadowManifold(timeseries,numlags,style)
        xy = np.array(list(pts))
        plt.plot(xy[:,0],xy[:,1])
    elif numlags == 1:
        plt.plot(timeseries,style)
    if show:
        plt.show()



