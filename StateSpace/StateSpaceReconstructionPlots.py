#third party modules
import numpy as np
import matplotlib as mpl
mpl.use('Pdf') #comment out if you want to see figures at run time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#my modules
import StateSpaceReconstruction as SSR
import CCM, CCMAlternatives,Weights

mpl.rcParams.update({'font.size': 22})

def plotManifold(timeseries,show=1,hold=0,style='b-',titlestr=None):
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
    if titlestr != None:  
        plt.title(titlestr)
    if show:
        plt.show()



def plotShadowManifold(timeseries, numlags, lagsize, show=1,hold=0,style='b-'):
    '''
    timeseries is a sequence of observations.
    numlags is a integer indicating the dimension of the shadow manifold
    to be constructed. It is required that numlags < 4 for plotting.
    lagsize is the number of time points in each lag.

    '''
    pts = SSR.makeShadowManifold(timeseries,numlags,lagsize)
    plotManifold(pts,show,hold,style)

def plotEstShadowManifoldSugihara(ts1,ts2,numlags,lagsize,wgtfunc=Weights.makeExpWeights):
    est1,est2 = CCM.crossMap(ts1,ts2,numlags,lagsize,wgtfunc)
    plotShadowManifold(ts1,numlags,lagsize,0)
    plotShadowManifold(est1,numlags,lagsize,0,1,'r-')
    plotShadowManifold(ts2,numlags,lagsize,0,0,'k-')
    plotShadowManifold(est2,numlags,lagsize,1,1,'g-')

def plotEstShadowManifoldUs1(ts1,ts2,numlags,lagsize,wgtfunc=Weights.makeExpWeights):
    M1 = SSR.makeShadowManifold(ts1,numlags,lagsize)
    M2 = SSR.makeShadowManifold(ts2,numlags,lagsize)
    Mest1,Mest2 = CCMAlternatives.crossMapModified1(M1,M2,wgtfunc)
    plotManifold(M1,0)
    plotManifold(Mest1,0,1,'r-')
    plotManifold(M2,0,0,'k-')
    plotManifold(Mest2,1,1,'g-')

def plotEstShadowManifoldUs2(ts1,ts2,numlags,lagsize,wgtfunc=Weights.makeExpWeights):
    M1 = SSR.makeShadowManifold(ts1,numlags,lagsize)
    M2 = SSR.makeShadowManifold(ts2,numlags,lagsize)
    est1,est2 = CCMAlternatives.crossMapModified2(M1,M2,wgtfunc)
    plotShadowManifold(ts1,numlags,lagsize,0)
    plotShadowManifold(est1,numlags,lagsize,0,1,'r-')
    plotShadowManifold(ts2,numlags,lagsize,0,0,'k-')
    plotShadowManifold(est2,numlags,lagsize,1,1,'g-')

def plotEstShadowManifoldUs3(ts1,ts2,numlags,lagsize,proj,wgtfunc=Weights.makeExpWeights):
    M1 = SSR.makeShadowManifold(ts1,numlags,lagsize)
    M2 = SSR.makeShadowManifold(ts2,numlags,lagsize)
    est1,est2 = CCMAlternatives.crossMapModified3(M1,M2,proj,wgtfunc)
    plotShadowManifold(ts1,numlags,lagsize,0)
    plotShadowManifold(est1,numlags,lagsize,0,1,'r-')
    plotShadowManifold(ts2,numlags,lagsize,0,0,'k-')
    plotShadowManifold(est2,numlags,lagsize,1,1,'g-')

def plots(x,y,show=1,hold=0,stylestr=['b-'],leglabels=None,legloc=4,titlestr=None,xstr=None,ystr=None,fname=None):
    if not hold:
        fig = plt.figure()
    else:
        plt.hold('on')
    if len(y.shape) == 2 and len(x.shape)==1:
        for k in range(y.shape[1]):
            if leglabels != None:
                plt.plot(x,y[:,k],stylestr[k],linewidth=2.0,label=leglabels[k])
            else:
                plt.plot(x,y[:,k],stylestr[k],linewidth=2.0)
    elif len(y.shape) == 2 and len(x.shape)==2:
        for k in range(y.shape[1]):
            if leglabels != None:
                plt.plot(x[:,k],y[:,k],stylestr[k],linewidth=2.0,label=leglabels[k])
            else:
                plt.plot(x[:,k],y[:,k],stylestr[k],linewidth=2.0)
    else:
        if leglabels != None:
            plt.plot(x,y,stylestr[0],linewidth=2.0,label=leglabels[0]) 
        else:
            plt.plot(x,y,stylestr[0],linewidth=2.0) 
    if titlestr != None:  
        plt.title(titlestr)
    if xstr != None: 
        plt.xlabel(xstr)
    if ystr != None:
        plt.ylabel(ystr)
    if leglabels != None:
        plt.legend(loc=legloc)
    mpl.rc('font',size=22)
    if fname != None:
        plt.savefig(fname,format='pdf', bbox_inches="tight")
    if show:
        plt.show()

