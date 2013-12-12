#third party modules
import numpy as np
import matplotlib as mpl
# mpl.use('Pdf') #comment out if you want to see figures at run time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#my modules
import StateSpaceReconstruction as SSR
import CCM, CCMAlternatives,Weights

mpl.rcParams.update({'font.size': 22})

def plotManifold(timeseries,show=1,hold=0,style='b-',titlestr=None,scatter=False,color=None,axisequal=True):
    '''
    timeseries is a sequence of observations containing at most three columns

    '''
    if style and not color:
        color = style[0]
    if scatter and style[1] == '-':
        style = style[0] + 'o'
    if scatter and not style:
        style = 'o' 
    s = timeseries.shape
    if not hold:
        fig = plt.figure()
        fig.patch.set_alpha(0.0)
        if len(s) == 2 and s[1] > 2:
            ax = fig.add_subplot(111, projection='3d')
            if axisequal:
                # Create cubic bounding box to simulate equal aspect ratio
                X = timeseries[:,0]
                Y = timeseries[:,1]
                Z = timeseries[:,2]
                max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
                Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
                Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
                Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
                # Comment or uncomment following both lines to test the fake bounding box:
                for xb, yb, zb in zip(Xb, Yb, Zb):
                   ax.plot([xb], [yb], [zb], 'w')
    else:
        plt.hold('on')
        ax = plt.gca()
    if len(s) == 1:
        if not scatter:
            plt.plot(timeseries,style,color=color)
        else:
            plt.scatter(timeseries)
        # plt.ylabel(r'$x_0$')
    elif len(s) == 2 and s[1] == 2:
        if not scatter:
            plt.plot(timeseries[:,0],timeseries[:,1],style,color=color)
        else:
            plt.scatter(timeseries[:,0],timeseries[:,1])
            plt.axis('off')
        # plt.xlabel(r'$x_0$')
        # plt.ylabel(r'$x_1$')
    elif len(s) == 2 and s[1] == 3:
        if not scatter:
            ax.plot(timeseries[:,0],timeseries[:,1],timeseries[:,2],style,color=color)
            # ax._axis3don = False
            # plt.hold('on')
            # ax.plot([0.5],[0.5],[0.1],'r.')
            # ax.set_zlim3d(0,2)
        else:
            ax.scatter(timeseries[:,0],timeseries[:,1],timeseries[:,2],style,color=color)
        ax.patch.set_alpha(0.0)
        # plt.xlabel(r'$x_0$')
        # plt.ylabel(r'$x_1$')
        # ax.set_zlabel(r'$x_2$')
    elif len(s) == 2 and s[1] == 4:
        norm4 = (timeseries[:,-1] - np.min(timeseries[:,-1])) / (np.max(timeseries[:,-1]) - np.min(timeseries[:,-1]))
        ax.scatter(timeseries[:,0],timeseries[:,1],timeseries[:,2],'-',c=mpl.cm.jet(norm4))
        ax._axis3don = False
        ax.patch.set_alpha(0.0)
    else:
        print('A timeseries of dimension ' + str(timeseries.shape) + ' cannot be plotted')
        raise(SystemExit)
    if titlestr != None:  
        plt.title(titlestr)
    if axisequal:   
        plt.axis('equal')
    if show:
        plt.show()



def plotShadowManifold(timeseries, numlags, lagsize, show=1,hold=0,style='b-',titlestr=None,scatter=False, color=None,smooth=1,axisequal=True):
    '''
    timeseries is a sequence of observations.
    numlags is a integer indicating the dimension of the shadow manifold
    to be constructed. It is required that numlags < 4 for plotting.
    lagsize is the number of time points in each lag.

    '''
    pts = SSR.makeShadowManifold(timeseries,numlags,lagsize,smooth)
    plotManifold(pts,show=show,hold=hold,style=style,titlestr=titlestr,scatter=scatter,color=color,axisequal=axisequal)

def plotEstShadowManifoldSugihara(ts1,ts2,numlags,lagsize,wgtfunc=Weights.makeExpWeights):
    est1,est2 = CCM.crossMap(ts1,ts2,numlags,lagsize,wgtfunc)
    plotShadowManifold(ts1,numlags,lagsize,0)
    plotShadowManifold(est1,numlags,lagsize,0,1,'r-')
    plotShadowManifold(ts2,numlags,lagsize,0,0,'k-')
    plotShadowManifold(est2,numlags,lagsize,1,1,'g-')

def plotEstShadowManifoldUs1(ts1,ts2,numlags,lagsize,wgtfunc=Weights.makeExpWeights,smooth=1):
    M1 = SSR.makeShadowManifold(ts1,numlags,lagsize,smooth)
    M2 = SSR.makeShadowManifold(ts2,numlags,lagsize,smooth)
    Mest1,Mest2 = CCMAlternatives.crossMapModified1(M1,M2,wgtfunc)
    plotManifold(M1,0)
    plotManifold(Mest1,0,1,'r-')
    plotManifold(M2,0,0,'k-')
    plotManifold(Mest2,1,1,'g-')

def plotEstShadowManifoldUs2(ts1,ts2,numlags,lagsize,wgtfunc=Weights.makeExpWeights,smooth=1):
    M1 = SSR.makeShadowManifold(ts1,numlags,lagsize,smooth)
    M2 = SSR.makeShadowManifold(ts2,numlags,lagsize,smooth)
    est1,est2 = CCMAlternatives.crossMapModified2(M1,M2,wgtfunc)
    plotShadowManifold(ts1,numlags,lagsize,0)
    plotShadowManifold(est1,numlags,lagsize,0,1,'r-')
    plotShadowManifold(ts2,numlags,lagsize,0,0,'k-')
    plotShadowManifold(est2,numlags,lagsize,1,1,'g-')

def plotEstShadowManifoldUs3(ts1,ts2,numlags,lagsize,proj,wgtfunc=Weights.makeExpWeights,smooth=1):
    M1 = SSR.makeShadowManifold(ts1,numlags,lagsize,smooth)
    M2 = SSR.makeShadowManifold(ts2,numlags,lagsize,smooth)
    est1,est2 = CCMAlternatives.crossMapModified3(M1,M2,proj,wgtfunc)
    plotShadowManifold(ts1,numlags,lagsize,0)
    plotShadowManifold(est1,numlags,lagsize,0,1,'r-')
    plotShadowManifold(ts2,numlags,lagsize,0,0,'k-')
    plotShadowManifold(est2,numlags,lagsize,1,1,'g-')

def plotAutocorrelation(autocorr,title):
    plt.figure()
    plt.plot([1,len(autocorr)+1],[0,0],'k')
    plt.hold('on')
    plt.plot(range(1,len(autocorr)+1),autocorr)
    plt.ylabel('autocorrelation')
    plt.xlabel('lag index')
    plt.title(title)
    plt.show()

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

