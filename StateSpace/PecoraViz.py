import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})

import PecoraMethod as PM
import StateSpaceReconstruction as SSR


def lorenzTS(finaltime=80.0,dt=0.01):
    from LorenzEqns import solveLorenz
    timeseries = solveLorenz([1.0,0.5,0.5],finaltime,dt)
    eqns = 'Lorenz'
    names = ['x','y','z']
    return eqns,names,timeseries

def doublependulumTS(finaltime=2400.0,dt=0.025):
    from DoublePendulum import solvePendulum
    timeseries = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt)
    eqns = 'Double pendulum'
    names = ['x','y','z','w']
    return eqns,names,timeseries

def plotOutput(contconf,invcontconf,Nlist,title,mastereps):
    plt.figure(0)
    plt.plot(mastereps,contconf.transpose())
    plt.legend([str(N) for N in Nlist])
    plt.ylabel(r'\theta (C^0)')
    plt.xlabel(r'\epsilon')
    plt.title(title+' forward')
    plt.figure(1)
    plt.plot(mastereps,invcontconf.transpose())
    plt.legend([str(N) for N in Nlist])
    plt.ylabel(r'\theta (I^0)')
    plt.xlabel(r'\epsilon')
    plt.title(title+' inverse')
    plt.show()

def testLorenz(Nlistmaster,mastereps):
    eqns,names,ts = lorenzTS()
    compind1=0
    compind2=1
    numlags = 3 #num dims
    lagsize = 8
    # lagsize = SSR.numlagsFromFirstZeroOfAutocorrelation(ts[:,compind1])
    # print(lagsize)
    Mx = SSR.makeShadowManifold(ts[:,compind1], numlags, lagsize)
    My = SSR.makeShadowManifold(ts[:,compind2], numlags, lagsize)
    Nlist = (Nlistmaster*len(ts)).astype(int)
    contconf, invcontconf = PM.convergenceWithContinuityTest(Mx,My,Nlist,mastereps)
    title = eqns + ' M{0} -> M{1}'.format(compind1,compind2)
    plotOutput(contconf,invcontconf,Nlist,title,mastereps)

if __name__ == '__main__':
    Nlistmaster = np.arange(0.1,0.6,0.1)
    mastereps=np.array([0.005,0.01,0.02,0.05,0.1,0.2])
    testLorenz(Nlistmaster,mastereps)