#!/usr/bin/sh python

import numpy as np
import rk4
import matplotlib.pyplot as plt
import StateSpaceReconstruction as SSR
import StateSpaceReconstructionPlots as SSRPlots
import PecoraMethodModified as PM
import PecoraViz as PV

def complotkavolterra4D(t,x,a=0,b=0,r=1,s=1,p=1,q=1):
    #Baigent notes
    rates=np.array([-1,1,-1,1])
    interactionmatrix=np.array([[0,r,0,0],[-s,0,a,0],[0,-b,0,p],[0,0,-q,0]])
    return x*(rates + np.sum(interactionmatrix*x,1))

def complotkavolterra2D(t,x,r=1,s=1):
    #Baigent notes
    rates=np.array([-1,1])
    interactionmatrix=np.array([[0,r],[-s,0]])
    return x*(rates + np.sum(interactionmatrix*x,1))

def solvePredPrey(init,finaltime,dt=0.025,odes=complotkavolterra4D,kwargs={'A':1,'B':1,'C':1,'D':1}):
    '''
    init = [x[0],y[0]] are the initial conditions
    finaltime is length of simulation, dt is the time step
    odes is the function handle to be integrated
    kwargs dict contains parameters for odes function
    '''
    times = np.arange(0,finaltime,dt)
    timeseries = np.zeros((len(times),len(init)))
    timeseries[0,:] = init
    for k,ti in enumerate(times[:-1]):
        timeseries[k+1,:] = rk4.solverp(ti,timeseries[k,:],dt,odes,**kwargs)
    return timeseries

def figstogether(x,y):
    plt.figure()
    plt.plot(x,y,'b-')
    plt.show()

def figsapart(x):
    plt.figure()
    for c in range(x.shape[1]):
        plt.plot(x[:,c])
        plt.hold('on')
    plt.show()

def run2D():
    ts = solvePredPrey([0.21,0.19],400,odes=complotkavolterra2D,kwargs={'r':np.sqrt(3),'s':np.sqrt(3)/3.})
    lags=[63,63]
    dim=2
    tsprops=np.arange(0.4,1.1,0.2)
    epsprops=np.array([0.001, 0.002, 0.005,0.01,0.02,0.05])
    forwardconf, inverseconf,epslist1,epslist2,forwardprobs,inverseprobs = PM.convergenceWithContinuityTestFixedLagsFixedEps(ts[:,0],ts[:,1],dim,lags[0],lags[1],tsprops=tsprops,epsprops=epsprops)
    PV.plotOutput(forwardconf,inverseconf,epsprops,tsprops,'Mx1 -> Mx2','Mx2 -> Mx1',logs = [0,0])

def run4D():
    ts = solvePredPrey([0.21,0.19,0.15,0.18],400,kwargs={'a':0.4,'b':0.05,'r':np.sqrt(3),'s':np.sqrt(3)/3.,'p':np.sqrt(2),'q':np.sqrt(2)/4.})
    lags=[20,20,60,60]
    dim=4
    tsprops=np.arange(0.4,1.1,0.2)
    epspropslo=np.array([0.001, 0.002, 0.005,0.01,0.02,0.05])
    epspropshi=np.array([0.005,0.01,0.02,0.05,0.1,0.2])
    c1=1
    c2=2
    epsprops=epspropshi
    forwardconf, inverseconf,epslist1,epslist2,forwardprobs,inverseprobs = PM.convergenceWithContinuityTestFixedLagsFixedEps(ts[:,c1],ts[:,c2],dim,lags[c1],lags[c2],tsprops=tsprops,epsprops=epsprops)
    PV.plotOutput(forwardconf,inverseconf,epsprops,tsprops,'Mx'+str(c1+1)+' -> Mx'+str(c2+1),'Mx'+str(c2+1)+' -> Mx'+str(c1+1),logs = [0,0])

def lag4D():
    ts = solvePredPrey([0.21,0.19,0.15,0.18],400,kwargs={'a':0.4,'b':0.05,'r':np.sqrt(3),'s':np.sqrt(3)/3.,'p':np.sqrt(2),'q':np.sqrt(2)/4.})
    Mlens = [int(p*ts.shape[0]) for p in np.arange(0.1,1.1,0.1)]
    SSR.chooseLags(ts,Mlens,Tp=None)

def viz4D():
    ts = solvePredPrey([0.21,0.19,0.15,0.18],400,kwargs={'a':0.4,'b':0.05,'r':np.sqrt(3),'s':np.sqrt(3)/3.,'p':np.sqrt(2),'q':np.sqrt(2)/4.})
    figstogether(ts[:,0],ts[:,1])
    figstogether(ts[:,2],ts[:,3])
    figsapart(ts)
    lags=[20,20,60,60]
    dim=4
    for c in range(ts.shape[1]):
        SSRPlots.plotShadowManifold(ts[:,c], dim, lags[c], show=1,hold=0,style='b-',titlestr='x'+str(c+1),scatter=False, color=None,smooth=1,axisequal=True)

# lag4D()
# viz4D()
run4D()
