#!/usr/bin/sh python

import numpy as np
import rk4
import matplotlib.pyplot as plt

def complotkavolterra4D(t,x,a=0,b=0,r=1,p=1,q=1):
    #Baigent notes
    rates=np.array([-1,1,-1,1])
    interactionmatrix=np.array([[0,r,0,0],[-r,0,a,0],[0,-b,0,p],[0,0,-q,0]])
    return x*(rates + np.sum(interactionmatrix*x,1))

def solvePredPrey(init,finaltime,dt=0.025,odes=predprey,kwargs={'A':1,'B':1,'C':1,'D':1}):
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

ts = solvePredPrey([0.2,0.25,0.15,0.18],400,dt=0.025,odes=complotkavolterra4D,kwargs={'a':0.4,'b':0.05,'p':np.sqrt(2),'q':np.sqrt(2)/2.})
figstogether(ts[:,0],ts[:,1])
figstogether(ts[:,2],ts[:,3])
figsapart(ts)

