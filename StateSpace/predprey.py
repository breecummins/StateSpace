#!/usr/bin/sh python

import numpy as np
import rk4
import matplotlib.pyplot as plt

# def predprey(t,x,A=1,B=1,C=1,D=1):
#     dx = np.zeros(x.shape)
#     dx[0] = x[0]*(A-x[0])-B*x[1]
#     dx[1] = x[1]*(C-x[1])+D*x[0]
#     return dx

# def lotkavolterra(t,x,mu=1):
#     dx = np.zeros(x.shape)
#     dx[0] = x[0]*(1-x[1])
#     dx[1] = mu*x[1]*(x[0]-1)
#     return dx

# def logisticlotkavolterra(t,x,A=1,B=1,C=1,D=1):
#     dx = np.zeros(x.shape)
#     dx[0] = x[0]*(A-A*x[0]-B*x[1])
#     dx[1] = x[1]*(C-C*x[1]+D*x[0])
#     return dx

# def sugiharamap(t,x,A=3.8,B=0.02,C=3.5,D=0.1):
#     dx = np.zeros(x.shape)
#     dx[0] = x[0]*(A-A*x[0]-B*x[1])
#     dx[1] = x[1]*(C-C*x[1]-D*x[0])
#     return dx

# def twopredoneprey(t,x,A=1,B=1,C=1,D=1,E=1,F=1,G=1,H=1,I=1):
#     dx = np.zeros(x.shape)
#     dx[0] = x[0]*(A-A*x[0]-B*x[1]-E*x[2])
#     dx[1] = x[1]*(C-C*x[1]+D*x[0]-F*x[2])
#     dx[2] = x[2]*(G-G*x[1]+H*x[0]-I*x[1])
#     return dx

# def vanocompetitivelk(t,x,rates=np.array([1,0.72,1.53,1.27]),interactionmatrix=np.array([[1,1.09,1.52,0],[0,1,0.44,1.36],[2.33,0,1,0.47],[1.21,0.51,0.35,1]])):
#     #Vano et al (2006), Chaos in low-dimensional Lotka-Volterra models of competition
#     return rates*x*(np.ones(x.shape) - np.sum(interactionmatrix*x,1))

# def complotkavolterra3D(t,x,rates=np.array([1,0.5,1.5]),interactionmatrix=np.array([[0,1.,-1.],[-1.,0,1],[1.,-1,0]])):
#     #Baigent notes
#     return x*(rates + np.sum(interactionmatrix*x,1))

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

# ts = solvePredPrey([0.6,0.2],1000,dt=0.1,odes=predprey,kwargs={'A':1,'B':0.1,'C':1,'D':0.1})
# figstogether(ts[:,0],ts[:,1])

# ts = solvePredPrey([0.6,0.2],1000,dt=0.1,odes=logisticlotkavolterra,kwargs={'A':1,'B':0.5,'C':1,'D':0.1})
# figstogether(ts[:,0],ts[:,1])

# ts = solvePredPrey([1,0.105,0.099],200,dt=0.025,odes=twopredoneprey,kwargs={'A':1,'B':0.5,'C':1,'D':0.1,'E':0.5,'F':0.105,'G':1,'H':0.1,'I':0.1})
# figsapart(ts)

# ts = solvePredPrey([1,0.105,0.099],200,dt=0.025,odes=twopredoneprey,kwargs={'A':2,'B':0.495,'C':1,'D':0.1,'E':0.5,'F':0.105,'G':1,'H':0.1,'I':0.1})
# figsapart(ts)

# ts = solvePredPrey([0.6,0.2],1000,dt=0.1,odes=lotkavolterra,kwargs={'mu':0.1})
# figstogether(ts[:,0],ts[:,1])

# ts = solvePredPrey([0.55,0.5],300,dt=1,odes=sugiharamap,kwargs={})
# figstogether(ts[:,0],ts[:,1])
# figsapart(ts)

# ts = solvePredPrey([1,0.9,1.1],100,dt=0.025,odes=complotkavolterra3D,kwargs={})
# figstogether(ts[:,0],ts[:,1])

