import numpy as np
import rk4

def solveRossler(init,T,dt=0.01,a=0.2,b=0.2,c=5.7):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,Rossler,a=a,b=b,c=c)
    return x

def Rossler(t,x,a=0.0,b=0.0,c=0.0):
    dx = np.zeros(x.shape)
    dx[0] = -x[1]-x[2]
    dx[1] = x[0] + a*x[1]
    dx[2] = b+ x[2]*(x[0]-c)
    return dx

def solveDiamond(init,T,dt=0.01,mu=4.0,beta=1.2,A=2.0,a=0.2,b=0.2,c=5.7,d=3.5,B=1.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,diamond,mu=mu,beta=beta,A=A,a=a,b=b,c=c,d=d,B=B)
    return x

def diamond(t,x,mu=0.0,beta=0.0,A=0.0,a=0.0,b=0.0,c=0.0,d=0.0,B=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = x[3]
    dx[3] = -x[3] - beta*np.sin(x[2]) + A*np.sin(x[0])
    dx[4] = -x[5] - x[6] + d*np.sin(x[1])
    dx[5] = x[4] + a*x[5]
    dx[6] = b + x[6]*(x[4] - c)
    dx[7] = np.sin(x[7]) + B*np.sin(x[6])*np.sin(x[2])
    return dx

def solveRosslerExp1(init,T,dt=0.01,a=0.2,b=0.2,c=5.7):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,RosslerExperiment1,a=a,b=b,c=c)
    return x

def RosslerExperiment1(t,x,a=0.0,b=0.0,c=0.0):
    dx = np.zeros(x.shape)
    dx[0] = -x[1]**3-x[2]**3
    dx[1] = x[0] + a*x[1]
    dx[2] = b + x[2]*(x[0]-c)
    return dx



if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    # # x = solveRossler([5,4,3],500)
    # # SSRPlots.plotManifold(x)
    # x = solveDiamond([1.0,2.0,3.0,2.0,5.0,4.0,3.0,3.0],600.0)
    # SSRPlots.plotManifold(x[:,4:7],show=0,titlestr='s,u,v')
    # SSRPlots.plotManifold(x[:,[0,1,4]],show=0,titlestr='x,y,s')
    # SSRPlots.plotManifold(x[:,[0,1,5]],show=0,titlestr='x,y,u')
    # SSRPlots.plotManifold(x[:,[0,1,6]],show=0,titlestr='x,y,v')
    # SSRPlots.plotManifold(x[:,[0,1,7]],show=0,titlestr='x,y,p')
    # SSRPlots.plotManifold(x[:,[2,6,7]],show=1,titlestr='z,v,p')
    #########################
    x = solveRossler([5.0,4.0,3.0],600.0)
    # SSRPlots.plotShadowManifold(x[:,2], 3, 60, show=0, titlestr='v, lag 60')
    SSRPlots.plotShadowManifold(x[:,2], 3, 30, show=0, titlestr='v, lag 30')
    SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=0, titlestr='u, lag 60')
    SSRPlots.plotShadowManifold(x[:,0], 3, 60, show=0, titlestr='s, lag 60')
    SSRPlots.plotManifold(x,show=1,titlestr='phase space')
    # #########################
    # x = solveRosslerExp1([5.0,4.0,3.0],600.0)
    # SSRPlots.plotShadowManifold(x[:,2], 3, 30, show=0, titlestr='Exp 1, v, lag 30')
    # SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=0, titlestr='Exp 1, u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,0], 3, 60, show=0, titlestr='Exp 1, s, lag 60')
    # SSRPlots.plotManifold(x,show=1,titlestr='phase space')
