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

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    x = solveRossler([5,4,3],500)
    SSRPlots.plotManifold(x)