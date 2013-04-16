import numpy as np
import rk4

def solvePendulum(init,T,dt=0.05,mu=4.0,beta=2.0,A=2.0):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,doublePendulum,mu=mu,beta=beta,A=A)
    return x

def doublePendulum(t,x,mu=0.0,beta=0.0,A=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0]
    dx[2] = x[3]
    dx[3] = -x[3] - beta*np.sin(x[2]) + A*np.sin(x[0])
    return dx

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    x = solvePendulum([1.0,2.0,3.0,2.0],300.0)
    numlags = 2
    lagsize = 16
    SSRPlots.plotManifold(x[:,:2],show=0,titlestr='x and y')
    # SSRPlots.plotManifold(x[:,(0,2)],show=0,titlestr='x and z')
    SSRPlots.plotManifold(x[:,(0,3)],show=0,titlestr='x and w')
    # SSRPlots.plotManifold(x[:,(1,2)],show=0,titlestr='y and z')
    SSRPlots.plotManifold(x[:,(1,3)],show=0,titlestr='y and w')
    SSRPlots.plotShadowManifold(x[:,0],numlags,lagsize,show=0,hold=0,style='r-')
    SSRPlots.plotShadowManifold(x[:,-1],numlags,lagsize,show=1,hold=0,style='g-')
    # SSRPlots.plotManifold(x[:,2:],show=1,titlestr='z and w')
    # SSRPlots.plotManifold(x[:,:-1],show=0,titlestr='x, y, and z')
    # SSRPlots.plotManifold(x[:,(0,2,3)],show=0,titlestr='x, z, and w')
    # SSRPlots.plotManifold(x[:,1:],show=1,titlestr='y, z, and w')
