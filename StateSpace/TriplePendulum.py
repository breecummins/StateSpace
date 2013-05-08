import numpy as np
import rk4

def solvePendulum(init,T,dt=0.05,mu=4.0,beta=2.0,gamma=3.5,delta=2.0,A=2.0,B=4.0,C=1.0,D=1.0):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,triplePendulum,mu=mu,beta=beta,gamma=gamma,delta=delta,A=A,B=B,C=C,D=D)
    return x

def triplePendulum(t,x,mu=0.0,beta=0.0,gamma=0.0,delta=0.0,A=0.0,B=0.0,C=0.0,D=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0]
    dx[2] = x[3]
    dx[3] = -x[3] - beta*np.sin(x[2]) + A*np.sin(x[0])
    dx[4] = x[5]
    dx[5] = -x[5] - gamma*np.sin(x[4]) + B*np.sin(x[0])
    dx[6] = x[7]
    dx[7] = -x[7] - delta*np.sin(x[6]) + C*np.sin(x[2]) + D*np.sin(x[4])
    return dx

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    x = solvePendulum([1.0,2.0,3.0,2.0,1.0,1.5,2.5,2.0],300.0)
    numlags = 2
    lagsize = 24
    SSRPlots.plotManifold(x[:,(0,1,3)],show=0,titlestr='x, y, w')
    SSRPlots.plotManifold(x[:,(0,1,5)],show=0,titlestr='x, y, u')
    SSRPlots.plotManifold(x[:,(0,1,7)],show=0,titlestr='x, y, r')
    SSRPlots.plotManifold(x[:,(0,1)],show=0,titlestr='x and y')
    SSRPlots.plotManifold(x[:,(2,3)],show=0,titlestr='z and w')
    SSRPlots.plotManifold(x[:,(4,5)],show=0,titlestr='v and u')
    SSRPlots.plotManifold(x[:,(6,7)],show=0,titlestr='s and r')
    SSRPlots.plotShadowManifold(x[:,0],numlags,lagsize,show=0,hold=0,style='m-',titlestr='x, lag '+str(lagsize))
    SSRPlots.plotShadowManifold(x[:,1],numlags,lagsize,show=0,hold=0,style='r-',titlestr='y, lag '+str(lagsize))
    SSRPlots.plotShadowManifold(x[:,3],numlags,lagsize,show=0,hold=0,style='g-',titlestr='w, lag '+str(lagsize))
    SSRPlots.plotShadowManifold(x[:,5],numlags,lagsize,show=0,hold=0,style='k-',titlestr='u, lag '+str(lagsize))
    SSRPlots.plotShadowManifold(x[:,7],numlags,lagsize,show=1,hold=0,style='b-',titlestr='r, lag '+str(lagsize))
    # SSRPlots.plotManifold(x[:,2:],show=1,titlestr='z and w')
    # SSRPlots.plotManifold(x[:,:-1],show=0,titlestr='x, y, and z')
    # SSRPlots.plotManifold(x[:,(0,2,3)],show=0,titlestr='x, z, and w')
    # SSRPlots.plotManifold(x[:,1:],show=1,titlestr='y, z, and w')
