import numpy as np
import rk4

def solvePendulum(init,T,dt=0.1,mu=4.0,beta=2.0,A=2.0):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,doublePendulum,mu=mu,beta=beta,A=A)
    return x

def doublePendulum(t,x,mu=0.0,beta=0.0,A=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = x[3]
    dx[3] = -x[3] - beta*np.sin(x[2]) + A*np.sin(x[0])
    return dx

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    import StateSpaceReconstruction as SSR
    dt = 0.025
    finaltime = 600.0
    x = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt=dt)
    numlags = 2
    lagsize = int(0.8/dt) #because dt=0.1 with lagsize 8 works well with smooth manifold
    lagsize0 = SSR.lagsizeFromFirstZeroOfAutocorrelation(x[:,0])
    lagsize1 = SSR.lagsizeFromFirstZeroOfAutocorrelation(x[:,1])
    lagsize2 = SSR.lagsizeFromFirstZeroOfAutocorrelation(x[:,2])
    lagsize3 = SSR.lagsizeFromFirstZeroOfAutocorrelation(x[:,3])
    print((lagsize0,lagsize1,lagsize2,lagsize3))
    # times = np.arange(0,finaltime,dt)
    # SSRPlots.plotManifold(x[:,:3],show=0,titlestr='x, y, z')
    SSRPlots.plotManifold(x[:,(0,1,3)],show=0,titlestr='x, y, w',style='g-')
    SSRPlots.plotManifold(x[:,(0,1,2)],show=0,titlestr='x, y, z',style='r-')
    # SSRPlots.plotShadowManifold(x[:,2],numlags,lagsize2,show=0,hold=0,style='r-')
    SSRPlots.plotShadowManifold(x[:,3],numlags,lagsize3,show=0,hold=0,style='g-',titlestr='lagsize {0}'.format(lagsize3))
    SSRPlots.plotShadowManifold(x[:,2],numlags,lagsize3,show=0,hold=0,style='r-',titlestr='lagsize {0}'.format(lagsize3))
    SSRPlots.plotShadowManifold(x[:,3],numlags,lagsize,show=0,hold=0,style='g-',titlestr='lagsize {0}'.format(lagsize))
    SSRPlots.plotShadowManifold(x[:,2],numlags,lagsize,show=0,hold=0,style='r-',titlestr='lagsize {0}'.format(lagsize))
    SSRPlots.plotShadowManifold(x[:,3],numlags,75,show=0,hold=0,style='g-',titlestr='lagsize 75')
    SSRPlots.plotShadowManifold(x[:,2],numlags,75,show=1,hold=0,style='r-',titlestr='lagsize 75')
    # SSRPlots.plotManifold(np.hstack([x[:,(0,1)],-4*np.ones((x.shape[0],1))]),show=1,hold=1,style='r')
    # SSRPlots.plotManifold(x[:,(1,2)],show=0,titlestr='y and z')
    # SSRPlots.plotManifold(x[:,(1,3)],show=0,titlestr='y and w')
    # SSRPlots.plotShadowManifold(x[:,2]+x[:,3],numlags,lagsize,show=0,hold=0,style='r-')
    # SSRPlots.plotShadowManifold(x[:len(x)/4,3],numlags,lagsize,show=0,hold=0,style='r-',smooth=0)
    # SSRPlots.plotShadowManifold(x[:len(x)/2,3],numlags,lagsize,show=0,hold=0,style='b-',smooth=0)
    # SSRPlots.plotShadowManifold(x[:len(x)/1,3],numlags,lagsize,show=1,hold=0,style='g-',smooth=0)
    # SSRPlots.plotShadowManifold(x[:len(x)/4,3],numlags,lagsize,show=0,hold=0,style='r-')
    # SSRPlots.plotShadowManifold(x[:len(x)/2,0],numlags,lagsize,show=0,hold=0,style='r-',titlestr='x')
    # SSRPlots.plotShadowManifold(x[:len(x)/2,3],numlags,lagsize,show=1,hold=0,style='g-',titlestr='w')
    # SSRPlots.plots(times,x[:,2:],show=1,hold=0,stylestr=['r-','g-'])
    # SSRPlots.plots(times,x[:,3],show=1,hold=0)
    # SSRPlots.plotManifold(x[:,2:],show=1,titlestr='z and w')
    # SSRPlots.plotManifold(x[:,:-1],show=0,titlestr='x, y, and z')
    # SSRPlots.plotManifold(x[:,(0,2,3)],show=0,titlestr='x, z, and w')
    # SSRPlots.plotManifold(x[:,1:],show=1,titlestr='y, z, and w')
