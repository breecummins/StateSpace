import rk4
import numpy as np

def solveLorenz(init,finaltime,dt=0.01,sigma=10.,rho=28.,beta=8/3.):
    '''
    init = [x[0],y[0],z[0]] are the initial conditions
    dt is the time step
    last three args are Lorenz parameters
    '''
    times = np.arange(0,finaltime,dt)
    timeseries = np.zeros((len(times),len(init)))
    timeseries[0,:] = init
    for k,ti in enumerate(times[:-1]):
        timeseries[k+1,:] = rk4.solverp(ti,timeseries[k,:],dt,lorenz,sigma=sigma,rho=rho,beta=beta)
    return timeseries


def lorenz(t,xyz,sigma=0.,rho=0.,beta=0.):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    dxyz = np.zeros(xyz.shape)
    dxyz[0] = sigma*(y-x)
    dxyz[1] = x*(rho - z) - y
    dxyz[2] = x*y - beta*z
    return dxyz

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    timeseries = solveLorenz([1.0,0.5,0.5],80.0)
    # SSRPlots.plotManifold(timeseries,show=0)
    # SSRPlots.plotShadowManifold(timeseries[:,0],3,8,show=0,hold=1,style='r-')
    SSRPlots.plotShadowManifold(timeseries[:,0],2,8,show=0,hold=0,style='r-')
    SSRPlots.plotShadowManifold(timeseries[:,1],2,8,show=0,hold=0,style='g-')
    SSRPlots.plotShadowManifold(timeseries[:,2],2,8,show=1,hold=0,style='k-')
