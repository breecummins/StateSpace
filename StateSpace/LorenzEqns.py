import rk4
import numpy as np

def solveLorenz(init,finaltime,dt=0.025,sigma=10.,rho=28.,beta=8/3.):
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

def lorenz(t,x,sigma=0.,rho=0.,beta=0.):
    dx = np.zeros(x.shape)
    dx[0] = sigma*(x[1]-x[0])
    dx[1] = x[0]*(rho - x[2]) - x[1]
    dx[2] = x[0]*x[1] - beta*x[2]
    return dx

def solveLorenzVarChange(init,finaltime,dt=0.025,sigma=10.,rho=28.,beta=8/3.):
    '''
    init = [x[0],y[0],z[0]] are the initial conditions
    dt is the time step
    last three args are Lorenz parameters
    '''
    times = np.arange(0,finaltime,dt)
    timeseries = np.zeros((len(times),len(init)))
    timeseries[0,:] = init
    for k,ti in enumerate(times[:-1]):
        timeseries[k+1,:] = rk4.solverp(ti,timeseries[k,:],dt,lorenzVarChange,sigma=sigma,rho=rho,beta=beta)
    return timeseries

def lorenzVarChange(t,x,sigma=0.,rho=0.,beta=0.):
    dx = np.zeros(x.shape)
    dx[0] = sigma*(x[1]-x[0])
    dx[1] = x[0]*(rho - x[2] + x[0]) - x[1]
    dx[2] = sigma*(x[1]-x[0]) + x[0]*x[1] - beta*(x[2]-x[0])
    return dx

def solveRotatedLorenz(init,finaltime,dt=0.025,sigma=10.,rho=28.,beta=8/3.):
    '''
    init = [x[0],y[0],z[0]] are the initial conditions
    dt is the time step
    last three args are Lorenz parameters
    '''
    times = np.arange(0,finaltime,dt)
    timeseries = np.zeros((len(times),len(init)))
    timeseries[0,:] = init
    for k,ti in enumerate(times[:-1]):
        timeseries[k+1,:] = rk4.solverp(ti,timeseries[k,:],dt,rotatedLorenz,sigma=sigma,rho=rho,beta=beta)
    return timeseries

def rotatedLorenz(t,x,sigma=0.,rho=0.,beta=0.):
    dx = np.zeros(x.shape)
    dx[0] = 0.5*( -(1+ beta+rho)*x[0] + (1-beta+rho)*x[1] + (beta -1 +rho)*x[2] + (x[1]+x[2]-x[0])*(x[2]-x[1]) )
    dx[1] = (sigma - beta/2.)*x[0] - (sigma + beta/2.)*x[1] + x[2]*beta/2. -0.25*((x[0]-x[1])**2 - x[2]**2)
    dx[2] = 0.5*(  (2*sigma-1-rho)*x[0] - (2*sigma-1-rho)*x[1] + (rho-1)*x[2] + 0.5*((x[0]-x[2])**2 - x[1]**2)  ) 
    return dx

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    dt = 0.025
    dt = 0.01
    lag=10
    lag=25
    finaltime = 100.0
    # timeseries = solveLorenzVarChange([1.0,0.5,-0.5],finaltime,dt)
    # numlags = 3
    # lagsize = int(0.08/dt) #because lagsize=8 is good with dt = 0.01
    # SSRPlots.plotManifold(timeseries,show=0)
    # # # SSRPlots.plotShadowManifold(timeseries[:,0],3,8,show=0,hold=1,style='r-')
    # SSRPlots.plotShadowManifold(timeseries[:,0],3,8,show=0,hold=0,style='r-')
    # SSRPlots.plotShadowManifold(timeseries[:,1],3,8,show=0,hold=0,style='g-')
    # SSRPlots.plotShadowManifold(timeseries[:,2],3,8,show=1,hold=0,style='k-')
    # # SSRPlots.plotShadowManifold(timeseries[:,0],3,8,show=0,hold=0,style='r-',smooth=0)
    # # SSRPlots.plotShadowManifold(timeseries[:,1],3,8,show=0,hold=0,style='g-',smooth=0)
    # # SSRPlots.plotShadowManifold(timeseries[:,2],3,8,show=1,hold=0,style='k-',smooth=0)
    ##############################
    timeseries = solveLorenz([1.0,0.5,0.5],finaltime,dt)
    numlags = 3
    # SSRPlots.plotManifold(timeseries,show=0)
    SSRPlots.plotShadowManifold(timeseries[:,2],numlags,lag,show=1,hold=0,color=(0.8,0.2,0.1))
    # SSRPlots.plotShadowManifold(timeseries[:,1],3,10,show=0,hold=0,style='g-')
    # SSRPlots.plotShadowManifold(timeseries[:,2],3,7,show=1,hold=0,style='r-')
