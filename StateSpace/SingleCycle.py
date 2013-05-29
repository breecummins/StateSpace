import rk4
import numpy as np

#FIXME: need something with chaotic behavior

def solveCycle(init,finaltime,dt=0.1,a=2.0,b=2.0,c=2.0):
    '''
    init = [x[0],y[0],z[0]] are the initial conditions
    dt is the time step
    finaltime is the length of time to solve for

    '''
    times = np.arange(0,finaltime,dt)
    timeseries = np.zeros((len(times),len(init)))
    timeseries[0,:] = init
    for k,ti in enumerate(times[:-1]):
        timeseries[k+1,:] = rk4.solverp(ti,timeseries[k,:],dt,singlecycle,a=a,b=b,c=c)
    return timeseries


def singlecycle(t,xyz,a=0.0,b=0.0,c=0.0):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    dxyz = np.zeros(xyz.shape)
    dxyz[0] = -b*x+a*z
    dxyz[1] = -c*y+b*x
    dxyz[2] = -a*z+c*y
    return dxyz

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    timeseries = solveCycle([4.0,0.0,5.0],5000.0)
    SSRPlots.plotManifold(timeseries,show=1)
    # SSRPlots.plotShadowManifold(timeseries[:,0],3,8,show=0,hold=1,style='r-')
    # SSRPlots.plotShadowManifold(timeseries[:,0],2,8,show=0,hold=0,style='r-')
    # SSRPlots.plotShadowManifold(timeseries[:,1],2,8,show=0,hold=0,style='g-')
    # SSRPlots.plotShadowManifold(timeseries[:,2],2,8,show=1,hold=0,style='k-')
