import numpy as np

def solve2Species(init,T,dt=0.01,rx=3.8,ry=3.5,bxy=0.02,byx=0.1):
    trange = np.arange(0,T,dt)
    xy = np.zeros((trange.shape[0],2))
    xy[0,:] = init
    for k,t in enumerate(trange[:-1]):
        xy[k+1,0] = xy[k,0]*(rx - rx*xy[k,0] - bxy*xy[k,1])
        xy[k+1,1] = xy[k,1]*(ry - ry*xy[k,1] - byx*xy[k,0])
    return xy

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    timeseries = solve2Species([0.4,0.2],4,bxy=0)
    SSRPlots.plotManifold(timeseries,show=1)
    SSRPlots.plotManifold(timeseries[275:375,0],show=0)
    SSRPlots.plotManifold(timeseries[275:375,1],show=0,hold=1,style='k-')
    SSRPlots.plotShadowManifold(timeseries[:,0],2,8,show=0)
    SSRPlots.plotShadowManifold(timeseries[:,1],2,8,show=1,hold=0,style='k-')

