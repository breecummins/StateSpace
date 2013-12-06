import numpy as np
import rk4

def solvePendulum(init,T,dt=0.1,mu=4.0,beta=1.2,A=2.0):
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
    dt = 0.02
    # finaltime = 600.0
    # x = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt=dt)
    # acc3 = SSR.getAutocorrelation(x[:,3],int(x.shape[0] / 10.))
    # SSRPlots.plots(np.arange(1,len(acc3)+1),np.array(acc3),show=0,titlestr='w autocc')
    # finaltime = 1200.0
    # x = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt=dt)
    # newlags = SSR.chooseLagSize(x)
    # print(newlags)
    # Mz = SSR.makeShadowManifold(x[:,2],3,115*60)
    # print(Mz.shape)
    # acc3 = SSR.getAutocorrelation(x[:,3],int(x.shape[0] / 10.))
    # SSRPlots.plots(np.arange(1,len(acc3)+1),np.array(acc3),show=0,titlestr='w autocc')
    finaltime = 2400.0
    finaltime = 600.0
    x = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt=dt)
    # SSRPlots.plots(np.arange(0,x.shape[0]*0.25,0.25),0.01*x[:,0]-2*0.01,show=1)
    # ac = SSR.getAutocorrelation(x[:,2],int(x.shape[0] / 3.))
    # SSRPlots.plots(np.arange(1,len(ac)+1),np.array(ac),show=1,titlestr='z autocorr')
    # numlags = 3
    # lagsize = int(0.8/dt) #because dt=0.1 with lagsize 8 works well with smooth manifold
    # lagsize0 = SSR.lagsizeFromFirstZeroOfAutocorrelation(x[:,0])
    # lagsize1 = SSR.lagsizeFromFirstZeroOfAutocorrelation(x[:,1])
    # lagsize2 = SSR.lagsizeFromFirstZeroOfAutocorrelation(x[:,2],int(x.shape[0] / 2.))
    # print(lagsize2)
    # lagsize3 = SSR.lagsizeFromFirstZeroOfAutocorrelation(x[:,3])
    # acc0 = SSR.getAutocorrelation(x[:,0],int(x.shape[0] / 10.))
    # acc1 = SSR.getAutocorrelation(x[:,1],int(x.shape[0] / 10.))
    # acc2 = SSR.getAutocorrelation(x[:,2],int(x.shape[0] / 2.))
    # acc3 = SSR.getAutocorrelation(x[:,3],int(x.shape[0] / 10.))
    # zeros0 = SSR.findAllZeros(acc0)
    # zeros1 = SSR.findAllZeros(acc1)
    # zeros2 = SSR.findAllZeros(acc2)
    # zeros3 = SSR.findAllZeros(acc3)
    # SSRPlots.plots(np.arange(1,len(acc0)+1),np.array(acc0),show=0,titlestr='x autocc')
    # SSRPlots.plots(np.arange(1,len(acc1)+1),np.array(acc1),show=0,titlestr='y autocc')
    # SSRPlots.plots(np.arange(1,len(acc2)+1),np.array(acc2),show=1,titlestr='z autocc')
    # SSRPlots.plots(np.arange(1,len(acc3)+1),np.array(acc3),show=1,titlestr='w autocc')
    # times = np.arange(0,finaltime,dt)
    # SSRPlots.plotManifold(x[:,:3],show=0,titlestr='x, y, z',color=(2./255,39./255,129./255))
    # SSRPlots.plotManifold(x[:,(0,1,3)],show=1,titlestr='x, y, w',color=(129./255,0./255,23./255))
    # SSRPlots.plotManifold(x,show=0,titlestr='x, y, z, w',scatter=True)
    # SSRPlots.plotManifold(x[:,(0,1,3,2)],show=0,titlestr='x, y, w, z',scatter=True)
    # SSRPlots.plotManifold(x[:,(1,2,3,0)],show=1)
    SSRPlots.plotManifold(x[:,:2],show=1,scatter=True,style='bo')#color=(2./255,39./255,129./255))
    # SSRPlots.plotShadowManifold(x[:,0],3,100,show=0,hold=0,style='k-',titlestr=r'$M_x$')
    # SSRPlots.plotShadowManifold(x[:,1],3,100,show=0,hold=0,titlestr=r'$M_y$',color=(0,129./255,20./255))
    # SSRPlots.plotShadowManifold(x[:,2],3,115,show=0,hold=0,titlestr=r'$M_z$',color=(2./255,39./255,129./255))
    # SSRPlots.plotShadowManifold(x[:,3],3,100,show=1,hold=0,titlestr=r'$M_w$',color=(129./255,0./255,23./255))
    # SSRPlots.plotShadowManifold(x[:,2],numlags,int(float(lagsize2)/lagsize3)*lagsize3,show=1,hold=0,style='r-',titlestr='Mz, lagsize {0} = {1}*{2}'.format((int(float(lagsize2)/lagsize3)*lagsize3),lagsize3,int(float(lagsize2)/lagsize3)))

    # SSRPlots.plotShadowManifold(x[:,3],numlags,lagsize,show=0,hold=0,style='g-',titlestr='lagsize {0}'.format(lagsize))
    # SSRPlots.plotShadowManifold(x[:,2],numlags,lagsize,show=0,hold=0,style='r-',titlestr='lagsize {0}'.format(lagsize))
    # SSRPlots.plotShadowManifold(x[:,3],numlags,zeros3[1],show=0,hold=0,style='g-',titlestr='lagsize {0}'.format(zeros3[1]))
    # SSRPlots.plotShadowManifold(x[:,2],numlags,zeros3[1],show=1,hold=0,style='r-',titlestr='lagsize {0}'.format(zeros3[1]))
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
