import numpy as np
import rk4

def doublePendulumDiamond(t,x,mu=0.0,beta=0.0,A=0.0,gamma=0.0,B=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - gamma*x[0] #Van der Pol oscillator
    dx[2] = x[3]
    dx[3] = beta*(1.0 - x[2]**2)*x[3] - x[2] + A*np.sin(x[0]) # driven Van der Pol oscillator
    dx[4] = x[5]
    dx[5] = beta*(1.0 - x[4]**2)*x[5] + 0.5*x[4] - 0.5*x[4]**3 + A*np.sin(x[0]) # driven Duffing
    dx[6] = x[7]
    dx[7] = mu*(1.0 - x[6]**2)*x[7] - x[6] + B*(np.sin(x[2]) + np.sin(x[4])) # doubly driven Van der Pol
    return dx

def solveDiamond(init,T,dt=0.1,mu=4.0,beta=2.0,A=2.0,gamma=10.0,B=1.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,doublePendulumDiamond,mu=mu,beta=beta,A=A,gamma=gamma,B=B)
    return x

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    dt = 0.025
    finaltime = 600.0
    x = solveDiamond([1.0,2.0,1.0,2.0,2.0,2.0,1.0,2.0],finaltime,dt=dt)
    # SSRPlots.plotManifold(x[:,(0,1,3)],show=0,titlestr='x, y, w',style='g-')
    # SSRPlots.plotManifold(x[:,(0,1,2)],show=0,titlestr='x, y, z',style='r-')
    # SSRPlots.plotManifold(x[:,(0,1,5)],show=0,titlestr='x, y, v',style='k-')
    # SSRPlots.plotManifold(x[:,(0,1,4)],show=0,titlestr='x, y, u',style='m-')
    SSRPlots.plotManifold(x[:,(0,1,7)],show=0,titlestr='x, y, q',style='b-')
    SSRPlots.plotManifold(x[:,(0,1,6)],show=1,titlestr='x, y, p',style='c-')
    SSRPlots.plotShadowManifold(x[:,3],3,70,show=0,hold=0,style='r-',titlestr='w')
    SSRPlots.plotShadowManifold(x[:,5],3,60,show=0,hold=0,style='g-',titlestr='v')
    SSRPlots.plotShadowManifold(x[:,7],3,90,show=0,hold=0,style='k-',titlestr='q')
    SSRPlots.plotShadowManifold(x[:,2],3,70,show=0,hold=0,style='b-',titlestr='z')
    SSRPlots.plotShadowManifold(x[:,4],3,90,show=0,hold=0,style='m-',titlestr='u')
    SSRPlots.plotShadowManifold(x[:,6],3,90,show=1,hold=0,style='c-',titlestr='p')
