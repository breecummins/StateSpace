import numpy as np
import rk4

def doublePendulumForm1(t,x,mu=0.0,beta=0.0,A=0.0,gamma=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - gamma*x[0] #Van der Pol oscillator
    dx[2] = x[3]
    dx[3] = beta*(1.0 - x[2]**2)*x[3] + 0.5*x[2] - 0.5*x[2]**3 + A*np.sin(x[0])
    return dx

def doublePendulumForm2(t,x,mu=0.0,beta=0.0,A=0.0,gamma=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1] + mu*( x[0] - (x[0]**3)/3 )
    dx[1] = - gamma*x[0] #Van der Pol oscillator
    dx[2] = x[3] + beta*( x[2] - (x[2]**3)/3 )
    dx[3] =  0.5*x[2] - 0.5*x[2]**3 + A*np.sin(x[0])
    return dx

def solvePendulum(init,T,func=doublePendulumForm1,dt=0.1,mu=4.0,beta=2.0,A=2.0,gamma=10.0):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,func,mu=mu,beta=beta,A=A,gamma=gamma)
    return x

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    dt = 0.025
    finaltime = 600.0
    x = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt=dt)
    SSRPlots.plotManifold(x[:,(0,1,3)],show=0,titlestr='x, y, w',style='g-')
    SSRPlots.plotManifold(x[:,(0,1,2)],show=1,titlestr='x, y, z',style='r-')
    x = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt=dt,func=doublePendulumForm2)
    SSRPlots.plotManifold(x[:,(0,1,3)],show=0,titlestr='x, y, w',style='g-')
    SSRPlots.plotManifold(x[:,(0,1,2)],show=1,titlestr='x, y, z',style='r-')
