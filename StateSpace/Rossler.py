import numpy as np
import rk4
from scipy.integrate import ode
from functools import partial

def solveRossler(init,T,dt=0.01,a=0.2,b=0.2,c=5.7):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,Rossler,a=a,b=b,c=c)
    return x

def Rossler(t,x,a=0.0,b=0.0,c=0.0):
    dx = np.zeros(x.shape)
    dx[0] = -x[1]-x[2]
    dx[1] = x[0] + a*x[1]
    dx[2] = b+ x[2]*(x[0]-c)
    return dx

def solveDiamond(init,T,dt=0.01,mu=4.0,beta=1.2,A=2.0,a=0.2,b=0.2,c=5.7,d=2.0,B=1.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,diamond,mu=mu,beta=beta,A=A,a=a,b=b,c=c,d=d,B=B)
    return x

def diamond(t,x,mu=0.0,beta=0.0,A=0.0,a=0.0,b=0.0,c=0.0,d=0.0,B=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = x[3]
    dx[3] = -x[3] - beta*np.sin(x[2]) + A*np.sin(x[0])
    dx[4] = -(x[5] + x[6]) + d*np.sin(x[1]) 
    dx[5] = x[4] + a*x[5] + d*np.sin(x[1])
    dx[6] = b + x[6]*(x[4] - c) 
    dx[7] = -x[7] + B*np.sin(x[6])*np.sin(x[2])
    return dx

def solveRosslerVarChange(init,T,dt=0.01,a=0.2,b=0.2,c=5.7):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,RosslerVarChange,a=a,b=b,c=c)
    return x

def RosslerVarChange(t,x,a=0.0,b=0.0,c=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[0]-x[1]-x[2]
    dx[1] = x[0] + a*x[1]
    dx[2] = x[0]-x[1]-x[2] + b + (x[2]-x[0])*(x[0]-c)
    return dx

def solveDrivenRosslerVarChange(init,T,dt=0.01,a=0.2,b=0.2,c=5.7,d=2.0,f=2*np.pi):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,drivenRosslerVarChange,a=a,b=b,c=c,d=d,f=f)
    return x

def solveDrivenRosslerVarChange_SciPySolver(init,T,dt=0.01,a=0.2,b=0.2,c=5.7,d=2.0,f=2*np.pi):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    solver = ode(partial(drivenRosslerVarChange,a=a,b=b,c=c,d=d,f=f)).set_integrator('vode',method='bdf',order=15,nsteps=5000).set_initial_value(init,0)
    c=0
    while solver.successful() and solver.t < T-2*dt:
        solver.integrate(solver.t+dt)
        c += 1
        x[c,:] = solver.y
    return x

def drivenRosslerVarChange(t,x,a=0.0,b=0.0,c=0.0,d=0.0,f=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[0]-x[1]-x[2] + d*np.sin(f*t + np.pi/5)
    dx[1] = x[0] + a*x[1]
    dx[2] = x[0]-x[1]-x[2] + b + (x[2]-x[0])*(x[0]-c)
    return dx

def solveDrivenRossler(init,T,dt=0.01,a=0.2,b=0.2,c=5.7,d=5.0,f=np.pi):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,drivenRossler,a=a,b=b,c=c,d=d,f=f)
    return x

def drivenRossler(t,x,a=0.0,b=0.0,c=0.0,d=0.0,f=0.0):
    dx = np.zeros(x.shape)
    dx[0] = -x[1]-x[2] + d*np.sin(f*t + np.pi/5)
    dx[1] = x[0] + a*x[1]
    dx[2] = b+ x[2]*(x[0]-c)
    return dx

def solveRotatedRossler(init,T,dt=0.01,a=0.2,b=0.2,c=5.7):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,rotatedRossler,a=a,b=b,c=c)
    return x

def rotatedRossler(t,x,a=0.0,b=0.0,c=0.0):
    dx = np.zeros(x.shape)
    dx[0] = 0.5*( (a-1-c)*x[0] + (-a+1-c)*x[1] + (1+a+c)*x[2] +2*b + 0.5*(-(x[0] - x[2])**2 + x[1]**2))
    dx[1] = (1/2.)*(-(c+2)*x[0] - c*x[1] + c*x[2] + 2*b + 0.5*(-(x[0] - x[2])**2 + x[1]**2))
    dx[2] = 0.5*((a-3)*x[0] + (1-a)*x[1] + (1+a)*x[2])
    return dx

def solveDiamondVarChange(init,T,dt=0.01,mu=4.0,beta=1.2,A=2.0,a=0.2,b=0.2,c=5.7,d=0.2,B=1.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,diamondVarChange,mu=mu,beta=beta,A=A,a=a,b=b,c=c,d=d,B=B)
    return x

def diamondVarChange(t,x,mu=0.0,beta=0.0,A=0.0,a=0.0,b=0.0,c=0.0,d=0.0,B=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = x[3]
    dx[3] = -x[3] - beta*np.sin(x[2]) + A*np.sin(x[0])
    dx[4] = x[4]-x[5] - x[6] + d*np.sin(x[1])
    dx[5] = x[4] + a*x[5]
    dx[6] = x[4]-x[5] - x[6] + b + (x[6]-x[4])*(x[4] - c)
    dx[7] = -x[7] + B*np.sin(x[6])*np.sin(x[2])
    return dx

def solveCappedPendulumRossler(init,T,dt=0.01,mu=4.0,a=0.2,b=0.2,c=5.7,d=0.1,B=1.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,cappedPendulumRossler,mu=mu,a=a,b=b,c=c,d=d,B=B)
    return x

def cappedPendulumRossler(t,x,mu=0.0,a=0.0,b=0.0,c=0.0,d=0.0,B=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = x[2]-x[3] - x[4] + d*np.sin(x[1])
    dx[3] = x[2] + a*x[3]
    dx[4] = x[2]-x[3] - x[4] + b + (x[4]-x[2])*(x[2] - c)
    dx[5] = -x[5] + B*np.sin(x[4])
    return dx

def solveCappedPendulumRotatedRossler(init,T,dt=0.01,mu=4.0,a=0.2,b=0.2,c=5.7,d=0.1,B=1.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,cappedPendulumRotatedRossler,mu=mu,a=a,b=b,c=c,d=d,B=B)
    return x

def cappedPendulumRotatedRossler(t,x,mu=0.0,a=0.0,b=0.0,c=0.0,d=0.0,B=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = 0.5*( (a-1-c)*x[2] + (-a+1-c)*x[3] + (1+a+c)*x[4] +2*b + 0.5*(-(x[2] - x[4])**2 + x[3]**2)) + d*np.sin(x[1])
    dx[3] = (1/2.)*(-(c+2)*x[2] - c*x[3] + c*x[4] + 2*b + 0.5*(-(x[2] - x[4])**2 + x[3]**2))
    dx[4] = 0.5*((a-3)*x[2] + (1-a)*x[3] + (1+a)*x[4])
    dx[5] = -x[5] + B*np.sin(x[4])
    return dx

def solvePendulumRotatedRossler(init,T,dt=0.01,mu=4.0,a=0.2,b=0.2,c=5.7,d=0.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,pendulumRotatedRossler,mu=mu,a=a,b=b,c=c,d=d)
    return x

def pendulumRotatedRossler(t,x,mu=0.0,a=0.0,b=0.0,c=0.0,d=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = 0.5*( (a-1-c)*x[2] + (-a+1-c)*x[3] + (1+a+c)*x[4] +2*b + 0.5*(-(x[2] - x[4])**2 + x[3]**2)) + d*np.sin(x[1])
    dx[3] = (1/2.)*(-(c+2)*x[2] - c*x[3] + c*x[4] + 2*b + 0.5*(-(x[2] - x[4])**2 + x[3]**2))
    dx[4] = 0.5*((a-3)*x[2] + (1-a)*x[3] + (1+a)*x[4])
    return dx

def solveDiamondRotated(init,T,dt=0.01,mu=4.0,beta=1.2,A=2.0,a=0.2,b=0.2,c=5.7,d=0.1,B=1.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,diamondRotated,mu=mu,beta=beta,A=A,a=a,b=b,c=c,d=d,B=B)
    return x

def diamondRotated(t,x,mu=0.0,beta=0.0,A=0.0,a=0.0,b=0.0,c=0.0,d=0.0,B=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = x[3]
    dx[3] = -x[3] - beta*np.sin(x[2]) + A*np.sin(x[0]) #linear oscillator
    dx[4] = 0.5*( (a-1-c)*x[4] + (-a+1-c)*x[5] + (1+a+c)*x[6] +2*b + 0.5*(-(x[4] - x[6])**2 + x[5]**2)) + d*np.sin(x[1])
    dx[5] = (1/2.)*(-(c+2)*x[4] - c*x[5] + c*x[6] + 2*b + 0.5*(-(x[4] - x[6])**2 + x[5]**2))
    dx[6] = 0.5*((a-3)*x[4] + (1-a)*x[5] + (1+a)*x[6]) #Rossler
    dx[7] = -x[7] + B*np.sin(x[6])*np.sin(x[2]) #top variable
    return dx

def solveDiamondRotatedMoreDrive(init,T,dt=0.01,mu=4.0,beta=1.2,A=2.0,a=0.2,b=0.2,c=5.7,d=0.4,B=1.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,diamondRotatedMoreDrive,mu=mu,beta=beta,A=A,a=a,b=b,c=c,d=d,B=B)
    return x

def diamondRotatedMoreDrive(t,x,mu=0.0,beta=0.0,A=0.0,a=0.0,b=0.0,c=0.0,d=0.0,B=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = x[3]
    dx[3] = -x[3] - beta*np.sin(x[2]) + A*np.sin(x[0]) #linear oscillator
    dx[4] = 0.5*( (a-1-c)*x[4] + (-a+1-c)*x[5] + (1+a+c)*x[6] +2*b + 0.5*(-(x[4] - x[6])**2 + x[5]**2)) +d*np.sin(x[1])
    dx[5] = 0.5*(-(c+2)*x[4] - c*x[5] + c*x[6] + 2*b + 0.5*(-(x[4] - x[6])**2 + x[5]**2)) +d*np.sin(x[0])
    dx[6] = 0.5*((a-3)*x[4] + (1-a)*x[5] + (1+a)*x[6]) + d*np.sin(x[1]) # driven Rossler
    dx[7] = -x[7] + B*np.sin(x[6])*np.sin(x[2]) #top variable
    return dx

def solveDiamondRotatedDrivenXYMult(init,T,dt=0.01,mu=4.0,beta=1.2,A=2.0,a=0.2,b=0.2,c=5.7,d=0.5,B=1.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,diamondRotatedDrivenXYMult,mu=mu,beta=beta,A=A,a=a,b=b,c=c,d=d,B=B)
    return x

def diamondRotatedDrivenXYMult(t,x,mu=0.0,beta=0.0,A=0.0,a=0.0,b=0.0,c=0.0,d=0.0,B=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = x[3]
    dx[3] = -x[3] - beta*np.sin(x[2]) + A*np.sin(x[0]) #linear oscillator
    dx[4] = 0.5*( (a-1-c)*x[4] + (-a+1-c)*x[5] + (1+a+c)*x[6] +2*b + 0.5*(-(x[4] - x[6])**2 + x[5]**2)) + d*x[1]*x[0]
    dx[5] = 0.5*(-(c+2)*x[4] - c*x[5] + c*x[6] + 2*b + 0.5*(-(x[4] - x[6])**2 + x[5]**2)) + d*x[0]*x[1]
    dx[6] = 0.5*((a-3)*x[4] + (1-a)*x[5] + (1+a)*x[6]) + d*x[1]*x[0] #Rossler
    dx[7] = -x[7] + B*np.sin(x[6])*np.sin(x[2]) #top variable
    return dx

def solveDiamondRotatedInteralMult(init,T,dt=0.025,mu=4.0,beta=1.2,A=2.0,a=0.2,b=0.2,c=5.7,d=0.25,B=1.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,diamondRotatedInternalMult,mu=mu,beta=beta,A=A,a=a,b=b,c=c,d=d,B=B)
    return x

def diamondRotatedInternalMult(t,x,mu=0.0,beta=0.0,A=0.0,a=0.0,b=0.0,c=0.0,d=0.0,B=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = x[3]
    dx[3] = -x[3] - beta*np.sin(x[2]) + A*np.sin(x[0]) #linear oscillator
    dx[4] = 0.5*( (a-1-c)*x[4] + (-a+1-c)*x[5] + (1+a+c)*x[6] +2*b + 0.5*(-d*x[1]*(x[4] - x[6])**2+ x[5]**2)) 
    dx[5] = 0.5*(-(c+2)*x[4] - c*x[5] + c*x[6] + 2*b + 0.5*(-d*x[1]*(x[4] - x[6])**2 + x[5]**2))
    dx[6] = 0.5*((a-3)*x[4] + (1-a)*x[5] + (1+a)*x[6]) #Rossler
    dx[7] = -x[7] + B*np.sin(x[6])*np.sin(x[2]) #top variable
    return dx

def solvePendulumRotatedInteralMult(init,T,dt=0.025,mu=4.0,beta=1.2,A=2.0,a=0.2,b=0.2,c=5.7,d=0.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,pendulumRotatedInternalMult,mu=mu,beta=beta,A=A,a=a,b=b,c=c,d=d)
    return x

def pendulumRotatedInternalMult(t,x,mu=0.0,beta=0.0,A=0.0,a=0.0,b=0.0,c=0.0,d=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = 0.5*( (a-1-c)*x[2] + (-a+1-c)*x[3] + (1+a+c)*x[4] +2*b + 0.5*(-d*x[1]*(x[2] - x[4])**2+ x[3]**2)) 
    dx[3] = 0.5*(-(c+2)*x[2] - c*x[3] + c*x[4] + 2*b + 0.5*(-d*x[1]*(x[2] - x[4])**2 + x[3]**2))
    dx[4] = 0.5*((a-3)*x[2] + (1-a)*x[3] + (1+a)*x[4]) #Rossler
    return dx

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    # x = solveRossler([5,4,3],600)
    # SSRPlots.plotShadowManifold(x[:,2], 3, 60, show=0, titlestr=r'$M_z$',style='k-')
    # SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=0, titlestr=r'$M_y$',style='r-')
    # SSRPlots.plotShadowManifold(x[:,0], 3, 50, show=0, titlestr=r'$M_x$',style='g-')
    # SSRPlots.plotManifold(x,show=0,titlestr='x,y,z',style='k-')
    # x = solveDiamond([1.0,2.0,3.0,2.0,5.0,4.0,3.0,3.0],600.0)
    # SSRPlots.plotManifold(x[:,4:7],show=0,titlestr='s,u,v')
    # SSRPlots.plotManifold(x[:,[0,1,4]],show=0,titlestr='x,y,s')
    # SSRPlots.plotManifold(x[:,[0,1,5]],show=0,titlestr='x,y,u')
    # SSRPlots.plotManifold(x[:,[0,1,6]],show=0,titlestr='x,y,v')
    # SSRPlots.plotManifold(x[:,[0,1,7]],show=0,titlestr='x,y,p')
    # SSRPlots.plotManifold(x[:,[2,6,7]],show=1,titlestr='z,v,p')
    #########################
    # x = solveRossler([5.0,4.0,3.0],600.0)
    # SSRPlots.plotShadowManifold(x[:,2], 3, 60, show=0, titlestr='v, lag 60')
    # SSRPlots.plotShadowManifold(x[:,2], 3, 30, show=0, titlestr='v, lag 30')
    # SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=0, titlestr='u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,0], 3, 60, show=0, titlestr='s, lag 60')
    # SSRPlots.plotManifold(x,show=1,titlestr='phase space')
    # #########################
    # x = solveRosslerVarChange([5.0,4.0,3.0],1200.0)
    # SSRPlots.plotShadowManifold(x[:,2], 3, 60, show=0, titlestr='var change, v, lag 60')
    # SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=0, titlestr='var change, u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,0], 3, 60, show=0, titlestr='var change, s, lag 60')
    # SSRPlots.plotManifold(x,show=1,titlestr='phase space')
    # #########################
    # x = solveDiamond([1.0,2.0,3.0,2.0,5.0,4.0,3.0,0.75],600.0,d=2.0)
    # SSRPlots.plotShadowManifold(x[:,6], 3, 60, show=0, titlestr='var change, v, lag 60')
    # SSRPlots.plotShadowManifold(x[:,5], 3, 60, show=0, titlestr='var change, u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr='var change, s, lag 60')
    # SSRPlots.plotManifold(x[:,[0,1,7]],show=0,titlestr='x,y,p')
    # SSRPlots.plotManifold(x[:,4:7],show=1,titlestr='phase space')
    # #########################
    # x = solveDiamondVarChange([1.0,2.0,3.0,2.0,5.0,4.0,3.0,0.75],600.0)
    # SSRPlots.plotShadowManifold(x[:,6], 3, 60, show=0, titlestr='var change, v, lag 60')
    # SSRPlots.plotShadowManifold(x[:,5], 3, 60, show=0, titlestr='var change, u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr='var change, s, lag 60')
    # SSRPlots.plotManifold(x[:,4:7],show=0,titlestr='Rossler phase space')
    # # SSRPlots.plotManifold(x[:,[0,1,4]],show=0,titlestr='x,y,s')
    # # SSRPlots.plotManifold(x[:,[0,1,5]],show=0,titlestr='x,y,u')
    # # SSRPlots.plotManifold(x[:,[0,1,6]],show=0,titlestr='x,y,v')
    # SSRPlots.plotManifold(x[:,[0,1,7]],show=1,titlestr='x,y,p')
    # #########################
    # x = solveCappedPendulumRossler([1.0,2.0,5.0,4.0,3.0,0.75],600.0)
    # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr='var change, v, lag 60')
    # SSRPlots.plotShadowManifold(x[:,3], 3, 60, show=0, titlestr='var change, u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,2], 3, 60, show=0, titlestr='var change, s, lag 60')
    # SSRPlots.plotManifold(x[:,2:5],show=0,titlestr='Rossler phase space')
    # SSRPlots.plotManifold(x[:,[0,1,5]],show=1,titlestr='x,y,p')
    # #########################
    # x = solveDrivenRosslerVarChange([0.1,0.1,0.1],600.0)
    # SSRPlots.plotShadowManifold(x[:,2], 3, 60, show=0, titlestr='var change, v, lag 60')
    # SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=0, titlestr='var change, u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,0], 3, 60, show=0, titlestr='var change, s, lag 60')
    # SSRPlots.plotManifold(x,show=1,titlestr='Rossler phase space, RK4')
    # x = solveDrivenRosslerVarChange_SciPySolver([5.0,4.0,3.0],600.0)
    # SSRPlots.plotShadowManifold(x[:,2], 3, 60, show=0, titlestr='var change, v, lag 60')
    # SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=0, titlestr='var change, u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,0], 3, 60, show=0, titlestr='var change, s, lag 60')
    # SSRPlots.plotManifold(x,show=1,titlestr='Rossler phase space, scipy ode')
    # #########################
    # x = solveDrivenRossler([5.0,4.0,3.0],600.0)
    # SSRPlots.plotShadowManifold(x[:,2], 3, 60, show=0, titlestr='var change, v, lag 60')
    # SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=0, titlestr='var change, u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,0], 3, 60, show=0, titlestr='var change, s, lag 60')
    # SSRPlots.plotManifold(x,show=1,titlestr='Rossler phase space')
    # #########################
    x = solveRotatedRossler([5,4,3],600.0)
    # SSRPlots.plotShadowManifold(x[:,2], 3, 60, show=0, titlestr=r'$M_v$',style='k-')
    # SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=0, titlestr=r'$M_u$',style='r-')
    # SSRPlots.plotShadowManifold(x[:,0], 3, 50, show=1, titlestr=r'$M_s$',style='g-')
    SSRPlots.plotManifold(x,show=1,color=(2./255,39./255,129./255))
    # #########################
    # x = solveCappedPendulumRotatedRossler([1.0,2.0,5.0,4.0,3.0,0.75],600.0)
    # SSRPlots.plotShadowManifold(x[:,5], 3, 100, show=1, titlestr='var change, p, lag 100')
    # # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr='var change, v, lag 60')
    # # SSRPlots.plotShadowManifold(x[:,3], 3, 60, show=0, titlestr='var change, u, lag 60')
    # # SSRPlots.plotShadowManifold(x[:,2], 3, 50, show=0, titlestr='var change, s, lag 50')
    # # SSRPlots.plotManifold(x[:,2:5],show=0,titlestr='Rossler phase space')
    # # SSRPlots.plotManifold(x[:,[0,1,5]],show=1,titlestr='x,y,p')
    # #########################
    # x = solveDiamondRotated([1.0,2.0,3.0,2.0,5.0,4.0,3.0,0.75],600.0)
    # SSRPlots.plotShadowManifold(x[:,6], 3, 60, show=0, titlestr='rotated v, lag 60')
    # SSRPlots.plotShadowManifold(x[:,5], 3, 60, show=0, titlestr='rotated u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr='rotated s, lag 60')
    # SSRPlots.plotManifold(x[:,4:7],show=0,titlestr='Rossler phase space')
    # SSRPlots.plotManifold(x[:,0:2],show=1,titlestr='Oscillator phase space')
    # # SSRPlots.plotManifold(x[:,[0,1,4]],show=0,titlestr='x,y,s')
    # # SSRPlots.plotManifold(x[:,[0,1,5]],show=0,titlestr='x,y,u')
    # # SSRPlots.plotManifold(x[:,[0,1,6]],show=0,titlestr='x,y,v')
    # # SSRPlots.plotManifold(x[:,[0,1,7]],show=1,titlestr='x,y,p')
    # #########################
    # x = solveDiamondRotatedMoreDrive([1.0,2.0,3.0,2.0,5.0,4.0,3.0,0.75],600.0)
    # SSRPlots.plotShadowManifold(x[:,6], 3, 60, show=0, titlestr='rotated v, lag 60')
    # SSRPlots.plotShadowManifold(x[:,5], 3, 60, show=0, titlestr='rotated u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr='rotated s, lag 60')
    # SSRPlots.plotManifold(x[:,4:7],show=0,titlestr='Rossler phase space')
    # SSRPlots.plotManifold(x[:,[0,1,4]],show=0,titlestr='x,y,s')
    # SSRPlots.plotManifold(x[:,[0,1,5]],show=0,titlestr='x,y,u')
    # SSRPlots.plotManifold(x[:,[0,1,6]],show=0,titlestr='x,y,v')
    # SSRPlots.plotManifold(x[:,[0,1,7]],show=1,titlestr='x,y,p')
    # #########################
    # x = solveDiamondRotatedDrivenXYMult([1.0,2.0,3.0,2.0,5.0,4.0,3.0,0.75],600.0,dt=0.01)
    # # SSRPlots.plotShadowManifold(x[:,6], 3, 60, show=0, titlestr='rotated v, lag 60')
    # # SSRPlots.plotShadowManifold(x[:,5], 3, 60, show=0, titlestr='rotated u, lag 60')
    # # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr='rotated s, lag 60')
    # # SSRPlots.plotManifold(x[:,4:7],show=0,titlestr='Rossler phase space')
    # # SSRPlots.plotManifold(x[:,0:2],show=1,titlestr='Oscillator phase space')
    # # SSRPlots.plotManifold(x[:,[0,1,4]],show=0,titlestr='x,y,s')
    # # SSRPlots.plotManifold(x[:,[0,1,5]],show=0,titlestr='x,y,u')
    # # SSRPlots.plotManifold(x[:,[0,1,6]],show=0,titlestr='x,y,v')
    # SSRPlots.plotShadowManifold(x[:,7], 3, 120, show=0, titlestr='p, lag 120')
    # SSRPlots.plotShadowManifold(x[:,7], 3, 250, show=0, titlestr='p, lag 250')
    # SSRPlots.plotManifold(x[:,[0,1,7]],show=1,titlestr='x,y,p')
    # #########################
    # x = solveDiamondRotatedExp([1.0,2.0,3.0,2.0,5.0,4.0,3.0,0.75],600.0,dt=0.01)
    # SSRPlots.plotShadowManifold(x[:,6], 3, 60, show=0, titlestr='rotated v, lag 60')
    # SSRPlots.plotShadowManifold(x[:,5], 3, 60, show=0, titlestr='rotated u, lag 60')
    # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr='rotated s, lag 60')
    # SSRPlots.plotManifold(x[:,4:7],show=0,titlestr='Rossler phase space')
    # # SSRPlots.plotManifold(x[:,0:2],show=1,titlestr='Oscillator phase space')
    # # SSRPlots.plotManifold(x[:,[0,1,4]],show=0,titlestr='x,y,s')
    # # SSRPlots.plotManifold(x[:,[0,1,5]],show=0,titlestr='x,y,u')
    # # SSRPlots.plotManifold(x[:,[0,1,6]],show=0,titlestr='x,y,v')
    # SSRPlots.plotManifold(x[:,[0,1,7]],show=1,titlestr='x,y,p')
    # #########################
    # x = solvePendulumRotatedRossler([1.0,2.0,5.0,4.0,3.0],600.0,d=0.65)
    # # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr=r'$M_v$')
    # # SSRPlots.plotShadowManifold(x[:,3], 3, 60, show=0, titlestr=r'$M_u$')
    # # SSRPlots.plotShadowManifold(x[:,2], 3, 50, show=0, titlestr=r'$M_s$',style='g-')
    # SSRPlots.plotManifold(x[:,2:],show=1,titlestr='s,u,v')
    # #########################
    # x = solvePendulumRotatedInteralMult([1.0,2.0,5.0,4.0,3.0],600.0)
    # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr=r'$M_v$')
    # SSRPlots.plotShadowManifold(x[:,3], 3, 60, show=0, titlestr=r'$M_u$')
    # SSRPlots.plotShadowManifold(x[:,2], 3, 50, show=0, titlestr=r'$M_s$',style='g-')
    # SSRPlots.plotManifold(x[:,2:],show=1,titlestr='s,u,v')
