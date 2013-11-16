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

def solveDiamond(init,T,dt=0.01,mu=4.0,beta=1.2,A=2.0,a=0.2,b=0.2,c=5.7,d=0.15,B=1.25):
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
    dx[4] = -(x[5] + x[6]) 
    dx[5] = x[4] + a*x[5] 
    dx[6] = b - c*x[6] - d*(x[0] - 3)*(x[4]**2 + x[6]**2)/4.0 + (2 + d*(x[0] - 3))*(x[4]*x[6])/2.0
    dx[7] = -x[7] + B*np.sin(x[6])*np.sin(x[2])
    return dx

def solveDiamondNonlinearTerm(init,T,dt=0.01,mu=4.0,beta=1.2,A=2.0,a=0.2,b=0.2,c=5.7,d=0.1,B=1.25):
    times = np.arange(0,T,dt)
    x = np.zeros((len(times),len(init)))
    x[0,:] = init
    for k,t in enumerate(times[:-1]):
        x[k+1,:] = rk4.solverp(t,x[k,:],dt,diamondNonlinearTerm,mu=mu,beta=beta,A=A,a=a,b=b,c=c,d=d,B=B)
    return x

def diamondNonlinearTerm(t,x,mu=0.0,beta=0.0,A=0.0,a=0.0,b=0.0,c=0.0,d=0.0,B=0.0):
    dx = np.zeros(x.shape)
    dx[0] = x[1]
    dx[1] = mu*(1.0 - x[0]**2)*x[1] - x[0] #Van der Pol oscillator
    dx[2] = x[3]
    dx[3] = -x[3] - beta*np.sin(x[2]) + A*np.sin(x[0])
    dx[4] = -(x[5] + x[6])
    dx[5] = x[4] + a*x[5] 
    dx[6] = b + x[6]*(x[4] - c) + d*x[0]**2
    dx[7] = -x[7] + B*np.sin(x[6])*np.sin(x[2])
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

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    # x = solveRossler([0.5,0.5,0.1],600)
    # SSRPlots.plotShadowManifold(x[:,2], 3, 60, show=0, titlestr=r'$M_z$',style='k-')
    # SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=0, titlestr=r'$M_y$',style='r-')
    # SSRPlots.plotShadowManifold(x[:,0], 3, 50, show=0, titlestr=r'$M_x$',style='g-')
    # SSRPlots.plotManifold(x,show=1,titlestr='x,y,z',style='k-')
    #########################
    x = solveDiamond([1.0,2.0,3.0,2.0,0.5,0.5,0.1,0.75],600.0,d=0.2)
    # SSRPlots.plotShadowManifold(x[:,6], 3, 50, show=0, titlestr=r'$M_v$, lag 50')
    # SSRPlots.plotShadowManifold(x[:,5], 3, 65, show=0, titlestr=r'$M_u$, lag 65')
    # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr=r'$M_s$, lag 60')
    # SSRPlots.plotManifold(x[:,[0,1,7]],show=0,titlestr='x,y,p')
    # SSRPlots.plotManifold(x[:,4:7],show=1,titlestr='phase space')
    # SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=1,color=(0.2,0.5,0.2))
    SSRPlots.plotShadowManifold(x[:,5], 3, 60, show=1,color=(0.7,0,0))
    # SSRPlots.plotManifold(x[:,4:7],show=1,style='k-')
    # #########################
    # x = solveDiamondNonlinearTerm([1.0,2.0,3.0,2.0,4.0,4.0,4.0,0.75],600.0,d=0.4)
    # SSRPlots.plotShadowManifold(x[:,6], 3, 50, show=0, titlestr=r'$M_v$, lag 50')
    # SSRPlots.plotShadowManifold(x[:,5], 3, 65, show=0, titlestr=r'$M_u$, lag 65')
    # SSRPlots.plotShadowManifold(x[:,4], 3, 60, show=0, titlestr=r'$M_s$, lag 60')
    # SSRPlots.plotManifold(x[:,[0,1,7]],show=0,titlestr='x,y,p')
    # SSRPlots.plotManifold(x[:,4:7],show=1,titlestr='phase space')
    # # #########################
    # x = solveRotatedRossler([5,4,3],600.0)
    # # SSRPlots.plotShadowManifold(x[:,2], 3, 60, show=0, titlestr=r'$M_v$',style='k-')
    # # SSRPlots.plotShadowManifold(x[:,1], 3, 60, show=0, titlestr=r'$M_u$',style='r-')
    # # SSRPlots.plotShadowManifold(x[:,0], 3, 50, show=1, titlestr=r'$M_s$',style='g-')
    # SSRPlots.plotManifold(x,show=1,color=(2./255,39./255,129./255))
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
