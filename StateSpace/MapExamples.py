#!/usr/bin/sh python

import numpy as np
import matplotlib.pyplot as plt

def mapSugi(x0,y0,N):
    x=[x0]
    y=[y0]
    for k in range(1,N+1):
        x.append(x[k-1]*(3.8-3.8*x[k-1]-0.02*y[k-1]))
        y.append(y[k-1]*(3.5-3.5*y[k-1]-0.1*x[k-1]))
    return x,y

def map1(x0,y0,N):
    x=[x0]
    y=[y0]
    for k in range(1,N+1):
        x.append(x[k-1]*(3.8-3.8*x[k-1]-0.02*y[k-1]))
        y.append(y[k-1]*(3.5-3.5*y[k-1]-0.1*(x[k-1])**2))
    return x,y

def map2(x0,y0,N):
    x=[x0]
    y=[y0]
    for k in range(1,N+1):
        x.append(x[k-1]*(3.8-3.8*x[k-1]-0.02*(y[k-1])**(-1)))
        y.append(y[k-1]*(3.5-3.5*y[k-1]-0.1*(x[k-1])**2))
    return x,y

def figs(x,y):
    plt.figure()
    plt.plot(x,'b-')
    plt.hold('on')
    plt.plot(y,'r-')
    plt.show()

# x,y=mapSugi(0.1,0.2,1000)
# figs(x,y)
x,y=map2(0.1,0.2,100)
figs(x,y)

