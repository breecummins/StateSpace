#!/usr/bin/sh python

#third party modules
import numpy as np
import matplotlib as mpl
# mpl.use('Pdf') #comment out if you want to see figures at run time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

t = np.linspace(0, 2*np.pi, 100)
x = np.cos(t)

# segments = [ np.array([ [t[k],x[k]],[t[k+1],x[k+1]] ]) for k in range(len(t)-1) ]
points = np.array([t, x]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0,1))
lc.set_array(x**2)

plt.figure()
plt.gca().add_collection(lc)
plt.xlim(0, 2*np.pi)
plt.ylim(-1, 1)
plt.show()



