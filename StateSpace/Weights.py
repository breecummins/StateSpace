import numpy as np

def makeExpWeights(dists):
    u = np.exp( -dists/dists[0])
    return u / u.sum()

