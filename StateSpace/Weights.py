import numpy as np

def makeExpWeights(dists):
    '''
    dists is 1D array
    '''
    u = np.exp( -dists/dists[0])
    return u / u.sum()

def makeUniformWeights(dists):
    '''
    dists is 1D array
    '''
    return (1. / dists.shape[0])*np.ones(dists.shape)

def makeLambdaWeights(dists,lam=0.5):
    '''
    dists is 1D array
    '''
    powers = [lam**n for n in range(dists.shape[0])]
    return np.array([p/sum(powers) for p in powers])




