import numpy as np
from scipy.misc import comb

def getBinomialMax(n,p):
    '''
    Return the maximum value of the binomial distribution 
    parameterized by n observations with p probability of 
    success.

    '''
    maxloc = np.floor( (n+1)*p )
    return comb(n,maxloc)*(p**maxloc)*((1-p)**(n-maxloc)) 

def getContinuityConfidence(neps,ndelta,N):
    '''
    Compare the probability of observing that all ndelta points
    map to the epsilon set with the null hypthesis that the
    points are randomly distributed. Random distribution 
    corresponds to neps/N probability per point, where neps
    is the number of points observed within eps of the image
    point. Successes (falling within eps of the image) should
    follow a binomial distribution for a random distribution.
    If ndelta is large enough to be unlikely, then we can reject
    the null hypothesis of random distribution. Since the points
    are consistent with the definition of continuity, we say we 
    are relatively confident (or not) of the presence of a 
    continuous function.

    '''
    p = float(neps) / N
    pmax = getBinomialMax(ndelta,p)
    return 1 - (p**ndelta)/pmax

def chooseEpsilons(M):
    '''
    Estimate the standard deviation of the manifold M (sensu Pecora)
    and take fractions of it for different epsilons.

    '''
    meanM = np.mean(M,axis=0)
    dists = np.sqrt(((M - meanM)**2).sum(axis=1))
    stdM = np.std(dists)
    return std*np.array([0.01,0.02,0.05,0.1,0.2])