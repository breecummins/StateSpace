import numpy as np

def corrCoeffPearson(ts1,ts2):
    shift1 = ts1 - np.mean(ts1)
    shift2 = ts2 - np.mean(ts2)
    s12 = ( shift1*shift2 ).sum()
    s11 = ( shift1*shift1 ).sum()
    s22 = ( shift2*shift2 ).sum()
    return s12 / np.sqrt(s11*s22)

def L2error(M1,M2):
    return np.sqrt(((M1-M2)**2).sum(1)).sum(0) / M1.shape[0]

