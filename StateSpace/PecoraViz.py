import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})
import fileops

def plotOutput(forwardconf,inverseconf,mastereps,masterts,tslength,forwardtitle,inversetitle):
    Mlens = (masterts*tslength).astype(int)
    plt.figure(0)
    plt.plot(mastereps,forwardconf.transpose())
    plt.legend([str(m) for m in Mlens],loc=0)
    plt.ylabel(r'$\theta_{C^0}$')
    plt.xlabel(r'$\epsilon$')
    plt.title(forwardtitle)
    plt.figure(1)
    plt.plot(mastereps,inverseconf.transpose())
    plt.legend([str(m) for m in Mlens],loc=0)
    plt.ylabel(r'$\theta_{I^0}$')
    plt.xlabel(r'$\epsilon$')
    plt.title(inversetitle)
    plt.show()

if __name__ == '__main__':
    # import PecoraScripts as PS
    # masterts = np.arange(0.2,1.1,0.2)
    # mastereps=np.array([0.01,0.05,0.1,0.2])
    # # outdict = PS.testLorenz(masterts,mastereps)
    # outdict = PS.testDoublePendulum(masterts,mastereps)
    outdict = fileops.loadPickle('DP_1200pts_307lag_zw.pickle')
    plotOutput(outdict['forwardconf'],outdict['inverseconf'],outdict['mastereps'],outdict['masterts'],len(outdict['ts']),outdict['forwardtitle'],outdict['inversetitle'])