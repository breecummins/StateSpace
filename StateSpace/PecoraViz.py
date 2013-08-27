import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})
import fileops

def plotOutput(forwardconf,inverseconf,mastereps,masterts,tslength,forwardtitle,inversetitle):
    Mlens = (masterts*tslength).astype(int)
    plt.figure()
    plt.semilogy(mastereps,forwardconf.transpose())
    plt.legend([str(m) for m in Mlens],loc=0)
    plt.ylabel(r'$\theta_{C^0}$')
    plt.xlabel(r'$\epsilon$')
    plt.title(forwardtitle)
    plt.figure()
    plt.semilogy(mastereps,inverseconf.transpose())
    plt.legend([str(m) for m in Mlens],loc=0)
    plt.ylabel(r'$\theta_{I^0}$')
    plt.xlabel(r'$\epsilon$')
    plt.title(inversetitle)
    plt.show()

def plotAutocorrelation(autocorr,title):
    plt.figure()
    plt.plot([1,len(autocorr)+1],[0,0],'k')
    plt.hold('on')
    plt.plot(range(1,len(autocorr)+1),autocorr)
    plt.ylabel('autocorrelation')
    plt.xlabel('lag index')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    import PecoraScriptsModified as PS
    import StateSpaceReconstruction as SSR
    masterts = np.arange(0.5,0.85,0.1)
    mastereps=np.array([0.005,0.0075,0.01,0.0125,0.015,0.02,0.04])
    # # outdict = PS.testLorenz(masterts,mastereps)
    # outdict = PS.testDoublePendulum(masterts,mastereps)
    # outdict = fileops.loadPickle('DP_600pts_28lag_zw.pickle')
    # outdict = fileops.loadPickle('DP_1200pts_117lag_zw.pickle')
    # outdict = fileops.loadPickle('DP_1200pts_307lag_zw.pickle')
    # outdict = fileops.loadPickle('DP_2400pts_504lag_zw.pickle')
    # outdict = fileops.loadPickle('DP_2400pts_6900lag_zw_closeup.pickle')
    basedir = '/Users/bree/SimulationResults/TimeSeries/PecoraMethod/'
    fname = 'DPMod_1200time_twofixedlags_xw.pickle'
    PS.testDoublePendulumModified(masterts,mastereps,fname = basedir+fname,lags = [[100,5600]])
    outdict = fileops.loadPickle(basedir+fname)
    print(outdict['mastereps'])
    print(outdict['forwardconf'])
    print(outdict['inverseconf'])
    plotOutput(outdict['forwardconf'],outdict['inverseconf'],outdict['mastereps'],outdict['masterts'],len(outdict['ts']),outdict['forwardtitle'],outdict['inversetitle'])




    # #autocorrelation pics
    # eqns,names,ts = PS.doublependulummodifiedTS(finaltime=1200.0)
    # compind1=2
    # compind2=3
    # # autocorr1 = SSR.getAutocorrelation(ts[:,compind1],int(ts.shape[0] / 3.))
    # # plotAutocorrelation(autocorr1,"length {0}".format(ts.shape[0]))
    # Mlens = ts.shape[0]*np.arange(0.2,1.1,0.2)
    # for L in Mlens:
    #     autocorr1 = SSR.getAutocorrelation(ts[:L,compind1],int(0.5*L))
    #     autocorr2 = SSR.getAutocorrelation(ts[:L,compind2],int(0.5*L))
    #     plotAutocorrelation(autocorr1,"z autocorr, length {0}".format(L))
    #     plotAutocorrelation(autocorr2,"w autocorr, length {0}".format(L))