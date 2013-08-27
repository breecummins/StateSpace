import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})
import fileops

def plotOutput(forwardconf,inverseconf,epsprops,tsprops,tslength,forwardtitle,inversetitle,logs = [1,1]):
    Mlens = (tsprops*tslength).astype(int)
    if logs[0]:
        if np.any(forwardconf):
            plt.figure()
            plt.semilogy(epsprops,forwardconf.transpose())
            plt.legend([str(m) for m in Mlens],loc=0)
            plt.ylabel(r'$\theta_{C^0}$')
            plt.xlabel(r'$\epsilon$')
            plt.title(forwardtitle)
        else:
            print("Data has no positive values in the forward direction.")
    else:   
        plt.figure()
        plt.plot(epsprops,forwardconf.transpose())
        plt.legend([str(m) for m in Mlens],loc=0)
        plt.ylabel(r'$\theta_{C^0}$')
        plt.xlabel(r'$\epsilon$')
        plt.title(forwardtitle)
    if logs[1]:
        if np.any(inverseconf):
            plt.figure()
            plt.semilogy(epsprops,inverseconf.transpose())
            plt.legend([str(m) for m in Mlens],loc=0)
            plt.ylabel(r'$\theta_{I^0}$')
            plt.xlabel(r'$\epsilon$')
            plt.title(inversetitle)
        else:
            print("Data has no positive values in the inverse direction.")
    else:   
        plt.figure()
        plt.plot(epsprops,inverseconf.transpose())
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
    tsprops = np.arange(0.5,0.85,0.1)
    epsprops=np.array([0.005,0.0075,0.01,0.0125,0.015,0.02,0.04]) #for z and w
    # epsprops=np.array([0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.0075]) #for x and y
    # epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for x and w
    # # outdict = PS.testLorenz(tsprops,epsprops)
    # outdict = PS.testDoublePendulum(tsprops,epsprops)
    # outdict = fileops.loadPickle('DP_600pts_28lag_zw.pickle')
    # outdict = fileops.loadPickle('DP_1200pts_117lag_zw.pickle')
    # outdict = fileops.loadPickle('DP_1200pts_307lag_zw.pickle')
    # outdict = fileops.loadPickle('DP_2400pts_504lag_zw.pickle')
    # outdict = fileops.loadPickle('DP_2400pts_6900lag_zw_closeup.pickle')
    basedir = '/Users/bree/SimulationResults/TimeSeries/PecoraMethod/'
    fname = 'DPMod_1200time_samefixedlags_zw.pickle'
    # PS.testDoublePendulumModified([2,3],tsprops,epsprops,fname = basedir+fname,lags = [[5600,5600]])
    outdict = fileops.loadPickle(basedir+fname)
    logs = [1,1]
    print(outdict['epsprops'])
    print(outdict['forwardconf'])
    print(outdict['inverseconf'])
    plotOutput(outdict['forwardconf'],outdict['inverseconf'],outdict['epsprops'],outdict['tsprops'],len(outdict['ts']),outdict['forwardtitle'],outdict['inversetitle'],logs)

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