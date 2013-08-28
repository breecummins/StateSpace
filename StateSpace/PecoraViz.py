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

def plotContinuityConfWrapper(basedir,fname,logs=[1,1]):
    outdict = fileops.loadPickle(basedir+fname)
    np.set_printoptions(linewidth=125)
    print("Epsilon as proportions of (changing) standard deviations: {0}.".format(outdict['epsprops']))
    print("Forward continuity confidence: ")
    print(outdict['forwardconf'])
    print("Inverse continuity confidence: ")
    print(outdict['inverseconf'])
    print("Real epsilon values for M1: ")
    for e in outdict['epsM1']: 
        print(e)
    print("Real epsilon values for M2: ")
    for e in outdict['epsM2']: 
        print(e)
    print("Forward probability of one point mapping into M2 epsilon: ")
    print(outdict['forwardprobs'])
    print("Inverse probability of one point mapping into M1 epsilon: ")
    print(outdict['inverseprobs'])    
    plotOutput(outdict['forwardconf'],outdict['inverseconf'],outdict['epsprops'],outdict['tsprops'],len(outdict['ts']),outdict['forwardtitle'],outdict['inversetitle'],logs)

def plotAutocorrelation(autocorr,title):
    plt.figure()
    plt.plot([1,len(autocorr)+1],[0,0],'k')
    plt.hold('on')
    plt.plot(range(1,len(autocorr)+1),autocorr)
    plt.ylabel('autocorrelation')
    plt.xlabel('lag index')
    plt.title(title)
    plt.show()

def plotAutoCorrWrapper():
    import PecoraScriptsModified as PS
    import StateSpaceReconstruction as SSR
    #autocorrelation pics
    eqns,names,ts = PS.doublependulummodifiedTS(finaltime=1200.0)
    compind1=2
    compind2=3
    Mlens = ts.shape[0]*np.arange(0.2,1.1,0.2)
    for L in Mlens:
        autocorr1 = SSR.getAutocorrelation(ts[:L,compind1],int(0.5*L))
        autocorr2 = SSR.getAutocorrelation(ts[:L,compind2],int(0.5*L))
        plotAutocorrelation(autocorr1,"z autocorr, length {0}".format(L))
        plotAutocorrelation(autocorr2,"w autocorr, length {0}".format(L))

if __name__ == '__main__':
    basedir = '/Users/bree/SimulationResults/TimeSeries/PecoraMethod/workingstuff/'
    fname = 'DPMod_1200time_samefixedlags_zw.pickle'
    plotContinuityConfWrapper(basedir,fname)
    fname = 'DPMod_1200time_samefixedlags_xy.pickle'
    plotContinuityConfWrapper(basedir,fname,[0,0])
    fname = 'DPMod_1200time_difffixedlags_xw.pickle'
    plotContinuityConfWrapper(basedir,fname,[1,0])

