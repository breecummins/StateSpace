import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})
import fileops

def plotOutput(forwardconf,inverseconf,epsprops,legendarray,forwardtitle,inversetitle,logs = [1,1],forwardfname='',inversefname=''):
    colormap = plt.cm.jet
    plt.figure()
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1.0, len(legendarray))])
    if logs[0]:
        if np.any(forwardconf):
            plt.semilogy(epsprops,forwardconf.transpose())
            plt.ylim([0,1])
            # plt.legend([str(m) for m in Mlens],loc=0)
            plt.legend([str(int(m*100))+'%' for m in legendarray],loc=0,prop={'size':16})
            plt.ylabel(r'$\Theta$',rotation='horizontal')
            plt.xlabel(r'$\epsilon$')
            plt.title(forwardtitle[-12:])
            if forwardfname:
                plt.savefig(forwardfname)
                plt.close()
            else:
                plt.show()
        else:
            print("Data has no positive values in the forward direction.")
    else:   
        plt.plot(epsprops,forwardconf.transpose())
        plt.ylim([0,1])
        # plt.legend([str(m) for m in Mlens],loc=0)
        plt.legend([str(int(m*100))+'%' for m in legendarray],loc=0,prop={'size':16})
        plt.ylabel(r'$\Theta$',rotation='horizontal')
        plt.xlabel(r'$\epsilon$')
        plt.title(forwardtitle[-12:])
        if forwardfname:
            plt.savefig(forwardfname)
            plt.close()
        else:
            plt.show()
    plt.figure()
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1.0, len(legendarray))])
    if logs[1]:
        if np.any(inverseconf):
            plt.semilogy(epsprops,inverseconf.transpose())
            plt.ylim([0,1])
            # plt.legend([str(m) for m in Mlens],loc=0)
            plt.legend([str(int(m*100))+'%' for m in legendarray],loc=0,prop={'size':16})
            plt.ylabel(r'$\Theta$',rotation='horizontal')
            plt.xlabel(r'$\epsilon$')
            plt.title(inversetitle[-12:])
            if inversefname:
                plt.savefig(inversefname)
                plt.close()
            else:
                plt.show()
        else:
            print("Data has no positive values in the inverse direction.")
    else:   
        plt.plot(epsprops,inverseconf.transpose())
        plt.ylim([0,1])
        # plt.legend([str(m) for m in Mlens],loc=0)
        plt.legend([str(int(m*100))+'%' for m in legendarray],loc=0,prop={'size':16})
        plt.ylabel(r'$\Theta$',rotation='horizontal')
        plt.xlabel(r'$\epsilon$')
        plt.title(inversetitle[-12:])
        if inversefname:
            plt.savefig(inversefname)
            plt.close()
        else:
            plt.show()

def plotContinuityConfWrapper_SaveFigs(basedir,fname,logs=[0,0]):
    outdict = fileops.loadPickle(basedir+fname)
    base = basedir+fname[:-7] #get rid of .pickle at end of fname
    var1 = base[-2]
    var2 = base[-1]
    forwardfname = base + '_M' + var1 + 'toM' + var2 + '.eps'
    inversefname = base + '_M' + var2 + 'toM' + var1 + '.eps'    
    plotOutput(outdict['forwardconf'],outdict['inverseconf'],outdict['epsprops'],outdict['tsprops'],outdict['forwardtitle'],outdict['inversetitle'],logs,forwardfname,inversefname)

def plotContinuityConfWrapper_SaveFigs_Noise(basedir,fnames,noises,logs=[0,0]):
    forconf = []
    invconf = []
    for f in fnames:
        outdict = fileops.loadPickle(basedir+f)
        forconf.append(outdict['forwardconf'][-1].squeeze())
        invconf.append(outdict['inverseconf'][-1].squeeze())
    base = basedir+f[:-7] #get rid of .pickle at end of fname
    var1 = base[-2]
    var2 = base[-1]
    chunk = 'noise{:02d}_'.format(int(noises[-1]*100))
    forwardfname = base.replace(chunk,'') + '_M' + var1 + 'toM' + var2 + '.png'
    inversefname = base.replace(chunk,'') + '_M' + var2 + 'toM' + var1 + '.png' 
    plotOutput(np.array(forconf),np.array(invconf),outdict['epsprops'],noises,outdict['forwardtitle'],outdict['inversetitle'],logs,forwardfname,inversefname)

def plotContinuityConfWrapper(basedir,fname,logs=[0,0]):
    outdict = fileops.loadPickle(basedir+fname)
    np.set_printoptions(linewidth=125)
    print("Epsilon as proportions of (changing) standard deviations: {0}.".format(outdict['epsprops']))
    print(outdict['forwardtitle'])
    print("Forward continuity confidence: ")
    print(outdict['forwardconf'])
    print(outdict['inversetitle'])
    print("Inverse continuity confidence: ")
    print(outdict['inverseconf'])
    # print("Real epsilon values for M1: ")
    # for e in outdict['epsM1']: 
    #     print(e)
    # print("Real epsilon values for M2: ")
    # for e in outdict['epsM2']: 
    #     print(e)
    # print("Forward probability of one point mapping into M2 epsilon: ")
    # print(outdict['forwardprobs'])
    # print("Inverse probability of one point mapping into M1 epsilon: ")
    # print(outdict['inverseprobs'])    
    # plotOutput(outdict['forwardconf'],outdict['inverseconf'],outdict['epsprops'],outdict['tsprops'],len(outdict['ts']),outdict['forwardtitle'],outdict['inversetitle'],logs)
    # plt.show()

def plotAutoCorrWrapper(finaltime=1200.0):
    import PecoraScriptsModified as PS
    import StateSpaceReconstruction as SSR
    import StateSpaceReconstructionPlots as SSRPlots
    #autocorrelation pics
    eqns,names,ts = PS.doublependulummodifiedTS(finaltime)
    compind1=2
    # compind2=3
    Mlens = ts.shape[0]*np.arange(0.5,1.1,0.5)
    for L in Mlens:
        autocorr1 = SSR.getAutocorrelation(ts[:L,compind1],int(0.5*L))
        # autocorr2 = SSR.getAutocorrelation(ts[:L,compind2],int(0.5*L))
        SSRPlots.plotAutocorrelation(autocorr1,"z autocorr, length {0}".format(L))
        # SSRPlots.plotAutocorrelation(autocorr2,"w autocorr, length {0}".format(L))

def plotConfForCstp():
    from scipy.misc import comb
    def calcConf(n,p):
        k = np.floor( (n+1)*p )
        return 1 - (1./comb(n,k))*p**(n-k)/(1-p)**(n-k)
    ps = [1.e-6,1.e-5,1.e-4,1.e-3,1.e-2,5.e-2]
    nums = range(1,5)
    for p in ps:
        conf = []
        for n in nums:
            conf.append(calcConf(n,p))
        plt.figure()
        plt.plot(nums,conf)
        plt.ylabel(r'$\theta$')
        plt.xlabel(r'$n_{\delta}$')
        plt.title(r'$p$ = {0}'.format(p))
    plt.show()



if __name__ == '__main__':
    basedir = '/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPchangedbeta/'
    fname = 'DP_1200time_samefixedlags_fixedeps_zw.pickle'
    plotContinuityConfWrapper(basedir,fname)
    fname = 'DP_1200time_samefixedlags_fixedeps_xy.pickle'
    plotContinuityConfWrapper(basedir,fname,[0,0])
    fname = 'DP_1200time_samefixedlags_fixedeps_xw.pickle'
    plotContinuityConfWrapper(basedir,fname,[1,0])
    # ################################
    # basedir = '/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPModified/changedbeta/'
    # fname = 'DPMod_1200time_samefixedlags_fixedeps_beta1-2_zw.pickle'
    # plotContinuityConfWrapper(basedir,fname)
    # fname = 'DPMod_1200time_samefixedlags_fixedeps_beta1-2_xy.pickle'
    # plotContinuityConfWrapper(basedir,fname,[0,0])
    # fname = 'DPMod_1200time_difffixedlags_fixedeps_beta1-2_xw.pickle'
    # plotContinuityConfWrapper(basedir,fname,[1,0])
    # ##############################
    # plotConfForCstp()
    ##############################
    # plotAutoCorrWrapper(2400.0)
