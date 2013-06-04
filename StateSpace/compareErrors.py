import numpy as np
import CCM, CCMAlternatives, Similarity, Weights
import StateSpaceReconstruction as SSR
# import StateSpaceReconstructionPlots as SSRPlots
import random

def lorenzTS(finaltime=80.0,dt=0.01):
    from LorenzEqns import solveLorenz
    timeseries = solveLorenz([1.0,0.5,0.5],finaltime,dt)
    eqns = 'Lorenz'
    names = ['x','y','z']
    numlags = 3
    lagsize = int(0.08/dt)
    return eqns,names,numlags,lagsize,timeseries

def doublependulumTS(finaltime=2400.0,dt=0.025):
    from DoublePendulum import solvePendulum
    timeseries = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt)
    eqns = 'Double pendulum'
    names = ['x','y','z','w']
    numlags=4 
    lagsize=int(0.8/dt) 
    return eqns,names,numlags,lagsize,timeseries

def wholeManifoldComparison(names,numlags,lagsize,timeseries,compind1,compind2,startind,endind,corr):
    M1=SSR.makeShadowManifold(timeseries[startind:endind,compind1],numlags,lagsize)
    M2=SSR.makeShadowManifold(timeseries[startind:endind,compind2],numlags,lagsize)

    #function defs
    def calcErrs(M1est,M2est,method,methodnote,M1ref=M1[corr:,:],M2ref=M2[corr:,:],name1='M'+names[compind1],name2='M'+names[compind2]):
        err1 = method(M1ref,M1est)
        err2 = method(M2ref,M2est)
        print("    "+ methodnote+" between M"+name1+" and M"+name1+"' is " + str(err1))
        print("    "+ methodnote+" between M"+name2+" and M"+name2+"' is " + str(err2))

    # Sugihara method
    est1,est2=CCM.crossMap(timeseries[startind:endind,compind1],timeseries[startind:endind,compind2],numlags,lagsize,Weights.makeExpWeights)
    M1Sug=SSR.makeShadowManifold(est1,numlags,lagsize)
    M2Sug=SSR.makeShadowManifold(est2,numlags,lagsize)
    print('############################################################################')
    print("Sugihara method with reconstruction:")
    calcErrs(M1Sug,M2Sug,Similarity.RootMeanSquaredErrorManifold,'RMSE')
    calcErrs(M1Sug,M2Sug,Similarity.MeanErrorManifold,"Mean error per point")
    calcErrs(M1Sug,M2Sug,Similarity.HausdorffDistance,"Hausdorff dist")
    # weighted sums in the embedding space
    print('############################################################################')
    print("Direct estimation of the manifold using exponential weights:")
    M1us1,M2us1=CCMAlternatives.crossMapModified1(M1,M2,Weights.makeExpWeights)
    calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.RootMeanSquaredErrorManifold,'RMSE')
    calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.MeanErrorManifold,'Mean error per point')
    calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.HausdorffDistance,"Hausdorff dist")
    print('############################################################################')
    print("Direct estimation of the manifold using uniform weights:")
    M1us1,M2us1=CCMAlternatives.crossMapModified1(M1,M2,Weights.makeUniformWeights)
    calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.RootMeanSquaredErrorManifold,'RMSE')
    calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.MeanErrorManifold,'Mean error per point')
    calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.HausdorffDistance,"Hausdorff dist")
    print('############################################################################')
    print("Direct estimation of the manifold using weights from powers of 1/2:")
    M1us1,M2us1=CCMAlternatives.crossMapModified1(M1,M2,Weights.makeLambdaWeights)
    calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.RootMeanSquaredErrorManifold,'RMSE')
    calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.MeanErrorManifold,'Mean error per point')
    calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.HausdorffDistance,"Hausdorff dist")

def sequenceOfReconstructions(names,numlags,lagsize,timeseries,compind1,compind2,startind,endind,corr,listoflens,numiters):
    def printResults(note,short,avg1,std1,avg2,std2,name1='M'+names[compind1],name2='M'+names[compind2]):
        print(note + " between " + name1 + " and " + name1 +"': " + ' '.join(["{0:0.6f}".format(i) for i in avg1]))
        # print("Standard deviations for " + short + ' ' + name1 + " and " + name1 +"': " + ' '.join(["{0:0.6f}".format(i) for i in std1]))
        print(note + " between " + name2 + " and " + name2 +"': " + ' '.join(["{0:0.6f}".format(i) for i in avg2]))
        # print("Standard deviations for " + short + ' ' + name2 + " and " + name2 +"': " + ' '.join(["{0:0.6f}".format(i) for i in std2]))

    # Sugihara method with corr coeff
    lol,avgs1,avgs2,stds1,stds2 = CCM.testCausality(timeseries[startind:endind,compind1],timeseries[startind:endind,compind2],numlags,lagsize,listoflens,numiters,Weights.makeExpWeights,[Similarity.corrCoeffPearson,Similarity.RootMeanSquaredErrorTS])
    print('############################################################################')
    print("Sugihara method with correlation coefficient:")
    print("Lengths: " + str(lol))
    printResults("Mean correlation coefficients", "CC", [avgs1[_k][0] for _k in range(len(avgs1))],[stds1[_k][0] for _k in range(len(avgs1))],[avgs2[_k][0] for _k in range(len(avgs1))],[stds2[_k][0] for _k in range(len(avgs1))],names[compind1],names[compind2])
    print('############################################################################')
    print("Sugihara method with root mean square error:")
    print("Lengths: " + str(lol))
    printResults("Mean time series RMSE", "RMSE", [avgs1[_k][1] for _k in range(len(avgs1))],[stds1[_k][1] for _k in range(len(avgs1))],[avgs2[_k][1] for _k in range(len(avgs1))],[stds2[_k][1] for _k in range(len(avgs1))],names[compind1],names[compind2])
    # Sugihara with intermediate reconstruction
    avgs1 = []
    stds1 = []
    avgs2 = []
    stds2 = []
    for l in listoflens:
        startinds = [ s + startind for s in random.sample(range(endind-startind-l),numiters)]
        a1=[]
        a2=[]
        for s in startinds:
            M1orig=SSR.makeShadowManifold(timeseries[s:s+l,compind1],numlags,lagsize)
            M2orig=SSR.makeShadowManifold(timeseries[s:s+l,compind2],numlags,lagsize)
            est1,est2=CCM.crossMapManifold(M1orig,M2orig,numlags,lagsize,Weights.makeExpWeights)
            M1Sug=SSR.makeShadowManifold(est1,numlags,lagsize)
            M2Sug=SSR.makeShadowManifold(est2,numlags,lagsize)
            M1SugRMSE, M2SugRMSE = calcErrs(M1Sug,M2Sug,Similarity.RootMeanSquaredErrorManifold,M1ref=M1orig[corr:,:],M2ref=M2orig[corr:,:])
            M1SugHD, M2SugHD = calcErrs(M1Sug,M2Sug,Similarity.HausdorffDistance,M1ref=M1orig[corr:,:],M2ref=M2orig[corr:,:])
            M1SugME, M2SugME = calcErrs(M1Sug,M2Sug,Similarity.MeanErrorManifold,M1ref=M1orig[corr:,:],M2ref=M2orig[corr:,:])
            a1.extend([M1SugRMSE,M1SugME,M1SugHD])
            a2.extend([M2SugRMSE,M2SugME,M2SugHD])
        avgs1.append([np.mean(a1[_k::3]) for _k in range(3)])
        stds1.append([np.std(a1[_k::3]) for _k in range(3)])
        avgs2.append([np.mean(a2[_k::3]) for _k in range(3)])
        stds2.append([np.std(a2[_k::3]) for _k in range(3)])
    print('############################################################################')
    print("Sugihara method with intermediate reconstruction:")
    print("Lengths: " + str(listoflens))
    printResults("Mean RMSE","RMSE",[avgs1[_k][0] for _k in range(len(avgs1))],[stds1[_k][0] for _k in range(len(avgs1))],[avgs2[_k][0] for _k in range(len(avgs1))],[stds2[_k][0] for _k in range(len(avgs1))])
    printResults("Mean error per point","ME",[avgs1[_k][1] for _k in range(len(avgs1))],[stds1[_k][1] for _k in range(len(avgs1))],[avgs2[_k][1] for _k in range(len(avgs1))],[stds2[_k][1] for _k in range(len(avgs1))])
    printResults("Mean Hausdorff distance","ME",[avgs1[_k][2] for _k in range(len(avgs1))],[stds1[_k][2] for _k in range(len(avgs1))],[avgs2[_k][2] for _k in range(len(avgs1))],[stds2[_k][2] for _k in range(len(avgs1))])

    # weighted sums in the embedding space
    lol,avgs1,avgs2,stds1,stds2 = CCMAlternatives.testCausalityReconstruction(timeseries[startind:endind,compind1],timeseries[startind:endind,compind2],numlags,lagsize,listoflens,numiters,wgtfunc=Weights.makeExpWeights,simMeasure=[Similarity.RootMeanSquaredErrorManifold,Similarity.MeanErrorManifold,Similarity.HausdorffDistance])
    print('############################################################################')
    print("Direct estimation of the manifold using exponential weights:")
    print("Lengths: " + str(listoflens))
    printResults("Mean RMSE","RMSE",[avgs1[_k][0] for _k in range(len(avgs1))],[stds1[_k][0] for _k in range(len(avgs1))],[avgs2[_k][0] for _k in range(len(avgs1))],[stds2[_k][0] for _k in range(len(avgs1))])
    printResults("Mean error per point","ME",[avgs1[_k][1] for _k in range(len(avgs1))],[stds1[_k][1] for _k in range(len(avgs1))],[avgs2[_k][1] for _k in range(len(avgs1))],[stds2[_k][1] for _k in range(len(avgs1))])
    printResults("Mean Hausdorff distance","ME",[avgs1[_k][2] for _k in range(len(avgs1))],[stds1[_k][2] for _k in range(len(avgs1))],[avgs2[_k][2] for _k in range(len(avgs1))],[stds2[_k][2] for _k in range(len(avgs1))])

    lol,avgs1,avgs2,stds1,stds2 = CCMAlternatives.testCausalityReconstruction(timeseries[startind:endind,compind1],timeseries[startind:endind,compind2],numlags,lagsize,listoflens,numiters,wgtfunc=Weights.makeUniformWeights,simMeasure=[Similarity.RootMeanSquaredErrorManifold,Similarity.MeanErrorManifold,Similarity.HausdorffDistance])
    print('############################################################################')
    print("Direct estimation of the manifold using uniform weights:")
    print("Lengths: " + str(listoflens))
    printResults("Mean RMSE","RMSE",[avgs1[_k][0] for _k in range(len(avgs1))],[stds1[_k][0] for _k in range(len(avgs1))],[avgs2[_k][0] for _k in range(len(avgs1))],[stds2[_k][0] for _k in range(len(avgs1))])
    printResults("Mean error per point","ME",[avgs1[_k][1] for _k in range(len(avgs1))],[stds1[_k][1] for _k in range(len(avgs1))],[avgs2[_k][1] for _k in range(len(avgs1))],[stds2[_k][1] for _k in range(len(avgs1))])
    printResults("Mean Hausdorff distance","ME",[avgs1[_k][2] for _k in range(len(avgs1))],[stds1[_k][2] for _k in range(len(avgs1))],[avgs2[_k][2] for _k in range(len(avgs1))],[stds2[_k][2] for _k in range(len(avgs1))])

    lol,avgs1,avgs2,stds1,stds2 = CCMAlternatives.testCausalityReconstruction(timeseries[startind:endind,compind1],timeseries[startind:endind,compind2],numlags,lagsize,listoflens,numiters,wgtfunc=Weights.makeLambdaWeights,simMeasure=[Similarity.RootMeanSquaredErrorManifold,Similarity.MeanErrorManifold,Similarity.HausdorffDistance])
    print('############################################################################')
    print("Direct estimation of the manifold using weights made from powers of 1/2:")
    print("Lengths: " + str(listoflens))
    printResults("Mean RMSE","RMSE",[avgs1[_k][0] for _k in range(len(avgs1))],[stds1[_k][0] for _k in range(len(avgs1))],[avgs2[_k][0] for _k in range(len(avgs1))],[stds2[_k][0] for _k in range(len(avgs1))])
    printResults("Mean error per point","ME",[avgs1[_k][1] for _k in range(len(avgs1))],[stds1[_k][1] for _k in range(len(avgs1))],[avgs2[_k][1] for _k in range(len(avgs1))],[stds2[_k][1] for _k in range(len(avgs1))])
    printResults("Mean Hausdorff distance","ME",[avgs1[_k][2] for _k in range(len(avgs1))],[stds1[_k][2] for _k in range(len(avgs1))],[avgs2[_k][2] for _k in range(len(avgs1))],[stds2[_k][2] for _k in range(len(avgs1))])

if __name__=='__main__':
    # make a time series
    dt = 0.025
    eqns,names,numlags,lagsize,timeseries = doublependulumTS(dt=dt)

    # time series indices
    startind = 500 #how much to cut off the front
    endind=len(timeseries)-0 #how much to cut off the back
    corr = (numlags-1)*lagsize #correction term for shadow manifold creation

    # comparison variables
    compind1 = 2
    compind2 = 3

    # parameters for a sequence of measurements of manifolds of lengths in listoflens with numiters different starting locations (only needed for sequenceOfReconstructions)
    listoflens = range(1000,22600,1000)
    numiters = 25

    # print info about the analysis to be done.
    print('{0} with lagsize of {1!s}*dt with dt = {2!s} and reconstruction dimension {3!s}.'.format(eqns,lagsize,dt,numlags))
    print('If looking at a sequence of measurements, the lengths range from {0!s} to {1!s} and the number of iterations per length is {2!s}.'.format(listoflens[0],listoflens[-1],numiters))

    # run the analysis
    sequenceOfReconstructions(names,numlags,lagsize,timeseries,compind1,compind2,startind,endind,corr,listoflens,numiters)
    # wholeManifoldComparison(names,numlags,lagsize,timeseries,compind1,compind2,startind,endind,corr)











    # # average over the different estimates of a point in time
    # est1,est2=CCMAlternatives.crossMapModified2(M1,M2,Weights.makeExpWeights)
    # M1us2=SSR.makeShadowManifold(est1,numlags,lagsize)
    # M2us2=SSR.makeShadowManifold(est2,numlags,lagsize)
    # M1us2RMSE, M2us2RMSE = calcErrs(M1us2,M2us2,Similarity.RootMeanSquaredError)
    # M1us2HD, M2us2HD = calcErrs(M1us2,M2us2,Similarity.HausdorffDistance)
    # M1us2CM, M2us2CM = calcErrs(M1us2,M2us2,Similarity.countingMeasure)
    # print("Our method 2, average the different time series estimations:")
    # printMe("RMSE",M1us2RMSE,M2us2RMSE)
    # printMe("Hausdorff dist",M1us2HD,M2us2HD)
    # printMe("Counting measure",M1us2CM,M2us2CM)
    # # take a different projection than Sugihara
    # proj = numlags - 1 
    # est1,est2=CCMAlternatives.crossMapModified3(M1,M2,proj,Weights.makeExpWeights)
    # M1us3=SSR.makeShadowManifold(est1,numlags,lagsize)
    # M2us3=SSR.makeShadowManifold(est2,numlags,lagsize)
    # M1us3RMSE, M2us3RMSE = calcErrs(M1us3,M2us3,Similarity.RootMeanSquaredError,M1[corr-proj:-proj,:],M2[corr-proj:-proj,:])
    # M1us3HD, M2us3HD = calcErrs(M1us3,M2us3,Similarity.HausdorffDistance,M1[corr-proj:-proj,:],M2[corr-proj:-proj,:])
    # M1us3CM, M2us3CM = calcErrs(M1us3,M2us3,Similarity.countingMeasure,M1[corr-proj:-proj,:],M2[corr-proj:-proj,:])
    # print("Our method 3, take a different projection (index "+ str(proj) +"):")
    # printMe("RMSE",M1us3RMSE,M2us3RMSE)
    # printMe("Hausdorff dist",M1us3HD,M2us3HD)
    # printMe("Counting measure",M1us3CM,M2us3CM)

    # # plot the shadow manifolds and their estimates
    # SSRPlots.plotEstShadowManifoldSugihara(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)     
    # SSRPlots.plotEstShadowManifoldUs1(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)        
    # SSRPlots.plotEstShadowManifoldUs2(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)        
    # SSRPlots.plotEstShadowManifoldUs3(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize,proj)        

