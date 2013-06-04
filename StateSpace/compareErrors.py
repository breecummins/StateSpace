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

def wholeManifoldComparison(names,numlags,lagsize,timeseries,compind1,compind2):

    ts1 = timeseries[:,compind1]
    ts2 = timeseries[:,compind2]
    M1=SSR.makeShadowManifold(ts1,numlags,lagsize)
    M2=SSR.makeShadowManifold(ts2,numlags,lagsize)
    corr = (numlags-1)*lagsize #correction term for shadow manifold creation

    #function defs
    def printMe(methodnote,name1,name2,err1,err2):
        print("    {0} between {1} and {1}' is {2!s}".format(methodnote,name1,err1))
        print("    {0} between {1} and {1}' is {2!s}".format(methodnote,name2,err2))

    def calcErrs(summary,M1est,M2est,M1ref=M1[corr:,:],M2ref=M2[corr:,:],name1='M{0}'.format(names[compind1]),name2='M{0}'.format(names[compind2])):
        print('############################################################################')
        print(summary)
        err1 = Similarity.RootMeanSquaredErrorManifold(M1ref,M1est)
        err2 = Similarity.RootMeanSquaredErrorManifold(M2ref,M2est)
        printMe('RMSE',name1,name2,err1,err2)
        err1 = Similarity.MeanErrorManifold(M1ref,M1est)
        err2 = Similarity.MeanErrorManifold(M2ref,M2est)
        printMe('Mean error per point',name1,name2,err1,err2)
        err1 = Similarity.HausdorffDistance(M1ref,M1est)
        err2 = Similarity.HausdorffDistance(M2ref,M2est)
        printMe('Hausdorff distance',name1,name2,err1,err2)

    # Sugihara method
    est1,est2=CCM.crossMap(ts1,ts2,numlags,lagsize,Weights.makeExpWeights)
    M1Sug=SSR.makeShadowManifold(est1,numlags,lagsize)
    M2Sug=SSR.makeShadowManifold(est2,numlags,lagsize)
    calcErrs("Sugihara method with reconstruction:",M1Sug,M2Sug)
    # weighted sums in the embedding space
    M1us,M2us=CCMAlternatives.crossMapModified1(M1,M2,Weights.makeExpWeights)
    calcErrs("Direct estimation of the manifold using exponential weights:",M1us[corr:,:],M2us[corr:,:])
    M1us,M2us=CCMAlternatives.crossMapModified1(M1,M2,Weights.makeUniformWeights)
    calcErrs("Direct estimation of the manifold using uniform weights:",M1us[corr:,:],M2us[corr:,:])
    M1us,M2us=CCMAlternatives.crossMapModified1(M1,M2,Weights.makeLambdaWeights)
    calcErrs("Direct estimation of the manifold using weights from powers of 1/2:",M1us[corr:,:],M2us[corr:,:])

def sequenceOfReconstructions(names,numlags,lagsize,timeseries,compind1,compind2,listoflens,numiters):

    ts1 = timeseries[:,compind1]
    ts2 = timeseries[:,compind2]
    corr = (numlags-1)*lagsize #correction term for shadow manifold creation

    allstartinds = []
    for l in listoflens:
        allstartinds.append(random.sample(range(len(ts1)-l),numiters))

    def printResults(lol,summary,notes,shorts,avgs1,stds1,avgs2,stds2,name1,name2,printstd=0):
        print('#####################################################################')
        print(summary)
        print("Lengths: {0!s}".format(lol))
        for j,note in enumerate(notes):
            print("    {0} between {1} and {1}': ".format(note,name1) + ' '.join(["{0:0.6f}".format(_i) for _i in [avgs1[_k][j] for _k in range(len(avgs1))]]))
            if printstd:
                print("    Standard deviations for {0} {1} and {1}': ".format(shorts[j],name1) + ' '.join(["{0:0.6f}".format(_i) for _i in [stds1[_k][j] for _k in range(len(stds1))]]))
            print("    {0} between {1} and {1}': ".format(note,name2) + ' '.join(["{0:0.6f}".format(_i) for _i in [avgs2[_k][j] for _k in range(len(avgs2))]]))
            if printstd:
                print("    Standard deviations for {0} {1} and {1}': ".format(shorts[j],name2) + ' '.join(["{0:0.6f}".format(_i) for _i in [stds2[_k][j] for _k in range(len(stds2))]]))
    
    def calcSequence(method,wgtfunc,simMeasure,summary,notes,shorts,name1='M{0}'.format(names[compind1]),name2='M{0}'.format(names[compind2]),printstd=0):
        lol,avgs1,avgs2,stds1,stds2 = CCM.causalityWrapper(ts1,ts2,numlags,lagsize,listoflens,numiters,allstartinds,causalitytester=method,morefunctions={'wgtfunc':wgtfunc,'simMeasure':simMeasure})
        printResults(lol,summary,notes,shorts,avgs1,stds1,avgs2,stds2,name1,name2,printstd)

    #############################################
    # Sugihara method with corr coeff
    calcSequence(CCM.testCausality,Weights.makeExpWeights,[Similarity.corrCoeffPearson,Similarity.RootMeanSquaredErrorTS],"Sugihara method:",["Mean correlation coefficients","Mean time series RMSE"], ["CC","RMSE"],names[compind1],names[compind2])
    #############################################
    # Sugihara with intermediate reconstruction
    calcSequence(CCM.testCausalityReconstruction,Weights.makeExpWeights,[Similarity.RootMeanSquaredErrorManifold,Similarity.MeanErrorManifold,Similarity.HausdorffDistance],"Sugihara method with intermediate reconstruction:",["Mean RMSE","Mean error per point","Mean Hausdorff distance"],["RMSE","ME","HD"])
    #############################################
    # exponential weighted sum in the embedding space
    calcSequence(CCMAlternatives.testCausalityReconstruction,Weights.makeExpWeights,[Similarity.RootMeanSquaredErrorManifold,Similarity.MeanErrorManifold,Similarity.HausdorffDistance],"Direct estimation of the manifold using exponential weights:",["Mean RMSE","Mean error per point","Mean Hausdorff distance"],["RMSE","ME","HD"])
    #############################################
    # uniform weighted sum in the embedding space
    calcSequence(CCMAlternatives.testCausalityReconstruction,Weights.makeUniformWeights,[Similarity.RootMeanSquaredErrorManifold,Similarity.MeanErrorManifold,Similarity.HausdorffDistance],"Direct estimation of the manifold using uniform weights:",["Mean RMSE","Mean error per point","Mean Hausdorff distance"],["RMSE","ME","HD"])
    #############################################
    # weighted sum using powers of 2 in the embedding space
    calcSequence(CCMAlternatives.testCausalityReconstruction,Weights.makeLambdaWeights,[Similarity.RootMeanSquaredErrorManifold,Similarity.MeanErrorManifold,Similarity.HausdorffDistance],"Direct estimation of the manifold using weights made from powers of 1/2:",["Mean RMSE","Mean error per point","Mean Hausdorff distance"],["RMSE","ME","HD"])

if __name__=='__main__':
    # make a time series
    dt = 0.025
    eqns,names,numlags,lagsize,timeseries = doublependulumTS(350,dt=dt)

    # truncate time series if desired
    startind = 0#500 #how much to cut off the front
    endind = len(timeseries) #how much to leave at the back
    ts = timeseries[startind:endind,:] 

    # comparison variables
    compind1 = 2
    compind2 = 3

    # parameters for a sequence of measurements of manifolds of lengths in listoflens with numiters different starting locations (only needed for sequenceOfReconstructions)
    listoflens = range(200,400,100)#range(1000,22600,1000)
    numiters = 25

    # print info about the analysis to be done.
    print('{0} with lagsize of {1!s}*dt with dt = {2!s} and reconstruction dimension {3!s}.'.format(eqns,lagsize,dt,numlags))
    print('If looking at a sequence of measurements, the lengths range from {0!s} to {1!s} and the number of iterations per length is {2!s}.'.format(listoflens[0],listoflens[-1],numiters))

    # run the analysis
    sequenceOfReconstructions(names,numlags,lagsize,ts,compind1,compind2,listoflens,numiters)
    wholeManifoldComparison(names,numlags,lagsize,ts,compind1,compind2)











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

