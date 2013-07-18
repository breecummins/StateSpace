import numpy as np
import CCM, CCMAlternatives, Similarity, Weights
import StateSpaceReconstruction as SSR
from CircleExample import getLagDim
# import StateSpaceReconstructionPlots as SSRPlots
import random, sys

def lorenzTS(finaltime=80.0,dt=0.01):
    from LorenzEqns import solveLorenz
    timeseries = solveLorenz([1.0,0.5,0.5],finaltime,dt)
    eqns = 'Lorenz'
    names = ['x','y','z']
    return eqns,names,timeseries

def doublependulumTS(finaltime=2400.0,dt=0.025):
    from DoublePendulum import solvePendulum
    timeseries = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt)
    eqns = 'Double pendulum'
    names = ['x','y','z','w']
    return eqns,names,timeseries

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
        sys.stdout.flush()

    def calcErrs(summary,M1est,M2est,M1ref=M1[corr:,:],M2ref=M2[corr:,:],name1='M{0}'.format(names[compind1]),name2='M{0}'.format(names[compind2])):
        print('############################################################################')
        print(summary)
        sys.stdout.flush()
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

def sequenceOfReconstructions(names,numlags,lagsize,timeseries,compind1,compind2,listoflens,numiters,allstartinds,growtraj):

    ts1 = timeseries[:,compind1]
    ts2 = timeseries[:,compind2]
    corr = (numlags-1)*lagsize #correction term for shadow manifold creation

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
        sys.stdout.flush()

    def calcSequence(method,wgtfunc,simMeasure,summary,notes,shorts,name1='M{0}'.format(names[compind1]),name2='M{0}'.format(names[compind2]),printstd=0):
        lol,avgs1,avgs2,stds1,stds2 = CCM.causalityWrapper(ts1,ts2,numlags,lagsize,listoflens,numiters,allstartinds,growtraj,causalitytester=method,morefunctions={'wgtfunc':wgtfunc,'simMeasure':simMeasure})
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

def sequenceOfDiffeomorphismChecks(names,numlags,lagsize,timeseries,compind1,compind2,listoflens,numiters,allstartinds,listofskips):

    ts1 = timeseries[:,compind1]
    ts2 = timeseries[:,compind2]
    corr = (numlags-1)*lagsize #correction term for shadow manifold creation

    poi_whole = random.sample(range(len(ts1)-corr),min(listoflens)-corr)
    poi = range(min(listoflens)-corr)

    def printResults(lol,summary,notes,shorts,avgs1,stds1,avgs2,stds2,name1='M{0}'.format(names[compind1]),name2='M{0}'.format(names[compind2]),printstd=0,skips=0):
        print('#####################################################################')
        print(summary)
        if not skips:
            print("Lengths: {0!s}".format(lol))
        else:
            print("Skips: {0!s}".format(lol))
        try:
            for j,note in enumerate(notes):
                print("    {0} between {1} and {2}: ".format(note,name1,name2) + ' '.join(["{0:0.6f}".format(_i) for _i in [avgs1[_k][j] for _k in range(len(avgs1))]]))
                if printstd:
                    print("    Standard deviations for {0} {1} and {2}: ".format(shorts[j],name1,name2) + ' '.join(["{0:0.6f}".format(_i) for _i in [stds1[_k][j] for _k in range(len(stds1))]]))
                print("    {0} between {1} and {2}: ".format(note,name2,name1) + ' '.join(["{0:0.6f}".format(_i) for _i in [avgs2[_k][j] for _k in range(len(avgs2))]]))
                if printstd:
                    print("    Standard deviations for {0} {1} and {2}: ".format(shorts[j],name2,name1) + ' '.join(["{0:0.6f}".format(_i) for _i in [stds2[_k][j] for _k in range(len(stds2))]]))
        except:
            for j,note in enumerate(notes):
                print("    {0} between {1} and {2}: ".format(note,name1,name2) + ' '.join(["{0:0.6f}".format(_i) for _i in avgs1[j::len(notes)]]))
                if printstd:
                    print("    Standard deviations for {0} {1} and {2}: ".format(shorts[j],name1,name2) + ' '.join(["{0:0.6f}".format(_i) for _i in stds1[j::len(notes)]]))
                print("    {0} between {1} and {2}: ".format(note,name2,name1) + ' '.join(["{0:0.6f}".format(_i) for _i in avgs2[j::len(notes)]]))
                if printstd:
                    print("    Standard deviations for {0} {1} and {2}: ".format(shorts[j],name2,name1) + ' '.join(["{0:0.6f}".format(_i) for _i in stds2[j::len(notes)]]))
        sys.stdout.flush()

    lol,avgs1,avgs2,stds1,stds2 = CCMAlternatives.testDiffeomorphism(ts1,ts2,numlags,lagsize,listoflens,numiters,allstartinds,simMeasure=[Similarity.neighborDistance,Similarity.countingMeasure],N=numlags+1)
    printResults(lol,"Comparing manifolds directly (no estimation) with random, changing starting positions for subintervals",["Neighbor distance","Counting measure"],["ND","CM"],avgs1,stds1,avgs2,stds2)
    lol,avgs1,avgs2 = CCMAlternatives.testDiffeomorphismSamePoints(ts1,ts2,numlags,lagsize,listoflens,simMeasure=[Similarity.maxNeighborDistMean,Similarity.meanNeighborDist],N=numlags+1,poi=poi)
    printResults(lol,"Comparing manifolds directly (no estimation) with fixed starting positions for subintervals near the beginning of the series",["Max neighbor distance","Mean neighbor dist"],["MND","mND"],avgs1,None,avgs2,None)
    lol,avgs1,avgs2 = CCMAlternatives.testDiffeomorphismSamePointsFillIn(ts1,ts2,numlags,lagsize,listofskips,simMeasure=[Similarity.maxNeighborDistMeanWithSkip,Similarity.meanNeighborDistWithSkip],N=numlags+1,poi=poi_whole)
    printResults(lol,"Comparing manifolds directly (no estimation) with fixed starting positions for subintervals over the whole series",["Max neighbor distance with skip","Mean neighbor dist with skip"],["MNDws","mNDws"],avgs1,None,avgs2,None)



if __name__=='__main__':
    # make a time series
    dt = 0.1#0.025
    finaltime = 1200.0
    eqns,names,timeseries = doublependulumTS(finaltime,dt)
    
    # truncate time series if desired
    startind = int(50/dt)#2000 #how much to cut off the front
    timeseries = timeseries[startind:,:]
    
    # comparison variables
    # compind1 = 2
    compind1 = 0
    compind2 = 3

    # # get the lagsize and number of lags to construct the shadow manifold
    # lagsize,numlags = getLagDim(timeseries,cols=[compind1,compind2],dims=12)
    # print('lagsize = {0}, numlags = {1}'.format(lagsize,numlags))
    # sys.stdout.flush() #Forces immediate print to screen. Useful if dumping long analysis to text file.
    lagsize = 5 #need to try 11 also, did getLagDim before running full script
    numlags = 28

    # subsample time series according to lagsize
    # this will analyze the subsequence of multiples of lagsize*dt in the time series
    ts = timeseries[::lagsize,:]
    newlagsize = 1

    # parameters for a sequence of measurements of manifolds of lengths in listoflens with numiters different starting locations (only needed for sequenceOfReconstructions)
    listoflens = range(200,1300,200)
    listofskips = [2**n for n in range(4,-1,-1)]
    numiters = 10
    allstartinds = []
    for l in listoflens:
        allstartinds.append(random.sample(range(ts.shape[0]-l),numiters))

    # print info about the analysis to be done.
    print('{0} with lagsize of {1}*dt with dt = {2} and reconstruction dimension {3} using only times located at multiples of lagsize*dt.'.format(eqns,lagsize,dt,numlags))
    print('If looking at a sequence of measurements, the lengths range from {0} to {1} and the number of iterations per length is {2}.'.format(listoflens[0],listoflens[-1],numiters))
    sys.stdout.flush() #Forces immediate print to screen. Useful if dumping long analysis to text file.

    # run the analysis
    print('#####################################################################')
    print("Whole manifold checks between M{0} and M{0}' and M{1} and M{1}'.".format(names[compind1],names[compind2]))
    wholeManifoldComparison(names,numlags,newlagsize,ts,compind1,compind2)

    print('#####################################################################')
    print("Convergence checks between original quantities and estimates (M{0} and M{0}', M{1} and M{1}', {0} and {0}', and {1} and {1}') with random starting positions at each subinterval length.".format(names[compind1],names[compind2]))
    sequenceOfReconstructions(names,numlags,newlagsize,ts,compind1,compind2,listoflens,numiters,allstartinds,0)

    print('#####################################################################')
    print("Convergence checks between original quantities and estimates (M{0} and M{0}', M{1} and M{1}', {0} and {0}', and {1} and {1}') with fixed starting positions for all subinterval lengths.".format(names[compind1],names[compind2]))
    sequenceOfReconstructions(names,numlags,newlagsize,ts,compind1,compind2,listoflens,numiters,allstartinds,1)

    print('#####################################################################')
    print('Convergence checks between M{0} and M{1} directly.'.format(names[compind1],names[compind2]))
    sequenceOfDiffeomorphismChecks(names,numlags,newlagsize,ts,compind1,compind2,listoflens,numiters,allstartinds,listofskips)










    # # # average over the different estimates of a point in time
    # # est1,est2=CCMAlternatives.crossMapModified2(M1,M2,Weights.makeExpWeights)
    # # M1us2=SSR.makeShadowManifold(est1,numlags,lagsize)
    # # M2us2=SSR.makeShadowManifold(est2,numlags,lagsize)
    # # M1us2RMSE, M2us2RMSE = calcErrs(M1us2,M2us2,Similarity.RootMeanSquaredError)
    # # M1us2HD, M2us2HD = calcErrs(M1us2,M2us2,Similarity.HausdorffDistance)
    # # M1us2CM, M2us2CM = calcErrs(M1us2,M2us2,Similarity.countingMeasure)
    # # print("Our method 2, average the different time series estimations:")
    # # printMe("RMSE",M1us2RMSE,M2us2RMSE)
    # # printMe("Hausdorff dist",M1us2HD,M2us2HD)
    # # printMe("Counting measure",M1us2CM,M2us2CM)
    # # # take a different projection than Sugihara
    # # proj = numlags - 1 
    # # est1,est2=CCMAlternatives.crossMapModified3(M1,M2,proj,Weights.makeExpWeights)
    # # M1us3=SSR.makeShadowManifold(est1,numlags,lagsize)
    # # M2us3=SSR.makeShadowManifold(est2,numlags,lagsize)
    # # M1us3RMSE, M2us3RMSE = calcErrs(M1us3,M2us3,Similarity.RootMeanSquaredError,M1[corr-proj:-proj,:],M2[corr-proj:-proj,:])
    # # M1us3HD, M2us3HD = calcErrs(M1us3,M2us3,Similarity.HausdorffDistance,M1[corr-proj:-proj,:],M2[corr-proj:-proj,:])
    # # M1us3CM, M2us3CM = calcErrs(M1us3,M2us3,Similarity.countingMeasure,M1[corr-proj:-proj,:],M2[corr-proj:-proj,:])
    # # print("Our method 3, take a different projection (index "+ str(proj) +"):")
    # # printMe("RMSE",M1us3RMSE,M2us3RMSE)
    # # printMe("Hausdorff dist",M1us3HD,M2us3HD)
    # # printMe("Counting measure",M1us3CM,M2us3CM)

    # # # plot the shadow manifolds and their estimates
    # # SSRPlots.plotEstShadowManifoldSugihara(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)     
    # # SSRPlots.plotEstShadowManifoldUs1(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)        
    # # SSRPlots.plotEstShadowManifoldUs2(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)        
    # # SSRPlots.plotEstShadowManifoldUs3(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize,proj)        

