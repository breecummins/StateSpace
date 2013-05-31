#!/usr/bin/env python

import numpy as np
import CCM, CCMAlternatives, Similarity, Weights
import StateSpaceReconstruction as SSR
# import StateSpaceReconstructionPlots as SSRPlots
import random
# #Make a time series
# from LorenzEqns import solveLorenz
# timeseries = solveLorenz([1.0,0.5,0.5],40.0)
# from differenceEqns import solve2Species
# timeseries = solve2Species([0.4,0.2],80.0)
from DoublePendulum import solvePendulum
dt = 0.1
timeseries = solvePendulum([1.0,2.0,3.0,2.0],300.0,dt=dt)
# quantities needed for the different methods
eqns = 'Double pendulum'
numlags=4
lagsize=8 
compind1 = 0
name1 = 'x'
compind2 = 3
name2 = 'y'
startind = 0
endind=len(timeseries)
corr = (numlags-1)*lagsize
M1=SSR.makeShadowManifold(timeseries[startind:endind,compind1],numlags,lagsize)
M2=SSR.makeShadowManifold(timeseries[startind:endind,compind2],numlags,lagsize)
#function defs
def calcErrs(M1est,M2est,method,M1ref=M1[corr:,:],M2ref=M2[corr:,:]):
    err1 = method(M1ref,M1est)
    err2 = method(M2ref,M2est)
    return err1,err2

def printMe(method,err1,err2,name1=name1,name2=name2):
    print("    "+ method+" between M"+name1+" and M"+name1+"' is " + str(err1))
    print("    "+ method+" between M"+name2+" and M"+name2+"' is " + str(err2))

print(eqns+' with lagsize of '+str(lagsize)+'*dt with dt = '+str(dt)+' and reconstruction dimension '+str(numlags))

# # Measure whole manifold

# # Sugihara method
# est1,est2=CCM.crossMap(timeseries[startind:endind,compind1],timeseries[startind:endind,compind2],numlags,lagsize,Weights.makeExpWeights)
# M1Sug=SSR.makeShadowManifold(est1,numlags,lagsize)
# M2Sug=SSR.makeShadowManifold(est2,numlags,lagsize)
# M1SugRMSE, M2SugRMSE = calcErrs(M1Sug,M2Sug,Similarity.RootMeanSquaredErrorManifold)
# M1SugHD, M2SugHD = calcErrs(M1Sug,M2Sug,Similarity.HausdorffDistance)
# M1SugME, M2SugME = calcErrs(M1Sug,M2Sug,Similarity.MeanErrorManifold)
# print("Sugihara method:")
# printMe("RMSE",M1SugRMSE,M2SugRMSE)
# printMe("Mean error per point",M1SugME,M2SugME)
# printMe("Hausdorff dist",M1SugHD,M2SugHD)
# # weighted sums in the embedding space
# M1us1,M2us1=CCMAlternatives.crossMapModified1(M1,M2,Weights.makeExpWeights)
# M1us1RMSE, M2us1RMSE = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.RootMeanSquaredErrorManifold)
# M1us1HD, M2us1HD = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.HausdorffDistance)
# M1us1ME, M2us1ME = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.MeanErrorManifold)
# print("Direct estimation of the manifold using exponential weights:")
# printMe("RMSE",M1us1RMSE,M2us1RMSE)
# printMe("Mean error per point",M1us1ME,M2us1ME)
# printMe("Hausdorff dist",M1us1HD,M2us1HD)
# M1us1,M2us1=CCMAlternatives.crossMapModified1(M1,M2,Weights.makeUniformWeights)
# M1us1RMSE, M2us1RMSE = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.RootMeanSquaredErrorManifold)
# M1us1HD, M2us1HD = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.HausdorffDistance)
# M1us1ME, M2us1ME = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.MeanErrorManifold)
# print("Direct estimation of the manifold using uniform weights:")
# printMe("RMSE",M1us1RMSE,M2us1RMSE)
# printMe("Mean error per point",M1us1ME,M2us1ME)
# printMe("Hausdorff dist",M1us1HD,M2us1HD)
# M1us1,M2us1=CCMAlternatives.crossMapModified1(M1,M2,Weights.makeLambdaWeights)
# M1us1RMSE, M2us1RMSE = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.RootMeanSquaredErrorManifold)
# M1us1HD, M2us1HD = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.HausdorffDistance)
# M1us1ME, M2us1ME = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.MeanErrorManifold)
# print("Direct estimation of the manifold using weights from powers of 1/2:")
# printMe("RMSE",M1us1RMSE,M2us1RMSE)
# printMe("Mean error per point",M1us1ME,M2us1ME)
# printMe("Hausdorff dist",M1us1HD,M2us1HD)

# Construct a series of manifolds

listoflens = range(400,3000,400)
numiters = 25

def makeSeries(wgtfunc,simMeasure,ts1=timeseries[startind:endind,compind1],ts2=timeseries[startind:endind,compind2]):
    lol,avg1,avg2,std1,std2 = CCMAlternatives.testCausalityReconstruction(ts1,ts2,numlags,lagsize,listoflens,numiters,wgtfunc=wgtfunc,simMeasure=simMeasure)
    print("Mean error per point between " + name1 + " and " + name1 +"': " + str(["{0:0.6f}".format(i) for i in avg1]))
    print("Standard deviations for ME" + name1 + " and " + name1 +"': " + str(["{0:0.6f}".format(i) for i in std1]))
    print("Mean error per point between " + name2 + " and " + name2 +"': " + str(["{0:0.6f}".format(i) for i in avg2]))
    print("Standard deviations for ME" + name2 + " and " + name2 +"': " + str(["{0:0.6f}".format(i) for i in std2]))

# Sugihara method, full
lol,avg1,avg2,std1,std2 = CCM.testCausality(timeseries[startind:endind,compind1],timeseries[startind:endind,compind2],numlags,lagsize,listoflens,numiters,Weights.makeExpWeights)
print("Sugihara method with correlation coefficient:")
print("Lengths: " + str(lol))
print("Mean correlation coefficients between " + name1 + " and " + name1 +"': " + str(["{0:0.6f}".format(i) for i in avg1]))
print("Standard deviations for " + name1 + " and " + name1 +"': " + str(["{0:0.6f}".format(i) for i in std1]))
print("Mean correlation coefficients between " + name2 + " and " + name2 +"': " + str(["{0:0.6f}".format(i) for i in avg2]))
print("Standard deviations for " + name2 + " and " + name2 +"': " + str(["{0:0.6f}".format(i) for i in std2]))
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
        est1,est2=CCM.crossMap(timeseries[s:s+l,compind1],timeseries[s:s+l,compind2],numlags,lagsize,Weights.makeExpWeights)
        M1Sug=SSR.makeShadowManifold(est1,numlags,lagsize)
        M2Sug=SSR.makeShadowManifold(est2,numlags,lagsize)
        M1SugRMSE, M2SugRMSE = calcErrs(M1Sug,M2Sug,Similarity.RootMeanSquaredErrorManifold,M1ref=M1[s+2*corr:s+l,:],M2ref=M2[s+2*corr:s+l,:])
        M1SugHD, M2SugHD = calcErrs(M1Sug,M2Sug,Similarity.HausdorffDistance,M1ref=M1[s+2*corr:s+l,:],M2ref=M2[s+2*corr:s+l,:])
        M1SugME, M2SugME = calcErrs(M1Sug,M2Sug,Similarity.MeanErrorManifold,M1ref=M1[s+2*corr:s+l,:],M2ref=M2[s+2*corr:s+l,:])
        a1.append(M1SugRMSE)
        a1.append(M1SugHD)
        a1.append(M1SugME)
        a2.append(M2SugRMSE)
        a2.append(M2SugHD)
        a2.append(M2SugME)
    avgs1.append(np.mean(a1[0::3]))
    avgs1.append(np.mean(a1[1::3]))
    avgs1.append(np.mean(a1[2::3]))
    stds1.append(np.std(a1[0::3]))
    stds1.append(np.std(a1[1::3]))
    stds1.append(np.std(a1[2::3]))
    avgs2.append(np.mean(a2[0::3]))
    avgs2.append(np.mean(a2[1::3]))
    avgs2.append(np.mean(a2[2::3]))
    stds2.append(np.std(a2[0::3]))
    stds2.append(np.std(a2[1::3]))
    stds2.append(np.std(a2[2::3]))
print("Sugihara method with intermediate reconstruction:")
print("Lengths: " + str(listoflens))
print("Mean RMSE between " + name1 + " and " + name1 +"': " + str(["{0:0.6f}".format(i) for i in avgs1[0::3]]))
print("Standard deviations for RMSE" + name1 + " and " + name1 +"': " + str(["{0:0.6f}".format(i) for i in stds1[0::3]]))
print("Mean RMSE between " + name2 + " and " + name2 +"': " + str(["{0:0.6f}".format(i) for i in avgs2[0::3]]))
print("Standard deviations for RMSE" + name2 + " and " + name2 +"': " + str(["{0:0.6f}".format(i) for i in stds2[0::3]]))
print("Mean error per point between " + name1 + " and " + name1 +"': " + str(["{0:0.6f}".format(i) for i in avgs1[2::3]]))
print("Standard deviations for ME" + name1 + " and " + name1 +"': " + str(["{0:0.6f}".format(i) for i in stds1[2::3]]))
print("Mean error per point between " + name2 + " and " + name2 +"': " + str(["{0:0.6f}".format(i) for i in avgs2[2::3]]))
print("Standard deviations for ME" + name2 + " and " + name2 +"': " + str(["{0:0.6f}".format(i) for i in stds2[2::3]]))
print("Mean Hausdorff distance between " + name1 + " and " + name1 +"': " + str(["{0:0.6f}".format(i) for i in avgs1[1::3]]))
print("Standard deviations for HD" + name1 + " and " + name1 +"': " + str(["{0:0.6f}".format(i) for i in stds1[1::3]]))
print("Mean Hausdorff distance between " + name2 + " and " + name2 +"': " + str(["{0:0.6f}".format(i) for i in avgs2[1::3]]))
print("Standard deviations for HD" + name2 + " and " + name2 +"': " + str(["{0:0.6f}".format(i) for i in stds2[1::3]]))
# weighted sums in the embedding space
makeSeries(Weights.makeExpWeights,Similarity.RootMeanSquaredErrorManifold)
print("Direct estimation of the manifold using exponential weights:")
print("Lengths: " + str(listoflens))
makeSeries(Weights.makeExpWeights,Similarity.MeanErrorManifold)
makeSeries(Weights.makeExpWeights,Similarity.HausdorffDistance)
makeSeries(Weights.makeUniformWeights,Similarity.RootMeanSquaredErrorManifold)
print("Direct estimation of the manifold using uniform weights:")
print("Lengths: " + str(listoflens))
makeSeries(Weights.makeUniformWeights,Similarity.MeanErrorManifold)
makeSeries(Weights.makeUniformWeights,Similarity.HausdorffDistance)
makeSeries(Weights.makeLambdaWeights,Similarity.RootMeanSquaredErrorManifold)
print("Direct estimation of the manifold using weights made from powers of 1/2:")
print("Lengths: " + str(listoflens))
makeSeries(Weights.makeLambdaWeights,Similarity.MeanErrorManifold)
makeSeries(Weights.makeLambdaWeights,Similarity.HausdorffDistance)












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

