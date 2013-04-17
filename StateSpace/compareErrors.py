import numpy as np
import CCM, CCMAlternatives, Similarity, Weights
import StateSpaceReconstruction as SSR
import StateSpaceReconstructionPlots as SSRPlots
# #Make a time series
from LorenzEqns import solveLorenz
timeseries = solveLorenz([1.0,0.5,0.5],80.0)
# from differenceEqns import solve2Species
# timeseries = solve2Species([0.4,0.2],80.0)
# from DoublePendulum import solvePendulum
# timeseries = solvePendulum([1.0,2.0,3.0,2.0],300.0)
# quantities needed for the different methods
numlags=2
lagsize=8#40 
compind1 = 0
compind2 = 1#3
endind=2000#len(timeseries) 
corr = (numlags-1)*lagsize
M1=SSR.makeShadowManifold(timeseries[:endind,compind1],numlags,lagsize)
M2=SSR.makeShadowManifold(timeseries[:endind,compind2],numlags,lagsize)
#function defs
def calcErrs(M1est,M2est,method,M1ref=M1[corr:,:],M2ref=M2[corr:,:]):
    err1 = method(M1ref,M1est)
    err2 = method(M2ref,M2est)
    return err1,err2

def printMe(method,err1,err2):
    print("    "+ method+" between Mx and estimated Mx is " + str(err1))
    print("    "+ method+" between My and estimated My is " + str(err2))

# Sugihara method
est1,est2=CCM.crossMap(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize,Weights.makeExpWeights)
M1Sug=SSR.makeShadowManifold(est1,numlags,lagsize)
M2Sug=SSR.makeShadowManifold(est2,numlags,lagsize)
M1SugRMSE, M2SugRMSE = calcErrs(M1Sug,M2Sug,Similarity.RootMeanSquaredError)
M1SugHD, M2SugHD = calcErrs(M1Sug,M2Sug,Similarity.HausdorffDistance)
M1SugCM, M2SugCM = calcErrs(M1Sug,M2Sug,Similarity.countingMeasure)
print("Sugihara method:")
printMe("RMSE",M1SugRMSE,M2SugRMSE)
printMe("Hausdorff dist",M1SugHD,M2SugHD)
printMe("Counting measure",M1SugCM,M2SugCM)
# weighted sum in the embedding space
M1us1,M2us1=CCMAlternatives.crossMapModified1(M1,M2,Weights.makeExpWeights)
M1us1RMSE, M2us1RMSE = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.RootMeanSquaredError)
M1us1HD, M2us1HD = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.HausdorffDistance)
M1us1CM, M2us1CM = calcErrs(M1us1[corr:,:],M2us1[corr:,:],Similarity.countingMeasure)
print("Our method 1, estimate the manifold:")
printMe("RMSE",M1us1RMSE,M2us1RMSE)
printMe("Hausdorff dist",M1us1HD,M2us1HD)
printMe("Counting measure",M1us1CM,M2us1CM)
# average over the different estimates of a point in time
est1,est2=CCMAlternatives.crossMapModified2(M1,M2,Weights.makeExpWeights)
M1us2=SSR.makeShadowManifold(est1,numlags,lagsize)
M2us2=SSR.makeShadowManifold(est2,numlags,lagsize)
M1us2RMSE, M2us2RMSE = calcErrs(M1us2,M2us2,Similarity.RootMeanSquaredError)
M1us2HD, M2us2HD = calcErrs(M1us2,M2us2,Similarity.HausdorffDistance)
M1us2CM, M2us2CM = calcErrs(M1us2,M2us2,Similarity.countingMeasure)
print("Our method 2, average the different time series estimations:")
printMe("RMSE",M1us2RMSE,M2us2RMSE)
printMe("Hausdorff dist",M1us2HD,M2us2HD)
printMe("Counting measure",M1us2CM,M2us2CM)
# take a different projection than Sugihara
proj = numlags - 1 
est1,est2=CCMAlternatives.crossMapModified3(M1,M2,proj,Weights.makeExpWeights)
M1us3=SSR.makeShadowManifold(est1,numlags,lagsize)
M2us3=SSR.makeShadowManifold(est2,numlags,lagsize)
M1us3RMSE, M2us3RMSE = calcErrs(M1us3,M2us3,Similarity.RootMeanSquaredError,M1[corr-(numlags-proj):-(numlags-proj),:],M2[corr-(numlags-proj):-(numlags-proj),:])
M1us3HD, M2us3HD = calcErrs(M1us3,M2us3,Similarity.HausdorffDistance,M1[corr-(numlags-proj):-(numlags-proj),:],M2[corr-(numlags-proj):-(numlags-proj),:])
M1us3CM, M2us3CM = calcErrs(M1us3,M2us3,Similarity.countingMeasure,M1[corr-(numlags-proj):-(numlags-proj),:],M2[corr-(numlags-proj):-(numlags-proj),:])
print("Our method 3, take a different projection (index "+ str(proj) +"):")
printMe("RMSE",M1us3RMSE,M2us3RMSE)
printMe("Hausdorff dist",M1us3HD,M2us3HD)
printMe("Counting measure",M1us3CM,M2us3CM)

# plot the shadow manifolds and their estimates
SSRPlots.plotEstShadowManifoldSugihara(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)     
SSRPlots.plotEstShadowManifoldUs1(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)        
SSRPlots.plotEstShadowManifoldUs2(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)        
SSRPlots.plotEstShadowManifoldUs3(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize,proj)        

