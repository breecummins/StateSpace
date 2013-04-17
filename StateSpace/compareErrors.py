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
# Sugihara method
est1,est2=CCM.crossMap(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize,Weights.makeExpWeights)
M1Sug=SSR.makeShadowManifold(est1,numlags,lagsize)
M2Sug=SSR.makeShadowManifold(est2,numlags,lagsize)
M1SugRMSE = Similarity.RootMeanSquaredError(M1[corr:,:],M1Sug)
M2SugRMSE = Similarity.RootMeanSquaredError(M2[corr:,:],M2Sug)
M1SugHD = Similarity.HausdorffDistance(M1[corr:,:],M1Sug)
M2SugHD = Similarity.HausdorffDistance(M2[corr:,:],M2Sug)
print("Sugihara method:")
print("    RMSE between Mx and estimated Mx is " + str(M1SugRMSE))
print("    RMSE between My and estimated My is " + str(M2SugRMSE))
print("    Hausdorff dist between Mx and estimated Mx is " + str(M1SugHD))
print("    Hausdorff dist between My and estimated My is " + str(M2SugHD))
# weighted sum in the embedding space
M1us1,M2us1=CCMAlternatives.crossMapModified1(M1,M2,Weights.makeExpWeights)
M1us1RMSE = Similarity.RootMeanSquaredError(M1[corr:,:],M1us1[corr:,:])
M2us1RMSE = Similarity.RootMeanSquaredError(M2[corr:,:],M2us1[corr:,:])
M1us1HD = Similarity.HausdorffDistance(M1[corr:,:],M1us1[corr:,:])
M2us1HD = Similarity.HausdorffDistance(M2[corr:,:],M2us1[corr:,:])
print("Our method 1:")
print("    RMSE between Mx and estimated Mx is " + str(M1us1RMSE))
print("    RMSE between My and estimated My is " + str(M2us1RMSE))
print("    Hausdorff dist between Mx and estimated Mx is " + str(M1us1HD))
print("    Hausdorff dist between My and estimated My is " + str(M2us1HD))
# average over the different estimates of a point in time
est1,est2=CCMAlternatives.crossMapModified2(M1,M2,Weights.makeExpWeights)
M1us2=SSR.makeShadowManifold(est1,numlags,lagsize)
M2us2=SSR.makeShadowManifold(est2,numlags,lagsize)
M1us2RMSE = Similarity.RootMeanSquaredError(M1[corr:,:],M1us2)
M2us2RMSE = Similarity.RootMeanSquaredError(M2[corr:,:],M2us2)
M1us2HD = Similarity.HausdorffDistance(M1[corr:,:],M1us2)
M2us2HD = Similarity.HausdorffDistance(M2[corr:,:],M2us2)
print("Our method 2:")
print("    RMSE between Mx and estimated Mx is " + str(M1us2RMSE))
print("    RMSE between My and estimated My is " + str(M2us2RMSE))
print("    Hausdorff dist between Mx and estimated Mx is " + str(M1us2HD))
print("    Hausdorff dist between My and estimated My is " + str(M2us2HD))
# take a different projection than Sugihara
proj = numlags - 1 
est1,est2=CCMAlternatives.crossMapModified3(M1,M2,proj,Weights.makeExpWeights)
M1us3=SSR.makeShadowManifold(est1,numlags,lagsize)
M2us3=SSR.makeShadowManifold(est2,numlags,lagsize)
M1us3RMSE = Similarity.RootMeanSquaredError(M1[corr-(numlags-proj):-(numlags-proj),:],M1us3)
M2us3RMSE = Similarity.RootMeanSquaredError(M2[corr-(numlags-proj):-(numlags-proj),:],M2us3)
M1us3HD = Similarity.HausdorffDistance(M1[corr-(numlags-proj):-(numlags-proj),:],M1us3)
M2us3HD = Similarity.HausdorffDistance(M2[corr-(numlags-proj):-(numlags-proj),:],M2us3)
print("Our method 3:")
print("    RMSE between Mx and estimated Mx is " + str(M1us3RMSE))
print("    RMSE between My and estimated My is " + str(M2us3RMSE))
print("    Hausdorff dist between Mx and estimated Mx is " + str(M1us3HD))
print("    Hausdorff dist between My and estimated My is " + str(M2us3HD))

# plot the shadow manifolds and their estimates
SSRPlots.plotEstShadowManifoldSugihara(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)     
SSRPlots.plotEstShadowManifoldUs1(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)        
SSRPlots.plotEstShadowManifoldUs2(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)        
SSRPlots.plotEstShadowManifoldUs3(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize,proj)        

