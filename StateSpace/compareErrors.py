import numpy as np
import CCM, Similarity, Weights
import StateSpaceReconstruction as SSR
import StateSpaceReconstructionPlots as SSRPlots

#FIXME - refactor SSR

# from LorenzEqns import solveLorenz
# timeseries = solveLorenz([1.0,0.5,0.5],80.0)
# from differenceEqns import solve2Species
# timeseries = solve2Species([0.4,0.2],80.0)
from DoublePendulum import solvePendulum
timeseries = solvePendulum([1.0,2.0,3.0,2.0],300.0)
numlags=2#3
lagsize=40#8 
compind1 = 0
compind2 = 3 #1
endind=len(timeseries)#2000 
M1=np.array(list(SSR.makeShadowManifold(timeseries[:endind,compind1],numlags,lagsize)))
M2=np.array(list(SSR.makeShadowManifold(timeseries[:endind,compind2],numlags,lagsize)))
est1,est2=CCM.crossMap(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize,Weights.makeExpWeights)
M1Sug=np.array(list(SSR.makeShadowManifold(est1,numlags,lagsize)))
M2Sug=np.array(list(SSR.makeShadowManifold(est2,numlags,lagsize)))
corr = (numlags-1)*lagsize
l = M1Sug.shape[0]
M1SugErr = Similarity.RootMeanSquaredError(M1[corr:corr+l,:],M1Sug)
M2SugErr = Similarity.RootMeanSquaredError(M2[corr:corr+l,:],M2Sug)
print("Sugihara method RMSE:")
print("    RMSE between Mx and estimated Mx is " + str(M1SugErr))
print("    RMSE between My and estimated My is " + str(M2SugErr))
M1us1,M2us1=CCM.crossMapModified1(M1,M2,Weights.makeExpWeights)
print("Our method 1 RMSE:")
print("    RMSE between Mx and estimated Mx is " + str(Similarity.RootMeanSquaredError(M1,M1us1)))
print("    RMSE between My and estimated My is " + str(Similarity.RootMeanSquaredError(M2,M2us1)))    
est1,est2=CCM.crossMapModified2(M1,M2,Weights.makeExpWeights)
M1us2=np.array(list(SSR.makeShadowManifold(est1,numlags,lagsize)))
M2us2=np.array(list(SSR.makeShadowManifold(est2,numlags,lagsize)))
corr = (numlags-1)*lagsize
l = M1us2.shape[0]
M1err2 = Similarity.RootMeanSquaredError(M1[corr:corr+l,:],M1us2)
M2err2 = Similarity.RootMeanSquaredError(M2[corr:corr+l,:],M2us2)
print("Our method 2 RMSE:")
print("    RMSE between Mx and estimated Mx is " + str(M1err2))
print("    RMSE between My and estimated My is " + str(M2err2))   
proj = 0#numlags - 3 
est1,est2=CCM.crossMapModified3(M1,M2,proj,Weights.makeExpWeights)
M1us3=np.array(list(SSR.makeShadowManifold(est1,numlags,lagsize)))
M2us3=np.array(list(SSR.makeShadowManifold(est2,numlags,lagsize)))
corr = (numlags-1)*lagsize
l = M1us3.shape[0]
M1err3 = Similarity.RootMeanSquaredError(M1[corr-(numlags-proj):corr+l-(numlags-proj),:],M1us3)
M2err3 = Similarity.RootMeanSquaredError(M2[corr-(numlags-proj):corr+l-(numlags-proj),:],M2us3)
print("Our method 3 RMSE:")
print("    RMSE between Mx and estimated Mx is " + str(M1err3))
print("    RMSE between My and estimated My is " + str(M2err3))    


SSRPlots.plotEstShadowManifoldSugihara(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)     
SSRPlots.plotEstShadowManifoldUs1(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)        
SSRPlots.plotEstShadowManifoldUs2(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize)        
SSRPlots.plotEstShadowManifoldUs3(timeseries[:endind,compind1],timeseries[:endind,compind2],numlags,lagsize,proj)        

