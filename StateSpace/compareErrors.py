import numpy as np
import CCM, Similarity, Weights
import StateSpaceReconstruction as SSR
import StateSpaceReconstructionPlots as SSRPlots
from LorenzEqns import solveLorenz
timeseries = solveLorenz([1.0,0.5,0.5],80.0)
# from differenceEqns import solve2Species
# timeseries = solve2Species([0.4,0.2],80.0)
numlags=2#3
lagsize=8 
endind=len(timeseries)#2000 
M1=np.array(list(SSR.makeShadowManifold(timeseries[:endind,0],numlags,lagsize)))
M2=np.array(list(SSR.makeShadowManifold(timeseries[:endind,1],numlags,lagsize)))
est1,est2=CCM.crossMap(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize,Weights.makeExpWeights)
M1Sug=np.array(list(SSR.makeShadowManifold(est1,numlags,lagsize)))
M2Sug=np.array(list(SSR.makeShadowManifold(est2,numlags,lagsize)))
corr = (numlags-1)*lagsize
l = M1Sug.shape[0]
M1SugErr = Similarity.L2error(M1[corr:corr+l,:],M1Sug)
M2SugErr = Similarity.L2error(M2[corr:corr+l,:],M2Sug)
print("Sugihara method L2 error:")
print("    L2 error between Mx and estimated Mx is " + str(M1SugErr))
print("    L2 error between My and estimated My is " + str(M2SugErr))
M1us1,M2us1=CCM.crossMapModified1(M1,M2,Weights.makeExpWeights)
print("Our method 1 L2 error:")
print("    L2 error between Mx and estimated Mx is " + str(Similarity.L2error(M1,M1us1)))
print("    L2 error between My and estimated My is " + str(Similarity.L2error(M2,M2us1)))    
est1,est2=CCM.crossMapModified2(M1,M2,Weights.makeExpWeights)
M1us2=np.array(list(SSR.makeShadowManifold(est1,numlags,lagsize)))
M2us2=np.array(list(SSR.makeShadowManifold(est2,numlags,lagsize)))
corr = (numlags-1)*lagsize
l = M1us2.shape[0]
M1err2 = Similarity.L2error(M1[corr:corr+l,:],M1us2)
M2err2 = Similarity.L2error(M2[corr:corr+l,:],M2us2)
print("Our method 2 L2 error:")
print("    L2 error between Mx and estimated Mx is " + str(M1err2))
print("    L2 error between My and estimated My is " + str(M2err2))   
proj = 0#numlags - 3 
est1,est2=CCM.crossMapModified3(M1,M2,proj,Weights.makeExpWeights)
M1us3=np.array(list(SSR.makeShadowManifold(est1,numlags,lagsize)))
M2us3=np.array(list(SSR.makeShadowManifold(est2,numlags,lagsize)))
corr = (numlags-1)*lagsize
l = M1us3.shape[0]
M1err3 = Similarity.L2error(M1[corr-(numlags-proj):corr+l-(numlags-proj),:],M1us3)
M2err3 = Similarity.L2error(M2[corr-(numlags-proj):corr+l-(numlags-proj),:],M2us3)
print("Our method 3 L2 error:")
print("    L2 error between Mx and estimated Mx is " + str(M1err3))
print("    L2 error between My and estimated My is " + str(M2err3))    


SSRPlots.plotEstShadowManifoldSugihara(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize)     
SSRPlots.plotEstShadowManifoldUs1(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize)        
SSRPlots.plotEstShadowManifoldUs2(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize)        
SSRPlots.plotEstShadowManifoldUs3(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize,proj)        

