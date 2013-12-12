import numpy as np
import StateSpace.StateSpaceReconstruction as SSR
import StateSpace.StateSpaceReconstructionPlots as SSRPlots
import parsers

def vizdata(anglesonlyfname):
    names,times,angarray = parsers.parseangles(anglesonlyfname)
    drop = range(9,12)+range(21,24)+range(28,30)+range(37,39)+range(42,45)+range(45,48)+range(54,57)+range(57,60)
    inds = [ i for i in range(0,angarray.shape[1],3) if i not in drop ]
    lags = SSR.chooseLags(angarray[:,inds],[angarray.shape[0]])
    ns = [names[int(np.floor(j/3))] for j in inds]
    print(ns)
    return None
    inds = [i for i in range(0,angarray.shape[1],3) if i not in [3*3,3*7,3*14,3*15,3*18,3*19]]
    for i in inds:
        SSRPlots.plotManifold(angarray[:,i:i+3],show=0,hold=0,style='b-',titlestr=names[int(i/3)],scatter=False,color=None)
        SSRPlots.plotShadowManifold(angarray[:,i], 3, 10, show=1,hold=0,style='r-',titlestr=names[int(i/3)]+' reconstruction',scatter=False, color=None,smooth=1)

def ssr():
    pass

if __name__ == '__main__':
    import os
    vizdata(os.path.join(os.path.expanduser('~'),'MotionCaptureData/20110202-GBNNN-VDEF-08_PabloDataSet1_AnglesOnly.csv'))