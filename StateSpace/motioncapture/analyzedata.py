import numpy as np
import StateSpace.StateSpaceReconstruction as SSR
import StateSpace.StateSpaceReconstructionPlots as SSRPlots
import parsers

def vizdata(anglesonlyfname):
    names,times,angarray = parsers.parseangles(anglesonlyfname)
    laginds = range(angarray.shape[1])
    lags = SSR.chooseLags(angarray[:,laginds],[angarray.shape[0]])
    modlags = [l if l <= 30 else 30 for l in lags[0]]
    print(modlags)
    drop = [9,21,42,45,54,57]
    inds = [i for i in laginds if not i%3 and i not in drop]
    for i in inds:
        SSRPlots.plotManifold(angarray[:,i:i+3],show=0,hold=0,style='b-',titlestr=names[i/3],scatter=False,color=None)
        SSRPlots.plotShadowManifold(angarray[:,i], 3, modlags[i], show=0,hold=0,style='r-',titlestr=names[i/3]+', M_x',scatter=False, color=None,smooth=1)
        SSRPlots.plotShadowManifold(angarray[:,i+1], 3, modlags[i+1], show=0,hold=0,style='g-',titlestr=names[i/3]+', M_y',scatter=False, color=None,smooth=1)
        SSRPlots.plotShadowManifold(angarray[:,i+2], 3, modlags[i+2], show=1,hold=0,style='k-',titlestr=names[i/3]+', M_z',scatter=False, color=None,smooth=1)

def ssr():
    pass

if __name__ == '__main__':
    import os
    vizdata(os.path.join(os.path.expanduser('~'),'MotionCaptureData/20110202-GBNNN-VDEF-08_PabloDataSet1_AnglesOnly.csv'))