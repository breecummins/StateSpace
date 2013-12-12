import numpy as np

def parseangles(anglesonlyfname):
    f = open(anglesonlyfname,'r')
    flag = 0
    while not flag:
        l = f.readline()
        try:
            int(l[0])
            flag = 1
        except:
            if l[:4] == 'Time':
                names = filter(None,l.rstrip().split(','))[1:]
            continue
    L = [float(n) for n in l.rstrip().split(',')]
    times = [L.pop(0)]
    anglists=[L]
    for l in f:
        try:
            L = [float(n) for n in l.rstrip().split(',')]
            times.append(L.pop(0))
            anglists.append(L)
        except:
            if not l.rstrip():
                continue
            else:
                raise
    return names,times,np.array(anglists)

if __name__ == '__main__':
    import os
    names,times,anglesarray = parseangles(os.path.join(os.path.expanduser('~'),'MotionCaptureData/20110202-GBNNN-VDEF-08_PabloDataSet1_AnglesOnly.csv'))
    print(names)
    print(len(times))
    print(anglesarray.shape)