import matplotlib.pyplot as plt
import StateSpace.StateSpaceReconstruction as SSR
import StateSpace.PecoraMethodModified as PM

def extractRates(fname='physdata_13081200_NewSummary.csv'):
    f = open(fname,'r')
    f.readline()
    data = f.readlines()
    f.close()
    heartrate = []
    breathingrate = []
    for d in data: 
        words = d.split(',')
        heartrate.append(float(words[3]))
        breathingrate.append(float(words[5]))
    return heartrate,breathingrate

def plotRates(hr,br,start=0,stop=None):
    if not stop:
        stop = len(hr)
    timeindex = range(start,stop)
    plt.plot(timeindex,hr[start:stop],'r')
    plt.hold('on')
    plt.plot(timeindex,br[start:stop],'b')
    plt.legend(['Heart rate','Breathing rate'])
    plt.xlim([start,stop])
    plt.show()

def getLags(hr,br):
    pass

def getDims(hr,br):
    pass

def continuityTesting(hr,br,lags,dim):
    pass


if __name__ == "__main__":
    hr, br = extractRates()
    plotRates(hr,br,90,720)


