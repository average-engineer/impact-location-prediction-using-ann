from scipy import signal
import numpy as np

def expDataSegment(Arr):
    peakS1, _ = signal.find_peaks(Arr[:,1], height=None)
    peakS2, _ = signal.find_peaks(Arr[:,2], height=None)
    peakS3, _ = signal.find_peaks(Arr[:,3], height=None)
    peakS4, _ = signal.find_peaks(Arr[:,4], height=None)
    
    S1 = {}
    S2 = {}
    S3 = {}
    S4 = {}
    
    for i in peakS1:
        # Peaks detected at maximum absolute (negative and positive) values
        #S1[i] = np.absolute(Arr[i,1])
        S1[i] = Arr[i,1]
    k1 = max(S1, key=S1.get)
    
    for i in peakS2:
        #S2[i] = np.absolute(Arr[i,2])
        S2[i] = Arr[i,2]
    k2 = max(S2, key=S2.get)
    
    for i in peakS3:
        #S3[i] = np.absolute(Arr[i,3])
        S3[i] = Arr[i,3]
    k3 = max(S3, key=S3.get)
    
    for i in peakS4:
        #S4[i] = np.absolute(Arr[i,4])
        S4[i] = Arr[i,4]
    k4 = max(S4, key=S4.get)
    
    
    fIndex = min(k1,k2,k3,k4) - 50 # 50 datapoints before the first peak
    lIndex = fIndex + 300 # 300 datapoints considered
    
    dataCut = Arr[fIndex:lIndex + 1,:]
    
    return dataCut 
