# Function for cutting the data after the first highest peak

from scipy import signal
import numpy as np

def dataSegment(Arr,dataNum):
    peakS1, _ = signal.find_peaks(Arr[:,1], height=None)
    peakS2, _ = signal.find_peaks(Arr[:,2], height=None)
    peakS3, _ = signal.find_peaks(Arr[:,3], height=None)
    peakS4, _ = signal.find_peaks(Arr[:,4], height=None)
    
    # S1 = {}
    # S2 = {}
    # S3 = {}
    # S4 = {}

    # for i in peakS1:
    #     # Peaks detected at maximum absolute (negative and positive) values
    #     #S1[i] = np.absolute(Arr[i,1])
    #     S1[i] = Arr[i,1]
    # k1 = max(S1, key=S1.get)
    
    # for i in peakS2:
    #     #S2[i] = np.absolute(Arr[i,2])
    #     S2[i] = Arr[i,2]
    # k2 = max(S2, key=S2.get)
    
    # for i in peakS3:
    #     #S3[i] = np.absolute(Arr[i,3])
    #     S3[i] = Arr[i,3]
    # k3 = max(S3, key=S3.get)
    
    # for i in peakS4:
    #     #S4[i] = np.absolute(Arr[i,4])
    #     S4[i] = Arr[i,4]
    # k4 = max(S4, key=S4.get)
    
#     print(k1, k2, k3, k4)

    peakVals1 = np.zeros(peakS1.shape)
    peakVals2 = np.zeros(peakS2.shape)
    peakVals3 = np.zeros(peakS3.shape)
    peakVals4 = np.zeros(peakS4.shape)
    
    k1 = 0
    k2 = 0
    k3 = 0
    k4 = 0
    
    for i in range(0,peakS1.shape[0]):
        peakVals1[i] = Arr[peakS1[i],1]
        
    for i in range(0,peakS1.shape[0]):
        if peakVals1[i] == max(abs(peakVals1)):
            # k1 = i
            k1 = peakS1[i]
            
    for i in range(0,peakS2.shape[0]):
        peakVals2[i] = Arr[peakS2[i],2]
        
    for i in range(0,peakS2.shape[0]):
        if peakVals2[i] == max(abs(peakVals2)):
            # k2 = i
            k2 = peakS2[i]
            
    for i in range(0,peakS3.shape[0]):
        peakVals3[i] = Arr[peakS3[i],3]
        
    for i in range(0,peakS3.shape[0]):
        if peakVals3[i] == max(abs(peakVals3)):
            # k3 = i
            k3 = peakS3[i]
            
    for i in range(0,peakS4.shape[0]):
        peakVals4[i] = Arr[peakS4[i],4]
        
    for i in range(0,peakS4.shape[0]):
        if peakVals4[i] == max(abs(peakVals4)):
            # k4 = i
            k4 = peakS4[i]
    
    
    
    # Each sensor data is cut after its highest peak
    dS1 = Arr[:k1,1]
    dS2 = Arr[:k2,2]
    dS3 = Arr[:k3,3]
    dS4 = Arr[:k4,4]
    
    r = max(k1,k2,k3,k4) # The peak with the maximum index is selected
    
    # The segmented data will have r rows
    # Each sensor data will have data till its highest peak and then be zero
    # till r rows are populated
    
    data_cut = np.zeros(((r,7)))
    data_cut[:,0] = Arr[:r,0]
    data_cut[:k1,1] = dS1
    data_cut[:k2,2] = dS2
    data_cut[:k3,3] = dS3
    data_cut[:k4,4] = dS4
    # Impact Locations
    data_cut[:,5] = Arr[:r,5] 
    data_cut[:,6] = Arr[:r,6]
    
    # Resampling the data
    dataCut = np.zeros((dataNum,7))
    
    dataCut[:,1:5] = signal.resample(data_cut[:,1:5],dataNum) # Resampling sensor Values
    if data_cut.shape[0] > dataNum:
        dataCut[:,5:] = data_cut[:dataNum,5:] # Impact Locations
        
    else:
        dataCut[:,5:] = data_cut[0,5:]*np.ones((dataNum,1))
        
    dataCut[:,0] = np.linspace(data_cut[0,0],data_cut[-1,0],dataNum) # Time Vector
    
    
    
    return dataCut
