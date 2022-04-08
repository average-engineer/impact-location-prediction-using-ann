#**************************** Filtering Data *********************************
# Filter Used: 4th Order Low Pass Butterworth Filter


# Importing the signal module from Scipy
from scipy import signal

def dataFilter(in_signal,order,coFreq,sampFreq):
    
    [b,a] = signal.butter(order,coFreq/(sampFreq/2),'low')
    out_signal = signal.filtfilt(b,a,in_signal)
    return out_signal


