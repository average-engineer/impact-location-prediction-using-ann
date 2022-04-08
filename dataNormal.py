# Function for Normalising Data

import numpy as np
def dataNormal(Arr):
    # Input is a 3D array (X,Y,Z) size
    for i in range(Arr.shape[0]):
        Arr[i,:,1] = (Arr[i,:,1] - np.min(Arr[i,:,1]))/(np.max(Arr[i,:,1]) - np.min(Arr[i,:,1]))
        Arr[i,:,2] = (Arr[i,:,2] - np.min(Arr[i,:,2]))/(np.max(Arr[i,:,2]) - np.min(Arr[i,:,2]))
    
        Arr[i,:,4] = (Arr[i,:,4] - np.min(Arr[i,:,4]))/(np.max(Arr[i,:,4]) - np.min(Arr[i,:,4]))
        Arr[i,:,3] = (Arr[i,:,3] - np.min(Arr[i,:,3]))/(np.max(Arr[i,:,3]) - np.min(Arr[i,:,3]))
        
        
    return Arr
