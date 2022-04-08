# Function for augmenting sensor data
# A sensor time series data is the input and according to the impact location of the data
# the equivalent sensor data is generated for the other quadrants

# For eg: a sensor data is input whose corresponding impact location is in the 
# 3rd quadrant
# Then the sensor data for the mirrors of that impact location in all the other
# quadrants is generated

import numpy as np

from mirrorPoint import mirrorPoint

def dataAugment(data):
    
    # data will be a 2D array of size nx7
    
    # Corresponding quadrant data
    data1 = np.zeros(data.shape)
    data2 = np.zeros(data.shape)
    data3 = np.zeros(data.shape)
    
    # 1st row: time, 2-5th rows: sensor data, 6-7 rows: impact locations

        
    # Eqv. point across Y axis
    # Sensor1_new = Sensor 1
    # Sensor2_new = Sensor 4
    # Sensor3_new = Sensor 3
    # Sensor4_new = Sensor 2
    
    # Eqv. point across X axis
    # Sensor1_new = Sensor 3
    # Sensor2_new = Sensor 2
    # Sensor3_new = Sensor 1
    # Sensor4_new = Sensor 4
    
    # Eqv. point across Origin
    # Sensor1_new = Sensor 3
    # Sensor2_new = Sensor 4
    # Sensor3_new = Sensor 1
    # Sensor4_new = Sensor 2
    
    # Generating point in Quadrant 2 -> Mirroring across Y axis
    x1,y1 = mirrorPoint(data[0,5],data[0,6],250,250,'Y')
    
    data1[:,1] = data[:,1]
    data1[:,2] = data[:,4]
    data1[:,3] = data[:,3]
    data1[:,4] = data[:,2]
    data1[:,0] = data[:,0]
    data1[:,5] = x1
    data1[:,6] = y1
    
    # Generating point in Quadrant 3 -> Mirroring across Origin
    x2,y2 = mirrorPoint(data[0,5],data[0,6],250,250,'Origin')
    
    data2[:,1] = data[:,3]
    data2[:,2] = data[:,4]
    data2[:,3] = data[:,1]
    data2[:,4] = data[:,2]
    data2[:,0] = data[:,0]
    data2[:,5] = x2
    data2[:,6] = y2
    
    # Generating point in Quadrant 4 -> Mirroring across X axis
    x3,y3 = mirrorPoint(data[0,5],data[0,6],250,250,'X')
    
    data3[:,1] = data[:,3]
    data3[:,2] = data[:,2]
    data3[:,3] = data[:,1]
    data3[:,4] = data[:,4]
    data3[:,0] = data[:,0]
    data3[:,5] = x3
    data3[:,6] = y3
    
    return data1, data2, data3
        
    