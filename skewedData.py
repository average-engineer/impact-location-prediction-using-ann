import numpy as np
from dataAugment import dataAugment
from removeArrFromList import removeArrFromList
import random

#%% Data Balancing
# A balanced training and validation data (i.e.) equal number of impact locations from each quadrant
# is expected to give better results for the CNN

# The numerical data is balanced due to augmentation
# 976 impact locations in Numerical simulation data
# 244 locations per quadrant

# 12 sets from the experimental validation data are also used for training and validation
# Structure of Experimenetal validation data -> 4 sensor data corresponding to each location
# Thus, 3 locations are taken for the training and validation data
# Locations: (230,240) -> Quadrant 3, (205,200) -> Quadrant 3, (250,260) -> Axis (Quadrant 1 and 2)

# Loading the numerical simulation data
finalNumData_load = np.loadtxt('saved_variables/finalNumData.txt')
# Reshaping
finalNumData = finalNumData_load.reshape(finalNumData_load.shape[0], finalNumData_load.shape[1] // 7, 7)

dataBase = finalNumData # Sxy is the final training and validation sensor data

# Loading the experimental validation data
finalexpData_load = np.loadtxt('saved_variables/finalexpData.txt')
# Reshaping
finalexpData = finalexpData_load.reshape(finalexpData_load.shape[0], finalexpData_load.shape[1] // 7, 7)

# The first 12 datasets (for training and validation)
SxyExp = finalexpData[:12,:,:]

# Augmented Experimental data
# For each impact location in SxyExp, 3 more corresponding impact locations are generated
Sxy_augExp = np.zeros((3*SxyExp.shape[0],SxyExp.shape[1],SxyExp.shape[2]))

count = 0 # Counter variable for keeping track of the iterations over SxyExp
# Each of these datasets are augmented in a loop
for ii in range(0, SxyExp.shape[0]):
    data1 = np.zeros(SxyExp[ii,:,:].shape)
    data2 = np.zeros(SxyExp[ii,:,:].shape)
    data3 = np.zeros(SxyExp[ii,:,:].shape)
    data1, data2, data3 = dataAugment(SxyExp[ii,:,:])
    Sxy_augExp[count,:,:] = data1
    Sxy_augExp[count+1,:,:] = data2
    Sxy_augExp[count+2,:,:] = data3



    # Updating counter
    count = count + 3

# Complete Sensor data
dataBase = np.concatenate((dataBase,SxyExp, Sxy_augExp),axis = 0) # SxyExp[:,:;1:5] denotes the original unaugmented impact locations in the experimental data

#%% Directories based on skewness degree
dir1 = 'saved_variables/Skewed_Data/Q2/'
dir2 = 'saved_variables/Skewed_Data/Q2Q4/'
dir3 = 'saved_variables/Skewed_Data/Q2Q4Q3/'


#%% Segregating the database into groups based on the quadrants of the impact locations

# Empty lists for segregating the database
axis = [] # Impact Locations on the axis
q1 = [] # Impact Locations in Q1
q2 = [] # Impact Locations in Q2
q3 = [] # Impact Locations in Q3
q4 = [] # Impact Locations in Q4

for ii in range(0,dataBase.shape[0]):

    if dataBase[ii,0,5] == 250 or dataBase[ii,0,6] == 250:
        axis.append(dataBase[ii,:,:])

    # elif dataBase[ii,0,5] >= 250 and dataBase[ii,0,5] <= 325 and dataBase[ii,0,6] >= 250 and dataBase[ii,0,6] <= 325:
    #     q1.append(dataBase[ii,:,:])

    elif dataBase[ii,0,5] >= 175 and dataBase[ii,0,5] <= 250 and dataBase[ii,0,6] >= 250 and dataBase[ii,0,6] <= 325:
        q2.append(dataBase[ii,:,:])

    elif dataBase[ii,0,5] >= 175 and dataBase[ii,0,5] <= 250 and dataBase[ii,0,6] >= 175 and dataBase[ii,0,6] <= 250:
        q3.append(dataBase[ii,:,:])

    elif dataBase[ii,0,5] >= 250 and dataBase[ii,0,5] <= 325 and dataBase[ii,0,6] >= 175 and dataBase[ii,0,6] <= 250:
        q4.append(dataBase[ii,:,:])

# Segregation strategy
# 4 arrays taken from the list of arrays with impact locations on the axis
# 24 arrays taken from each list corresponding to the quadrants
# Then they are clubbed as validation data
# The remaining data is the training data

# Empty list for storing the training data from each quadrant

# Removing 4 arrays from the axis arrays
axisVal = random.sample(axis,4)
for i in range(0,len(axisVal)):
    axis = removeArrFromList(axis, axisVal[i])



# Removing 24 arrays from the q1 arrays
# q1Val = random.sample(q1,24)
# for i in range(0,len(q1Val)):
#     q1 = removeArrFromList(q1, q1Val[i])

# Removing 24 arrays from the q2 arrays
q2Val = random.sample(q2,24)
for i in range(0,len(q2Val)):
    q2 = removeArrFromList(q2, q2Val[i])

# Removing 24 arrays from the q3 arrays
q3Val = random.sample(q3,24)
for i in range(0,len(q3Val)):
    q3 = removeArrFromList(q3, q3Val[i])

# Removing 24 arrays from the q4 arrays
q4Val = random.sample(q4,24)
for i in range(0,len(q4Val)):
    q4 = removeArrFromList(q4, q4Val[i])


# Assembling training and validation data list
trainList = [*axis,*q2,*q4,*q3]
valList = [*axisVal,*q2Val,*q4Val,*q3Val]

# Complete Dataset
compList = [*trainList,*valList]


# Training Data Array
Sxy_train = np.zeros((len(trainList),finalNumData.shape[1],4)) # Training sensor data
Oxy_train = np.zeros((len(trainList),2)) # Training sensor data

# Validation Data Array
Sxy_val = np.zeros((len(valList),finalNumData.shape[1],4)) # Training sensor data
Oxy_val = np.zeros((len(valList),2)) # Training sensor data

# Complete Data (Training + Validation)
Sxy = np.zeros((len(compList),finalNumData.shape[1],4)) # Training sensor data
Oxy = np.zeros((len(compList),2)) # Training sensor data

for ii in range(0,len(trainList)):
    Sxy_train[ii,:,:] = trainList[ii][:,1:5]
    Oxy_train[ii,:] = trainList[ii][0,5:]

for ii in range(0,len(valList)):
    Sxy_val[ii,:,:] = valList[ii][:,1:5]
    Oxy_val[ii,:] = valList[ii][0,5:]

for ii in range(0,len(compList)):
    Sxy[ii,:,:] = compList[ii][:,1:5]
    Oxy[ii,:] = compList[ii][0,5:]

#%% Locally saving the training and test data
np.savetxt(dir3 + 'Oxy_train.txt',Oxy_train)
np.savetxt(dir3 + 'Oxy_val.txt',Oxy_val)
Sxy_train_rshpd = Sxy_train.reshape(Sxy_train.shape[0], -1)
np.savetxt(dir3 + 'Sxy_train.txt',Sxy_train_rshpd)
Sxy_val_rshpd = Sxy_val.reshape(Sxy_val.shape[0], -1)
np.savetxt(dir3 + 'Sxy_val.txt',Sxy_val_rshpd)

np.savetxt(dir3 + 'Oxy.txt',Oxy)
Sxy_rshpd = Sxy.reshape(Sxy.shape[0], -1)
np.savetxt(dir3 + 'Sxy.txt',Sxy_rshpd)

print(Oxy_train.shape, Sxy_train_rshpd.shape, Oxy_val.shape, Sxy_val_rshpd.shape,Oxy.shape, Sxy_rshpd.shape)
