#%% Importing Modules
import matplotlib.pyplot as plt
import pandas as pd
import os
from io import BytesIO
import owncloud
from datetime import datetime
import numpy as np
from scipy import signal
import re
import scipy.io
import glob
from scipy.fft import fft, fftfreq, rfft, rfftfreq # For FFT
from sklearn.preprocessing import MinMaxScaler # For Data Normalisation
from sklearn.model_selection import train_test_split

#%% Importing Methods
from dataFilter import dataFilter
from mirrorPoint import mirrorPoint
from pointQuad import pointQuad
from dataSegment import dataSegment
from expDataSegment import expDataSegment
from dataNormal import dataNormal
#%% Settings
plt.close('all') # Closing all plots at the start of execution

#%% Exracting Data
directory = ['/EPOT_Data/','/Validation_augmented_data/','/Experimental_validation/'] # Remote Directories

#%% Accessing the remote directory
numSim_Q1 = [] # Empty List for storing all individual .mat files of the numerical simulations for impact locations in quadrant 1
numSim_Q2 = [] # Empty List for storing all individual .mat files of the numerical simulations for impact locations in quadrant 1
numSim_Q3 = [] # Empty List for storing all individual .mat files of the numerical simulations for impact locations in quadrant 1
numSim_Q4 = [] # Empty List for storing all individual .mat files of the numerical simulations for impact locations in quadrant 1
valAug_data = [] # Empty List for storing all individual .mat files of the validation augmented data
valAug_loc = [] # Empty List for storing the impact coorinates included in the validation augmented data
checkAugData = [] # Empty List for storing the augmented datasets which need to be validated
exp_data = []
numSimData = [] # Empty List for storing all numerical simulation data of all quadrants 
checkExpData = []
 
public_link = 'https://rwth-aachen.sciebo.de/s/Q114EFBAp1QP3fq'
folder_password = 'CIE_B'
grp14_data = 'https://rwth-aachen.sciebo.de/s/cvQW1nT5Kzgvgms'
grp14_pw = 'CIE'
oc = owncloud.Client.from_public_link(public_link, folder_password=folder_password)
oc1 = owncloud.Client.from_public_link(grp14_data, folder_password=grp14_pw)
# =============================================================================
# content = oc.get_file('/EPOT_Data/'+files[0].get_name()) # Downloading the .mat file locally so scipy.io can read it
# print(type(files[0].get_name()))
# mat = scipy.io.loadmat(files[0].get_name())
# os.remove(files[0].get_name())
# =============================================================================

# Looping through all the files in the remote directory
# We obtain each .mat file as a seperate dictionary


#%% Validation Augmented Data
files = oc.list(directory[1], depth = 'infinity')

for file in files:
    colNames = ['Time', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4']
    mat_df = pd.DataFrame(columns = colNames)
    content = oc.get_file(directory[1] + file.get_name())
    # Loading the .mat file
    mat = scipy.io.loadmat(file.get_name())
    
    # Extracting the Impact Locations from the file names
    x = float(file.get_name().split('_')[1]) # X coordinate of the Impact Location
    y = float(file.get_name().split('_')[2].split('.')[0]) # Y coordinate of the Impact Location
    
    os.remove(file.get_name()) # Removing the file from the local directory once it has been read
    
    mat = mat['num_data']
    
    # Rearranging the numpy arrays into dataframes
    mat_df['Time'] = mat[:,0]
    mat_df['Sensor 1'] = mat[:,1]
    mat_df['Sensor 2'] = mat[:,2]
    mat_df['Sensor 3'] = mat[:,3]
    mat_df['Sensor 4'] = mat[:,4]
    
    # Appending the obtained array to the numSim_data list
    valAug_data.append(mat_df)
    valAug_loc.append((x,y))

#%% Numerical Simulation Data
files_num = oc.list(directory[0], depth = 'infinity') # Obtaining file info and names in the remote folder

for file in files_num:
    colNames = ['Time', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'X', 'Y']
    q1_df = pd.DataFrame(columns = colNames)
    q2_df = pd.DataFrame(columns = colNames)
    q3_df = pd.DataFrame(columns = colNames)
    q4_df = pd.DataFrame(columns = colNames)
    content = oc.get_file(directory[0] + file.get_name()) # Downloading the mat file
    # Loading the .mat file
    mat = scipy.io.loadmat(file.get_name())
    
    # Extracting the Impact Locations from the file names
    x = float(file.get_name().split('_')[1]) # X coordinate of the Impact Location
    y = float(file.get_name().split('_')[2].split('.')[0]) # Y coordinate of the Impact Location
    
    
    os.remove(file.get_name()) # Removing the file from the local directory once it has been read
    
    # We obtain each mat as a dictionary
    # The actual array is stored as the value corresponding to the 4th key ('num_data')
    mat = mat['num_data']
    
    # Evaluating the Quadrant of Impact depending on the Impact Locations
    quadrant = pointQuad(x,y,mat.shape[0])
    
    # Rearranging the numpy arrays into dataframes
    q1_df['Time'] = mat[:,0]
    q1_df['Sensor 1'] = mat[:,1]
    q1_df['Sensor 2'] = mat[:,2]
    q1_df['Sensor 3'] = mat[:,3]
    q1_df['Sensor 4'] = mat[:,4]
    q1_df['X'] = x
    q1_df['Y'] = y
    
    
    # Appending the obtained array to the numSim_data list
    numSim_Q1.append(q1_df)
    
    
# Format of each .mat file -> 20000 rows and 5 columns
#                          -> First Column: Time Vector
#                          -> Next 4 columns: Electric Potential of 4 Piezoelectric Transducers
    
    # DATA AUGMENTATION
    # The provided simulation data only covers impact locations in 1st Quadrant
    # Each file corresponds to one impact location in the first quadrant
    # For each impact location in Quadrant 1, we can generate the corresponding mirror impact locations in other quadrants
    # For any point P in quadrant 1:
        # Mirror Point in Qudrant 2 -> Mirroring the point across Y axis
        # Mirror Point in Quadrant 3 -> Mirroring the point across origin/Mirroring quad 2 point across X axis
        # Mirror Point in Quadrant 4 -> Mirroring the point across X axis/ Mirroring quad 3 point across Y axis
        
    # The considered coordinate system his shifted and has its origin at (250,250)
        
    # Mirror Point in Quadrant 2
    x,y = mirrorPoint(x,y,250,250,'Y')
    quadrant = pointQuad(x,y,mat.shape[0])
    
    # Rearranging the numpy arrays into dataframes
    # For Quadrant 2 impact location, sensor 4 becomes equivalent to sensor 2 (in quad 1) and vice versa
    q2_df['Time'] = mat[:,0]
    q2_df['Sensor 1'] = mat[:,1]
    q2_df['Sensor 2'] = mat[:,4]
    q2_df['Sensor 3'] = mat[:,3]
    q2_df['Sensor 4'] = mat[:,2]
    q2_df['X'] = x
    q2_df['Y'] = y
    
    numSim_Q2.append(q2_df)
    
    
    # Mirror Point in Quadrant 3
    x,y = mirrorPoint(x,y,250,250,'X')
    quadrant = pointQuad(x,y,mat.shape[0])
    
    # Rearranging the numpy arrays into dataframes
    # For Quadrant 3 impact location, sensor 3 becomes equivalent to sensor 1 (in quad 1) and vice versa
    q3_df['Time'] = mat[:,0]
    q3_df['Sensor 1'] = mat[:,3]
    q3_df['Sensor 2'] = mat[:,4]
    q3_df['Sensor 3'] = mat[:,1]
    q3_df['Sensor 4'] = mat[:,2]
    q3_df['X'] = x
    q3_df['Y'] = y
    
    numSim_Q3.append(q3_df)
    
    # Mirror Point in Quadrant 4
    x,y = mirrorPoint(x,y,250,250,'Y')
    quadrant = pointQuad(x,y,mat.shape[0])
    
    # Rearranging the numpy arrays into dataframes
    # For Quadrant 4 impact location, sensor 4 becomes equivalent to sensor 1 (in quad 1) and vice versa
    q4_df['Time'] = mat[:,0]
    q4_df['Sensor 1'] = mat[:,3]
    q4_df['Sensor 2'] = mat[:,2]
    q4_df['Sensor 3'] = mat[:,1]
    q4_df['Sensor 4'] = mat[:,4]
    q4_df['X'] = x
    q4_df['Y'] = y
    
    # Comparing the augmented data with the validation augmented data provided
    # In the validation data provided, the locations are the corresponding mirrors of (275,265) in quadrant 1
    
    if float(file.get_name().split('_')[1]) == 275 and float(file.get_name().split('_')[2].split('.')[0]) == 265:
        checkAugData.append(q3_df)
        checkAugData.append(q2_df)
        checkAugData.append(q4_df)
        checkAugData.append(q1_df)
        
    if float(file.get_name().split('_')[1]) == 250 and float(file.get_name().split('_')[2].split('.')[0]) == 260:
        # checkExpData.append(q3_df)
        # checkExpData.append(q2_df)
        # checkExpData.append(q4_df)
        checkExpData.append(q1_df)  
    
        
    
    # For each given impact location in the first quadrant, its corresponding augmented 
    # simulation data in all the other quadrants is also stored
    numSimData.append(q1_df)
    numSimData.append(q2_df)
    numSimData.append(q3_df)
    numSimData.append(q4_df)
    
# Converting the numerical data into a 3D Numpy Array
numSimData = np.array(numSimData)

#%% Plotting raw numerical data
num = 14
fig = plt.figure(figsize = (6,6))
plt.plot(numSimData[num,:,4])
plt.title('Sensor 4 Data')
plt.xlabel('Number of Datapoints',fontsize = 12)
plt.ylabel('Sensor Data',fontsize = 12)
plt.grid()

#%% Normalisation
# numSimData = dataNormal(numSimData)

# Using MinMaxScaler method
for i in range(0,numSimData.shape[0]):
    numSimData[i,:,1:5] = MinMaxScaler(feature_range = (-1,1)).fit_transform(numSimData[i,:,1:5]) # Normalising between -1 and 1
    
#%% Cutting the numerical data till the highest peaks and resampling 
dataNum = 1000 # number of datapoints in numerical data
finalNumData = np.zeros((numSimData.shape[0],dataNum,numSimData.shape[2]))
for ii in range(0,numSimData.shape[0]):
    finalNumData[ii,:,:] = dataSegment(numSimData[ii,:,:],dataNum)
    
    
       
#%% Experimental Validation Data
files = oc.list(directory[2], depth = 'infinity')


for file in files:
    if not file.is_dir():
        # going through each text file
        # Downloading each text file
        
        content = oc.get_file(file.get_path()+ '/' + file.get_name()) # Downloadinf the file
        
        # Reading each text file
        
        # File format of each experimental data text file
        # File:    C:\exports\piezo_171.txt
        # Created: Freitag, 17. Dezember 2021 14:27:58
        # Header time format: Absolute
        # Time of first sample: 351 14:27:50.745076300
        # Title:   

        # Time	Ch A1	Ch A2	Ch A3	Ch A4
        # s	V	V	V	V
        
        # The first 8 lines have to be skipped and are not important
        # The first entry of the time series is the 9th line
        # We start reading from the 9th line (8th Index Number)
        
        f = open(file.get_name(),'r')
        contents = f.readlines() # readlines() returns the entire line as a single string
        # The multiple lines are thus returned as multiple strings, all part of a list
        contents = contents[8:]
        
        expT = [] # Empty list for storing time vector of each reading
        expS1 = [] # Empty list for storing sensor 1 reading 
        expS2 = [] # Empty list for storing sensor 2 reading
        expS3 = [] # Empty list for storing sensor 3 reading
        expS4 = [] # Empty list for storing sensor 4 reading
        expData = []
        X = []
        Y = []
        
        for i in range(0,len(contents)): # going through each line string
            
            # Each line contains the '\n' character at the end to signify the next line
            # Removing '\n' from each line string
            contents[i] = contents[i].replace('\n','') # repalce() doesn't modify the original list implicitly
            
            # The decimals in the file are written as commas which may create problems
            # Replacing all the commas with decimal points
            # replace(oldString, newString, # times the replacement has to be carried out)
            contents[i] = contents[i].replace(',','.',contents[i].count(','))
            
            
            # Each line string now has to be split with '\t' as the splitting tool
            # split() will split each line string wrt '\t' and return a list of the splitted strings
            # for each line
            # First split string -> Time vector element
            expT.append(float(contents[i].split('\t')[0]))
            # Second split string -> Sensor 1 reading
            expS1.append(float(contents[i].split('\t')[1]))
            # Third split string -> Sensor 2 reading
            expS2.append(float(contents[i].split('\t')[2]))

            # Fourth split string -> Sensor 3 reading
            expS3.append(float(contents[i].split('\t')[4]))

            # Fifth split string -> Sensor 4 reading
            expS4.append(float(contents[i].split('\t')[3]))
            
            # Impact Location X Coordinate
            X.append(int(file.get_name().split('_')[2]))
            
            # Impact Location Y Coordinate
            Y.append(int(file.get_name().split('_')[3]))

            
            
            
            
        
        
        f.close() # Closing the access to the text file
        colNames = ['Time', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4']
        txt_df = pd.DataFrame(columns = colNames)
        
        txt_df['Time'] = expT
        # expS1 = dataFilter(expS1,4,10000,len(expS1)/expT[-1])
        # expS2 = dataFilter(expS1,4,10000,len(expS2)/expT[-1])
        # expS3 = dataFilter(expS1,4,10000,len(expS3)/expT[-1])
        # expS4 = dataFilter(expS1,4,10000,len(expS4)/expT[-1])
        txt_df['Sensor 1'] = expS1
        txt_df['Sensor 2'] = expS2
        txt_df['Sensor 3'] = expS3
        txt_df['Sensor 4'] = expS4
        txt_df['X'] = X
        txt_df['Y'] = Y
        
        
        # Deleting the read text file from local directory
        os.remove(file.get_name())
        
        
        
        # content = oc.get_file(directory[1] + 'SE1_a_t/' + file.get_name())
        # Loading the .text file
        # os.remove(file.get_name()) # Removing the file from the local directory once it has been read
    
        # Appending the dataframe correspnding to each file to the experimental data list
        exp_data.append(txt_df)
        
exp_data = np.array(exp_data)
        
#%% Final Experimental Validation Data
                
# Cutting Garbage Data
for ii in range(0,exp_data.shape[0]):
    dataCut = expDataSegment(exp_data[ii,:,:])
    dataCut[:,1:5] = -dataCut[:,1:5] # Inverting the Experimental Data
    expData.append(dataCut) 
                
expData = np.array(expData, dtype = object)

#%% Filtering the Data
# for i in range(0,expData.shape[0]):
#     expData[i,:,:] = dataFilter(expData[i,:,:],1,expData[i,:,:].shape[0]*(expData[i,-1,0] - expData[i,0,0])/2.5,expData[i,:,:].shape[0]*(expData[i,-1,0] - expData[i,0,0]))

#%% Normalisation
# expData = dataNormal(expData) 

# Using MinMaxScaler method
for i in range(0,expData.shape[0]):
    expData[i,:,1:5] = MinMaxScaler(feature_range = (-1,1)).fit_transform(expData[i,:,1:5]) # Normalising between -1 and 1

#%% Data Resampling to datapoints same as Numerical Data and cutting it after the highest peak
finalexpData = np.zeros((expData.shape[0],dataNum,7))
for ii in range(0,expData.shape[0]):
    finalexpData[ii,:,:] = dataSegment(expData[ii,:,:],dataNum)

#%% Experimental Data (Group Specific)
dataNum = 1000 # number of datapoints in numerical data
expData = []
exp_data = []

#%% Experimental Data (Group Specific)
files = oc1.list('/', depth = 'infinity')


for file in files:
    #if not file.is_dir():
    # going through each text file
    # Downloading each text file
    
    content = oc1.get_file(file.get_path() + file.get_name()) # Downloading the file
    
    # Reading each text file
    
    # File format of each experimental data text file
    # File:    C:\exports\piezo_171.txt
    # Created: Freitag, 17. Dezember 2021 14:27:58
    # Header time format: Absolute
    # Time of first sample: 351 14:27:50.745076300
    # Title:   

    # Time	Ch A1	Ch A2	Ch A3	Ch A4
    # s	V	V	V	V
    
    # The first 8 lines have to be skipped and are not important
    # The first entry of the time series is the 9th line
    # We start reading from the 9th line (8th Index Number)
    
    f = open(file.get_name(),'r')
    contents = f.readlines() # readlines() returns the entire line as a single string
    # The multiple lines are thus returned as multiple strings, all part of a list
    contents = contents[8:]
    
    expT = [] # Empty list for storing time vector of each reading
    expS1 = [] # Empty list for storing sensor 1 reading 
    expS2 = [] # Empty list for storing sensor 2 reading
    expS3 = [] # Empty list for storing sensor 3 reading
    expS4 = [] # Empty list for storing sensor 4 reading
    expData = []
    X = []
    Y = []
    
    for i in range(0,len(contents)): # going through each line string
        
        # Each line contains the '\n' character at the end to signify the next line
        # Removing '\n' from each line string
        contents[i] = contents[i].replace('\n','') # repalce() doesn't modify the original list implicitly
        
        # The decimals in the file are written as commas which may create problems
        # Replacing all the commas with decimal points
        # replace(oldString, newString, # times the replacement has to be carried out)
        contents[i] = contents[i].replace(',','.',contents[i].count(','))
        
        
        # Each line string now has to be split with '\t' as the splitting tool
        # split() will split each line string wrt '\t' and return a list of the splitted strings
        # for each line
        # First split string -> Time vector element
        expT.append(float(contents[i].split('\t')[0]))
        # Second split string -> Sensor 1 reading
        expS1.append(float(contents[i].split('\t')[1]))
        # Third split string -> Sensor 2 reading
        expS2.append(float(contents[i].split('\t')[2]))

        # Fourth split string -> Sensor 3 reading
        expS3.append(float(contents[i].split('\t')[4]))

        # Fifth split string -> Sensor 4 reading
        expS4.append(float(contents[i].split('\t')[3]))
        
#         # Impact Location X Coordinate
#         X.append(int(file.get_name().split('_')[2]))
            
#         # Impact Location Y Coordinate
#         Y.append(int(file.get_name().split('_')[3]))

        
        
        
        
    
    
    f.close() # Closing the access to the text file
    colNames = ['Time', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4','X','Y']
    txt_df = pd.DataFrame(columns = colNames)
    
    txt_df['Time'] = expT
    # expS1 = dataFilter(expS1,4,10000,len(expS1)/expT[-1])
    # expS2 = dataFilter(expS1,4,10000,len(expS2)/expT[-1])
    # expS3 = dataFilter(expS1,4,10000,len(expS3)/expT[-1])
    # expS4 = dataFilter(expS1,4,10000,len(expS4)/expT[-1])
    txt_df['Sensor 1'] = expS1
    txt_df['Sensor 2'] = expS2
    txt_df['Sensor 3'] = expS3
    txt_df['Sensor 4'] = expS4
    
    
    # Deleting the read text file from local directory
    os.remove(file.get_name())
    
    
    
    # content = oc.get_file(directory[1] + 'SE1_a_t/' + file.get_name())
    # Loading the .text file
    # os.remove(file.get_name()) # Removing the file from the local directory once it has been read

    # Appending the dataframe correspnding to each file to the experimental data list
    exp_data.append(txt_df)
        
exp_data = np.array(exp_data)
        
#%% Final Experimental Validation Data
                
# Cutting Garbage Data
for ii in range(0,exp_data.shape[0]):
    dataCut = expDataSegment(exp_data[ii,:,:])
    dataCut[:,1:5] = -dataCut[:,1:5] # Inverting the Experimental Data
    expData.append(dataCut) 
                
expData = np.array(expData, dtype = object)

# Normalisation
# expData = dataNormal(expData)

# Using MinMaxScaler method
for i in range(0,expData.shape[0]):
    expData[i,:,1:5] = MinMaxScaler(feature_range = (-1,1)).fit_transform(expData[i,:,1:5]) # Normalising between -1 and 1

# Segmenting the data after highest peak
finalpredData = np.zeros((expData.shape[0],dataNum,7))
for ii in range(0,expData.shape[0]):
    finalpredData[ii,:,:] = dataSegment(expData[ii,:,:],dataNum)

    
    
#%% Saving experimental and numerical parsed data
finalexpData_rshpd = finalexpData.reshape(finalexpData.shape[0], -1)
np.savetxt('saved_variables/finalexpData.txt',finalexpData_rshpd)

# Preprocessed numerical simulation data
finalNumData_rshpd = finalNumData.reshape(finalNumData.shape[0], -1)
np.savetxt('saved_variables/finalNumData.txt',finalNumData_rshpd)

#%% The above numerical and experimental data is unbalanced
# Splitting numerical and Experimental data into Sensor Sets
Sxy1 = finalNumData[:,:,1:5] # Sensor Numerical Data
Sxy2 = finalexpData[:12,:,1:5] # Half of experimental data
Sxy = np.concatenate((Sxy1,Sxy2),axis = 0)
#Sxy = Sxy.reshape(-1,4*dataNum,1) # Reshaping
Oxy1 = finalNumData[:,0,5:] # Impact Locations in Numerical Data
Oxy2 = finalexpData[:12,0,5:] # Impact Locations in Experimental Data
Oxy = np.concatenate((Oxy1,Oxy2),axis = 0) # Training and Validation Data outputs

# Splitting the data into Training and Validation Data
Sxy_train, Sxy_val, Oxy_train, Oxy_val = train_test_split(Sxy, Oxy, test_size=0.1, shuffle=True)

#%% Test Data
Sxy_test = finalexpData[12:,:,1:5] # Sensor Experimental Data
Oxy_test = finalexpData[12:,0,5:] # Impact Locations

#%% Preediction Data
Sxy_pred = finalpredData[:,:,1:5]
# Actual experimental impact  (Provided Later)
Oxy_act = np.array([[250,250],
[245,255],
[265,235],
[250,265],
[265,255],
[255,265],
[265,235],
[225,200],
[235,235],
[235,300],
[250,265],
[200,225],
[235,235],
[300,285],
[270,300],
[260,320],
[245,255],
[250,250]])

#%% Normalising training, validation and test data outputs
Oxy_scale = MinMaxScaler().fit(Oxy)
Oxy_train_n = Oxy_scale.transform(Oxy_train)
Oxy_val_n = Oxy_scale.transform(Oxy_val)
Oxy_test_n = Oxy_scale.transform(Oxy_test)

#%% Locally Saving Data
# UNBALANCED DATA
# Training and Validation Data outputs
# 2D array
np.savetxt('saved_variables/UnbalancedData/Oxy.txt',Oxy)
np.savetxt('saved_variables/UnbalancedData/Oxy_train.txt',Oxy_train)
np.savetxt('saved_variables/UnbalancedData/Oxy_val.txt',Oxy_val)

# Testing Data outputs
# 2D array
np.savetxt('saved_variables/Oxy_test.txt',Oxy_test)

# Normalised Testing Data outputs
# 2D array
np.savetxt('saved_variables/Oxy_test_n.txt',Oxy_test_n)

# Sensor Test Data
# Reshaping 3D arrays to 2D arrays
Sxy_test_rshpd = Sxy_test.reshape(Sxy_test.shape[0], -1)
np.savetxt('saved_variables/Sxy_test.txt',Sxy_test_rshpd)

# Training and Validation sensor data
# Reshaping 3D arrays to 2D arrays
Sxy_rshpd = Sxy.reshape(Sxy.shape[0], -1)
Sxy_train_rshpd = Sxy_train.reshape(Sxy_train.shape[0], -1)
Sxy_val_rshpd = Sxy_val.reshape(Sxy_val.shape[0], -1)
np.savetxt('saved_variables/UnbalancedData/Sxy.txt',Sxy_rshpd)
np.savetxt('saved_variables/UnbalancedData/Sxy_train.txt',Sxy_train_rshpd)
np.savetxt('saved_variables/UnbalancedData/Sxy_val.txt',Sxy_val_rshpd)

# Prediction Data
# Actual Impact Locations (Desired Outputs)
# 2D array
np.savetxt('saved_variables/Oxy_act.txt',Oxy_act)

# Sensor Test Data
# Reshaping 3D arrays to 2D arrays
Sxy_pred_rshpd = Sxy_pred.reshape(Sxy_pred.shape[0], -1)
np.savetxt('saved_variables/Sxy_pred.txt',Sxy_pred_rshpd)