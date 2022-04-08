# Feed Forward Neural Network
#%% Libraries
import glob
import os
import re
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv1D,MaxPooling1D,AveragePooling1D
from keras.layers.advanced_activations import LeakyReLU
from scipy import signal
from numpy.random import seed
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

#%% Directories for local variables
dir1 = 'saved_variables/Balanced_Data/'
dir2 = 'saved_variables/Skewed_Data/Q2/'
dir3 = 'saved_variables/Skewed_Data/Q2Q4/'
dir4 = 'saved_variables/Skewed_Data/Q2Q4Q3/'
dir = dir1

#%% Loading training, validation and test data
# Loading Sxy_test and Oxy_test data from the local repository
# Test Data ouput
Oxy_test = np.loadtxt('saved_variables/Oxy_test.txt')
# Test Data Normalised output
Oxy_test_n = np.loadtxt('saved_variables/Oxy_test_n.txt')

# Test Data Sensor Data
Sxy_test_load = np.loadtxt('saved_variables/Sxy_test.txt')
# Reshaping Sxy_test_load to a 3D array
Sxy_test = Sxy_test_load.reshape(Sxy_test_load.shape[0],Sxy_test_load.shape[1] // 4 , 4)

# Training Data Sensor Data
Sxy_train_load = np.loadtxt(dir + 'Sxy_train.txt')
# Reshaping Sxy_test_load to a 3D array
Sxy_train = Sxy_train_load.reshape(Sxy_train_load.shape[0],Sxy_train_load.shape[1] // 4 , 4)

# Training Data ouput
Oxy_train = np.loadtxt(dir + 'Oxy_train.txt')

# Validation Data output
Oxy_val = np.loadtxt(dir + 'Oxy_val.txt')

# Validation Data Sensor Data
Sxy_val_load = np.loadtxt(dir + 'Sxy_val.txt')
# Reshaping Sxy_test_load to a 3D array
Sxy_val = Sxy_val_load.reshape(Sxy_val_load.shape[0],Sxy_val_load.shape[1] // 4 , 4)

# Complete Data
Sxy_load = np.loadtxt(dir + 'Sxy.txt')
# Reshaping Sxy_test_load to a 3D array
Sxy = Sxy_load.reshape(Sxy_load.shape[0],Sxy_load.shape[1] // 4 , 4)

Oxy = np.loadtxt(dir + 'Oxy.txt')

# Group Specific Experimental Data (Prediction Data)
Oxy_act = np.loadtxt('saved_variables/Oxy_act.txt')
# Normalising Oxy_act
Sxy_pred_load = np.loadtxt('saved_variables/Sxy_pred.txt')
# Reshaping Sxy_pred_load to a 3D array
Sxy_pred = Sxy_pred_load.reshape(Sxy_pred_load.shape[0],Sxy_load.shape[1] // 4 , 4)

#%% Normalising Training and Validation Outputs
# Using MinMaxScalar()
Oxy_scale = MinMaxScaler().fit(Oxy) # Scale for normalisation decided on the basis of Oxy
Oxy_train_n = Oxy_scale.transform(Oxy_train)
Oxy_val_n = Oxy_scale.transform(Oxy_val)
Oxy_act_n = Oxy_scale.transform(Oxy_act)

#%% Using Savgol filtering to reduce noise from the input sensor data
# Sxy_test (test data) and Sxy (training plus validation data) and Sxy_pred (Group specific experimental data/prediction data)
for i in range(Sxy.shape[0]):
    Sxy[i,:,0] = savgol_filter(Sxy[i,:,0], 61, 3)
    Sxy[i,:,1] = savgol_filter(Sxy[i,:,1], 61, 3)
    Sxy[i,:,2] = savgol_filter(Sxy[i,:,2], 61, 3)
    Sxy[i,:,3] = savgol_filter(Sxy[i,:,3], 61, 3)
    
for i in range(Sxy_test.shape[0]):
    Sxy_test[i,:,0] = savgol_filter(Sxy_test[i,:,0], 61, 3)
    Sxy_test[i,:,1] = savgol_filter(Sxy_test[i,:,1], 61, 3)
    Sxy_test[i,:,2] = savgol_filter(Sxy_test[i,:,2], 61, 3)
    Sxy_test[i,:,3] = savgol_filter(Sxy_test[i,:,3], 61, 3)
    
for i in range(Sxy_pred.shape[0]):
    Sxy_pred[i,:,0] = savgol_filter(Sxy_pred[i,:,0], 61, 3)
    Sxy_pred[i,:,1] = savgol_filter(Sxy_pred[i,:,1], 61, 3)
    Sxy_pred[i,:,2] = savgol_filter(Sxy_pred[i,:,2], 61, 3)
    Sxy_pred[i,:,3] = savgol_filter(Sxy_pred[i,:,3], 61, 3)

# %% Building and running the model and then locally saving it
# epochs = 100 # Number of epochs
# seed(1000)
# tf.random.set_seed(1000)
# model_xy = Sequential()
# model_xy.add(Flatten(input_shape=(1000,4)))
# model_xy.add(Dense(1000, activation='relu'))
# model_xy.add(LeakyReLU(alpha=0.1))
# model_xy.add(Dense(326, activation='linear'))
# model_xy.add(Dropout(0.2))
# model_xy.add(Dense(2))

# # Compiling the model
# model_xy.compile(loss="mse", optimizer="adam", metrics=['mse','mae'])
# model_xy.summary()

# callback = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=5)

# # Training the Model on the balanced data
# model_train_xy = model_xy.fit(Sxy_train, Oxy_train_n, epochs=epochs, callbacks = [callback], verbose=1, validation_data = (Sxy_val, Oxy_val_n), shuffle = True) # With Validation

# # Save the entire model as a SavedModel.
# model_xy.save('saved_model/ffnn_model_xy_Balanced')



#%% Loading the saved model and then using the test data on saved model
# For continuous testing, the code above this doesn't need to be executed againin and again
model_xy = tf.keras.models.load_model('saved_model/ffnn_model_xy_Balanced')

#%% Testing the model on the Test Data

print("******** Prediction on Test Data *********")
Oxy_pred = model_xy.predict(Sxy_test) # For regression of Test Data
print(model_xy.evaluate(Sxy_test, Oxy_test_n)) 
print('MSE: %4f' %mean_squared_error(Oxy_test_n, Oxy_pred))
print('MAE: %4f' %mean_absolute_error(Oxy_test_n, Oxy_pred))

# Denormalising predicted impact locations
obj = MinMaxScaler().fit(Oxy)
Oxy_pred_DN = obj.inverse_transform(Oxy_pred)
np.round(Oxy_pred_DN), Oxy_test

# Denormalised MSE and MAE
print('MSE: %4f' %mean_squared_error(Oxy_pred_DN, Oxy_test))
print('MAE: %4f' %mean_absolute_error(Oxy_pred_DN, Oxy_test))

#%% Testing the model on the Prediction Data

print("******** Prediction on Prediction Data *********")
Oxy_pred = model_xy.predict(Sxy_pred) # For regression of Test Data
print(model_xy.evaluate(Sxy_pred, Oxy_act_n)) 
print('MSE: %4f' %mean_squared_error(Oxy_act_n, Oxy_pred))
print('MAE: %4f' %mean_absolute_error(Oxy_act_n, Oxy_pred))

# Denormalising predicted impact locations
obj = MinMaxScaler().fit(Oxy)
Oxy_pred_DN = obj.inverse_transform(Oxy_pred)
np.round(Oxy_pred_DN), Oxy_act

# Denormalised MSE and MAE
print('MSE: %4f' %mean_squared_error(Oxy_pred_DN, Oxy_act))
print('MAE: %4f' %mean_absolute_error(Oxy_pred_DN, Oxy_act))


#%% Predicted X Comparison Plot
# Sx_ax = range(Oxy_pred.shape[0])
# plt.plot(Sx_ax, Oxy_act[:,0], lw=0.8, color='blue', label='original')
# plt.plot(Sx_ax, Oxy_pred_DN[:,0], lw=0.8, color='red', label='predicted')
# plt.title('X Prediction (Feed Forward Neural Network)')
# plt.legend()
# plt.grid()
# plt.show()


# #%% Predicted Y Comparison Plot
# Sy_ax = range(Oxy_pred.shape[0])
# plt.plot(Sy_ax, Oxy_act[:,1], lw=0.8, color='blue', label='original')
# plt.plot(Sy_ax, Oxy_pred_DN[:,1], lw=0.8, color='red', label='predicted')
# plt.title('Y Prediction (Feed Forward Neural Network)')
# plt.legend()
# plt.grid()
# plt.show()

#%% Impact Location Coordinates (X,Y) Comparison
Sx_ax = range(1,Oxy_pred.shape[0]+1)
Sy_ax = range(1,Oxy_pred.shape[0]+1)
fig,(ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8,10))
ax1.plot(Sx_ax, Oxy_act[:,0], lw=0.8, color='blue', label='original')
ax1.plot(Sx_ax, Oxy_pred_DN[:,0], lw=0.8, color='red', label='predicted')
ax1.set_title('X Prediction (Feed Forward Neural Network)')
ax1.set_xlabel('Trial Number')
ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
ax1.set_xlim(0, 19)
ax1.legend()
ax1.grid()
ax2.plot(Sy_ax, Oxy_act[:,1], lw=0.8, color='blue', label='original')
ax2.plot(Sy_ax, Oxy_pred_DN[:,1], lw=0.8, color='red', label='predicted')
ax2.set_title('Y Prediction (Feed Forward Neural Network)')
ax2.set_xlabel('Trial Number')
ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
ax2.set_xlim(0, 19)
ax2.legend()
ax2.grid()
plt.show()