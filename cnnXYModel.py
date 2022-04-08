# CNN Model (With 2 Output layers)

#%% Libraries and modules
import matplotlib.pyplot as plt
import pandas as pd
import os
from io import BytesIO
import owncloud
from datetime import datetime
import numpy as np
from numpy.random import seed
from scipy import signal
import re
import scipy.io
from scipy.fft import fft, fftfreq, rfft, rfftfreq # For FFT
import pickle # For storing variables in local txt files

# Neural Network Based Modules
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv1D,MaxPooling1D, AveragePooling1D
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler # For Data Normalisation
from keras.callbacks import EarlyStopping
# import visualkeras
from keras.models import load_model # for saving the trained and validated model and all its parameters

#%% Directories for local variables
dir0 = 'saved_variables/'
dir1 = 'saved_variables/Balanced_Data/'
dir2 = 'saved_variables/Skewed_Data/Q2/'
dir3 = 'saved_variables/Skewed_Data/Q2Q4/'
dir4 = 'saved_variables/Skewed_Data/Q2Q4Q3/'
dir = dir3

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
Oxy_scale = MinMaxScaler().fit(Oxy) # Scaling based on outputs of training and validation data
Oxy_act_n = Oxy_scale.transform(Oxy_act)
Sxy_pred_load = np.loadtxt('saved_variables/Sxy_pred.txt')
# Reshaping Sxy_pred_load to a 3D array
Sxy_pred = Sxy_pred_load.reshape(Sxy_pred_load.shape[0],Sxy_load.shape[1] // 4 , 4) 

#%% Normalising Training and Validation Outputs
# Using MinMaxScalar()
Oxy_scale = MinMaxScaler().fit(Oxy) # Scale for normalisation decided on the basis of Oxy
Oxy_train_n = Oxy_scale.transform(Oxy_train)
Oxy_val_n = Oxy_scale.transform(Oxy_val)

#%% CNN Model
# Number of Epochs
# epochs = 100
# seed(1000)
# tf.random.set_seed(1000)
# model_xy = Sequential()
# model_xy.add(Conv1D(16, kernel_size=8, kernel_initializer='he_uniform', strides=1, padding='causal', activation='relu', input_shape=(1000,4)))
# #model_xy.add(Conv1D(16, kernel_size=6, kernel_initializer='he_uniform', strides=4, padding='causal', activation='relu', input_shape=(4*dataNum,1)))
# model_xy.add(LeakyReLU(alpha=0.1))
# # model_xy.add(Dropout(0.95))
# model_xy.add(AveragePooling1D(pool_size=10))
# # model_xy.add(MaxPooling1D(pool_size=8))
# model_xy.add(Conv1D(32, kernel_size=3, padding='causal', activation='linear'))
# model_xy.add(LeakyReLU(alpha=0.1))
# # model_xy.add(Dropout(0.95))
# model_xy.add(MaxPooling1D(pool_size=5))
# model_xy.add(Conv1D(64, kernel_size=2, padding='causal', activation='linear'))
# model_xy.add(LeakyReLU(alpha=0.1))
# # model_xy.add(Dropout(0.95))
# model_xy.add(MaxPooling1D(pool_size=2))
# model_xy.add(Flatten())
# model_xy.add(Dense(128))
# model_xy.add(LeakyReLU(alpha=0.1))
# model_xy.add(Dropout(0.7))
# model_xy.add(Dense(2))

# # Compiling the CNN Model
# model_xy.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse','mae']) # for regression problem
# # model_y.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) # for classification problem

# callback = tf.keras.callbacks.EarlyStopping(monitor = 'mae', patience = 5)

# # Training the model
# model_train_xy = model_xy.fit(Sxy_train, Oxy_train_n, epochs=epochs, callbacks = [callback], verbose=1, validation_data = (Sxy_val, Oxy_val_n), shuffle = True) # With Validation


# # Save the entire model as a SavedModel.
# model_xy.save('saved_model/cnn_model')

# For continuous testing, the code above this doesn't need to be executed again and again
model_xy = tf.keras.models.load_model('saved_model/cnn_model_xy_50Skewness')

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
# plt.title('X Prediction (Convolutional Neural Network)')
# plt.legend()
# plt.grid()
# plt.show()


# #%% Predicted Y Comparison Plot
# Sy_ax = range(Oxy_pred.shape[0])
# plt.plot(Sy_ax, Oxy_act[:,1], lw=0.8, color='blue', label='original')
# plt.plot(Sy_ax, Oxy_pred_DN[:,1], lw=0.8, color='red', label='predicted')
# plt.title('Y Prediction (Convolutional Neural Network)')
# plt.legend()
# plt.grid()
# plt.show()

#%% Impact Location Coordinates (X,Y) Comparison
Sx_ax = range(1,Oxy_pred.shape[0]+1)
Sy_ax = range(1,Oxy_pred.shape[0]+1)
fig,(ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8,10))
ax1.plot(Sx_ax, Oxy_act[:,0], lw=0.8, color='blue', label='original')
ax1.plot(Sx_ax, Oxy_pred_DN[:,0], lw=0.8, color='red', label='predicted')
ax1.set_title('X Prediction (Convolutional Neural Network)')
ax1.set_xlabel('Trial Number')
ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
ax1.set_xlim(0, 19)
ax1.legend()
ax1.grid()
ax2.plot(Sy_ax, Oxy_act[:,1], lw=0.8, color='blue', label='original')
ax2.plot(Sy_ax, Oxy_pred_DN[:,1], lw=0.8, color='red', label='predicted')
ax2.set_title('Y Prediction (Convolutional Neural Network)')
ax2.set_xlabel('Trial Number')
ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
ax2.set_xlim(0, 19)
ax2.legend()
ax2.grid()
plt.show()



