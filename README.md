Documentation of the project

# 1. Install Packages 
## Use the requirements.txt file to install all necessary packages

In command Line:

```
cd impact-location-prediction-using-ann-group-14
```
```
pip install -r requirements.txt
```

# 2. Data
Run `dataParser.py` to parse all numerical data, experimental validation data and group specific data from the remote directory Sciebo maintained by RWTH Aachen University. `dataParser.py` saves the unbalanced datasets to the local directory `saved_variables`. Run `dataBalance.py` to save the completely balanced dataset to the local directory `saved_variables/Balanced_Data`. Run `skewedData.py` to save the datasets for various cases of skewness in the local directory `saved_variables/Skewed_Data`.

# 3. CNN Architecture
## cnnXYModel.py is the main program used to call different saved trained and validated CNN models. This program uses the directories saved_model and saved_variable to import the desired models.

### Details on the Saved Models:

- cnn_model_xy_Balanced --> Single model predicting both X and Y Coordinates with balanced data.
- cnn_single_model_x_Balanced --> Independent model predicting X Coordinates with balanced data.
- cnn_single_model_y_Balanced --> Independent model predicting Y Coordinates with balanced data.
- cnn_model_xy_25Skewness --> Single model predicting both X and Y Coordinates with 25% of balanced data.
- cnn_model_xy_50Skewness --> Single model predicting both X and Y Coordinates with 50% of balanced data.
- cnn_model_xy_75Skewness --> Single model predicting both X and Y Coordinates with 75% of balanced data.

# 3. FFNN Architecture
## ffnnXYModel.py is the main program used to call saved models for the Feed-Forward Architecture. This program also uses the directories saved_model and saved_variable to import the desired models.

### Details on the Saved Models:

- ffnn_model_xy_Balanced --> Single Feed-Forward model predicting both X and Y Coordinates with balanced data.

# 4. RNN Architecture 
## The RNN directory contains the config.yaml file used to define the Recurrent Neural Network Architecture.

>Note: The Data has been preprocessed in the jupter notebook rnn_test.ipynb and saved to rnn.csv. This program also uses the directory saved_variable to import the desired variables.

In order to train this model run the command:

```
cd RNN
```
```
ludwig train --dataset /rnn.csv --config /config.yaml
```
# 5. Details on the Saved Variables:

- finalexpData.txt --> Complete experimental data.
- finalNumData.txt --> Complete numerical/simulated data.
- Oxy_act.txt --> Desired Output for group-specific data.
- Oxy_test_n.txt --> Normalized desired output for the experimental data.
- Oxy_test.txt -->  Desired output for the experimental data.
- Oxy.txt --> Desired output for the numerically simulated data.
- Sxy_pred.txt --> Sensor Data for group-specific data.
- Sxy_test.txt --> Sensor Data for the experimental data.
- Sxy.txt --> Sensor Data for the numerical data.

## 6. CIE_ProjectB_Group14.ipynb
The Jupyter Notebook submitted for the presentation is CIE_ProjectB_Group14.ipynb

## 7. dataParser.py
The dataParser.py file is the root file used to execute all the corresponding python methods (dataFilter.py, dataNormal.py, dataSegment.py, expDataSegment.py, mirrorPoint.py, pointQuad.py read_data.py)

## 8. dataBalance.py
The dataBalance.py is used to save the balanced data in the saved_variables directory (here the dataAugment.py and removeArrFromList.py are used under the hood)

## 9. skewedData.py
The skewedData.py is used to save the skewed data in the saved_variables directory (here the dataAugment.py and removeArrFromList.py are used under the hood)
