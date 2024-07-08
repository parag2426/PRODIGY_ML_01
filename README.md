# Linear Regression for Housing Price Prediction

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)

## Introduction

This project uses a linear regression model to predict housing prices based on various features. The dataset used for this project is from a housing dataset that includes 81 features about each house.

## Dataset

The dataset used in this project is a CSV file containing housing data. The dataset is divided into training and test sets.

- **Training data**: Used to train the model.
- **Test data**: Used to evaluate the model.


## Usage

Import the libraries and load the data:
python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
House_data = pd.read_csv('path/to/train.csv')
House_data2 = pd.read_csv('path/to/test.csv')
Preprocess the data and train the model:
python
Copy code
# Handle missing values
imputer_num = SimpleImputer(strategy='mean')
House_data[numerical_cols] = imputer_num.fit_transform(House_data[numerical_cols])
House_data = House_data.fillna('None')

# Encode categorical features
label_encoders = {}
for column in categorical_cols:
    label_encoders[column] = LabelEncoder()
    House_data[column] = label_encoders[column].fit_transform(House_data[column])

# Train the model
model = LinearRegression()
x = House_data.drop('SalePrice', axis=1)
y = House_data['SalePrice']
model.fit(x, y)
Make predictions and evaluate the model:

y_pred = model.predict(x)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
print(f'MSE: {mse}, RMSE: {rmse}, R2: {r2}')
## Model Training and Evaluation
The model is trained using the linear regression algorithm. The performance of the model is evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).

MSE: Measures the average squared difference between the predicted and actual values.
RMSE: The square root of MSE, representing the standard deviation of the residuals.
R2: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
## Results
The model achieved the following performance metrics on the training data:

MSE: 925,105,572.84
RMSE: 30,415.55
R2: 0.853
These metrics indicate that the model has a good fit to the training data, but further evaluation on the test data is necessary to confirm its performance
