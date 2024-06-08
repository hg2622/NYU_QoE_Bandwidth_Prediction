#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:08:16 2024

@author: kevinguo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("7Train1.csv",header=None)
bandwidth=data.iloc[:,0]
length=len(bandwidth)


np.random.seed(0)
N = length  # Number of samples
filter_length = 5  # Length of the filter
delta = 0.01 


x=bandwidth#original data
predicted = np.zeros(N)  # Initialize predicted array
w = np.zeros((filter_length, N))  # Initialize weight matrix
P = (1 / delta) * np.eye(filter_length)  # Initialize inverse correlation matrix
e = np.zeros(N)  # Initialize error array


# Initialize input buffer
x_buffer = np.zeros(filter_length)

for k in range(filter_length, N-1):
    x_buffer = x[k:k-filter_length:-1]  # Current input buffer

    # Predict next value
    predicted[k+1] = np.dot(w[:, k], x_buffer)

    # Error calculation
    e[k+1] = x[k+1] - predicted[k+1]

    # Gain vector
    K = np.dot(P, x_buffer) / (1 + np.dot(np.dot(x_buffer.T, P), x_buffer))

    # Update weights
    w[:, k+1] = w[:, k] + K * np.sign(e[k+1])

    # Update inverse correlation matrix
    P = P - np.outer(K, np.dot(x_buffer.T, P))
    
# Calculate RMSE
rmse = np.sqrt(np.mean(e[filter_length+1:]**2))
mae = np.mean(np.abs(e[filter_length+1:]))
print("RMSE: ", rmse)
print("MAE: ", mae)

mean=[np.mean(x)]*N
difference=mean-x
error_ratio_rmse=np.sqrt(np.mean(difference**2))
error_ratio_mae=np.mean(np.abs(difference))

print("Error ratio for RMSE: ", rmse/error_ratio_rmse*100)
print("Error ratio for MAE: ", mae/error_ratio_mae*100)



plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, N+1), x, 'b', label='Original Signal')
plt.plot(np.arange(1, N+1), predicted, 'r--', label='Predicted Signal')
plt.xlim((1200,1400))
plt.title('Original vs. Predicted Signal (RMSE)')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Signal Value')

