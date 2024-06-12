# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:12:40 2024
 
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
if not os.path.exists("fig/training/"):
    os.makedirs("fig/training/")

def smooth_list(input_list):
    # Ensure the list has enough elements to smooth
    if len(input_list) < 5:
        raise ValueError("List must contain at least 5 elements to apply smoothing.")

    smoothed_list = []
    for i in range(len(input_list)):
        # Determine the window bounds
        start = max(0, i - 2)
        end = min(len(input_list), i + 3)
        
        # Extract the window and calculate the average
        window = input_list[start:end]
        window_average = sum(window) / len(window)
        
        smoothed_list.append(window_average)
    
    return smoothed_list


# Read the CSV file
df = pd.read_csv('visualization/metrics-operaCT.csv')

# # Specify the column you want to read
# column_name = 'valid_acc'

# # Extract the column values to a list and drop NaN values
# data_list = df[column_name].dropna().tolist()

# # Plot the curve
# plt.figure(dpi=300)
# plt.plot(data_list)
# plt.title('Validation Acc', fontsize=18)
# plt.xlim((-5, 151))
# plt.xlabel('Epoch', fontsize=18)
# plt.ylabel('Value', fontsize=18)
# # plt.show()
# plt.savefig("fig/training/operaCT_acc.png")


# Specify the column you want to read
column_name = 'valid_loss'

# Extract the column values to a list and drop NaN values
data_list = df[column_name].dropna().tolist()

# Plot the curve
plt.figure(dpi=300)
plt.plot(data_list)
plt.title('Validation Loss', fontsize=18)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Value', fontsize=18)
plt.xlim((-5, 201))
# plt.show()
plt.savefig("fig/training/operaCT_val_loss.png")

# labels = {0: "covidbreath", 1:"covidcough", 2: "icbhi", 3: "coughvid", 4: "hf_lung", 5: "covidUKexhalation", 6:"covidUKcough"}
# labels = ["covid19sounds(breath)", "covid19sounds(cough)", "icbhi",  "coughvid", "hf_lung", "UKcovid19(exhalation)", "UKcovid19(cough)"]
labels = ["covid19sounds(breath)", "covid19sounds(cough)", "icbhi", "icbhicycle",  "coughvid", "hf_lung", "UKcovid19(exhalation)", "UKcovid19(cough)"]
# labels = ["covidbreath", "covidcough",  "coughvid", "covidUKexhalation", "covidUKcough"]

# Plot the curve
plt.figure(figsize=(6,4), dpi=300)
for i in range(8):
    if i == 3:
        continue
    # Specify the column you want to read
    column_name = 'train' + str(i) + '_loss'
    # column_name = 'train_loss'

    # Extract the column values to a list and drop NaN values
    non_nan_indices = df[column_name].dropna().index
    data_list = df.loc[non_nan_indices, column_name].tolist()
    data_list = smooth_list(data_list)

    plt.plot(non_nan_indices*50, data_list, '-o', markersize=1.5, label=labels[i])
plt.title('Training Loss', fontsize=14)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig("fig/training/operaCT_training_loss.png")

