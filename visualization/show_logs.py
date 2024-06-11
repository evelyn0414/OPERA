# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:12:40 2024
 
"""

import pandas as pd
import matplotlib.pyplot as plt


# Read the CSV file
df = pd.read_csv('src/plot/metrics.csv')

# Specify the column you want to read
column_name = 'valid_acc'

# Extract the column values to a list and drop NaN values
data_list = df[column_name].dropna().tolist()

# Plot the curve
plt.figure(dpi=300)
plt.plot(data_list)
plt.title('Validation Acc', fontsize=18)
plt.xlim((-5, 151))
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Value', fontsize=18)
# plt.show()
plt.savefig("fig/training/operaCE_acc.png")


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
plt.xlim((-5, 151))
# plt.show()
plt.savefig("fig/training/operaCE_val_loss.png")

# # labels = {0: "covidbreath", 1:"covidcough", 2: "icbhi", 3: "coughvid", 4: "hf_lung", 5: "covidUKexhalation", 6:"covidUKcough"}
# labels = ["covidbreath", "covidcough", "icbhi",  "coughvid", "hf_lung", "covidUKexhalation", "covidUKcough"]

# # labels = ["covidbreath", "covidcough",  "coughvid", "covidUKexhalation", "covidUKcough"]

# # Plot the curve
# plt.figure(figsize=(6,4), dpi=300)
# for i in range(6):
#     # Specify the column you want to read
#     column_name = 'train' + str(i) + '_loss'
#     # column_name = 'train_loss'

#     # Extract the column values to a list and drop NaN values
#     non_nan_indices = df[column_name].dropna().index
#     data_list = df.loc[non_nan_indices, column_name].tolist()

#     plt.plot(non_nan_indices*50, data_list, '-o', markersize=2, label=labels[i])
# plt.title('training loss')
# plt.xlabel('Training step')
# plt.ylabel('loss')
# plt.legend()
# # plt.show()
# plt.savefig("fig/training/htsatrepeat200_loss.png")

