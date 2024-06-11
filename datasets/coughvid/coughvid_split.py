# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def cdf(data):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    return sorted_data, yvals


# Load the data from the CSV file
df = pd.read_csv(
    'F:\SSL\RespSounds\datasets\coughvid\public_dataset\metadata_compiled.csv')
# df = pd.read_csv('datasets\coughvid\public_dataset\metadata_compiled.csv')

# cough_detected = df['cough_detected'].tolist()

# # Calculate CDF
# x, y = cdf(cough_detected)

# # Plot CDF
# plt.step(x, y, where='post')
# plt.xlabel('cough_detected')
# plt.ylabel('CDF')
# plt.title('Cumulative Distribution Function (CDF)')
# plt.grid(True)
# plt.show()

df = df[df['cough_detected'] >= 0.8]

# Display the first few rows of the dataframe to understand its structure
print(df.head())
# More detailed statistics for selected columns
print('-------------------------------')
print("Age statistics:")  # 34
print(df['age'].describe())
print('-------------------------------')
print("Gender distribution:")  # 7:4
print(df['gender'].value_counts())
print('-------------------------------')
print("Respiratory condition distribution:")  # 85:10
print(df['status'].value_counts())

# Filter the dataset for females and males separately
females_df = df[df['gender'] == 'female']
males_df = df[df['gender'] == 'male']

# Select fixed numbers of females and males for the test set
female_test_uuids = females_df.sample(n=1031, random_state=0)['uuid']
male_test_uuids = males_df.sample(n=1924, random_state=0)['uuid']

# Combine these UUIDs to form the full test set
test_uuids = pd.concat([female_test_uuids, male_test_uuids])

# The rest of the data is for training and validation
train_val_uuids = df[~df['uuid'].isin(test_uuids)]

# # Split the UUIDs into training and testing sets
# train_val_uuids, test_uuids = train_test_split(df['uuid'], test_size=0.2, random_state=9)

# Now split the temporary training set into the final training set and validation set
train_uuids, val_uuids = train_test_split(
    train_val_uuids, test_size=0.25, random_state=100)  # 0.25 x 0.8 = 0.2

print(len(train_uuids), len(val_uuids), len(test_uuids))

# Create dataframes for training, validation, and testing sets
train_df = df[df['uuid'].isin(train_uuids['uuid'])]
healthy_training_df = train_df[train_df['status'] == 'healthy']
covid_training_df = train_df[train_df['status'] == 'COVID-19']

val_df = df[df['uuid'].isin(val_uuids['uuid'])]
healthy_val_df = val_df[val_df['status'] == 'healthy']
covid_val_df = val_df[val_df['status'] == 'COVID-19']

test_df = df[df['uuid'].isin(test_uuids)]
print('-------------------------------')
print("test Gender distribution:")
print(test_df['gender'].value_counts())
print('-------------------------------')
healthy_test_df = test_df[test_df['status'] == 'healthy']
covid_test_df = test_df[test_df['status'] == 'COVID-19']


np.save('coughvid__train_uuids.npy', train_uuids['uuid'].values)
np.save('coughvid__val_uuids.npy', val_uuids['uuid'].values)
np.save('coughvid_gender_test_uuids.npy', test_uuids.values)

# More detailed statistics for selected columns
print('-------------------------------')
print('training')
print('-------------------------------')
print("Age statistics:")
print(healthy_training_df['age'].describe())
print('-------------------------------')
print("Gender distribution:")
print(healthy_training_df['gender'].value_counts())
print('-------------------------------')
print("Respiratory condition distribution:")
print(healthy_training_df['status'].value_counts())
print('-------------------------------')
print('-------------------------------')
print("Age statistics:")
print(covid_training_df['age'].describe())
print('-------------------------------')
print("Gender distribution:")
print(covid_training_df['gender'].value_counts())
print('-------------------------------')
print("Respiratory condition distribution:")
print(covid_training_df['status'].value_counts())

print('-------------------------------')
print('testing')
# More detailed statistics for selected columns
print('-------------------------------')
print("Age statistics:")
print(healthy_test_df['age'].describe())
print('-------------------------------')
print("Gender distribution:")
print(healthy_test_df['gender'].value_counts())
print('-------------------------------')
print("Respiratory condition distribution:")
print(healthy_test_df['status'].value_counts())
print('-------------------------------')
print('-------------------------------')
print("Age statistics:")
print(covid_test_df['age'].describe())
print('-------------------------------')
print("Gender distribution:")
print(covid_test_df['gender'].value_counts())
print('-------------------------------')
print("Respiratory condition distribution:")
print(covid_test_df['status'].value_counts())

healthy_test_uuids = healthy_test_df.sample(n=2237, random_state=0)['uuid']
covid_test_uuids = covid_test_df.sample(n=172, random_state=0)['uuid']
test_uuids = pd.concat([healthy_test_uuids, covid_test_uuids])

test_df = df[df['uuid'].isin(test_uuids)]
print('-------------------------------')
print("Respiratory condition distribution:")
print(test_df['status'].value_counts())


np.save('coughvid_covid_test_uuids.npy', test_uuids.values)
