# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

participant_file = 'audio_metadata.csv'
split_file = 'train_test_splits.csv'

 
participant_data = pd.read_csv(participant_file)
split_data = pd.read_csv(split_file)

exhalation_length = participant_data['exhalation_length'].tolist()
print(np.nanmean(exhalation_length))

cough_length = participant_data['cough_length'].tolist()
print(np.nanmean(cough_length))

# Merge participant data with split data
merged_data = pd.merge(participant_data, split_data, on='participant_identifier')

training_files = {}
testing_files = {}
val_files = {}

for index, row in merged_data.iterrows():
    participant_id = row['participant_identifier']
    exhalation_file = row['exhalation_file_name']
    split = row['splits']
    
    if split == 'train':
        training_files.setdefault(participant_id, []).append(exhalation_file) #dictionary
    elif split == 'test':
        testing_files.setdefault(participant_id, []).append(exhalation_file)
    elif split == 'val':
        val_files.setdefault(participant_id, []).append(exhalation_file)
        
    
# np.save('exhalation_training_files.npy', np.array(list(training_files.values())))
# np.save('exhalation_val_files.npy', np.array(list(val_files.values())))
# np.save('exhalation_testing_files.npy', np.array(list(testing_files.values())))


training_files = {}
testing_files = {}
val_files = {}

for index, row in merged_data.iterrows():
    participant_id = row['participant_identifier']
    exhalation_file = row['cough_file_name']
    split = row['splits']
    
    if split == 'train':
        training_files.setdefault(participant_id, []).append(exhalation_file) #dictionary
    elif split == 'test':
        testing_files.setdefault(participant_id, []).append(exhalation_file)
    elif split == 'val':
        val_files.setdefault(participant_id, []).append(exhalation_file)
        
        
# np.save('cough_training_files.npy', np.array(list(training_files.values())))
# np.save('cough_val_files.npy', np.array(list(val_files.values())))
# np.save('cough_testing_files.npy', np.array(list(testing_files.values())))