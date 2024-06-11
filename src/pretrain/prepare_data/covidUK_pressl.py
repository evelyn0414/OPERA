# -*- coding: utf-8 -*-
import argparse
import numpy as np
from tqdm import tqdm
from src.util import get_entire_signal_librosa
import os

### for pretraining
def preprocess_spectrogram_SSL(modality="modality", input_sec=2):
    
    #path = 'datasets/covidUK/'
    path = ''
    
    uids = np.load(path + modality + "_training_files.npy", allow_pickle=True).tolist()
    uids = [item for sublist in uids for item in sublist]
    
    uids_val = np.load(path + modality + "_val_files.npy", allow_pickle=True).tolist()
    uids_val = [item for sublist in uids_val for item in sublist]
    
    uids = uids_val + uids
    
    print('SSL training:', len(uids))
    
    invalid_data = 0

    filename_list = []

    # use metadata as outer loop to enable quality check
    for file in tqdm(uids):
        #print(file)
        userID = file.split('.')[0]
        if os.path.exists(path + 'audio/' + file):
            data = get_entire_signal_librosa(path + 'audio/', userID, spectrogram=True, input_sec=input_sec)

            if data is None:
                invalid_data += 1
                continue

        # saving to individual npy files
        np.save(path + "entire_spec_npy/" + userID + ".npy", data)
        filename_list.append("datasets/covidUK/entire_spec_npy/" + userID)
        
    np.save(path+"entire_" + modality + "_filenames.npy", filename_list)     
    print("finished preprocessing breathing: valid data", len(filename_list), "; invalid data", invalid_data)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default='exhalation')
    parser.add_argument("--input_sec", type=int, default=4)
    args = parser.parse_args()

    preprocess_spectrogram_SSL(modality=args.modality, input_sec=args.input_sec)
    