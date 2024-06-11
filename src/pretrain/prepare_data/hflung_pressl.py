import glob as gb
import argparse
import numpy as np
from tqdm import tqdm
from src.util import get_entire_signal_librosa

SR = 16000

def preprocess_entire_spectrogram(input_sec=8):
    
    #train_test = []
    filename_list = []
    invalid_data = 0
    
    path = 'datasets/hf_lung/HF_Lung_V1-master/train'
    sound_dir_loc = np.array(gb.glob(path + "/*.wav"))
    
    
    for i in tqdm(range(sound_dir_loc.shape[0])):
        filename = sound_dir_loc[i].strip().split('.')[0]
        fileID = filename.split('/')[-1].split('.')[0]
        print('==')
        print(fileID)
      
        data = get_entire_signal_librosa('', filename, spectrogram=True, input_sec=input_sec)

        if data is None:
            invalid_data += 1
            continue

        np.save("datasets/hf_lung/entire_spec_npy/" + fileID + ".npy", data)
        filename_list.append("datasets/hf_lung/entire_spec_npy/" + fileID)
        
    path = 'datasets/hf_lung/HF_Lung_V1_IP-main/train'
    sound_dir_loc = np.array(gb.glob(path + "/*.wav"))
    
    for i in tqdm(range(sound_dir_loc.shape[0])):
        filename = sound_dir_loc[i].strip().split('.')[0]
        fileID = filename.split('/')[-1].split('.')[0]
        print('==')
        print(fileID)
      
        data = get_entire_signal_librosa('', filename, spectrogram=True, input_sec=input_sec)

        if data is None:
            invalid_data += 1
            continue

        np.save("datasets/hf_lung/entire_spec_npy/" + fileID + ".npy", data)
        filename_list.append("datasets/hf_lung/entire_spec_npy/" + fileID)
        
    np.save("datasets/hf_lung/entire_spec_filenames.npy", filename_list)
    print("invalid_data", invalid_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_sec", type=int, default=8)
    args = parser.parse_args()

    preprocess_entire_spectrogram(input_sec=args.input_sec)


