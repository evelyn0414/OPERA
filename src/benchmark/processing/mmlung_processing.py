import glob as gb
import argparse
import librosa
import collections
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import random
import scipy.signal as sg
import soundfile
import os
from os.path import exists

data_dir = "datasets/mmlung/Trimmed_Data_from_phone/"
meta_dir = "datasets/mmlung/"
feature_dir = "feature/mmlung_eval/"

Used_modality = ['Deep_Breath_file', 'O_Single_file']

def process_label():
    df = pd.read_excel(meta_dir + 'All_path.xlsx')
    labels = []
    for y in ['FVC','FEV1',	'FEV1/FVC']:
        labels.append(df[y].tolist())
     
    labels = np.array(labels).T     

    # print(labels)
    np.save(feature_dir + "label.npy", labels)

def extract_and_save_embeddings_baselines(feature="opensmile"):
    from src.benchmark.baseline.extract_feature import extract_opensmile_features, extract_vgg_feature, extract_clap_feature, extract_audioMAE_feature
    
    df = pd.read_excel(meta_dir + 'All_path.xlsx')
    # print(df)
    for modality in Used_modality:
        sound_dir_loc = df[modality]  #.str.replace(' ', '_')
        
        sound_dir_loc = sound_dir_loc.tolist() 
        # print(sound_dir_loc)      
        sound_dir_loc = ['datasets/mmlung' + path[1:] for path in sound_dir_loc]

        if feature == "opensmile":
            opensmile_features = []
            for file in tqdm(sound_dir_loc):
                audio_signal, sr = librosa.load(file, sr=16000)
                opensmile_feature = extract_opensmile_features(file)
                opensmile_features.append(opensmile_feature)
            np.save(feature_dir +  modality + "_opensmile_feature.npy", np.array(opensmile_features))
        
        elif feature == "vggish":
            vgg_features = extract_vgg_feature(sound_dir_loc)
            np.save(feature_dir +  modality + "_vggish_feature.npy", np.array(vgg_features))
        elif feature == "clap":
            clap_features = extract_clap_feature(sound_dir_loc)
            np.save(feature_dir +  modality + "_clap_feature.npy", np.array(clap_features))
        elif feature == "audiomae":
            audiomae_feature = extract_audioMAE_feature(sound_dir_loc)
            np.save(feature_dir +  modality + "_audiomae_feature.npy", np.array(audiomae_feature))


def extract_and_save_embeddings(feature="operaCE", input_sec=5, dim=1280):
    from src.benchmark.model_util import extract_opera_feature
    df = pd.read_excel(meta_dir + 'All_path.xlsx')
    for modality in Used_modality:
        sound_dir_loc = df[modality] #.str.replace(' ', '_')
        sound_dir_loc = sound_dir_loc.tolist()       
        sound_dir_loc = ['datasets/mmlung' + path[1:] for path in sound_dir_loc]
   
        if 'operaCT' in feature:
            opera_features = extract_opera_feature(sound_dir_loc,  pretrain=feature, input_sec=input_sec, dim=dim, pad0=True)
        else:
            opera_features = extract_opera_feature(sound_dir_loc,  pretrain=feature, input_sec=input_sec, dim=dim)
        feature_name = feature + str(dim)
        np.save(feature_dir + modality + '_' + feature_name + "_feature.npy", np.array(opera_features))
 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--min_len_cnn", type=int, default=1)
    parser.add_argument("--min_len_htsat", type=int, default=20)
    parser.add_argument("--dim", type=int, default=1280)
    args = parser.parse_args()

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    
        process_label()
       
    if args.pretrain in ["vggish", "opensmile", "clap", "audiomae"]: 
        extract_and_save_embeddings_baselines(args.pretrain)
    else:
        if args.pretrain == "operaCT":
            input_sec = args.min_len_htsat
        elif args.pretrain == "operaCE":
            input_sec = args.min_len_cnn
        elif args.pretrain == "operaGT":
            input_sec = 5
        extract_and_save_embeddings(args.pretrain, input_sec=input_sec, dim=args.dim)
