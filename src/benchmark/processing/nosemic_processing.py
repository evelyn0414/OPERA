import glob as gb
import argparse
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from os.path import exists
import os

data_dir = "datasets/nosemic/audio/"
meta_dir = "datasets/nosemic/"
feature_dir = "feature/nosemic_eval/"


def process_label():
    labels = []
    uids = []
    #print(os.listdir(data_dir))
    for filename in sorted(os.listdir(data_dir)):
        user,_,_,label = filename[:-4].split('_')
        labels.append(label)
        uids.append(user)
        
    labels = np.array(labels).T    
    uids = np.array(uids).T
    np.save(feature_dir + "labels.npy", labels)
    np.save(feature_dir + "uids.npy", uids)

def extract_and_save_embeddings_baselines(feature="opensmile"):
    from src.benchmark.baseline.extract_feature import extract_opensmile_features, extract_vgg_feature, extract_clap_feature, extract_audioMAE_feature
    
    sound_dir_loc = [data_dir + file for file in sorted(os.listdir(data_dir))] 

    if feature == "opensmile":
        opensmile_features = []
        for file in tqdm(sound_dir_loc):
            audio_signal, sr = librosa.load(file, sr=16000)
            opensmile_feature = extract_opensmile_features(file)
            opensmile_features.append(opensmile_feature)
        np.save(feature_dir + "opensmile_feature.npy", np.array(opensmile_features))
    
    elif feature == "vggish":
        vgg_features = extract_vgg_feature(sound_dir_loc)
        np.save(feature_dir + "vggish_feature.npy", np.array(vgg_features))
    elif feature == "clap":
        clap_features = extract_clap_feature(sound_dir_loc)
        np.save(feature_dir + "clap_feature.npy", np.array(clap_features))
    elif feature == "audiomae":
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc)
        np.save(feature_dir + "audiomae_feature.npy", np.array(audiomae_feature))


def extract_and_save_embeddings(feature="operaCE", input_sec=8, dim=1280):
    from src.benchmark.model_util import extract_opera_feature
    sound_dir_loc = [data_dir + file for file in sorted(os.listdir(data_dir))]
    print('input:', input_sec)
    opera_features = extract_opera_feature(sound_dir_loc,  pretrain=feature, input_sec=input_sec, dim=dim)
    feature += str(dim)
    np.save(feature_dir + feature + "_feature.npy", np.array(opera_features))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--min_len_cnn", type=int, default=1)
    parser.add_argument("--min_len_htsat", type=int, default=30)
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
            input_sec = 8
        extract_and_save_embeddings(args.pretrain, input_sec=input_sec, dim=args.dim)

