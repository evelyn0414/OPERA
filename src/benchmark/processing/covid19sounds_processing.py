import glob as gb
import argparse
import librosa
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import scipy.signal as sg
from src.util import get_individual_segments_librosa, get_entire_signal_librosa, downsample
import os

task1_data_dir = "datasets/covid19-sounds/0426_EN_used_task1/"
task1_dir = "feature/covid19sounds_eval/"
task1_downsampled_dir = "feature/covid19sounds_eval/downsampled/"

if not os.path.exists(task1_data_dir):
    raise FileNotFoundError(f"Folder not found: {task1_data_dir}, please download the dataset.")

for path in [task1_dir, task1_downsampled_dir]:
    if not os.path.exists(path):
        os.makedirs(path)


def extract_opera_features(feature, task=1, modality="cough", input_sec=8, dim=1280):
    folders = {1: task1_downsampled_dir, 2: task2_dir}
    feature_dir = folders[task]
    from model_util import extract_opera_feature
    sound_dir_loc = np.load(feature_dir  + "sound_dir_loc_{}.npy".format(modality))
    cola_features = extract_opera_feature(sound_dir_loc,  pretrain=feature, input_sec=input_sec, dim=dim)
    feature += str(dim)
    print("saving feature to", feature_dir  +  feature + "_feature_{}.npy".format(modality))
    np.save(feature_dir  +   feature + "_feature_{}.npy".format(modality), np.array(cola_features))


def preprocess_task1(modality="cough"):
    """run once and shared by all methods"""
    data_split = []
    labels = []
    sound_dir_loc = [] 
    df = pd.read_csv("datasets/covid19-sounds/data_0426_en_task1.csv", delimiter=";")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        userID = row["Uid"]
        folder = row["Folder Name"]
        filename = row["{} filename".format(modality.capitalize())]
        split = row["split"]
        label = row["label"]
        sex = row["Sex"]
        if userID[:4] == "2020":
            userID = "form-app-users"
        file = "/".join(["datasets/covid19-sounds/0426_EN_used_task1", userID, folder, filename])

        labels.append(label)
        data_split.append(split)
        sound_dir_loc.append(file)

    np.save(task1_dir + "labels.npy", np.array(labels))
    np.save(task1_dir + "data_split.npy", np.array(data_split))
    np.save(task1_dir + "sound_dir_loc_{}.npy".format(modality), np.array(sound_dir_loc))


def task1_downsample(downsampling_factor=5):
    labels =  np.load(task1_dir + "labels.npy")
    print(labels)
    splits = np.load(task1_dir + "data_split.npy")
    train_idx = splits == 0
    val_idx = splits == 1
    test_idx = splits == 2

    train_labels = labels[train_idx]

    downsampled_train_labels = train_labels[::downsampling_factor]
    new_labels = np.concatenate([downsampled_train_labels, labels[val_idx], labels[test_idx]])
    np.save(task1_downsampled_dir  + "labels.npy", new_labels) 
    print(downsampled_train_labels)

    new_splits = np.concatenate([np.full_like(downsampled_train_labels, 0), splits[val_idx], splits[test_idx]])
    np.save(task1_downsampled_dir  + "data_split.npy", new_splits)
    print(new_splits)

    for modality in ["cough", "breath"]:
        sound_dir_loc = np.load(task1_dir + "sound_dir_loc_{}.npy".format(modality))
        train_sound_dir_loc = sound_dir_loc[train_idx]
        downsampled_train_sound_dir_loc = train_sound_dir_loc[::downsampling_factor]
        new_sound_dir_loc = np.concatenate([downsampled_train_sound_dir_loc, sound_dir_loc[val_idx], sound_dir_loc[test_idx]])
        np.save(task1_downsampled_dir  + "sound_dir_loc_{}.npy".format(modality), new_sound_dir_loc)


def extract_and_save_embeddings_baselines(task=1, modality="cough", feature="opensmile"):
    folders = {1: "feature/covid19sounds_eval/downsampled/"}
    feature_dir = folders[task]
    from src.benchmark.baseline.extract_feature import extract_opensmile_features, extract_vgg_feature, extract_clap_feature, extract_audioMAE_feature

    sound_dir_loc = np.load(feature_dir + "sound_dir_loc_{}.npy".format(modality))

    if feature == "opensmile":
        opensmile_features = []
        for file in tqdm(sound_dir_loc):
            audio_signal, sr = librosa.load(file, sr=16000)
            opensmile_feature = extract_opensmile_features(file)
            opensmile_features.append(opensmile_feature)
        np.save(feature_dir + "opensmile_feature_{}.npy".format(modality), np.array(opensmile_features))
    
    elif feature == "vggish":
        vgg_features = extract_vgg_feature(sound_dir_loc)
        np.save(feature_dir + "vggish_feature_{}.npy".format(modality), np.array(vgg_features))

    elif feature == "clap":
        clap_features = extract_clap_feature(sound_dir_loc)
        np.save(feature_dir + "clap_feature_{}.npy".format(modality), np.array(clap_features))
    
    elif feature == "audiomae":
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc)
        np.save(feature_dir + "audiomae_feature_{}.npy".format(modality), np.array(audiomae_feature))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="1")
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--min_len_cnn", type=int, default=8)
    parser.add_argument("--min_len_htsat", type=int, default=8)
    parser.add_argument("--modality", type=str, default="breath")
    args = parser.parse_args()

    if not os.path.exists(task1_dir):
        os.makedirs(task1_dir)
        os.makedirs(task1_downsampled_dir)
        
        for modality in ["breath", "cough"][]:
            preprocess_task1(modality)
        task1_downsample()
    
    if args.pretrain in ["vggish", "opensmile", "clap", "audiomae"]: 
        extract_and_save_embeddings_baselines(1, modality, args.pretrain)
    else:
        if args.pretrain == "operaCT":
            input_sec = args.min_len_htsat
        elif args.pretrain == "operaCE":
            input_sec = args.min_len_cnn
        elif args.pretrain == "operaGT":
            input_sec = 8.18
        extract_opera_features(args.pretrain, task=int(args.task), modality=args.modality, input_sec=input_sec, dim=args.dim)