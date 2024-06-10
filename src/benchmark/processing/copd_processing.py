# Yuwei (Evelyn) Zhang
# yz798@cam.ac.uk
# Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking
# https://github.com/evelyn0414/OPERA

import glob as gb
import argparse
import librosa
import collections
import numpy as np
from sklearn.model_selection import train_test_split
import random
import csv
from tqdm import tqdm
import os

data_dir = "datasets/copd/"
audio_dir = data_dir + "RespiratoryDatabase@TR"
feature_dir = "feature/copd_eval/"

if not os.path.exists(audio_dir):
    raise FileNotFoundError(f"Folder not found: {audio_dir}, please download the dataset.")


def get_annotaion_from_csv(path):
    data = {}
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip the header row
        for row in csvreader:
            data[row[0]] = int(row[1][-1])
    return data


def preprocess_split():
    label_dict = get_annotaion_from_csv(data_dir + "Labels.csv")

    patient_id = list(label_dict.keys())
    label = [label_dict[u] for u in patient_id]

    # split patients into subject independent splits

    _x_train, x_test, _y_train, y_test = train_test_split(
            patient_id, label, test_size=0.2, random_state=1337, stratify=label
        )

    x_train, x_val, y_train, y_val = train_test_split(
            _x_train, _y_train, test_size=0.2, random_state=1337, stratify=_y_train
        )
    
    print(collections.Counter(y_train))
    print(collections.Counter(y_val))
    print(collections.Counter(y_test))

    sound_dir_loc = np.array(gb.glob(data_dir + "RespiratoryDatabase@TR/*.wav"))
    np.save(feature_dir + "sound_dir_loc.npy", sound_dir_loc)

    audio_split = []
    labels = []
    for file in sound_dir_loc:
        u = file.split("/")[-1][:4]
        if u in x_train:
            audio_split.append("train")
        elif u in x_val:
            audio_split.append("val")
        else:
            audio_split.append("test")
        labels.append(label_dict[u])

    np.save(feature_dir + "train_test_split.npy", audio_split)
    np.save(feature_dir + "labels.npy", labels)


def check_demographic(trait="label"):

    print("checking training and testing", trait)

    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    labels = np.load(feature_dir + "labels.npy")
    split = np.load(feature_dir + "train_test_split.npy")

    train = sound_dir_loc[split == "train"]
    val = sound_dir_loc[split == "val"]
    test = sound_dir_loc[split == "test"]

    for sound_dir_loc in [train, val, test]:
        count = collections.defaultdict(int)
        for i in range(sound_dir_loc.shape[0]):
            filename = sound_dir_loc[i]
            if trait == "label":
                label = labels[i]
                count[label] += 1
        print({k: count[k] for k in sorted(count.keys())})


def extract_and_save_embeddings_baselines(feature="opensmile"):
    from src.benchmark.baseline.extract_feature import extract_opensmile_features, extract_vgg_feature, extract_clap_feature, extract_audioMAE_feature
    
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")

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
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")
    opera_features = extract_opera_feature(sound_dir_loc,  pretrain=feature, input_sec=input_sec, dim=dim)
    feature += str(dim)
    np.save(feature_dir + feature + "_feature.npy", np.array(opera_features))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--min_len_cnn", type=int, default=8)
    parser.add_argument("--min_len_htsat", type=int, default=8)
    parser.add_argument("--dim", type=int, default=1280)
    args = parser.parse_args()

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        preprocess_split()
        check_demographic()

    if args.pretrain in ["vggish", "opensmile", "clap", "audiomae"]: 
        extract_and_save_embeddings_baselines(args.pretrain)
    else:
        if args.pretrain == "operaCT":
            input_sec = args.min_len_htsat
        elif args.pretrain == "operaCE":
            input_sec = args.min_len_cnn
        elif args.pretrain == "operaGT":
            input_sec = 8.18
        extract_and_save_embeddings(args.pretrain, input_sec=input_sec, dim=args.dim)