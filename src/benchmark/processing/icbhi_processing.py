import os
import glob as gb
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

labels_data = pd.read_csv('datasets/icbhi/ICBHI_Challenge_diagnosis.txt',
                          dtype=str, sep='\t', names=['userID', 'class'])
splits_data = pd.read_csv('datasets/icbhi/ICBHI_challenge_train_test.txt',
                          dtype=str, sep='\t', names=['fileID', 'group'])
demographics_data = pd.read_csv('datasets/icbhi/ICBHI_Challenge_demographic_information.txt',
                                dtype=str, sep='\t', names=['userId', 'Age', 'Sex', 'Adult_BMI', 'Child Weight', 'Child Height'])

BATCH_SIZE = 512
SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = "datasets/icbhi/"
feature_dir = "feature/icbhidisease_eval/"

audio_dir = data_dir + "ICBHI_final_database/"

if not os.path.exists(audio_dir):
    raise FileNotFoundError(
        f"Folder not found: {audio_dir}, please download the dataset.")


def process_disease():
    sound_dir_loc = np.array(
        gb.glob("datasets/icbhi/ICBHI_final_database/*.wav"))
    labels = []
    filenames = []
    split = []

    for i in tqdm(range(sound_dir_loc.shape[0])):
        filename = sound_dir_loc[i].strip().split('.')[0]
        fileID = filename.split('/')[-1].split('.')[0]
        userID = filename.split('/')[-1].split('_')[0]
        disease_label = labels_data["class"][labels_data.userID ==
                                             userID].values[0]
        file_split = splits_data["group"][splits_data.fileID ==
                                          fileID].values[0]

        filenames.append(sound_dir_loc[i])
        labels.append(disease_label)
        split.append(file_split)

    np.save(feature_dir + "labels.npy", labels)
    np.save(feature_dir + "sound_dir_loc.npy", filenames)
    np.save(feature_dir + "split.npy", split)


def extract_and_save_embeddings_baselines(feature="opensmile"):
    from src.benchmark.baseline.extract_feature import extract_opensmile_features, extract_vgg_feature, extract_clap_feature, extract_audioMAE_feature
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")

    if feature == "opensmile":
        opensmile_features = []
        for file in tqdm(sound_dir_loc):
            opensmile_feature = extract_opensmile_features(file)
            opensmile_features.append(opensmile_feature)
        np.save(feature_dir + "opensmile_feature.npy",
                np.array(opensmile_features))

    elif feature == "vggish":
        vgg_features = extract_vgg_feature(sound_dir_loc)
        np.save(feature_dir + "vggish_feature.npy", np.array(vgg_features))
    elif feature == "clap":
        clap_features = extract_clap_feature(sound_dir_loc)
        np.save(feature_dir + "clap_feature.npy", np.array(clap_features))
    elif feature == "audiomae":
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc)
        np.save(feature_dir + "audiomae_feature.npy",
                np.array(audiomae_feature))


def extract_and_save_embeddings(feature="operaCE", input_sec=8, dim=1280):
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc.npy")

    from src.benchmark.model_util import extract_opera_feature
    opera_features = extract_opera_feature(
        sound_dir_loc, pretrain=feature, input_sec=input_sec, dim=dim)
    feature += str(dim)
    np.save(feature_dir + feature + "_feature.npy", np.array(opera_features))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--min_len_cnn", type=int, default=8)
    parser.add_argument("--min_len_htsat", type=int, default=8)

    args = parser.parse_args()

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        process_disease()

    if args.pretrain in ["vggish", "opensmile", "clap", "audiomae"]:
        extract_and_save_embeddings_baselines(args.pretrain)
    else:
        if args.pretrain == "operaCT":
            input_sec = args.min_len_htsat
        elif args.pretrain == "operaCE":
            input_sec = args.min_len_cnn
        elif args.pretrain == "operaGT":
            input_sec = 8.18
        extract_and_save_embeddings(
            args.pretrain, input_sec=input_sec, dim=args.dim)
