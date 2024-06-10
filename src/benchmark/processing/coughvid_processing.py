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
from os.path import exists
import os

data_dir = "datasets/coughvid/"
feature_dir = "feature/coughvid_eval/"
audio_dir = data_dir + "wav"
if not os.path.exists(audio_dir):
    raise FileNotFoundError(f"Folder not found: {audio_dir}, please download the dataset.")


train_uuid = np.load(data_dir + "coughvid__train_uuids.npy", allow_pickle = True)
val_uuid = np.load(data_dir + "coughvid__val_uuids.npy", allow_pickle = True)
covid_test_uuid = np.load(data_dir + "coughvid_covid_test_uuids.npy", allow_pickle = True)
gender_test_uuid = np.load(data_dir + "coughvid_gender_test_uuids.npy", allow_pickle = True)
all_uuid = list(train_uuid) + list(val_uuid) + list(gender_test_uuid)


def preprocess_label(label="covid"):
    df = pd.read_csv(data_dir + 'metadata_compiled.csv', index_col="uuid")
    df = df.replace(np.nan,'',regex=True)
    # df = df[df["gender"].str.contains("male")]

    gender_label_dict = {"female": 1, "male": 0, "pnts": None, "Other": None, "other": None,'':None}
    covid_label_dict = {"COVID-19": 1, "healthy": 0, "pnts": None, "Other": None, 'symptomatic': None, '':None}

    filename_list = []
    label_list = []
    split = []
    for uuid, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = data_dir + "wav/" + uuid + ".wav"
        if not exists(filename):
            # problem in data name
            filename = data_dir + "wav/" + uuid[:-1] + ".wav"
        if label == "gender":
            audio_label = gender_label_dict[row["gender"]]
            
        elif label == "covid":
            audio_label = covid_label_dict[row["status"]]
        
        if audio_label is None:
            continue
        if uuid not in all_uuid:
            # no in downstream
            continue

        label_list.append(audio_label)
        filename_list.append(filename)
        if uuid in train_uuid:
            split.append("train")
        elif uuid in val_uuid:
            split.append("val")
        else:
            split.append("test")

    np.save(feature_dir + "label_{}.npy".format(label), label_list)
    np.save(feature_dir + "sound_dir_loc_{}.npy".format(label), filename_list)
    np.save(feature_dir + "split_{}.npy".format(label), split)


def extract_and_save_embeddings_baselines(label, feature="opensmile"):
    from src.benchmark.baseline.extract_feature import extract_opensmile_features, extract_vgg_feature, extract_clap_feature, extract_audioMAE_feature
    
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc_{}.npy".format(label))

    if feature == "opensmile":
        opensmile_features = []
        for file in tqdm(sound_dir_loc):
            audio_signal, sr = librosa.load(file, sr=16000)
            opensmile_feature = extract_opensmile_features(file)
            opensmile_features.append(opensmile_feature)
        np.save(feature_dir + "opensmile_feature_{}.npy".format(label), np.array(opensmile_features))
    elif feature == "vggish":
        vgg_features = extract_vgg_feature(sound_dir_loc)
        np.save(feature_dir + "vggish_feature_{}.npy".format(label), np.array(vgg_features))
    elif feature == "clap":
        clap_features = extract_clap_feature(sound_dir_loc)
        np.save(feature_dir + "clap_feature_{}.npy".format(label), np.array(clap_features))
    elif feature == "audiomae":
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc)
        np.save(feature_dir + "audiomae_feature_{}.npy".format(label), np.array(audiomae_feature))


def extract_and_save_embeddings(feature="operaCE", label="covid", input_sec=2, dim=1280):
    from src.benchmark.model_util import extract_opera_feature
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc_{}.npy".format(label))
    opera_features = extract_opera_feature(sound_dir_loc,  pretrain=feature, input_sec=input_sec, dim=dim)
    feature += str(dim)
    np.save(feature_dir +  feature + "_feature_{}.npy".format(label), np.array(opera_features))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--label", type=str, default="covid")
    parser.add_argument("--min_len_cnn", type=int, default=2)
    parser.add_argument("--min_len_htsat", type=int, default=2)
    args = parser.parse_args()

    if not exists(feature_dir):
        os.makedirs(feature_dir)
        for label in ["covid", "gender"]:
            preprocess_label(label)
    
    if args.pretrain in ["vggish", "opensmile", "clap", "audiomae"]: 
        extract_and_save_embeddings_baselines(label, args.pretrain)
    else:
        if args.pretrain == "operaCT":
            input_sec = args.min_len_htsat
        elif args.pretrain == "operaCE":
            input_sec = args.min_len_cnn
        elif args.pretrain == "operaGT":
            input_sec = 8.18
        extract_and_save_embeddings(args.pretrain, args.label, input_sec, dim=args.dim)