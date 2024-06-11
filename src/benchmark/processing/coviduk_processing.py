import os
import csv
import argparse
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


data_dir = "datasets/covidUK/"
feature_dir = "feature/coviduk_eval/"
audio_dir = data_dir + "audio_selected/"

if not os.path.exists(audio_dir):
    raise FileNotFoundError(f"Folder not found: {audio_dir}, please download the dataset.")


def get_user_id_from_file(modality):
    data = {}
    col = 1 if modality == "exhalation" else 9
    with open(data_dir + 'audio_metadata.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip the header row
        for row in csvreader:
            data[row[col]] = row[0]
    # print(len(data.values()))
    return data


def process_label(modality="exhalation"):
    train_files = np.load(data_dir + "{}_training_files_downsample.npy".format(modality), allow_pickle = True)
    val_files = np.load(data_dir + "{}_val_files_downsample.npy".format(modality), allow_pickle = True)
    test_files = np.load(data_dir + "{}_testing_files_downsample.npy".format(modality), allow_pickle = True)
    
    all_files = list(train_files) + list(val_files) + list(test_files)

    df_user = pd.read_csv(data_dir + 'participant_metadata.csv', index_col="participant_identifier")
    audio_to_user = get_user_id_from_file(modality)
    df_audio = pd.read_csv(data_dir + 'audio_metadata.csv')

    label_dict = {"Negative":0, "Positive": 1}

    filename_list = []
    label_list = []
    split = []
    for _, row in tqdm(df_audio.iterrows(), total=df_audio.shape[0]):

        filename = row[modality + "_file_name"]
        if filename not in all_files:
            # no in downstream
            continue
        
        user = audio_to_user[filename]
        label = df_user.loc[user]["covid_test_result"]
        label = label_dict[label]

        label_list.append(label)
        filename_list.append(data_dir + "audio_selected/" + filename)
        if filename in train_files:
            split.append("train")
        elif filename in val_files:
            split.append("val")
        else:
            split.append("test")
    print(len(filename_list))
    np.save(feature_dir + "label_{}.npy".format(modality), label_list)
    np.save(feature_dir + "sound_dir_loc_{}.npy".format(modality), filename_list)
    np.save(feature_dir + "split_{}.npy".format(modality), split)


def extract_and_save_embeddings_baselines(modality="exhalation"):
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


def extract_and_save_embeddings(feature="operaCE", modality="exhalation", input_sec=8, dim=1280):
    from src.benchmark.model_util import extract_opera_feature
    sound_dir_loc = np.load(feature_dir + "sound_dir_loc_{}.npy".format(modality))
    opera_features = extract_opera_feature(sound_dir_loc,  pretrain=feature, input_sec=input_sec, dim=dim)
    feature += str(dim)
    np.save(feature_dir +  feature + "_feature_{}.npy".format(modality), np.array(opera_features))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--modality", type=str, default="exhalation")
    parser.add_argument("--min_len_cnn", type=int, default=2)
    parser.add_argument("--min_len_htsat", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)        
        for modality in ["exhalation", "cough"]:
            process_label(modality)
    
    if args.pretrain in ["vggish", "opensmile", "clap", "audiomae"]: 
        extract_and_save_embeddings_baselines(args.modality, args.pretrain)
    else:
        if args.pretrain == "operaCT":
            input_sec = args.min_len_htsat
        elif args.pretrain == "operaCE":
            input_sec = args.min_len_cnn
        elif args.pretrain == "operaGT":
            input_sec = 8.18
        extract_and_save_embeddings(args.pretrain, args.modality, input_sec, dim=args.dim)
