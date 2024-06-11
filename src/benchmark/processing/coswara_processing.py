import glob as gb
import argparse
import librosa
import collections
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import csv
from src.util import get_entire_signal_librosa
import os

feature_dir = "feature/coswara_eval/"  # "datasets/Coswara-Data/coswara_eval/"
data_dir = "datasets/Coswara-Data/Extracted_data/"

if not os.path.exists(data_dir):
    raise FileNotFoundError(
        f"Folder not found: {data_dir}, please download the dataset.")


def get_annotaion_from_csv(path):
    data = {}
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip the header row
        for row in csvreader:
            data[row[0]] = row[1]
    # print(data)
    return data


# not working: Coswara-Data/Extracted_data/20200418/C7Km0KttQRMMM6UoyocajfgZAOB3/breathing-shallow.wav
#  Coswara-Data/Extracted_data/20200418/C7Km0KttQRMMM6UoyocajfgZAOB3/cough-heavy.wav
# Coswara-Data/Extracted_data/20200814/kgjTguvo3vZJTO7F1qO9GxEicbA3/cough-heavy.wav

def preprocess_label(label="sex"):
    df = pd.read_csv('datasets/Coswara-Data/combined_data.csv', index_col="id")
    df = df.replace(np.nan, '', regex=True)

    sex_label_dict = {"female": 1, "male": 0, "pnts": None, "Other": None}
    smoker_label_dict = {"y": 1, "n": 0, "TRUE": 1,
                         "True": 1, "False": 0, "FALSE": 0, "": None}

    for modality in ["breathing-deep", "breathing-shallow", "cough-heavy", "cough-shallow"]:
        label_list = []
        filename_list = []
        annotation = get_annotaion_from_csv(
            "datasets/Coswara-Data/annotations/{}_labels.csv".format(modality))
        for uuid, row in tqdm(df.iterrows(), total=df.shape[0]):

            if uuid == "9hftEYixyhP1Neeq3fB7ZwITQC53" and modality == "cough-shallow":
                # this user has missing file
                continue

            if uuid in ["C7Km0KttQRMMM6UoyocajfgZAOB3", "kgjTguvo3vZJTO7F1qO9GxEicbA3"]:
                continue

            if annotation["_".join([uuid, modality])] == "0":
                # bad quality
                continue

            files = gb.glob(
                "/".join(["datasets/Coswara-Data/Extracted_data", "*", uuid, modality + ".wav"]))

            filename = files[0]
            if label == "sex":
                audio_label = sex_label_dict[row["g"]]
            elif label == "smoker":
                audio_label = smoker_label_dict[row["smoker"]]

            if audio_label is not None:
                label_list.append(audio_label)
                filename_list.append(filename)

        print(collections.Counter(label_list))
        np.save(feature_dir + "{}_label_{}.npy".format(label, modality), label_list)
        np.save(
            feature_dir + "entireaudio_filenames_{}_w_{}.npy".format(modality, label), filename_list)


def preprocess_modality(modality="breathing", label="sex"):
    uuid_dict = collections.defaultdict(int)
    submodalities = {"breathing": [
        "-deep", "-shallow"], "cough": ["-heavy", "-shallow"]}

    for submodality in submodalities[modality]:
        filenames = np.load(
            feature_dir + "entireaudio_filenames_{}_w_{}.npy".format(modality+submodality, label))
        for i, file in enumerate(filenames):
            uuid = file.split("/")[-2]
            uuid_dict[uuid] += 1

    for submodality in submodalities[modality]:
        labels = np.load(
            feature_dir + "{}_label_{}.npy".format(label, modality + submodality))
        filenames = np.load(
            feature_dir + "entireaudio_filenames_{}_w_{}.npy".format(modality+submodality, label))
        final_label = []
        final_filenames = []
        for i, file in enumerate(filenames):
            uuid = file.split("/")[-2]
            if uuid_dict[uuid] == 2:
                # user has both deep and shallow
                final_label.append(labels[i])
                final_filenames.append(file)
        print(collections.Counter(final_label))
        np.save(feature_dir + "{}_aligned_{}_label_{}.npy".format(modality,
                label, modality + submodality), final_label)
        np.save(feature_dir + "{}_aligned_filenames_{}_w_{}.npy".format(modality,
                label, modality + submodality), final_filenames)


def preprocess_split_google():
    modality, label, submodality = "cough",  "sex", "-heavy"
    num_pos, num_neg = 174, 478

    # split once, sequence same
    labels = np.load(
        feature_dir + "{}_aligned_{}_label_{}.npy".format(modality, label, modality + submodality))

    # Indices of positive and negative class
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]

    np.random.seed(42)
    # Shuffle indices to randomize data
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)

    # Split data into training and test sets
    test_positive_indices = positive_indices[:num_pos]
    test_negative_indices = negative_indices[:num_neg]

    test_indices = np.concatenate(
        (test_positive_indices, test_negative_indices))

    split = []
    for idx in range(len(labels)):
        if idx in test_indices:
            split.append("test")
        else:
            split.append("train")
    print(split)
    np.save(feature_dir + "google_{}_{}_split.npy".format(label, modality), split)


def split_set(modality, label="smoker"):

    broad_modality = modality.split("-")[0]
    sound_dir_loc = np.load(
        feature_dir + "{}_aligned_filenames_{}_w_{}.npy".format(broad_modality, label, modality))

    labels = np.load(
        feature_dir + "{}_aligned_{}_label_{}.npy".format(broad_modality, label, modality))

    X_train, X_test, y_train, y_test = train_test_split(
        sound_dir_loc, labels, test_size=0.2, random_state=1337, stratify=labels
    )
    split = np.array(
        ["train" if file in X_train else "test" for file in sound_dir_loc])
    np.save(feature_dir + "{}_aligned_train_test_split_{}_w_{}.npy".format(
        broad_modality, label, modality), split)


def check_demographic(modality, label="smoker", trait="label"):

    print("checking training and testing demographic", trait, modality, label)
    broad_modality = modality.split("-")[0]
    sound_dir_loc = np.load(
        feature_dir + "{}_aligned_filenames_{}_w_{}.npy".format(broad_modality, label, modality))

    labels = np.load(
        feature_dir + "{}_aligned_{}_label_{}.npy".format(broad_modality, label, modality))
    split = np.load(
        feature_dir + "{}_aligned_train_test_split_{}_w_{}.npy".format(broad_modality, label, modality))

    train = sound_dir_loc[split == "train"]
    test = sound_dir_loc[split == "test"]

    df = pd.read_csv('datasets/Coswara-Data/combined_data.csv', index_col="id")
    df = df.replace(np.nan, '', regex=True)

    for trait in ["label", "sex", "age"]:
        for sound_dir_loc in [train, test]:
            count = collections.defaultdict(int)
            for i in range(sound_dir_loc.shape[0]):
                filename = sound_dir_loc[i]
                uuid = filename.split("/")[-2]
                # print(uuid)
                row = df.loc[uuid]
                if trait == "label":
                    label = labels[i]
                    count[label] += 1
                if trait == "age":
                    age = int(row["a"] // 10) * 10
                    count[age] += 1
                if trait == "sex":
                    sex = row["g"]
                    count[sex] += 1
            print({k: count[k] for k in sorted(count.keys())})


def preprocess_spectrogram(modality, label="sex"):
    print("preprocessing spectrogram of {} with {} label".format(modality, label))
    audio_images = []
    broad_modality = modality.split("-")[0]
    # sound_dir_loc = np.load(feature_dir + "entireaudio_filenames_{}_w_{}.npy".format(modality, label))
    sound_dir_loc = np.load(
        feature_dir + "{}_aligned_filenames_{}_w_{}.npy".format(broad_modality, label, modality))
    print("number of files", len(sound_dir_loc))
    for file in tqdm(sound_dir_loc):
        data = get_entire_signal_librosa(
            "", file.split('.')[0], spectrogram=True, pad=True)
        audio_images.append(data)
    # np.savez(feature_dir + "entire_spec_all_{}_w_{}.npz".format(modality, label), *audio_images)
    np.savez(feature_dir + "{}_aligned_spec_{}_w_{}.npz".format(broad_modality,
             modality, label), *audio_images)


def extract_and_save_embeddings_baselines(modality, label="sex", feature="opensmile"):
    from src.benchmark.baseline.extract_feature import extract_opensmile_features,  extract_vgg_feature, extract_clap_feature, extract_audioMAE_feature
    opensmile_features = []

    broad_modality = modality.split("-")[0]
    sound_dir_loc = np.load(
        feature_dir + "{}_aligned_filenames_{}_w_{}.npy".format(broad_modality, label, modality))

    if feature == "opensmile":
        for file in tqdm(sound_dir_loc):
            audio_signal, sr = librosa.load(file, sr=16000)
            opensmile_feature = extract_opensmile_features(file)
            opensmile_features.append(opensmile_feature)
        np.save(feature_dir + "opensmile_feature_{}_{}.npy".format(modality,
                label), np.array(opensmile_features))
    elif feature == "vggish":
        vgg_features = extract_vgg_feature(sound_dir_loc)
        np.save(feature_dir + "vggish_feature_{}_{}.npy".format(modality,
                label), np.array(vgg_features))
    elif feature == "clap":
        clap_features = extract_clap_feature(sound_dir_loc)
        np.save(feature_dir + "clap_feature_{}_{}.npy".format(modality,
                label), np.array(clap_features))
    elif feature == "audiomae":
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc)
        np.save(feature_dir + "audiomae_feature_{}_{}.npy".format(modality,
                label), np.array(audiomae_feature))


def extract_and_save_embeddings(feature, modality, label="sex", input_sec=8, dim=1280):
    from src.benchmark.model_util import extract_opera_feature
    # sound_dir_loc = np.load("datasets/Coswara-Data/entireaudio_filenames_{}_w_{}.npy".format(modality, label))
    broad_modality = modality.split("-")[0]
    if input_sec == 2:
        audio_images = np.load(
            feature_dir + "{}_aligned_spec_pad2_{}_w_{}.npz".format(broad_modality, modality, label))
    else:
        # default 8
        audio_images = np.load(
            feature_dir + "{}_aligned_spec_{}_w_{}.npz".format(broad_modality, modality, label))
    audio_images = [audio_images[f] for f in audio_images.files]
    opera_features = extract_opera_feature(
        audio_images,  pretrain=feature, from_spec=True, input_sec=input_sec, dim=dim)
    feature += str(dim)
    np.save(feature_dir + feature + "_feature_{}_{}.npy".format(modality,
            label), np.array(opera_features))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--modality", type=str, default="cough-shallow")
    parser.add_argument("--label", type=str, default="sex")
    parser.add_argument("--min_len_cnn", type=int, default=2)
    parser.add_argument("--min_len_htsat", type=int, default=8)

    args = parser.parse_args()

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        preprocess_split_google()

        #  run once
        for label in ["sex", "smoker"]:
            # preprocess_label(label)
            for modality in ["breathing", "cough"][1:]:
                preprocess_modality(modality, label)
            for modality in ["breathing-deep", "breathing-shallow", "cough-heavy", "cough-shallow"][3:]:
                preprocess_spectrogram(modality, label)

    if args.pretrain in ["vggish", "opensmile", "clap", "audiomae"]:
        extract_and_save_embeddings_baselines(
            args.modality, args.label, args.pretrain)
    else:
        if args.pretrain == "operaCT":
            input_sec = args.min_len_htsat
        elif args.pretrain == "operaCE":
            input_sec = args.min_len_cnn
        elif args.pretrain == "operaGT":
            input_sec = 8.18
        extract_and_save_embeddings(
            args.pretrain, args.modality, args.label, input_sec, dim=args.dim)
