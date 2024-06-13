import os
import glob as gb
import argparse
import collections
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

feature_dir = "feature/kauh_eval/"

audio_dir = "datasets/KAUH/AudioFiles/"

if not os.path.exists(audio_dir):
    print(f"Folder not found: {audio_dir}, downloading the dataset")
    os.system('sh datasets/KAUH/download_data.sh')
    # raise FileNotFoundError(
    #     f"Folder not found: {audio_dir}, please download the dataset by running: sh datasets/KAUH/download_data.sh.")


def preprocess_subset():
    sound_dir_loc = np.array(gb.glob(audio_dir + "*.wav"))
    sound_dir_loc_subset = []
    labels = []
    # labels {'N': 105, 'Asthma': 51, 'asthma': 45, 'heart failure': 45, 'COPD': 24, 'pneumonia': 15, 'Lung Fibrosis': 12, 'Heart Failure': 9, 'BRON': 9, 'Plueral Effusion': 6, 'Heart Failure + COPD': 6, 'copd': 3, 'Heart Failure + Lung Fibrosis ': 3, 'Asthma and lung fibrosis': 3}

    for i in range(sound_dir_loc.shape[0]):
        filename = sound_dir_loc[i]

        label = filename.split('/')[-1].split(',')[0].split('_')[-1]

        if label == "N":
            label = "healthy"
        elif "asthma" in label or "Asthma" in label:
            label = "asthma"
        elif "COPD" in label:
            label = "COPD"
        else:
            continue
        sound_dir_loc_subset.append(filename)
        labels.append(label)

    np.save(feature_dir + "sound_dir_loc_subset.npy", sound_dir_loc_subset)
    np.save(feature_dir + "labels_both.npy", labels)


def split_set():
    sound_dir_loc_subset = np.load(feature_dir + "sound_dir_loc_subset.npy")
    labels = np.load(feature_dir + "labels_both.npy")
    user_ids, user_labels = [], []
    audio_split = []
    for i, filename in enumerate(sound_dir_loc_subset):
        user_id = filename.split('/')[-1].split('_')[0][2:]
        if user_id not in user_ids:
            user_ids.append(user_id)
            user_labels.append(labels[i])

    train_ratio = 0.7
    validation_ratio = 0.1
    test_ratio = 0.20

    seed = 42
    _x_train, x_test, _y_train, y_test = train_test_split(
        user_ids, user_labels, test_size=test_ratio, random_state=seed, stratify=user_labels
    )

    x_train, x_val, y_train, y_val = train_test_split(
        _x_train, _y_train, test_size=validation_ratio /
        (validation_ratio + train_ratio),
        random_state=seed, stratify=_y_train
    )

    for i, filename in enumerate(sound_dir_loc_subset):
        u = filename.split('/')[-1].split('_')[0][2:]
        if u in x_train:
            audio_split.append("train")
        else:
            audio_split.append("test")

    np.save(feature_dir + "train_test_split.npy", audio_split)


def check_demographic(trait="label"):

    print("checking training and testing demographic", trait)

    sound_dir_loc_subset = np.load(feature_dir + "sound_dir_loc_subset.npy")
    labels = np.load(feature_dir + "labels_both.npy")
    split = np.load(feature_dir + "train_test_split.npy")

    train = sound_dir_loc_subset[split == "train"]
    test = sound_dir_loc_subset[split == "test"]

    for sound_dir_loc in [train, test]:
        count = collections.defaultdict(int)
        for i in range(sound_dir_loc.shape[0]):
            filename = sound_dir_loc[i]
            if trait == "label":
                label = labels[i]
                count[label] += 1
            if trait == "age":
                age = (int(filename.split('/')
                       [-1].split('.')[0].split(',')[-2]) // 10) * 10
                count[age] += 1
            if trait == "sex":
                sex = filename.split('/')[-1].split('.')[0].split(',')[-1]
                count[sex] += 1
        print({k: count[k] for k in sorted(count.keys())})


def extract_and_save_embeddings_baselines(feature="opensmile"):
    from src.benchmark.baseline.extract_feature import extract_opensmile_features, extract_vgg_feature, extract_clap_feature, extract_audioMAE_feature
    sound_dir_loc_subset = np.load(feature_dir + "sound_dir_loc_subset.npy")
    if feature == "opensmile":
        opensmile_features = []
        for file in tqdm(sound_dir_loc_subset):
            opensmile_feature = extract_opensmile_features(file)
            opensmile_features.append(opensmile_feature)
        np.save(feature_dir + "opensmile_feature_both.npy",
                np.array(opensmile_features))
    elif feature == "vggish":
        vgg_features = extract_vgg_feature(sound_dir_loc_subset)
        np.save(feature_dir + "vggish_feature_both.npy", vgg_features)
    elif feature == "clap":
        clap_features = extract_clap_feature(sound_dir_loc_subset)
        np.save(feature_dir + "clap_feature_both.npy", clap_features)
    elif feature == "audiomae":
        audiomae_feature = extract_audioMAE_feature(sound_dir_loc_subset)
        np.save(feature_dir + "audiomae_feature_both.npy",
                np.array(audiomae_feature))


def extract_and_save_embeddings(feature="operaCE", input_sec=8,  dim=1280):
    sound_dir_loc_subset = np.load(feature_dir + "sound_dir_loc_subset.npy")

    from src.benchmark.model_util import extract_opera_feature
    opera_features = extract_opera_feature(
        sound_dir_loc_subset, pretrain=feature, input_sec=input_sec, dim=dim)
    feature += str(dim)
    np.save(feature_dir + feature + "_feature_both.npy",
            np.array(opera_features))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="operaCE")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--min_len_cnn", type=int, default=8)
    parser.add_argument("--min_len_htsat", type=int, default=8)
    args = parser.parse_args()

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        preprocess_subset()
        split_set()
        for trait in ["label", "sex", "age"]:
            check_demographic(trait)

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
