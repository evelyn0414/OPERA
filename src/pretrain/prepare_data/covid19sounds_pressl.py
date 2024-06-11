import glob as gb
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.util import get_entire_signal_librosa

# for pretraining


def preprocess_spectrogram_SSL(modality="breath", input_sec=8):

    except_uids = np.load("datasets/covid19-sounds/test_uid.npy").tolist()
    except_uids.append("MJQ296DCcN")

    # currently removing all data used, but can instead remove test set only
    task1_pd = pd.read_csv(
        "datasets/covid19-sounds/data_0426_en_task1.csv", delimiter=";")
    task1_pd = task1_pd[task1_pd["split"] == 2]
    task1_uids = task1_pd["Uid"].tolist()

    task2_pd = pd.read_csv("datasets/covid19-sounds/data_0426_en_task2.csv")
    task2_pd = task2_pd[task2_pd["fold"] == "test"]
    task2_uids = task2_pd["uid"].tolist()

    except_uids += task1_uids + task2_uids

    invalid_data = 0

    filename_list = []

    metadata_dir = np.array(
        gb.glob("datasets/covid19-sounds/covid19_data_0426_metadata/*.csv"))
    # use metadata as outer loop to enable quality check
    for file in metadata_dir:
        df = pd.read_csv(file, delimiter=";")

        if "cough" in modality:
            df = df[df["Cough check"].str.contains("c")]
        if "breath" in modality:
            df = df[df["Breath check"].str.contains("b")]
        if "voice" in modality:
            df = df[df["Voice check"].str.contains("v")]

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            userID = row["Uid"]

            # avoid users used in downstream task test set
            if userID in except_uids:
                continue

            folder = row["Folder Name"]

            file_loc = "/".join(["datasets/covid19-sounds/covid19_data_0426",
                                userID, folder, "*{}*.wav".format(modality)])
            file_list = gb.glob(file_loc)
            # data inconsistency in naming
            if len(file_list) == 0 and modality == "voice":
                file_list = gb.glob(
                    "/".join(["datasets/covid19-sounds/covid19_data_0426", userID, folder, "*read*.wav"]))
            filename = file_list[0]

            data = get_entire_signal_librosa("", filename.split(
                '.')[0], spectrogram=True, input_sec=input_sec)

            if data is None:
                invalid_data += 1
                continue

            # saving to individual npy files
            np.save("datasets/covid19-sounds/entire_spec_npy_8000/" +
                    "_".join([userID, folder, modality + ".npy"]), data)
            filename_list.append(
                "datasets/covid19-sounds/entire_spec_npy_8000/" + "_".join([userID, folder, modality]))
            np.save("datasets/covid19-sounds/SSL_entireaudio_filenames_8000_" +
                    modality + ".npy", filename_list)

    print("finished preprocessing breathing: valid data",
          len(filename_list), "; invalid data", invalid_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="breath")
    parser.add_argument("--input_sec", type=int, default=-1)
    args = parser.parse_args()

    if args.input_sec == -1:
        args.input_sec = 2 if args.modality == "cough" else 8

    preprocess_spectrogram_SSL(
        modality=args.modality, input_sec=args.input_sec)
