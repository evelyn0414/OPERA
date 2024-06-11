import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model
from src.benchmark.other_eval.finetuning import AudioDataset
from src.util import plot_tsne_individual


def visualize_embedding_multiple(pretrain="operaCE", data_source={"covidbreath": 251}, split="test", num_samples=500, fea_dim=1280):
    for dt, max_len in data_source.items():
        from_npy = True
        if dt in ['covidbreath', 'covidcough']:
            modality = dt[5:]
            filenames = list(np.load("datasets/covid19-sounds/SSL_entireaudio_filenames_{}.npy".format(modality)))

        elif dt == "icbhi":
            #  training with audio
            icbhi_filenames = np.load("datasets/icbhi/entire_spec_filenames.npy")
            train_test = np.load("datasets/icbhi/entire_spec_split.npy")
            filenames = list(icbhi_filenames[train_test == "train"]) #exclude testing

        elif dt == "icbhicycle":
            # training with cycle:
            icbhi_filenames = np.load("datasets/icbhi/cycle_spec_pad2_name.npy")
            train_test = np.load("datasets/icbhi/cycle_spec_split.npy")
            filenames = list(icbhi_filenames[train_test == "train"]) #exclude testing

        elif dt == "coughvid":
            filenames = list(np.load("datasets/coughvid/entire_spec_filenames.npy"))
        
        elif dt == "hf_lung":
            filenames = list(np.load("datasets/hf_lung/entire_spec_filenames.npy"))
        
        elif dt == "covidUKexhalation":
            filenames = list(np.load("datasets/covidUK/entire_exhalation_filenames.npy"))
            
        elif dt == "covidUKcough":
            filenames = list(np.load("datasets/covidUK/entire_cough_filenames.npy"))
        
        label_list = list(range(len(filenames)))
        encoder_path = get_encoder_path(pretrain)
        ckpt = torch.load(encoder_path)

        model = initialize_pretrained_model(pretrain)
        model.eval()
        model.load_state_dict(ckpt["state_dict"], strict=False)
        
        test_size = 0.05

        _train_x, test_x, _train_y, test_y = train_test_split(filenames, label_list, test_size=test_size, random_state=1337)
        train_x, val_x, train_y, val_y = train_test_split(_train_x, _train_y, test_size=test_size, random_state=1337)
        max_len = 50 if modality == "cough" else 200

        if split == "test":
            test_x = test_x[:num_samples]
            test_y = test_y[:num_samples]
        elif split == "train":
            test_x = train_x[:num_samples]
            test_y = train_y[:num_samples]
        

        self_label_list = np.repeat(list(range(len(test_x))), 4)

        test_x, test_y = np.repeat(test_x, 4, axis=0), np.repeat(test_y, 4, axis=0)
        test_data = AudioDataset((test_x, test_y), augment=False, from_npy=True, max_len=max_len, crop_mode="random")

        batch_size = 128

        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=4
        )

        features, y_data = [], []
        model.to("cuda")

        with torch.no_grad():
            for (X, y) in test_loader:
                if torch.cuda.is_available():
                    X = X.cuda()
                out = model.extract_feature(X, dim=fea_dim).detach().cpu()
                features.append(out)
                y_data.append(y)
        
        features = torch.cat(features)
        y_data = torch.cat(y_data)

        x_data = features.numpy()
        y_data = y_data.numpy()

        # print(x_data.shape, self_label_list.shape)
        plot_tsne_individual(x_data, self_label_list, title="/" + pretrain + "_" + dt + "_" + split , n_instance=num_samples)




if __name__ == "__main__":

    data_source = {"covidbreath": 200, "covidcough": 50, "icbhi":50, "coughvid":50, "hf_lung":200, "covidUKexhalation": 100, "covidUKcough": 50}
    visualize_embedding_multiple(pretrain="operaCT", data_source=data_source, split="test", fea_dim=768)