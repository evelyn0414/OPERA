# Yuwei (Evelyn) Zhang
# yz798@cam.ac.uk
# Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking
# https://github.com/evelyn0414/OPERA
# some code below is referenced from https://github.com/CVxTz/COLA_pytorch


import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from lightning.pytorch.utilities import CombinedLoader
from src.util import random_crop, random_mask, random_multiply
from src.model.models_cola import Cola, ColaMD


def combine_dataloaders(dataloaders, train=False):
    if train:
        return CombinedLoader(dataloaders, 'max_size_cycle')
    return CombinedLoader(dataloaders, "sequential")


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len=200, augment=True, from_npy=False, labels=None, method="cola"):
        """
        max len: 251 for 8 secs, 157 for 5 second, 126 for 4 seconds, 63 for 2 seconds, 32 for 1 second
        """
        self.data = data
        self.max_len = max_len
        self.augment = augment
        self.from_npy = from_npy
        self.labels = labels
        self.method = method

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.from_npy:
            npy_path = self.data[idx]
            x = np.load(npy_path + ".npy")
        else:
            x = self.data[idx]

        if self.method == "cola":

            if self.augment:
                x = random_mask(x)

            x1 = random_crop(x, crop_size=self.max_len)
            x2 = random_crop(x, crop_size=self.max_len)

            if self.augment:
                x1 = random_multiply(x1)
                x2 = random_multiply(x2)

            x1 = torch.tensor(x1, dtype=torch.float)
            x2 = torch.tensor(x2, dtype=torch.float)

        if self.labels is None:
            return x1, x2

        return (x1, x2), self.labels[idx]


class DecayLearningRate(pl.Callback):
    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = []
            for param_group in optimizer.param_groups:
                group.append(param_group["lr"])
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.99
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group


def train_multiple_data(title, data_source={"covidbreath": 251},  dim_fea=1280, dim_hidden=1280, dim_out=512, encoder="efficientnet", n_epoches=512, training_method="cola"):
    print(data_source)

    method = training_method

    batch_size = 128
    epochs = n_epoches

    num_batch = []

    #  constructing dataloaders
    train_loaders, val_loaders = [], []

    print('===============================================================================')
    print('start loading data:')
    for dt, max_len in data_source.items():
        from_npy = True
        if dt in ['covidbreath', 'covidcough']:
            modality = dt[5:]
            filenames = list(np.load(
                "datasets/covid19-sounds/SSL_entireaudio_filenames_{}.npy".format(modality)))

        elif dt == "icbhi":
            #  training with audio
            icbhi_filenames = np.load(
                "datasets/icbhi/entire_spec_filenames.npy")
            train_test = np.load("datasets/icbhi/entire_spec_split.npy")
            # exclude testing
            filenames = list(icbhi_filenames[train_test == "train"])

        elif dt == "icbhicycle":
            # training with cycle:
            icbhi_filenames = np.load(
                "datasets/icbhi/cycle_spec_pad2_name.npy")
            train_test = np.load("datasets/icbhi/cycle_spec_split.npy")
            # exclude testing
            filenames = list(icbhi_filenames[train_test == "train"])

        elif dt == "coughvid":
            filenames = list(
                np.load("datasets/coughvid/entire_spec_filenames.npy"))

        elif dt == "hf_lung":
            filenames = list(
                np.load("datasets/hf_lung/entire_spec_filenames.npy"))

        elif dt == "covidUKexhalation":
            filenames = list(
                np.load("datasets/covidUK/entire_exhalation_filenames.npy"))

        elif dt == "covidUKcough":
            filenames = list(
                np.load("datasets/covidUK/entire_cough_filenames.npy"))

        # # plotting data length distribution
        # data_lengths = []
        # for file_path in filenames:
        #     data = np.load(file_path + ".npy")
        #     data_lengths.append(data.shape[0])

        # import matplotlib.pyplot as plt
        # plt.hist(data_lengths, bins=20, color='skyblue', edgecolor='black')
        # plt.xlabel('Data Length')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Data Length ' + dt)
        # plt.grid(True)
        # plt.savefig("fig/training/{}_length_hist.png".format(dt))
        # plt.clf()

        train, test = train_test_split(
            filenames, test_size=0.1, random_state=1337)

        train_data = AudioDataset(
            train, augment=True, from_npy=True, max_len=max_len, method=method)
        val_data = AudioDataset(
            test, augment=True, from_npy=True, max_len=max_len, method=method)

        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=7
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=True, num_workers=7
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        print(dt, 'Length of Training, Validation',
              len(train_loader), len(val_loader))
        num_batch.append(len(train_loader))

    print('===============================================================================')
    train_loader = combine_dataloaders(train_loaders, train=True)
    val_loader = combine_dataloaders(val_loaders)

    if training_method == "cola":
        model = ColaMD(encoder=encoder, max_len=data_source, dim_fea=dim_fea,
                       dim_hidden=dim_hidden, dim_out=dim_out, num_batch=num_batch)
    logger = CSVLogger(
        save_dir="cks/logs",
        name="combined",
        version=title,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss", mode="min", dirpath="cks/model/combined/" + "_".join(data_source.keys()),
        filename='encoder-' + title +
        '-{epoch:02d}--{valid_acc:.2f}-{valid_loss:.4f}',
        every_n_epochs=50,
        save_top_k=5
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[DecayLearningRate(), checkpoint_callback],
    )

    print('======================SSL Training==============================')
    trainer.fit(model, train_loader, val_loader)
    print('======================SSL Testing==============================')
    trainer.test(dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str)
    parser.add_argument("--data", type=str, default="multiple")

    # for training with multiple data
    parser.add_argument("--covidbreath", type=bool, default=False)
    parser.add_argument("--covidcough", type=bool, default=False)
    parser.add_argument("--icbhi", type=bool, default=False)
    parser.add_argument("--icbhicycle", type=bool, default=False)
    parser.add_argument("--coughvid", type=bool, default=False)
    parser.add_argument("--hf_lung", type=bool, default=False)
    parser.add_argument("--covidUKexhalation", type=bool, default=False)
    parser.add_argument("--covidUKcough", type=bool, default=False)

    # control training
    parser.add_argument("--dim_hidden", type=int, default=1280)
    parser.add_argument("--dim_out", type=int, default=512)
    parser.add_argument("--encoder", type=str, default="efficientnet")
    parser.add_argument("--epoches", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    # training goal
    parser.add_argument("--method", type=str, default="cola")

    args = parser.parse_args()

    optimal_max_len = {"covidbreath": 200, "covidcough": 50, "icbhi": 50, "icbhicycle": 50,
                       "coughvid": 50, "hf_lung": 200, "covidUKexhalation": 100, "covidUKcough": 50}

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_source = {}
    for dt, max_len in optimal_max_len.items():
        if getattr(args, dt) is True:
            data_source[dt] = max_len
    train_multiple_data(args.title, data_source=data_source, dim_hidden=args.dim_hidden,
                        dim_out=args.dim_out, encoder=args.encoder, n_epoches=args.epoches, training_method=args.method)
