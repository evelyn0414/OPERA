# Yuwei (Evelyn) Zhang
# yz798@cam.ac.uk
# Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking
# https://github.com/evelyn0414/OPERA
# some code below is referenced from https://github.com/CVxTz/COLA_pytorch

import random
import librosa
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT

from torchaudio import transforms as T

n_mels = 64


def crop_first(data, crop_size=128):
    return data[0 : crop_size, :]


def random_crop(data, crop_size=128):
    start = int(random.random() * (data.shape[0] - crop_size))
    return data[start : (start + crop_size), :]


def random_mask(data, rate_start=0.1, rate_seq=0.2):
    new_data = data.copy()
    mean = new_data.mean()
    prev_zero = False
    for i in range(new_data.shape[0]):
        if random.random() < rate_start or (
            prev_zero and random.random() < rate_seq
        ):
            prev_zero = True
            new_data[i, :] = mean
        else:
            prev_zero = False

    return new_data


def random_multiply(data):
    new_data = data.copy()
    return new_data * (0.9 + random.random() / 5.)


# not used 
class SpecAugment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.time_masking = T.TimeMasking(time_mask_param=80)
        # self.freq_masking = T.FrequencyMasking(freq_mask_param=80)
        # self.F, self.T = 20, 50

        # self.F, self.m_F, self.T, self.m_T = 48, 2, 160, 2
        # htsat
        self.F, self.m_F, self.T, self.m_T = 8, 2, 40, 2
        self.mask = "mean"
        # self.spec_aug = torch.nn.Sequential(
        #     # T.TimeStretch(0.8, fixed_rate=True),
        #     T.FrequencyMasking(freq_mask_param=self.F),
        #     T.TimeMasking(time_mask_param=self.T)
        # )
        self.mel_spectrogram = None
    
    def forward(self, img):
        self.mel_spectrogram = img
        self.mel_spectrogram = self.freq_mask()
        self.mel_spectrogram = self.time_mask()

        # return self.spec_aug(img)
        return self.mel_spectrogram
    
    def time_mask(self):
        if self.mask == 'mean':
            # maksing to mean value
            mask_value = self.mel_spectrogram.mean()
        elif self.mask == 'zero':
            # maksing to zero value
            mask_value = 0.

        # tau = self.mel_spectrogram.shape[1] # time frames
        tau = self.mel_spectrogram.shape[0]
        
        # apply m_T time masks to the mel spectrogram
        for i in range(self.m_T):
            t = int(np.random.uniform(0, self.T)) # [0, T)
            t0 = random.randint(0, tau - t) # [0, tau - t)
            # self.mel_spectrogram[:, :, t0:t0 + t] = mask_value
            self.mel_spectrogram[t0:t0 + t, :] = mask_value
            
        return self.mel_spectrogram

    def freq_mask(self):
        if self.mask == 'mean':
            # maksing to mean value
            mask_value = self.mel_spectrogram.mean()
        elif self.mask == 'zero':
            # maksing to zero value
            mask_value = 0.

        v = self.mel_spectrogram.shape[1] # no. of mel bins
        
        # apply m_F frequency masks to the mel spectrogram
        for i in range(self.m_F):
            f = int(np.random.uniform(0, self.F)) # [0, F)
            f0 = random.randint(0, v - f) # [0, v - f)
            # self.mel_spectrogram[:, f0:f0 + f, :] = mask_value
            self.mel_spectrogram[:, f0:f0 + f] = mask_value
            
        return self.mel_spectrogram

