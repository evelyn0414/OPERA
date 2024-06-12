# Yuwei (Evelyn) Zhang
# yz798@cam.ac.uk
# Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking
# https://github.com/evelyn0414/OPERA
# some code below is referenced from https://github.com/raymin0223/patch-mix_contrastive_learning and https://github.com/CVxTz/COLA_pytorch



import os
import torch
import torchaudio
from torchaudio import transforms as T
from scipy.signal import butter, lfilter
import pandas as pd
import librosa
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import time


def crop_first(data, crop_size=128):
    return data[0: crop_size, :]


def random_crop(data, crop_size=128):
    start = int(random.random() * (data.shape[0] - crop_size))
    return data[start: (start + crop_size), :]


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


def _extract_lungsound_annotation(file_name, data_folder):
    tokens = file_name.strip().split('_')
    # print(tokens)
    recording_info = pd.DataFrame(data=[tokens], columns=[
                                  'Patient Number', 'Recording index', 'Chest location', 'Acquisition mode', 'Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(
        data_folder, file_name + '.txt'), names=['Start', 'End', 'Crackles', 'Wheezes'], delimiter='\t')

    return recording_info, recording_annotations


def get_annotations(class_split='cycle', data_folder="datasets/icbhi/ICBHI_final_database/"):
    if class_split == 'cycle':
        filenames = [f.strip().split('.')[0]
                     for f in os.listdir(data_folder) if '.wav' in f]
        # print(filenames[0])

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            annotation_dict[f] = ann

    elif class_split == 'diagnosis':
        filenames = [f.strip().split('.')[0]
                     for f in os.listdir(data_folder) if '.txt' in f]
        tmp = pd.read_csv(
            data_folder + 'ICBHI_Challenge_diagnosis.txt', names=['Disease'], delimiter='\t')

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            ann.drop(['Crackles', 'Wheezes'], axis=1, inplace=True)

            disease = tmp.loc[int(f.strip().split('_')[0]), 'Disease']
            ann['Disease'] = disease

            annotation_dict[f] = ann

    return annotation_dict


def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    return b, a


def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)

    return y


def _slice_data_librosa(start, end, data, sample_rate):
    """
    RespireNet paper..
    sample_rate denotes how many sample points for one second
    """
    max_ind = len(data)
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)

    return data[start_ind: end_ind]


def get_individual_segments_librosa(data_folder, filename, input_sec=8, sample_rate=16000, hop_sec=2, butterworth_filter=None, spectrogram=False):
    sample_data = []

    # load file with specified sample rate (also converts to mono)
    data, rate = librosa.load(os.path.join(
        data_folder, filename+'.wav'), sr=sample_rate)
    start = 0
    end = input_sec

    if butterworth_filter:
        # butter bandpass filter
        data = _butter_bandpass_filter(
            lowcut=200, highcut=1800, fs=sample_rate, order=butterworth_filter)

    # Trim leading and trailing silence from an audio signal.
    FRAME_LEN = int(sample_rate / 10)  #
    HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
    yt, index = librosa.effects.trim(
        data, frame_length=FRAME_LEN, hop_length=HOP
    )

    # check audio not too short
    duration = librosa.get_duration(y=yt, sr=rate)
    if duration < 2:
        print("Warning: audio too short, skipped")
        return []

    # samples that are long enough
    while end <= duration:
        audio_chunk = _slice_data_librosa(start, end, yt, rate)
        sample_data.append(audio_chunk)
        start += hop_sec
        end += hop_sec

    #  if last segment shorter than 8s,  but still longer than 2
    if start + 2 < duration:
        audio_chunk = _slice_data_librosa(start, end, yt, rate)
        audio_chunk = split_pad_sample([audio_chunk, 0, 0], 8, rate)[0][0]
        sample_data.append(audio_chunk)
    # print(filename, "audio duration", duration, "segments", len(sample_data))

    # directly process to spectrogram
    if spectrogram:
        audio_image = []
        for audio in sample_data:
            image = pre_process_audio_mel_t(audio.squeeze())
            audio_image.append(image)
        return audio_image

    return sample_data


def get_entire_signal_librosa(data_folder, filename, input_sec=8, sample_rate=16000, butterworth_filter=None, spectrogram=False, pad=False, from_cycle=False, yt=None):

    if not from_cycle:

        # load file with specified sample rate (also converts to mono)
        data, rate = librosa.load(os.path.join(
            data_folder, filename+'.wav'), sr=sample_rate)

        if butterworth_filter:
            # butter bandpass filter
            data = _butter_bandpass_filter(
                lowcut=200, highcut=1800, fs=sample_rate, order=butterworth_filter)

        # Trim leading and trailing silence from an audio signal.
        FRAME_LEN = int(sample_rate / 10)  #
        HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
        yt, index = librosa.effects.trim(
            data, frame_length=FRAME_LEN, hop_length=HOP
        )

    # check audio not too short

    duration = librosa.get_duration(y=yt, sr=sample_rate)
    if duration < input_sec:
        if not pad:
            print("Warning: audio too short, skipped")
            return None
        else:
            yt = split_pad_sample([yt, 0, 0], input_sec, sample_rate)[0][0]

    # directly process to spectrogram
    if spectrogram:
        # # visualization for testing the spectrogram parameters
        # plot_melspectrogram(yt.squeeze(), title=filename.replace("/", "-"))
        return pre_process_audio_mel_t(yt.squeeze(), f_max=8000)

    return yt


def get_split_signal_librosa(data_folder, filename, input_sec=8, sample_rate=16000, butterworth_filter=None, spectrogram=False, trim_tail=False):
    # print(os.path.join(data_folder, filename+'.wav'))

    # load file with specified sample rate (also converts to mono)
    data, rate = librosa.load(os.path.join(
        data_folder, filename+'.wav'), sr=sample_rate)

    if butterworth_filter:
        # butter bandpass filter
        data = _butter_bandpass_filter(
            lowcut=200, highcut=1800, fs=sample_rate, order=butterworth_filter)

    # Trim leading and trailing silence from an audio signal.
    FRAME_LEN = int(sample_rate / 10)  #
    HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
    yt, index = librosa.effects.trim(
        data, frame_length=FRAME_LEN, hop_length=HOP
    )

    # # check audio not too short
    # duration = librosa.get_duration(y=yt, sr=rate)
    # if duration < input_sec:
    #     print("Warning: audio too short, skipped")
    #     return None

    drop_last = False
    if trim_tail:
        drop_last = decide_droplast(yt, rate, input_sec)

    audio_chunks = [res[0]
                    for res in split_pad_sample([yt, 0, 0], input_sec, rate)]
    if drop_last:
        audio_chunks.pop()

    if not spectrogram:
        return audio_chunks

    # directly process to spectrogram
    audio_image = []
    for audio in audio_chunks:
        image = pre_process_audio_mel_t(audio.squeeze(), f_max=8000)
        audio_image.append(image)
    return audio_image

    # return [pre_process_audio_mel_t(chunk.squeeze()) for chunk in audio_chunks]


def decide_droplast(yt, sr, input_sec):
    duration = librosa.get_duration(y=yt, sr=sr)
    return duration > input_sec and (duration % input_sec) * 2 < input_sec


def get_individual_cycles_librosa(class_split, recording_annotations, data_folder, filename, sample_rate, n_cls, butterworth_filter=None):
    """
    RespireNet paper..
    Used to split each individual sound file into separate sound clips containing one respiratory cycle each
    output: [(audio_chunk:np.array, label:int), (...)]
    """
    sample_data = []

    # load file with specified sample rate (also converts to mono)
    data, rate = librosa.load(os.path.join(
        data_folder, filename+'.wav'), sr=sample_rate)

    if butterworth_filter:
        # butter bandpass filter
        data = _butter_bandpass_filter(
            lowcut=200, highcut=1800, fs=sample_rate, order=butterworth_filter)

    for idx in recording_annotations.index:
        row = recording_annotations.loc[idx]

        start = row['Start']  # time (second)
        end = row['End']  # time (second)
        audio_chunk = _slice_data_librosa(start, end, data, rate)

        if class_split == 'cycle':
            crackles = row['Crackles']
            wheezes = row['Wheezes']
            sample_data.append(
                (audio_chunk, _get_lungsound_label(crackles, wheezes, n_cls)))
        elif class_split == 'diagnosis':
            disease = row['Disease']
            sample_data.append(
                (audio_chunk, _get_diagnosis_label(disease, n_cls)))

    return sample_data
    fade_samples_ratio = 16
    desired_length = 8
    sample_rate = 16000
    fade_samples = int(sample_rate / fade_samples_ratio)
    fade_out = T.Fade(
        fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = desired_length * sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
    else:
        # if args.pad_types == 'zero':
        #     tmp = torch.zeros(1, target_duration, dtype=torch.float32)
        #     diff = target_duration - data.shape[-1]
        #     tmp[..., diff//2:data.shape[-1]+diff//2] = data
        #     data = tmp
        # elif args.pad_types == 'repeat':
        ratio = math.ceil(target_duration / data.shape[-1])
        data = data.repeat(1, ratio)
        data = data[..., :target_duration]
        data = fade_out(data)

    return data


def _get_lungsound_label(crackle, wheeze, n_cls):
    if n_cls == 4:
        if crackle == 0 and wheeze == 0:
            return 0
        elif crackle == 1 and wheeze == 0:
            return 1
        elif crackle == 0 and wheeze == 1:
            return 2
        elif crackle == 1 and wheeze == 1:
            return 3

    elif n_cls == 2:
        if crackle == 0 and wheeze == 0:
            return 0
        else:
            return 1


def _get_diagnosis_label(disease, n_cls):
    if n_cls == 3:
        if disease in ['COPD', 'Bronchiectasis', 'Asthma']:
            return 1
        elif disease in ['URTI', 'LRTI', 'Pneumonia', 'Bronchiolitis']:
            return 2
        else:
            return 0

    elif n_cls == 2:
        if disease == 'Healthy':
            return 0
        else:
            return 1


def pre_process_audio_mel_t(audio, sample_rate=16000, n_mels=64, f_min=50, f_max=2000, nfft=1024, hop=512):
    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    # convert scale to dB from magnitude
    S = librosa.power_to_db(S, ref=np.max)
    if S.max() != S.min():
        mel_db = (S - S.min()) / (S.max() - S.min())
    else:
        mel_db = S
        print("warning in producing spectrogram!")

    return mel_db.T


def _zero_padding(source, output_length):
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)

    frac = src_length / output_length
    if frac < 0.5:
        # tile forward sounds to fill empty space
        cursor = 0
        while (cursor + src_length) < output_length:
            copy[cursor:(cursor + src_length)] = source[:]
            cursor += src_length
    else:
        # [src_length:] part will be zeros
        copy[:src_length] = source[:]

    return copy


def _equally_slice_pad_sample(sample, desired_length, sample_rate):
    """
    pad_type == 0: zero-padding
    if sample length > desired_length, 
    all equally sliced samples with samples_per_slice number are zero-padded or recursively duplicated
    """
    output_length = int(
        desired_length * sample_rate)  # desired_length is second
    soundclip = sample[0].copy()
    n_samples = len(soundclip)

    total_length = n_samples / sample_rate  # length of cycle in seconds
    # get the minimum number of slices needed
    n_slices = int(math.ceil(total_length / desired_length))
    samples_per_slice = n_samples // n_slices

    output = []  # holds the resultant slices
    src_start = 0  # staring index of the samples to copy from the sample buffer
    for i in range(n_slices):
        src_end = min(src_start + samples_per_slice, n_samples)
        length = src_end - src_start

        copy = _zero_padding(soundclip[src_start:src_end], output_length)
        output.append((copy, sample[1], sample[2]))
        src_start += length

    return output


def _duplicate_padding(sample, source, output_length, sample_rate, types):
    # pad_type == 1 or 2
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)
    left = output_length - src_length  # amount to be padded

    if types == 'repeat':
        aug = sample
    # else:
    #     aug = augment_raw_audio(sample, sample_rate)

    while len(aug) < left:
        aug = np.concatenate([aug, aug])

    prob = random.random()
    if prob < 0.5:
        # pad the back part of original sample
        copy[left:] = source
        copy[:left] = aug[len(aug)-left:]
    else:
        # pad the front part of original sample
        copy[:src_length] = source[:]
        copy[src_length:] = aug[:left]

    return copy


def split_pad_sample(sample, desired_length, sample_rate, types='repeat'):
    """
    if the audio sample length > desired_length, then split and pad samples
    else simply pad samples according to pad_types
    * types 'zero'   : simply pad by zeros (zero-padding)
    * types 'repeat' : pad with duplicate on both sides (half-n-half)
    * types 'aug'    : pad with augmented sample on both sides (half-n-half)	
    """
    if types == 'zero':
        return _equally_slice_pad_sample(sample, desired_length, sample_rate)

    output_length = int(desired_length * sample_rate)
    soundclip = sample[0].copy()
    n_samples = len(soundclip)

    output = []
    if n_samples > output_length:
        """
        if sample length > desired_length, slice samples with desired_length then just use them,
        and the last sample is padded according to the padding types
        """
        # frames[j] = x[j * hop_length : j * hop_length + frame_length]
        frames = librosa.util.frame(
            soundclip, frame_length=output_length, hop_length=output_length//2, axis=0)
        for i in range(frames.shape[0]):
            output.append((frames[i], sample[1], sample[2]))

        # get the last sample
        last_id = frames.shape[0] * (output_length//2)
        last_sample = soundclip[last_id:]

        padded = _duplicate_padding(
            soundclip, last_sample, output_length, sample_rate, types)
        output.append((padded, sample[1], sample[2]))
    else:  # only pad
        padded = _duplicate_padding(
            soundclip, soundclip, output_length, sample_rate, types)
        output.append((padded, sample[1], sample[2]))

    return output


def train_test_split_from_list(X, Y, train_test):
    print(X.shape, len(Y), len(train_test))
    X_train, X_test, y_train, y_test = [], [], [], []
    for i, split in enumerate(train_test):
        if split == "train":
            X_train.append(X[i])
            y_train.append(Y[i])
        else:
            X_test.append(X[i])
            y_test.append(Y[i])
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def plot_tsne(x_plot, y_plot, order=None, color="hls", title=""):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40,
                n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(x_plot)
    plt.figure(figsize=(16, 10))
    if color == "paired":
        cm = sns.color_palette("Paired", 10)
    else:
        cm = sns.color_palette()
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=y_plot,
        hue_order=order,
        palette=cm,
        # sns.color_palette("hls", 8),
        # palette=sns.color_palette("Paired", 10),
        legend="full",
        alpha=0.7
    )
    if title == "":
        title = str(time.time())
    plt.savefig("fig/tsne/" + title + ".png", bbox_inches='tight')
    print("t-sne plot saved to", "fig/tsne/" + title + ".png")


def plot_tsne_individual(x_plot, y_plot, order=None, title="", n_instance=1401):
    from sklearn.manifold import TSNE
    # import colorcet as cc
    sns.set_theme()
    # palette = sns.color_palette(cc.glasbey, n_colors=n_instance)
    # print("using cc palette")
    palette = sns.color_palette("hls", 10)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(x_plot)
    plt.figure(figsize=(4, 4))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=y_plot,
        hue_order=order,
        palette=palette,
        # legend="auto",
        # legend="brief",
        legend=False,
        alpha=0.85,
        s=50
    )
    if title == "":
        title = str(time.time())
    
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    plt.xlabel("T-SNE dim 1", fontsize=14)
    plt.ylabel("T-SNE dim 2", fontsize=14)
    plt.savefig("fig/tsne_individual/" + title + ".png", bbox_inches='tight')
    print("fig/tsne_individual/" + title + ".png")


def plot_melspectrogram(audio, title="", sample_rate=16000, n_mels=64, f_min=50, f_max=2000, nfft=1024, hop=512):
    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sample_rate,
                                   fmin=f_min,
                                   fmax=f_max, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    if title == "":
        title = str(time.time())
    ax.set(title='Mel-frequency spectrogram ' + title)
    plt.savefig("fig/spectrogram/" + title + ".png")
    plt.clf()


def get_weighted_loss_icbhi():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_nums = [16, 399]  # training class distribution
    weights = torch.tensor(class_nums, dtype=torch.float32)
    weights = 1.0 / (weights / weights.sum())
    weights /= weights.sum()
    loss_fun = nn.CrossEntropyLoss(weight=weights.to(device))
    return loss_fun


def get_weighted_loss_icbhidisease():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_nums = [2063, 1215, 501, 363]  # training class distribution
    weights = torch.tensor(class_nums, dtype=torch.float32)
    weights = 1.0 / (weights / weights.sum())
    weights /= weights.sum()
    loss_fun = nn.CrossEntropyLoss(weight=weights.to(device))
    return loss_fun


def downsample_balanced_dataset(x_train, y_train):
    # Find unique classes in y_train
    classes = np.unique(y_train)

    # Find the minimum number of samples among classes
    min_samples = min(np.bincount(y_train))

    # Initialize lists to store downsampled data
    x_downsampled = []
    y_downsampled = []

    # Downsample each class
    for c in classes:
        # Get indices of samples belonging to class c
        indices = np.where(y_train == c)[0]

        # Randomly select min_samples samples
        selected_indices = np.random.choice(
            indices, min_samples, replace=False)

        # Add selected samples to downsampled data
        x_downsampled.extend(x_train[selected_indices])
        y_downsampled.extend(y_train[selected_indices])

    # Convert lists to numpy arrays
    x_downsampled = np.array(x_downsampled)
    y_downsampled = np.array(y_downsampled)

    return x_downsampled, y_downsampled


def get_split_signal_fbank_pad(data_folder, filename, input_sec=8, sample_rate=16000, butterworth_filter=None, spectrogram=False, trim_tail=False):
    from src.util import split_pad_sample

    data, rate = librosa.load(os.path.join(
        data_folder, filename+'.wav'), sr=sample_rate)

    if butterworth_filter:
        # butter bandpass filter
        data = _butter_bandpass_filter(
            lowcut=200, highcut=1800, fs=sample_rate, order=butterworth_filter)

    # Trim leading and trailing silence from an audio signal.
    FRAME_LEN = int(sample_rate / 10)  #
    HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
    yt, index = librosa.effects.trim(
        data, frame_length=FRAME_LEN, hop_length=HOP
    )

    drop_last = False
    if trim_tail:
        drop_last = decide_droplast(yt, rate, input_sec)

    audio_chunks = [res[0]
                    for res in split_pad_sample([yt, 0, 0], input_sec, rate)]

    if drop_last:
        audio_chunks.pop()

    if not spectrogram:
        return audio_chunks

    # directly process to spectrogram
    audio_image = []
    for waveform in audio_chunks:
        # image = pre_process_audio_mel_t(audio.squeeze(), f_max=8000)

        waveform = waveform - waveform.mean()
        waveform = torch.tensor(waveform).reshape([1, -1])
        # print(waveform.shape)
        if waveform.shape[1] > 400:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, channel=0, frame_length=25, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
                                                      window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

            # print( waveform.shape[1]/sample_rate, fbank.shape)
            audio_image.append(fbank)
    return audio_image
    """Normalization of Input Batch.

    Note:
        Unlike other blocks, use this with *batch inputs*.

    Args:
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, axis=[0, 2, 3]):
        super().__init__()
        self.axis = axis

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _mean = X.mean(dim=self.axis, keepdims=True)
        _std = torch.clamp(X.std(dim=self.axis, keepdims=True),
                           torch.finfo().eps, torch.finfo().max)
        return ((X - _mean) / _std)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(axis={self.axis})'
        return format_string
