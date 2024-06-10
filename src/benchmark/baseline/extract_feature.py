# Yuwei (Evelyn) Zhang
# yz798@cam.ac.uk
# Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking
# https://github.com/evelyn0414/OPERA

import numpy as np
import librosa
import os
import json
import glob as gb
from tqdm import tqdm
import torch
import torchaudio
import opensmile


SR = 22050  # sample rate

def extract_opensmile_features(audio_file):
    # the emobase feature set with 988 acoustic features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    opensmile_features = smile.process_file(audio_file)
    return opensmile_features


def opensmile_feature_names():
    return smile.feature_names


def extract_vgg_feature(sound_dir_loc, from_signal=False):
    import tensorflow as tf
    import urllib
    import sys
    sys.path.append('./src/benchmark/baseline/vggish')
    from src.benchmark.baseline.vggish import vggish_input
    from src.benchmark.baseline.vggish import vggish_params
    from src.benchmark.baseline.vggish import vggish_slim

    SR_VGG = 16000  # VGG pretrained model sample rate
    x_data = []

    checkpoint_path = "./src/benchmark/baseline/vggish/vggish_model.ckpt"
    pca_params_path = "./src/benchmark/baseline/vggish/vggish_pca_params.npz"
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        # load pre-trained model
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME
        )
        for file in tqdm(sound_dir_loc):

            if from_signal:
                y, sr = file, SR
            else:
                y, sr = librosa.load(
                    file, sr=SR, mono=True, offset=0.0, duration=None
                )

            duration = librosa.get_duration(y=y, sr=sr)

            input_batch = vggish_input.waveform_to_examples(
                y, SR_VGG
            )  # 3x96x64 --> 3x128
            [features] = sess.run(
                [embedding_tensor], feed_dict={features_tensor: input_batch}
            )
            # print(features.shape) # (num_frames, 128)
            features_sta = np.mean(features, axis=0)
            x_data.append(features_sta.tolist())
    
    x_data = np.array(x_data)
    return x_data


def extract_clap_feature(sound_dir_loc, single_file=False, version='2022'):
    from src.benchmark.baseline.msclap import CLAP
    clap_model = CLAP(version=version, use_cuda=True)
    
    if single_file:
        audio_embeddings = clap_model.get_audio_embeddings(sound_dir_loc)
        return np.array(audio_embeddings)

    x_data = []
    num_files = len(sound_dir_loc)
    batch_size = 512
    num_batches = (num_files + batch_size - 1) // batch_size 

    for batch_index in tqdm(range(num_batches)):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, num_files)
        batch_files = sound_dir_loc[start_index:end_index]
        audio_embeddings = clap_model.get_audio_embeddings(batch_files)
        x_data.extend(audio_embeddings.cpu())
    x_data = np.array(x_data)
    print(x_data.shape)
    return x_data


def extract_audioMAE_feature(sound_dir_loc, input_sec=10):
    """
    input_sec and trim_tail deprecated
    trim_tail: drop last residual segment if too short, shorter than one half of input_sec
    """
    from tqdm import tqdm
    from src.benchmark.baseline.audioMAE.models_mae import mae_vit_small, vit_base_patch16

    encoder_path = "src/benchmark/baseline/audioMAE/ViTB_pretrained.pth"
    ckpt = torch.load(encoder_path)

    model = vit_base_patch16(
        in_chans=1,
        img_size=(1024,128),
        drop_path_rate=0.1,
        global_pool=True,
        mask_2d=False,
        use_custom_patch=False)
    
    model.eval()
    model.load_state_dict(ckpt["model"], strict=False)
    mae_features = []

    for audio_file in tqdm(sound_dir_loc):
        data = get_split_signal_fbank("", audio_file[:-4], input_sec=10) 
        features = []
        for x in data:
            # print(x.shape)
            if x.shape[1]>=16: # Kernel size can't be greater than actual input size
                x = np.expand_dims(x, axis=0)
                
                x = torch.tensor(x, dtype=torch.float)
                fea = model.forward_feature(x).detach().numpy()

                features.append(fea)

        features_sta = np.mean(features, axis=0)

        mae_features.append(features_sta.tolist())

    x_data = np.array(mae_features)
    x_data = x_data.squeeze(1)
    print(x_data.shape)
    return x_data


def get_split_signal_fbank(data_folder, filename, input_sec=10, sample_rate=16000):
  
    data, rate = librosa.load(os.path.join(data_folder, filename+'.wav'), sr=sample_rate)

    # Trim leading and trailing silence from an audio signal.
    FRAME_LEN = int(sample_rate / 10)  # 
    HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
    yt, index = librosa.effects.trim(
                data, frame_length=FRAME_LEN, hop_length=HOP
            )

    audio_chunks = [res for res in split_sample(yt, input_sec, rate)]
    
    # directly process to spectrogram
    audio_image = []
    for waveform in audio_chunks:
        waveform = waveform - waveform.mean()
        waveform = torch.tensor(waveform).reshape([1,-1])
        #print(waveform.shape)
        if waveform.shape[1] > 400: 
            fbank = torchaudio.compliance.kaldi.fbank(waveform, channel=0, frame_length=25, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
                                                    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        

            #print( waveform.shape[1]/sample_rate, fbank.shape)
            audio_image.append(fbank)
    return audio_image


def split_sample(sample, desired_length, sample_rate, hop_len=0):

    output_length = int(desired_length * sample_rate)
    soundclip = sample.copy()
    n_frames = int(np.ceil(len(soundclip) / output_length ))
    output=[]
    for i in range(n_frames):
        frame = soundclip[output_length*i: output_length*(i+1)]
        output.append(frame)

       
    return output

