# Yuwei (Evelyn) Zhang
# yz798@cam.ac.uk
# Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking
# https://github.com/evelyn0414/OPERA

import os
import numpy as np
import torch
from src.model.models_cola import Cola
from src.model.models_mae import mae_vit_small
from huggingface_hub.file_download import hf_hub_download

SR = 16000

# You can add your own model path here.

ENCODER_PATH_OPERA_CE_EFFICIENTNET = "cks/model/encoder-operaCE.ckpt"
ENCODER_PATH_OPERA_CT_HT_SAT = "cks/model/encoder-operaCT.ckpt"
ENCODER_PATH_OPERA_GT_VIT =  "cks/model/encoder-operaGT.ckpt"


def get_encoder_path(pretrain):
    encoder_paths = {
        "operaCT": ENCODER_PATH_OPERA_CT_HT_SAT,
        "operaCE": ENCODER_PATH_OPERA_CE_EFFICIENTNET,
        "operaGT": ENCODER_PATH_OPERA_GT_VIT
        }
    if not os.path.exists(encoder_paths[pretrain]):
        print("model ckpt not found, trying to download from huggingface")
        download_ckpt(pretrain)
    return encoder_paths[pretrain]


def download_ckpt(pretrain):
    model_repo = "evelyn0414/OPERA"
    model_name = "encoder-" + pretrain
    hf_hub_download(model_repo, model_name, local_dir="cks/model")


def extract_opera_feature(sound_dir_loc, pretrain="operaCE", input_sec=8, from_spec=False, dim=1280, pad0=False):
    """

    """
    from src.util import get_split_signal_librosa, pre_process_audio_mel_t, split_pad_sample, decide_droplast, get_entire_signal_librosa
    from tqdm import tqdm

    print("extracting feature from {} model with input_sec {}".format(pretrain, input_sec))

    MAE = ("mae" in pretrain or "GT" in pretrain)

    encoder_path = get_encoder_path(pretrain)
    ckpt = torch.load(encoder_path)
    model = initialize_pretrained_model(pretrain)
    model.eval()
    model.load_state_dict(ckpt["state_dict"], strict=False)

    opera_features = []

    for audio_file in tqdm(sound_dir_loc):

        if MAE:
            if from_spec:
                data = [audio_file[i: i+256] for i in range(0, len(audio_file), 256)]
            else:
                data = get_split_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=input_sec) ##8.18s --> T=256
            features = []
            for x in data:
                if x.shape[0]>=16: # Kernel size can't be greater than actual input size
                    x = np.expand_dims(x, axis=0)
                    x = torch.tensor(x, dtype=torch.float)
                    fea = model.forward_feature(x).detach().numpy()
                    features.append(fea)
            features_sta = np.mean(features, axis=0)
            # print('MAE ViT feature dim:', features_sta.shape)
            opera_features.append(features_sta.tolist())
        else:
            #  put entire audio into the model
            if from_spec:
                data = audio_file
            else:
                # input is filename of an audio
                if pad0:
                    data = get_entire_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=input_sec, pad=True, types='zero')
                else:
                    data = get_entire_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=input_sec, pad=True)
            
            data = np.array(data)

            # for entire audio, batchsize = 1
            data = np.expand_dims(data, axis=0)

            x = torch.tensor(data, dtype=torch.float)
            features = model.extract_feature(x, dim).detach().numpy()

            # for entire audio, batchsize = 1
            opera_features.append(features.tolist()[0])

    x_data = np.array(opera_features)
    if MAE: x_data = x_data.squeeze(1) 
    print(x_data.shape)
    return x_data


def initialize_pretrained_model(pretrain):
    if pretrain == "operaCT":
        model = Cola(encoder="htsat")
    elif pretrain == "operaCE":
        model = Cola(encoder="efficientnet")
    elif pretrain == "operaGT":
        model =  mae_vit_small(norm_pix_loss=False,
                            in_chans=1, audio_exp=True,
                            img_size=(256,64),
                            alpha=0.0, mode=0, use_custom_patch=False,
                            split_pos=False, pos_trainable=False, use_nce=False,
                            decoder_mode=1, #decoder mode 0: global attn 1: swined local attn
                            mask_2d=False, mask_t_prob=0.7, mask_f_prob=0.3,
                            no_shift=False).float()
    else:
        raise NotImplementedError(f"Model not exist: {pretrain}, please check the parameter.")
    return model


