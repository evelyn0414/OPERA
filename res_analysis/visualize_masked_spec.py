
import os
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torch
from src.model.models_mae import mae_vit_small

encoder_path =  "cks/model/encoder-operaGT.ckpt"

if not os.path.exists("fig/maked_spec/"):
    os.makedirs("fig/maked_spec/")


def random_masking(x, mask_ratio=0.7):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

def generate_mask(array, mask_indices, patch_size=(4,4)):
    patch_height, patch_width = patch_size
    rows, cols = array.shape
    
    array = np.array(array)
    masked_array = array.copy()
    
  
    patch_idx = 0
    for row in range(0, rows, patch_height):
        for col in range(0, cols, patch_width):
            #print(row, col, patch_idx)
            if patch_idx < len(mask_indices) and mask_indices[patch_idx] == 1:
                  masked_array[row:row+patch_height, col:col+patch_width] = 1
            patch_idx += 1
    
    return masked_array   
    
def replace_generation(x, pre, mask_indices, patch_size=(4,4)):
    patch_height, patch_width = patch_size
    rows, cols = pre.shape
    
    pre_new = pre.copy()
    #print(mask_indices)
    patch_idx = 0
    for row in range(0, rows, patch_height):
        for col in range(0, cols, patch_width):
            
            if patch_idx < len(mask_indices) and mask_indices[patch_idx] == 0:
                  # print(row, col, patch_idx)
                  pre_new[row:row+patch_height, col:col+patch_width] = x[row:row+patch_height, col:col+patch_width]
                
            patch_idx += 1
    return pre_new
       
def clipped_relu(x):
    return np.clip(x, 0, 1)

def visualize_spec(pretrain="operaGT", data_source='ICBHI'):

    #path = "datasets/hf_lung/entire_spec_npy/"
    # path = "datasets/covid/entire_spec_npy_8000/"
    if data_source == 'COUGHVID': 
        path = "datasets/coughvid/entire_spec_npy/"
        i = 0
        for file in os.listdir(path):
            data = np.load(path + file)
            i += 1
            if i>25: break
        name = 'coughvid25'
        data = data[:64,:]  #disply the firs 2s

    if data_source == 'ICBHI': 
        path = "datasets/icbhi/entire_spec_npy/"

        i = 0
        for file in os.listdir(path):
            data = np.load(path + file)
            i += 1
            if i>150: break
        name = 'icbhi_150'
        data = data[:256,:]  #disply the firs 8s


    
    plt.figure(figsize=(12, 6))  # Set the figure size
    sns.heatmap(np.flipud(data.T), annot=False, cmap='magma', cbar=False)
    plt.xlabel('Time frame', fontsize=15)
    plt.ylabel('Frequency bin', fontsize=15)
    plt.savefig('fig/maked_spec/orignal_Spec_' + name +'.png')

    data = data.reshape(1,-1,64)
    

   
    ckpt = torch.load(encoder_path)
    model =  mae_vit_small(norm_pix_loss=False,	
                                in_chans=1, audio_exp=True,	
                                img_size=(256,64),	
                                alpha=0.0, mode=0, use_custom_patch=False,	
                                split_pos=False, pos_trainable=False, use_nce=False,
                                decoder_mode=1, #decoder mode 0: global attn 1: swined local attn
                                mask_2d=False, mask_t_prob=0.7, mask_f_prob=0.3, 
                                no_shift=False).float() 
    model.load_state_dict(ckpt["state_dict"], strict=False)
    print('successufully load model from: ', encoder_path)
    model.eval()


    x = torch.tensor(data, dtype=torch.float)
    print(x.shape) #1x128x64
    # fea = model.forward_feature(x).detach().numpy()

    emb_enc, mask, ids_restore, _ = model.forward_encoder(x, mask_ratio=0.7, mask_2d=False)
    print('Encoder:',emb_enc.shape)

    x_show = generate_mask(data[0], mask[0])
    plt.figure(figsize=(12, 6))  # Set the figure size
    sns.heatmap(np.flipud(x_show.T), annot=False, cmap='magma', cbar=False)
    plt.xlabel('Time frame', fontsize=15)
    plt.ylabel('Frequency bin', fontsize=15)
    plt.savefig('fig/maked_spec/masked_Spec_' + name + '.png')


    pred, _, _ = model.forward_decoder(emb_enc, ids_restore)  # [N, L, p*p*3]

    print(pred.shape)

    pred = model.unpatchify(pred)
    print(pred.shape)
    prediction = pred[0,0,:,:].detach().numpy()


    prediction_new = replace_generation(data[0], prediction, mask[0]) # show the original patch for the non-masked pixel
  



    prediction_new = clipped_relu(prediction_new)
    plt.figure(figsize=(12, 6))  # Set the figure size
    sns.heatmap(np.flipud(prediction_new.T), annot=False, cmap='magma', cbar=False)
    plt.xlabel('Time frame', fontsize=15)
    plt.ylabel('Frequency bin', fontsize=15)
    plt.savefig('fig/maked_spec/recovered_Spec_' + name + '.png')



if __name__ == "__main__":

 
    visualize_spec(pretrain="operaGT", data_source='ICBHI')
     