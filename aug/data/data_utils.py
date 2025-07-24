import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import random

def list_wav_files(directory: str) -> List[str]:
    """
    Lists all .wav files in a directory.
    
    Args:
        directory (str): Directory to search for .wav files.
    
    Returns:
        List[str]: List of .wav filenames in the directory.
    """

    return [f for f in os.listdir(directory) if f.endswith('.wav')]

def modify_sound_dir_loc(sound_dir_loc: np.ndarray, output_path: str) -> np.ndarray:
    """
    Modifies the sound_dir_loc to point to the output_path.
    """
    modified_paths = []
    for path in sound_dir_loc:
        filename = os.path.basename(path)
        modified_path = os.path.join(output_path, filename)
        modified_paths.append(modified_path)
    return np.array(modified_paths)

def load_dataset_from_npy(config: dict, split: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Loads dataset information from preprocessed .npy files using explicit file names from config.
    
    Args:
        feature_dir (str): Directory containing the .npy files (e.g., 'feature/coughvid_eval/')
        config (dict): Configuration dictionary containing file names
        split (str): Data split to filter by ('train', 'val', 'test', or 'all' for all splits)
    
    Returns:
        Tuple[List[str], List[int], List[str]]: 
            - List of audio file paths (modified to point to output_path)
            - List of labels
            - List of split assignments
    """
    
    # Load the .npy files using explicit file names from config
    feature_dir = config['feature_dir']
    sound_dir_loc = np.load(os.path.join(feature_dir, config['sound_dir_loc_file']), allow_pickle=True)
    labels = np.load(os.path.join(feature_dir, config['labels_file']), allow_pickle=True)
    splits = np.load(os.path.join(feature_dir, config['split_file']), allow_pickle=True)
    
    # Filter by split (or return all if split is 'all')
    if split == 'all':
        return sound_dir_loc.tolist(), labels.tolist(), splits.tolist()
    else:
        split_mask = np.array(splits) == split
        filtered_paths = sound_dir_loc[split_mask]
        filtered_labels = labels[split_mask]
        filtered_splits = np.array(splits)[split_mask]
        
        return filtered_paths.tolist(), filtered_labels.tolist(), filtered_splits.tolist()

def validate_dataset_files(feature_dir: str, config: dict) -> bool:
    """
    Validates that all required .npy files exist for a dataset using explicit file names from config.
    
    Args:
        feature_dir (str): Directory containing the .npy files
        config (dict): Configuration dictionary containing file names
    
    Returns:
        bool: True if all required files exist, False otherwise
    """
    required_files = [
        config['sound_dir_loc_file'],
        config['labels_file'],
        config['split_file']
    ]
    
    for file in required_files:
        file_path = os.path.join(feature_dir, file)
        if not os.path.exists(file_path):
            print(f"Missing required file: {file_path}")
            return False
    
    return True 

def random_crop_noisy_pair(noisy_audio, clean_audio, crop_size=128):
    """
    Randomly crops a noisy audio pair to a specified size.

    Args:
        noisy_audio (np.ndarray): Noisy audio spectrogram
        clean_audio (np.ndarray): Clean audio spectrogram
        crop_size (int): Size of the crop

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of noisy and clean audio spectrograms
    """
    n_frames = noisy_audio.shape[0]
    if n_frames < crop_size:
        # Pad with zeros
        pad_width = crop_size - n_frames
        noisy_audio_padded = np.pad(noisy_audio, ((0, pad_width), (0, 0)), mode='constant')
        clean_audio_padded = np.pad(clean_audio, ((0, pad_width), (0, 0)), mode='constant')
        return noisy_audio_padded, clean_audio_padded
    else:
        start = int(random.random() * (n_frames - crop_size))
        return noisy_audio[start: (start + crop_size), :], clean_audio[start: (start + crop_size), :]