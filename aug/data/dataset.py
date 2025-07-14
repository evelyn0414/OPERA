import os
import numpy as np
import torch
from torch.utils.data import Dataset
from aug.utils import load_audio, process_audio_length, pre_process_audio_mel, peak_normalize
from aug.data.data_utils import load_dataset_from_npy, list_wav_files
from aug.augment.pipeline import AugmentationPipeline

class NoisyICBHIDataGenerator(Dataset):
    """
    PyTorch Dataset for generating noisy/augmented ICBHI audio samples on-the-fly.

    Each sample is loaded from the ICBHI dataset, optionally augmented with either
    synthetic artefacts or noise from the ESC-50 dataset, or left clean, according to
    probabilities specified in the configuration. The dataset can be used for both
    training and evaluation, and supports debug mode for easier inspection.
    """
    def __init__(self, config, split: str, debug: bool = False):
        """
        Initialize the dataset.

        Args:
            config: Configuration object or dictionary with all pipeline settings.
            split (str): Data split to use ('train' or 'test').
            debug (bool): If True, disables mel conversion and uses peak normalization for easier inspection.
        """
        super().__init__()
        
        self.config = config
        self.debug = debug
        self.seed = config.get('random_seed', 42)
        
        self.sr = config['sample_rate']
        self.noise_type = config['noise_type']
        self.duration = config['target_duration_sec']

        self.clean_data_path = config['clean_data_path']
        self.noise_data_path = config['noise_data_path']
        self.file_paths, self.labels, self.splits = load_dataset_from_npy(config, split)
        
        self.noise_files = list_wav_files(self.noise_data_path)
        self.noise_files.sort()
        
        self.artefact_specs = [
            {'type': 'low_pass', 'cutoff_hz': 2000, 'order': 5},
            {'type': 'low_pass', 'cutoff_hz': 4000, 'order': 5},
            {'type': 'clipping', 'gain_db': 6.0},
            {'type': 'clipping', 'gain_db': 12.0},
        ]
        
        self.pipeline = AugmentationPipeline(
            config,
            self.noise_files,
            self.artefact_specs,
            self.sr,
            self.duration,
            self.noise_type
        )

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads and augments a sample by index.

        Args:
            idx (int): Index of the sample to load.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]:
                - Augmented audio (torch.Tensor)
                - Clean audio (torch.Tensor)
                - Augmentation type (str)
        Notes:
            - If debug is True, returns peak-normalized waveforms.
            - If debug is False, returns mel-spectrograms.
            - If the audio file is missing, returns silent audio.
        """
        clean_audio = load_audio(self.file_paths[idx], sr=self.sr)

        if clean_audio is None:
            silent_len = int(self.duration * self.sr)
            silent_audio = np.zeros(silent_len, dtype=np.float32)
            if self.debug:
                silent_tensor = torch.from_numpy(peak_normalize(silent_audio))
                return silent_tensor, silent_tensor, "silent"
            else:
                silent_mel = pre_process_audio_mel(silent_audio, f_max=8000, sr=self.sr)
                silent_tensor = torch.from_numpy(silent_mel)
                return silent_tensor, silent_tensor, "silent"

        clean_audio = clean_audio
        # clean_audio_processed = process_audio_length(clean_audio_raw, self.duration, self.sr)
        aug_audio, aug_type = self.pipeline(clean_audio, self.seed)

        if self.debug:
            pass
            # aug_audio = peak_normalize(aug_audio)
            # clean_audio = peak_normalize(clean_audio)
        else:
            aug_audio = pre_process_audio_mel(aug_audio, f_max=8000, sr=self.sr)
            clean_audio = pre_process_audio_mel(clean_audio, f_max=8000, sr=self.sr)

        return (
            torch.from_numpy(aug_audio),
            torch.from_numpy(clean_audio),
            aug_type
        ) 