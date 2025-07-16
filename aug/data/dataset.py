import numpy as np
import torch
from torch.utils.data import Dataset
from aug.utils import load_audio, pre_process_audio_mel
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
    def __init__(self, config, split: str, debug: bool = False, training: bool = False):
        """
        Initialize the dataset.

        Args:
            config: Configuration object or dictionary with all pipeline settings.
            split (str): Data split to use ('train' or 'test').
            debug (bool): If True, disables mel conversion and uses peak normalization for easier inspection.
        """
        super().__init__()
        
        self.config = config
        self.training = training
        self.seed = config.get('random_seed', 42)

        self.clean_data_path = config['clean_data_path']
        self.noise_data_path = config['noise_data_path']
        self.file_paths, self.labels, self.splits = load_dataset_from_npy(config, split)
        self.noise_files = list_wav_files(self.noise_data_path)
        self.noise_files.sort()

        self.noise_type = config['noise_type']
        self.duration = config['target_duration_sec']

        # If training, use the sample rate from the config, otherwise, load the sample rate from the first audio file
        # Adjusts between audio file or mel spectrogram generation
        if training:
            self.sr = config['sample_rate']
        else:
            _, sr = load_audio(self.file_paths[0], sr=None)
            print(f"[DEBUG] Dataset Sample rate: {sr}")
            self.sr = sr
            
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
            - If training is True, returns mel-spectrograms.
            - If training is False, returns waveforms.
            - If the audio file is missing, returns silent audio.
        """

        # Load the clean audio
        clean_audio, _ = load_audio(self.file_paths[idx], sr=self.sr)

        if clean_audio is None:
            print(f"[ERROR] Audio file found at {self.file_paths[idx]}")
            print(f"[ERROR] Returning silent audio")

            # Create silent audio
            silent_len = int(self.duration * self.sr)
            silent_audio = np.zeros(silent_len, dtype=np.float32)

            # If training, convert to mel spectrogram
            if self.training:
                silent_mel = pre_process_audio_mel(silent_audio, f_max=8000, sr=self.sr)
                return (
                    torch.from_numpy(silent_mel),
                    torch.from_numpy(silent_mel),
                    "silent"
                )
            else:
                return (
                    torch.from_numpy(silent_audio),
                    torch.from_numpy(silent_audio),
                    "silent"
                )

        # Augment the audio
        aug_audio, aug_type = self.pipeline(clean_audio, self.seed)

        # If training, convert to mel spectrogram
        if self.training:
            aug_audio = pre_process_audio_mel(aug_audio, f_max=8000, sr=self.sr)
            clean_audio = pre_process_audio_mel(clean_audio, f_max=8000, sr=self.sr)

        # Return the augmented audio and the clean audio
        return (
            torch.from_numpy(aug_audio),
            torch.from_numpy(clean_audio),
            aug_type
        ) 