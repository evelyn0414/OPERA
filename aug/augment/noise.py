import numpy as np
from aug.augment.base import BaseAugmentation
from aug.utils import process_audio_length
import augly.audio as audaugs

class NoiseAugmentation(BaseAugmentation):
    """
    Augmentation that adds ambient noise to a clean audio signal at a specified dBFS level.
    """
    def __init__(self, config, noise_audio: np.ndarray, sr: int, duration: float, noise_level: float, noise_type: str):
        """
        Args:
            noise_audio (np.ndarray): The noise audio signal.
            sr (int): Sample rate.
            duration (float): Target duration in seconds.
            background_noise_db_level (float): Target dBFS for noise.
        """

        self.config = config

        self.noise_audio = noise_audio
        self.sr = sr
        self.duration = duration
        self.noise_level = noise_level
        self.noise_type = noise_type

    def __call__(self, clean_audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adds noise to the clean audio at the specified SNR.
        
        Args:
            clean_audio (np.ndarray): Clean input audio signal.
        
        Returns:
            np.ndarray: Noisy audio signal.
        """
        clean_proc = process_audio_length(clean_audio, self.duration, self.sr)
        noise_proc = process_audio_length(self.noise_audio, self.duration, self.sr)
        
        noisy_audio, _ = audaugs.add_background_noise(
                clean_audio, sample_rate = self.sr, background_audio=self.noise_audio, snr_level_db=self.noise_level, seed=self.config.get('random_seed', 42)
            )
        return noisy_audio 