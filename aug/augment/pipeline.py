import os
import random
from typing import List, Optional, Tuple
import numpy as np
from aug.utils import load_audio
import augly.audio as A
import librosa
from aug.augment.augly_effects import ARTEFACT_EFFECTS

class AugmentationPipeline:
    """
    Composes and applies augmentations to audio based on configuration and random selection.
    """
    def __init__(self, config, noise_files: List[str], sr: float, duration: float, noise_type: str):
        """
        Args:
            config: Configuration object or dictionary.
            noise_files (List[str]): List of available noise file names.
            sr (int): Sample rate.
            duration (float): Target duration in seconds.
            noise_loader (callable): Function to load noise audio by filename.
        """

        self.config = config
        self.noise_data_path = config['noise_data_path']

        self.noise_files = noise_files
        
        self.sr = sr
        self.duration = duration
        
        gen_settings = config['generator_settings']
        self.p_clean = gen_settings.get('clean_sample_prob', 0.1)
        self.p_artefact = gen_settings.get('artefact_prob', 0.45)
        self.p_noise = gen_settings.get('ambient_noise_prob', 0.45)
        self.noise_type = noise_type
        self.noise_levels = config.get('noise_level', [-40])

    @staticmethod
    def _ensure_mono(audio: np.ndarray) -> np.ndarray:
        """
        Ensures the audio is mono. If stereo (2 channels), converts to mono by averaging channels.
        Args:
            audio (np.ndarray): Input audio array.
        Returns:
            np.ndarray: Mono audio array.
        """
        if audio.ndim == 2:
            # If shape is (2, N) or (N, 2), average across the channel axis
            if audio.shape[0] == 2:
                return np.mean(audio, axis=0)
            elif audio.shape[1] == 2:
                return np.mean(audio, axis=1)
            elif audio.shape[0] == 1:
                return audio[0]
            elif audio.shape[1] == 1:
                return audio[:, 0]
        return audio

    def __call__(self, clean_audio: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Randomly selects and applies an augmentation to the input audio.
        
        Args:
            clean_audio (np.ndarray): Clean input audio signal.
        
        Returns:
            Tuple[np.ndarray, str]: Augmented audio and a string describing the augmentation type.
        """

        augmentation_options = ['clean', 'artefact', 'noise']
        augmentation_probs = [self.p_clean, self.p_artefact, self.p_noise]
        chosen_augment = random.choices(augmentation_options, weights=augmentation_probs, k=1)[0]
        
        if chosen_augment == 'clean':
            # print(f"[INFO] Applying clean")
            audio = self._ensure_mono(clean_audio)
            return audio, 'clean'
        
        elif chosen_augment == 'artefact':
            # print(f"[INFO] Applying artefact")
            apply_artefact = random.choice(ARTEFACT_EFFECTS)
            noisy_audio, _ = apply_artefact(clean_audio, sample_rate=int(self.sr))
            audio = self._ensure_mono(noisy_audio)
            return audio, f"artefact"
        
        elif chosen_augment == 'noise':
            # print(f"[INFO] Applying noise with {self.noise_levels} dB noise")
            noise_file = random.choice(self.noise_files)
            audio, label = self._inject_ambient_noise(clean_audio, noise_file, self.noise_levels)
            audio = self._ensure_mono(audio)
            return audio, label
        
        else:
            raise ValueError(f"Unknown augmentation type: {chosen_augment}")
    
    def _inject_ambient_noise(self, clean_audio: np.ndarray, noise_file: str, noise_level: float) -> Tuple[np.ndarray, str]:
        
        # Load the noise audio
        noise_audio = self._load_noise_audio(noise_file, self.sr)
        
        # Add the noise to the clean audio
        noise_level = random.choice(self.noise_levels)
        
        # If the noise level is infinite, return the clean audio
        if noise_level == "inf":
            noisy_audio = clean_audio
        else:
            noisy_audio, _ = A.add_background_noise(
                    clean_audio, sample_rate=int(self.sr), background_audio=noise_audio, snr_level_db=noise_level
                )
        noisy_audio = self._ensure_mono(noisy_audio)
        return noisy_audio, f"noise_{noise_file}_{self.noise_type}_{noise_level}"
    
    def _load_noise_audio(self, noise_file: str, sr: float) -> np.ndarray:
        noise_path = os.path.join(self.noise_data_path, noise_file)
        noise_audio, _ = load_audio(noise_path, sr=sr)
        noise_audio = self._process_audio_length(noise_audio)
        return noise_audio
    
    def _process_audio_length(self, audio: np.ndarray) -> np.ndarray:
        # Get the length of the audio
        duration = librosa.get_duration(y=audio, sr=self.sr)
        if duration >  20.0:
            # Trim the audio
            audio_trimmed = librosa.util.fix_length(audio, size=int(20.0 * self.sr))
            return audio_trimmed
        elif duration < 10.0:
            # Pad the audio
            audio_padded = librosa.util.fix_length(audio, size=int(10.0 * self.sr))
            return audio_padded
        else:
            # Return the audio as is
            return audio