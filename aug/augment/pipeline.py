import os
import random
from typing import List, Tuple
import numpy as np
from aug.augment.artefact import ArtefactAugmentation
from aug.utils import load_audio
import augly.audio as A
import librosa

class AugmentationPipeline:
    """
    Composes and applies augmentations to audio based on configuration and random selection.
    """
    def __init__(self, config, noise_files: List[str], artefact_specs: List[dict], sr: int, duration: float, noise_type: str):
        """
        Args:
            config: Configuration object or dictionary.
            noise_files (List[str]): List of available noise file names.
            artefact_specs (List[dict]): List of artefact specifications.
            sr (int): Sample rate.
            duration (float): Target duration in seconds.
            noise_loader (callable): Function to load noise audio by filename.
        """

        self.config = config
        self.noise_data_path = config['noise_data_path']

        self.noise_files = noise_files
        self.artefact_specs = artefact_specs
        
        self.sr = sr
        self.duration = duration
        
        gen_settings = config['generator_settings']
        self.p_clean = gen_settings.get('clean_sample_prob', 0.1)
        self.p_artefact = gen_settings.get('artefact_prob', 0.45)
        self.p_noise = gen_settings.get('ambient_noise_prob', 0.45)
        self.noise_type = noise_type
        self.noise_levels = config.get('noise_level', [-40])

    def __call__(self, clean_audio: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
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
            return clean_audio, 'clean'
        
        elif chosen_augment == 'artefact':
            spec = random.choice(self.artefact_specs)
            artefact_aug = ArtefactAugmentation(spec, self.sr, self.duration)
            return artefact_aug(clean_audio), f"artefact_{spec['type']}"
        
        elif chosen_augment == 'noise':
            
            # Load a randomnoise audio
            noise_file = random.choice(self.noise_files)
            noise_audio = self._load_noise_audio(noise_file)
            noise_level = random.choice(self.noise_levels)
            
            noisy_audio = self.add_noise(clean_audio, noise_audio, noise_level, seed)
            return noisy_audio, f"noise_{noise_file}_{self.noise_type}_{noise_level}"
        
        else:
            raise ValueError(f"Unknown augmentation type: {chosen_augment}")
    
    def _load_noise_audio(self, noise_file: str) -> np.ndarray:
        noise_path = os.path.join(self.noise_data_path, noise_file)
        noise_audio = load_audio(noise_path, sr=self.sr)
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

    
    def add_noise(self, clean_audio: np.ndarray, noise_audio: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
        noisy_audio, _ = A.add_background_noise(
                clean_audio, sample_rate = self.sr, background_audio=noise_audio, snr_level_db=noise_level, seed=seed
            )
        return noisy_audio