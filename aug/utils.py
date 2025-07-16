import random
import numpy as np
import torch
import soundfile as sf
import librosa
from typing import Optional, Tuple

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_audio(path: str, audio: np.ndarray, sr: float):
    """
    Saves a numpy audio array to disk as a WAV file.
    
    Args:
        path (str): Output file path.
        audio (np.ndarray): Audio data to save.
        sr (float): Sample rate.
    """
    
    sf.write(path, audio, sr)

def load_audio(path: str, sr: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Loads an audio file as a numpy array.
    
    Args:
        path (str): Path to the audio file.
        sr (float): Target sample rate.
    
    Returns:
        np.ndarray: Loaded audio signal, or empty array if loading fails.
    """
    
    try:
        audio, native_sr = librosa.load(path, sr=sr, mono=True)
        if native_sr == 0.0:
            raise ValueError(f"[ERROR] Sample rate is 0.0 for {path}")
        return audio, native_sr
    except Exception as e:
        print(f"[ERROR] Could not load audio from {path}: {e}")
        return np.array([]), 0.0

def process_audio_length(audio: np.ndarray, duration: float, sr: int) -> np.ndarray:
    """
    Pads or truncates audio to a target duration.
    
    Args:
        audio (np.ndarray): Input audio signal.
        duration (float): Target duration in seconds.
        sr (float): Sample rate.
    
    Returns:
        np.ndarray: Audio of the specified length.
    """

    target_len = int(duration * sr)
    
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        pad_width = target_len - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')
    
    return audio

def peak_normalize(audio: np.ndarray) -> np.ndarray:
    """
    Normalizes audio to have a peak amplitude of 1.0.
    
    Args:
        audio (np.ndarray): Input audio signal.
    
    Returns:
        np.ndarray: Peak-normalized audio.
    """

    peak = np.max(np.abs(audio))
    
    if peak > 0:
        return audio / peak
    
    return audio

def pre_process_audio_mel(audio: np.ndarray, f_max: int = 8000, sr: float = 16000, n_mels: int = 64) -> np.ndarray:
    """
    Converts audio to a mel-spectrogram in dB scale.
    
    Args:
        audio (np.ndarray): Input audio signal.
        f_max (int): Maximum frequency for mel filterbank.
        sr (float): Sample rate.
        n_mels (int): Number of mel bands.
    
    Returns:
        np.ndarray: Mel-spectrogram in dB.
    """

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=f_max)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    return mel_db.astype(np.float32) 