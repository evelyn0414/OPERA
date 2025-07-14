import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from aug.config import Config
from aug.utils import load_audio, process_audio_length, pre_process_audio_mel, save_audio, peak_normalize
import augly.audio as A

# ---- USER CONFIGURABLE ----
CLEAN_FILES = [
    "138_2p2_Tc_mc_AKGC417L",
    "130_2b4_Ll_mc_AKGC417L",
    "135_2b1_Ar_mc_LittC2SE",
    "220_1b2_Al_mc_LittC2SE",
    "130_2b3_Pr_mc_AKGC417L"
]

NOISE_FILES = [
    "1-137-A-32",
    "1-4211-A-12",
    "1-5996-A-6",
    "1-7057-A-12",
    "1-7456-A-13",
    "1-9841-A-13",
    "1-9887-A-49",
    "1-11687-A-47",
    "1-17092-A-27",
    "1-24524-C-19"
]
CONFIG_PATH = 'debug_config_icbhidisease.yaml'
SEED = 42

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_rms(audio):
    """Compute root mean square (RMS) of an audio signal."""
    return np.sqrt(np.mean(audio ** 2))

def normalize_to_rms(audio, target_rms):
    """
    Scale audio to have the specified RMS.
    Args:
        audio (np.ndarray): Input audio.
        target_rms (float): Desired RMS value.
    Returns:
        np.ndarray: Scaled audio.
    """
    rms = compute_rms(audio)
    if rms > 0:
        return audio * (target_rms / rms)
    else:
        return audio

def normalize_to_dbfs(audio, target_dbfs):
    """Scale audio to have the specified RMS dBFS."""
    rms = compute_rms(audio)
    if rms > 0:
        target_rms = 10 ** (target_dbfs / 20.0)
        return audio * (target_rms / rms)
    else:
        return audio

def compute_dbfs(audio):
    """Compute average dBFS of an audio signal."""
    rms = compute_rms(audio)
    if rms > 0:
        return 20 * np.log10(rms)
    else:
        return -float('inf')

def plot_and_save_mels(clean, noise, noisy, sr, out_path, title_suffix=""):
    """
    Plot and save mel-spectrograms of clean, noise, and noisy audio side-by-side,
    all sharing the same dB color scale. Titles include average dBFS.
    """
    mels = [
        pre_process_audio_mel(clean, sr=sr),
        pre_process_audio_mel(noise, sr=sr),
        pre_process_audio_mel(noisy, sr=sr),
        # pre_process_audio_mel(noisy - clean, sr=sr),
    ]
    diff = mels[2] - mels[0]
    diff = diff - 80.0
    all_mels = mels + [diff]
    vmin = -80
    vmax = 0

    # Compute dBFS for each signal
    dbfs_clean = compute_dbfs(clean)
    dbfs_noise = compute_dbfs(noise)
    dbfs_noisy = compute_dbfs(noisy)
    dbfs_diff = compute_dbfs(noisy - clean)

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    titles = [
        f"Clean (avg dBFS: {dbfs_clean:.3f} dB) Mel-Spectrogram{title_suffix}",
        f"Noise (avg dBFS: {dbfs_noise:.3f} dB) Mel-Spectrogram{title_suffix}",
        f"Noisy (avg dBFS: {dbfs_noisy:.3f} dB) Mel-Spectrogram{title_suffix}",
        f"Difference (avg dBFS: {(dbfs_diff):.3f} dB) Mel-Spectrogram{title_suffix}"
    ]
    for i, (mel, ax) in enumerate(zip(all_mels, axs)):
        img = ax.imshow(mel, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        ax.set_title(titles[i])
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def main():
    set_deterministic(SEED)
    config = Config(CONFIG_PATH)
    sr = config['sample_rate']
    duration = config['target_duration_sec']
    clean_dir = config['clean_data_path']
    noise_dir = config['noise_data_path']

    noise_level = config.get('noise_level', None)
    if len(noise_level) == 1:
        background_noise_db_level = noise_level[0]
    else:
        raise ValueError("Background noise level must be a single value")

    OUTPUT_DIR = f'debug_noise_injection__snr__{noise_level}dB'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for clean_file in CLEAN_FILES:
        clean_path = os.path.join(clean_dir, clean_file + ".wav")
        clean_audio = load_audio(clean_path, sr)
        if clean_audio is None:
            print(f"[ERROR] Could not load clean file: {clean_file}")
            continue
        # clean_audio = process_audio_length(clean_audio, duration, sr)
        # clean_audio = peak_normalize(clean_audio)

        for noise_file in NOISE_FILES:
            noise_path = os.path.join(noise_dir, noise_file + ".wav")
            noise_audio = load_audio(noise_path, sr)
            if noise_audio is None:
                print(f"[ERROR] Could not load noise file: {noise_file}")
                continue
            # noise_audio = process_audio_length(noise_audio, duration, sr)

            # Scale noise to fixed dBFS before mixing
            # noise_audio_scaled = normalize_to_dbfs(noise_audio, background_noise_db_level)
            # noisy_audio = clean_audio + noise_audio_scaled
            noisy_audio, _ = A.add_background_noise(
                    clean_audio, sample_rate = sr, background_audio=noise_audio, snr_level_db=background_noise_db_level, seed=config.get('random_seed', 42)
                )
            print(f"[INFO] Used fixed [dBFS] noise scaling: {background_noise_db_level} dBFS")

            # Save audio files
            base = f"{os.path.splitext(clean_file)[0]}__{os.path.splitext(noise_file)[0]}"
            save_audio(os.path.join(OUTPUT_DIR, f"{base}_clean.wav"), clean_audio, sr)
            save_audio(os.path.join(OUTPUT_DIR, f"{base}_noise.wav"), noise_audio, sr)
            save_audio(os.path.join(OUTPUT_DIR, f"{base}_noisy.wav"), noisy_audio, sr)

            # Save mel-spectrogram PNG
            plot_and_save_mels(
                clean_audio, noise_audio, noisy_audio, sr,
                os.path.join(OUTPUT_DIR, f"{base}__mels.png"),
                title_suffix=f"\n({clean_file} + {noise_file})"
            )
            print(f"[INFO] Saved debug outputs for {clean_file} + {noise_file}")

            print("================================================")
            print(f"[DEBUG] Clean RMS: {compute_rms(clean_audio):.3f} || dBFS: {compute_dbfs(clean_audio):.3f} dB")
            print(f"[DEBUG] Noise RMS: {compute_rms(noise_audio):.3f} || dBFS: {compute_dbfs(noise_audio):.3f} dB")
            print(f"[DEBUG] Noisy RMS: {compute_rms(noisy_audio):.3f} || dBFS: {compute_dbfs(noisy_audio):.3f} dB")
            print(f"[DEBUG] Difference RMS: {compute_rms(noisy_audio - clean_audio):.3f} || dBFS: {compute_dbfs(noisy_audio - clean_audio):.3f} dB")
            print("================================================")

if __name__ == "__main__":
    main()
