import os
import numpy as np
import yaml
from tqdm import tqdm

from aug.utils import load_audio, set_deterministic
from aug.augment.pipeline import AugmentationPipeline
from aug.data.data_utils import load_dataset_from_npy, random_crop_noisy_pair

from src.util import pre_process_audio_mel_t
import librosa

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(config_path):
    # Load config
    config = load_config(config_path)
    feature_dir = config['feature_dir']
    output_dir = config.get('output_path', os.path.join(feature_dir, "noisy_clean_mels"))
    output_dir = os.path.join(output_dir, "noisy_clean_mels")
    ensure_dir(output_dir)

    # Load file paths (and optionally labels/splits)
    file_paths, _, _ = load_dataset_from_npy(config, split='all')

    # Set up augmentation pipeline (noise only)
    noise_files = os.listdir(config['noise_data_path'])
    noise_files = [f for f in noise_files if f.endswith('.wav')]
    sr = config['sample_rate']
    print(f"[DEBUG] Loaded {len(noise_files)} noise files")
    print(f"[DEBUG] sample rate: {sr}")
    input_sec = config['target_duration_sec']
    noise_type = config.get('noise_type', 'augly')
    pipeline = AugmentationPipeline(config, noise_files, sr, input_sec, noise_type)

    # Determine target training set size
    train_set_size = config.get('train_set_size', len(file_paths))
    n_files = len(file_paths)

    idx = 0
    for i in tqdm(range(train_set_size), desc="Generating training set"):  # Loop until reaching train_set_size
        file_idx = i % n_files
        identifier = i // n_files
        audio_path = file_paths[file_idx]
        print(f"[DEBUG] Processing {audio_path} with idx {file_idx} and i {i}")
        # Load clean audio
        clean_audio, _ = load_audio(audio_path, sr=sr)
        if clean_audio is None:
            print(f"[WARNING] Could not load {audio_path}, skipping.")
            continue

        duration = librosa.get_duration(y=clean_audio, sr=sr)
        if duration < input_sec:
            # pad the audio to the target duration
            clean_audio = np.pad(clean_audio, (0, int(input_sec * sr) - len(clean_audio)), mode='constant')

        # Clean mel
        clean_mel = pre_process_audio_mel_t(clean_audio, f_max=8000, sample_rate=sr)

        # Noisy audio (using pipeline, force noise)
        noisy_audio, aug_type = pipeline(clean_audio, seed=idx)
        noisy_mel = pre_process_audio_mel_t(noisy_audio, f_max=8000, sample_rate=sr)

        noisy_mel, clean_mel = random_crop_noisy_pair(noisy_mel, clean_mel, crop_size=251)

        # Save both
        base = os.path.splitext(os.path.basename(audio_path))[0]
        print(f"[DEBUG] Saving {base}_sample{identifier}_clean.npy and {base}_sample{identifier}_noisy.npy")
        np.save(os.path.join(output_dir, f"{base}_sample{identifier}_clean.npy"), clean_mel)
        np.save(os.path.join(output_dir, f"{base}_sample{identifier}_noisy.npy"), noisy_mel)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Universal noisy/clean mel-spectrogram preprocessor")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    # Set deterministic seed
    set_deterministic()

    # Run main function
    main(args.config)