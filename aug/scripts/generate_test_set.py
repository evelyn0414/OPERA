import os
from aug.config import Config
from aug.data.dataset import NoisyICBHIDataGenerator
from aug.utils import set_deterministic, save_audio
import argparse
from torch.autograd.anomaly_mode import set_detect_anomaly
import numpy as np
from aug.utils import load_audio

set_detect_anomaly(True)

def generate_entire_augmented_dataset(config: Config, split: str = 'train'):
    """
    Generates the entire augmented audio dataset for a given split and saves to disk.
    
    Args:
        config_path (str): Path to the YAML config file.
        output_dir (str): Directory to save the generated audio files.
        split (str): Data split to generate ('train' or 'test').
    """
    print(f"--- Generating Entire Audio Dataset for '{split}' split ---")
    
    dataset = NoisyICBHIDataGenerator(config, split, training=False)
    print(f"[DEBUG] Dataset sample rate: {dataset.sr}")
    dataset_sr = dataset.sr
    output_dir = config['output_path']
    
    if len(dataset) == 0:
        print(f"[WARNING] The generator for the '{split}' split contains 0 files. Cannot proceed with generation.")
        return
    print(f"\nGenerator instantiated successfully. Found {len(dataset)} files.")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Audio files will be saved in: '{output_dir}/'")
    
    for i in range(len(dataset)):
        result = dataset[i]
        if len(result) == 3:
            noisy_tensor, clean_tensor, aug_type = result
        else:
            noisy_tensor, clean_tensor = result
        file_name = os.path.basename(dataset.file_paths[i])
        
        # Convert to numpy arrays
        noisy_tensor = noisy_tensor.numpy()
        clean_tensor = clean_tensor.numpy()
        
        save_audio(os.path.join(output_dir, f"{file_name}"), noisy_tensor, dataset_sr)
        if i % 100 == 0:
            print(f"Saved {i} samples so far")
    
    print("\n--- Full Audio Dataset Generation Complete ---")

if __name__ == '__main__':
    """
    Main entry point for generating the full augmented dataset for both train and test splits.
    """
    parser = argparse.ArgumentParser(description='Generate augmented datasets using preprocessed .npy files')
    parser.add_argument('--config', type=str, default='config_icbhidisease_generic.yaml', required=True, help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set deterministic behavior
    config = Config(args.config)
    set_deterministic(args.seed)
    print(f"=" * 10, f"[INFO] Using seed: {args.seed}", "=" * 10)
    print(f"=" * 10, f"[INFO] Generating dataset with noise level: {config['noise_level']} dB", "=" * 10)
    
    generate_entire_augmented_dataset(config, split='train')
    generate_entire_augmented_dataset(config, split='test')
    try:
        generate_entire_augmented_dataset(config, split='val')
    except Exception as e:
        print(f"[DEBUG] No val split found with error: {e}")