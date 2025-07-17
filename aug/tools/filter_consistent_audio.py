import os
import shutil
import librosa
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "/projects/prjs1635/datasets/fsd50k/FSD50K.selected_eval"      # <-- Set this to your source directory
OUTPUT_DIR = "/projects/prjs1635/datasets/fsd50k/filtered"    # <-- Set this to your target directory
MIN_CONSISTENT_SEC = 4.0
RMS_THRESHOLD = 0.01
FRAME_LENGTH = 2048
HOP_LENGTH = 512

def has_consistent_audio(y, sr, min_sec, rms_thresh):
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    frame_duration = HOP_LENGTH / sr
    consistent_frames = np.sum(rms > rms_thresh)
    consistent_duration = consistent_frames * frame_duration
    return consistent_duration >= min_sec

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audio_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.wav', '.flac', '.mp3', '.ogg'))]
    print(f"Found {len(audio_files)} audio files.")

    for fname in tqdm(audio_files, desc="Processing audio files"):
        in_path = os.path.join(INPUT_DIR, fname)
        try:
            y, sr = librosa.load(in_path, sr=None, mono=True)
            if has_consistent_audio(y, sr, MIN_CONSISTENT_SEC, RMS_THRESHOLD):
                shutil.copy2(in_path, os.path.join(OUTPUT_DIR, fname))
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    main()