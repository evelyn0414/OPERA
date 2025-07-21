import streamlit as st
import os
import numpy as np
import yaml
import glob
import matplotlib.pyplot as plt

# Helper to load config
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Helper to load file IDs from sound_dir_loc.npy
def load_file_ids(config):
    feature_dir = config['feature_dir']
    sound_dir_loc_file = config['sound_dir_loc_file']
    sound_dir_loc_path = os.path.join(feature_dir, sound_dir_loc_file)
    if not os.path.exists(sound_dir_loc_path):
        return []
    file_paths = np.load(sound_dir_loc_path, allow_pickle=True)
    # Use base name (without extension) as file ID
    file_ids = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]
    return file_ids

# Helper to find all passes for a file ID
def find_passes_for_file(output_dir, file_id):
    clean_files = sorted(glob.glob(os.path.join(output_dir, f'{file_id}_sample*_clean.npy')))
    noisy_files = sorted(glob.glob(os.path.join(output_dir, f'{file_id}_sample*_noisy.npy')))
    # Extract pass numbers
    def extract_pass_num(fname, kind):
        # fname: .../{file_id}_sample{num}_{kind}.npy
        base = os.path.basename(fname)
        try:
            sample_part = base.split(f'{file_id}_sample')[1]
            num_str = sample_part.split(f'_{kind}')[0]
            return int(num_str)
        except Exception:
            return None
    clean_passes = {extract_pass_num(f, 'clean'): f for f in clean_files}
    noisy_passes = {extract_pass_num(f, 'noisy'): f for f in noisy_files}
    # Only keep passes that have both clean and noisy and are not None
    valid_clean_passes = {k: v for k, v in clean_passes.items() if k is not None}
    valid_noisy_passes = {k: v for k, v in noisy_passes.items() if k is not None}
    common_passes = sorted(set(valid_clean_passes.keys()) & set(valid_noisy_passes.keys()))
    return [(p, valid_clean_passes[p], valid_noisy_passes[p]) for p in common_passes]

# Helper to load mel
@st.cache_data
def load_mel(path):
    return np.load(path)

st.title("Noisy/Clean Mel-Spectrogram Viewer (Multi-Pass)")

st.markdown("""
This app lets you browse and visualize all passes of clean and noisy mel-spectrograms for each file in your dataset.
""")

config_path = st.text_input("Path to config YAML", value="debug_config_icbhidisease.yaml")

if not os.path.exists(config_path):
    st.error(f"Config file not found: {config_path}")
    st.stop()

config = load_config(config_path)
output_dir = config.get('output_path', 'noisy_clean_mels')
output_dir = os.path.join(output_dir, "noisy_clean_mels")

print(f"[DEBUG] Output directory: {output_dir}")

if not os.path.exists(output_dir):
    st.error(f"Output directory not found: {output_dir}")
    st.stop()

file_ids = load_file_ids(config)
if not file_ids:
    st.warning("No file IDs found in sound_dir_loc.npy.")
    st.stop()

num_samples = st.number_input("Number of file IDs to display", min_value=1, max_value=len(file_ids), value=min(5, len(file_ids)))
selected_file_ids = st.multiselect("Select file IDs to display", file_ids, default=file_ids[:num_samples])

for file_id in selected_file_ids:
    st.header(f"File ID: {file_id}")
    passes = find_passes_for_file(output_dir, file_id)
    if not passes:
        st.warning(f"No clean/noisy pairs found for file ID {file_id}.")
        continue
    for pass_num, clean_path, noisy_path in passes:
        st.subheader(f"Pass #{pass_num}")
        try:
            clean_mel = load_mel(clean_path)
            noisy_mel = load_mel(noisy_path)
        except Exception as e:
            st.error(f"Error loading pass {pass_num} for {file_id}: {e}")
            continue
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].imshow(clean_mel.T, aspect='auto', origin='lower', cmap='magma')
        axs[0].set_title('Clean Mel-Spectrogram')
        axs[1].imshow(noisy_mel.T, aspect='auto', origin='lower', cmap='magma')
        axs[1].set_title('Noisy Mel-Spectrogram')
        for ax in axs:
            ax.set_xlabel('Time')
            ax.set_ylabel('Mel Bin')
        st.pyplot(fig)
        st.markdown(f"**Clean:** `{os.path.basename(clean_path)}`  ")
        st.markdown(f"**Noisy:** `{os.path.basename(noisy_path)}`  ") 